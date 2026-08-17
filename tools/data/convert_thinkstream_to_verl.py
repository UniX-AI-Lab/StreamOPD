"""
Convert ThinkStream datasets to verl-compatible parquet format for On-Policy Distillation.

ThinkStream has two datasets:
1. streaming_cot_cold (110K): Multi-turn streaming reasoning (SFT cold start)
2. streaming_rlvr (8.9K): Single-turn QA with verifiable answers (RL)

For OPD, we use RLVR data (clear correct/incorrect signal) as the primary training set.
The COT data can optionally be used for warm-up SFT.

verl expects parquet with columns:
- prompt: JSON string of messages list (OpenAI format)
- data_source: string identifier for routing

Usage:
    python tools/data/convert_thinkstream_to_verl.py \
        --thinkstream-dir /path/to/ThinkStream --output-dir data/raw
"""

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

THINKSTREAM_DIR = ""
OUTPUT_DIR = ""


def check_video_exists(video_path: str) -> bool:
    """Check if video file exists."""
    return os.path.exists(video_path)


def convert_rlvr_to_verl(
    input_path: str,
    output_path: str,
    max_video_duration: float = 30.0,
    skip_missing_videos: bool = True,
):
    """
    Convert streaming_rlvr data to verl parquet format.

    RLVR data: single user question + single assistant answer, with timestamp.
    For OPD: we construct a prompt that includes the video up to the timestamp,
    and the question. The student will generate a response, and teacher will score it.

    Args:
        input_path: Path to streaming_rlvr_processed_abspath.jsonl
        output_path: Path to output .parquet file
        max_video_duration: Maximum seconds of video to include before timestamp
        skip_missing_videos: Skip entries with missing video files
    """
    records = []
    skipped = 0
    total = 0

    with open(input_path) as f:
        for line in tqdm(f, desc="Converting RLVR"):
            item = json.loads(line)
            total += 1

            video_path = item["video_path"]
            conversations = item["conversations"]
            response_format = item.get("response_format", "")

            # Skip if video doesn't exist
            if skip_missing_videos and not check_video_exists(video_path):
                skipped += 1
                continue

            # Extract user question and answer
            user_msg = conversations[0]
            assistant_msg = conversations[1]
            timestamp = float(user_msg["timestamp"])
            question = user_msg["content"]
            ground_truth = assistant_msg["content"]

            # Calculate video window: [max(0, timestamp - max_duration), timestamp]
            video_start = max(0.0, timestamp - max_video_duration)
            video_end = timestamp

            # Build verl prompt format (OpenAI messages style)
            # For VL models, video is embedded in the user message content
            prompt = [{
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_path,
                        "video_start": video_start,
                        "video_end": video_end,
                    },
                    {
                        "type": "text",
                        "text": question,
                    },
                ],
            }]

            records.append({
                "prompt": json.dumps(prompt, ensure_ascii=False),
                "ground_truth": ground_truth,
                "response_format": response_format,
                "timestamp": timestamp,
                "video_path": video_path,
                "data_source": "thinkstream_rlvr",
            })

    print(f"Total: {total}, Converted: {len(records)}, Skipped (missing video): {skipped}")

    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"Saved to: {output_path} ({len(df)} rows)")
    return df


def convert_cot_to_verl(
    input_path: str,
    output_path: str,
    max_video_duration: float = 30.0,
    skip_missing_videos: bool = True,
    max_samples: int = None,
):
    """
    Convert streaming_cot_cold data to verl parquet format.

    COT data has multi-turn conversations. For OPD, we take the user's question
    (first turn) and use it as the prompt. The model needs to generate a streaming
    response. We use the first assistant response as ground truth for reference.

    Args:
        input_path: Path to streaming_cot_cold_processed_5_20_abspath.jsonl
        output_path: Path to output .parquet file
        max_video_duration: Maximum seconds of video before question timestamp
        skip_missing_videos: Skip entries with missing video files
        max_samples: Optional limit on number of samples
    """
    records = []
    skipped = 0
    total = 0

    with open(input_path) as f:
        for line in tqdm(f, desc="Converting COT"):
            item = json.loads(line)
            total += 1

            if max_samples and len(records) >= max_samples:
                break

            video_path = item["video_path"]
            conversations = item["conversations"]
            response_format = item.get("response_format", "")

            if skip_missing_videos and not check_video_exists(video_path):
                skipped += 1
                continue

            # First turn is always user
            user_msg = conversations[0]
            timestamp = float(user_msg["timestamp"])
            question = user_msg["content"]

            # Collect all assistant responses as ground truth
            assistant_responses = []
            for msg in conversations[1:]:
                if msg["role"] == "assistant":
                    assistant_responses.append(msg["content"])
            ground_truth = " ".join(assistant_responses) if assistant_responses else ""

            video_start = max(0.0, timestamp - max_video_duration)
            video_end = timestamp

            prompt = [{
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_path,
                        "video_start": video_start,
                        "video_end": video_end,
                    },
                    {
                        "type": "text",
                        "text": question,
                    },
                ],
            }]

            records.append({
                "prompt": json.dumps(prompt, ensure_ascii=False),
                "ground_truth": ground_truth,
                "response_format": response_format,
                "timestamp": timestamp,
                "video_path": video_path,
                "data_source": "thinkstream_cot",
            })

    print(f"Total: {total}, Converted: {len(records)}, Skipped (missing video): {skipped}")

    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"Saved to: {output_path} ({len(df)} rows)")
    return df


def create_train_val_split(df: pd.DataFrame, val_ratio: float = 0.05, seed: int = 42):
    """Split dataframe into train and validation sets."""
    df_shuffled = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    val_size = max(1, int(len(df_shuffled) * val_ratio))
    val_df = df_shuffled[:val_size]
    train_df = df_shuffled[val_size:]
    return train_df, val_df


def main():
    global THINKSTREAM_DIR, OUTPUT_DIR
    ap = argparse.ArgumentParser(description="Convert ThinkStream jsonl to verl parquet")
    ap.add_argument("--thinkstream-dir", required=True,
                    help="directory holding the ThinkStream *_abspath.jsonl files")
    ap.add_argument("--output-dir", default="data/raw", help="where to write the parquets")
    args = ap.parse_args()
    THINKSTREAM_DIR = args.thinkstream_dir
    OUTPUT_DIR = args.output_dir

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("Converting ThinkStream → verl parquet for OPD")
    print("=" * 60)

    # 1. Convert RLVR data (primary training data for OPD)
    print("\n--- RLVR Dataset (8.9K, verifiable answers) ---")
    rlvr_df = convert_rlvr_to_verl(
        input_path=f"{THINKSTREAM_DIR}/streaming_rlvr_processed_abspath.jsonl",
        output_path=f"{OUTPUT_DIR}/rlvr_all.parquet",
    )

    # Split RLVR into train/val
    rlvr_train, rlvr_val = create_train_val_split(rlvr_df, val_ratio=0.05)
    rlvr_train.to_parquet(f"{OUTPUT_DIR}/rlvr_train.parquet", index=False)
    rlvr_val.to_parquet(f"{OUTPUT_DIR}/rlvr_val.parquet", index=False)
    print(f"  Train: {len(rlvr_train)}, Val: {len(rlvr_val)}")

    # 2. Convert COT data (optional, for warm-up or additional training signal)
    print("\n--- COT Dataset (110K, streaming reasoning) ---")
    cot_df = convert_cot_to_verl(
        input_path=f"{THINKSTREAM_DIR}/streaming_cot_cold_processed_5_20_abspath.jsonl",
        output_path=f"{OUTPUT_DIR}/cot_all.parquet",
    )

    # Split COT into train/val
    cot_train, cot_val = create_train_val_split(cot_df, val_ratio=0.02)
    cot_train.to_parquet(f"{OUTPUT_DIR}/cot_train.parquet", index=False)
    cot_val.to_parquet(f"{OUTPUT_DIR}/cot_val.parquet", index=False)
    print(f"  Train: {len(cot_train)}, Val: {len(cot_val)}")

    # 3. Print summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Files created:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        if f.endswith(".parquet"):
            size = os.path.getsize(f"{OUTPUT_DIR}/{f}") / 1024 / 1024
            print(f"  {f}: {size:.1f} MB")


if __name__ == "__main__":
    main()
