#!/usr/bin/env python3
"""Convert cot_train.parquet (deterministic-answer subset) into the rlvr/verl
format used by the OPD training pipeline.

Input  : <data-dir>/cot_train.parquet  (79,156 samples, JSON-string prompts)
Output : <data-dir>/cot_rlvr_train_filtered.parquet  (17,369 samples after filters)

Filters:
  1. response_format in {Multiple Choice, Binary, Counting}  -> drops 70% open-ended
  2. ground_truth is "clean" (single letter / Yes-No / pure digit)
  3. 0.5s <= clip duration <= 10s   (matches rlvr_train_filtered)

The output schema matches rlvr_train_filtered.parquet exactly so the file can
be passed to verl via:
    data.train_files=['rlvr_train_filtered.parquet', 'cot_rlvr_train_filtered.parquet']
"""
import json
import os
import re
from pathlib import Path

import pandas as pd

# Directory holding the parquets emitted by convert_thinkstream_to_verl.py.
DATA_DIR = Path(os.environ.get("STREAMOPD_RAW_DIR", "data/raw"))
SRC = DATA_DIR / "cot_train.parquet"
DST = DATA_DIR / "cot_rlvr_train_filtered.parquet"

MIN_DUR = 0.5   # drop sub-half-second clips (decord struggles)
MAX_DUR = 10.0  # match rlvr_train_filtered

CLEAN = {
    "Multiple Choice": lambda s: s in {"A", "B", "C", "D", "E", "F"},
    "Binary":          lambda s: s in {"Yes", "No"},
    "Counting":        lambda s: s.isdigit(),
}


def normalize_gt(gt: str, fmt: str) -> str | None:
    """Return a canonical GT string, or None if it can't be cleaned."""
    s = str(gt).strip().rstrip(".")  # 'No.' -> 'No'
    if fmt == "Binary":
        s = s.capitalize()            # 'yes' -> 'Yes'
    if CLEAN[fmt](s):
        return s
    return None


def convert_row(row):
    msgs = json.loads(row["prompt"])
    contents = msgs[0]["content"]

    text_part  = next(c for c in contents if c["type"] == "text")
    video_part = next(c for c in contents if c["type"] == "video")

    duration = float(video_part["video_end"]) - float(video_part["video_start"])
    if not (MIN_DUR <= duration <= MAX_DUR):
        return None

    gt = normalize_gt(row["ground_truth"], row["response_format"])
    if gt is None:
        return None

    question = text_part["text"].strip()
    # Make sure '<video>' marker is present at the very front of the user message.
    # This is what verl's rl_dataset.py keys off of.
    content = "<video>" + question

    return {
        "prompt": [{"role": "user", "content": content}],
        "videos": [{
            "video":       video_part["video"],
            "video_start": float(video_part["video_start"]),
            "video_end":   float(video_part["video_end"]),
        }],
        "ground_truth":    gt,
        "response_format": row["response_format"],
        "data_source":     "thinkstream_rlvr",          # reuse registered reward fn
        "reward_model":    {"ground_truth": gt},
    }


def main():
    print(f"[load] {SRC}")
    df = pd.read_parquet(SRC)
    print(f"  {len(df):,} rows total")

    fmt_counts = df["response_format"].value_counts().to_dict()
    print(f"  format breakdown: {fmt_counts}")

    # Filter to deterministic-answer formats first (cheap)
    det_mask = df["response_format"].isin(["Multiple Choice", "Binary", "Counting"])
    det = df[det_mask].reset_index(drop=True)
    print(f"[filter] deterministic formats: {len(det):,}")

    # Convert (also applies duration + cleanliness filters)
    out = []
    skipped_dur = 0
    skipped_gt  = 0
    skipped_err = 0
    for _, row in det.iterrows():
        try:
            rec = convert_row(row)
        except Exception:
            skipped_err += 1
            continue
        if rec is None:
            # Determine the reason cheaply
            try:
                msgs = json.loads(row["prompt"])
                vid = next(c for c in msgs[0]["content"] if c["type"] == "video")
                dur = float(vid["video_end"]) - float(vid["video_start"])
                if not (MIN_DUR <= dur <= MAX_DUR):
                    skipped_dur += 1
                    continue
            except Exception:
                pass
            skipped_gt += 1
            continue
        out.append(rec)

    print(f"[skip] duration out-of-range: {skipped_dur:,}")
    print(f"[skip] dirty ground_truth:    {skipped_gt:,}")
    print(f"[skip] parse error:           {skipped_err:,}")
    print(f"[keep] final:                 {len(out):,}")

    out_df = pd.DataFrame(out)
    print(f"\n[write] {DST}")
    out_df.to_parquet(DST, index=False)
    print(f"  saved {len(out_df):,} rows -> {DST.stat().st_size / 1e6:.1f} MB")

    # Sanity: verify schema matches rlvr_train_filtered exactly
    rlvr = pd.read_parquet(DATA_DIR / "rlvr_train_filtered.parquet")
    cols_match = list(out_df.columns) == list(rlvr.columns)
    print(f"\n[verify] columns match rlvr_train_filtered: {cols_match}")
    print(f"  rlvr columns:     {list(rlvr.columns)}")
    print(f"  cot_rlvr columns: {list(out_df.columns)}")

    # Show distributions
    print(f"\n[stats] response_format:")
    print(out_df["response_format"].value_counts().to_string())
    print(f"\n[stats] ground_truth (top 10):")
    print(out_df["ground_truth"].value_counts().head(10).to_string())

    durs = out_df["videos"].apply(lambda v: v[0]["video_end"] - v[0]["video_start"])
    print(f"\n[stats] duration: min={durs.min():.2f}s max={durs.max():.2f}s "
          f"mean={durs.mean():.2f}s median={durs.median():.2f}s")


if __name__ == "__main__":
    main()
