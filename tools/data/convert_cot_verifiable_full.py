#!/usr/bin/env python3
"""Extract ALL verifiable-answer COT samples (Multiple Choice / Binary / Counting)
from ThinkStream COT data into a *separate* verl-format parquet.

This differs from `convert_cot_to_rlvr.py` in two ways:
  1. NO 10-second upper duration cap  — keeps every verifiable sample regardless
     of clip length (only drops sub-0.5s windows that crash decord).
  2. Writes to a NEW, independent file name so it does not mix with the existing
     training pools (rlvr_*, cot_rlvr_train_filtered, train20k_*).

Source : <data-dir>/cot_train.parquet + cot_val.parquet
         (these already passed the video-existence check during the
          convert_thinkstream_to_verl.py run, so we reuse them instead of
          re-scanning the video store.)
Output : <data-dir>/cot_verifiable_all.parquet  (23,884 samples)

Schema (matches train20k_filtered_8343.parquet / rlvr_train_filtered.parquet exactly):
    prompt          list[{role, content}]   content = "<video>" + question
    videos          list[{video, video_start, video_end}]
    ground_truth    str
    response_format str
    data_source     str   ("thinkstream_rlvr" -> reuses registered exact-match reward)
    reward_model    dict  {"ground_truth": gt}
"""
import json
import os
from pathlib import Path

import pandas as pd

# Directory holding the parquets emitted by convert_thinkstream_to_verl.py.
DATA_DIR = Path(os.environ.get("STREAMOPD_RAW_DIR", "data/raw"))
SRC_FILES = [DATA_DIR / "cot_train.parquet", DATA_DIR / "cot_val.parquet"]
DST = DATA_DIR / "cot_verifiable_all.parquet"

MIN_DUR = 0.5     # drop sub-half-second clips (decord struggles); NO upper cap.

CLEAN = {
    "Multiple Choice": lambda s: s in {"A", "B", "C", "D", "E", "F"},
    "Binary":          lambda s: s in {"Yes", "No"},
    "Counting":        lambda s: s.isdigit(),
}


def normalize_gt(gt: str, fmt: str) -> str | None:
    """Return a canonical GT string, or None if it can't be cleaned."""
    s = str(gt).strip().rstrip(".")  # 'No.' -> 'No'
    if fmt == "Binary":
        s = s.capitalize()           # 'yes' -> 'Yes'
    if fmt in CLEAN and CLEAN[fmt](s):
        return s
    return None


def convert_row(row):
    msgs = json.loads(row["prompt"])
    contents = msgs[0]["content"]

    text_part = next(c for c in contents if c["type"] == "text")
    video_part = next(c for c in contents if c["type"] == "video")

    duration = float(video_part["video_end"]) - float(video_part["video_start"])
    if duration < MIN_DUR:            # only a lower bound, NO upper cap
        return None, "dur"

    gt = normalize_gt(row["ground_truth"], row["response_format"])
    if gt is None:
        return None, "gt"

    question = text_part["text"].strip()
    content = "<video>" + question    # verl rl_dataset keys off the <video> marker

    return {
        "prompt": [{"role": "user", "content": content}],
        "videos": [{
            "video":       video_part["video"],
            "video_start": float(video_part["video_start"]),
            "video_end":   float(video_part["video_end"]),
        }],
        "ground_truth":    gt,
        "response_format": row["response_format"],
        "data_source":     "thinkstream_rlvr",   # reuse registered exact-match reward fn
        "reward_model":    {"ground_truth": gt},
    }, None


def main():
    frames = []
    for f in SRC_FILES:
        print(f"[load] {f}")
        df = pd.read_parquet(f)
        print(f"  {len(df):,} rows")
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    print(f"[merge] cot train+val total: {len(df):,}")

    fmt_counts = df["response_format"].value_counts().to_dict()
    print(f"  format breakdown: {fmt_counts}")

    det_mask = df["response_format"].isin(["Multiple Choice", "Binary", "Counting"])
    det = df[det_mask].reset_index(drop=True)
    print(f"[filter] deterministic formats: {len(det):,}")

    out = []
    skipped = {"dur": 0, "gt": 0, "err": 0}
    for _, row in det.iterrows():
        try:
            rec, reason = convert_row(row)
        except Exception:
            skipped["err"] += 1
            continue
        if rec is None:
            skipped[reason] += 1
            continue
        out.append(rec)

    print(f"[skip] sub-0.5s window: {skipped['dur']:,}")
    print(f"[skip] dirty gt:        {skipped['gt']:,}")
    print(f"[skip] parse error:     {skipped['err']:,}")
    print(f"[keep] final:           {len(out):,}")

    out_df = pd.DataFrame(out)

    # dedup on (video + question + gt) in case train/val overlap
    before = len(out_df)
    out_df["_key"] = out_df.apply(
        lambda r: (r["videos"][0]["video"], r["prompt"][0]["content"], r["ground_truth"]), axis=1
    )
    out_df = out_df.drop_duplicates("_key").drop(columns="_key").reset_index(drop=True)
    print(f"[dedup] {before:,} -> {len(out_df):,}")

    print(f"\n[write] {DST}")
    out_df.to_parquet(DST, index=False)
    print(f"  saved {len(out_df):,} rows -> {DST.stat().st_size / 1e6:.1f} MB")

    # Verify schema matches the training parquet exactly
    ref = pd.read_parquet(DATA_DIR / "train20k_filtered_8343.parquet")
    cols_match = list(out_df.columns) == list(ref.columns)
    print(f"\n[verify] columns match train20k_filtered_8343: {cols_match}")
    print(f"  ref columns: {list(ref.columns)}")
    print(f"  new columns: {list(out_df.columns)}")

    print(f"\n[stats] response_format:")
    print(out_df["response_format"].value_counts().to_string())
    print(f"\n[stats] ground_truth (top 10):")
    print(out_df["ground_truth"].value_counts().head(10).to_string())
    durs = out_df["videos"].apply(lambda v: v[0]["video_end"] - v[0]["video_start"])
    print(f"\n[stats] duration: min={durs.min():.2f}s max={durs.max():.2f}s "
          f"mean={durs.mean():.2f}s median={durs.median():.2f}s")
    print(f"  >10s: {(durs > 10).sum():,}  (these are the ones convert_cot_to_rlvr.py dropped)")


if __name__ == "__main__":
    main()
