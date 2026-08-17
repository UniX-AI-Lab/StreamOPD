#!/usr/bin/env python3
"""
Concatenate training parquets and drop duplicates.

Two rows are the same sample when their prompt text and ground truth match; earlier files
win, so list the more carefully filtered pool first. This is how the 25,118-row training
set is built:

    python tools/data/merge_dedup.py \
        --inputs data/train20k_filtered_8343.parquet data/raw/cot_verifiable_all.parquet \
        --out data/filtered8k_plus_cot_verifiable_dedup_25118.parquet
"""

import argparse
import hashlib
import json

import pandas as pd


def row_key(row) -> str:
    prompt = row["prompt"]
    text = json.dumps(prompt.tolist() if hasattr(prompt, "tolist") else prompt,
                      sort_keys=True, default=str)
    return hashlib.md5(f"{text}||{row['ground_truth']}".encode()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True, help="parquets to merge, highest priority first")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    frames = []
    for path in args.inputs:
        df = pd.read_parquet(path)
        print(f"[load] {path}: {len(df):,} rows")
        frames.append(df)

    merged = pd.concat(frames, ignore_index=True)
    merged["_key"] = merged.apply(row_key, axis=1)
    deduped = merged.drop_duplicates(subset="_key", keep="first").drop(columns="_key")

    print(f"[merge] {len(merged):,} concatenated -> {len(deduped):,} after dedup "
          f"({len(merged) - len(deduped):,} overlapping)")
    deduped.reset_index(drop=True).to_parquet(args.out, index=False)
    print(f"[out] {args.out}")


if __name__ == "__main__":
    main()
