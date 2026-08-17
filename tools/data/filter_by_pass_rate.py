#!/usr/bin/env python3
"""
Apply the pass-rate filter to a training pool.

Keeps a sample when the teacher is at least as accurate as the student and the pair is not
already saturated:

    keep = teacher.n_correct >= student.n_correct and not (student == teacher == n_samples)

Dropping "both perfect" removes rows whose KL signal is ~0; dropping "student better than
teacher" removes rows where distillation would pull the student toward a wrong answer.
Applied to the 20,031-row pool with n=8 this yields the 8,343-row subset used in the paper.

    python tools/data/filter_by_pass_rate.py \
        --pool data/raw/train20k.parquet \
        --student data/passrate/student_part1.jsonl data/passrate/student_part2.jsonl \
        --teacher data/passrate/teacher_part1.jsonl data/passrate/teacher_part2.jsonl \
        --out data/train20k_filtered_8343.parquet
"""

import argparse
import json
from collections import Counter

import pandas as pd


def load_counts(paths):
    """idx -> n_correct, taking the last record for repeated idx."""
    counts = {}
    for path in paths:
        with open(path) as fh:
            for line in fh:
                rec = json.loads(line)
                if "idx" in rec and "n_correct" in rec:
                    counts[int(rec["idx"])] = int(rec["n_correct"])
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True, help="parquet holding the unfiltered training pool")
    ap.add_argument("--student", nargs="+", required=True, help="student pass-rate jsonl files")
    ap.add_argument("--teacher", nargs="+", required=True, help="teacher pass-rate jsonl files")
    ap.add_argument("--n-samples", type=int, default=8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    df = pd.read_parquet(args.pool)
    student = load_counts(args.student)
    teacher = load_counts(args.teacher)

    stats = Counter()
    keep = []
    for idx in range(len(df)):
        s, t = student.get(idx), teacher.get(idx)
        if s is None or t is None:
            stats["missing_pass_rate"] += 1
            continue
        if s == t == args.n_samples:
            stats["both_perfect"] += 1
        elif t > s:
            stats["teacher_better"] += 1
            keep.append(idx)
        elif t == s:
            stats["tied_below_perfect"] += 1
            keep.append(idx)
        else:
            stats["student_better"] += 1

    for name, n in stats.most_common():
        print(f"  {name:<20} {n:>7,}")
    out = df.iloc[keep].reset_index(drop=True)
    out.to_parquet(args.out, index=False)
    print(f"[out] {args.out}: {len(out):,} / {len(df):,} rows kept")


if __name__ == "__main__":
    main()
