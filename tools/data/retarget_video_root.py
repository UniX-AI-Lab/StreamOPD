#!/usr/bin/env python3
"""Rewrite the video-root prefix stored inside the training/validation parquets.

Each parquet row stores an absolute path to a video file. The released parquets ship with
the placeholder root ``/path/to/video_root``; point them at your local dataset root once
after downloading the public videos (see docs/DATA.md).

    python tools/data/retarget_video_root.py --root /mnt/datasets
    python tools/data/retarget_video_root.py --root /mnt/datasets --check-exists

Re-running with a different ``--root`` is safe: pass ``--old-root`` to override the prefix
that is being replaced.
"""

import argparse
import glob
import os

import pandas as pd

DEFAULT_OLD_ROOT = "/path/to/video_root"


def rewrite_videos(cell, old_root: str, new_root: str):
    out = []
    for entry in cell:
        entry = dict(entry)
        path = entry["video"]
        if path.startswith(old_root):
            entry["video"] = new_root + path[len(old_root) :]
        out.append(entry)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="local dataset root holding the downloaded videos")
    ap.add_argument("--old-root", default=DEFAULT_OLD_ROOT, help="prefix to replace")
    ap.add_argument("--data-dir", default="data", help="directory containing the parquets")
    ap.add_argument("--check-exists", action="store_true", help="report rows whose video file is missing")
    args = ap.parse_args()

    new_root = args.root.rstrip("/")
    files = sorted(glob.glob(os.path.join(args.data_dir, "*.parquet")))
    if not files:
        raise SystemExit(f"no parquet found under {args.data_dir}")

    for path in files:
        df = pd.read_parquet(path)
        if "videos" not in df.columns:
            print(f"[skip] {path}: no 'videos' column")
            continue
        df["videos"] = df["videos"].map(lambda c: rewrite_videos(c, args.old_root, new_root))
        df.to_parquet(path)

        msg = f"[ok] {path}: {len(df)} rows -> {new_root}"
        if args.check_exists:
            missing = sum(not os.path.exists(dict(list(c)[0])["video"]) for c in df["videos"])
            msg += f"  (missing videos: {missing})"
        print(msg)


if __name__ == "__main__":
    main()
