#!/usr/bin/env bash
# Collect the numbers produced by scripts/eval/run_all.sh into one summary.
#
#   bash scripts/eval/score_all.sh <RUN_NAME>

set -euo pipefail
cd "$(dirname "$0")/../.."

RUN_NAME="${1:?Usage: score_all.sh <RUN_NAME>}"
PYTHON=${PYTHON:-python3}
RESULTS=${RESULTS:-results}

$PYTHON - "$RESULTS" "$RUN_NAME" <<'PY'
import glob
import json
import os
import subprocess
import sys

base, name = sys.argv[1], sys.argv[2]


def section(title):
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


def lmms_metric(task_dir, metric):
    """Read one metric out of an lmms-eval output directory."""
    files = glob.glob(f"{task_dir}/*/*.json") if os.path.isdir(task_dir) else []
    for path in sorted(files):
        stem = os.path.basename(path)
        if "results" not in stem or "samples" in stem:
            continue
        payload = json.load(open(path)).get("results", {})
        for value in payload.values():
            if metric in value:
                return value[metric]
    return None


section("StreamingBench")
sb = f"{base}/{name}_streamingbench/results_incremental.jsonl"
if os.path.exists(sb):
    rows = [json.loads(line) for line in open(sb)]
    hit = sum(1 for r in rows if r.get("correct"))
    print(f"  accuracy: {hit}/{len(rows)} = {hit / len(rows) * 100:.2f}%")
else:
    print("  not found")

section("OVO-Bench")
ovo_dir = f"{base}/{name}_ovo"
ovo_files = glob.glob(f"{ovo_dir}/*.json") if os.path.isdir(ovo_dir) else []
if ovo_files:
    subprocess.run(
        [sys.executable, "-m", "streamopd.scoring.ovo_bench", "--result_path", sorted(ovo_files)[0]],
        check=False,
    )
else:
    print("  not found")

section("lmms-eval")
vme = lmms_metric(f"{base}/lmms_eval/videomme_{name}", "videomme_perception_score,none")
lvb = lmms_metric(f"{base}/lmms_eval/longvideobench_{name}", "lvb_acc,none")
print(f"  VideoMME:       {vme:.2f}%" if isinstance(vme, float) else "  VideoMME:       not found")
print(f"  LongVideoBench: {lvb * 100:.2f}%" if isinstance(lvb, float) else "  LongVideoBench: not found")
PY
