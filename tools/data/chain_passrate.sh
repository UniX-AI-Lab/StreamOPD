#!/usr/bin/env bash
# Run pass_rate_vllm.py over several seeds back to back against one already-running vLLM
# server, so a GPU keeps producing pass-rate samples without supervision.
#
#   bash tools/data/chain_passrate.sh <role> <port> <model> <seed1,seed2,...> [limit] [data]
#   bash tools/data/chain_passrate.sh student 8001 Qwen/Qwen3.5-4B 111,222,333 3000
#   bash tools/data/chain_passrate.sh teacher 8002 Qwen/Qwen3.5-9B 111,222,333 3000
#
# One jsonl per seed lands in $OUT_DIR. pass_rate_vllm.py resumes, so re-running a finished
# seed costs nothing.

set -uo pipefail
cd "$(dirname "$0")/../.."

export FORCE_QWENVL_VIDEO_READER=${FORCE_QWENVL_VIDEO_READER:-decord}
PYTHON=${PYTHON:-python3}
OUT_DIR=${OUT_DIR:-data/passrate}

ROLE=$1; PORT=$2; MODEL=$3; SEEDS=$4
LIMIT=${5:-3000}
DATA=${6:-data/filtered8k_plus_cot_verifiable_dedup_25118.parquet}

mkdir -p "$OUT_DIR"
IFS=',' read -ra SEED_ARR <<< "$SEEDS"

echo "[chain] role=$ROLE port=$PORT model=$MODEL seeds=$SEEDS limit=$LIMIT data=$DATA"
for seed in "${SEED_ARR[@]}"; do
    OUT="$OUT_DIR/${ROLE}_n${LIMIT}_s${seed}.jsonl"
    done_n=$(wc -l < "$OUT" 2>/dev/null || echo 0)
    if [ "$done_n" -ge "$LIMIT" ]; then
        echo "[chain] seed=$seed already complete ($done_n rows), skipping"
        continue
    fi
    echo "[chain] seed=$seed starting ($done_n rows already present)"
    $PYTHON tools/data/pass_rate_vllm.py \
        --role "$ROLE" --port "$PORT" --model-name "$MODEL" \
        --data-files "$DATA" \
        --limit "$LIMIT" --seed "$seed" --n-samples 8 --fps 2 --thinking \
        --max-new-tokens 1024 --concurrency 8 \
        --out "$OUT"
    echo "[chain] seed=$seed done"
done
echo "[chain] all seeds finished: $SEEDS"
