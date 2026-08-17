#!/usr/bin/env bash
# Run the four benchmarks of the paper in parallel, one per GPU.
#
#   bash scripts/eval/run_all.sh <MODEL_PATH> <RUN_NAME> [GPU_IDS]
#   bash scripts/eval/run_all.sh checkpoints/<exp>/global_step_1400/actor/huggingface opd_25k_step1400 0,1,2,3
#
#   GPU 0 -> LongVideoBench (lmms-eval)
#   GPU 1 -> StreamingBench (recent-window, instruct mode)
#   GPU 2 -> OVO-Bench      (recent-window, instruct mode)
#   GPU 3 -> VideoMME       (lmms-eval)
#
# MODEL_PATH must already be in HuggingFace format; convert verl FSDP shards first with
# scripts/merge_checkpoint.sh. Benchmark videos are expected under $BENCH_ROOT, see
# docs/EVALUATION.md.
#
# lmms-eval downloads task metadata from the HuggingFace hub, so export HF_TOKEN (and a
# proxy, if your network needs one) before launching.

set -euo pipefail
cd "$(dirname "$0")/../.."

MODEL_PATH="${1:?Usage: run_all.sh <MODEL_PATH> <RUN_NAME> [GPU_IDS]}"
RUN_NAME="${2:?Usage: run_all.sh <MODEL_PATH> <RUN_NAME> [GPU_IDS]}"
GPU_IDS="${3:-0,1,2,3}"

IFS=',' read -ra GPUS <<< "$GPU_IDS"
GPU_LVB="${GPUS[0]:-0}"
GPU_SB="${GPUS[1]:-1}"
GPU_OVO="${GPUS[2]:-2}"
GPU_VME="${GPUS[3]:-3}"

PYTHON=${PYTHON:-python3}
BENCH_ROOT=${BENCH_ROOT:-data/benchmarks}
RESULTS=${RESULTS:-results}
LOGS=${LOGS:-logs/eval}
MAX_NUM_FRAMES=${MAX_NUM_FRAMES:-32}
RECENT_FRAMES=${RECENT_FRAMES:-4}

# Pinned explicitly because the vision budget changes the scores and different lmms-eval
# releases ship different defaults for it. These are the values the reported numbers used.
MIN_PIXELS=${MIN_PIXELS:-$((64 * 32 * 32))}
MAX_PIXELS=${MAX_PIXELS:-$((128 * 32 * 32))}
TOTAL_PIXELS=${TOTAL_PIXELS:-$((224 * 1024 * 32 * 32))}
LMMS_MODEL_ARGS="pretrained=${MODEL_PATH},enable_thinking=False,max_num_frames=${MAX_NUM_FRAMES}"
LMMS_MODEL_ARGS="${LMMS_MODEL_ARGS},min_pixels=${MIN_PIXELS},max_pixels=${MAX_PIXELS},total_pixels=${TOTAL_PIXELS}"

export FORCE_QWENVL_VIDEO_READER=${FORCE_QWENVL_VIDEO_READER:-decord}
mkdir -p "$LOGS" "$RESULTS"

echo "============================================================"
echo "model : $MODEL_PATH"
echo "run   : $RUN_NAME"
echo "gpus  : LVB=$GPU_LVB SB=$GPU_SB OVO=$GPU_OVO VideoMME=$GPU_VME"
echo "============================================================"

CUDA_VISIBLE_DEVICES=$GPU_LVB nohup $PYTHON -m lmms_eval \
    --model qwen3_5 \
    --model_args "$LMMS_MODEL_ARGS" \
    --tasks longvideobench_val_v \
    --batch_size 1 \
    --output_path "$RESULTS/lmms_eval/longvideobench_${RUN_NAME}" \
    > "$LOGS/${RUN_NAME}_longvideobench.log" 2>&1 &
PID_LVB=$!

CUDA_VISIBLE_DEVICES=$GPU_SB nohup $PYTHON -m streamopd.eval.streamingbench \
    --anno-path "$BENCH_ROOT/streamingbench/questions_real.json" \
    --video-dir "$BENCH_ROOT/streamingbench/videos" \
    --qa-model "$MODEL_PATH" \
    --top-k 0 --recent-frames-only "$RECENT_FRAMES" --chunk-duration 1.0 --fps 1.0 \
    --output-dir "$RESULTS/${RUN_NAME}_streamingbench" \
    > "$LOGS/${RUN_NAME}_streamingbench.log" 2>&1 &
PID_SB=$!

CUDA_VISIBLE_DEVICES=$GPU_OVO nohup $PYTHON -m streamopd.eval.ovo_bench \
    --model_path "$MODEL_PATH" \
    --anno_path "$BENCH_ROOT/ovo_bench/ovo_bench_new.json" \
    --chunked_dir "$BENCH_ROOT/ovo_bench/chunked_videos" \
    --result_dir "$RESULTS/${RUN_NAME}_ovo" \
    --recent_frames_only "$RECENT_FRAMES" \
    > "$LOGS/${RUN_NAME}_ovo.log" 2>&1 &
PID_OVO=$!

CUDA_VISIBLE_DEVICES=$GPU_VME nohup $PYTHON -m lmms_eval \
    --model qwen3_5 \
    --model_args "$LMMS_MODEL_ARGS" \
    --tasks videomme \
    --batch_size 1 \
    --output_path "$RESULTS/lmms_eval/videomme_${RUN_NAME}" \
    > "$LOGS/${RUN_NAME}_videomme.log" 2>&1 &
PID_VME=$!

echo "launched: LVB=$PID_LVB SB=$PID_SB OVO=$PID_OVO VideoMME=$PID_VME"
echo "monitor : tail -f $LOGS/${RUN_NAME}_*.log"
echo "then    : bash scripts/eval/score_all.sh $RUN_NAME"

wait $PID_LVB $PID_SB $PID_OVO $PID_VME
echo "all benchmarks finished"
