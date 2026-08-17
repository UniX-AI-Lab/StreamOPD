#!/usr/bin/env bash
# Shared setup for every StreamOPD training launcher. Sourced by scripts/train/*.sh.
#
# Each launcher declares its own defaults with `: "${VAR:=...}"` *before* sourcing this
# file; the values here are the project-wide fallbacks. Anything exported in the
# environment wins over both, e.g.
#   STUDENT_MODEL=/local/Qwen3.5-4B TOTAL_EPOCHS=2 bash scripts/train/opd.sh
#
# Expects 8 GPUs: 0-3 run the student (vLLM rollout TP=2 + FSDP training), 4-7 run the
# frozen teacher (vLLM inference TP=4).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# ---------------------------------------------------------------- models
: "${STUDENT_MODEL:=Qwen/Qwen3.5-4B}"
: "${TEACHER_MODEL:=Qwen/Qwen3.5-9B}"

# ---------------------------------------------------------------- data / output
: "${DATA_DIR:=$REPO_ROOT/data}"
: "${OUTPUT_ROOT:=$REPO_ROOT/checkpoints}"
: "${TRAIN_DATA:=filtered8k_plus_cot_verifiable_dedup_25118.parquet}"
: "${VAL_DATA:=rlvr_val.parquet}"
: "${TRAIN_FILE:=$DATA_DIR/$TRAIN_DATA}"
: "${VAL_FILE:=$DATA_DIR/$VAL_DATA}"

# ---------------------------------------------------------------- runtime
: "${PYTHON:=python3}"
export FORCE_QWENVL_VIDEO_READER="${FORCE_QWENVL_VIDEO_READER:-decord}"

# Weights & Biases is optional: export WANDB_API_KEY (plus WANDB_ENTITY / WANDB_PROJECT if
# you use them) before launching. Without a key the run logs to the console only.
if [[ -n "${WANDB_API_KEY:-}" ]]; then
    export WANDB_PROJECT="${WANDB_PROJECT:-streamopd}"
    TRAINER_LOGGER='["console","wandb"]'
else
    TRAINER_LOGGER='["console"]'
fi

# ---------------------------------------------------------------- resources
: "${NNODES:=1}"
: "${STUDENT_GPUS:=4}"
: "${TEACHER_GPUS:=4}"

# ---------------------------------------------------------------- optimisation
: "${ACTOR_LR:=1e-6}"
: "${TRAIN_BATCH_SIZE:=32}"
: "${PPO_MINI_BATCH_SIZE:=32}"
: "${TOTAL_EPOCHS:=4}"
: "${SAVE_FREQ:=100}"
: "${TEST_FREQ:=100}"
: "${VAL_BEFORE_TRAIN:=True}"

# Sequence budget. max_prompt_length has to stay generous: a 10 s clip at fps=2 already
# costs several thousand vision tokens. filter_overlong_prompts stays off because the
# pre-scan would single-process decode every video in the set.
: "${MAX_PROMPT_LENGTH:=16000}"
: "${MAX_RESPONSE_LENGTH:=512}"
: "${PPO_MAX_TOKEN_LEN_PER_GPU:=32768}"

# ---------------------------------------------------------------- rollout / teacher
: "${ROLLOUT_N:=1}"
: "${ROLLOUT_TP:=2}"
: "${ROLLOUT_GPU_MEM_UTIL:=0.3}"
: "${TEACHER_TP:=4}"
: "${TEACHER_GPU_MEM_UTIL:=0.70}"

# ---------------------------------------------------------------- distillation loss
: "${DISTILLATION_LOSS_MODE:=k1}"
: "${USE_POLICY_GRADIENT:=True}"
: "${DISTILLATION_TOPK:=64}"

: "${PROJECT_NAME:=streamopd}"

# ---------------------------------------------------------------- helpers

preflight() {
    for f in "$TRAIN_FILE" "$VAL_FILE"; do
        [[ -f "$f" ]] || { echo "ERROR: data file not found: $f" >&2; exit 1; }
    done
    $PYTHON -c "import verl; from verl.trainer.distillation.losses import is_distillation_enabled" \
        || { echo "ERROR: the patched verl in this repo is not importable; see docs/INSTALL.md" >&2; exit 1; }
}

# Fills COMMON_ARGS with the hydra overrides shared by every method and resolves
# OUTPUT_DIR / MAX_NUM_TOKENS. EXPERIMENT_NAME must be set by the caller.
build_common_args() {
    : "${EXPERIMENT_NAME:?EXPERIMENT_NAME must be set before calling build_common_args}"

    MAX_NUM_TOKENS=$(( MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH + 1 ))
    : "${OUTPUT_DIR:=$OUTPUT_ROOT/$EXPERIMENT_NAME}"
    mkdir -p "$OUTPUT_DIR"

    COMMON_ARGS=(
        algorithm.adv_estimator=grpo
        algorithm.use_kl_in_reward=False

        data.train_files="['$TRAIN_FILE']"
        data.val_files="['$VAL_FILE']"
        data.train_batch_size="${TRAIN_BATCH_SIZE}"
        data.max_prompt_length="${MAX_PROMPT_LENGTH}"
        data.max_response_length="${MAX_RESPONSE_LENGTH}"
        data.filter_overlong_prompts=False
        data.truncation=left
        data.video_key=videos

        actor_rollout_ref.model.path="$STUDENT_MODEL"
        actor_rollout_ref.model.use_remove_padding=True
        actor_rollout_ref.model.enable_gradient_checkpointing=True
        +actor_rollout_ref.model.override_config.attn_implementation=flash_attention_2

        actor_rollout_ref.actor.optim.lr="${ACTOR_LR}"
        actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}"
        actor_rollout_ref.actor.use_dynamic_bsz=True
        actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${PPO_MAX_TOKEN_LEN_PER_GPU}"
        actor_rollout_ref.actor.fsdp_config.param_offload=True
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=True

        actor_rollout_ref.rollout.name=vllm
        actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP}"
        actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEM_UTIL}"
        actor_rollout_ref.rollout.n="${ROLLOUT_N}"
        actor_rollout_ref.rollout.max_model_len="${MAX_NUM_TOKENS}"
        actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
        actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="${PPO_MAX_TOKEN_LEN_PER_GPU}"
        +actor_rollout_ref.rollout.engine_kwargs.vllm.disable_custom_all_reduce=True

        trainer.balance_batch=True
        trainer.logger="$TRAINER_LOGGER"
        trainer.project_name="${PROJECT_NAME}"
        trainer.experiment_name="${EXPERIMENT_NAME}"
        trainer.n_gpus_per_node="${STUDENT_GPUS}"
        trainer.nnodes="${NNODES}"
        trainer.default_local_dir="${OUTPUT_DIR}"
        trainer.val_before_train="${VAL_BEFORE_TRAIN}"
        trainer.save_freq="${SAVE_FREQ}"
        trainer.test_freq="${TEST_FREQ}"
        trainer.total_epochs="${TOTAL_EPOCHS}"
    )
}

# Fills TEACHER_ARGS with the frozen-teacher pool overrides. Must run after
# build_common_args (it reuses MAX_NUM_TOKENS).
build_teacher_args() {
    : "${TEACHER_MAX_MODEL_LEN:=$MAX_NUM_TOKENS}"
    TEACHER_ARGS=(
        distillation.enabled=True
        distillation.n_gpus_per_node="${TEACHER_GPUS}"
        distillation.nnodes="${NNODES}"
        distillation.teacher_models.teacher_model.model_path="$TEACHER_MODEL"
        distillation.teacher_models.teacher_model.inference.name=vllm
        distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size="${TEACHER_TP}"
        distillation.teacher_models.teacher_model.inference.gpu_memory_utilization="${TEACHER_GPU_MEM_UTIL}"
        distillation.teacher_models.teacher_model.inference.max_model_len="${TEACHER_MAX_MODEL_LEN}"
        +distillation.teacher_models.teacher_model.inference.engine_kwargs.vllm.disable_custom_all_reduce=True
        distillation.distillation_loss.loss_mode="${DISTILLATION_LOSS_MODE}"
        distillation.distillation_loss.topk="${DISTILLATION_TOPK}"
        distillation.distillation_loss.use_task_rewards=False
        distillation.distillation_loss.use_policy_gradient="${USE_POLICY_GRADIENT}"
        distillation.distillation_loss.loss_max_clamp=10.0
        distillation.distillation_loss.log_prob_min_clamp=-10.0
    )
}

banner() {
    local title="$1"; shift
    echo "============================================================"
    echo "$title"
    echo "============================================================"
    echo "student    : $STUDENT_MODEL"
    [[ "${USE_TEACHER:-1}" == "1" ]] && echo "teacher    : $TEACHER_MODEL"
    echo "train data : $TRAIN_FILE"
    echo "val data   : $VAL_FILE"
    echo "rollout.n  : $ROLLOUT_N   batch: $TRAIN_BATCH_SIZE   lr: $ACTOR_LR   epochs: $TOTAL_EPOCHS"
    for line in "$@"; do echo "$line"; done
    echo "experiment : $EXPERIMENT_NAME"
    echo "output     : $OUTPUT_DIR"
    echo "============================================================"
}
