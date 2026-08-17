#!/usr/bin/env bash
# GRPO: pure RL with verifiable rewards, no teacher. The reference point for
# "post-training without distillation", and the run that exposes the response-format drift
# discussed in the paper.
#
# All 8 GPUs go to the student because there is no teacher pool. Defaults follow the run
# reported in the paper: n=8 samples, 8.3k pass-rate-filtered data, 10 epochs.

: "${TRAIN_DATA:=train20k_filtered_8343.parquet}"
: "${STUDENT_GPUS:=8}"
: "${ROLLOUT_N:=8}"
: "${TOTAL_EPOCHS:=10}"
USE_TEACHER=0

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

: "${KL_LOSS_COEF:=0.001}"
: "${KL_LOSS_TYPE:=low_var_kl}"
: "${ENTROPY_COEF:=0.0}"
: "${EXPERIMENT_NAME:=qwen35_4b_grpo_n${ROLLOUT_N}_bs${TRAIN_BATCH_SIZE}_lr${ACTOR_LR}_8k_ep${TOTAL_EPOCHS}}"

preflight
build_common_args

banner "GRPO — pure RL with verifiable rewards (no teacher)" \
    "kl loss    : $KL_LOSS_COEF ($KL_LOSS_TYPE)   entropy: $ENTROPY_COEF"

exec $PYTHON -m verl.trainer.main_ppo \
    "${COMMON_ARGS[@]}" \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef="${KL_LOSS_COEF}" \
    actor_rollout_ref.actor.kl_loss_type="${KL_LOSS_TYPE}" \
    actor_rollout_ref.actor.entropy_coeff="${ENTROPY_COEF}" \
    trainer.validation_data_dir="${OUTPUT_DIR}/val_rollouts" \
    trainer.log_val_generations=20 \
    distillation.enabled=False \
    "$@"
