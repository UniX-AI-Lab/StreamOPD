#!/usr/bin/env bash
# ExOPD: extrapolated on-policy distillation (ungrounded privilege baseline).
#
#     advantage = -(student - base) + lambda * (teacher - base)
#
# lambda=1 recovers standard OPD; lambda>1 extrapolates past the teacher along the
# base->teacher direction. It needs a third frozen model (the student's initial weights),
# so the inference pool is split into 2 GPUs for the 9B teacher and 2 for the 4B base.

: "${TEACHER_TP:=2}"
: "${TEACHER_GPU_MEM_UTIL:=0.75}"

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

: "${BASE_MODEL:=$STUDENT_MODEL}"
: "${BASE_TP:=2}"
: "${BASE_GPU_MEM_UTIL:=0.75}"
: "${EXOPD_LAMBDA:=1.25}"
: "${EXPERIMENT_NAME:=qwen35_4b_exopd_n${ROLLOUT_N}_k1pg_lambda${EXOPD_LAMBDA}_25k_ep${TOTAL_EPOCHS}}"

preflight
build_common_args
build_teacher_args

banner "ExOPD — extrapolated on-policy distillation" \
    "base model : $BASE_MODEL (frozen)" \
    "lambda     : $EXOPD_LAMBDA"

exec $PYTHON -m verl.trainer.main_ppo \
    "${COMMON_ARGS[@]}" \
    "${TEACHER_ARGS[@]}" \
    +distillation.distillation_loss.exopd_enabled=True \
    +distillation.distillation_loss.exopd_lambda="${EXOPD_LAMBDA}" \
    +distillation.base_model.model_path="$BASE_MODEL" \
    +distillation.base_model.num_replicas=1 \
    +distillation.base_model.inference.name=vllm \
    +distillation.base_model.inference.tensor_model_parallel_size="${BASE_TP}" \
    +distillation.base_model.inference.gpu_memory_utilization="${BASE_GPU_MEM_UTIL}" \
    +distillation.base_model.inference.max_model_len="${MAX_NUM_TOKENS}" \
    "$@"
