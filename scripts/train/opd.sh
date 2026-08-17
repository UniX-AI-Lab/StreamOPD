#!/usr/bin/env bash
# Standard on-policy distillation: Qwen3.5-4B student <- Qwen3.5-9B teacher.
#
# The student rolls out on-policy, the frozen teacher scores those same responses, and the
# reverse KL (loss_mode=k1, applied through the policy gradient) is the training signal.
# This is the main distillation baseline of the paper: 25k merged data, ~3.1k steps.

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

: "${EXPERIMENT_NAME:=qwen35_4b_opd_n${ROLLOUT_N}_k1pg_25k_ep${TOTAL_EPOCHS}}"

preflight
build_common_args
build_teacher_args

banner "On-Policy Distillation (OPD)" "distill    : ${DISTILLATION_LOSS_MODE} + policy gradient"

exec $PYTHON -m verl.trainer.main_ppo "${COMMON_ARGS[@]}" "${TEACHER_ARGS[@]}" "$@"
