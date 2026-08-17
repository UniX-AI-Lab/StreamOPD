#!/usr/bin/env bash
# ViCuR (cue-only): teacher conditioned on a grounded spatio-temporal cue, no gating.
#
# This is ViCuR's teacher-side visual cue without its student-side cue recovery module, so
# the student and the inference path stay identical to standard OPD and the comparison is
# single-variable. It is also what ST-CueGate reduces to at CUE_GATE_ALPHA=0: the cue is
# supplied uniformly to every rollout instead of being weighted by how much it actually
# moved the teacher.
#
# Rows whose cue was rejected during screening keep `teacher_prompt == prompt` and
# therefore train exactly like standard OPD.

: "${TRAIN_DATA:=train25k_with_cue_instruct.parquet}"

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

: "${TEACHER_PROMPT_KEY:=teacher_prompt}"
: "${EXPERIMENT_NAME:=qwen35_4b_vicur_cueonly_n${ROLLOUT_N}_k1pg_25k_ep${TOTAL_EPOCHS}}"

preflight
build_common_args
build_teacher_args

banner "ViCuR (cue-only) — passive visual-cue teacher conditioning" \
    "teacher key: $TEACHER_PROMPT_KEY"

exec $PYTHON -m verl.trainer.main_ppo \
    "${COMMON_ARGS[@]}" \
    "${TEACHER_ARGS[@]}" \
    +data.teacher_prompt_key="${TEACHER_PROMPT_KEY}" \
    "$@"
