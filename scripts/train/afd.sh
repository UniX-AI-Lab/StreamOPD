#!/usr/bin/env bash
# AFD: asymmetric frame-budget distillation (grounded-but-passive privilege baseline).
#
# The teacher sees a frame *superset* of what the student saw: every student frame is kept
# and INSERT_PER_GAP extra frames are interpolated into each gap. Keeping the student's own
# frames is what makes the privilege recoverable — the teacher's view strictly contains the
# student's. INSERT_PER_GAP=0 degenerates to standard OPD.
#
# The teacher context is sized independently of the student's, because the denser frame
# sequence produces a much longer prompt.

: "${TEACHER_MAX_MODEL_LEN:=32768}"

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

: "${AFD_ENABLED:=True}"
: "${INSERT_PER_GAP:=1}"          # frames inserted between adjacent student frames
: "${TEACHER_MAX_FRAMES:=64}"     # cap; when exceeded only interpolated frames are dropped
: "${EXPERIMENT_NAME:=qwen35_4b_afd_n${ROLLOUT_N}_k1pg_superset_r${INSERT_PER_GAP}_t${TEACHER_MAX_FRAMES}_25k_ep${TOTAL_EPOCHS}}"

preflight
build_common_args
build_teacher_args

banner "AFD — asymmetric frame-budget distillation (frame superset)" \
    "insert/gap : $INSERT_PER_GAP" \
    "teacher cap: $TEACHER_MAX_FRAMES frames / $TEACHER_MAX_MODEL_LEN tokens"

exec $PYTHON -m verl.trainer.main_ppo \
    "${COMMON_ARGS[@]}" \
    "${TEACHER_ARGS[@]}" \
    "+distillation.teacher_frame_budget={enabled:${AFD_ENABLED},frame_mode:superset,insert_per_gap:${INSERT_PER_GAP},max_frames:${TEACHER_MAX_FRAMES}}" \
    "$@"
