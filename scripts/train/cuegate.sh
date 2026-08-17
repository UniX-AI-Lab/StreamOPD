#!/usr/bin/env bash
# ST-CueGate: spatio-temporal cue gating, the paper's teacher-privilege extension.
#
# The teacher is conditioned on a grounded spatio-temporal cue (the `teacher_prompt` column
# of train25k_with_cue_instruct.parquet); the student prompt is untouched. A second teacher
# forward scores the same response on the *un-cued* prompt, and the nested contrast
#     delta_t = log p_teacher(y_t | q, c) - log p_teacher(y_t | q)
# is averaged into a response-level proxy, standardised within a comparison group, and used
# to reweight the OPD advantage. Responses the cue helped more than their peers are
# up-weighted; the ranking is relative, not absolute.
#
# CUE_GATE_ALPHA=0 gives w == 1 and reproduces ViCuR cue-only exactly (scripts/train/vicur.sh).
#
# Headline configuration (n=4, uid grouping):
#   ROLLOUT_N=4 CUE_GATE_GROUP_MODE=uid PPO_MAX_TOKEN_LEN_PER_GPU=16384 \
#   PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True bash scripts/train/cuegate.sh
#
# uid grouping standardises within each prompt's sibling rollouts, so it needs n>1: at n=1
# every group is a singleton and the gate collapses to the neutral value.
#
# On-policy self-distillation (OPSD) is the same script with the teacher pointed at the
# student's initial policy: TEACHER_MODEL=Qwen/Qwen3.5-4B bash scripts/train/cuegate.sh

: "${TRAIN_DATA:=train25k_with_cue_instruct.parquet}"

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

: "${TEACHER_PROMPT_KEY:=teacher_prompt}"
: "${CUE_GATE_ALPHA:=0.5}"              # gate strength; 0 -> exactly ViCuR cue-only
: "${CUE_GATE_GAMMA:=1.0}"              # penalty applied to negative delta
: "${CUE_GATE_W_MIN:=0.0}"
: "${CUE_GATE_W_MAX:=2.0}"
: "${CUE_GATE_GROUP_MODE:=batch}"       # batch (any n) | uid (needs n>1)
: "${CUE_GATE_LEVEL:=seq}"              # seq = one gate per response | token = per-token gates

: "${EXPERIMENT_NAME:=qwen35_4b_cuegate_n${ROLLOUT_N}_k1pg_alpha${CUE_GATE_ALPHA}_group${CUE_GATE_GROUP_MODE}_${CUE_GATE_LEVEL}_cueinstruct_25k_ep${TOTAL_EPOCHS}}"

preflight
build_common_args
build_teacher_args

banner "ST-CueGate — nested cue-removal gating" \
    "cue gate   : alpha=$CUE_GATE_ALPHA gamma=$CUE_GATE_GAMMA w=[$CUE_GATE_W_MIN,$CUE_GATE_W_MAX] group=$CUE_GATE_GROUP_MODE level=$CUE_GATE_LEVEL" \
    "note       : the extra no-cue teacher forward costs ~20-35% more time per step"

exec $PYTHON -m verl.trainer.main_ppo \
    "${COMMON_ARGS[@]}" \
    "${TEACHER_ARGS[@]}" \
    +data.teacher_prompt_key="${TEACHER_PROMPT_KEY}" \
    +distillation.distillation_loss.cue_gate_enabled=True \
    +distillation.distillation_loss.cue_gate_alpha="${CUE_GATE_ALPHA}" \
    +distillation.distillation_loss.cue_gate_gamma="${CUE_GATE_GAMMA}" \
    +distillation.distillation_loss.cue_gate_w_min="${CUE_GATE_W_MIN}" \
    +distillation.distillation_loss.cue_gate_w_max="${CUE_GATE_W_MAX}" \
    +distillation.distillation_loss.cue_gate_group_mode="${CUE_GATE_GROUP_MODE}" \
    +distillation.distillation_loss.cue_gate_level="${CUE_GATE_LEVEL}" \
    "$@"
