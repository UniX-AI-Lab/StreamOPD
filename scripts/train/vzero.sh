#!/usr/bin/env bash
# V-Zero style contrastive evidence gating, adapted to video (the negative-view baseline).
#
# Mechanically identical to ST-CueGate — a teacher likelihood contrast between two views,
# aggregated and used to reweight the OPD advantage — but the negative view is a *degraded
# video* (temporally shuffled frames) rather than the un-cued prompt. It is the control that
# shows the gate needs a cue-specific reference: holding frames and response fixed and
# removing only the cue works, while corrupting the video does not.
#
# NEG_VIEW_MODE selects the degradation: shuffle (reported), black, or textonly.

: "${ROLLOUT_N:=4}"
: "${PPO_MAX_TOKEN_LEN_PER_GPU:=16384}"

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

: "${NEG_VIEW_MODE:=shuffle}"
: "${EVIDENCE_GATE_ALPHA:=0.5}"
: "${EVIDENCE_GATE_GAMMA:=1.0}"
: "${EVIDENCE_GATE_W_MIN:=0.0}"
: "${EVIDENCE_GATE_W_MAX:=2.0}"
: "${EXPERIMENT_NAME:=qwen35_4b_vzero_n${ROLLOUT_N}_k1pg_${NEG_VIEW_MODE}_25k_ep${TOTAL_EPOCHS}}"

preflight
build_common_args
build_teacher_args

banner "V-Zero — contrastive evidence gating (degraded-video negative view)" \
    "neg view   : $NEG_VIEW_MODE" \
    "gate       : alpha=$EVIDENCE_GATE_ALPHA gamma=$EVIDENCE_GATE_GAMMA w=[$EVIDENCE_GATE_W_MIN,$EVIDENCE_GATE_W_MAX]"

exec $PYTHON -m verl.trainer.main_ppo \
    "${COMMON_ARGS[@]}" \
    "${TEACHER_ARGS[@]}" \
    +distillation.distillation_loss.contrastive_gate_enabled=True \
    +distillation.distillation_loss.neg_view_mode="${NEG_VIEW_MODE}" \
    +distillation.distillation_loss.evidence_gate_alpha="${EVIDENCE_GATE_ALPHA}" \
    +distillation.distillation_loss.evidence_gate_gamma="${EVIDENCE_GATE_GAMMA}" \
    +distillation.distillation_loss.evidence_gate_w_min="${EVIDENCE_GATE_W_MIN}" \
    +distillation.distillation_loss.evidence_gate_w_max="${EVIDENCE_GATE_W_MAX}" \
    "$@"
