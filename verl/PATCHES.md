# StreamOPD modifications to verl

This directory is a vendored copy of [verl](https://github.com/volcengine/verl) (0.8.0.dev,
Apache-2.0) carrying the changes StreamOPD needs. Everything listed here is gated behind a
config flag that defaults to off, so with the flags unset the training loop behaves exactly
like upstream verl.

Search the sources for the bracketed tags below (`[Cue-Gate]`, `[ExOPD]`, `[AFD]`,
`[ViCue]`, `[V-Zero]`, `[DAD]`) to find every touched region.

## New modules

| Path | Purpose |
|------|---------|
| `verl/trainer/distillation/losses.py` | On-policy distillation losses (k1/k2/k3 reverse KL, policy-gradient form) and the advantage gating applied by CueGate, ExOPD, DAD and V-Zero |
| `verl/trainer/distillation/evidence_weighting.py` | Turns per-token teacher log-prob gaps into per-sequence gate weights: `compute_cue_gate_weights` (CueGate) and `compute_relative_evidence_opd_weights` (V-Zero) |
| `verl/experimental/teacher_loop/teacher_manager.py` | Routes log-prob requests to the frozen teacher pool; extracts response-only log probs and pads them onto the student sequence |
| `verl/experimental/teacher_loop/teacher_model.py` | Teacher inference server wrapper |
| `verl/workers/config/distillation.py` | All distillation config: teacher pool, loss mode, and the CueGate / ExOPD / AD-ExOPD / DAD / V-Zero / AFD flags |
| `verl/utils/reward_score/thinkstream.py` | Verifiable reward for the three answer formats: multiple choice, binary yes/no, and counting |

## Modified upstream files

| Path | Change |
|------|--------|
| `verl/experimental/agent_loop/agent_loop.py` | Tolerates undecodable videos instead of failing the batch; builds the teacher's own prompt when a cue is in use (`_build_teacher_prompt_inputs`); runs the extra no-cue teacher pass for CueGate; re-decodes dense frames for AFD; fetches frozen-base log probs for ExOPD |
| `verl/experimental/teacher_loop/teacher_manager.py` | Response-only log-prob extraction, needed because the teacher's prompt length differs from the student's whenever the teacher is conditioned differently (cue, denser frames) |
| `verl/trainer/ppo/ray_trainer.py` | Computes the CueGate / evidence / difficulty weights once per batch and stores them for the loss |
| `verl/utils/dataset/rl_dataset.py` | Wraps `process_vision_info` in try/except to skip corrupted videos; reads the optional `teacher_prompt_key` column |
| `verl/utils/reward_score/__init__.py` | Registers the `thinkstream_rlvr`, `streamingbench` and `ovo-bench` reward functions |

## Enabling the features

| Feature | Flag |
|---------|------|
| On-policy distillation | `distillation.enabled=True` |
| Teacher conditioned on a visual cue (ViCue) | `+data.teacher_prompt_key=teacher_prompt` |
| CueGate | the above, plus `+distillation.distillation_loss.cue_gate_enabled=True` |
| ExOPD | `+distillation.distillation_loss.exopd_enabled=True` + `+distillation.base_model.*` |
| AFD | `+distillation.teacher_frame_budget={enabled:True,frame_mode:superset,...}` |
| DAD | `+distillation.distillation_loss.difficulty_adaptive_enabled=True` |
| V-Zero evidence gating | `+distillation.distillation_loss.evidence_gate_enabled=True` |

`cue_gate_alpha=0` makes the gate identically 1, which reproduces ViCue exactly; likewise
`exopd_lambda=1` reproduces standard OPD and `insert_per_gap=0` disables AFD.
