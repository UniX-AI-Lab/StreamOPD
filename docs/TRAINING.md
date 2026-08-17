# Training

Every launcher in `scripts/train/` sources `common.sh`, which holds the shared environment,
the project defaults, and the hydra overrides common to all methods. A launcher only adds
what makes its method different.

```bash
STUDENT_MODEL=Qwen/Qwen3.5-4B TEACHER_MODEL=Qwen/Qwen3.5-9B \
    bash scripts/train/cuegate.sh
```

Any variable in `common.sh` can be overridden from the environment, and any extra argument
is forwarded to hydra:

```bash
TOTAL_EPOCHS=2 ACTOR_LR=5e-7 bash scripts/train/opd.sh trainer.save_freq=200
```

Logging goes to the console unless `WANDB_API_KEY` is exported, in which case the run also
reports to Weights & Biases under `WANDB_PROJECT` (default `streamopd`).

## The launchers

| Script | Method | Data | Distinguishing config |
|--------|--------|------|-----------------------|
| `opd.sh` | on-policy distillation | 25k | — (the distillation baseline) |
| `cuegate.sh` | **ST-CueGate** | 25k cue-instruct | `cue_gate_enabled=True`, `alpha`, `group_mode` |
| `vicur.sh` | ViCuR cue-only, no gate | 25k cue-instruct | `teacher_prompt_key` only |
| `exopd.sh` | extrapolated distillation | 25k | `exopd_lambda`, frozen base model |
| `vzero.sh` | contrastive evidence gating | 25k | `contrastive_gate_enabled`, `neg_view_mode` |
| `afd.sh` | asymmetric frame budget | 25k | `teacher_frame_budget`, `insert_per_gap` |
| `grpo.sh` | pure RL, no teacher | 8.3k | `distillation.enabled=False`, all 8 GPUs to the student |

The headline ST-CueGate configuration is n=4 with uid-level grouping:

```bash
ROLLOUT_N=4 CUE_GATE_GROUP_MODE=uid \
PPO_MAX_TOKEN_LEN_PER_GPU=16384 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    bash scripts/train/cuegate.sh
```

`uid` mode needs n>1: with a single rollout per prompt every group is a singleton, the
within-group z-score is 0, and the gate collapses to a constant. The `n=1` default therefore
uses batch-level grouping instead.

**On-policy self-distillation (OPSD)** needs no separate script — point the teacher at the
student's own initial policy. This is the variant that recovers abstention (HLD 57.0):

```bash
TEACHER_MODEL=Qwen/Qwen3.5-4B ROLLOUT_N=4 CUE_GATE_GROUP_MODE=uid \
    bash scripts/train/cuegate.sh          # ST-CueGate (OPSD)
TEACHER_MODEL=Qwen/Qwen3.5-4B bash scripts/train/opd.sh   # plain OPSD
```

Reproducing other rows of the ablation tables is a matter of environment variables:

```bash
ROLLOUT_N=4 bash scripts/train/opd.sh                    # OPD (n=4)
CUE_GATE_ALPHA=0.25 bash scripts/train/cuegate.sh        # gate-strength ablation
CUE_GATE_LEVEL=token bash scripts/train/cuegate.sh       # per-token instead of per-response
TEACHER_MODEL=Qwen/Qwen3.5-27B bash scripts/train/cuegate.sh   # teacher-scale ablation
NEG_VIEW_MODE=black bash scripts/train/vzero.sh          # alternative negative view
```

## Configuration that must not regress

These are not tuning knobs; changing them breaks the run outright.

| Setting | Value | Why |
|---------|-------|-----|
| `actor.fsdp_config.param_offload` / `optimizer_offload` | `True` | removing either OOMs immediately |
| `rollout.gpu_memory_utilization` | `0.3` | student vLLM shares the GPU with FSDP training |
| teacher `gpu_memory_utilization` | `0.70` | raising it OOMs |
| `data.filter_overlong_prompts` | `False` | `True` single-process decodes every video at startup, turning 10 minutes into 2 hours |
| `data.max_prompt_length` | `16000`, `truncation=left` | long video prompts |
| `trainer.default_local_dir` | absolute path | otherwise checkpoints land in the working directory |

## Resource layout

8 GPUs, split 4 student / 4 teacher, all inside one Ray node. The student runs vLLM rollout
at TP=2 alongside FSDP training; the teacher runs frozen vLLM inference at TP=4. ExOPD is
the exception: its inference pool holds both the 9B teacher (TP=2) and the frozen 4B base
(TP=2). GRPO gives all 8 GPUs to the student.

## Running and monitoring

```bash
mkdir -p logs
nohup bash scripts/train/cuegate.sh > logs/cuegate.log 2>&1 &
```

Startup is Ray init → dataset build → model load (5-15 min) → `val_before_train` baseline
(~60%) → step 1.

```bash
# model loading
grep -E "Loading weights|Loading safetensors" logs/cuegate.log
# the step-0 baseline
grep "val-core/.*acc/mean@1" logs/cuegate.log | head -1
# training actually started
grep "training/global_step:1" logs/cuegate.log
# real errors, filtering out the noisy video-decode warnings
grep -iE "traceback|out of memory|cuda error|assertionerror" logs/cuegate.log \
    | grep -viE "decode video|video_reader|threaded_decoder|UserWarning|FutureWarning"
```

This message appears dozens of times per step and is **normal** — it is the response-only
log-prob extraction doing its job whenever the teacher's prompt length differs from the
student's:

```
[TeacherLoop] Multimodal length mismatch: teacher_ids=..., align_len=L, response_length=R.
              Extracting response only.
```

Useful metrics: `val-core/thinkstream_rlvr/acc/mean@1`, `actor/distillation/abs_loss`,
`actor/distillation/loss`, and the teacher-student log-prob gap diagnostics
`distillation/ts_logprob_gap`, `ts_gap_front50`, `ts_gap_back50`.

## Step times

| Configuration | Per step | Total steps | Wall clock |
|---------------|---------:|------------:|-----------:|
| OPD n=1, 25k | 35-50 s | ~3,140 | 18-25 h |
| ST-CueGate n=1, 25k | 45-65 s | ~3,140 | 24-32 h |
| ST-CueGate n=4, 25k | 150-200 s | early peak | stop at the val peak |
| GRPO n=8, 8.3k | ~120 s | ~2,600 | — |

The extra no-cue teacher forward that ST-CueGate needs costs 20-35% per step and no extra
GPUs: it reuses the student's decoded frames and the same teacher pool.

## Choosing a checkpoint

`save_freq=100` keeps every hundredth step. Pick the step with the highest held-out
validation accuracy:

```bash
grep -oE "step:[0-9]+ -.*val-core/thinkstream_rlvr/acc/mean@1:np.float64\([0-9.]+\)" logs/cuegate.log \
    | grep -oE "^step:[0-9]+|acc/mean@1:np.float64\([0-9.]+\)" | paste - -
```

Validation is noisy at n=1, so evaluate two or three checkpoints around the peak rather
than trusting a single point. Note also that validation runs with thinking on while the
benchmarks run in instruct mode, so it is a signal for ranking checkpoints, not a predictor
of the downstream number.

There is no need to run all 4 epochs. Validation typically peaks partway through epoch 2 —
around step 1400 for OPD on 25k, step 2600 for ST-CueGate n=1, and much earlier, near step
600-800, for ST-CueGate n=4 — and declines afterwards. Stop once the peak is a few
checkpoints behind you.

Reported results are means over three seeds, selecting one checkpoint per run on the
held-out validation aggregate and evaluating that same checkpoint on all four benchmarks.
Single-run numbers move by a few tenths, so treat a sub-point difference between two
configurations as noise unless it is reproduced across seeds.

## Multi-machine runs

Independent experiments on separate machines only collide through names: give each run a
distinct `EXPERIMENT_NAME`, since it determines the checkpoint directory, the log name and
the W&B run identity. Stagger the starts by 5-10 minutes if both read models from the same
shared filesystem.
