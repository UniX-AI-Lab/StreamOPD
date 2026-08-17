# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
from dataclasses import dataclass, field
from typing import Optional

from verl.base_config import BaseConfig
from verl.utils.config import omega_conf_to_dataclass

from .rollout import RolloutConfig

__all__ = ["DistillationLossConfig", "DistillationTeacherModelConfig", "DistillationConfig"]

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@dataclass
class DistillationLossConfig(BaseConfig):
    """Configuration for distillation loss settings.

    loss_mode (str):
        Distillation loss function to use.
    topk (int, optional):
        Number of top tokens to consider for top-k distillation losses.
    use_task_rewards (bool):
        Whether to include task rewards alongside distillation loss.
    distillation_loss_coef (float):
        Coefficient for distillation loss when combined with task rewards.
    loss_max_clamp (float, optional):
        Maximum value to clamp distillation loss. If None, no clamping is applied.
    log_prob_min_clamp (float, optional):
        Minimum value to clamp log probabilities for stability, e.g., log q - log p where p or q are
        very close to zero. If None, no clamping is applied.
    use_policy_gradient (bool):
        Whether to incorporate distillation loss as a reward, as done
        by https://thinkingmachines.ai/blog/on-policy-distillation/. Recommended to use loss_mode=k1.
        Otherwise, distillation loss is directly backpropagated as a supervised loss,
        as in https://arxiv.org/abs/2306.13649. Recommended to use loss_mode=k3 or forward_kl_topk.
    policy_loss_mode (str):
        Name of the policy loss to use when use_policy_gradient is true.
    clip_ratio (float):
        PPO clipping ratio for policy loss.
    clip_ratio_low (float):
        Lower bound for PPO clipping ratio.
    clip_ratio_high (float):
        Upper bound for PPO clipping ratio.
    loss_settings (DistillationLossSettings, optional):
        Runtime-populated settings based on loss_mode. Not set by user.
    """

    loss_mode: str = "k3"
    topk: Optional[int] = 128
    use_task_rewards: bool = True
    distillation_loss_coef: float = 1.0
    loss_max_clamp: Optional[float] = 10.0
    log_prob_min_clamp: Optional[float] = -10.0

    use_policy_gradient: bool = True
    policy_loss_mode: str = "vanilla"
    clip_ratio: float = 0.2
    clip_ratio_low: float = 0.2
    clip_ratio_high: float = 0.2

    # [ExOPD] Extrapolated On-Policy Distillation (G-OPD, arXiv 2602.12125).
    # Uses a frozen base model (student initial weights) as a third reference point.
    # advantage = -(student - base) + lambda*(teacher - base)
    # lambda=1.0 degenerates to standard OPD; lambda>1 extrapolates beyond teacher.
    exopd_enabled: bool = False
    exopd_lambda: float = 1.5

    # [AD-ExOPD] Axis-Decomposed ExOPD: decompose the single extrapolation direction
    # (teacher - base) into two orthogonal axes using a third reference base_dense
    # (same 4B base model but with DENSE frames, same budget as teacher):
    #   temporal axis  = base_dense - base_sparse   (capacity fixed, more frames)
    #   capacity axis  = teacher    - base_dense    (frames fixed, bigger model)
    # advantage = -(student - base_sparse)
    #             + lambda_temporal*(base_dense - base_sparse)
    #             + lambda_capacity*(teacher - base_dense)
    # Set lambda_temporal>1 to extrapolate along the frame-budget axis (what a sliding-window
    # student can exploit) while keeping lambda_capacity<=1 (a 4B student cannot match 9B capacity).
    # lambda_temporal=lambda_capacity=exopd_lambda degenerates to standard ExOPD.
    exopd_axis_decomposed: bool = False
    exopd_lambda_temporal: float = 1.5
    exopd_lambda_capacity: float = 1.0

    # [TSKL] Temporal-Selective KL: scale advantages by sample-level temporal_weight.
    # Requires training parquet to have a `temporal_weight` column (float).
    temporal_selective_kl: bool = False
    temporal_weight_default: float = 1.0

    # [V-Zero-Video] Contrastive Evidence Gating (V-Zero, arXiv 2606.25319, adapted to video).
    # The teacher scores each student rollout under TWO views: the positive view = the full
    # video (== the standard teacher forward, reused as-is), and a NEGATIVE view = a degraded
    # video. Per token the evidence gap Delta = logp_teacher(pos) - logp_teacher(neg) measures
    # how much that token relies on genuine visual evidence; aggregated to a trajectory score
    # p^(g), standardized WITHIN each prompt's G sibling rollouts to a group-relative advantage
    # a^(g), then turned into a non-negative gate w^(g)=clip(1+a, w_min, w_max) that scales the
    # (positive-view) distillation advantage. Requires rollout.n>1 (needs sibling group for
    # mean/std). answer-label-free. The gate is computed at batch level in ray_trainer (group
    # visibility) and consumed here; the negative-view logprobs are produced in agent_loop.
    # contrastive_gate_enabled=False (default) → exact standard-OPD behavior, no negative pass.
    contrastive_gate_enabled: bool = False
    # Negative-view construction, one of:
    #   "black"    — replace the student's decoded frames with zero tensors (isolates "does this
    #                token depend on ANY visual input" → anti-hallucination). Most robust; verify first.
    #   "shuffle"  — permute the frames' temporal order (isolates "does this token depend on TEMPORAL
    #                order" → the streaming-video novelty; paper does not do this).
    #   "textonly" — drop the <video> element from the teacher prompt (video vs language-prior).
    neg_view_mode: str = "black"
    evidence_gate_w_min: float = 0.0   # V-Zero default
    evidence_gate_w_max: float = 2.0   # V-Zero default
    # V-Zero gate math (match upstream evidence_weighting defaults):
    #   token_score = relu(Δ) - gamma*relu(-Δ)   (gamma=1.0 → token_score == Δ)
    #   weight      = clip(1 + alpha * group_zscore(score), w_min, w_max)
    # NOTE: upstream V-Zero's main results use alpha=0.5 (softer gate), NOT 1.0.
    evidence_gate_gamma: float = 1.0
    evidence_gate_alpha: float = 0.5
    evidence_gate_eps: float = 1e-6

    # [DAD] On-policy Difficulty-Adaptive Distillation: reweight the distillation advantage
    # per-sample by the CURRENT student-teacher reverse-KL gap, focusing learning on samples
    # the student has not yet mastered. Mean-preserving (batch mean weight = 1) so it does not
    # change the effective learning rate, and self-anneals (uniform early, focused later).
    # difficulty_alpha=0 degenerates exactly to standard distillation.
    difficulty_adaptive_enabled: bool = False
    difficulty_alpha: float = 1.0          # weighting aggressiveness; 0 = uniform
    difficulty_weight_min: float = 0.2     # clamp floor (avoid zero-gradient samples)
    difficulty_weight_max: float = 3.0     # clamp ceil (avoid outlier domination)
    difficulty_mode: str = "exp"           # "exp" (only mode for now; reserved for rank/linear)
    # Per-batch difficulty standardization mode (decouples weighting scale from small-batch noise):
    #   "zscore" (default) — original behavior: z = (d - mean) / std. Sensitive to outliers when
    #            the per-batch sample count is small (n=1 rollout, batch=32), which empirically
    #            slams the clamp floor/ceil almost every step (see DAD alpha=1.0 run 20260629).
    #   "robust" — z = (d - median) / (1.4826 * MAD). Outlier-resistant; keeps weights inside the
    #            clamp band instead of pinning them, so difficulty_alpha controls a smooth focus.
    difficulty_norm_mode: str = "zscore"

    # [OPSD token-KL-clip] Per-token divergence clipping (siyan-zhao/OPSD, arXiv 2601.18734).
    # Caps each response token's |distillation divergence| at `token_kl_clip` BEFORE it is
    # used (as PG reward or supervised loss). Motivation: style tokens (e.g. "wait", "think")
    # can exhibit 6-15x higher KL than content tokens and dominate the training signal; clipping
    # stabilizes training and prevents think-token KL from drowning out answer-token KL.
    # Distinct from `loss_max_clamp` (a coarse ±10 numerical-stability guard): this is a tunable,
    # typically-small (e.g. 0.05-0.5) per-token cap applied only when token_kl_clip is not None.
    # token_kl_clip=None (default) → exact original behavior, no effect.
    token_kl_clip: Optional[float] = None

    # [Cue-Gate] Privileged-Information-Gated Distillation (ViCue loss-side extension).
    # When ViCue teacher-prompt cue is active, quantify the cue's PER-TOKEN marginal utility
    #   Δ_token = logp_teacher(y | cue_prompt) − logp_teacher(y | no_cue_prompt)
    # and scale the distillation advantage by a gate derived from Δ, so distillation is
    # STRENGTHENED where the cue actually helps the teacher (large Δ = privileged info the
    # student can't see) and DAMPENED where it doesn't (Δ≈0 = student already knows).
    # positive view = existing `teacher_logprobs` (cue prompt); negative view = an extra no-cue
    # teacher forward on the ORIGINAL prompt (produced in agent_loop, same teacher pool, no extra
    # GPU). Gate math mirrors V-Zero but standardizes at BATCH level by default (see group_mode),
    # so it works with rollout.n=1 (no sibling group needed).
    # cue_gate_enabled=False (default) → exact standard behavior: no negative pass, no gate.
    cue_gate_enabled: bool = False
    cue_gate_alpha: float = 0.5            # gate aggressiveness; 0 → gate≡1 (exact no-op)
    cue_gate_gamma: float = 1.0            # negative-Δ penalty; 1.0 → token_score == Δ
    cue_gate_w_min: float = 0.0
    cue_gate_w_max: float = 2.0
    cue_gate_eps: float = 1e-6
    # Δ standardization scope:
    #   "batch" (default) — standardize Δ across the whole batch; works for rollout.n=1.
    #   "uid"             — group-relative z-score within sibling rollouts (needs rollout.n>1;
    #                       degenerates to neutral gate=1 for singleton groups, like V-Zero).
    cue_gate_group_mode: str = "batch"
    # Gate granularity:
    #   "seq" (default) — one weight per sequence (collapse tokens to a mean score first).
    #   "token"         — one weight per response token (z-normalize each token's own Δ-score
    #                     over the group's pooled valid tokens); masked tokens → weight 1.
    cue_gate_level: str = "seq"


    # Store global batch info for loss aggregation:
    # dp_size: data parallel size
    # batch_num_tokens: number of valid tokens in global batch
    # global_batch_size: global batch size
    global_batch_info: dict = field(default_factory=dict)

    # Store distillation loss settings for computing the specified loss_mode
    # Not set by user, populated at runtime
    loss_settings: Optional[dict] = None

    def __post_init__(self):
        self._mutable_fields.add("loss_settings")
        from verl.trainer.distillation.losses import DistillationLossSettings, get_distillation_loss_settings

        self.loss_settings: DistillationLossSettings = get_distillation_loss_settings(self.loss_mode)

        if self.policy_loss_mode != "vanilla":
            raise NotImplementedError(
                f"Only vanilla policy loss is currently supported when use_policy_gradient is True, "
                f"but got {self.policy_loss_mode}."
            )

        if self.use_policy_gradient and self.loss_mode == "forward_kl_topk":
            print(
                "WARNING: forward_kl_topk is most effective as a supervised distillation loss "
                "(use_policy_gradient=False). With policy gradient, the update uses only the sampled"
                " token's logprob ∇logπ(a), so the top-k distributional signal (how non-sampled logits "
                "should move) is largely unused."
            )

        if not self.use_policy_gradient and self.loss_mode == "k1":
            raise ValueError(
                "Directly backpropagating k1 loss is incorrect since gradient of k1 loss"
                " wrt model weights does not depend on teacher log probabilities."
            )


@dataclass
class DistillationTeacherModelConfig(BaseConfig):
    """Configuration for on-policy distillation teacher.

    key (str, optional):
        Identifier to route examples to the teacher model in multi-teacher setting.
    model_path (str, optional):
        Model path for the teacher model. Can be a local path or a Hugging Face model
    inference (RolloutConfig):
        Rollout configuration for the teacher model inference during distillation.
    num_replicas (int):
        Number of inference replicas of this teacher to launch. Each replica occupies
        `per_replica_world_size` GPUs (= inference.data_parallel_size *
        inference.tensor_model_parallel_size * inference.pipeline_model_parallel_size),
        so the teacher's total GPU footprint is
        `num_replicas * per_replica_world_size`.
    """

    _mutable_fields = BaseConfig._mutable_fields | {"num_replicas", "key"}

    key: Optional[str] = None
    model_path: Optional[str] = None
    inference: RolloutConfig = field(default_factory=RolloutConfig)
    num_replicas: Optional[int] = 0

    @property
    def per_replica_world_size(self) -> int:
        return (
            self.inference.tensor_model_parallel_size
            * self.inference.data_parallel_size
            * self.inference.pipeline_model_parallel_size
        )

    @property
    def world_size(self) -> int:
        return self.num_replicas * self.per_replica_world_size

    def check_configured(self):
        if self.model_path is None:
            raise ValueError("model_path must be specified for distillation teacher model config.")
        if self.key is None:
            raise ValueError("key must be specified for distillation teacher model config.")
        if self.num_replicas is None:
            raise ValueError("num_replicas must be specified for distillation teacher model config.")

    def validate_and_prepare_for_distillation(self, use_topk: bool, topk: Optional[int]) -> None:
        # Prompt + Response from student are fed into teacher as context
        max_model_len = self.inference.max_model_len
        student_prompt_length = self.inference.prompt_length
        student_response_length = self.inference.response_length
        required_context_len = student_prompt_length + student_response_length + 1
        if max_model_len is not None and required_context_len > max_model_len:
            raise ValueError(
                "Distillation teacher inference requires room for the student prompt, the full student "
                f"response, and one generated token, but got {student_prompt_length=}, "
                f"{student_response_length=}, {required_context_len=}, {max_model_len=}."
            )
        self.inference.prompt_length = self.inference.prompt_length + self.inference.response_length
        self.inference.response_length = 1
        self._validate_topk_logprobs(use_topk=use_topk, topk=topk)

    def _validate_topk_logprobs(self, use_topk: bool, topk: Optional[int]) -> None:
        if not use_topk:
            return
        if topk is None:
            raise ValueError("topk must be specified when use_topk is True.")

        engine_name = self.inference.name
        engine_kwargs = self.inference.engine_kwargs
        match engine_name:
            case "vllm":
                vllm_engine_kwargs = dict(engine_kwargs.get("vllm", {}))
                max_logprobs = vllm_engine_kwargs.get("max_logprobs")
                if max_logprobs is None:
                    vllm_engine_kwargs["max_logprobs"] = topk
                    max_logprobs = topk
                if max_logprobs < topk:
                    raise ValueError(
                        f"VLLM max_logprobs ({max_logprobs}) must be >= distillation_loss topk "
                        f"({topk}) to enable distillation loss computation."
                    )
                engine_kwargs["vllm"] = vllm_engine_kwargs
            case "sglang":
                # SGLang's top_logprobs_num is a per-request parameter, so there is no
                # engine-boot cap to align (unlike vLLM's max_logprobs). The async
                # server translates sampling_params["prompt_logprobs"] into
                # return_logprob + logprob_start_len=0 + top_logprobs_num at call time.
                pass
            case _:
                raise NotImplementedError(
                    f"DistillationTeacherModelConfig does not support inference engine {engine_name}"
                )


@dataclass
class DistillationConfig(BaseConfig):
    """Configuration for on-policy distillation.

    enabled (bool):
        Whether on-policy distillation is enabled.
    n_gpus_per_node (int):
        Number of GPUs per node in the teacher resource pool.
    nnodes (int):
        Number of nodes in the teacher resource pool.
    teacher_models (dict[str, TeacherModelConfig]):
        Configurations for teacher models used for multi-teacher distillation.
    teacher_key (str):
        Key to route examples to the appropriate teacher model in multi-teacher setups. Should correspond to a field in
        the data proto, e.g., data_source.
    distillation_loss (DistillationLossConfig):
    Configuration for distillation loss settings.

    NOTE: The `teacher_model` entry is in the `teacher_models` dict by default.
    Since it is popped when other teacher entries are added, using `teacher_model` as
    one of several keys silently drops it. For example, the following CLI overrides result
    in ONLY `teacher_model2` being used:

    ```bash
    distillation.teacher_models.teacher_model.key=openai/gsm8k
    distillation.teacher_models.teacher_model.model_path=Qwen/Qwen3-4B
    +distillation.teacher_models.teacher_model2.key=hiyouga/geometry3k
    +distillation.teacher_models.teacher_model2.model_path=Qwen/Qwen3-VL-4B-Instruct
    ```
    Instead, give the first teacher a different name:

    ```bash
    +distillation.teacher_models.teacher_model1.key=openai/gsm8k
    +distillation.teacher_models.teacher_model1.model_path=Qwen/Qwen3-4B
    +distillation.teacher_models.teacher_model2.key=hiyouga/geometry3k
    +distillation.teacher_models.teacher_model2.model_path=Qwen/Qwen3-VL-4B-Instruct
    ```
    """

    _mutable_fields = BaseConfig._mutable_fields | {"teacher_models", "n_gpus_per_node", "nnodes", "base_model"}

    enabled: bool = False
    n_gpus_per_node: int = 0
    nnodes: int = 0
    teacher_models: dict[str, DistillationTeacherModelConfig] = field(default_factory=dict)
    teacher_key: str = "data_source"
    distillation_loss: DistillationLossConfig = field(default_factory=DistillationLossConfig)
    # [AFD] Asymmetric Frame-Budget Distillation: optional dict with keys
    # {enabled, fps, max_frames, min_frames, nframes} to make the teacher decode
    # denser frames than the student. Default None → teacher reuses student frames (no-op).
    teacher_frame_budget: Optional[dict] = None
    # [ExOPD] Frozen base model (student initial weights) for 3-way advantage.
    # Configured like a teacher but launched separately. Only used when
    # distillation_loss.exopd_enabled=True.
    base_model: Optional[DistillationTeacherModelConfig] = None

    def __post_init__(self):
        if not self.enabled:
            return

        self.teacher_models = self._resolve_teacher_models()
        teacher_world_size_sum = 0
        for teacher_model in self.teacher_models.values():
            teacher_model.validate_and_prepare_for_distillation(
                use_topk=self.distillation_loss.loss_settings.use_topk,
                topk=self.distillation_loss.topk,
            )
            teacher_world_size_sum += teacher_model.world_size

        # [ExOPD] Validate base model config if exopd is enabled
        if self.distillation_loss.exopd_enabled:
            if self.base_model is None:
                raise ValueError("base_model must be configured when exopd_enabled=True.")
            self.base_model = omega_conf_to_dataclass(self.base_model, dataclass_type=DistillationTeacherModelConfig)
            self.base_model.key = "exopd_base"
            self.base_model.validate_and_prepare_for_distillation(
                use_topk=False, topk=None,
            )
            self.teacher_models["exopd_base"] = self.base_model
            teacher_world_size_sum += self.base_model.world_size

        total_pool_size = self.n_gpus_per_node * self.nnodes
        if teacher_world_size_sum != total_pool_size:
            raise ValueError(
                f"Sum of teacher (num_replicas * per_replica_world_size) ({teacher_world_size_sum}) must match "
                f"the distillation resource pool size "
                f"({self.n_gpus_per_node=} * {self.nnodes=} = {total_pool_size})."
            )

    def _resolve_teacher_models(self) -> dict[str, DistillationTeacherModelConfig]:
        assert "teacher_model" in self.teacher_models
        if len(self.teacher_models) == 1:
            # Single teacher occupies the teacher resource pool (minus ExOPD base if present).
            teacher_model = self.teacher_models["teacher_model"]
            inference = teacher_model.inference
            per_replica = (
                inference.tensor_model_parallel_size
                * inference.data_parallel_size
                * inference.pipeline_model_parallel_size
            )
            pool_size = self.n_gpus_per_node * self.nnodes
            # [ExOPD] Reserve GPUs for base model
            if self.distillation_loss.exopd_enabled and self.base_model is not None:
                base_cfg = omega_conf_to_dataclass(self.base_model, dataclass_type=DistillationTeacherModelConfig)
                base_world_size = base_cfg.num_replicas * (
                    base_cfg.inference.tensor_model_parallel_size
                    * base_cfg.inference.data_parallel_size
                    * base_cfg.inference.pipeline_model_parallel_size
                )
                pool_size -= base_world_size
            if pool_size % per_replica != 0:
                raise ValueError(
                    f"Single teacher's per_replica_world_size ({per_replica}) must divide the distillation "
                    f"resource pool size (available={pool_size})."
                )
            teacher_model.num_replicas = pool_size // per_replica
            teacher_model.key = "default"
        else:
            # Multiple teachers: remove default single teacher config
            self.teacher_models.pop("teacher_model")

        # Teacher models dict is keyed by teacher_key instead of YAML entry name
        teacher_models = {}
        for teacher_config in self.teacher_models.values():
            teacher_config = omega_conf_to_dataclass(teacher_config, dataclass_type=DistillationTeacherModelConfig)
            teacher_config.check_configured()
            if teacher_config.key in teacher_models:
                raise ValueError(f"Duplicate teacher key {teacher_config.key} found in teacher models.")
            teacher_models[teacher_config.key] = teacher_config
        return teacher_models
