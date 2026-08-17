# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import torch
from tensordict import TensorDict

from verl.base_config import BaseConfig
from verl.trainer.ppo.core_algos import agg_loss, get_policy_loss_fn, kl_penalty
from verl.utils.metric import AggregationType, Metric
from verl.workers.config import ActorConfig, DistillationConfig, DistillationLossConfig
from verl.workers.utils.losses import ppo_loss
from verl.workers.utils.padding import no_padding_2_padding

DistillationLossFn = Callable[
    [
        ActorConfig,  # actor_config
        DistillationConfig,  # distillation_config
        dict,  # model_output
        TensorDict,  # micro batch input
    ],
    tuple[torch.Tensor, dict[str, Any]],
]


def is_distillation_enabled(config: Optional[DistillationConfig]) -> bool:
    """Check if distillation is enabled based on the provided configuration."""
    if config is None:
        return False
    return config.enabled


@dataclass
class DistillationLossSettings(BaseConfig):
    """
    Settings for a distillation loss function to be registered.

    Args:
        names (str | list[str]): Name(s) to register the distillation loss function under.
        use_topk (bool): Whether the loss function uses top-k log probabilities.
        use_estimator (bool): Whether the loss function uses single-sample KL estimators.
    """

    names: str | list[str] = field(default_factory=list)
    use_topk: bool = False
    use_estimator: bool = False

    _mutable_fields = {"names"}

    def __post_init__(self):
        self.names = [self.names] if isinstance(self.names, str) else self.names
        if sum([self.use_topk, self.use_estimator]) != 1:
            raise ValueError(
                f"Expected only one of use_estimator, use_topk, but got {self.use_estimator=}, {self.use_topk=}."
            )


DISTILLATION_LOSS_REGISTRY: dict[str, DistillationLossFn] = {}
DISTILLATION_SETTINGS_REGISTRY: dict[str, DistillationLossSettings] = {}


def register_distillation_loss(
    loss_settings: DistillationLossSettings,
) -> Callable[[DistillationLossFn], DistillationLossFn]:
    """Register a distillation loss function with the given name."""

    def decorator(func: DistillationLossFn) -> DistillationLossFn:
        for name in loss_settings.names:
            if name in DISTILLATION_LOSS_REGISTRY:
                raise ValueError(f"Distillation loss function with name '{name}' is already registered.")
            DISTILLATION_LOSS_REGISTRY[name] = func
            DISTILLATION_SETTINGS_REGISTRY[name] = loss_settings
        return func

    return decorator


def get_distillation_loss_fn(loss_name: str) -> DistillationLossFn:
    """Get the distillation loss function with a given name."""
    if loss_name not in DISTILLATION_LOSS_REGISTRY:
        raise ValueError(
            f"Unsupported loss mode: {loss_name}. Supported modes are: {list(DISTILLATION_LOSS_REGISTRY.keys())}"
        )
    return DISTILLATION_LOSS_REGISTRY[loss_name]


def get_distillation_loss_settings(loss_name: str) -> DistillationLossSettings:
    """Get the distillation loss settings with a given name."""
    if loss_name not in DISTILLATION_SETTINGS_REGISTRY:
        raise ValueError(
            f"Unsupported loss mode: {loss_name}. Supported modes are: {list(DISTILLATION_SETTINGS_REGISTRY.keys())}"
        )
    return DISTILLATION_SETTINGS_REGISTRY[loss_name]


def compute_distillation_loss_range(
    distillation_losses: torch.Tensor, response_mask: torch.Tensor
) -> dict[str, Metric]:
    """Compute min and max distillation loss over valid response tokens."""
    if response_mask.is_nested:
        distillation_losses_response = distillation_losses[response_mask.bool().to_padded_tensor(False)]
    else:
        distillation_losses_response = distillation_losses[response_mask.bool()]
    return {
        "distillation/loss_min": Metric(AggregationType.MIN, distillation_losses_response.min()),
        "distillation/loss_max": Metric(AggregationType.MAX, distillation_losses_response.max()),
    }


def compute_topk_loss(
    config: ActorConfig,
    distillation_config: DistillationConfig,
    data: TensorDict,
    student_logits: torch.Tensor,
    data_format: str,
) -> torch.Tensor:
    """Compute the topk loss in logit processor.

    Returns:
    - distillation_losses: (bsz, seqlen/cp_size)
    - student_mass: (bsz, seqlen/cp_size)
    - teacher_mass: (bsz, seqlen/cp_size)
    """
    match config.strategy:
        # VeOmni uses FSDP2 internally, so its loss computation is identical to FSDP.
        case "fsdp" | "veomni":
            import verl.trainer.distillation.fsdp.losses as fsdp_losses

            distillation_loss_fn = fsdp_losses.compute_forward_kl_topk
        case "megatron":
            import verl.trainer.distillation.megatron.losses as megatron_losses

            distillation_loss_fn = megatron_losses.compute_forward_kl_topk
        case _:
            raise NotImplementedError(f"Unsupported strategy: {config.strategy=}")

    outputs = distillation_loss_fn(
        student_logits=student_logits,
        teacher_topk_log_probs=data["teacher_logprobs"],
        teacher_topk_ids=data["teacher_ids"],
        config=distillation_config,
        data_format=data_format,
    )

    expected_shape = student_logits.shape[:2]
    for k, v in outputs.items():
        assert v.shape == expected_shape, f"Expected shape {expected_shape}, but got {v.shape} for {k=}."

    return outputs


def distillation_ppo_loss(
    config: ActorConfig,
    distillation_config: Optional[DistillationConfig],
    model_output: dict = None,
    data: TensorDict = None,
    dp_group=None,
    student_logits: torch.Tensor = None,
    data_format: str = "thd",
):
    """Loss function used both for logit processor and final policy loss.
    - student_logits is not None, compute the topk loss in logit processor.
    - student_logits is None, compute final policy loss.

    [split sequence across sp/cp groups]
                   |
    [model forward and output logits: (bsz, seqlen/cp_size, vocab_size/tp_size)]
                   |
    [logits processor compute topk loss: (bsz, seqlen/cp_size)]
                   |
    [all gather topk loss across sp/cp groups: (bsz, seqlen)]
                   |
    [combine topk loss with policy loss]

    Args:
        config: Actor configuration.
        distillation_config: Distillation configuration.
        model_output: Model output, including log_probs, entropy.
        data: Micro input batch, contains
          - teacher_logprobs: (bsz, seqlen, topk)
          - teacher_ids: (bsz, seqlen, topk)
        student_logits: (bsz, seqlen/cp_size, vocab_size/tp_size).
        data_format: "thd" or "bshd", models not support THD format, e.g GPT-OSS, Qwen3.5

    Returns:
    - student_logits is not None, return the topk loss tensor (bsz, seqlen/cp_size).
    - student_logits is None, return the final policy loss scalar and metrics.
    """

    # Called as logits processor
    if student_logits is not None:
        return compute_topk_loss(config, distillation_config, data, student_logits, data_format)

    # Called as final policy loss
    distillation_loss_config = distillation_config.distillation_loss
    distill_loss, distill_metrics = distillation_loss(config, distillation_config, model_output, data)
    policy_loss, policy_metrics = ppo_loss(config, model_output, data, dp_group)
    if not distillation_loss_config.use_task_rewards:
        policy_loss = 0.0

    # Combine distillation with policy loss
    policy_metrics.update(distill_metrics)
    distillation_loss_coef = (
        distillation_loss_config.distillation_loss_coef if distillation_loss_config.use_task_rewards else 1.0
    )
    policy_loss += distill_loss * distillation_loss_coef
    policy_metrics["distillation/loss"] = Metric(value=distill_loss, aggregation=AggregationType.SUM)

    return policy_loss, policy_metrics


def distillation_loss(
    config: ActorConfig,
    distillation_config: DistillationConfig,
    model_output: dict,
    data: TensorDict,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute the distillation loss and related metrics.

    Returns:
    - distillation_loss: Aggregated distillation loss scalar.
    - distillation_metrics: Dictionary of metrics.
    """
    assert distillation_config is not None
    loss_config: DistillationLossConfig = distillation_config.distillation_loss
    distillation_loss_fn = get_distillation_loss_fn(loss_config.loss_mode)
    distillation_losses, distillation_metrics = distillation_loss_fn(
        config=config,
        distillation_config=distillation_config,
        model_output=model_output,
        data=data,
    )
    response_mask = data["response_mask"]
    loss_agg_mode = config.loss_agg_mode

    distillation_metrics.update(
        compute_distillation_loss_range(distillation_losses=distillation_losses, response_mask=response_mask)
    )
    if loss_config.loss_max_clamp is not None:
        # clamping min is for k1 loss which can be negative
        distillation_losses = distillation_losses.clamp(min=-loss_config.loss_max_clamp, max=loss_config.loss_max_clamp)

    # [OPSD token-KL-clip] Cap each token's divergence MAGNITUDE at token_kl_clip so that
    # high-divergence style tokens ("wait"/"think") do not dominate the signal over content
    # tokens (siyan-zhao/OPSD, arXiv 2601.18734). k1 losses are signed, so we clamp the
    # absolute value symmetrically (preserving sign), which matches OPSD's per-token clamp on
    # the (non-negative) JSD while remaining correct for signed reverse-KL. Gated: when
    # token_kl_clip is None (default) this is a no-op and behavior is exactly unchanged.
    token_kl_clip = getattr(loss_config, "token_kl_clip", None)
    if token_kl_clip is not None:
        with torch.no_grad():
            resp_mask_bool = (
                response_mask.bool().to_padded_tensor(False) if response_mask.is_nested else response_mask.bool()
            )
            pre_clip = distillation_losses[resp_mask_bool].abs()
            clipped_frac = (pre_clip > token_kl_clip).float().mean() if pre_clip.numel() > 0 else pre_clip.new_zeros(())
        distillation_losses = distillation_losses.clamp(min=-token_kl_clip, max=token_kl_clip)
        distillation_metrics.update(
            {
                "distillation/token_kl_clip_frac": Metric(AggregationType.MEAN, clipped_frac),
            }
        )

    if loss_config.use_policy_gradient:
        # Use negative distillation loss as reward, as done by https://thinkingmachines.ai/blog/on-policy-distillation/.
        policy_loss_fn = get_policy_loss_fn(loss_config.policy_loss_mode)
        for k, v in config.global_batch_info.items():
            loss_config.global_batch_info[k] = v
        log_prob = no_padding_2_padding(model_output["log_probs"], data)
        old_log_prob = data["old_log_probs"]
        if old_log_prob.is_nested:
            old_log_prob = data["old_log_probs"].to_padded_tensor(0.0)
        if response_mask.is_nested:
            response_mask = response_mask.to_padded_tensor(False)
        rollout_is_weights = data.get("rollout_is_weights", None)

        # [ExOPD] 3-way advantage: -(student - base) + lambda*(teacher - base)
        if loss_config.exopd_enabled and "base_logprobs" in data.keys():
            base_log_probs = no_padding_2_padding(data["base_logprobs"], data).squeeze(-1)
            teacher_log_probs = no_padding_2_padding(data["teacher_logprobs"], data).squeeze(-1)
            # [AD-ExOPD] Axis-decomposed: split (teacher - base) into a frame-budget
            # (temporal) axis and a model-capacity axis via a dense-frame base reference.
            #   advantage = -(student - base_sparse)
            #               + lt*(base_dense - base_sparse)   [temporal/frame-budget axis]
            #               + lc*(teacher    - base_dense)     [model-capacity axis]
            if loss_config.exopd_axis_decomposed and "base_dense_logprobs" in data.keys():
                base_dense_log_probs = no_padding_2_padding(data["base_dense_logprobs"], data).squeeze(-1)
                lt = loss_config.exopd_lambda_temporal
                lc = loss_config.exopd_lambda_capacity
                temporal_axis = base_dense_log_probs - base_log_probs
                capacity_axis = teacher_log_probs - base_dense_log_probs
                student_dev = old_log_prob - base_log_probs
                advantages = -(student_dev - lt * temporal_axis - lc * capacity_axis)
            elif loss_config.exopd_lambda == 1.0:
                advantages = -(old_log_prob - teacher_log_probs)
            else:
                reverse_kl = old_log_prob - base_log_probs
                reward_correction = teacher_log_probs - base_log_probs
                advantages = -(reverse_kl - reward_correction * loss_config.exopd_lambda)
            advantages = advantages.detach()
        else:
            advantages = -distillation_losses.detach()

        # [DAD] On-policy difficulty-adaptive weighting: focus learning on samples where the
        # student still diverges most from the teacher. Difficulty = per-sample mean of the
        # (always-positive) per-token distillation KL `distillation_losses` — a clean magnitude
        # of student-teacher disagreement, NOT the signed logprob gap. Mean-preserving (batch
        # mean weight = 1) so the effective learning rate is unchanged; self-annealing (uniform
        # early when all samples are hard, increasingly focused as the student masters the easy
        # mass). difficulty_alpha=0 degenerates exactly to the un-weighted advantage.
        if loss_config.difficulty_adaptive_enabled and loss_config.difficulty_alpha != 0.0:
            rm_dad = response_mask.float()
            tok_dad = rm_dad.sum(dim=-1) + 1e-8
            # per-token KL magnitude -> per-sample difficulty (detached)
            difficulty = (distillation_losses.detach().abs() * rm_dad).sum(dim=-1) / tok_dad
            # Standardize difficulty before the exp() weighting. "zscore" (default) reproduces the
            # original behavior; "robust" uses median/MAD, which is far less sensitive to the
            # heavy-tailed per-batch difficulty distribution that pins weights to the clamp under n=1.
            if loss_config.difficulty_norm_mode == "robust":
                med = difficulty.median()
                mad = (difficulty - med).abs().median()
                z = (difficulty - med) / (1.4826 * mad + 1e-8)
            else:
                z = (difficulty - difficulty.mean()) / (difficulty.std() + 1e-8)
            w = torch.exp(loss_config.difficulty_alpha * z)
            w = w / (w.mean() + 1e-8)  # mean-preserving (before clamp)
            w = w.clamp(loss_config.difficulty_weight_min, loss_config.difficulty_weight_max)
            advantages = advantages * w.unsqueeze(-1)
            distillation_metrics.update(
                {
                    "distillation/dad_difficulty_mean": Metric(AggregationType.MEAN, difficulty.mean()),
                    "distillation/dad_difficulty_std": Metric(AggregationType.MEAN, difficulty.std()),
                    "distillation/dad_weight_max": Metric(AggregationType.MAX, w.max()),
                    "distillation/dad_weight_min": Metric(AggregationType.MIN, w.min()),
                }
            )

        # [TSKL] Temporal-Selective KL: scale advantages by sample-level weight
        if loss_config.temporal_selective_kl and "temporal_weight" in data.keys():
            temporal_weight = data["temporal_weight"]
            if temporal_weight.dim() == 1:
                temporal_weight = temporal_weight.unsqueeze(-1)
            advantages = advantages * temporal_weight

        # [V-Zero-Video] Contrastive evidence gate: scale advantages by the group-relative
        # trajectory gate w^(g)=clip(1+(p-μ)/σ, w_min, w_max) computed at batch level in ray_trainer
        # from the positive/negative teacher views. Strengthens OPD on rollouts better supported by
        # genuine visual evidence, suppresses weakly-grounded (e.g. hallucinated) ones. Gated: no-op
        # when contrastive_gate_enabled is False or evidence_gate is absent.
        if loss_config.contrastive_gate_enabled and "evidence_gate" in data.keys():
            evidence_gate = data["evidence_gate"]
            if evidence_gate.dim() == 1:
                evidence_gate = evidence_gate.unsqueeze(-1)
            advantages = advantages * evidence_gate

        # [Cue-Gate] Privileged-information gate: scale advantages by the cue-utility weight
        # w=clip(1+alpha*z(Δ), w_min, w_max), Δ=logp(y|cue)−logp(y|no_cue), computed at batch level
        # in ray_trainer. Strengthens distillation where the cue actually helped the teacher.
        # Gated: no-op when cue_gate_enabled is False or cue_gate is absent (→ standard behavior).
        if loss_config.cue_gate_enabled and "cue_gate" in data.keys():
            cue_gate = data["cue_gate"]
            if cue_gate.dim() == 1:
                cue_gate = cue_gate.unsqueeze(-1)
            advantages = advantages * cue_gate

        distillation_loss, pg_metrics = policy_loss_fn(
            old_log_prob=old_log_prob,
            log_prob=log_prob,
            advantages=advantages,
            response_mask=response_mask,
            loss_agg_mode=loss_agg_mode,
            config=loss_config,
            rollout_is_weights=rollout_is_weights,
        )
        pg_metrics = {f"distillation/{k[len('actor/') :]}": v for k, v in pg_metrics.items()}
        distillation_metrics.update(pg_metrics)
    else:
        # Directly backpropagate distillation loss as a supervised loss, as in https://arxiv.org/abs/2306.13649.
        if response_mask.is_nested:
            response_mask = response_mask.to_padded_tensor(False)
        distillation_loss = agg_loss(
            loss_mat=distillation_losses,
            loss_mask=response_mask,
            loss_agg_mode=loss_agg_mode,
            **config.global_batch_info,
        )

    return distillation_loss, distillation_metrics


@register_distillation_loss(DistillationLossSettings(names=["forward_kl_topk"], use_topk=True))  # type: ignore[arg-type]
def compute_forward_kl_topk(
    config: ActorConfig,
    distillation_config: DistillationConfig,
    model_output: dict,
    data: TensorDict,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute forward KL distillation loss and related metrics using top-k log probabilities.

    Returns:
    - distillation_losses: (bsz, resp_len)
    - distillation_metrics: Dictionary of metrics.
    """
    # topk loss has been computed in logits processor
    distillation_losses = no_padding_2_padding(model_output["distillation_losses"], data)
    student_mass = no_padding_2_padding(model_output["student_mass"], data)
    teacher_mass = no_padding_2_padding(model_output["teacher_mass"], data)
    if data["response_mask"].is_nested:
        response_mask_bool = data["response_mask"].bool().to_padded_tensor(False)
    else:
        response_mask_bool = data["response_mask"].bool()
    assert distillation_losses.shape == student_mass.shape == teacher_mass.shape == response_mask_bool.shape

    # Log amount of mass in the top-k log probabilities for both student and teacher.
    student_mass = student_mass[response_mask_bool]
    teacher_mass = teacher_mass[response_mask_bool]
    distillation_metrics = {
        "distillation/student_mass": student_mass.mean().item(),
        "distillation/student_mass_min": Metric(AggregationType.MIN, student_mass.min()),
        "distillation/student_mass_max": Metric(AggregationType.MAX, student_mass.max()),
        "distillation/teacher_mass": teacher_mass.mean().item(),
        "distillation/teacher_mass_min": Metric(AggregationType.MIN, teacher_mass.min()),
        "distillation/teacher_mass_max": Metric(AggregationType.MAX, teacher_mass.max()),
    }

    # Due to use of top-k, student and teacher distributions don't sum to 1 -> divergences can be negative.
    distillation_losses = distillation_losses.clamp_min(0.0)

    return distillation_losses, distillation_metrics


@register_distillation_loss(
    DistillationLossSettings(names=["kl", "k1", "abs", "mse", "k2", "low_var_kl", "k3"], use_estimator=True)
)  # type: ignore[arg-type]
def compute_distillation_loss_reverse_kl_estimator(
    config: ActorConfig,
    distillation_config: DistillationConfig,
    model_output,
    data: TensorDict,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute the distillation loss and related metrics using single-sample KL estimators.

    Uses the kl_penalty function from core_algos which supports various KL divergence
    estimators: "kl", "k1", "abs", "mse", "k2", "low_var_kl", "k3".

    Returns:
    - distillation_losses: (bsz, resp_len)
    - distillation_metrics: Dictionary of metrics.
    """
    student_log_probs = no_padding_2_padding(model_output["log_probs"], data)
    teacher_log_probs = no_padding_2_padding(data["teacher_logprobs"], data).squeeze(-1)
    if data["response_mask"].is_nested:
        response_mask_bool = data["response_mask"].bool().to_padded_tensor(False)
    else:
        response_mask_bool = data["response_mask"].bool()
    assert teacher_log_probs.shape == student_log_probs.shape == response_mask_bool.shape

    loss_config: DistillationLossConfig = distillation_config.distillation_loss
    distillation_losses = kl_penalty(
        logprob=student_log_probs, ref_logprob=teacher_log_probs, kl_penalty=loss_config.loss_mode
    )
    # Since k1 can be negative, log the mean absolute loss.
    metrics = {
        "distillation/abs_loss": Metric(AggregationType.MEAN, distillation_losses[response_mask_bool].abs().mean()),
    }
    # [ViCuR-style diagnostic] Signed teacher-student log-prob gap on response tokens
    # (gap = teacher_logprob - student_logprob, averaged over valid tokens). This is the
    # quantity plotted in ViCuR Fig.3: a gap that DECREASES over training means the
    # teacher's distribution is becoming realizable for the student (good). A gap that
    # stays flat/high means the target is unrealizable (the AFD v2 failure mode).
    # Also split by token position within each response (front 50% / back 50%) to expose
    # where the student lags the teacher most (early reasoning vs. final answer).
    with torch.no_grad():
        gap = teacher_log_probs - student_log_probs  # (bsz, resp_len)
        valid = response_mask_bool
        if valid.any():
            metrics["distillation/ts_logprob_gap"] = Metric(AggregationType.MEAN, gap[valid].mean())
            # Per-row front/back split by valid-token position (not raw column index).
            front_mask = torch.zeros_like(valid)
            back_mask = torch.zeros_like(valid)
            for r in range(valid.shape[0]):
                cols = valid[r].nonzero(as_tuple=True)[0]
                if cols.numel() == 0:
                    continue
                mid = cols.numel() // 2
                if mid > 0:
                    front_mask[r, cols[:mid]] = True
                back_mask[r, cols[mid:]] = True
            if front_mask.any():
                metrics["distillation/ts_gap_front50"] = Metric(AggregationType.MEAN, gap[front_mask].mean())
            if back_mask.any():
                metrics["distillation/ts_gap_back50"] = Metric(AggregationType.MEAN, gap[back_mask].mean())
    return distillation_losses, metrics
