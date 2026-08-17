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
#
# [V-Zero-Video] Ported from eVI-group-SCU/V-Zero (arXiv 2606.25319),
# verl/trainer/distillation/evidence_weighting.py, adapted for the video setting:
# the positive/negative views here are full-video vs degraded-video (black/shuffle/
# textonly), produced in agent_loop, instead of the paper's image crop vs downsampled
# crop. The gating math (per-token asymmetric evidence gap → group-relative z-score →
# clipped weight) is kept identical to upstream so results are comparable.

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch


def _squeeze_single_logprob(name: str, logprobs: torch.Tensor) -> torch.Tensor:
    if logprobs.dim() == 2:
        return logprobs
    if logprobs.dim() == 3 and logprobs.shape[-1] == 1:
        return logprobs.squeeze(-1)
    raise ValueError(
        f"{name} must contain sampled-token logprobs with shape (bsz, seq_len) or "
        f"(bsz, seq_len, 1), got {tuple(logprobs.shape)}."
    )


def compute_relative_evidence_opd_weights(
    *,
    teacher_logprobs: torch.Tensor,
    teacher_neg_logprobs: torch.Tensor,
    response_mask: torch.Tensor,
    prompt_width: int,
    uids: Sequence[Any] | np.ndarray,
    gamma: float,
    alpha: float,
    w_min: float,
    w_max: float,
    eps: float,
    return_diagnostics: bool = False,
) -> tuple[torch.Tensor, dict[str, float]] | tuple[torch.Tensor, dict[str, float], dict[str, torch.Tensor]]:
    """Compute per-sequence relative evidence weights for OPD (V-Zero).

    `teacher_logprobs` (positive view) and `teacher_neg_logprobs` (negative view) are
    each padded to the FULL prompt+response sequence (as emitted by `_pad_teacher_outputs`);
    the response slice `[:, prompt_width : prompt_width+response_len]` is selected here.
    The returned weights have shape `(bsz,)`.

    weight^(g) = clip(1 + alpha * z_group(score^(g)), w_min, w_max), where
        score^(g) = mean_k [ relu(Δ_k) - gamma * relu(-Δ_k) ],  Δ_k = logp_pos - logp_neg
    and z_group standardizes within each prompt's sibling rollouts (grouped by uid).
    Singleton groups (no sibling) get advantage 0 → weight 1 (neutral).
    """

    if len(uids) != response_mask.shape[0]:
        raise ValueError(f"uid count ({len(uids)}) must match batch size ({response_mask.shape[0]}).")

    plus_full = _squeeze_single_logprob("teacher_logprobs", teacher_logprobs).float()
    minus_full = _squeeze_single_logprob("teacher_neg_logprobs", teacher_neg_logprobs).float()
    response_len = response_mask.shape[1]
    response_end = prompt_width + response_len
    if plus_full.shape != minus_full.shape:
        raise ValueError(f"teacher positive/negative logprob shapes differ: {plus_full.shape} vs {minus_full.shape}.")
    if response_end > plus_full.shape[1]:
        raise ValueError(
            f"Teacher logprob sequence is too short for prompt_width+response_len: "
            f"{plus_full.shape[1]=}, {prompt_width=}, {response_len=}."
        )

    plus = plus_full[:, prompt_width:response_end]
    minus = minus_full[:, prompt_width:response_end]
    mask = response_mask.to(device=plus.device, dtype=plus.dtype)
    mask_bool = mask.bool()

    delta = plus - minus
    token_scores = torch.relu(delta) - float(gamma) * torch.relu(-delta)
    lengths = mask.sum(dim=-1).clamp_min(float(eps))
    scores = (token_scores * mask).sum(dim=-1) / lengths

    advantages = torch.zeros_like(scores)
    uid_values = np.asarray(uids, dtype=object)
    for uid in np.unique(uid_values):
        indices = np.nonzero(uid_values == uid)[0]
        if indices.size <= 1:
            # Singleton group: no sibling to compare against → leave advantage 0 (neutral gate).
            continue
        group_idx = torch.as_tensor(indices, device=scores.device, dtype=torch.long)
        group_scores = scores.index_select(0, group_idx)
        group_mean = group_scores.mean()
        group_std = torch.sqrt(torch.mean((group_scores - group_mean).square()))
        advantages[group_idx] = (group_scores - group_mean) / (group_std + float(eps))

    weights = torch.clamp(1.0 + float(alpha) * advantages, min=float(w_min), max=float(w_max)).detach()

    valid_delta = delta[mask_bool]
    if valid_delta.numel() == 0:
        delta_mean = delta_pos_frac = 0.0
    else:
        delta_mean = valid_delta.mean().detach().item()
        delta_pos_frac = (valid_delta > 0).float().mean().detach().item()

    metrics = {
        "distillation/evidence_score_mean": scores.mean().detach().item(),
        "distillation/evidence_score_std": scores.std(unbiased=False).detach().item(),
        "distillation/evidence_adv_mean": advantages.mean().detach().item(),
        "distillation/evidence_adv_std": advantages.std(unbiased=False).detach().item(),
        "distillation/evidence_gate_mean": weights.mean().detach().item(),
        "distillation/evidence_gate_min": weights.min().detach().item(),
        "distillation/evidence_gate_max": weights.max().detach().item(),
        "distillation/evidence_gate_zero_frac": (weights <= float(w_min) + float(eps)).float().mean().detach().item(),
        "distillation/evidence_delta_mean": delta_mean,
        "distillation/evidence_delta_pos_frac": delta_pos_frac,
    }
    weights = weights.to(dtype=torch.float32)
    if return_diagnostics:
        diagnostics = {
            "scores": scores.detach().to(dtype=torch.float32),
            "advantages": advantages.detach().to(dtype=torch.float32),
            "weights": weights.detach(),
        }
        return weights, metrics, diagnostics
    return weights, metrics


def compute_cue_gate_weights(
    *,
    teacher_logprobs: torch.Tensor,
    teacher_nocue_logprobs: torch.Tensor,
    response_mask: torch.Tensor,
    prompt_width: int,
    alpha: float,
    gamma: float,
    w_min: float,
    w_max: float,
    eps: float,
    group_mode: str = "batch",
    uids: Sequence[Any] | np.ndarray | None = None,
    level: str = "seq",
) -> tuple[torch.Tensor, dict[str, float]]:
    """[Cue-Gate] Gate from the cue's marginal utility to the teacher.

    positive view = `teacher_logprobs` (teacher on the CUE prompt);
    negative view = `teacher_nocue_logprobs` (teacher on the ORIGINAL no-cue prompt).
    Both are padded to the FULL student prompt+response sequence; the response slice
    `[:, prompt_width : prompt_width+response_len]` is scored here.

    Math (mirrors V-Zero token scoring):
        Δ_k         = logp_cue - logp_nocue                         (per response token)
        score^(g)   = mean_k [ relu(Δ_k) - gamma * relu(-Δ_k) ]     (per sequence)
        weight^(g)  = clip(1 + alpha * z(score), w_min, w_max)

    level (granularity of the gate):
      * level="seq" (default): collapse each sequence's tokens to score^(g), z-normalize
        the per-sequence scores, return one weight per sequence `(bsz,)`. Original behavior.
      * level="token": SKIP the per-sequence mean; z-normalize each token's own
        `relu(Δ_k)-gamma*relu(-Δ_k)` over the pooled VALID tokens of the same group, and
        weight each token individually. Returns `(bsz, response_len)`; masked/padding
        tokens get weight 1. Finer granularity — only re-weights the exact tokens the cue
        helped, instead of the whole response.

    z(...) standardization scope:
      * group_mode="batch" (default): standardize across the WHOLE batch — works for
        rollout.n=1 (no sibling group needed).
      * group_mode="uid": group-relative z-score within each prompt's sibling rollouts
        (needs rollout.n>1; singleton/too-few groups get advantage 0 → weight 1, like V-Zero).

    alpha=0 → weight ≡ 1 everywhere (exact no-op; identical to standard distillation), both levels.
    """
    plus_full = _squeeze_single_logprob("teacher_logprobs", teacher_logprobs).float()
    minus_full = _squeeze_single_logprob("teacher_nocue_logprobs", teacher_nocue_logprobs).float()
    response_len = response_mask.shape[1]
    response_end = prompt_width + response_len
    if plus_full.shape != minus_full.shape:
        raise ValueError(f"cue/no-cue teacher logprob shapes differ: {plus_full.shape} vs {minus_full.shape}.")
    if response_end > plus_full.shape[1]:
        raise ValueError(
            f"Teacher logprob sequence too short for prompt_width+response_len: "
            f"{plus_full.shape[1]=}, {prompt_width=}, {response_len=}."
        )

    plus = plus_full[:, prompt_width:response_end]
    minus = minus_full[:, prompt_width:response_end]
    mask = response_mask.to(device=plus.device, dtype=plus.dtype)
    mask_bool = mask.bool()

    delta = plus - minus
    token_scores = torch.relu(delta) - float(gamma) * torch.relu(-delta)

    if level == "token":
        # [Cue-Gate per-token] Weight each response token by its OWN cue-utility score,
        # z-normalized over the pooled VALID tokens of the same group (uid sibling group,
        # or the whole batch). Returns (bsz, response_len); masked/padding tokens → weight 1.
        token_weights = torch.ones_like(token_scores)
        if group_mode == "uid":
            if uids is None or len(uids) != token_scores.shape[0]:
                raise ValueError("group_mode='uid' requires uids matching batch size.")
            uid_values = np.asarray(uids, dtype=object)
            for uid in np.unique(uid_values):
                indices = np.nonzero(uid_values == uid)[0]
                group_idx = torch.as_tensor(indices, device=token_scores.device, dtype=torch.long)
                g_scores = token_scores.index_select(0, group_idx)
                g_valid = mask_bool.index_select(0, group_idx)
                vals = g_scores[g_valid]
                if vals.numel() <= 1:
                    continue  # too few valid tokens in group → neutral gate (weight 1)
                mean = vals.mean()
                std = torch.sqrt(torch.mean((vals - mean).square()))
                z = (g_scores - mean) / (std + float(eps))
                w = torch.clamp(1.0 + float(alpha) * z, min=float(w_min), max=float(w_max))
                token_weights[group_idx] = torch.where(g_valid, w, torch.ones_like(w))
        else:  # "batch": pool all valid tokens in the batch
            vals = token_scores[mask_bool]
            if vals.numel() > 1:
                mean = vals.mean()
                std = torch.sqrt(torch.mean((vals - mean).square()))
                z = (token_scores - mean) / (std + float(eps))
                w = torch.clamp(1.0 + float(alpha) * z, min=float(w_min), max=float(w_max))
                token_weights = torch.where(mask_bool, w, torch.ones_like(w))
        weights = token_weights.detach()
        gate_valid = weights[mask_bool]
        vd = delta[mask_bool]
        metrics = {
            "distillation/cue_gate_level_token": 1.0,
            "distillation/cue_gate_mean": gate_valid.mean().detach().item() if gate_valid.numel() else 1.0,
            "distillation/cue_gate_min": gate_valid.min().detach().item() if gate_valid.numel() else 1.0,
            "distillation/cue_gate_max": gate_valid.max().detach().item() if gate_valid.numel() else 1.0,
            "distillation/cue_delta_mean": vd.mean().detach().item() if vd.numel() else 0.0,
            "distillation/cue_delta_pos_frac": (vd > 0).float().mean().detach().item() if vd.numel() else 0.0,
        }
        return weights.to(dtype=torch.float32), metrics

    # ---- level == "seq" (default): per-sequence gate (original behavior) ----
    lengths = mask.sum(dim=-1).clamp_min(float(eps))
    scores = (token_scores * mask).sum(dim=-1) / lengths

    advantages = torch.zeros_like(scores)
    if group_mode == "uid":
        if uids is None or len(uids) != scores.shape[0]:
            raise ValueError("group_mode='uid' requires uids matching batch size.")
        uid_values = np.asarray(uids, dtype=object)
        for uid in np.unique(uid_values):
            indices = np.nonzero(uid_values == uid)[0]
            if indices.size <= 1:
                continue  # singleton → neutral gate (advantage 0)
            group_idx = torch.as_tensor(indices, device=scores.device, dtype=torch.long)
            group_scores = scores.index_select(0, group_idx)
            group_mean = group_scores.mean()
            group_std = torch.sqrt(torch.mean((group_scores - group_mean).square()))
            advantages[group_idx] = (group_scores - group_mean) / (group_std + float(eps))
    else:  # "batch": standardize across the whole batch (works for n=1)
        if scores.shape[0] > 1:
            b_mean = scores.mean()
            b_std = torch.sqrt(torch.mean((scores - b_mean).square()))
            advantages = (scores - b_mean) / (b_std + float(eps))
        # batch size 1 → advantage stays 0 → neutral gate

    weights = torch.clamp(1.0 + float(alpha) * advantages, min=float(w_min), max=float(w_max)).detach()

    valid_delta = delta[mask_bool]
    if valid_delta.numel() == 0:
        delta_mean = delta_pos_frac = 0.0
    else:
        delta_mean = valid_delta.mean().detach().item()
        delta_pos_frac = (valid_delta > 0).float().mean().detach().item()

    metrics = {
        "distillation/cue_score_mean": scores.mean().detach().item(),
        "distillation/cue_score_std": scores.std(unbiased=False).detach().item(),
        "distillation/cue_gate_mean": weights.mean().detach().item(),
        "distillation/cue_gate_min": weights.min().detach().item(),
        "distillation/cue_gate_max": weights.max().detach().item(),
        "distillation/cue_delta_mean": delta_mean,
        "distillation/cue_delta_pos_frac": delta_pos_frac,
    }
    return weights.to(dtype=torch.float32), metrics

