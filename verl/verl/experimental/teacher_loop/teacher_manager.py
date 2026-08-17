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
from typing import Any, Optional
from uuid import uuid4

import torch
from omegaconf import DictConfig
from torch.nn import functional as F

from verl.utils.config import omega_conf_to_dataclass
from verl.workers.config import (
    DistillationConfig,
    DistillationLossConfig,
    DistillationTeacherModelConfig,
)
from verl.workers.rollout.llm_server import LLMServerClient


def _get_teacher_sampling_params(
    teacher_model_config: DistillationTeacherModelConfig,
    distillation_loss_config: DistillationLossConfig,
) -> dict[str, Any]:
    """Get sampling parameters for teacher model when computing log probabilities for distillation."""
    if teacher_model_config.inference.temperature != 1.0:
        raise NotImplementedError("vLLM does not support temperature for prompt_logprobs.")

    num_logprobs = distillation_loss_config.topk if distillation_loss_config.loss_settings.use_topk else 0
    return {
        "max_tokens": 1,
        "temperature": teacher_model_config.inference.temperature,
        "prompt_logprobs": num_logprobs,
    }


def _pad_teacher_outputs(
    teacher_ids: torch.Tensor,
    teacher_logprobs: torch.Tensor,
    prompt_width: int,
    response_width: int,
    prompt_length: int,
    response_length: int,
    pad_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    # TODO(wuxibin): remove padding and use tensordict.
    left_pad_size = prompt_width - prompt_length
    right_pad_size = response_width - response_length
    padding = (0, 0, left_pad_size, right_pad_size)
    return (
        F.pad(teacher_ids, padding, value=pad_token_id).unsqueeze(0),
        F.pad(teacher_logprobs, padding, value=0.0).unsqueeze(0),
    )


class AsyncTeacherLLMServerManager:
    """Teacher-specific async client used for distillation logprob computation."""

    def __init__(
        self,
        config: DictConfig,
        teacher_client: dict[str, LLMServerClient],
    ):
        self.distillation_config: DistillationConfig = omega_conf_to_dataclass(config.distillation)
        self.distillation_loss_config: DistillationLossConfig = self.distillation_config.distillation_loss
        self.teacher_key: str = self.distillation_config.teacher_key

        self.teacher_model_configs: dict[str, DistillationTeacherModelConfig] = self.distillation_config.teacher_models
        expected = set(self.teacher_model_configs)
        if set(teacher_client.keys()) != expected:
            raise ValueError(
                f"teacher client keys {sorted(teacher_client.keys())} "
                f"do not match teacher routing keys {sorted(expected)}."
            )
        self.teacher_client: dict[str, LLMServerClient] = teacher_client

    def _resolve_teacher_key(self, routing_key: Optional[str]) -> str:
        # [ExOPD] Direct routing to specific model (e.g. "exopd_base") bypasses normal logic
        if routing_key is not None and routing_key in self.teacher_model_configs:
            return routing_key
        # Exclude ExOPD base from normal teacher routing
        normal_teachers = {k: v for k, v in self.teacher_model_configs.items() if k != "exopd_base"}
        if len(normal_teachers) == 1:
            return next(iter(normal_teachers))
        if routing_key is None:
            raise ValueError(
                f"Routing key is required for multi-teacher distillation "
                f"(configured via distillation.teacher_key={self.teacher_key!r})."
            )
        if routing_key not in normal_teachers:
            raise ValueError(
                f"No teacher configured for routing key {routing_key!r}. "
                f"Configured teachers: {sorted(normal_teachers)}."
            )
        return routing_key

    async def compute_teacher_logprobs_single(
        self,
        sequence_ids: list[int],
        multi_modal_data: Optional[dict[str, Any]] = None,
        mm_processor_kwargs: Optional[dict[str, Any]] = None,
        routing_key: Optional[str] = None,
        response_length: Optional[int] = None,
        pad_to_sequence_ids: Optional[list[int]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute teacher log probabilities for a single unpadded sequence.

        pad_to_sequence_ids: [AFD] when the teacher runs on a *different* sequence than the
            student (e.g. denser frames → longer prompt), the teacher's logprobs must still be
            padded/aligned to the STUDENT sequence so that downstream batch cat + response-only
            distillation loss line up. Pass the student's (prompt_ids + response_ids) here.
            When None (default), pad against `sequence_ids` itself — identical to original behavior.
        """
        multi_modal_data = multi_modal_data or {}
        teacher_key = self._resolve_teacher_key(routing_key)
        teacher_model_config = self.teacher_model_configs[teacher_key]
        client = self.teacher_client[teacher_key]
        teacher_output = await client.generate(
            request_id=uuid4().hex,
            prompt_ids=sequence_ids,
            sampling_params=_get_teacher_sampling_params(teacher_model_config, self.distillation_loss_config),
            image_data=multi_modal_data.get("images"),
            video_data=multi_modal_data.get("videos"),
            audio_data=multi_modal_data.get("audios"),
            mm_processor_kwargs=mm_processor_kwargs,
        )
        # Shapes: teacher vLLM returns [S, 1] or [S, K] for both ids and logprobs
        teacher_ids = torch.tensor(teacher_output.extra_fields["prompt_ids"], dtype=torch.int32)
        teacher_logprobs = torch.tensor(teacher_output.extra_fields["prompt_logprobs"])

        # The sequence we pad/align the output against. For AFD this is the student's sequence
        # (teacher prompt differs in vision-token count, but only the response portion is used
        # for the distillation loss, and it must line up with the student's response tokens).
        align_ids = pad_to_sequence_ids if pad_to_sequence_ids is not None else sequence_ids

        # For multimodal inputs, teacher vLLM may expand vision tokens differently,
        # causing full sequence length mismatch. Only return the response portion
        # (last response_length tokens) which is pure text and always consistent.
        if response_length is not None and teacher_ids.shape[0] != len(align_ids):
            print(f"[TeacherLoop] Multimodal length mismatch: teacher_ids={teacher_ids.shape}, "
                  f"teacher_logprobs={teacher_logprobs.shape}, align_len={len(align_ids)}, "
                  f"response_length={response_length}. Extracting response only.", flush=True)
            # Extract response portion from the end
            teacher_ids = teacher_ids[-response_length:]
            teacher_logprobs = teacher_logprobs[-response_length:]
            # Build padded prompt portion matching the STUDENT (align_ids) shape
            prompt_length = len(align_ids) - response_length
            # teacher_ids is [response_length, K] -> pad prompt as [prompt_length, K]
            K = teacher_ids.shape[-1] if teacher_ids.dim() == 2 else 1
            prompt_ids_pad = torch.tensor(align_ids[:prompt_length], dtype=torch.int32).unsqueeze(-1).expand(-1, K)
            teacher_ids = torch.cat([prompt_ids_pad, teacher_ids], dim=0)
            # teacher_logprobs is [response_length, K] -> pad prompt as [prompt_length, K]
            K_lp = teacher_logprobs.shape[-1] if teacher_logprobs.dim() == 2 else 1
            prompt_logprobs_pad = torch.zeros(prompt_length, K_lp)
            if teacher_logprobs.dim() == 1:
                teacher_logprobs = teacher_logprobs.unsqueeze(-1)
            teacher_logprobs = torch.cat([prompt_logprobs_pad, teacher_logprobs], dim=0)
        return teacher_ids, teacher_logprobs
