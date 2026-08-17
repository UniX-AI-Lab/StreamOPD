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
"""
Agent framework for multi-turn rollout and agentic reinforcement learning.
- AgentLoopBase: coroutine based abstract base class for agent loop.
  - SingleTurnAgentLoop: single turn agent loop.
  - ToolAgentLoop: ReAct agent loop with tool calling, with user defined tools.
- AgentLoopWorker: worker class for running agent loop coroutines in parallel.
- AgentLoopManager: manager class for running agent loop workers in parallel.

AgentLoopManager is one specific agent-framework implementation in verl,
and is designed to be fully replaceable by other agent frameworks such as:
- NVIDIA Nemo-Gym
- AWS Bedrock AgentCore
- SWE-agent
- ...
"""

import asyncio
import logging
import os
import random
from abc import ABC, abstractmethod
from typing import Any, Optional
from uuid import uuid4

import hydra
import numpy as np
import ray
import torch
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from pydantic import BaseModel, ConfigDict
from tensordict import TensorDict
from transformers import AutoProcessor, AutoTokenizer

from verl.experimental.agent_loop.utils import resolve_config_path
from verl.protocol import DataProto
from verl.tools.tool_registry import load_all_tools
from verl.trainer.distillation import is_distillation_enabled
from verl.utils.chat_template import apply_chat_template, initialize_system_prompt
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.dataset.rl_dataset import RLHFDataset, get_dataset_class
from verl.utils.model import compute_position_id_with_mask
from verl.utils.profiler import simple_timer
from verl.utils.ray_utils import auto_await, get_event_loop
from verl.utils.rollout_trace import (
    RolloutTraceConfig,
    rollout_trace_attr,
)
from verl.utils.tokenizer import (
    build_multimodal_processor_inputs,
    get_processor_token_id,
    normalize_token_ids,
)
from verl.workers.config import (
    HFModelConfig,
    RolloutConfig,
)
from verl.workers.rollout.llm_server import LLMServerClient

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

DEFAULT_ROUTING_CACHE_SIZE = 10000


class AgentLoopMetrics(BaseModel):
    """Agent loop performance metrics."""

    generate_sequences: float = 0.0
    tool_calls: float = 0.0
    compute_score: float = 0.0
    num_preempted: int = -1  # -1 means not available


class AgentLoopOutput(BaseModel):
    """Agent loop output."""

    prompt_ids: list[int]
    """Prompt token ids."""
    response_ids: list[int]
    """Response token ids including LLM generated token, tool response token."""
    response_mask: list[int]
    """Response mask, 1 for LLM generated token, 0 for tool response token."""
    response_logprobs: Optional[list[float]] = None
    """Log probabilities for the response tokens."""
    routed_experts: Optional[Any] = None
    """Routed experts for the total tokens."""
    multi_modal_data: Optional[dict[str, Any]] = None
    """Multi-modal data for multi-modal tools."""
    reward_score: Optional[float] = None
    """Reward score for the trajectory."""
    num_turns: int = 0
    """Number of chat turns, including user, assistant, tool."""
    metrics: AgentLoopMetrics
    """Auxiliary performance metrics"""
    extra_fields: dict[str, Any] = {}
    """Extra fields for dynamic addition."""
    mm_processor_kwargs: Optional[dict[str, Any]] = None
    """Processor/backend kwargs that must stay aligned across rollout and training paths."""

    def as_dict(self) -> dict[str, Any]:
        """Convert agent loop output to a dictionary."""
        output = self.model_dump(exclude_unset=True)

        output["prompts"] = torch.tensor(output.pop("prompt_ids"), dtype=torch.int64)
        output["responses"] = torch.tensor(output.pop("response_ids"), dtype=torch.int64)
        output["response_mask"] = torch.tensor(output.pop("response_mask"), dtype=torch.int64)

        response_logprobs = output.pop("response_logprobs", None)
        if response_logprobs is not None:
            output["rollout_log_probs"] = torch.tensor(response_logprobs, dtype=torch.float32)

        routed_experts = output.pop("routed_experts", None)
        if routed_experts is not None:
            output["routed_experts"] = torch.tensor(routed_experts, dtype=torch.int64)

        # rm_scores: reward score for each token
        reward_score = output.pop("reward_score", None)
        if reward_score is not None:
            rm_scores = torch.zeros_like(output["response_mask"], dtype=torch.float32)
            rm_scores[-1] = reward_score
            output["rm_scores"] = rm_scores

        teacher_ids, teacher_logprobs = (
            output["extra_fields"].pop("teacher_ids", None),
            output["extra_fields"].pop("teacher_logprobs", None),
        )
        if teacher_ids is not None:
            output["teacher_ids"] = teacher_ids
        if teacher_logprobs is not None:
            output["teacher_logprobs"] = teacher_logprobs
        return output


class _InternalAgentLoopOutput(AgentLoopOutput):
    """Internal agent loop output with padded sequences."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    prompt_ids: torch.Tensor
    """Padded prompt token ids."""
    response_ids: torch.Tensor
    """Padded response token ids."""
    input_ids: torch.Tensor
    """Padded input ids(prompt_ids + response_ids)."""
    position_ids: torch.Tensor
    """Padded position ids."""
    response_mask: torch.Tensor
    """Padded response mask."""
    attention_mask: torch.Tensor
    """Padded attention mask."""
    response_logprobs: Optional[torch.Tensor] = None
    """Padded log probabilities for the response tokens."""
    teacher_logprobs: Optional[torch.Tensor] = None
    """Padded log probabilities from teacher model for prompt/response tokens."""
    teacher_ids: Optional[torch.Tensor] = None
    """Padded token ids corresponding to the teacher log probabilities."""
    base_logprobs: Optional[torch.Tensor] = None
    """[ExOPD] Padded log probabilities from frozen base model."""
    base_ids: Optional[torch.Tensor] = None
    """[ExOPD] Padded token ids corresponding to the base model log probabilities."""
    base_dense_logprobs: Optional[torch.Tensor] = None
    """[AD-ExOPD] Padded log probabilities from frozen base model on DENSE frames."""
    base_dense_ids: Optional[torch.Tensor] = None
    """[AD-ExOPD] Padded token ids corresponding to the dense-frame base log probabilities."""
    neg_view_logprobs: Optional[torch.Tensor] = None
    """[V-Zero-Video] Padded teacher log probabilities on the NEGATIVE (degraded) video view,
    aligned to the student sequence. Used with `teacher_logprobs` (positive view) to compute the
    contrastive evidence gate at batch level."""
    nocue_logprobs: Optional[torch.Tensor] = None
    """[Cue-Gate] Padded teacher log probabilities on the NO-CUE (original) prompt, aligned to the
    student sequence. Used with `teacher_logprobs` (cue view) to compute the cue-utility gate."""
    routed_experts: Optional[torch.Tensor] = None
    """Padded routed experts for the total tokens."""
    multi_modal_inputs: Optional[dict[str, torch.Tensor]] = None
    """Multi-modal inputs for processors (e.g. pixel_values, image_grid_thw, video_grid_thw)."""
    extra_fields: dict[str, Any] = {}
    """Extra fields for dynamic addition."""


class DictConfigWrap:
    """Wrapper for DictConfig to avoid hydra.utils.instantiate recursive resolve."""

    def __init__(self, config: DictConfig):
        self.config = config


class ToolListWrap:
    """Wraps a tool list so ``hydra.utils.instantiate`` doesn't recursively
    resolve its elements (which would demote them to ``DictConfig``)."""

    def __init__(self, tools: list):
        self.tools = tools


class AgentLoopBase(ABC):
    """An agent loop takes an input message, chat with OpenAI compatible LLM server and interact with various
    environments.

    Args:
        trainer_config (DictConfig): whole config for main entrypoint.
        server_manager (LLMServerClient): OpenAI compatible LLM server manager.
        tokenizer (AutoTokenizer): Tokenizer for tokenize messages.
        processor (AutoProcessor): Processor for process messages.
        dataset_cls (type[Dataset]): Dataset class for creating dataset, Defaults to RLHFDataset.
        data_config (DictConfigWrap): Dataset config.
    """

    def __init__(
        self,
        trainer_config: DictConfigWrap,
        server_manager: LLMServerClient,
        tokenizer: AutoTokenizer,
        processor: AutoProcessor,
        dataset_cls: type[RLHFDataset],
        data_config: DictConfigWrap,
        **kwargs,
    ):
        self.config = trainer_config.config
        self.rollout_config = self.config.actor_rollout_ref.rollout
        self.server_manager = server_manager
        self.tokenizer = tokenizer
        self.processor = processor
        self.dataset_cls = dataset_cls
        self.data_config = data_config.config
        self.apply_chat_template_kwargs = self.data_config.get("apply_chat_template_kwargs", {})
        self.mm_processor_kwargs = self.data_config.get("mm_processor_kwargs", {})
        processing_class = self.processor if self.processor is not None else self.tokenizer
        self.system_prompt = initialize_system_prompt(processing_class, **self.apply_chat_template_kwargs)
        self.loop = get_event_loop()

    def _get_mm_processor_kwargs(self, audio_data: Optional[list[Any]] = None) -> dict[str, Any]:
        mm_processor_kwargs = dict(self.mm_processor_kwargs or {})
        if audio_data is not None and "sampling_rate" not in mm_processor_kwargs:
            sampling_rate = getattr(getattr(self.processor, "feature_extractor", None), "sampling_rate", None)
            if sampling_rate is not None:
                mm_processor_kwargs["sampling_rate"] = int(sampling_rate)
        return mm_processor_kwargs

    async def process_vision_info(self, messages: list[dict]) -> dict:
        """Backward-compatible wrapper for multi-modal extraction."""
        return await self.process_multi_modal_info(messages)

    async def process_multi_modal_info(self, messages: list[dict]) -> dict:
        """Extract images, videos and audios from messages.

        Args:
            messages (list[dict]): Input messages.

        Returns:
            dict: Multi-modal data with keys like "images", "videos" and "audios".
        """
        multi_modal_data = {}
        if self.processor is not None:
            image_patch_size = getattr(getattr(self.processor, "image_processor", None), "patch_size", 14)
            try:
                if hasattr(self.dataset_cls, "process_multi_modal_info"):
                    images, videos, audios = await self.dataset_cls.process_multi_modal_info(
                        messages, image_patch_size=image_patch_size, config=self.data_config
                    )
                else:
                    images, videos = await self.dataset_cls.process_vision_info(
                        messages, image_patch_size=image_patch_size, config=self.data_config
                    )
                    audios = None
            except Exception as e:
                print(f"[AgentLoop] Failed to process multi-modal info, using dummy frame: {e}", flush=True)
                # Create a dummy single-frame black video so downstream code doesn't crash
                import torch
                dummy_video = torch.zeros(1, 3, 224, 224, dtype=torch.float32)
                images, audios = None, None
                videos = [(dummy_video, {"fps": 1.0, "num_frames": 1})]
            if images is not None:
                multi_modal_data["images"] = images
            if videos is not None:
                multi_modal_data["videos"] = videos
            if audios is not None:
                multi_modal_data["audios"] = audios

        return multi_modal_data

    async def apply_chat_template(
        self,
        messages: list[dict],
        tools: list[dict] = None,
        images: list[Image.Image] = None,
        videos: list[tuple[torch.Tensor, dict]] = None,
        audios: list[Any] = None,
        mm_processor_kwargs: Optional[dict[str, Any]] = None,
        remove_system_prompt: bool = False,
    ):
        """Apply chat template to messages with optional tools, images, and videos.

        Args:
            messages (list[dict]): Input messages.
            tools (list[dict], optional): Tools schemas. Defaults to None.
            images (list[Image.Image], optional): Input images. Defaults to None.
            videos (list[tuple[torch.Tensor, dict]], optional): Input videos. Defaults to None.
            remove_system_prompt (bool, optional): Whether to remove system prompt. Defaults to False.

        Returns:
            list[int]: Prompt token ids.
        """
        if self.processor is not None:
            raw_prompt = await self.loop.run_in_executor(
                None,
                lambda: apply_chat_template(
                    self.processor,
                    messages,
                    tools=tools,
                    add_generation_prompt=True,
                    tokenize=False,
                    **self.apply_chat_template_kwargs,
                ),
            )

            model_inputs = build_multimodal_processor_inputs(
                self.processor,
                text=[raw_prompt],
                images=images,
                videos=videos,
                audio=audios,
                mm_processor_kwargs=mm_processor_kwargs
                if mm_processor_kwargs is not None
                else self._get_mm_processor_kwargs(audios),
            )
            prompt_ids = normalize_token_ids(model_inputs.pop("input_ids"))
        else:
            tokenized_prompt = await self.loop.run_in_executor(
                None,
                lambda: apply_chat_template(
                    self.tokenizer,
                    messages,
                    tools=tools,
                    add_generation_prompt=True,
                    tokenize=True,
                    **self.apply_chat_template_kwargs,
                ),
            )
            prompt_ids = normalize_token_ids(tokenized_prompt)

        if remove_system_prompt:
            prompt_ids = prompt_ids[len(self.system_prompt) :]

        return prompt_ids

    @abstractmethod
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        """Run agent loop to interact with LLM server and environment.

        Args:
            sampling_params (Dict[str, Any]): LLM sampling params.
            **kwargs: dataset fields from `verl.utils.dataset.RLHFDataset`.

        Returns:
            AgentLoopOutput: Agent loop output.
        """
        raise NotImplementedError


"""Agent loop registry: key is agent_name, value is a dict of agent loop config
used by hydra.utils.instantiate to initialize agent loop instance.

https://hydra.cc/docs/advanced/instantiate_objects/overview/
"""
_agent_loop_registry: dict[str, dict] = {}


def register(agent_name: str):
    """Register agent loop class."""

    def decorator(subclass: type[AgentLoopBase]) -> type[AgentLoopBase]:
        fqdn = f"{subclass.__module__}.{subclass.__qualname__}"
        _agent_loop_registry[agent_name] = {"_target_": fqdn}
        return subclass

    return decorator


class AgentLoopWorker:
    """Agent loop worker takes a batch of messages and run each message in an agent loop.

    Args:
        config (DictConfig): whole config for main entrypoint.
        llm_client (LLMServerClient): Client for the LLM server.
        teacher_client (dict[str, LLMServerClient]): Client for multiple teacher servers.
        reward_loop_worker_handles (List[ray.actor.ActorHandle]): Actor handles for streaming reward computation.
    """

    def __init__(
        self,
        config: DictConfig,
        llm_client: LLMServerClient,
        teacher_client: dict[str, LLMServerClient] = None,
        reward_loop_worker_handles: list[ray.actor.ActorHandle] = None,
    ):
        self.config = config
        self.llm_client = llm_client
        self.teacher_client = teacher_client
        self.reward_loop_worker_handles = reward_loop_worker_handles

        rollout_config, model_config = config.actor_rollout_ref.rollout, config.actor_rollout_ref.model
        self.rollout_config: RolloutConfig = omega_conf_to_dataclass(rollout_config)
        self.model_config: HFModelConfig = omega_conf_to_dataclass(model_config)

        self.dataset_cls = get_dataset_class(config.data)
        self.tokenizer = self.model_config.tokenizer
        self.processor = self.model_config.processor
        self.mm_processor_kwargs = config.data.get("mm_processor_kwargs", {})

        # Online policy distillation
        self.distillation_enabled = is_distillation_enabled(config.distillation)
        if self.distillation_enabled:
            from verl.experimental.teacher_loop.teacher_manager import AsyncTeacherLLMServerManager

            self.teacher_key: str = config.distillation.teacher_key
            self.teacher_server_manager = AsyncTeacherLLMServerManager(
                config=config,
                teacher_client=teacher_client,
            )

        # [ExOPD] Track whether base model logprobs are needed
        self.exopd_enabled = (
            self.distillation_enabled
            and config.distillation.get("distillation_loss", {}).get("exopd_enabled", False)
        )

        # [AD-ExOPD] Axis-decomposed ExOPD needs a second base forward on dense frames.
        self.exopd_axis_decomposed = (
            self.exopd_enabled
            and config.distillation.get("distillation_loss", {}).get("exopd_axis_decomposed", False)
        )

        # [V-Zero-Video] Contrastive evidence gating: the teacher additionally scores each rollout
        # under a NEGATIVE (degraded) video view. When enabled, we do one extra teacher logprob pass
        # per rollout on the degraded frames and stash it as `neg_view_logprobs`; the group-relative
        # gate itself is computed at batch level in ray_trainer. Backward-compatible: default False.
        self.contrastive_gate_enabled = (
            self.distillation_enabled
            and config.distillation.get("distillation_loss", {}).get("contrastive_gate_enabled", False)
        )
        self.neg_view_mode = (
            config.distillation.get("distillation_loss", {}).get("neg_view_mode", "black")
            if self.contrastive_gate_enabled
            else "black"
        )
        if self.contrastive_gate_enabled:
            logger.info(f"[V-Zero] Contrastive evidence gating ENABLED: neg_view_mode={self.neg_view_mode!r}")

        # [Cue-Gate] Privileged-Information-Gated Distillation. When ViCue cue is active, also run
        # a no-cue teacher forward (on the ORIGINAL prompt) so the loss can weight distillation by
        # the cue's per-token marginal utility Δ = logp(y|cue) − logp(y|no_cue). Backward-compatible:
        # when cue_gate_enabled=False (default), no extra forward is done and behavior is unchanged.
        self.cue_gate_enabled = (
            self.distillation_enabled
            and config.distillation.get("distillation_loss", {}).get("cue_gate_enabled", False)
        )
        if self.cue_gate_enabled:
            logger.info("[Cue-Gate] Privileged-info gated distillation ENABLED: no-cue teacher pass active")

        # [Asymmetric Frame-Budget Distillation] Optional: let the teacher see a
        # *denser* set of frames than the student. Backward-compatible — when
        # `distillation.teacher_frame_budget` is unset (default), the teacher reuses
        # the student's decoded frames exactly as before, so behavior is unchanged.
        self.teacher_frame_budget = None
        if self.distillation_enabled:
            tfb = config.distillation.get("teacher_frame_budget", None)
            if tfb is not None and bool(tfb.get("enabled", False)):
                self.teacher_frame_budget = tfb
                logger.info(f"[AFD] Asymmetric frame-budget distillation ENABLED: {dict(tfb)}")

        # [Visual-Cue Distillation] Optional: let the teacher see a *different prompt* than the
        # student (student prompt + an appended visual-cue hint). Backward-compatible — when
        # `data.teacher_prompt_key` is unset (default), the teacher reuses the student's prompt
        # exactly as before, so behavior is unchanged. The cue is text-only privilege injected
        # into the teacher prompt; the student prompt (and inference interface) is untouched.
        self.teacher_prompt_key = None
        if self.distillation_enabled:
            tpk = config.data.get("teacher_prompt_key", None)
            if tpk:
                self.teacher_prompt_key = tpk
                logger.info(f"[ViCue] Visual-cue teacher-prompt distillation ENABLED: key={tpk!r}")

        # [Teacher-Think Distillation] Optional: let the teacher score the student's response with a
        # DIFFERENT thinking-mode chat template than the student. Motivated by OPSD (arXiv 2601.18734,
        # Table 5): student thinking-OFF + teacher thinking-ON maximizes the per-token KL signal on
        # content tokens. The student rollout keeps its own `data.apply_chat_template_kwargs.enable_thinking`
        # (typically False); only the teacher's prompt is re-tokenized with enable_thinking overridden.
        # Video frames are REUSED from the student (only the text template differs). Backward-compatible:
        # when `data.teacher_enable_thinking` is unset (default None), the teacher reuses the student's
        # prompt exactly as before. Mutually exclusive with AFD/ViCue (those take precedence).
        self.teacher_enable_thinking = None
        if self.distillation_enabled:
            tet = config.data.get("teacher_enable_thinking", None)
            if tet is not None:
                self.teacher_enable_thinking = bool(tet)
                logger.info(f"[TeacherThink] teacher enable_thinking override = {self.teacher_enable_thinking}")

        # Load tools once per worker; each trajectory just reuses self.tools.
        tool_config_path = self.rollout_config.multi_turn.tool_config_path
        function_tool_path = self.rollout_config.multi_turn.function_tool_path
        self.tools = load_all_tools(
            tool_config_path=resolve_config_path(tool_config_path) if tool_config_path else None,
            function_tool_path=resolve_config_path(function_tool_path) if function_tool_path else None,
        )

        # Load custom agent loop implementations from config path
        agent_loop_config_path = self.rollout_config.agent.agent_loop_config_path
        if agent_loop_config_path:
            resolved_path = resolve_config_path(agent_loop_config_path)
            agent_loop_configs = OmegaConf.load(resolved_path)
            for agent_loop_config in agent_loop_configs:
                _agent_loop_registry[agent_loop_config.name] = agent_loop_config
        if self.model_config.get("custom_chat_template", None) is not None:
            if self.model_config.processor is not None:
                self.model_config.processor.chat_template = self.model_config.custom_chat_template
            self.model_config.tokenizer.chat_template = self.model_config.custom_chat_template

        trace_config = self.rollout_config.trace
        RolloutTraceConfig.init(
            self.rollout_config.trace.project_name,
            self.rollout_config.trace.experiment_name,
            trace_config.get("backend"),
            trace_config.get("token2text", False),
            trace_config.get("max_samples_per_step_per_worker", None),
        )

    def _get_mm_processor_kwargs(self, audio_data: Optional[list[Any]] = None) -> dict[str, Any]:
        """Return multimodal processor kwargs with audio sampling-rate defaults."""
        mm_processor_kwargs = dict(self.mm_processor_kwargs or {})
        if audio_data is not None and "sampling_rate" not in mm_processor_kwargs:
            sampling_rate = getattr(getattr(self.processor, "feature_extractor", None), "sampling_rate", None)
            if sampling_rate is not None:
                mm_processor_kwargs["sampling_rate"] = int(sampling_rate)
        return mm_processor_kwargs

    async def generate_sequences(self, batch: DataProto) -> DataProto:
        """Generate sequences from agent loop.

        Args:
            batch (DataProto): Input batch.

        Returns:
            DataProto: Output batch.
            - prompts: [bsz, prompt_length], prompt token ids from dataset.
            - responses: [bsz, response_length], output token ids include response tokens
              from LLM generation and observation tokens from tool_calls.
            - response_mask: [bsz, response_length], 1 for LLM generated tokens, 0 for observation/padding tokens.
            - input_ids: [bsz, prompt_length + response_length], whole sequence token ids, including prompt tokens
              and response tokens.
            - attention_mask: [bsz, prompt_length + response_length], 0 for padding tokens, 1 for other tokens.
            - position_ids: [bsz, prompt_length + response_length], incremental position ids.

            For multi-turn conversations:
            responses:     |<- LLM generation ->|<- tool_calls ->|<- LLM generation ->|<- padding ->|
            response_mask: | 1, 1, 1, ..., 1, 1 | 0, 0, .., 0, 0 | 1, 1, 1, ..., 1, 1 | 0, 0, ..., 0|
        """
        config = self.rollout_config
        validate = batch.meta_info.get("validate", False)
        sampling_params = dict(
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            repetition_penalty=1.0,
            logprobs=config.calculate_log_probs,
        )

        def apply_greedy_sampling_params(params: dict[str, Any]) -> None:
            params["top_p"] = 1.0
            params["top_k"] = -1
            params["temperature"] = 0

        # override sampling params for validation
        if validate:
            sampling_params["top_p"] = config.val_kwargs.top_p
            sampling_params["top_k"] = config.val_kwargs.top_k
            sampling_params["temperature"] = config.val_kwargs.temperature

        # by default, we assume it's a single turn agent
        if "agent_name" not in batch.non_tensor_batch:
            default_agent_loop = config.agent.default_agent_loop
            batch.non_tensor_batch["agent_name"] = np.array([default_agent_loop] * len(batch), dtype=object)

        if "index" in batch.non_tensor_batch:
            index = batch.non_tensor_batch["index"]
        else:
            index = np.arange(len(batch))

        max_samples_per_worker = RolloutTraceConfig.get_instance().max_samples_per_step_per_worker

        # For n rollouts per sample, we trace all n rollouts for selected samples
        # Note: This sampling happens per-worker, so total traces = max_samples_per_worker * num_workers * n
        if max_samples_per_worker is not None:
            unique_sample_indices = np.unique(index)
            if max_samples_per_worker < len(unique_sample_indices):
                selected_samples = set(
                    np.random.choice(unique_sample_indices, max_samples_per_worker, replace=False).tolist()
                )
                traced_indices = set(i for i in range(len(batch)) if index[i] in selected_samples)
            else:
                traced_indices = set(range(len(batch)))
        else:
            traced_indices = set(range(len(batch)))

        trajectory_info = await get_trajectory_info(
            batch.meta_info.get("global_steps", -1), index.tolist(), batch.meta_info.get("validate", False)
        )

        # NOTE: __do_sample__ is an internal per-sample override used by REMAX combined rollout.
        # Do not forward it to concrete agent loops, which may reject unknown kwargs.
        per_sample_do_sample = batch.non_tensor_batch.get("__do_sample__")
        tasks = []
        for i in range(len(batch)):
            trace_this_sample = i in traced_indices
            kwargs = {k: v[i] for k, v in batch.non_tensor_batch.items() if k != "__do_sample__"}
            sample_sampling_params = dict(sampling_params)
            if not validate and per_sample_do_sample is not None and not bool(per_sample_do_sample[i]):
                apply_greedy_sampling_params(sample_sampling_params)
            tasks.append(
                asyncio.create_task(
                    self._run_agent_loop(sample_sampling_params, trajectory_info[i], trace=trace_this_sample, **kwargs)
                )
            )
        outputs = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out failed samples (e.g. video decode errors, prompt too long)
        failed_indices = []
        for i, out in enumerate(outputs):
            if isinstance(out, BaseException):
                print(f"[AgentLoop] Sample {i} failed: {type(out).__name__}: {out}", flush=True)
                failed_indices.append(i)

        if failed_indices:
            good_idx = next((i for i in range(len(outputs)) if not isinstance(outputs[i], BaseException)), None)
            if good_idx is None:
                # All samples failed - retry with shorter max_tokens to get at least one through
                print(f"[AgentLoop] WARNING: All {len(outputs)} samples failed. Retrying sample 0 with truncated prompt.", flush=True)
                # Re-run first sample without video (text-only fallback)
                kwargs_retry = {k: v[0] for k, v in batch.non_tensor_batch.items() if k != "__do_sample__"}
                # Strip video from raw_prompt to make it short enough
                raw_prompt = kwargs_retry.get("raw_prompt", [])
                if isinstance(raw_prompt, list):
                    for msg in raw_prompt:
                        if isinstance(msg.get("content"), list):
                            msg["content"] = [c for c in msg["content"] if c.get("type") != "video"]
                retry_task = asyncio.create_task(
                    self._run_agent_loop(sampling_params, trajectory_info[0], trace=False, **kwargs_retry)
                )
                retry_result = await asyncio.gather(retry_task, return_exceptions=True)
                if isinstance(retry_result[0], BaseException):
                    # Even retry failed - re-raise
                    raise outputs[0]
                # Use retry result for all slots
                for i in range(len(outputs)):
                    outputs[i] = retry_result[0]
            else:
                for fi in failed_indices:
                    outputs[fi] = outputs[good_idx]

        output = self._postprocess(
            outputs, input_non_tensor_batch=batch.non_tensor_batch, validate=batch.meta_info.get("validate", False)
        )
        return output

    async def _run_agent_loop(
        self,
        sampling_params: dict[str, Any],
        trajectory: dict[str, Any],
        *,
        agent_name: str,
        trace: bool = True,
        **kwargs,
    ) -> _InternalAgentLoopOutput:
        with rollout_trace_attr(
            step=trajectory["step"],
            sample_index=trajectory["sample_index"],
            rollout_n=trajectory["rollout_n"],
            validate=trajectory["validate"],
            name="agent_loop",
            trace=trace,
        ):
            assert agent_name in _agent_loop_registry, (
                f"Agent loop {agent_name} not registered, registered agent loops: {_agent_loop_registry.keys()}"
            )

            agent_loop_config = _agent_loop_registry[agent_name]
            agent_loop = hydra.utils.instantiate(
                config=agent_loop_config,
                trainer_config=DictConfigWrap(config=self.config),
                server_manager=self.llm_client,
                tokenizer=self.tokenizer,
                processor=self.processor,
                dataset_cls=self.dataset_cls,
                data_config=DictConfigWrap(self.config.data),
                tools=ToolListWrap(self.tools),
            )
            output: AgentLoopOutput = await agent_loop.run(sampling_params, **kwargs)
            return await self._agent_loop_postprocess(output, trajectory["validate"], **kwargs)

    async def _agent_loop_postprocess(self, output, validate, **kwargs) -> _InternalAgentLoopOutput:
        """Perform post-processing operations on the output of each individual agent loop."""
        output.extra_fields["raw_prompt"] = kwargs["raw_prompt"]

        # Some AgentLoop may have already computed the reward score, e.g SWE-agent.

        # NOTE: consistent with the legacy batch version of generate_sequences that existed in the
        # deprecated vLLM SPMD rollout implementation.
        # prompt_ids: left padded with zeros (e.g., [0,0,0,0,1,2,3,4])
        # response_ids: right padded with zeros (e.g., [5,6,7,8,0,0,0,0])
        # input_ids: concatenation of prompt + response
        # Mask:
        # For example, if the prompt is [1,2,3,4] and the response is [5,6,7,(tool start)8,9(tool end),10,11,12]
        # - prompt_attention_mask: 0s for padding, 1s for tokens
        #   e.g., [0,0,0,0,1,1,1,1]
        # - response_attention_mask: 0s for padding, 1s for tokens
        #   e.g., [1,1,1,1,1,1,1,1,1,1,1,0,0,0,0]
        # attention_mask: concatenation of prompt_attention_mask and response_attention_mask
        #   e.g., [0,0,0,0,1,1,1,1(prompt),1,1,1,1,1,1,1,1,1,1,1,0,0,0,0(response)]
        # - response_mask: 1s for LLM generated tokens, 0 for tool response/padding tokens
        #   e.g., [1,1,1,1,1,1,1,(tool start),0,0(tool end),1,1,0,0,0,0]
        # - position_ids: sequential positions for tokens, starting at 0
        #   e.g., [0,0,0,0,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,0,0,0,0]

        # TODO(wuxibin): remove padding and use tensordict.
        self.tokenizer.padding_side = "left"
        prompt_output = self.tokenizer.pad(
            {"input_ids": output.prompt_ids},
            padding="max_length",
            max_length=self.rollout_config.prompt_length,
            return_tensors="pt",
            return_attention_mask=True,
        )
        if prompt_output["input_ids"].dim() == 1:
            prompt_output["input_ids"] = prompt_output["input_ids"].unsqueeze(0)
            prompt_output["attention_mask"] = prompt_output["attention_mask"].unsqueeze(0)

        self.tokenizer.padding_side = "right"
        response_output = self.tokenizer.pad(
            {"input_ids": output.response_ids},
            padding="max_length",
            max_length=self.rollout_config.response_length,
            return_tensors="pt",
            return_attention_mask=True,
        )
        if response_output["input_ids"].dim() == 1:
            response_output["input_ids"] = response_output["input_ids"].unsqueeze(0)
            response_output["attention_mask"] = response_output["attention_mask"].unsqueeze(0)

        response_mask_output = self.tokenizer.pad(
            {"input_ids": output.response_mask},
            padding="max_length",
            max_length=self.rollout_config.response_length,
            return_tensors="pt",
            return_attention_mask=False,
        )
        if response_mask_output["input_ids"].dim() == 1:
            response_mask_output["input_ids"] = response_mask_output["input_ids"].unsqueeze(0)

        response_logprobs = None
        if output.response_logprobs is not None:
            pad_size = self.rollout_config.response_length - len(output.response_logprobs)
            response_logprobs = torch.tensor(output.response_logprobs + [0.0] * pad_size).unsqueeze(0)

        response_mask = response_mask_output["input_ids"] * response_output["attention_mask"]
        attention_mask = torch.cat([prompt_output["attention_mask"], response_output["attention_mask"]], dim=1)
        input_ids = torch.cat([prompt_output["input_ids"], response_output["input_ids"]], dim=1)

        routed_experts = None
        if output.routed_experts is not None:
            total_length = input_ids.shape[1]
            length, layer_num, topk_num = output.routed_experts.shape
            if isinstance(output.routed_experts, np.ndarray):
                routed_experts_array = output.routed_experts
                if not routed_experts_array.flags.writeable:
                    routed_experts_array = routed_experts_array.copy()
                experts_tensor = torch.from_numpy(routed_experts_array)
            elif isinstance(output.routed_experts, torch.Tensor):
                experts_tensor = output.routed_experts
            else:
                raise TypeError(f"Unsupported type for routed_experts: {type(output.routed_experts)}")
            routed_experts = torch.zeros(1, total_length, layer_num, topk_num, dtype=experts_tensor.dtype)

            # Calculate start position: left padding means original prompt starts at the end
            start_pos = prompt_output["input_ids"].shape[1] - len(output.prompt_ids)
            end_pos = min(start_pos + length, total_length)

            # Add boundary checks for robustness
            if start_pos < 0 or end_pos > total_length:
                raise ValueError(
                    f"Invalid position range: start_pos={start_pos}, end_pos={end_pos}, total_length={total_length}"
                )

            routed_experts[:, start_pos:end_pos] = experts_tensor.unsqueeze(0)

        multi_modal_inputs = self._compute_multi_modal_inputs(output, input_ids)
        position_ids = self._compute_position_ids(
            input_ids,
            attention_mask,
            multi_modal_inputs,
            output.mm_processor_kwargs
            if output.mm_processor_kwargs is not None
            else self._get_mm_processor_kwargs(
                output.multi_modal_data.get("audios") if output.multi_modal_data else None
            ),
        )
        await self._compute_score([output], kwargs=kwargs)
        await self._compute_teacher_logprobs(
            output,
            prompt_ids=output.prompt_ids,
            response_ids=output.response_ids,
            validate=validate,
            sample_kwargs=kwargs,
        )
        teacher_ids, teacher_logprobs = (
            output.extra_fields.pop("teacher_ids", None),
            output.extra_fields.pop("teacher_logprobs", None),
        )
        if teacher_ids is not None and teacher_logprobs is not None:
            # TODO(wuxibin): remove padding and use tensordict.
            from verl.experimental.teacher_loop.teacher_manager import _pad_teacher_outputs

            teacher_ids, teacher_logprobs = _pad_teacher_outputs(
                teacher_ids,
                teacher_logprobs,
                prompt_width=prompt_output["input_ids"].shape[1],
                response_width=response_output["input_ids"].shape[1],
                prompt_length=len(output.prompt_ids),
                response_length=len(output.response_ids),
                pad_token_id=self.tokenizer.pad_token_id,
            )

        # [ExOPD] Pad base model outputs the same way
        base_ids, base_logprobs = (
            output.extra_fields.pop("base_ids", None),
            output.extra_fields.pop("base_logprobs", None),
        )
        if base_ids is not None and base_logprobs is not None:
            from verl.experimental.teacher_loop.teacher_manager import _pad_teacher_outputs

            base_ids, base_logprobs = _pad_teacher_outputs(
                base_ids,
                base_logprobs,
                prompt_width=prompt_output["input_ids"].shape[1],
                response_width=response_output["input_ids"].shape[1],
                prompt_length=len(output.prompt_ids),
                response_length=len(output.response_ids),
                pad_token_id=self.tokenizer.pad_token_id,
            )

        # [AD-ExOPD] Pad dense-frame base model outputs the same way
        base_dense_ids, base_dense_logprobs = (
            output.extra_fields.pop("base_dense_ids", None),
            output.extra_fields.pop("base_dense_logprobs", None),
        )
        if base_dense_ids is not None and base_dense_logprobs is not None:
            from verl.experimental.teacher_loop.teacher_manager import _pad_teacher_outputs

            base_dense_ids, base_dense_logprobs = _pad_teacher_outputs(
                base_dense_ids,
                base_dense_logprobs,
                prompt_width=prompt_output["input_ids"].shape[1],
                response_width=response_output["input_ids"].shape[1],
                prompt_length=len(output.prompt_ids),
                response_length=len(output.response_ids),
                pad_token_id=self.tokenizer.pad_token_id,
            )

        # [V-Zero-Video] Pad negative-view teacher outputs the same way
        neg_view_ids, neg_view_logprobs = (
            output.extra_fields.pop("neg_view_ids", None),
            output.extra_fields.pop("neg_view_logprobs", None),
        )
        if neg_view_ids is not None and neg_view_logprobs is not None:
            from verl.experimental.teacher_loop.teacher_manager import _pad_teacher_outputs

            neg_view_ids, neg_view_logprobs = _pad_teacher_outputs(
                neg_view_ids,
                neg_view_logprobs,
                prompt_width=prompt_output["input_ids"].shape[1],
                response_width=response_output["input_ids"].shape[1],
                prompt_length=len(output.prompt_ids),
                response_length=len(output.response_ids),
                pad_token_id=self.tokenizer.pad_token_id,
            )

        # [Cue-Gate] Pad no-cue teacher outputs the same way
        nocue_ids, nocue_logprobs = (
            output.extra_fields.pop("nocue_ids", None),
            output.extra_fields.pop("nocue_logprobs", None),
        )
        if nocue_ids is not None and nocue_logprobs is not None:
            from verl.experimental.teacher_loop.teacher_manager import _pad_teacher_outputs

            nocue_ids, nocue_logprobs = _pad_teacher_outputs(
                nocue_ids,
                nocue_logprobs,
                prompt_width=prompt_output["input_ids"].shape[1],
                response_width=response_output["input_ids"].shape[1],
                prompt_length=len(output.prompt_ids),
                response_length=len(output.response_ids),
                pad_token_id=self.tokenizer.pad_token_id,
            )

        return _InternalAgentLoopOutput(
            prompt_ids=prompt_output["input_ids"],
            response_ids=response_output["input_ids"],
            input_ids=input_ids,
            position_ids=position_ids,
            response_mask=response_mask,
            attention_mask=attention_mask,
            response_logprobs=response_logprobs,
            routed_experts=routed_experts,
            multi_modal_inputs=multi_modal_inputs,
            multi_modal_data=output.multi_modal_data,
            mm_processor_kwargs=output.mm_processor_kwargs,
            teacher_logprobs=teacher_logprobs,
            teacher_ids=teacher_ids,
            base_logprobs=base_logprobs,
            base_ids=base_ids,
            base_dense_logprobs=base_dense_logprobs,
            base_dense_ids=base_dense_ids,
            neg_view_logprobs=neg_view_logprobs,
            nocue_logprobs=nocue_logprobs,
            reward_score=output.reward_score,
            num_turns=output.num_turns,
            metrics=output.metrics,
            extra_fields=output.extra_fields,
        )

    def _compute_multi_modal_inputs(self, output, input_ids) -> dict[str, torch.Tensor]:
        """Compute multi-modal inputs with image, video and audio."""
        multi_modal_inputs = {}
        if self.processor is None:
            return multi_modal_inputs

        multi_modal_data = output.multi_modal_data or {}
        images = multi_modal_data.get("images")
        videos = multi_modal_data.get("videos")
        audios = multi_modal_data.get("audios")
        current_text = self.tokenizer.decode(input_ids.squeeze(0), skip_special_tokens=True)

        multi_modal_inputs = build_multimodal_processor_inputs(
            self.processor,
            text=[current_text],
            images=images,
            videos=videos,
            audio=audios,
            mm_processor_kwargs=output.mm_processor_kwargs
            if output.mm_processor_kwargs is not None
            else self._get_mm_processor_kwargs(audios),
        )
        multi_modal_inputs.pop("input_ids", None)
        multi_modal_inputs.pop("attention_mask", None)

        # We must use dict(multi_modal_inputs) to convert BatchFeature values to a new dict
        # because np.array() only keeps the keys for BatchFeature.
        multi_modal_inputs = dict(multi_modal_inputs.convert_to_tensors("pt"))
        image_grid_thw = multi_modal_inputs.get("image_grid_thw")
        if image_grid_thw is not None:
            images_seqlens = torch.repeat_interleave(image_grid_thw[:, 1] * image_grid_thw[:, 2], image_grid_thw[:, 0])
            multi_modal_inputs["images_seqlens"] = images_seqlens
        return multi_modal_inputs

    def _compute_position_ids(
        self,
        input_ids,
        attention_mask,
        multi_modal_inputs,
        mm_processor_kwargs: Optional[dict[str, Any]] = None,
    ) -> torch.Tensor:
        """Compute position ids for multi-modal inputs."""
        if self.processor is None:
            return compute_position_id_with_mask(attention_mask)  # (1, seq_len)

        multi_modal_kwargs = {
            "image_grid_thw": multi_modal_inputs.get("image_grid_thw"),
            "video_grid_thw": multi_modal_inputs.get("video_grid_thw"),
        }
        # For transformers>=5.3.0, mm_token_type_ids is only used to calculate position ids.
        if multi_modal_inputs.pop("mm_token_type_ids", None) is not None:
            mm_token_type_ids = torch.zeros_like(input_ids)
            image_token_id = get_processor_token_id(self.processor, "image")
            video_token_id = get_processor_token_id(self.processor, "video")
            if image_token_id is not None:
                mm_token_type_ids[0][input_ids[0] == image_token_id] = 1
            if video_token_id is not None:
                mm_token_type_ids[0][input_ids[0] == video_token_id] = 2
            multi_modal_kwargs["mm_token_type_ids"] = mm_token_type_ids

        # Model's get_rope_index has been dynamically bind to the processor.
        vision_position_ids, _ = self.processor.get_rope_index(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **multi_modal_kwargs,
        )
        vision_position_ids = vision_position_ids.transpose(0, 1)  # (3, 1, seq_len) => (1, 3, seq_len)

        valid_mask = attention_mask[0].bool()
        text_position_ids = torch.ones((1, len(input_ids[0])), dtype=torch.long)
        text_position_ids[0, valid_mask] = torch.arange(valid_mask.sum().item())
        text_position_ids = text_position_ids.unsqueeze(0)
        position_ids = torch.cat((text_position_ids, vision_position_ids), dim=1)  # (1, 4, seq_length)
        return position_ids

    async def _compute_score(self, outputs: list[AgentLoopOutput], kwargs: dict) -> None:
        """Compute reward score for all outputs in a trajectory; assigns result to outputs[-1]."""
        enable_async_reward = self.reward_loop_worker_handles is not None

        final_output = outputs[-1]
        if final_output.reward_score is None and enable_async_reward:
            timing = {}
            with simple_timer("compute_score", timing):
                all_prompts, all_responses, all_input_ids, all_attention_mask, all_position_ids = [], [], [], [], []
                for output in outputs:
                    prompts = torch.tensor(output.prompt_ids, dtype=torch.int64)
                    responses = torch.tensor(output.response_ids, dtype=torch.int64)
                    input_ids = torch.cat([prompts, responses], dim=0)
                    attention_mask = torch.ones_like(input_ids, dtype=torch.int64)
                    multi_modal_inputs = self._compute_multi_modal_inputs(output, input_ids)
                    position_ids = self._compute_position_ids(
                        input_ids.unsqueeze(0),
                        attention_mask.unsqueeze(0),
                        multi_modal_inputs,
                        output.mm_processor_kwargs
                        if output.mm_processor_kwargs is not None
                        else self._get_mm_processor_kwargs(
                            output.multi_modal_data.get("audios") if output.multi_modal_data else None
                        ),
                    ).squeeze(0)
                    all_prompts.append(prompts)
                    all_responses.append(responses)
                    all_input_ids.append(input_ids)
                    all_attention_mask.append(attention_mask)
                    all_position_ids.append(position_ids)

                n = len(outputs)
                batch = TensorDict(
                    {
                        "prompts": torch.nn.utils.rnn.pad_sequence(all_prompts, batch_first=True, padding_value=0),
                        "responses": torch.nn.utils.rnn.pad_sequence(all_responses, batch_first=True, padding_value=0),
                        "attention_mask": torch.nn.utils.rnn.pad_sequence(
                            all_attention_mask, batch_first=True, padding_value=0
                        ),
                        "input_ids": torch.nn.utils.rnn.pad_sequence(all_input_ids, batch_first=True, padding_value=0),
                        "position_ids": torch.nn.utils.rnn.pad_sequence(
                            all_position_ids, batch_first=True, padding_value=0
                        ),
                    },
                    batch_size=n,
                )
                non_tensor_batch = {
                    **{k: np.array([v] * n) for k, v in kwargs.items()},
                    "__num_turns__": np.array([o.num_turns for o in outputs]),
                    "tool_extra_fields": np.array([o.extra_fields for o in outputs], dtype=object),
                    "prompt_len": np.array([len(o.prompt_ids) for o in outputs]),
                    "response_len": np.array([len(o.response_ids) for o in outputs]),
                }

                data = DataProto(
                    batch=batch,
                    non_tensor_batch=non_tensor_batch,
                )
                selected_reward_loop_worker_handle = random.choice(self.reward_loop_worker_handles)
                result = await selected_reward_loop_worker_handle.compute_score.remote(data)
                final_output.reward_score = result["reward_score"]
                final_output.extra_fields["reward_extra_info"] = result["reward_extra_info"]
            final_output.metrics.compute_score = timing["compute_score"]

    async def _compute_teacher_logprobs(
        self,
        output: AgentLoopOutput,
        prompt_ids: list[int],
        response_ids: list[int],
        validate: bool,
        sample_kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        """Compute teacher logprobs for single sample."""
        if self.distillation_enabled and not validate:
            routing_key = None
            if sample_kwargs is not None:
                routing_value = sample_kwargs.get(self.teacher_key)
                if routing_value is not None:
                    # Non-tensor batch values arrive as 0-d numpy objects / arrays; normalize to Python.
                    routing_key = routing_value.item() if hasattr(routing_value, "item") else routing_value

            # Default path (unchanged): teacher reuses the student's decoded frames.
            teacher_sequence_ids = prompt_ids + response_ids
            teacher_mm_data = output.multi_modal_data
            teacher_mm_kwargs = output.mm_processor_kwargs
            student_sequence_ids = prompt_ids + response_ids  # for AFD alignment

            # [Asymmetric Frame-Budget Distillation] When enabled, re-decode a *denser*
            # set of frames for the teacher and re-tokenize its prompt. The student keeps
            # its sparse frames; only the response tokens (pure text, frame-count
            # independent) are used for the distillation loss, so alignment still holds
            # via the existing teacher_loop length-mismatch handling.
            afd_active = False
            cue_active = False
            if self.teacher_frame_budget is not None and sample_kwargs is not None:
                try:
                    teacher_prompt_ids, teacher_mm_data, teacher_mm_kwargs = (
                        await self._build_teacher_dense_inputs(sample_kwargs, output=output)
                    )
                    teacher_sequence_ids = teacher_prompt_ids + response_ids
                    afd_active = True
                except Exception as e:
                    logger.warning(
                        f"[AFD] teacher dense-frame decode failed, falling back to student frames: {e}"
                    )
                    teacher_sequence_ids = prompt_ids + response_ids
                    teacher_mm_data = output.multi_modal_data
                    teacher_mm_kwargs = output.mm_processor_kwargs

            # [Visual-Cue Distillation] When enabled and this sample carries a teacher_raw_prompt
            # (student prompt + appended cue hint), re-tokenize the teacher prompt. Video frames
            # are REUSED from the student (only the text prompt differs). Student prompt untouched.
            # Only the response tokens are used for the distillation loss, so teacher/student
            # prompt-length mismatch is handled by pad_to_sequence_ids (same as AFD). Mutually
            # exclusive with AFD (AFD takes precedence if both somehow enabled).
            elif self.teacher_prompt_key is not None and sample_kwargs is not None:
                teacher_raw_prompt = sample_kwargs.get("teacher_raw_prompt")
                if teacher_raw_prompt is not None:
                    try:
                        teacher_prompt_ids, teacher_mm_data, teacher_mm_kwargs = (
                            await self._build_teacher_prompt_inputs(teacher_raw_prompt, output=output)
                        )
                        teacher_sequence_ids = teacher_prompt_ids + response_ids
                        cue_active = True
                    except Exception as e:
                        logger.warning(
                            f"[ViCue] teacher-prompt tokenize failed, falling back to student prompt: {e}"
                        )
                        teacher_sequence_ids = prompt_ids + response_ids
                        teacher_mm_data = output.multi_modal_data
                        teacher_mm_kwargs = output.mm_processor_kwargs

            # [Teacher-Think Distillation] Re-tokenize the teacher's prompt with enable_thinking
            # overridden (e.g. teacher thinking-ON while student rolled out thinking-OFF). Reuses the
            # student's raw_prompt messages and decoded video frames; only the chat-template thinking
            # flag differs, so the teacher prompt length may differ from the student's — handled by
            # response-only extraction + pad_to_sequence_ids (same mechanism as AFD/ViCue).
            elif self.teacher_enable_thinking is not None and sample_kwargs is not None:
                teacher_raw_prompt = sample_kwargs.get("raw_prompt")
                if teacher_raw_prompt is not None:
                    try:
                        teacher_prompt_ids, teacher_mm_data, teacher_mm_kwargs = (
                            await self._build_teacher_prompt_inputs(
                                teacher_raw_prompt,
                                output=output,
                                enable_thinking_override=self.teacher_enable_thinking,
                            )
                        )
                        teacher_sequence_ids = teacher_prompt_ids + response_ids
                        cue_active = True
                    except Exception as e:
                        logger.warning(
                            f"[TeacherThink] teacher-prompt tokenize failed, falling back to student prompt: {e}"
                        )
                        teacher_sequence_ids = prompt_ids + response_ids
                        teacher_mm_data = output.multi_modal_data
                        teacher_mm_kwargs = output.mm_processor_kwargs

            teacher_ids, teacher_logprobs = await self.teacher_server_manager.compute_teacher_logprobs_single(
                sequence_ids=teacher_sequence_ids,
                multi_modal_data=teacher_mm_data,
                mm_processor_kwargs=teacher_mm_kwargs,
                routing_key=routing_key,
                response_length=len(response_ids),
                # AFD/ViCue: teacher sequence differs from student (denser frames or appended cue);
                # align/pad the teacher logprobs to the STUDENT sequence so batch cat + response-only
                # loss line up.
                pad_to_sequence_ids=student_sequence_ids if (afd_active or cue_active) else None,
            )
            output.extra_fields["teacher_ids"] = teacher_ids
            output.extra_fields["teacher_logprobs"] = teacher_logprobs

            # [Cue-Gate] When ViCue cue is active, also score the SAME student response under the
            # teacher on the ORIGINAL (no-cue) prompt. positive view = teacher_logprobs above (cue);
            # negative view = this no-cue pass. Δ = logp(y|cue) − logp(y|no_cue) is the cue's
            # per-token marginal utility, consumed in the loss to gate the distillation advantage.
            # Uses the same teacher pool (routing_key), reuses the student's decoded frames, and
            # pads to the STUDENT sequence so it lines up token-for-token with teacher_logprobs.
            if self.cue_gate_enabled and cue_active:
                try:
                    nocue_ids, nocue_logprobs = await self.teacher_server_manager.compute_teacher_logprobs_single(
                        sequence_ids=prompt_ids + response_ids,  # original (no-cue) prompt
                        multi_modal_data=output.multi_modal_data,
                        mm_processor_kwargs=output.mm_processor_kwargs,
                        routing_key=routing_key,
                        response_length=len(response_ids),
                        pad_to_sequence_ids=student_sequence_ids,
                    )
                    output.extra_fields["nocue_ids"] = nocue_ids
                    output.extra_fields["nocue_logprobs"] = nocue_logprobs
                except Exception as e:
                    logger.warning(
                        f"[Cue-Gate] no-cue teacher pass failed; gate defaults to 1 for this sample: {e}"
                    )

            # [ExOPD] Fetch base model logprobs using the same infrastructure
            if self.exopd_enabled:
                base_ids, base_logprobs = await self.teacher_server_manager.compute_teacher_logprobs_single(
                    sequence_ids=prompt_ids + response_ids,
                    multi_modal_data=output.multi_modal_data,
                    mm_processor_kwargs=output.mm_processor_kwargs,
                    routing_key="exopd_base",
                    response_length=len(response_ids),
                )
                output.extra_fields["base_ids"] = base_ids
                output.extra_fields["base_logprobs"] = base_logprobs

            # [AD-ExOPD] Fetch a SECOND base-model logprob with DENSE frames (same budget
            # as the teacher) to decompose the extrapolation direction into a frame-budget
            # (temporal) axis and a model-capacity axis. Reuses the same 4B base pool
            # (routing_key="exopd_base") — no extra GPU — just an additional forward with
            # the dense frames produced by the AFD dense-input builder.
            if self.exopd_enabled and self.exopd_axis_decomposed:
                if self.teacher_frame_budget is None or sample_kwargs is None:
                    logger.warning(
                        "[AD-ExOPD] axis_decomposed enabled but teacher_frame_budget/sample_kwargs "
                        "unavailable; base_dense falls back to sparse frames (degenerates toward standard ExOPD)."
                    )
                    bd_sequence_ids = prompt_ids + response_ids
                    bd_mm_data = output.multi_modal_data
                    bd_mm_kwargs = output.mm_processor_kwargs
                else:
                    try:
                        bd_prompt_ids, bd_mm_data, bd_mm_kwargs = await self._build_teacher_dense_inputs(
                            sample_kwargs, output=output
                        )
                        bd_sequence_ids = bd_prompt_ids + response_ids
                    except Exception as e:
                        logger.warning(f"[AD-ExOPD] base dense-frame decode failed: {e}; fallback to sparse")
                        bd_sequence_ids = prompt_ids + response_ids
                        bd_mm_data = output.multi_modal_data
                        bd_mm_kwargs = output.mm_processor_kwargs
                base_dense_ids, base_dense_logprobs = await self.teacher_server_manager.compute_teacher_logprobs_single(
                    sequence_ids=bd_sequence_ids,
                    multi_modal_data=bd_mm_data,
                    mm_processor_kwargs=bd_mm_kwargs,
                    routing_key="exopd_base",
                    response_length=len(response_ids),
                    pad_to_sequence_ids=prompt_ids + response_ids,
                )
                output.extra_fields["base_dense_ids"] = base_dense_ids
                output.extra_fields["base_dense_logprobs"] = base_dense_logprobs

            # [V-Zero-Video] Contrastive evidence gating: score the SAME student response under a
            # NEGATIVE (degraded) video view with the SAME teacher (routing_key="default", no extra
            # GPU). The degraded frames change token count vs the student only for "textonly"; in all
            # cases we pad/align the teacher logprobs back to the STUDENT sequence via
            # pad_to_sequence_ids, so the batch-level gate lines up per response token. The positive
            # view is the standard `teacher_logprobs` already computed above.
            if self.contrastive_gate_enabled and sample_kwargs is not None:
                try:
                    neg_prompt_ids, neg_mm_data, neg_mm_kwargs = await self._build_neg_view_inputs(
                        sample_kwargs, output=output, mode=self.neg_view_mode
                    )
                    neg_sequence_ids = neg_prompt_ids + response_ids
                    neg_view_ids, neg_view_logprobs = await self.teacher_server_manager.compute_teacher_logprobs_single(
                        sequence_ids=neg_sequence_ids,
                        multi_modal_data=neg_mm_data,
                        mm_processor_kwargs=neg_mm_kwargs,
                        routing_key=routing_key,
                        response_length=len(response_ids),
                        pad_to_sequence_ids=student_sequence_ids,
                    )
                    output.extra_fields["neg_view_ids"] = neg_view_ids
                    output.extra_fields["neg_view_logprobs"] = neg_view_logprobs
                except Exception as e:
                    logger.warning(
                        f"[V-Zero] negative-view teacher pass failed ({self.neg_view_mode}); "
                        f"gate will default to 1 for this sample: {e}"
                    )

    async def _build_neg_view_inputs(
        self,
        sample_kwargs: dict[str, Any],
        output: Optional[AgentLoopOutput],
        mode: str,
    ) -> tuple[list[int], Optional[dict[str, Any]], Optional[dict[str, Any]]]:
        """[V-Zero-Video] Build teacher inputs on a DEGRADED (negative) video view, reusing the
        student's already-decoded frames. Returns (prompt_ids, multi_modal_data, mm_processor_kwargs).

        Modes:
          * "black"    — replace every decoded frame with a zero tensor (same shape/resize, so token
                         count is identical; isolates dependence on ANY visual input).
          * "shuffle"  — permute the frames' temporal order with a DETERMINISTIC (index-based)
                         permutation so resume is reproducible (isolates dependence on temporal order).
          * "textonly" — drop the <video> element entirely (video vs language prior; cheapest pass).
        """
        import copy

        raw_prompt = sample_kwargs.get("raw_prompt")
        if raw_prompt is None:
            raise ValueError("raw_prompt missing from sample_kwargs; cannot build negative-view inputs")
        if self.processor is None:
            raise ValueError("Contrastive evidence gating requires a multimodal processor")

        messages = copy.deepcopy(list(raw_prompt))
        video_items = [
            item
            for msg in messages
            if isinstance(msg.get("content"), list)
            for item in msg["content"]
            if isinstance(item, dict) and item.get("type") == "video"
        ]

        if mode == "textonly":
            # Remove every <video> content element from the messages (also strip any now-empty text
            # references are left as-is; the chat template just drops the visual tokens).
            for msg in messages:
                content = msg.get("content")
                if isinstance(content, list):
                    msg["content"] = [
                        it for it in content if not (isinstance(it, dict) and it.get("type") == "video")
                    ]
            image_patch_size = getattr(getattr(self.processor, "image_processor", None), "patch_size", 14)
            images, videos, audios = await self.dataset_cls.process_multi_modal_info(
                messages, image_patch_size=image_patch_size, config=self.config.data
            )
            neg_mm_data: dict[str, Any] = {}
            if images is not None:
                neg_mm_data["images"] = images
            if videos is not None:
                neg_mm_data["videos"] = videos
            if audios is not None:
                neg_mm_data["audios"] = audios
            neg_mm_kwargs = self._get_mm_processor_kwargs(audios)
            apply_kwargs = self.config.data.get("apply_chat_template_kwargs", {})
            raw_text = apply_chat_template(
                self.processor, messages, tools=None, add_generation_prompt=True, tokenize=False, **apply_kwargs
            )
            model_inputs = build_multimodal_processor_inputs(
                self.processor, text=[raw_text], images=images, videos=videos, audio=audios,
                mm_processor_kwargs=neg_mm_kwargs,
            )
            return normalize_token_ids(model_inputs.pop("input_ids")), neg_mm_data, neg_mm_kwargs

        # "black" / "shuffle": reuse the student's decoded frames and perturb their pixels/order.
        student_videos = None
        if output is not None and output.multi_modal_data is not None:
            student_videos = output.multi_modal_data.get("videos")
        if not student_videos:
            raise ValueError("[V-Zero] student decoded videos unavailable; cannot build negative view")
        if len(video_items) != len(student_videos):
            raise ValueError(
                f"[V-Zero] video item count ({len(video_items)}) != student decoded videos "
                f"({len(student_videos)}); cannot align"
            )

        for vid_idx, (item, stu_video) in enumerate(zip(video_items, student_videos)):
            # stu_video is (tensor[T,C,H,W], metadata) when return_video_metadata=True; be tolerant.
            if isinstance(stu_video, (tuple, list)) and len(stu_video) == 2:
                stu_tensor, _stu_meta = stu_video
            else:
                stu_tensor = stu_video
            resized_h, resized_w = int(stu_tensor.shape[-2]), int(stu_tensor.shape[-1])
            n_frames = int(stu_tensor.shape[0])

            if mode == "black":
                frames_tensor = torch.zeros_like(stu_tensor)
            elif mode == "shuffle":
                # Deterministic index-based permutation (reverse + prime stride) so the same rollout
                # always yields the same negative order across resume; avoids global RNG.
                if n_frames > 1:
                    stride = 7 if n_frames % 7 != 0 else 5
                    perm = [((i * stride + vid_idx) % n_frames) for i in range(n_frames)]
                    # ensure it is a genuine permutation, not a collision-y mapping
                    if len(set(perm)) != n_frames:
                        perm = list(range(n_frames - 1, -1, -1))  # fallback: reverse
                    frames_tensor = stu_tensor[perm]
                else:
                    frames_tensor = stu_tensor
            else:
                raise ValueError(f"[V-Zero] unknown neg_view_mode={mode!r}")

            # Convert [T,C,H,W] tensor to a PIL frame list so qwen_vl_utils skips its own sampling
            # and uses exactly these (degraded) frames at the student's resize (token count preserved).
            arr = frames_tensor.permute(0, 2, 3, 1).to(torch.uint8).cpu().numpy()
            pil_frames = [Image.fromarray(arr[i]) for i in range(arr.shape[0])]
            item["video"] = pil_frames
            item["resized_height"] = resized_h
            item["resized_width"] = resized_w
            for k in ("fps", "nframes", "max_frames", "min_frames", "video_start", "video_end"):
                item.pop(k, None)

        image_patch_size = getattr(getattr(self.processor, "image_processor", None), "patch_size", 14)
        images, videos, audios = await self.dataset_cls.process_multi_modal_info(
            messages, image_patch_size=image_patch_size, config=self.config.data
        )
        neg_mm_data = {}
        if images is not None:
            neg_mm_data["images"] = images
        if videos is not None:
            neg_mm_data["videos"] = videos
        if audios is not None:
            neg_mm_data["audios"] = audios
        neg_mm_kwargs = self._get_mm_processor_kwargs(audios)
        apply_kwargs = self.config.data.get("apply_chat_template_kwargs", {})
        raw_text = apply_chat_template(
            self.processor, messages, tools=None, add_generation_prompt=True, tokenize=False, **apply_kwargs
        )
        model_inputs = build_multimodal_processor_inputs(
            self.processor, text=[raw_text], images=images, videos=videos, audio=audios,
            mm_processor_kwargs=neg_mm_kwargs,
        )
        return normalize_token_ids(model_inputs.pop("input_ids")), neg_mm_data, neg_mm_kwargs

    async def _build_teacher_dense_inputs(
        self, sample_kwargs: dict[str, Any], output: Optional[AgentLoopOutput] = None
    ) -> tuple[list[int], Optional[dict[str, Any]], Optional[dict[str, Any]]]:
        """[AFD] Re-decode dense frames for the teacher and re-tokenize its prompt.

        Returns (teacher_prompt_ids, teacher_multi_modal_data, teacher_mm_processor_kwargs).

        Two modes, selected by ``teacher_frame_budget.frame_mode``:

        * ``"resample"`` (default / unset — AFD v2): inject frame-budget hints
          (fps / max_frames / nframes) into each <video> element so qwen_vl_utils
          RE-SAMPLES more frames for the teacher via its internal linspace. The teacher
          frame set is a DISJOINT resampling — it does NOT contain the student's frames
          (verified: linspace(a,b,2m) shares only endpoints with linspace(a,b,m)). This
          is the behavior that empirically failed (unrealizable target; see
          DESIGN_afd_v3_frame_superset.md).

        * ``"superset"`` (AFD v3): read the frame indices the STUDENT actually sampled
          (from ``output.multi_modal_data["videos"][i][1]["frames_indices"]``), build a
          denser index set T ⊇ S by inserting intermediate frames, decode the teacher at
          exactly those explicit indices (bypassing qwen linspace), and resize to the
          SAME (H,W) the student used. The teacher's frames are then a pure visual
          increment over the student's — recoverable privilege.

        NOTE: This runs inside AgentLoopWorker (not AgentLoopBase), so it calls the
        module-level helpers (apply_chat_template / build_multimodal_processor_inputs)
        and the dataset classmethod directly rather than AgentLoopBase instance methods.
        """
        import copy

        raw_prompt = sample_kwargs.get("raw_prompt")
        if raw_prompt is None:
            raise ValueError("raw_prompt missing from sample_kwargs; cannot build teacher dense inputs")
        if self.processor is None:
            raise ValueError("AFD requires a multimodal processor")

        tfb = self.teacher_frame_budget
        frame_mode = tfb.get("frame_mode", "resample") if tfb is not None else "resample"

        if frame_mode == "superset":
            return await self._build_teacher_superset_inputs(raw_prompt, output, sample_kwargs=sample_kwargs)

        messages = copy.deepcopy(list(raw_prompt))
        # Inject frame-budget hints into every video element of the messages.
        for msg in messages:
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            for item in content:
                if isinstance(item, dict) and item.get("type") == "video":
                    if tfb.get("fps", None) is not None:
                        item["fps"] = float(tfb.get("fps"))
                    if tfb.get("max_frames", None) is not None:
                        item["max_frames"] = int(tfb.get("max_frames"))
                    if tfb.get("min_frames", None) is not None:
                        item["min_frames"] = int(tfb.get("min_frames"))
                    if tfb.get("nframes", None) is not None:
                        item["nframes"] = int(tfb.get("nframes"))

        # 1. decode dense frames via the dataset classmethod (same path the student uses)
        image_patch_size = getattr(getattr(self.processor, "image_processor", None), "patch_size", 14)
        images, videos, audios = await self.dataset_cls.process_multi_modal_info(
            messages, image_patch_size=image_patch_size, config=self.config.data
        )
        teacher_mm_data: dict[str, Any] = {}
        if images is not None:
            teacher_mm_data["images"] = images
        if videos is not None:
            teacher_mm_data["videos"] = videos
        if audios is not None:
            teacher_mm_data["audios"] = audios
        teacher_mm_kwargs = self._get_mm_processor_kwargs(audios)

        # 2. apply chat template + tokenize with the dense frames embedded
        apply_kwargs = self.config.data.get("apply_chat_template_kwargs", {})
        raw_text = apply_chat_template(
            self.processor,
            messages,
            tools=None,
            add_generation_prompt=True,
            tokenize=False,
            **apply_kwargs,
        )
        model_inputs = build_multimodal_processor_inputs(
            self.processor,
            text=[raw_text],
            images=images,
            videos=videos,
            audio=audios,
            mm_processor_kwargs=teacher_mm_kwargs,
        )
        teacher_prompt_ids = normalize_token_ids(model_inputs.pop("input_ids"))
        return teacher_prompt_ids, teacher_mm_data, teacher_mm_kwargs

    async def _build_teacher_prompt_inputs(
        self,
        teacher_raw_prompt,
        output: Optional[AgentLoopOutput],
        enable_thinking_override: Optional[bool] = None,
    ) -> tuple[list[int], Optional[dict[str, Any]], Optional[dict[str, Any]]]:
        """[Visual-Cue / Teacher-Think] Build teacher inputs from a DIFFERENT text prompt than the
        student, REUSING the student's already-decoded video frames.

        Two callers:
        - ViCue: teacher_raw_prompt = student prompt + appended cue hint (text privilege).
        - Teacher-Think: teacher_raw_prompt = student's raw_prompt, but re-tokenized with
          `enable_thinking_override` (e.g. thinking-ON) different from the student's rollout template.

        In both cases only the TEXT prompt differs; the video pixels are identical (we re-decode the
        same frames via the dataset classmethod, using the same messages' <video> elements without any
        frame-budget override). The distillation loss uses only the response tokens, so any teacher
        prompt-length difference is handled by response-only extraction + pad_to_sequence_ids
        (same mechanism as AFD).
        """
        if teacher_raw_prompt is None:
            raise ValueError("teacher_raw_prompt missing; cannot build teacher-prompt inputs")
        if self.processor is None:
            raise ValueError("Teacher-prompt distillation requires a multimodal processor")

        messages = copy.deepcopy(list(teacher_raw_prompt))
        # Decode video/image exactly as the student would (no frame-budget override) —
        # only the TEXT template differs, the frames stay identical to the student's.
        image_patch_size = getattr(getattr(self.processor, "image_processor", None), "patch_size", 14)
        images, videos, audios = await self.dataset_cls.process_multi_modal_info(
            messages, image_patch_size=image_patch_size, config=self.config.data
        )
        teacher_mm_data: dict[str, Any] = {}
        if images is not None:
            teacher_mm_data["images"] = images
        if videos is not None:
            teacher_mm_data["videos"] = videos
        if audios is not None:
            teacher_mm_data["audios"] = audios
        teacher_mm_kwargs = self._get_mm_processor_kwargs(audios)

        apply_kwargs = dict(self.config.data.get("apply_chat_template_kwargs", {}))
        # [Teacher-Think] Override the thinking-mode flag for the teacher's chat template only.
        if enable_thinking_override is not None:
            apply_kwargs["enable_thinking"] = enable_thinking_override
        raw_text = apply_chat_template(
            self.processor,
            messages,
            tools=None,
            add_generation_prompt=True,
            tokenize=False,
            **apply_kwargs,
        )
        model_inputs = build_multimodal_processor_inputs(
            self.processor,
            text=[raw_text],
            images=images,
            videos=videos,
            audio=audios,
            mm_processor_kwargs=teacher_mm_kwargs,
        )
        teacher_prompt_ids = normalize_token_ids(model_inputs.pop("input_ids"))
        return teacher_prompt_ids, teacher_mm_data, teacher_mm_kwargs

    async def _build_teacher_superset_inputs(
        self,
        raw_prompt,
        output: Optional[AgentLoopOutput],
        sample_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[list[int], Optional[dict[str, Any]], Optional[dict[str, Any]]]:
        """[AFD v3 / v3.1] Build teacher inputs whose frame set is a SUPERSET of the student's.

        For each <video>, read the student's actual sampled frame indices S (from the
        decoded video metadata attached by qwen_vl_utils), construct T ⊇ S by inserting
        ``insert_per_gap`` intermediate frames per gap, decode the teacher at exactly
        those explicit indices, and resize each frame to the SAME (H, W) the student
        used — so the frames shared with the student are pixel-identical and the teacher
        only gains extra in-between frames (recoverable privilege).

        [v3.1 span-directed] If this sample carries an ``answer_span`` (relative [s, e] in
        [0,1], from ``sample_kwargs``), the extra frames are inserted ONLY inside that
        evidence window instead of uniformly across the clip — concentrating the teacher's
        extra visual budget where the question's answer actually is. When ``answer_span`` is
        absent/None (OFF / non-grounded samples), it degenerates to the uniform superset.
        """
        import copy

        import decord

        from verl.utils.dataset.frame_superset import build_span_directed_indices, build_superset_indices

        tfb = self.teacher_frame_budget
        insert_per_gap = int(tfb.get("insert_per_gap", 1))
        max_frames = tfb.get("max_frames", None)
        max_frames = int(max_frames) if max_frames is not None else None

        # [v3.1] per-sample evidence span (relative [s,e] or None). None → uniform superset.
        answer_span = sample_kwargs.get("answer_span") if sample_kwargs is not None else None

        # Student decoded videos: list of (tensor[T,C,H,W], metadata{frames_indices,...}).
        student_videos = None
        if output is not None and output.multi_modal_data is not None:
            student_videos = output.multi_modal_data.get("videos")
        if not student_videos:
            raise ValueError("[AFD-superset] student decoded videos unavailable; cannot build superset")

        messages = copy.deepcopy(list(raw_prompt))

        # Collect video items from messages in order (they line up with student_videos).
        video_items = [
            item
            for msg in messages
            if isinstance(msg.get("content"), list)
            for item in msg["content"]
            if isinstance(item, dict) and item.get("type") == "video"
        ]
        if len(video_items) != len(student_videos):
            raise ValueError(
                f"[AFD-superset] video item count ({len(video_items)}) != student decoded "
                f"videos ({len(student_videos)}); cannot align"
            )

        for item, stu_video in zip(video_items, student_videos):
            # stu_video is (tensor, metadata) when return_video_metadata=True.
            if not (isinstance(stu_video, (tuple, list)) and len(stu_video) == 2):
                raise ValueError("[AFD-superset] student video missing metadata; need return_video_metadata=True")
            stu_tensor, stu_meta = stu_video
            student_indices = stu_meta.get("frames_indices")
            if student_indices is None:
                raise ValueError("[AFD-superset] student metadata has no frames_indices")
            # Match the student's resized frame size so shared frames are pixel-identical.
            # stu_tensor is [T, C, H, W].
            resized_h, resized_w = int(stu_tensor.shape[-2]), int(stu_tensor.shape[-1])

            T = build_superset_indices(student_indices, insert_per_gap=insert_per_gap, max_frames=max_frames)

            # Decode the teacher's frames at exactly the superset indices.
            video_path = item.get("video")
            vr = decord.VideoReader(video_path)
            n_total = len(vr)
            # [v3.1] span-directed: densify only inside the evidence window (needs n_total to
            # map relative span → absolute frames). answer_span=None → uniform (== above).
            if answer_span is not None:
                T = build_span_directed_indices(
                    student_indices,
                    n_total=n_total,
                    span=answer_span,
                    insert_per_gap=insert_per_gap,
                    max_frames=max_frames,
                )
            T = [min(max(0, idx), n_total - 1) for idx in T]
            frames = vr.get_batch(T).asnumpy()  # [len(T), H0, W0, C]
            pil_frames = [Image.fromarray(frames[i]) for i in range(frames.shape[0])]

            # Replace the video element with a pre-loaded frame list + fixed resize, so
            # qwen_vl_utils skips its own linspace sampling and uses exactly these frames.
            item["video"] = pil_frames
            item["resized_height"] = resized_h
            item["resized_width"] = resized_w
            # Drop any sampling hints that would conflict with an explicit frame list.
            for k in ("fps", "nframes", "max_frames", "min_frames", "video_start", "video_end"):
                item.pop(k, None)

        # Decode + tokenize the teacher prompt with the explicit superset frames.
        image_patch_size = getattr(getattr(self.processor, "image_processor", None), "patch_size", 14)
        images, videos, audios = await self.dataset_cls.process_multi_modal_info(
            messages, image_patch_size=image_patch_size, config=self.config.data
        )
        teacher_mm_data: dict[str, Any] = {}
        if images is not None:
            teacher_mm_data["images"] = images
        if videos is not None:
            teacher_mm_data["videos"] = videos
        if audios is not None:
            teacher_mm_data["audios"] = audios
        teacher_mm_kwargs = self._get_mm_processor_kwargs(audios)

        apply_kwargs = self.config.data.get("apply_chat_template_kwargs", {})
        raw_text = apply_chat_template(
            self.processor,
            messages,
            tools=None,
            add_generation_prompt=True,
            tokenize=False,
            **apply_kwargs,
        )
        model_inputs = build_multimodal_processor_inputs(
            self.processor,
            text=[raw_text],
            images=images,
            videos=videos,
            audio=audios,
            mm_processor_kwargs=teacher_mm_kwargs,
        )
        teacher_prompt_ids = normalize_token_ids(model_inputs.pop("input_ids"))
        return teacher_prompt_ids, teacher_mm_data, teacher_mm_kwargs



    def _postprocess(
        self,
        inputs: list[_InternalAgentLoopOutput],
        input_non_tensor_batch: dict | None = None,
        validate: bool = False,
    ) -> DataProto:
        """Process the padded outputs from _run_agent_loop and combine them into a batch."""
        # Convert lists back to tensors and stack them to create a batch.
        prompt_ids = torch.cat([input.prompt_ids for input in inputs], dim=0)
        response_ids = torch.cat([input.response_ids for input in inputs], dim=0)
        response_mask = torch.cat([input.response_mask for input in inputs], dim=0)
        attention_mask = torch.cat([input.attention_mask for input in inputs], dim=0)
        input_ids = torch.cat([input.input_ids for input in inputs], dim=0)
        position_ids = torch.cat([input.position_ids for input in inputs], dim=0)
        optional_outputs = {}
        if inputs[0].response_logprobs is not None:
            optional_outputs["rollout_log_probs"] = torch.cat([input.response_logprobs for input in inputs], dim=0)
        if inputs[0].routed_experts is not None:
            optional_outputs["routed_experts"] = torch.cat([input.routed_experts for input in inputs], dim=0)
        if inputs[0].teacher_logprobs is not None and inputs[0].teacher_ids is not None:
            optional_outputs["teacher_logprobs"] = torch.cat([input.teacher_logprobs for input in inputs], dim=0)
            optional_outputs["teacher_ids"] = torch.cat([input.teacher_ids for input in inputs], dim=0)
        if inputs[0].base_logprobs is not None:
            optional_outputs["base_logprobs"] = torch.cat([input.base_logprobs for input in inputs], dim=0)
        if inputs[0].base_dense_logprobs is not None:
            optional_outputs["base_dense_logprobs"] = torch.cat(
                [input.base_dense_logprobs for input in inputs], dim=0
            )
        # [V-Zero-Video] Negative-view teacher logprobs. Per-sample failures fall back to the
        # positive view (teacher_logprobs) so Δ=0 → neutral gate for that sample. Emitted whenever
        # ANY sample in the batch has it (contrastive gating enabled).
        if any(input.neg_view_logprobs is not None for input in inputs):
            optional_outputs["neg_view_logprobs"] = torch.cat(
                [
                    input.neg_view_logprobs if input.neg_view_logprobs is not None else input.teacher_logprobs
                    for input in inputs
                ],
                dim=0,
            )
        # [Cue-Gate] Samples without a no-cue pass (fallback-OPD rows, or a failed pass) fall back to
        # teacher_logprobs → Δ=0 → neutral gate (weight 1) for that sample.
        if any(input.nocue_logprobs is not None for input in inputs):
            optional_outputs["nocue_logprobs"] = torch.cat(
                [
                    input.nocue_logprobs if input.nocue_logprobs is not None else input.teacher_logprobs
                    for input in inputs
                ],
                dim=0,
            )
        batch = TensorDict(
            {
                "prompts": prompt_ids,  # [bsz, prompt_length]
                "responses": response_ids,  # [bsz, response_length]
                "response_mask": response_mask,  # [bsz, response_length]
                "input_ids": input_ids,  # [bsz, prompt_length + response_length]
                "attention_mask": attention_mask,  # [bsz, prompt_length + response_length]
                # position_ids: [bsz, 3, prompt_length + response_length] or [bsz, prompt_length + response_length]
                "position_ids": position_ids,
                **optional_outputs,
            },
            batch_size=len(inputs),
        )

        scores = [input.reward_score for input in inputs]
        if all(score is not None for score in scores):
            prompt_length = prompt_ids.size(1)
            response_length = attention_mask[:, prompt_length:].sum(dim=1) - 1
            rm_scores = torch.zeros_like(response_mask, dtype=torch.float32)
            rm_scores[torch.arange(response_mask.size(0)), response_length] = torch.tensor(scores, dtype=torch.float32)
            batch["rm_scores"] = rm_scores

        non_tensor_batch = {
            "__num_turns__": np.array([input.num_turns for input in inputs], dtype=np.int32),
        }
        if self.reward_loop_worker_handles is None and input_non_tensor_batch:
            non_tensor_batch.update(input_non_tensor_batch)

        # add reward_extra_info to non_tensor_batch
        reward_extra_infos = [input.extra_fields.get("reward_extra_info", {}) for input in inputs]
        reward_extra_keys = list(reward_extra_infos[0].keys())
        for key in reward_extra_keys:
            non_tensor_batch[key] = np.array([info[key] for info in reward_extra_infos])

        # Add multi_modal_inputs to non_tensor_batch if any samples have them
        multi_modal_inputs_list = [input.multi_modal_inputs for input in inputs]
        if any(mmi is not None for mmi in multi_modal_inputs_list):
            non_tensor_batch["multi_modal_inputs"] = np.array(multi_modal_inputs_list, dtype=object)

        metrics = [input.metrics.model_dump() for input in inputs]
        # Collect extra fields from all inputs and convert them to np.ndarray
        # Keep a stable set of keys so downstream batch concat stays consistent across agent loops.
        extra_fields = {}
        default_extra_keys = {
            "turn_scores",
            "tool_rewards",
            "min_global_steps",
            "max_global_steps",
            "extras",
        }
        all_keys = set(key for input_item in inputs for key in input_item.extra_fields) | default_extra_keys
        for key in all_keys:
            temp_arr = np.empty(len(inputs), dtype=object)
            temp_arr[:] = [input.extra_fields.get(key) for input in inputs]
            extra_fields[key] = temp_arr

        non_tensor_batch.update(extra_fields)

        # Only include reward_extra_keys in meta_info if rm_scores is in batch
        # This avoids conflicts when reward_tensor is merged later in ray_trainer.py
        if "rm_scores" in batch.keys():
            meta_info = {"metrics": metrics, "reward_extra_keys": reward_extra_keys}
        else:
            meta_info = {"metrics": metrics}

        return DataProto(
            batch=batch,
            non_tensor_batch=non_tensor_batch,
            meta_info=meta_info,
        )


async def get_trajectory_info(step, index, validate):
    """Get trajectory info.

    Args:
        step (int): global steps in the trainer.
        index (list): form datastore extra_info.index column.
        validate (bool): whether is a validate step.

    Returns:
        list: trajectory.
    """
    trajectory_info = []
    rollout_n = 0
    for i in range(len(index)):
        if i > 0 and index[i - 1] == index[i]:
            rollout_n += 1
        else:
            rollout_n = 0
        trajectory_info.append({"step": step, "sample_index": index[i], "rollout_n": rollout_n, "validate": validate})
    return trajectory_info


class AgentLoopManager:
    """Agent loop manager that manages a group of agent loop workers.

    Args:
        config (DictConfig): whole config for main entrypoint.
        llm_client (LLMServerClient): Client for the LLM server.
        teacher_client (dict[str, LLMServerClient]): Client for multiple teacher servers.
        reward_loop_worker_handles (List[ray.actor.ActorHandle]): Actor handles for streaming reward computation.
    """

    def __init__(
        self,
        config: DictConfig,
        llm_client: LLMServerClient,
        teacher_client: dict[str, LLMServerClient] = None,
        reward_loop_worker_handles: list[ray.actor.ActorHandle] = None,
    ):
        self.config = config
        self.rollout_config = config.actor_rollout_ref.rollout
        self.model_config = config.actor_rollout_ref.model
        self.llm_client = llm_client
        self.teacher_client = teacher_client
        self.reward_loop_worker_handles = reward_loop_worker_handles

        if not hasattr(self, "agent_loop_workers_class"):
            self.agent_loop_workers_class = ray.remote(AgentLoopWorker)

    @classmethod
    @auto_await
    async def create(cls, *args, **kwargs):
        """Create agent loop manager."""
        instance = cls(*args, **kwargs)
        await instance._init_agent_loop_workers()
        return instance

    async def _init_agent_loop_workers(self):
        self.agent_loop_workers = []
        num_workers = self.rollout_config.agent.num_workers

        node_ids = [node["NodeID"] for node in ray.nodes() if node["Alive"] and node["Resources"].get("CPU", 0) > 0]
        for i in range(num_workers):
            # Round-robin scheduling over the all nodes
            node_id = node_ids[i % len(node_ids)]
            self.agent_loop_workers.append(
                self.agent_loop_workers_class.options(
                    name=f"agent_loop_worker_{i}" + f"_{uuid4().hex[:8]}",
                    scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                        node_id=node_id, soft=True
                    ),
                ).remote(
                    self.config,
                    self.llm_client,
                    self.teacher_client,
                    self.reward_loop_worker_handles,
                )
            )

    @auto_await
    async def generate_sequences(self, prompts: DataProto) -> DataProto:
        """Split input batch and dispatch to agent loop workers.

        Args:
            prompts (DataProto): Input batch.

        Returns:
            DataProto: Output batch.
        """
        chunkes = prompts.chunk(len(self.agent_loop_workers))
        outputs = await asyncio.gather(
            *[
                worker.generate_sequences.remote(chunk)
                for worker, chunk in zip(self.agent_loop_workers, chunkes, strict=True)
            ]
        )
        output = DataProto.concat(outputs)

        # calculate performance metrics
        metrics = [output.meta_info.pop("metrics") for output in outputs]  # List[List[Dict[str, str]]]
        timing = self._performance_metrics(metrics, output)

        output.meta_info = {"timing": timing, **outputs[0].meta_info}
        return output

    def _performance_metrics(self, metrics: list[list[dict[str, str]]], output: DataProto) -> dict[str, float]:
        timing = {}
        t_generate_sequences = np.array([metric["generate_sequences"] for chunk in metrics for metric in chunk])
        t_tool_calls = np.array([metric["tool_calls"] for chunk in metrics for metric in chunk])
        t_compute_score = np.array([metric["compute_score"] for chunk in metrics for metric in chunk])
        num_preempted = np.array([metric["num_preempted"] for chunk in metrics for metric in chunk])
        timing["agent_loop/num_preempted/min"] = num_preempted.min()
        timing["agent_loop/num_preempted/max"] = num_preempted.max()
        timing["agent_loop/num_preempted/mean"] = num_preempted.mean()
        timing["agent_loop/generate_sequences/min"] = t_generate_sequences.min()
        timing["agent_loop/generate_sequences/max"] = t_generate_sequences.max()
        timing["agent_loop/generate_sequences/mean"] = t_generate_sequences.mean()
        timing["agent_loop/tool_calls/min"] = t_tool_calls.min()
        timing["agent_loop/tool_calls/max"] = t_tool_calls.max()
        timing["agent_loop/tool_calls/mean"] = t_tool_calls.mean()
        timing["agent_loop/compute_score/min"] = t_compute_score.min()
        timing["agent_loop/compute_score/max"] = t_compute_score.max()
        timing["agent_loop/compute_score/mean"] = t_compute_score.mean()

        # batch sequence generation is bounded by the slowest sample
        slowest = np.argmax(t_generate_sequences + t_tool_calls + t_compute_score)
        prompt_length = output.batch["prompts"].shape[1]
        timing["agent_loop/slowest/generate_sequences"] = t_generate_sequences[slowest]
        timing["agent_loop/slowest/tool_calls"] = t_tool_calls[slowest]
        timing["agent_loop/slowest/compute_score"] = t_compute_score[slowest]
        timing["agent_loop/slowest/num_preempted"] = num_preempted[slowest]

        if "attention_mask" in output.batch:
            attention_mask = output.batch["attention_mask"][slowest]
            timing["agent_loop/slowest/prompt_length"] = attention_mask[:prompt_length].sum().item()
            timing["agent_loop/slowest/response_length"] = attention_mask[prompt_length:].sum().item()

        return timing
