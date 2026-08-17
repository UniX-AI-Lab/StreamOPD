"""
Qwen3.5 backend for the recent-window streaming protocol.

Qwen3.5 specifics handled here (they differ from the Qwen3-VL family):
1. mm_token_type_ids drives the multimodal RoPE position encoding
2. video frames use video_token_id (248057), distinct from image_token_id (248056)
3. get_rope_index requires the mm_token_type_ids argument
4. video frames are separated by timestamps in the token sequence
5. hybrid architecture: Gated DeltaNet (linear attention) + full attention layers,
   which needs the flash-linear-attention and causal-conv1d packages
"""

from __future__ import annotations

import copy
import os
import time
from collections import deque
from dataclasses import dataclass

import torch
from PIL import Image

from streamopd.streaming.recent_window import (
    RecentWindowQAModel as _BaseRecentWindowQAModel,
    RecentWindowResult,
    build_ovo_prompt,
    decode_video_to_chunks_qwen,
    flatten_gathered_results,
    print_ovo_results,
)

import re

def _strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks from Qwen3.5 output."""
    # Remove thinking blocks (possibly multiline)
    text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)
    return text.strip()


class RecentWindowQAModel(_BaseRecentWindowQAModel):
    """Qwen3.5-VL wrapper with mm_token_type_ids and video_token_id support."""

    def __init__(
        self,
        model_name: str,
        device: str | torch.device = "auto",
        max_new_tokens: int = 256,
        attn_implementation: str = "sdpa",
        enable_thinking: bool = False,
    ) -> None:
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.model_name = model_name
        self.device = device
        self.max_new_tokens = int(max_new_tokens)
        self.enable_thinking = bool(enable_thinking)
        self._last_ttft_seconds = 0.0
        self._last_num_vision_tokens = 0
        self._last_num_vision_frames = 0

        proc_kwargs: dict[str, object] = {}
        if os.environ.get("MIN_PIXELS"):
            proc_kwargs["min_pixels"] = int(os.environ["MIN_PIXELS"])
        if os.environ.get("MAX_PIXELS"):
            proc_kwargs["max_pixels"] = int(os.environ["MAX_PIXELS"])
        self.processor = AutoProcessor.from_pretrained(model_name, **proc_kwargs)

        model_kwargs: dict[str, object] = {
            "torch_dtype": torch.bfloat16,
            "attn_implementation": attn_implementation,
        }
        if device == "auto":
            model_kwargs["device_map"] = "auto"

        saved_world_size = os.environ.pop("WORLD_SIZE", None)
        try:
            self.model = AutoModelForImageTextToText.from_pretrained(model_name, **model_kwargs)
        finally:
            if saved_world_size is not None:
                os.environ["WORLD_SIZE"] = saved_world_size
        if device != "auto":
            self.model.to(device)
        self.model.eval()

        # Model architecture references
        self._hf_model = self.model
        self._visual = self.model.model.visual
        self._text_model = self.model.model

        # Token IDs
        self.image_token_id = self.model.config.image_token_id      # 248056
        self.video_token_id = self.model.config.video_token_id      # 248057
        self.vision_start_id = self.model.config.vision_start_token_id  # 248053
        self.vision_end_id = self.model.config.vision_end_token_id      # 248054
        self.im_start_id = self.processor.tokenizer.convert_tokens_to_ids("<|im_start|>")
        self.im_end_id = self.processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
        self.merge_size = self.model.model.visual.spatial_merge_size

    def _extract_vision_embeds(self, features) -> torch.Tensor:
        """Extract vision embeddings from model output.

        In transformers 5.x, get_image_features returns BaseModelOutputWithPooling.
        The pooler_output contains a list of tensors (one per image).
        """
        if isinstance(features, torch.Tensor):
            return features
        # BaseModelOutputWithPooling - extract pooler_output
        if hasattr(features, "pooler_output"):
            pooler = features.pooler_output
            if isinstance(pooler, (list, tuple)):
                return torch.cat(list(pooler), dim=0)
            if isinstance(pooler, torch.Tensor):
                return pooler
        # Try last_hidden_state
        if hasattr(features, "last_hidden_state"):
            return features.last_hidden_state
        # Fallback to generic flattening
        return self._flatten_vision_features(features)

    @torch.inference_mode()
    def encode_vision(self, frames: list[Image.Image]) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode frames using the processor and vision encoder.

        For Qwen3.5, we pass frames as images to the processor and get vision embeddings.
        """
        content = [{"type": "image", "image": frame} for frame in frames]
        content.append({"type": "text", "text": "."})
        messages = [{"role": "user", "content": content}]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_tensors="pt",
        )

        pixel_values = inputs["pixel_values"].to(self.model.device)
        image_grid_thw = inputs["image_grid_thw"].to(self.model.device)
        image_embeds = self._extract_vision_embeds(
            self._get_image_feature_model().get_image_features(pixel_values, image_grid_thw)
        )

        del pixel_values
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return image_embeds, image_grid_thw

    @torch.inference_mode()
    def encode_vision_batched(
        self,
        frames_per_chunk: list[list[Image.Image]],
        max_frames_per_batch: int = 8,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Batch encode vision frames, returning embeddings per chunk."""
        if not frames_per_chunk:
            return []

        flat_pairs: list[tuple[int, Image.Image]] = []
        for chunk_index, frames in enumerate(frames_per_chunk):
            for frame in frames:
                flat_pairs.append((chunk_index, frame))

        hidden_size = int(getattr(self.model.config, "hidden_size",
                          getattr(self.model.config.vision_config, "out_hidden_size", 4096)))
        model_dtype = getattr(self.model, "dtype", torch.bfloat16)
        empty_emb = torch.empty((0, hidden_size), dtype=model_dtype, device="cpu")
        empty_grid = torch.empty((0, 3), dtype=torch.long, device="cpu")
        if not flat_pairs:
            return [(empty_emb, empty_grid) for _ in frames_per_chunk]

        merge_area = max(1, int(self.merge_size)) ** 2
        chunk_embeds: list[list[torch.Tensor]] = [[] for _ in frames_per_chunk]
        chunk_grids: list[list[torch.Tensor]] = [[] for _ in frames_per_chunk]

        batch_size = max(1, int(max_frames_per_batch))
        offset_flat = 0
        while offset_flat < len(flat_pairs):
            pairs = flat_pairs[offset_flat : offset_flat + batch_size]
            content = [{"type": "image", "image": frame} for _, frame in pairs]
            content.append({"type": "text", "text": "."})
            messages = [{"role": "user", "content": content}]

            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False,
                return_dict=True,
                return_tensors="pt",
            )
            pixel_values = inputs["pixel_values"].to(self.model.device)
            image_grid_thw = inputs["image_grid_thw"].to(self.model.device)
            image_embeds = self._extract_vision_embeds(
                self._get_image_feature_model().get_image_features(pixel_values, image_grid_thw)
            )

            frame_token_counts = [
                max(1, int(row[0].item() * row[1].item() * row[2].item()) // merge_area)
                for row in image_grid_thw
            ]
            expected_tokens = sum(frame_token_counts)
            if expected_tokens != int(image_embeds.shape[0]) or len(frame_token_counts) != len(pairs):
                # Fallback: encode individually
                grouped: dict[int, list[Image.Image]] = {}
                for chunk_index, frame in pairs:
                    grouped.setdefault(chunk_index, []).append(frame)
                for chunk_index, frames in grouped.items():
                    emb, grid = self.encode_vision(frames)
                    chunk_embeds[chunk_index].append(emb.to(dtype=torch.bfloat16, device="cpu"))
                    chunk_grids[chunk_index].append(grid.cpu())
                offset_flat += len(pairs)
                del pixel_values, image_grid_thw, image_embeds
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue

            offset = 0
            for (chunk_index, _), token_count, row in zip(pairs, frame_token_counts, image_grid_thw):
                end = offset + token_count
                chunk_embeds[chunk_index].append(image_embeds[offset:end].to(dtype=torch.bfloat16, device="cpu"))
                chunk_grids[chunk_index].append(row.unsqueeze(0).cpu())
                offset = end
            offset_flat += len(pairs)

            del pixel_values, image_grid_thw, image_embeds
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        outputs: list[tuple[torch.Tensor, torch.Tensor]] = []
        for chunk_index in range(len(frames_per_chunk)):
            if chunk_embeds[chunk_index]:
                outputs.append(
                    (
                        torch.cat(chunk_embeds[chunk_index], dim=0),
                        torch.cat(chunk_grids[chunk_index], dim=0),
                    )
                )
            else:
                outputs.append((empty_emb, empty_grid))
        return outputs

    @torch.inference_mode()
    def generate_with_vision_features(
        self,
        vision_embeds: torch.Tensor,
        vision_grid_thw: torch.Tensor,
        question: str,
    ) -> str:
        """Generate answer from pre-encoded vision features.

        Qwen3.5 key differences:
        - Uses mm_token_type_ids to distinguish text (0), image (1), and video (2) tokens
        - get_rope_index requires mm_token_type_ids
        - Each frame gets its own <vision_start>...<vision_end> block
        - We use image_token_id (treating frames as images for simplicity)
        - Output may contain <think>...</think> tags which we strip
        """
        device = self.model.device
        tokenizer = self.processor.tokenizer
        text_model = self._get_text_model()

        num_vision_tokens = int(vision_embeds.shape[0])
        self._last_num_vision_tokens = num_vision_tokens
        self._last_num_vision_frames = int(vision_grid_thw.shape[0]) if vision_grid_thw is not None else 0

        question_ids = tokenizer.encode(question, add_special_tokens=False)
        grid_rows = vision_grid_thw.to(device)
        tokens_per_frame = (grid_rows.prod(dim=-1) // (self.merge_size**2)).tolist()
        expected_tokens = sum(int(n) for n in tokens_per_frame)
        if expected_tokens != num_vision_tokens:
            raise ValueError(
                "vision token count mismatch: "
                f"embeds={num_vision_tokens} vs grid={expected_tokens}"
            )

        # Build input_ids with per-frame vision blocks
        # Format: <|im_start|>user\n<|vision_start|>[image_pad]*N<|vision_end|>...\n{question}<|im_end|>\n<|im_start|>assistant\n/no_think\n
        input_ids_list: list[int] = []
        mm_token_type_list: list[int] = []  # 0=text, 1=image, 2=video

        # <|im_start|>user\n
        prefix_ids = [self.im_start_id] + tokenizer.encode("user\n", add_special_tokens=False)
        input_ids_list.extend(prefix_ids)
        mm_token_type_list.extend([0] * len(prefix_ids))

        # Per-frame vision blocks (using image tokens)
        for frame_token_count in tokens_per_frame:
            # <|vision_start|>
            input_ids_list.append(self.vision_start_id)
            mm_token_type_list.append(0)  # vision_start is text type
            # [image_pad] * token_count
            input_ids_list.extend([self.image_token_id] * int(frame_token_count))
            mm_token_type_list.extend([1] * int(frame_token_count))  # image type
            # <|vision_end|>
            input_ids_list.append(self.vision_end_id)
            mm_token_type_list.append(0)  # vision_end is text type

        # \n{question}<|im_end|>\n<|im_start|>assistant\n
        # Then explicitly toggle thinking by appending either:
        #   <think>\n            (enable_thinking=True  — model continues reasoning, then </think>, then answer)
        #   <think>\n\n</think>\n\n (enable_thinking=False — model sees thinking already closed, gives direct answer)
        # This matches Qwen3.5's official chat_template (tokenizer_config.json L149-152).
        suffix_ids = tokenizer.encode("\n", add_special_tokens=False)
        suffix_ids.extend(question_ids)
        suffix_ids.append(self.im_end_id)
        suffix_ids.extend(tokenizer.encode("\n", add_special_tokens=False))
        suffix_ids.append(self.im_start_id)
        if self.enable_thinking:
            suffix_ids.extend(tokenizer.encode("assistant\n<think>\n", add_special_tokens=False))
        else:
            suffix_ids.extend(tokenizer.encode("assistant\n<think>\n\n</think>\n\n", add_special_tokens=False))
        input_ids_list.extend(suffix_ids)
        mm_token_type_list.extend([0] * len(suffix_ids))

        input_ids = torch.tensor([input_ids_list], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids)
        mm_token_type_ids = torch.tensor([mm_token_type_list], dtype=torch.int, device=device)

        # Get input embeddings and substitute vision tokens
        inputs_embeds = text_model.get_input_embeddings()(input_ids)
        vision_embeds = vision_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        image_mask = input_ids == self.image_token_id
        image_mask_expanded = image_mask.unsqueeze(-1).expand_as(inputs_embeds)
        inputs_embeds = inputs_embeds.masked_scatter(image_mask_expanded, vision_embeds)

        # Compute 3D RoPE position ids with mm_token_type_ids
        position_ids, _ = text_model.get_rope_index(
            input_ids=input_ids,
            mm_token_type_ids=mm_token_type_ids,
            image_grid_thw=grid_rows,
            video_grid_thw=None,
            attention_mask=attention_mask,
        )

        answer = self._generate_from_model_inputs(
            prompt_length=len(input_ids[0]),
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )

        # Strip thinking tags if present
        answer = _strip_thinking(answer)
        return answer

    @torch.inference_mode()
    def generate_from_frames(self, frames: list[Image.Image], question: str) -> str:
        """Convenience: encode + generate in one call."""
        vision_embeds, vision_grid_thw = self.encode_vision(frames)
        return self.generate_with_vision_features(vision_embeds, vision_grid_thw, question)


@dataclass
class EncodedChunk:
    vision_emb: torch.Tensor
    grid_thw: torch.Tensor
    chunk_index: int
    start_time: float
    end_time: float


def _combine_window_embeddings(
    window: deque[EncodedChunk],
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor]:
    combined_embeds = torch.cat([item.vision_emb.to(device) for item in window], dim=0)
    combined_grid_thw = torch.cat([item.grid_thw.to(device) for item in window], dim=0)
    return combined_embeds, combined_grid_thw


def query_recent_window(
    qa: RecentWindowQAModel,
    video_path: str,
    prompt: str,
    chunk_duration: float,
    fps: float,
    recent_frames_only: int,
    video_start: float | None = None,
    video_end: float | None = None,
    nframes: int | None = None,
) -> tuple[RecentWindowResult, str]:
    chunks, decode_backend = decode_video_to_chunks_qwen(
        video_path=video_path,
        chunk_duration=chunk_duration,
        fps=fps,
        recent_frames_only=recent_frames_only,
        video_start=video_start,
        video_end=video_end,
        nframes=nframes,
    )
    if not chunks:
        raise ValueError(f"No chunks decoded from video: {video_path}")

    if nframes is not None:
        window_size = len(chunks)
    else:
        window_size = max(1, int(recent_frames_only))
    recent_chunks = chunks[-window_size:]
    encoded_chunks: list[EncodedChunk] = []
    encoded_outputs = qa.encode_vision_batched([chunk.frames for chunk in recent_chunks], max_frames_per_batch=8)
    for chunk, (vision_emb, grid_thw) in zip(recent_chunks, encoded_outputs):
        if int(vision_emb.shape[0]) == 0 or int(grid_thw.shape[0]) == 0:
            continue
        encoded_chunks.append(
            EncodedChunk(
                vision_emb=vision_emb,
                grid_thw=grid_thw,
                chunk_index=chunk.chunk_index,
                start_time=chunk.start_time,
                end_time=chunk.end_time,
            )
        )
    if not encoded_chunks:
        raise ValueError(f"No vision chunks encoded from video: {video_path}")

    encoded_window: deque[EncodedChunk] = deque(encoded_chunks, maxlen=window_size)
    t0 = time.perf_counter()
    combined_embeds, combined_grid_thw = _combine_window_embeddings(encoded_window, qa.model.device)
    answer = qa.generate_with_vision_features(combined_embeds, combined_grid_thw, prompt)
    generate_time = time.perf_counter() - t0
    ttft_seconds = getattr(qa, "_last_ttft_seconds", 0.0) or 0.0
    num_vision_tokens = qa._last_num_vision_tokens
    num_frames = qa._last_num_vision_frames

    return (
        RecentWindowResult(
            answer=answer,
            final_chunk_ids=[item.chunk_index for item in encoded_window],
            generate_time=generate_time,
            ttft_seconds=ttft_seconds,
            num_vision_tokens=num_vision_tokens,
            num_vision_tokens_before=num_vision_tokens,
            num_vision_tokens_after=num_vision_tokens,
            num_frames=num_frames,
        ),
        decode_backend,
    )


def evaluate_ovo_backward_realtime(
    anno: dict,
    chunked_dir: str,
    qa: RecentWindowQAModel,
    chunk_duration: float,
    fps: float,
    recent_frames_only: int,
    nframes: int | None = None,
) -> dict:
    video_path = os.path.join(chunked_dir, f"{anno['id']}.mp4")
    response = None
    metadata: dict = {}
    if os.path.exists(video_path):
        result, decode_backend = query_recent_window(
            qa=qa,
            video_path=video_path,
            prompt=build_ovo_prompt(anno["task"], anno),
            chunk_duration=chunk_duration,
            fps=fps,
            recent_frames_only=recent_frames_only,
            nframes=nframes,
        )
        response = result.answer
        metadata = {
            "decode_backend": decode_backend,
            "final_chunk_ids": result.final_chunk_ids,
            "generate_time": result.generate_time,
            "ttft_seconds": result.ttft_seconds,
            "num_vision_tokens": result.num_vision_tokens,
            "num_vision_tokens_before": result.num_vision_tokens_before,
            "num_vision_tokens_after": result.num_vision_tokens_after,
            "num_frames": result.num_frames,
        }
    return {
        "id": anno["id"],
        "video": anno["video"],
        "task": anno["task"],
        "question": anno["question"],
        "response": response,
        "ground_truth": chr(65 + anno["gt"]),
        **metadata,
    }


def evaluate_ovo_forward(
    anno: dict,
    chunked_dir: str,
    qa: RecentWindowQAModel,
    chunk_duration: float,
    fps: float,
    recent_frames_only: int,
    nframes: int | None = None,
) -> dict:
    result_anno = copy.deepcopy(anno)
    for index, test_info in enumerate(result_anno["test_info"]):
        video_path = os.path.join(chunked_dir, f"{anno['id']}_{index}.mp4")
        if not os.path.exists(video_path):
            test_info["response"] = None
            continue
        result, decode_backend = query_recent_window(
            qa=qa,
            video_path=video_path,
            prompt=build_ovo_prompt(anno["task"], anno, index=index),
            chunk_duration=chunk_duration,
            fps=fps,
            recent_frames_only=recent_frames_only,
            nframes=nframes,
        )
        test_info["response"] = result.answer
        test_info["decode_backend"] = decode_backend
        test_info["final_chunk_ids"] = result.final_chunk_ids
        test_info["generate_time"] = result.generate_time
        test_info["ttft_seconds"] = result.ttft_seconds
        test_info["num_vision_tokens"] = result.num_vision_tokens
        test_info["num_vision_tokens_before"] = result.num_vision_tokens_before
        test_info["num_vision_tokens_after"] = result.num_vision_tokens_after
        test_info["num_frames"] = result.num_frames
    return result_anno
