# lmms-eval Qwen3.5 adapter

`lmms-eval` 0.7.1 and later ship a `qwen3_5` model natively, so in most cases you only need

```bash
pip install "lmms-eval>=0.7.1"
```

and `scripts/eval/run_all.sh` works as is. **Pin the vision budget either way** — see
[Reproducing the reported numbers](#reproducing-the-reported-numbers) below.

## What is in here

The reported results were produced with a thin-wrapper adapter rather than upstream's
standalone implementation. Both files are kept for exact-parity runs:

| File | Install as |
|------|------------|
| `qwen3_5.py` | `lmms_eval/models/simple/qwen3_5.py` |
| `qwen3_5_chat_model.py` | `lmms_eval/models/chat/qwen3_5.py` |

Installing them takes three steps, not two. The wrapper subclasses `Qwen3_VL`, and that base
class picks its HF model class from the `pretrained` string, so on its own it would load a
Qwen3.5 checkpoint with `Qwen3VLForConditionalGeneration` and the wrong dtype keyword.

1. Copy both files into the paths above.
2. Register them in `lmms_eval/models/__init__.py`, in `AVAILABLE_MODELS` and
   `AVAILABLE_CHAT_TEMPLATE_MODELS`:

   ```python
   "qwen3_5": "Qwen3_5",
   ```

3. Teach `lmms_eval/models/simple/qwen3_vl.py` to resolve the model class from the config
   instead of the name, and call it where the model is built:

   ```python
   def _resolve_model_class(pretrained: str, is_moe: bool):
       """Return (model_class, dtype_kwarg_name) for a Qwen3 variant."""
       config = AutoConfig.from_pretrained(pretrained, trust_remote_code=True)
       if "qwen3_5" in getattr(config, "model_type", ""):
           from transformers import Qwen3_5ForConditionalGeneration, Qwen3_5MoeForConditionalGeneration
           return (Qwen3_5MoeForConditionalGeneration if is_moe else Qwen3_5ForConditionalGeneration), "torch_dtype"
       from transformers import Qwen3VLForConditionalGeneration, Qwen3VLMoeForConditionalGeneration
       return (Qwen3VLMoeForConditionalGeneration if is_moe else Qwen3VLForConditionalGeneration), "dtype"

   # in Qwen3_VL.__init__, replacing the hard-coded model class:
   model_cls, dtype_key = _resolve_model_class(pretrained, is_moe)
   model_kwargs = {dtype_key: "bfloat16", "device_map": self.device_map}
   self._model = model_cls.from_pretrained(pretrained, **model_kwargs).eval()
   ```

   Note the dtype keyword differs between the two families: Qwen3.5 takes `torch_dtype`,
   Qwen3-VL takes `dtype`.

## Reproducing the reported numbers

This matters regardless of which adapter you use. The wrapper overrides the vision budget
it inherits, and the two sets of defaults are far apart:

| Argument | `Qwen3_VL` default | Value used for the reported numbers |
|----------|-------------------:|------------------------------------:|
| `min_pixels` | 256·28·28 = 200,704 | 64·32·32 = 65,536 |
| `max_pixels` | 1,605,632 | 128·32·32 = 131,072 |
| `total_pixels` | unset | 224·1024·32·32 = 234,881,024 |

A 12× difference in `max_pixels` moves Video-MME and LongVideoBench scores, so
`scripts/eval/run_all.sh` passes all three explicitly through `--model_args`. Keep them
pinned if you invoke lmms-eval directly, and adjust them together with any change to
`max_num_frames`, which stays at 32.
