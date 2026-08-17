# Installation

Budget roughly one hour on a fresh machine. Most of it is compiling vLLM (~40 min) and
flash-attn (~9 min) from source, which is unavoidable — see [Why source builds](#why-source-builds).

## Requirements

| | |
|---|---|
| GPUs | 8 × 80GB+, compute capability 9.0 (sm_90) |
| CUDA toolkit | 12.9 |
| Driver | ≥ 535 |
| Python | 3.12 |
| Disk | ≥ 1 TB for models, videos and checkpoints |

Verify the machine before starting:

```bash
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
nvcc --version | grep release
ldd --version | head -1
```

## 1. Create the environment

```bash
ENV=/path/to/env/streamopd
uv venv $ENV --python python3.12
UV_PIP="uv pip install --index-strategy unsafe-best-match --python $ENV/bin/python"
```

## 2. PyTorch

```bash
$UV_PIP "torch==2.11.0+cu129" "torchvision==0.26.0+cu129" "torchaudio==2.11.0+cu129" \
    --extra-index-url https://download.pytorch.org/whl/cu129
```

## 3. The patched verl

Installing the vendored verl in editable mode also pulls in its ~80 transitive
dependencies (transformers, ray, tensordict, wandb, datasets, ...).

```bash
$UV_PIP "setuptools>=77.0.3,<81.0.0" "cmake>=3.26" setuptools_scm ninja packaging wheel
$UV_PIP -e ./verl --no-build-isolation
```

If the clone lives inside a git repository, mark it safe first or the editable build is
refused: `git config --global --add safe.directory "$(pwd)/verl"`.

## 4. Video and Qwen3.5 dependencies

```bash
$UV_PIP "transformers>=5.9" flash-linear-attention decord qwen-vl-utils pandas datasets accelerate

CAUSAL_CONV1D_FORCE_BUILD=TRUE CUDA_HOME=/usr/local/cuda \
TORCH_CUDA_ARCH_LIST="9.0" MAX_JOBS=16 \
    $UV_PIP "causal-conv1d==1.5.2" --no-build-isolation
```

`flash-linear-attention` and `causal-conv1d` are required by the Qwen3.5 hybrid attention
stack (three Gated DeltaNet layers per full-attention layer).

## 5. vLLM from source (~40 min)

```bash
export PATH="$ENV/bin:$PATH"          # ensure the venv's cmake 4.x is used
CUDA_HOME=/usr/local/cuda MAX_JOBS=16 TORCH_CUDA_ARCH_LIST="9.0" UV_LOCK_TIMEOUT=3600 \
    $UV_PIP "vllm==0.20.2" --no-binary vllm --no-build-isolation --reinstall-package vllm
```

## 6. flash-attn from source (~9 min)

```bash
CUDA_HOME=/usr/local/cuda FLASH_ATTENTION_FORCE_BUILD=TRUE \
TORCH_CUDA_ARCH_LIST="9.0" MAX_JOBS=32 \
    $UV_PIP "flash-attn>=2.7" --no-binary flash-attn --no-build-isolation
```

## 7. Two site-package patches

**vLLM duplicate op registration.** In `$ENV/lib/python3.12/site-packages/vllm/utils/torch_utils.py`,
make the op definition idempotent:

```python
my_lib = target_lib or vllm_lib
try:
    my_lib.define(op_name + schema_str, tags=tags)
except RuntimeError as e:
    if "same name and overload name multiple times" in str(e):
        return
    raise
my_lib.impl(op_name, op_func, dispatch_key=dispatch_key)
```

**qwen_vl_utils torchvision fallback.** In
`$ENV/lib/python3.12/site-packages/qwen_vl_utils/vision_process.py`, replace the
`VIDEO_READER_BACKENDS["torchvision"]` fallback with a `raise`. `torchvision.io.read_video`
was removed in torchvision 0.27, so silently falling back to it produces confusing errors
instead of a clear one. Always run with `FORCE_QWENVL_VIDEO_READER=decord`, which the
scripts in this repo set for you.

## 8. lmms-eval (needed for Video-MME and LongVideoBench)

```bash
uv pip install "lmms-eval>=0.7.1" --python $ENV/bin/python
uv pip install --no-deps evaluate scikit-learn sacrebleu tenacity portalocker colorama \
    lxml pytz scipy joblib threadpoolctl narwhals pytablewriter --python $ENV/bin/python
```

0.7.1 ships the `qwen3_5` model natively, which is what `scripts/eval/run_all.sh` asks for.

`pytablewriter` is easy to miss: lmms-eval only needs it for the final summary table, so
without it the run finishes and writes its results and then raises ImportError at the very
end. The scores are already on disk at that point.

The reported numbers were produced with a thin-wrapper adapter whose vision budget differs
sharply from the `Qwen3_VL` defaults (`max_pixels` by 12×). `run_all.sh` therefore passes
`min_pixels`, `max_pixels` and `total_pixels` explicitly, so it reproduces the published
scores against either adapter. If you invoke lmms-eval directly, pin them yourself —
see [`third_party/lmms_eval/README.md`](../third_party/lmms_eval/README.md), which also
carries the wrapper itself for exact-parity runs.

## 9. Verify

```bash
$ENV/bin/python -c "
import torch, vllm, verl, flash_attn, decord, transformers
print(f'torch {torch.__version__} cuda {torch.version.cuda}')
print(f'vllm {vllm.__version__}')
print(f'verl {verl.__version__} @ {verl.__file__}')
assert 'StreamOPD/verl/' in verl.__file__, 'verl is not the patched local copy'
from verl.trainer.distillation.losses import is_distillation_enabled
from verl.utils.reward_score import default_compute_score
assert default_compute_score('thinkstream_rlvr', 'The answer is B', 'B') == 1.0
print(f'flash_attn {flash_attn.__version__}, decord {decord.__version__}, transformers {transformers.__version__}')
print('ALL OK')
"
```

Expected: torch 2.11.0+cu129, vLLM 0.20.2, verl 0.8.0.dev resolved inside this repository,
flash_attn 2.8.3.post1, decord 0.6.0, transformers 5.12.1.

The `assert` on `verl.__file__` matters. Installing verl from PyPI or from upstream master
instead of the vendored copy fails at once with
`NotImplementedError: Reward function is not implemented for data_source='thinkstream_rlvr'`,
and if you work around that, again during distillation with a `teacher_ids != sequence_ids`
assertion.

## Why source builds

- **vLLM**: the PyPI wheel links against CUDA 13 (`libcudart.so.13`) and will not load on a
  CUDA 12.9 machine.
- **flash-attn**: the PyPI wheel requires GLIBC 2.32+, while the reference cluster (and most
  CentOS/RHEL-derived images) ships GLIBC 2.28.

Both are the reason for the long install; neither can be skipped on this configuration.
