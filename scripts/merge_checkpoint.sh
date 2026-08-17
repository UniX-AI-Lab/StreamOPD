#!/usr/bin/env bash
# Convert a verl FSDP checkpoint into a HuggingFace-format directory that the eval scripts
# and lmms-eval can load.
#
#   bash scripts/merge_checkpoint.sh <CKPT_DIR> [BASE_MODEL]
#   bash scripts/merge_checkpoint.sh checkpoints/<experiment>/global_step_1400
#
# CKPT_DIR is the directory that contains `actor/`. The merged model is written to
# <CKPT_DIR>/actor/huggingface. BASE_MODEL supplies the tokenizer/processor files and the
# full config: the merger writes a config that omits fields such as head_dim and
# layer_types, so they have to be copied back from the original student checkpoint.

set -euo pipefail

CKPT_DIR="${1:?Usage: merge_checkpoint.sh <CKPT_DIR> [BASE_MODEL]}"
BASE_MODEL="${2:-${STUDENT_MODEL:-Qwen/Qwen3.5-4B}}"
PYTHON=${PYTHON:-python3}
TARGET="$CKPT_DIR/actor/huggingface"

$PYTHON -m verl.model_merger merge \
    --backend fsdp \
    --local_dir "$CKPT_DIR/actor" \
    --target_dir "$TARGET" \
    --use_cpu_initialization

if [[ -d "$BASE_MODEL" ]]; then
    for f in config.json tokenizer_config.json tokenizer.json preprocessor_config.json \
             video_preprocessor_config.json merges.txt vocab.json chat_template.jinja; do
        [[ -f "$BASE_MODEL/$f" ]] && cp "$BASE_MODEL/$f" "$TARGET/"
    done
else
    echo "BASE_MODEL is not a local directory; pulling config/tokenizer from the hub instead"
    $PYTHON - "$BASE_MODEL" "$TARGET" <<'PY'
import shutil
import sys

from transformers import AutoConfig, AutoProcessor, AutoTokenizer

src, dst = sys.argv[1], sys.argv[2]
AutoConfig.from_pretrained(src).save_pretrained(dst)
AutoTokenizer.from_pretrained(src).save_pretrained(dst)
try:
    AutoProcessor.from_pretrained(src).save_pretrained(dst)
except Exception as exc:  # processor is optional for text-only checkpoints
    print(f"no processor copied: {exc}")
PY
fi

$PYTHON -c "
from transformers import AutoConfig, AutoTokenizer
AutoConfig.from_pretrained('$TARGET')
AutoTokenizer.from_pretrained('$TARGET')
print('merged checkpoint loads: $TARGET')
"
