# Evaluation

Four benchmarks, all in instruct mode (thinking off), all under the memory-free
recent-window protocol for the streaming ones.

| Benchmark | Runner | Key settings | Reported metric |
|-----------|--------|--------------|-----------------|
| StreamingBench | `streamopd.eval.streamingbench` | recent 4 frames, 1 fps, 1 s chunks | accuracy |
| OVO-Bench | `streamopd.eval.ovo_bench` | recent 4 frames | B+R macro, with and without HLD |
| LongVideoBench | lmms-eval `longvideobench_val_v` | `max_num_frames=32`, pinned pixel budget | overall accuracy |
| Video-MME | lmms-eval `videomme` | `max_num_frames=32`, pinned pixel budget | perception score |

The pixel budget is pinned deliberately. Different lmms-eval adapters default to very
different vision resolutions — `max_pixels` varies by 12× between the Qwen3-VL base class
and the Qwen3.5 wrapper the reported numbers used — and that moves the scores. `run_all.sh`
passes `min_pixels=65536`, `max_pixels=131072` and `total_pixels=234881024` explicitly so
the result does not depend on which adapter is installed. Override them with the
`MIN_PIXELS` / `MAX_PIXELS` / `TOTAL_PIXELS` environment variables if you need to, but then
the numbers are no longer comparable to the table in the README.

## 1. Convert the checkpoint

verl writes FSDP shards; the evaluators need HuggingFace format.

```bash
bash scripts/merge_checkpoint.sh checkpoints/<experiment>/global_step_1400
```

This merges the shards and then copies `config.json`, the tokenizer and the processor files
from the base student model. The copy is not optional: the merger emits a config that omits
fields such as `head_dim` and `layer_types`, and loading it fails without them.

## 2. Benchmark data

All four benchmarks are public.

| Benchmark | Source |
|-----------|--------|
| StreamingBench | <https://huggingface.co/datasets/mjuicem/StreamingBench> ([code](https://github.com/THUNLP-MT/StreamingBench)) |
| OVO-Bench | <https://huggingface.co/datasets/JoeLeelyf/OVO-Bench> ([code](https://github.com/joeleelyf/ovo-bench)) |
| Video-MME | <https://huggingface.co/datasets/lmms-lab/Video-MME> |
| LongVideoBench | <https://huggingface.co/datasets/longvideobench/LongVideoBench> |

### StreamingBench and OVO-Bench

These two are read directly from `BENCH_ROOT`, which defaults to `data/benchmarks`:

```
$BENCH_ROOT/
├── streamingbench/
│   ├── questions_real.json
│   └── videos/
└── ovo_bench/
    ├── ovo_bench_new.json
    └── chunked_videos/
```

StreamingBench ships its videos as per-task zip archives; extract them and run the
repository's `scripts/preprocess.sh`, which produces the flat `videos/` directory.

OVO-Bench offers two downloads. Take the **pre-chunked** one — `chunked_videos.tar.parta[a-o]`,
about 144 GB — because our evaluator's `--chunked_dir` expects exactly that layout:

```bash
cd $BENCH_ROOT/ovo_bench
cat chunked_videos.tar.parta* | tar -xf -
```

The alternative `src_videos.tar.parta[a-e]` (~44 GB) is smaller but then requires running
the OVO-Bench repository's `scripts/chunk_video.sh` to produce the chunks yourself.

### Video-MME and LongVideoBench

These two are driven by lmms-eval, which finds videos through its own cache rather than
`BENCH_ROOT`. Download them anywhere and symlink:

```bash
huggingface-cli download lmms-lab/Video-MME --repo-type dataset \
    --local-dir $DATA_ROOT/Video-MME
huggingface-cli download longvideobench/LongVideoBench --repo-type dataset \
    --local-dir $DATA_ROOT/LongVideoBench

mkdir -p ~/.cache/huggingface/videomme ~/.cache/huggingface/datasets/longvideobench
ln -sfn $DATA_ROOT/Video-MME/data_hf  ~/.cache/huggingface/videomme/data
ln -sfn $DATA_ROOT/Video-MME/subtitle ~/.cache/huggingface/videomme/subtitle
ln -sfn $DATA_ROOT/LongVideoBench/videos ~/.cache/huggingface/datasets/longvideobench/videos

# expect 900 and 3992
ls ~/.cache/huggingface/videomme/data/*.mp4 | wc -l
ls ~/.cache/huggingface/datasets/longvideobench/videos/*.mp4 | wc -l
```

Three things reliably go wrong here:

- **Video-MME ships in two layouts.** lmms-eval matches files by YouTube ID
  (`026dzf-vc5g.mp4`), which is the `data_hf/` layout. The numerically named `videos/`
  layout (`001.mp4`) will not match anything.
- **An empty cache directory is worse than no directory.** If `~/.cache/huggingface/videomme`
  exists without a populated `data/` inside, lmms-eval treats the dataset as cached, extracts
  nothing, and reports missing videos.
- **Do not set `HF_DATASETS_OFFLINE=1`.** lmms-eval still fetches a few MB of task metadata
  from the hub even though the videos are local; offline mode breaks the dataset builder.
  Export `HF_TOKEN` (and a proxy if your network needs one) before running.

## 3. Run

```bash
bash scripts/eval/run_all.sh <MODEL_PATH> <RUN_NAME> 0,1,2,3
bash scripts/eval/score_all.sh <RUN_NAME>
```

One benchmark per GPU, in parallel. Results land under `results/`, logs under `logs/eval/`.
OVO-Bench is the long pole at roughly two hours.

To run a single benchmark, invoke its module directly:

```bash
CUDA_VISIBLE_DEVICES=0 python -m streamopd.eval.streamingbench \
    --anno-path data/benchmarks/streamingbench/questions_real.json \
    --video-dir data/benchmarks/streamingbench/videos \
    --qa-model <MODEL_PATH> \
    --top-k 0 --recent-frames-only 4 --chunk-duration 1.0 --fps 1.0 \
    --output-dir results/sb_run
```

MLVU and LVBench are not part of `run_all.sh`; add them through lmms-eval with
`--tasks mlvu_dev` or `--tasks lvbench` and the same `max_num_frames=32`.

## Protocol notes

**Instruct mode, not thinking mode.** All reported numbers use thinking off. The same
checkpoint scores 78.02 on StreamingBench with thinking on and 84.29 with it off — a 6.3
point difference that has nothing to do with the method. Never compare across modes.

**OVO-Bench is reported as the B+R macro.** Average the three Backward sub-tasks, average
the six Realtime sub-tasks, then average those two category means. This is the official
macro, not a flat mean over all nine tasks. `streamopd.scoring.ovo_bench` prints it, and
prints it twice: once including HLD and once excluding it. HLD measures the ability to
abstain, which every distilled model loses relative to the untrained baseline, for reasons
unrelated to streaming skill — reporting only one of the two numbers is misleading in
either direction.

**Answer parsing.** Distilled checkpoints answer with a single character, so all reasonable
parsers agree and the score is parser-independent. That stops being true for GRPO, whose
outputs drift to ~1000 characters: parsing its first letter gives 35.10 on StreamingBench
while parsing its last gives 82.37. Any comparison involving a model that produces long
outputs has to state which parser was used.
