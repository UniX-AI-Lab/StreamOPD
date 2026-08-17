# Data

The parquets in `data/` hold questions, answers and video *references* — one row is a video
path plus a `[video_start, video_end]` window and a verifiable ground truth. The videos
themselves come from public datasets you download separately.

## Shipped files

| File | Rows | Used by |
|------|-----:|---------|
| `filtered8k_plus_cot_verifiable_dedup_25118.parquet` | 25,118 | OPD, ExOPD, AFD — the main training set |
| `train25k_with_cue_instruct.parquet` | 25,118 | ViCuR cue-only, ST-CueGate — same rows plus a `teacher_prompt` column |
| `train20k_filtered_8343.parquet` | 8,343 | GRPO — the pass-rate-filtered subset |
| `rlvr_val.parquet` | 326 | held-out validation during training |

`train25k_with_cue_instruct.parquet` differs from the main set by exactly one column:
`teacher_prompt`, the question wrapped with a visual-evidence instruction block. The
student-facing `prompt` column is identical, which is what makes ViCuR cue-only and ST-CueGate
single-variable comparisons against OPD.

## Getting the videos

The parquets reference 23,187 distinct clips from two public sources. Only five
LLaVA-Video-178K subsets are needed, so restrict the download rather than pulling the full
178K corpus.

```bash
export DATA_ROOT=/your/dataset/root
```

### LLaVA-Video-178K — 18,276 clips

<https://huggingface.co/datasets/lmms-lab/LLaVA-Video-178K>

| Subset | Clips used |
|--------|-----------:|
| `0_30_s_youtube_v0_1` | 14,636 |
| `0_30_s_academic_v0_1` | 2,900 |
| `1_2_m_academic_v0_1` | 474 |
| `1_2_m_activitynetqa` | 231 |
| `2_3_m_academic_v0_1` | 35 |

```bash
huggingface-cli download lmms-lab/LLaVA-Video-178K --repo-type dataset \
    --local-dir "$DATA_ROOT/LLaVA-Video-178K" \
    --include '0_30_s_youtube_v0_1/*' '0_30_s_academic_v0_1/*' \
              '1_2_m_academic_v0_1/*' '1_2_m_activitynetqa/*' \
              '2_3_m_academic_v0_1/*'

# videos ship as *_videos_N.tar.gz per subset; extract into a data/ subdirectory
cd "$DATA_ROOT/LLaVA-Video-178K"
for subset in 0_30_s_youtube_v0_1 0_30_s_academic_v0_1 1_2_m_academic_v0_1 \
              1_2_m_activitynetqa 2_3_m_academic_v0_1; do
    mkdir -p "$subset/data"
    for f in "$subset"/*_videos_*.tar.gz; do
        [ -e "$f" ] && tar -xzf "$f" -C "$subset/data"
    done
done
```

### Kinetics-700 via Tarsier2-Recap-585K — 4,911 clips

<https://huggingface.co/datasets/omni-research/Tarsier2-Recap-585K>

This dataset is gated: open the page and accept the conditions once before downloading, and
make sure `huggingface-cli login` has been run.

```bash
huggingface-cli download omni-research/Tarsier2-Recap-585K --repo-type dataset \
    --local-dir "$DATA_ROOT/tarsier2_unzip" --include 'Kinetics-700/*'

cd "$DATA_ROOT/tarsier2_unzip/Kinetics-700"
cat videos.tar.part-* | tar -xf -      # the archive is uploaded in split parts
```

### Expected layout

The parquets store absolute paths, so the nesting has to match:

```
$DATA_ROOT/
├── LLaVA-Video-178K/
│   ├── 0_30_s_youtube_v0_1/data/liwei_youtube_videos/videos/youtube_video_2024/ytb_*.mp4
│   ├── 0_30_s_academic_v0_1/data/academic_source/Charades/*.mp4
│   ├── 1_2_m_academic_v0_1/data/academic_source/Charades/*.mp4
│   ├── 1_2_m_activitynetqa/data/ActivityNet-QA/activitynet/train/v1-3/train_val/*.mp4
│   └── 2_3_m_academic_v0_1/data/academic_source/ego4d/*.mp4
└── tarsier2_unzip/Kinetics-700/videos/*.mp4
```

Now repoint the parquets, which ship with the placeholder root `/path/to/video_root`, and
verify in one step:

```bash
python tools/data/retarget_video_root.py --root "$DATA_ROOT" --check-exists
```

`--check-exists` reports how many rows still point at a missing file; it should print 0. A
non-zero count almost always means the archives extracted at a different nesting depth than
the layout above — compare one real file path against it and move or symlink the tree
rather than editing the parquets by hand.

## Rebuilding the training set from scratch

Only needed if you want to change the filters or extend the pool. The pipeline starts from
the public **ThinkStream** dataset (an 8.9K single-turn `streaming_rlvr` subset and a 110K
multi-turn `streaming_cot_cold` subset).

```
ThinkStream jsonl
   │  tools/data/convert_thinkstream_to_verl.py
   ▼
rlvr_train.parquet (6,208)  ──filters──▶  rlvr_train_filtered.parquet (2,662)
cot_train.parquet (79,156)  ──filters──▶  cot_rlvr_train_filtered.parquet (17,369)
   │                                            tools/data/convert_cot_to_rlvr.py
   ▼  concatenate
20k pool (20,031)
   │  tools/data/pass_rate_vllm.py  (n=8 samples from student and teacher)
   │  tools/data/filter_by_pass_rate.py
   ▼
train20k_filtered_8343.parquet (8,343)
   │  ∪ cot_verifiable_all.parquet (23,884, tools/data/convert_cot_verifiable_full.py)
   │  tools/data/merge_dedup.py
   ▼
filtered8k_plus_cot_verifiable_dedup_25118.parquet (25,118)
   │  tools/data/gen_visual_cue.py  →  tools/data/build_cue_instruct.py
   ▼
train25k_with_cue_instruct.parquet
```

### Filters

Rows survive only if the answer is verifiable — multiple choice (A-F), binary yes/no, or a
count — and the ground truth is clean after normalisation. Clips shorter than 0.5 s are
dropped because decord decodes them unreliably; `convert_cot_to_rlvr.py` additionally caps
clips at 10 s, while `convert_cot_verifiable_full.py` keeps the long ones, which is where
most of the extra 19k rows in the 25k set come from.

### Pass-rate filtering

Both models answer every sample 8 times at temperature 0.7, and a sample is kept when the
teacher is at least as accurate as the student and the pair is not already saturated:

```
keep = teacher.n_correct >= student.n_correct and not (student == teacher == 8)
```

Dropping "both perfect" removes rows whose KL signal is essentially zero; dropping "student
better than teacher" removes rows where distillation would actively teach a wrong answer.
On the 20,031-row pool this keeps 8,343 (41.7%). The measured distribution:

| Bucket | Rows | Share |
|--------|-----:|------:|
| both perfect | 7,920 | 39.5% |
| both wrong | 2,483 | 12.4% |
| teacher > student | 4,590 | 22.9% |
| student > teacher | 3,768 | 18.8% |
| tied below perfect | 1,270 | 6.3% |

Running it needs two vLLM servers, one per role:

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen3.5-4B --port 8001 \
    --tensor-parallel-size 1 --max-model-len 32768 --allowed-local-media-path / \
    --gpu-memory-utilization 0.85 --limit-mm-per-prompt '{"video":1,"image":1}' \
    --media-io-kwargs '{"video":{"num_frames":-1}}'
# and the same for Qwen/Qwen3.5-9B on port 8002

bash tools/data/chain_passrate.sh student 8001 Qwen/Qwen3.5-4B 111,222,333
bash tools/data/chain_passrate.sh teacher 8002 Qwen/Qwen3.5-9B 111,222,333
```

Roughly 320k vLLM calls for the full pool, about 6-8 hours on one GPU per role. The output
jsonl is resumable.

### Visual cue generation

`tools/data/gen_visual_cue.py` runs a larger model (Qwen3.5-27B) offline to produce, for
each sample, a spatio-temporal pointer saying *when and where to look* without revealing
the answer. Three layers guard against leakage:

1. the cue model never sees the answer options, only the question stem;
2. every cue is screened by rules (option letters, yes/no assertions, digits for counting
   questions, overlap with the correct option's wording) and regenerated up to three times
   at rising temperature;
3. surviving cues are graded LEAK/SAFE by the model itself.

Cues that never come out clean fall back to no cue, in which case `teacher_prompt == prompt`
and that row trains as standard OPD. About 96.6% of rows end up with a clean cue.

`tools/data/build_cue_instruct.py` then rewrites the inline hint into the explicit
instruction block that ViCuR cue-only and ST-CueGate train on.

Serving the 27B from `/dev/shm` rather than networked storage is worth it if several
shards run in parallel: six concurrent servers reading a 52GB model over a shared
filesystem slowed each shard from 8 s to 950 s.
