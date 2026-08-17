#!/usr/bin/env python3
"""
Off-policy pass-rate estimation, used to pre-filter the training pool.

Each sample is answered n times (n=8 in the paper) by both the student and the teacher; the
fraction of correct answers is that sample's pass rate. Keeping only samples where the
teacher is at least as good as the student, and where neither is already perfect, removes
rows that carry no distillation signal or that would teach the student wrong answers.

Inference goes through a vLLM OpenAI-compatible server, which handles the Qwen3.5 video
path natively and returns all n samples in a single request. Start one server per role:

  # student on GPU 0
  CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen3.5-4B --port 8001 \
      --tensor-parallel-size 1 --max-model-len 32768 \
      --allowed-local-media-path / --gpu-memory-utilization 0.85 \
      --limit-mm-per-prompt '{"video":1,"image":1}' --media-io-kwargs '{"video":{"num_frames":-1}}'
  # teacher on GPU 1, same command with Qwen/Qwen3.5-9B and --port 8002

  python tools/data/pass_rate_vllm.py --role student --port 8001 \
      --data-files data/raw/rlvr_train_filtered.parquet \
      --limit 500 --seed 42 --n-samples 8 --fps 2 \
      --out data/passrate/student.jsonl

The output jsonl is resumable: re-running skips rows already present.
"""
import argparse, json, os, re, sys, time
import concurrent.futures as cf


# ---------- answer extraction / ground-truth normalisation (A-F + Yes/No/True/False) ----
def extract_answer(text: str) -> str:
    """Normalise a model response to A-F or YES/NO/TRUE/FALSE.

    With thinking enabled the response is <think>...</think> followed by the answer, so
    only the text after the closing tag is searched.
    """
    if not text:
        return ""
    t = text.strip()
    # Drop the thinking block: it often mentions several options in passing.
    if "</think>" in t:
        t = t.rsplit("</think>", 1)[1].strip()
    tl = t.lower()

    # 1) explicit "answer is X", highest priority
    m = re.search(r"(?:answer\s*(?:is|:)?|答案\s*(?:是|为|:)?)\s*\(?\*?\*?([A-F])\b", t, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    # 2) yes/no/true/false, a common ground-truth form in this data
    m = re.search(r"\b(yes|no|true|false)\b", tl)
    if m:
        return {"yes": "YES", "no": "NO", "true": "TRUE", "false": "FALSE"}[m.group(1)]
    # 3) a leading option letter: "A." / "A)" / "(A)" / "**A**" / "A:"
    m = re.search(r"(?:^|\n)\s*\(?\*?\*?([A-F])[\.\)\:\,\s\*]", t)
    if m:
        return m.group(1).upper()
    # 4) last resort: the final standalone A-F, since the answer usually ends the response
    letters = re.findall(r"\b([A-F])\b", t)
    if letters:
        return letters[-1].upper()
    return ""


def gt_to_letter(gt) -> str:
    """Normalise ground truth: A-F letter, 0-based index, or yes/no/true/false."""
    s = str(gt).strip().upper()
    if s in ("YES", "NO", "TRUE", "FALSE"):
        return s
    if len(s) == 1 and s in "ABCDEF":
        return s
    # The data also uses the abbreviations 'Y' and 'N'.
    if s == "Y":
        return "YES"
    if s == "N":
        return "NO"
    try:
        return chr(65 + int(s))
    except (ValueError, TypeError):
        return s[:1].upper()


# ---------- build OpenAI messages from a training row (prompt + videos) ----------
def build_messages(prompt_field, videos_field):
    import numpy as np
    videos = list(videos_field) if videos_field is not None else []
    out = []
    vid_idx = 0
    for msg in prompt_field:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if not isinstance(content, str):
            out.append({"role": role, "content": content})
            continue
        parts = re.split(r"(<video>|<image>)", content)
        cl = []
        for seg in parts:
            if seg == "<video>":
                if vid_idx < len(videos):
                    v = dict(videos[vid_idx])
                    path = v["video"]
                    url = path if path.startswith(("http://", "https://", "file://")) else "file://" + os.path.abspath(path)
                    item = {"type": "video_url", "video_url": {"url": url}}
                    cl.append(item)
                    vid_idx += 1
            elif seg == "<image>":
                pass
            elif seg:
                cl.append({"type": "text", "text": seg})
        out.append({"role": role, "content": cl})
    return out


def one_request(client, model, messages, args):
    """Request n=args.n_samples samples for one row and return the predicted letters."""
    extra = {"top_k": args.top_k,
             "mm_processor_kwargs": {"fps": args.fps, "do_sample_frames": True}}
    if not args.thinking:
        extra["chat_template_kwargs"] = {"enable_thinking": False}
    resp = client.chat.completions.create(
        model=model, messages=messages,
        max_tokens=args.max_new_tokens, temperature=args.temperature,
        top_p=args.top_p, n=args.n_samples, extra_body=extra,
        timeout=args.timeout,
    )
    preds = []
    for ch in resp.choices:
        txt = ch.message.content or ""
        preds.append(extract_answer(txt))
    return preds


def run(args):
    import pandas as pd
    from openai import OpenAI

    client = OpenAI(base_url=f"http://127.0.0.1:{args.port}/v1", api_key="EMPTY")

    dfs = []
    for f in args.data_files:
        if os.path.isfile(f):
            dfs.append(pd.read_parquet(f))
        else:
            print(f"[warn] missing {f}", flush=True)
    df = pd.concat(dfs, ignore_index=True)
    print(f"[info] total {len(df)} samples", flush=True)
    if args.limit and args.limit < len(df):
        df = df.sample(n=args.limit, random_state=args.seed).reset_index(drop=True)
        print(f"[info] sampled {len(df)} (seed={args.seed})", flush=True)

    done = set()
    if os.path.isfile(args.out) and not args.no_resume:
        for line in open(args.out):
            try:
                done.add(json.loads(line)["idx"])
            except Exception:
                pass
        print(f"[info] resume: {len(done)} done", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fout = open(args.out, "a")

    todo = [i for i in range(len(df)) if i not in done]

    def work(idx):
        row = df.iloc[idx]
        gt = gt_to_letter(row["ground_truth"])
        sid = str(row["id"]) if "id" in row else str(idx)
        try:
            messages = build_messages(row["prompt"], row["videos"])
            preds = one_request(client, args.model_name, messages, args)
            n_corr = sum(1 for p in preds if p == gt and gt != "")
            return {"idx": idx, "sample_id": sid, "gt": gt, "n_samples": len(preds),
                    "n_correct": n_corr,
                    "pass_rate": round(n_corr / len(preds), 4) if preds else 0.0,
                    "preds": preds, "role": args.role}
        except Exception as e:
            return {"idx": idx, "sample_id": sid, "gt": gt, "n_samples": 0, "n_correct": 0,
                    "pass_rate": 0.0, "preds": [f"[ERROR] {str(e)[:120]}"],
                    "role": args.role, "error": True}

    n_done = 0
    sum_pass = 0.0
    n_err = 0
    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        for rec in ex.map(work, todo):
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            n_done += 1
            if rec.get("error"):
                n_err += 1
            else:
                sum_pass += rec["pass_rate"]
            if n_done % 20 == 0:
                rate = n_done / (time.time() - t0)
                ok = n_done - n_err
                print(f"[{args.role}] {n_done}/{len(todo)} done "
                      f"(err={n_err}) mean_pass={sum_pass/max(ok,1):.3f} "
                      f"{rate:.2f}/s", flush=True)
    fout.close()
    ok = n_done - n_err
    print(f"[{args.role}] DONE {n_done} (err={n_err}) mean_pass={sum_pass/max(ok,1):.4f}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--role", choices=["student", "teacher"], default="student")
    p.add_argument("--port", type=int, default=8001)
    p.add_argument("--model-name", type=str, default=None,
                   help="vLLM served model id; defaults to --model-path")
    p.add_argument("--model-path", type=str, default=None,
                   help="only used to derive the served name (vLLM serves under the path it was given)")
    p.add_argument("--data-files", nargs="+", default=[
        "data/raw/rlvr_train_filtered.parquet",
        "data/raw/cot_rlvr_train_filtered.parquet",
    ])
    p.add_argument("--limit", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-samples", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=24)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--fps", type=float, default=2.0)
    p.add_argument("--thinking", action="store_true", help="keep the CoT block (off by default)")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--timeout", type=float, default=300)
    p.add_argument("--out", type=str, default="data/passrate/out.jsonl")
    p.add_argument("--no-resume", action="store_true")
    args = p.parse_args()
    if args.model_name is None:
        # vLLM registers the model under the path it was launched with.
        args.model_name = args.model_path or os.environ.get("PR_MODEL_NAME", "")
    run(args)


if __name__ == "__main__":
    main()
