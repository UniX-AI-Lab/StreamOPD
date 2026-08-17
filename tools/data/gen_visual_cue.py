#!/usr/bin/env python3
"""
Offline generation of teacher visual cues with a larger VLM (Qwen3.5-27B in the paper).

A cue is a spatio-temporal pointer — when and where to look — that must never contain the
answer. Two safeguards keep it that way:
  - the cue model is shown the question stem only, with options and answer stripped, so it
    cannot copy an option;
  - every generated cue is screened for leakage (option letters, yes/no assertions, high
    overlap with the correct option text) and regenerated at increasing temperature, with
    a fallback to no cue at all.

Run a vLLM server first, then this client:

  CUDA_VISIBLE_DEVICES=0,1 vllm serve Qwen/Qwen3.5-27B --port 8010 \
      --tensor-parallel-size 2 --max-model-len 32768 \
      --allowed-local-media-path / --gpu-memory-utilization 0.85 \
      --limit-mm-per-prompt '{"video":1,"image":1}' --media-io-kwargs '{"video":{"num_frames":-1}}'

  python tools/data/gen_visual_cue.py \
      --port 8010 --model-path Qwen/Qwen3.5-27B \
      --data-files data/filtered8k_plus_cot_verifiable_dedup_25118.parquet \
      --limit 0 --fps 32 --out data/cue/cue_25k.jsonl

Use --limit 0 for the full set; --num-shards / --shard-id split the work across servers.
With --limit 0 the emitted `idx` is the original parquet row number, so results can be
joined back with df.iloc[idx].
"""
import argparse, json, os, re, time
import concurrent.futures as cf


SYSTEM_PROMPT = (
    "You are a video analyst. For a multiple-choice question about a video, output a POINTER that "
    "tells the student WHEN and WHERE to look — WITHOUT answering the question.\n"
    "Think of it as pointing a flashlight: you reveal the location and moment, never the finding.\n"
    "\n"
    "ABSOLUTE PROHIBITIONS (any violation = total failure):\n"
    "1. NEVER output an option letter or option text (no 'A)', 'B.', 'E) To ...', and no sentence "
    "that restates any option).\n"
    "2. NEVER name the specific entity/attribute the question asks about. If it asks WHO/WHICH "
    "PERSON, don't name the person; if WHERE/POSITION, don't state the position; if WHAT OBJECT, "
    "don't name the object; if WHAT COLOR, don't say the color; if WHAT HAPPENS NEXT, don't say "
    "the outcome. Refer only generically ('that person', 'the object in their hand', 'their "
    "position').\n"
    "3. NEVER quote or transcribe on-screen text, captions, subtitles, or spoken dialogue. Only say "
    "WHEN/WHERE text appears (e.g. 'a subtitle appears around 0:09'), never its wording.\n"
    "4. NEVER describe the outcome, result, or a conclusion. Pointer only, no interpretation.\n"
    "\n"
    "FORMAT: exactly ONE sentence, under 25 words, shape = 'Around <when/which action>, look at "
    "<which region/subject>.'  No extra clauses, no 'suggesting', no 'indicating'.\n"
    "\n"
    "GOOD (Q: what color are the sneakers when the hand enters?):\n"
    "  'Around the moment the hand first enters the frame, look at the person's feet.'\n"
    "GOOD (Q: who places a tile next after the man with glasses?):\n"
    "  'Right after the man with glasses moves, look at the other players seated around the board.'\n"
    "GOOD (Q: where is the red bowl relative to the pizza?):\n"
    "  'As the woman slices the pizza, look at the area surrounding the cutting board.'\n"
    "BAD — all leak, never do these:\n"
    "  'E) To make sure the image is captured.'  (option letter+text)\n"
    "  'The girl in the purple shirt.'  (names the answer entity)\n"
    "  'To the right of the pizza.'  (states the position asked)\n"
    "  'The caption says \"Pregnant\".'  (transcribes text)\n"
    "  '...standing together, suggesting they are on a break.'  (describes outcome/conclusion)"
)


# Yes/no questions state the attribute under test inside the question itself ("is the
# helmet blue?"), so any description of the evidence tends to give the answer away. For
# them the cue is reduced to a pure spatio-temporal pointer: when to look and at whom or
# what region, never the attribute and never a judgement about it.
BINARY_SYSTEM_PROMPT = (
    "You are a video analyst helping a student LOCATE where to look for a yes/no question, "
    "WITHOUT answering it.\n"
    "The question asks whether some attribute/event holds at a moment. Your job is ONLY to point "
    "to WHEN and WHO/WHERE to inspect — NEVER to say whether it holds.\n"
    "\n"
    "STRICT RULES (violating any = failure):\n"
    "1. NEVER say yes/no, true/false, is/is not, was/wasn't, there is/there isn't, correct/incorrect, "
    "or any confirmation or denial.\n"
    "2. NEVER mention the specific attribute the question asks about (e.g. if it asks 'blue helmet', "
    "do NOT mention any helmet color; if 'red grinder', do NOT mention any grinder color). Refer to "
    "the subject only generically (the helmet, the tool, the person's hand).\n"
    "3. NEVER describe the state/outcome. Only give a spatio-temporal pointer.\n"
    "4. Output ONE sentence, under 25 words, of the form: 'Around <when/which action>, look at "
    "<who/which region>.'\n"
    "\n"
    "Good example (Q: Is the child in the red canoe wearing a blue helmet?): "
    "\"Around the segment showing the child in the red canoe, look closely at the head area.\"\n"
    "Bad examples (all leak): \"No, the helmet is yellow.\" / \"The child wears a red helmet.\" / "
    "\"Yes, there is a blue helmet.\""
)


def strip_options(content: str) -> str:
    """Keep only the question stem: drop the <video> marker and the answer options.

    The cue model never sees the options, which removes the easiest way for it to leak.
    """
    c = content.replace("<video>", "").replace("<image>", "").strip()
    m = re.search(r"\n\s*[A-F][\.\)]\s", c)  # cut before the first "\nA. " / "\nA) "
    if m:
        c = c[: m.start()]
    return c.strip()


def build_messages(prompt_field, videos_field, response_format=None):
    """Build the OpenAI messages: video plus the option-stripped question stem."""
    videos = list(videos_field) if videos_field is not None else []
    question_text = None
    video_item = None
    vid_idx = 0
    for msg in prompt_field:
        content = msg.get("content", "")
        if not isinstance(content, str):
            continue
        if "<video>" in content and vid_idx < len(videos):
            v = dict(videos[vid_idx])
            path = v["video"]
            url = path if path.startswith(("http://", "https://", "file://")) else "file://" + os.path.abspath(path)
            video_item = {"type": "video_url", "video_url": {"url": url}}
            vid_idx += 1
        question_text = strip_options(content)
    sys_prompt = BINARY_SYSTEM_PROMPT if response_format == "Binary" else SYSTEM_PROMPT
    user_content = []
    if video_item is not None:
        user_content.append(video_item)
    user_content.append({"type": "text", "text": f"Question: {question_text}\nVisual evidence:"})
    return [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_content},
    ], question_text


# ---------- answer-leakage detection ----------
def detect_leak(cue: str, response_format: str, gt: str, options_text: str) -> list:
    """Return the leakage flags a cue triggers; an empty list means it is clean."""
    flags = []
    cl = cue.lower()
    # 1) an explicit option letter: "option A" / "(B)" / "answer is C" / leading "A)" "A." "A:"
    if (re.search(r"\boption\s+[a-f]\b", cl) or re.search(r"\banswer\s+is\b", cl)
            or re.search(r"\([a-f]\)", cl) or re.search(r"^\s*[a-f][\.\):]", cue.strip(), re.IGNORECASE)):
        flags.append("option_letter")
    # 2) yes/no questions: any assertion or denial
    if response_format == "Binary":
        if re.search(r"\b(yes|no|not)\b", cl) or re.search(r"\bthere (is|are|isn't|aren't)\b", cl):
            flags.append("binary_assertion")
    # 3) counting questions: any digit or number word
    if response_format == "Counting":
        if re.search(r"\b\d+\b", cl) or re.search(r"\b(one|two|three|four|five|six|seven|eight|nine|ten)\b", cl):
            flags.append("counting_number")
    # 4) quoted text counts as leakage only when it overlaps the correct option; quoting an
    #    unrelated on-screen reading (e.g. a '10.5x' zoom indicator) is harmless.
    for qm in re.findall(r"[\"“'『「]([^\"”'』」]{2,})[\"”'』」]", cue):
        if response_format == "Multiple Choice" and options_text:
            gt_opt = _extract_gt_option(options_text, gt)
            if gt_opt and _word_jaccard(qm.lower(), gt_opt.lower()) >= 0.3:
                flags.append("quoted_answer_text")
                break
        # outside MCQ, a quoted phrase of 3+ words is still transcribed screen text
        elif len(re.findall(r"\w+", qm)) >= 3:
            flags.append("quoted_text")
            break
    # 5) MCQ: the cue overlaps the correct option's wording
    if response_format == "Multiple Choice" and options_text:
        gt_opt = _extract_gt_option(options_text, gt)
        if gt_opt:
            j = _word_jaccard(cl, gt_opt.lower())
            if j >= 0.35:
                flags.append(f"mcq_overlap_{j:.2f}")
            # Require 2+ distinct content words at 60%+ coverage, so a single generic word
            # such as 'standing' or 'together' does not flag an otherwise clean cue.
            opt_words = [w for w in re.findall(r"\w+", gt_opt.lower()) if w not in _STOP and len(w) > 3]
            if opt_words:
                hit = sum(1 for w in opt_words if w in cl)
                if hit >= 2 and hit >= 0.6 * len(opt_words):
                    flags.append("mcq_keyword_hit")
    return flags


def _extract_gt_option(options_text: str, gt: str) -> str:
    """Pull the option text for letter `gt` out of a '\nA. xxx\nB. yyy' block."""
    gt = str(gt).strip().upper()
    if not (len(gt) == 1 and gt in "ABCDEF"):
        return ""
    m = re.search(rf"\n\s*{gt}[\.\)]\s*(.+?)(?:\n\s*[A-F][\.\)]|$)", options_text, re.DOTALL)
    return m.group(1).strip() if m else ""


def _word_jaccard(a: str, b: str) -> float:
    sa = set(re.findall(r"\w+", a)) - _STOP
    sb = set(re.findall(r"\w+", b)) - _STOP
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


_STOP = set("a an the of to in on at is are was were be by with and or for this that it its as from".split())


def one_request(client, model, messages, args, temperature=None):
    extra = {"mm_processor_kwargs": {"fps": args.fps, "do_sample_frames": True},
             "chat_template_kwargs": {"enable_thinking": False}}
    resp = client.chat.completions.create(
        model=model, messages=messages,
        max_tokens=args.max_new_tokens,
        temperature=args.temperature if temperature is None else temperature,
        top_p=args.top_p, n=1, extra_body=extra, timeout=args.timeout,
    )
    return (resp.choices[0].message.content or "").strip()


# Sent on regeneration after a leak is detected; demands a pure pointer.
CUE_RETRY_HINT = (
    "Your previous response LEAKED the answer (it stated or restated an option/answer/color/"
    "count/yes-no/on-screen text). Rewrite it as a PURE POINTER: one short sentence 'Around "
    "<when/action>, look at <which region/subject>.' Do NOT state or hint the answer, do NOT "
    "output any option letter, and do NOT quote on-screen text. Output only the pointer."
)

# Second-stage screen: the same model judges whether a cue leaks, catching what the
# regex rules miss.
JUDGE_SYSTEM = (
    "You grade whether a HINT leaks the answer to a video question. A good hint ONLY points to "
    "WHERE/WHEN to look; it may name a region, object, or moment, but must not reveal the ANSWER.\n"
    "\n"
    "Judge by this rule:\n"
    "- SAFE: the hint is a pure pointer of the form 'look at <region/subject> around <when>'. It is "
    "still SAFE even if it names the object/person/area being asked about, AS LONG AS it does NOT "
    "state the answered attribute (the color, the count, the yes/no, the identity, the outcome, or "
    "the on-screen text content).\n"
    "- LEAK: the hint states or strongly implies the answer — e.g. gives the color/count/yes-no, "
    "names WHICH specific person/object is the answer, quotes the on-screen text that IS the "
    "answer, or describes the outcome/next-action the question asks for.\n"
    "\n"
    "Examples:\n"
    "  Q: where is the chair relative to the basket? Hint: 'look at the object directly underneath "
    "the basket.' -> SAFE (points to a location to inspect, does not state the spatial answer)\n"
    "  Q: what does the on-screen text say? Hint: \"the text 'Pregnant' is visible\" -> LEAK (quotes "
    "the answer text)\n"
    "  Q: what happens next? Hint: 'she rolls forward and rises, preparing for the next drill' -> "
    "LEAK (describes the outcome asked)\n"
    "  Q: next action of the bull? Hint: 'look at the bull's body orientation and the man's posture' "
    "-> SAFE (points to what to observe, does not state the action)\n"
    "\n"
    "Reply with EXACTLY one word: LEAK or SAFE."
)


def judge_leak(client, model, question, gt, cue, timeout=120):
    """True when the judge considers the cue to leak. Failures count as a leak."""
    try:
        msgs = [
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": f"Question: {question}\nCorrect answer: {gt}\nHint: {cue}\n\nVerdict (LEAK or SAFE):"},
        ]
        resp = client.chat.completions.create(
            model=model, messages=msgs, max_tokens=5, temperature=0.0, n=1,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}}, timeout=timeout,
        )
        out = (resp.choices[0].message.content or "").strip().upper()
        return "LEAK" in out  # only an explicit SAFE lets the cue through
    except Exception:
        return True


def run(args):
    import pandas as pd
    from openai import OpenAI

    client = OpenAI(base_url=f"http://127.0.0.1:{args.port}/v1", api_key="EMPTY")

    df = pd.read_parquet(args.data_file)
    print(f"[info] total {len(df)} samples", flush=True)
    if args.limit and args.limit < len(df):
        df = df.sample(n=args.limit, random_state=args.seed).reset_index(drop=True)
        print(f"[info] sampled {len(df)} (seed={args.seed})", flush=True)
    # Remember each row's position in the (possibly sampled) table so shards can be merged
    # and each cue can be traced back to its video.
    orig_pos = list(range(len(df)))
    # Sharding by row number modulo shard count: no overlap, no gaps.
    if args.num_shards > 1:
        keep = [i for i in orig_pos if i % args.num_shards == args.shard_id]
        df = df.iloc[keep].reset_index(drop=True)
        orig_pos = keep
        print(f"[shard {args.shard_id}/{args.num_shards}] {len(df)} rows in this shard", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fout = open(args.out, "w")

    def work(idx):
        row = df.iloc[idx]
        gt = str(row["ground_truth"])
        rfmt = str(row["response_format"])
        raw_content = row["prompt"][0]["content"]

        def _detect(cue):
            leak = detect_leak(cue, rfmt, gt, raw_content)
            rc = (len(leak) == 0)
            jv = None
            if rc and not args.no_judge:
                jv = "LEAK" if judge_leak(client, args.model_name, qtext, gt, cue) else "SAFE"
            return leak, rc, jv, (rc and jv != "LEAK")

        try:
            messages, qtext = build_messages(row["prompt"], row["videos"], response_format=rfmt)
            cue = one_request(client, args.model_name, messages, args)
            leak, rule_clean, judge_verdict, ok = _detect(cue)
            n_attempts = 1
            # Regenerate up to max_retries times, raising the temperature each round and
            # adding the stricter retry prompt.
            retry_temps = [0.5, 0.7, 0.9]
            while (not ok) and (not args.no_retry) and (n_attempts <= args.max_retries):
                t = retry_temps[min(n_attempts - 1, len(retry_temps) - 1)]
                retry_msgs = messages + [
                    {"role": "assistant", "content": cue[:120]},
                    {"role": "user", "content": CUE_RETRY_HINT},
                ]
                cue_r = one_request(client, args.model_name, retry_msgs, args, temperature=t)
                leak_r, rc_r, jv_r, ok_r = _detect(cue_r)
                n_attempts += 1
                if ok_r:
                    cue, leak, rule_clean, judge_verdict, ok = cue_r, leak_r, rc_r, jv_r, ok_r
                    break
                # Still leaking: keep the latest attempt for review and try again.
                cue, leak, rule_clean, judge_verdict = cue_r, leak_r, rc_r, jv_r
            # Rows that never came out clean fall back to standard OPD (no cue injected).
            fallback = (not ok)
            return {"idx": orig_pos[idx], "response_format": rfmt, "gt": gt,
                    "question": qtext, "cue": cue,
                    "leak_flags": leak, "rule_clean": rule_clean,
                    "judge": judge_verdict, "n_attempts": n_attempts,
                    "clean": ok, "fallback_to_opd": fallback,
                    "n_words": len(cue.split())}
        except Exception as e:
            return {"idx": orig_pos[idx], "response_format": rfmt, "gt": gt,
                    "cue": f"[ERROR] {str(e)[:150]}", "error": True,
                    "clean": False, "fallback_to_opd": True, "leak_flags": []}

    n_done, n_leak, n_err = 0, 0, 0
    n_retried, n_judge_caught = 0, 0
    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        for rec in ex.map(work, range(len(df))):
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            n_done += 1
            if rec.get("error"):
                n_err += 1
            elif not rec["clean"]:
                n_leak += 1
            if rec.get("n_attempts", 1) > 1:
                n_retried += 1
            if rec.get("rule_clean") and rec.get("judge") == "LEAK":
                n_judge_caught += 1
    fout.close()
    dt = time.time() - t0
    ok = n_done - n_err
    n_fallback = n_leak  # anything still flagged falls back to standard OPD
    print(f"[DONE] {n_done} cues (err={n_err}) in {dt:.1f}s", flush=True)
    print(f"[QC] final clean: {ok-n_leak}/{ok} ({100*(ok-n_leak)/max(ok,1):.1f}%)", flush=True)
    print(f"[retry] regenerated at least once: {n_retried} (up to {args.max_retries} times)", flush=True)
    print(f"[judge] caught by the LLM judge after passing the rules: {n_judge_caught}", flush=True)
    print(f"[fallback] never came out clean, will train without a cue: {n_fallback}", flush=True)
    print(f"[out] {args.out}", flush=True)
    print(f"\nInspect cue quality with: python tools/data/gen_visual_cue.py --review --out {args.out}", flush=True)


def review(args):
    """Print the generated cues one by one for manual inspection."""
    recs = [json.loads(l) for l in open(args.out)]
    clean = [r for r in recs if r.get("clean")]
    flagged = [r for r in recs if not r.get("clean") and not r.get("error")]
    print(f"=== {len(recs)} cues | clean={len(clean)} | leak-flagged={len(flagged)} ===\n")
    for r in recs:
        tag = "✅" if r.get("clean") else ("⚠️LEAK" if not r.get("error") else "❌ERR")
        print(f"[{r['idx']}] {tag} fmt={r.get('response_format')} gt={r.get('gt')} "
              f"words={r.get('n_words','?')} flags={r.get('leak_flags')}")
        print(f"  Q  : {r.get('question','')[:200]}")
        print(f"  CUE: {r.get('cue','')[:300]}")
        print()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8010)
    p.add_argument("--model-name", type=str, default=None, help="vLLM served model id (defaults to --model-path)")
    p.add_argument("--model-path", type=str, default="Qwen/Qwen3.5-27B")
    p.add_argument("--data-file", type=str,
                   default="data/filtered8k_plus_cot_verifiable_dedup_25118.parquet")
    p.add_argument("--limit", type=int, default=20, help="0 = full set")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-new-tokens", type=int, default=50)
    p.add_argument("--temperature", type=float, default=0.3)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--fps", type=float, default=4.0)
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--timeout", type=float, default=300)
    p.add_argument("--out", type=str, default="data/cue/cue.jsonl")
    p.add_argument("--review", action="store_true", help="print already-generated cues for inspection; no model calls")
    p.add_argument("--no-retry", action="store_true", help="do not regenerate cues flagged as leaking")
    p.add_argument("--max-retries", type=int, default=3, help="regeneration attempts at temperature 0.5/0.7/0.9")
    p.add_argument("--no-judge", action="store_true", help="skip the LLM-as-judge leakage screen")
    p.add_argument("--num-shards", type=int, default=1, help="total shards when running several servers in parallel")
    p.add_argument("--shard-id", type=int, default=0, help="shard handled by this process, in [0, num_shards)")
    args = p.parse_args()
    if args.model_name is None:
        args.model_name = args.model_path
    if args.review:
        review(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
