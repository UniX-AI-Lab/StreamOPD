#!/usr/bin/env python3
"""
Derive train25k_with_cue_instruct.parquet from train25k_with_cue.parquet.

The cue starts out appended inline to the question (`... [Visual evidence hint: <cue>]`).
This rewrites it into an explicit instruction block, which is what the ViCuR cue-only and ST-CueGate
runs train on:

  <video>question?
  A. ...  B. ...  C. ...  D. ...

  Here is some visual evidence to help you locate the answer in the video:
  <cue>

  After considering this visual evidence, identify the exact moment or region it points to
  in the video, verify what actually happens there, then answer the question using your own
  judgment.

The only thing that changes is the *wrapping* of teacher_prompt: the cue text itself is
untouched and the student's `prompt` column is not modified at all, which keeps the
comparison single-variable. Rows without a cue keep teacher_prompt == prompt and therefore
train exactly like standard OPD. The <video> placeholder stays at the front of the question
so the dataset's message builder is unaffected.

Cue extraction relies on the invariant
    teacher_prompt user content == student content + "\n\n[Visual evidence hint: <cue>]"
which held for all 24,257 injected rows (861 rows fall back, 0 malformed).
"""
import argparse
import copy
import re

import pandas as pd

CUE_MARK = "[Visual evidence hint:"
CUE_PAT = re.compile(r"\[Visual evidence hint:\s*(.*?)\]\s*$", re.S)

# The instruction block appended after the question; {cue} is the extracted cue text.
INSTRUCT_TEMPLATE = (
    "\n\nHere is some visual evidence to help you locate the answer in the video:\n"
    "{cue}\n\n"
    "After considering this visual evidence, identify the exact moment or region it points to "
    "in the video, verify what actually happens there, then answer the question using your own judgment."
)


def user_content(messages):
    for m in messages:
        if m.get("role") == "user":
            return m.get("content")
    return None


def set_user_content(messages, new_content):
    for m in messages:
        if m.get("role") == "user":
            m["content"] = new_content
            return True
    return False


def extract_cue(student_c: str, teacher_c: str):
    """Return the bare cue text, or None for fallback and malformed rows."""
    if teacher_c == student_c:
        return None
    if not (teacher_c.startswith(student_c) and CUE_MARK in teacher_c[len(student_c):]):
        return None
    block = teacher_c[len(student_c):].strip()
    m = CUE_PAT.search(block)
    return m.group(1).strip() if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="opd_training/data/train25k_with_cue.parquet")
    ap.add_argument("--out", default="opd_training/data/train25k_with_cue_instruct.parquet")
    args = ap.parse_args()

    df = pd.read_parquet(args.inp)
    n = len(df)
    print(f"[load] {n} rows from {args.inp}")

    new_tp = []
    n_instruct = n_fallback = n_bad = 0
    for i in range(n):
        prompt = df.iloc[i]["prompt"]
        teacher = copy.deepcopy(df.iloc[i]["teacher_prompt"])
        s_c = user_content(prompt)
        t_c = user_content(teacher)

        cue = extract_cue(s_c, t_c)
        if cue is None:
            # Keep the prompt as is; a fallback row already equals the student prompt and
            # therefore trains as standard OPD.
            if t_c == s_c:
                n_fallback += 1
            else:
                n_bad += 1
            new_tp.append(teacher)
            continue

        new_c = s_c + INSTRUCT_TEMPLATE.format(cue=cue)
        set_user_content(teacher, new_c)
        new_tp.append(teacher)
        n_instruct += 1

    out_df = df.copy()
    out_df["teacher_prompt"] = new_tp
    out_df.to_parquet(args.out)
    print(f"[stats] instruct_reformatted={n_instruct}  fallback(no cue)={n_fallback}  unexpected={n_bad}")
    print(f"[out] {args.out}  ({len(out_df)} rows)")


if __name__ == "__main__":
    main()
