# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

"""Exact-match reward for ThinkStream-style streaming video QA.

Three answer types, dispatched on the *ground_truth* shape (the gt is clean;
the model response is free-form):

    Multiple Choice : single letter A-H        e.g. gt="B"
    Binary          : Yes / No                 e.g. gt="Yes"
    Counting        : non-negative integer     e.g. gt="3"

Design principle: trust the ground_truth to pick the parsing strategy, then
extract the corresponding token from the (noisy) response. This avoids the
old failure mode where a free-form "Yes." response fell through to a
"last character" heuristic and scored 0.
"""

import re

__all__ = ["compute_score", "classify_gt", "extract_letter", "extract_binary", "extract_count"]


def classify_gt(gt: str) -> str:
    """Return one of {'mcq', 'binary', 'counting', 'unknown'} for a clean gt."""
    s = str(gt).strip().rstrip(".").strip()
    if re.fullmatch(r"[A-Ha-h]", s):
        return "mcq"
    if s.lower() in ("yes", "no"):
        return "binary"
    if re.fullmatch(r"\d+", s):
        return "counting"
    return "unknown"


def _answer_span(text: str) -> str:
    """Return the text after an explicit 'answer is/:' cue if present, else the
    whole text. Helps focus extraction on the final answer when a model emits a
    long chain-of-thought followed by 'The answer is X'."""
    # Prefer the LAST occurrence of an answer cue (models often restate).
    matches = list(re.finditer(r"(?:answer|option|choice)\s*(?:is|:|=)?\s*", text, re.IGNORECASE))
    if matches:
        return text[matches[-1].end():]
    return text


def extract_letter(solution: str) -> str:
    """Extract a single MCQ option letter A-H from a response. Returns '' if none.

    Case handling is deliberately asymmetric to avoid matching the English
    articles 'a'/'A' or the pronoun 'I' as if they were option letters:
      - After an explicit answer cue ('answer is c'), accept lower OR upper case.
      - Without a cue, only accept UPPER-case standalone letters (models are
        prompted to answer with capital A/B/C/D), so 'there is a dog' -> ''.
    """
    span = _answer_span(solution)
    cued = span is not solution  # _answer_span sliced -> an explicit cue was present

    # whole response is just a bare letter ("B", "d") -> that's the answer
    stripped = solution.strip().rstrip(").:").strip()
    if re.fullmatch(r"[A-Ha-h]", stripped):
        return stripped.upper()

    if cued:
        # right after a cue, a lone letter (either case) is the answer
        m = re.search(r"\b([A-Ha-h])\b", span)
        if m:
            return m.group(1).upper()

    # no cue (or cue had no letter): only trust UPPER-case standalone letters,
    # last one wins (closest to a final answer)
    letters = re.findall(r"\b([A-H])\b", solution)
    if letters:
        return letters[-1]
    # very last resort: leading bare letter like "B)" / "C." / "d)" (option style)
    m = re.match(r"\s*([A-Ha-h])[\).:\s]", solution)
    return m.group(1).upper() if m else ""


def extract_binary(solution: str) -> str:
    """Extract 'Yes'/'No' from a response. Returns '' if neither found.

    Uses the FIRST yes/no token, since binary answers normally lead the response
    ('No, the spoon ...'). Word-boundary matched to avoid 'no' inside 'nothing'.
    """
    m = re.search(r"\b(yes|no)\b", solution, re.IGNORECASE)
    if m:
        return m.group(1).capitalize()
    return ""


def extract_count(solution: str) -> str:
    """Extract a non-negative integer from a response. Returns '' if none.

    Prefers a number right after an answer cue; otherwise the first standalone
    integer in the text.
    """
    span = _answer_span(solution)
    m = re.search(r"\b(\d+)\b", span)
    if m:
        return m.group(1)
    m = re.search(r"\b(\d+)\b", solution)
    return m.group(1) if m else ""


def compute_score(solution_str: str, ground_truth) -> float:
    """Return 1.0 if the response matches the ground_truth, else 0.0."""
    if solution_str is None:
        return 0.0
    gt_raw = str(ground_truth).strip().rstrip(".").strip()
    kind = classify_gt(gt_raw)

    if kind == "mcq":
        pred = extract_letter(solution_str)
        return 1.0 if pred == gt_raw.upper() else 0.0
    if kind == "binary":
        pred = extract_binary(solution_str)
        return 1.0 if pred.lower() == gt_raw.lower() else 0.0
    if kind == "counting":
        pred = extract_count(solution_str)
        return 1.0 if pred == gt_raw else 0.0

    # Unknown gt shape: fall back to case-insensitive exact match on the
    # stripped strings (keeps behaviour defined rather than crashing).
    return 1.0 if solution_str.strip().rstrip(".").strip().lower() == gt_raw.lower() else 0.0
