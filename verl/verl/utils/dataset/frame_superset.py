# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

"""[AFD v3] Frame-superset index construction for asymmetric frame-budget distillation.

Given the frame indices the STUDENT actually sampled (S), build a denser teacher
index set T such that ``T ⊇ S`` by inserting intermediate frames strictly between
adjacent student frames. This makes the teacher's extra visual evidence a pure
*increment* over the student's frames (recoverable privilege, ViCuR) rather than a
disjoint resampling — which is what AFD v2 did and which failed (the teacher's
distribution became unrealizable for the student; see DESIGN_afd_v3_frame_superset.md).

Pure functions only — no decord / torch-cuda / model deps — so this is unit-testable
in isolation.
"""

from __future__ import annotations

__all__ = ["build_superset_indices", "build_span_directed_indices"]


def build_superset_indices(
    student_indices,
    insert_per_gap: int = 1,
    max_frames: int | None = None,
) -> list[int]:
    """Return teacher frame indices ``T`` such that ``T ⊇ S``.

    Between each adjacent pair of (sorted, de-duplicated) student frames, insert
    ``insert_per_gap`` equally-spaced intermediate frame indices. The original
    student frames are ALWAYS retained (that is the whole point — superset).

    Args:
        student_indices: iterable of int, the frame indices the student actually saw (S).
            Order/duplicates do not matter; they are sorted and de-duplicated.
        insert_per_gap: number of frames ``r`` to insert into each gap.
            ``r <= 0`` returns exactly ``S`` (degenerates to standard OPD — the teacher
            reuses the student's frames). This is the correctness-ablation knob.
        max_frames: optional cap on ``len(T)``. When the constructed superset exceeds
            this budget, INSERTED frames are dropped uniformly while EVERY student frame
            is kept, so the superset property ``T ⊇ S`` is never violated. If the budget
            cannot even hold ``S`` itself, ``S`` is returned unchanged (teacher then sees
            exactly the student frames — still a valid, if trivial, superset).

    Returns:
        Sorted list[int] of teacher frame indices, guaranteed to contain every element
        of ``S``.
    """
    S = sorted({int(x) for x in student_indices})
    if len(S) == 0:
        return []
    if insert_per_gap <= 0:
        return _cap_keeping_student(S, [], max_frames)

    inserted: set[int] = set()
    student_set = set(S)
    for a, b in zip(S[:-1], S[1:]):
        gap = b - a
        if gap <= 1:
            # Adjacent frames already touch; no room to insert without colliding with S.
            continue
        for k in range(insert_per_gap):
            # Equally-spaced open-interval positions: (k+1)/(r+1) of the way from a to b.
            pos = round(a + gap * (k + 1) / (insert_per_gap + 1))
            if pos not in student_set:
                inserted.add(pos)

    return _cap_keeping_student(S, sorted(inserted), max_frames)


def _cap_keeping_student(S: list[int], inserted: list[int], max_frames: int | None) -> list[int]:
    """Merge S with inserted frames, honoring ``max_frames`` by dropping ONLY inserted frames.

    Never drops a student frame — preserves ``T ⊇ S``.
    """
    if max_frames is None or (len(S) + len(inserted)) <= max_frames:
        return sorted(set(S) | set(inserted))

    keep = max_frames - len(S)
    if keep <= 0:
        # Budget can't hold the extra frames (or not even S). Return S: trivial superset.
        return S
    # Uniformly subsample `keep` of the inserted frames (preserve spread across the video).
    n = len(inserted)
    if keep >= n:
        chosen = inserted
    else:
        # Pick `keep` evenly-spaced positions from the inserted list.
        chosen = [inserted[round(i * (n - 1) / (keep - 1))] for i in range(keep)] if keep > 1 else [inserted[n // 2]]
    return sorted(set(S) | set(chosen))


def build_span_directed_indices(
    student_indices,
    n_total: int,
    span,
    insert_per_gap: int = 1,
    max_frames: int | None = None,
) -> list[int]:
    """[AFD v3.1 — span-directed] Teacher superset that only densifies WITHIN an evidence span.

    Same superset guarantee as :func:`build_superset_indices` (``T ⊇ S``), but instead of
    inserting extra frames into *every* gap uniformly, extra frames are inserted ONLY into
    the gaps that fall inside the answer-evidence time window ``span`` (relative [0,1]).
    Outside the window the teacher keeps exactly the student's frames. This concentrates the
    teacher's extra visual budget on the moment the question is actually about — the value of
    the ``answer_span`` labels — rather than spreading it thin across the whole clip.

    Args:
        student_indices: iterable of int, the frame indices the student actually saw (S), as
            ABSOLUTE frame numbers into the decoded video.
        n_total: total number of frames in the video (used to map the relative span to
            absolute frame positions).
        span: ``[s, e]`` with ``0 <= s < e <= 1`` (relative position). If ``None`` (no /
            untrusted span), this degenerates EXACTLY to :func:`build_superset_indices`
            (uniform densification) — the caller uses this for OFF / non-grounded samples.
        insert_per_gap: number of frames ``r`` to insert into each *in-span* gap.
        max_frames: optional cap on ``len(T)``. Inserted frames are dropped uniformly to
            honor the cap while every student frame is kept (``T ⊇ S`` preserved).

    Returns:
        Sorted list[int] of teacher frame indices, guaranteed to contain every element of S.
    """
    # No / untrusted span → fall back to uniform superset (single code path for OFF samples).
    if span is None:
        return build_superset_indices(student_indices, insert_per_gap=insert_per_gap, max_frames=max_frames)

    S = sorted({int(x) for x in student_indices})
    if len(S) == 0:
        return []
    if insert_per_gap <= 0:
        return _cap_keeping_student(S, [], max_frames)

    s_rel, e_rel = float(span[0]), float(span[1])
    # Guard against degenerate / inverted spans → treat as "no useful window" → uniform.
    if not (0.0 <= s_rel < e_rel <= 1.0001):
        return build_superset_indices(S, insert_per_gap=insert_per_gap, max_frames=max_frames)

    lo = s_rel * max(n_total, 1)
    hi = e_rel * max(n_total, 1)

    inserted: set[int] = set()
    student_set = set(S)
    for a, b in zip(S[:-1], S[1:]):
        gap = b - a
        if gap <= 1:
            continue
        # Only densify gaps whose midpoint lies inside the evidence window.
        mid = (a + b) / 2.0
        if not (lo <= mid <= hi):
            continue
        for k in range(insert_per_gap):
            pos = round(a + gap * (k + 1) / (insert_per_gap + 1))
            if pos not in student_set:
                inserted.add(pos)
    return _cap_keeping_student(S, sorted(inserted), max_frames)
