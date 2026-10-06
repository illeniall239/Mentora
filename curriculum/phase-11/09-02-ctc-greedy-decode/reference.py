# Reference solution: only used by scripts/verify-exercises.mjs. Shown in the Breakdown only after the Learner's own solution passes.
from itertools import groupby


def ctc_greedy_decode(frame_ids: list[int], blank: int) -> list[int]:
    if blank < 0 or any(i < 0 for i in frame_ids):
        raise ValueError("blank and every frame id must be non-negative")
    return [i for i, _ in groupby(frame_ids) if i != blank]
