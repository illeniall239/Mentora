# Reference solution: only used by scripts/verify-exercises.mjs. Shown in the Breakdown only after the Learner's own solution passes.
import math


def expected_calibration_error(confidences: list[float], correct: list[bool], n_bins: int) -> float:
    if len(confidences) != len(correct) or not confidences or n_bins < 1:
        raise ValueError("need equal-length non-empty lists and n_bins >= 1")
    if any(not 0.0 <= c <= 1.0 for c in confidences):
        raise ValueError("confidences must be in [0, 1]")
    # Per bin: [count, correct count, confidence sum].
    bins = [[0, 0, 0.0] for _ in range(n_bins)]
    for c, ok in zip(confidences, correct):
        b = max(0, math.ceil(c * n_bins) - 1)
        bins[b][0] += 1
        bins[b][1] += int(ok)
        bins[b][2] += c
    n = len(confidences)
    ece = 0.0
    for count, hits, conf_sum in bins:
        if count:
            ece += (count / n) * abs(hits / count - conf_sum / count)
    return ece
