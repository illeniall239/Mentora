# Reference solution: only used by scripts/verify-exercises.mjs. Shown in the Breakdown only after the Learner's own solution passes.
from collections import Counter


def normalize(answer: str) -> str:
    return " ".join(answer.split()).lower()


def consistency_score(samples: list[str]) -> float:
    if not samples:
        raise ValueError("need at least one sample")
    normalized = [normalize(s) for s in samples]
    counts = Counter(normalized)
    # max() returns the first maximal element, and Counter keeps insertion order.
    majority = max(counts, key=counts.get)
    return counts[majority] / len(normalized)


def flag_inconsistent(samples: list[str], threshold: float) -> bool:
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    return consistency_score(samples) < threshold
