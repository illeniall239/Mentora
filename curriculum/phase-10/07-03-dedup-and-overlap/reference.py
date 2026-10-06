# Reference solution: only used by scripts/verify-exercises.mjs. Shown in the Breakdown only after the Learner's own solution passes.
import hashlib


def dedup_exact(docs: list[str]) -> list[str]:
    seen: set[bytes] = set()
    kept = []
    for doc in docs:
        digest = hashlib.sha256(doc.encode("utf-8")).digest()
        if digest not in seen:
            seen.add(digest)
            kept.append(doc)
    return kept


def _ngrams(text: str, n: int) -> set[tuple[str, ...]]:
    words = text.split()
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


def ngram_overlap(a: str, b: str, n: int) -> float:
    if n < 1:
        raise ValueError("n must be at least 1")
    A, B = _ngrams(a, n), _ngrams(b, n)
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)
