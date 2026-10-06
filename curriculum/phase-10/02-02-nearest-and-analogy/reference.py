# Reference solution: only used by scripts/verify-exercises.mjs. Shown in the Breakdown only after the Learner's own solution passes.
import numpy as np


def _cosines(query: np.ndarray, table: np.ndarray) -> np.ndarray:
    return (table @ query) / (np.linalg.norm(table, axis=1) * np.linalg.norm(query))


def nearest(query: np.ndarray, table: np.ndarray, k: int) -> list[int]:
    order = np.argsort(-_cosines(query, table))
    return [int(i) for i in order[:k]]


def analogy(a: str, b: str, c: str, table: np.ndarray, vocab: list[str]) -> str:
    for w in (a, b, c):
        if w not in vocab:
            raise ValueError(f"{w!r} is not in the vocabulary")
    ia, ib, ic = vocab.index(a), vocab.index(b), vocab.index(c)
    target = table[ib] - table[ia] + table[ic]
    for i in nearest(target, table, len(vocab)):
        if i not in (ia, ib, ic):
            return vocab[i]
    raise ValueError("vocabulary has no word besides a, b and c")
