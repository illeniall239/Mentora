# Reference solution: only used by scripts/verify-exercises.mjs. Shown in the Breakdown only after the Learner's own solution passes.
from typing import Callable

import numpy as np


def _softmax_rows(S: np.ndarray) -> np.ndarray:
    e = np.exp(S - S.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def self_attention(X: np.ndarray, params: dict[str, np.ndarray], causal: bool = False) -> np.ndarray:
    Q, K, V = X @ params["W_q"], X @ params["W_k"], X @ params["W_v"]
    S = Q @ K.T / np.sqrt(Q.shape[1])
    if causal:
        S = np.where(np.triu(np.ones_like(S, dtype=bool), k=1), -np.inf, S)
    return _softmax_rows(S) @ V


def permute_rows(X: np.ndarray, perm: list[int]) -> np.ndarray:
    if sorted(perm) != list(range(X.shape[0])):
        raise ValueError("perm must be a permutation of range(n)")
    return X[list(perm)].copy()


def is_permutation_equivariant(
    fn: Callable[[np.ndarray], np.ndarray], X: np.ndarray, perm: list[int], atol: float = 1e-6
) -> bool:
    return bool(np.allclose(fn(permute_rows(X, perm)), permute_rows(fn(X), perm), atol=atol))
