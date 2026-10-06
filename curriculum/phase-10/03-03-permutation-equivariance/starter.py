from typing import Callable

import numpy as np


def self_attention(X: np.ndarray, params: dict[str, np.ndarray], causal: bool = False) -> np.ndarray:
    """Project X (n, d) to Q, K, V with W_q, W_k, W_v and return softmax(Q K^T / sqrt(d_k)) V, (n, d_v)."""
    raise NotImplementedError


def permute_rows(X: np.ndarray, perm: list[int]) -> np.ndarray:
    """New array whose row i is X[perm[i]]; ValueError unless perm is a permutation of range(n)."""
    raise NotImplementedError


def is_permutation_equivariant(
    fn: Callable[[np.ndarray], np.ndarray], X: np.ndarray, perm: list[int], atol: float = 1e-6
) -> bool:
    """Whether fn(permute_rows(X, perm)) equals permute_rows(fn(X), perm) within atol."""
    raise NotImplementedError
