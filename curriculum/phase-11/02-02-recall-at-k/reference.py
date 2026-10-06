# Reference solution: only used by scripts/verify-exercises.mjs. Shown in the Breakdown only after the Learner's own solution passes.
import numpy as np


def _recall_rows(sim: np.ndarray, k: int) -> float:
    """Fraction of rows whose diagonal entry ranks in the top k, ties going to the lower index."""
    n = sim.shape[0]
    true_scores = np.diag(sim)[:, None]
    better = sim > true_scores  # strictly better entries always outrank the true one
    index = np.arange(n)
    earlier_tie = (sim == true_scores) & (index[None, :] < index[:, None])
    rank = better.sum(axis=1) + earlier_tie.sum(axis=1)
    return float(np.mean(rank < k))


def recall_at_k(sim_matrix: np.ndarray, k: int) -> tuple[float, float]:
    if sim_matrix.ndim != 2 or sim_matrix.shape[0] != sim_matrix.shape[1]:
        raise ValueError("sim_matrix must be square (N, N)")
    n = sim_matrix.shape[0]
    if n == 0:
        raise ValueError("sim_matrix must hold at least one pair")
    if not isinstance(k, int) or isinstance(k, bool) or k < 1 or k > n:
        raise ValueError("k must be an integer in [1, N]")
    return _recall_rows(sim_matrix, k), _recall_rows(sim_matrix.T, k)
