# Reference solution: only used by scripts/verify-exercises.mjs. Shown in the Breakdown only after the Learner's own solution passes.
import numpy as np


def reinforce_grad(logits: np.ndarray, actions: np.ndarray, returns: np.ndarray, baseline: float = 0.0) -> np.ndarray:
    logits = np.asarray(logits, dtype=float)
    actions = np.asarray(actions)
    returns = np.asarray(returns, dtype=float)
    if logits.ndim != 2 or logits.shape[0] == 0:
        raise ValueError("logits must be a non-empty (n, A) array")
    n, n_actions = logits.shape
    if actions.shape != (n,) or returns.shape != (n,):
        raise ValueError("actions and returns must both have shape (n,)")
    if actions.min() < 0 or actions.max() >= n_actions:
        raise ValueError("every action must be in [0, A)")

    shifted = logits - logits.max(axis=1, keepdims=True)  # stable softmax
    exp = np.exp(shifted)
    p = exp / exp.sum(axis=1, keepdims=True)

    grad = -p  # d log pi(a) / d logits = onehot(a) - p
    grad[np.arange(n), actions] += 1.0
    advantage = returns - baseline  # the baseline shifts the weight, not the log-prob
    return grad * advantage[:, None] / n
