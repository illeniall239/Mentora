def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimator: 1 - C(n-c, k) / C(n, k)."""
    raise NotImplementedError


def pass_at_k_from_results(results: list[bool], k: int) -> float:
    """pass@k from per-sample outcomes: n = len(results), c = number of True entries."""
    raise NotImplementedError
