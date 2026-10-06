def greedy_path_prob(dists: dict[tuple[str, ...], dict[str, float]], path: list[str]) -> float:
    """Product of p(path[i] | path[:i]) read from the table; 0.0 if a prefix or token is missing."""
    raise NotImplementedError


def greedy_path(dists: dict[tuple[str, ...], dict[str, float]]) -> list[str]:
    """From () pick the most probable token (ties: first listed) until the prefix has no entry."""
    raise NotImplementedError
