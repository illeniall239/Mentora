# Reference solution: only used by scripts/verify-exercises.mjs. Shown in the Breakdown only after the Learner's own solution passes.
def greedy_path_prob(dists: dict[tuple[str, ...], dict[str, float]], path: list[str]) -> float:
    prob = 1.0
    for i, token in enumerate(path):
        dist = dists.get(tuple(path[:i]))
        if dist is None or token not in dist:
            return 0.0
        prob *= dist[token]
    return prob


def greedy_path(dists: dict[tuple[str, ...], dict[str, float]]) -> list[str]:
    if () not in dists:
        raise ValueError("table has no distribution for the empty prefix")
    path: list[str] = []
    while tuple(path) in dists:
        dist = dists[tuple(path)]
        path.append(max(dist, key=dist.get))  # max returns the first maximal key on ties
    return path
