# Reference solution: only used by scripts/verify-exercises.mjs. Shown in the Breakdown only after the Learner's own solution passes.
KINDS = ("bidirectional", "causal", "prefix")


def make_mask(n: int, kind: str, prefix_len: int | None = None) -> list[list[bool]]:
    if not isinstance(n, int) or isinstance(n, bool) or n < 1:
        raise ValueError("n must be a positive integer")
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}")
    if kind == "prefix":
        if prefix_len is None:
            raise ValueError('kind "prefix" needs a prefix_len')
        if not isinstance(prefix_len, int) or isinstance(prefix_len, bool) or not 0 <= prefix_len <= n:
            raise ValueError("prefix_len must be an integer in 0..n")

    if kind == "bidirectional":
        return [[True] * n for _ in range(n)]
    if kind == "causal":
        return [[j <= i for j in range(n)] for i in range(n)]
    return [[j < prefix_len or j <= i for j in range(n)] for i in range(n)]
