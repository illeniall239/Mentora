# Reference solution: only used by scripts/verify-exercises.mjs. Shown in the Breakdown only after the Learner's own solution passes.
def elo_update(ra: float, rb: float, outcome: float, k: float) -> tuple[float, float]:
    if not 0.0 <= outcome <= 1.0:
        raise ValueError("outcome must be in [0, 1]")
    if k <= 0:
        raise ValueError("k must be positive")
    ea = 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))
    eb = 1.0 - ea
    return ra + k * (outcome - ea), rb + k * ((1.0 - outcome) - eb)
