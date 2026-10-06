def elo_update(ra: float, rb: float, outcome: float, k: float) -> tuple[float, float]:
    """One Elo update; outcome is A's score (1 win, 0.5 tie, 0 loss). Returns (new_ra, new_rb)."""
    raise NotImplementedError
