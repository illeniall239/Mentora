def bradley_terry_ratings(matches: list[tuple[str, str]], iters: int) -> dict[str, float]:
    """Fit Bradley-Terry strengths from (winner, loser) pairs by `iters` MM rounds, normalized to sum 1."""
    raise NotImplementedError


def judge_position_swap_agreement(a_first: list[str], b_first: list[str]) -> float:
    """Fraction of comparisons where the judge gave the same verdict with the candidates swapped."""
    raise NotImplementedError


def judge_win_rate(a_first: list[str], b_first: list[str]) -> float:
    """Candidate A's win rate, ties worth half, averaged over both presentation orders."""
    raise NotImplementedError
