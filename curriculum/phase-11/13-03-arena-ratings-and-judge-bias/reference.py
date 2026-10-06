# Reference solution: only used by scripts/verify-exercises.mjs. Shown in the Breakdown only after the Learner's own solution passes.
VERDICTS = {"A", "B", "tie"}


def bradley_terry_ratings(matches: list[tuple[str, str]], iters: int) -> dict[str, float]:
    if not matches:
        raise ValueError("need at least one match")
    if iters < 1:
        raise ValueError("iters must be at least 1")
    if any(winner == loser for winner, loser in matches):
        raise ValueError("a model cannot play itself")

    players = sorted({name for match in matches for name in match})
    wins = {name: 0 for name in players}
    losses = {name: 0 for name in players}
    played = {name: {} for name in players}
    for winner, loser in matches:
        wins[winner] += 1
        losses[loser] += 1
        played[winner][loser] = played[winner].get(loser, 0) + 1
        played[loser][winner] = played[loser].get(winner, 0) + 1
    for name in players:
        if wins[name] == 0 or losses[name] == 0:
            raise ValueError(f"{name} has no wins or no losses, so the fit is degenerate")

    p = {name: 1.0 for name in players}
    for _ in range(iters):
        nxt = {}
        for name in players:
            denominator = sum(n / (p[name] + p[other]) for other, n in played[name].items())
            nxt[name] = wins[name] / denominator
        total = sum(nxt.values())
        p = {name: value / total for name, value in nxt.items()}
    return p


def _check_verdicts(a_first: list[str], b_first: list[str]) -> None:
    if len(a_first) != len(b_first):
        raise ValueError("both orderings must cover the same comparisons")
    if not a_first:
        raise ValueError("need at least one comparison")
    if any(v not in VERDICTS for v in a_first + b_first):
        raise ValueError("verdicts must be 'A', 'B' or 'tie'")


def judge_position_swap_agreement(a_first: list[str], b_first: list[str]) -> float:
    _check_verdicts(a_first, b_first)
    agreed = sum(1 for x, y in zip(a_first, b_first) if x == y)
    return agreed / len(a_first)


def _rate(verdicts: list[str]) -> float:
    return sum(1.0 if v == "A" else 0.5 if v == "tie" else 0.0 for v in verdicts) / len(verdicts)


def judge_win_rate(a_first: list[str], b_first: list[str]) -> float:
    _check_verdicts(a_first, b_first)
    return (_rate(a_first) + _rate(b_first)) / 2
