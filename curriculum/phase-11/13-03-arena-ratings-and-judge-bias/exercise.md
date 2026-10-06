# Arena ratings and judge position bias

Topic: 13. Evaluating generative models
Difficulty: 3 of 3

## Problem

When there is no gold answer, evaluation becomes a tournament: show two outputs, record which one won, repeat. Two things then have to be built. First, turn a pile of pairwise wins into one number per model — the Bradley–Terry fit behind the LMSYS arena, where `P(i beats j) = p_i / (p_i + p_j)`. Second, check that the judge is not simply preferring whatever it was shown first, which LLM judges demonstrably do.

Pure Python, no numpy.

### `bradley_terry_ratings(matches: list[tuple[str, str]], iters: int) -> dict[str, float]`

`matches[m]` is `(winner, loser)`. Let `W_i` be `i`'s total wins and `N_ij` the number of matches `i` played against `j` in either direction. Start every player at `p_i = 1.0` and run exactly `iters` rounds of this minorization-maximization update:

```
p_i_new  =  W_i  /  Σ_{j ≠ i}  N_ij / (p_i + p_j)
```

Every player is updated **simultaneously** from the previous round's values (the right-hand side only ever reads the old `p`), and at the end of each round every `p_i` is divided by `Σ_k p_k_new`, so the returned ratings sum to 1. The normalization is not cosmetic: only ratios are identified, `p` and `2p` describe the same tournament.

Return `{name: rating}` for every player appearing in `matches`.

Raise `ValueError` if `matches` is empty, if `iters < 1`, if any match has the same name twice, or if some player has **no wins or no losses** — with an unbeaten player the likelihood has no maximum, the rating runs off to infinity, and real arenas handle that with a prior rather than by pretending.

### `judge_position_swap_agreement(a_first: list[str], b_first: list[str]) -> float`

The same `N` comparisons judged twice: in `a_first` candidate A was shown first, in `b_first` candidate B was shown first. Every verdict is `"A"`, `"B"` or `"tie"`, naming the **candidate**, never the slot. Return the fraction of comparisons where the two verdicts are identical — the judge's self-agreement under a position swap. A judge that always picks whatever came first scores 0.0; a judge immune to order scores 1.0.

### `judge_win_rate(a_first: list[str], b_first: list[str]) -> float`

Candidate A's win rate, with a tie worth half a win and the **two orderings averaged**:

```
rate(verdicts) = (#A + 0.5 · #tie) / N
win_rate       = (rate(a_first) + rate(b_first)) / 2
```

Dropping the ties, or reporting only `rate(a_first)`, gives a different and wrong number; both are tested.

Both verdict functions raise `ValueError` if the lists differ in length, are empty, or contain a label outside `{"A", "B", "tie"}`.

## Examples

```
bradley_terry_ratings([("A","B"), ("A","B"), ("A","B"), ("B","A")], 50)
    → {"A": 0.75, "B": 0.25}            A won 3 of 4; P(A beats B) = 0.75 / (0.75 + 0.25)

bradley_terry_ratings([("A","B"), ("B","C"), ("C","A")], 200)
    → {"A": 1/3, "B": 1/3, "C": 1/3}    rock-paper-scissors: one number per model cannot express it

bradley_terry_ratings([("A","B"), ("A","B"), ("B","C"), ("B","C"), ("C","A")], 200)
    → {"A": 0.5161117..., "B": 0.3043792..., "C": 0.1795090...}

bradley_terry_ratings([("A","B")], 10)   → ValueError   B never wins, A never loses

judge_position_swap_agreement(["A","A","A","A"], ["B","B","B","B"])  → 0.0   pure position bias
judge_position_swap_agreement(["A","A","A","A"], ["A","A","A","A"])  → 1.0
judge_position_swap_agreement(["tie","A"], ["tie","B"])              → 0.5

judge_win_rate(["A","A","A","A"], ["B","B","B","B"])   → 0.5   the bias cancels, and hides itself
judge_win_rate(["tie","A"], ["tie","A"])               → 0.75  (dropping ties would say 1.0)
judge_win_rate(["A","A"], ["B","A"])                   → 0.75  (a_first alone would say 1.0)
```

## Constraints

- Up to 50 players, 5000 matches, 2000 iterations; up to 5000 comparisons. Absolute tolerance 1e-6.
- The result must not depend on the order of `matches`.
- Pure Python. No optimizer library, no `scipy`.

## Hints

1. Before coding the update: what are `W_i` and `N_ij` for the four-match example, and what does one round starting from `p = (1, 1)` give? Does a second round change it?
2. Why must the right-hand side read the previous round's `p` for *every* player? What breaks, silently, if you update the dictionary in place as you loop?
3. The ratings are only defined up to a common factor. What would the loop do without the normalization step, and what does that tell you about comparing two arenas' raw numbers?
4. A judge picks whichever answer is shown first, every time. Write down `a_first` and `b_first`, then the win rate and the agreement. Which of the two numbers tells you anything, and what does that say about reporting win rates alone?

## Explain-back

- The rock-paper-scissors tournament gives all three models the same rating. Is the fit wrong, or is the question wrong? What is a one-dimensional rating assuming about preferences?
- A judge with heavy position bias can still produce a 0.5 win rate that looks perfectly fair. What exactly does averaging the two orderings fix, and what does it leave broken?
- Name two more biases an LLM judge has beyond position, and say which one gets worse when the two candidates come from the same model family.
- An arena rating is a ranking on *the arena's prompt distribution*. What would you need to change to make it predict performance on your own app, and why does a leaderboard jump of 10 points usually not survive that change?
