# Elo update for a model arena

Topic: 7. Evaluating image models
Difficulty: 2 of 3

## Problem

When metrics stop correlating with what people actually prefer, evaluation moves to an arena: show two models' outputs for the same prompt, let a human pick a winner, and update both ratings. That update is Elo, unchanged from chess. Write it in plain Python.

`elo_update(ra: float, rb: float, outcome: float, k: float) -> tuple[float, float]`

```
ea = 1 / (1 + 10 ** ((rb - ra) / 400))      expected score for A
eb = 1 - ea                                 expected score for B
new_ra = ra + k * (outcome - ea)
new_rb = rb + k * ((1 - outcome) - eb)
```

- `outcome` is **A's score**: `1.0` if A won, `0.0` if B won, `0.5` for a tie. Any float in `[0, 1]` is allowed. Raise `ValueError` if it falls outside that range.
- `k` is the step size (32 is the usual starting value). Raise `ValueError` if `k <= 0`.
- Return `(new_ra, new_rb)` as a tuple of floats. The pair is **zero-sum**: `new_ra + new_rb` equals `ra + rb` for every input, so the arena's total rating never moves. It is also symmetric: swapping the two players and replacing `outcome` with `1 - outcome` must give the swapped result.

The 400 and the base 10 are a convention, not physics: a 400-point gap means the stronger model is expected to win 10 games out of 11. A rating only has meaning relative to the other ratings in the same pool.

Plain Python with `math` if you want it; no numpy, no torch.

## Examples

```
elo_update(1500, 1500, 1.0, 32)   -> (1516.0, 1484.0)      even match, A wins
elo_update(1500, 1500, 0.5, 32)   -> (1500.0, 1500.0)      even match, tie: nothing moves
elo_update(1500, 1500, 0.0, 32)   -> (1484.0, 1516.0)

elo_update(1600, 1200, 1.0, 32)   -> (1602.909..., 1197.090...)   favourite wins, tiny gain
elo_update(1600, 1200, 0.0, 32)   -> (1570.909..., 1229.090...)   upset, big swing

elo_update(1500, 1500, 1.5, 32)   -> ValueError
elo_update(1500, 1500, 1.0, 0)    -> ValueError
```

## Constraints

- Plain Python floats; tolerance 1e-9.
- Ratings in roughly `[0, 4000]`, `k` in `(0, 200]`.
- No state: the function takes two ratings and returns two ratings.

## Hints

1. Which rating goes in the numerator of the exponent? Check your sign against a case you know the answer to: the favourite at 1600 against 1200 should expect far more than half a point.
2. `ea + eb` must equal 1. Does your `eb` satisfy that by construction, or did you compute it with a second formula that might disagree?
3. Add the two rating changes together and simplify symbolically. What has to cancel for the update to be zero-sum?
4. Why does a tie between equal players move nothing, while a tie between a 1600 and a 1200 moves both ratings? Which term differs?

## Explain-back

- Model A has an Elo of 1250 and model B of 1100 in two different arenas. Why does that comparison mean nothing, and what would you need to make it mean something?
- A new model enters an arena and wins its first five matches against mid-ranked opponents. Why is its rating still a poor estimate, and what do real arenas do about it?
- Elo converts a rating gap into a win probability. What does an Elo of 1500 say about whether the images are actually *good*?
- Pairwise preference is slow and expensive compared with FID. What does it capture that a feature-space distance cannot?
