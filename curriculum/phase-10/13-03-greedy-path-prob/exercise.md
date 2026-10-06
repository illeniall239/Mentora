# Greedy path probability

Topic: 13. Hallucination, mechanically
Difficulty: 2 of 3

## Problem

A language model never checks facts: at each step it emits a token from p(next | prefix), and greedy decoding takes the most probable one. Work with a toy conditional table to see how the most *fluent* continuation can be the wrong answer, and how greedy can even miss the most probable sequence. Pure Python.

The table `dists` is a `dict[tuple[str, ...], dict[str, float]]`: for a prefix (a tuple of tokens already emitted, `()` at the start) it gives the next-token distribution as `{token: probability}`. A prefix with **no entry** in `dists` is a finished sequence.

- `greedy_path_prob(dists, path: list[str]) -> float` returns `p(path) = Π_i p(path[i] | path[:i])`, multiplying the probabilities read from the table step by step. If some prefix along the way has no distribution, or a token is missing from its distribution, return `0.0`. An empty path has probability `1.0`.
- `greedy_path(dists) -> list[str]` starts from `()` and repeatedly appends the most probable next token (ties: the token listed **first** in that dict) until it reaches a prefix that has no entry. Return the list of tokens. Raise `ValueError` if `()` itself is not in `dists`.

The tests use two tables. In the first, the question "capital of Australia?" has a fluent, confident, wrong greedy answer ("Sydney") that outranks the correct one ("Canberra"). In the second, greedy picks a first token whose best continuation is worse than the sequence it passed over, so `greedy_path` is **not** the highest-probability path.

## Examples

```
AUS = {
    (): {"Sydney": 0.5, "Canberra": 0.4, "I": 0.1},
    ("Sydney",): {".": 1.0},
    ("Canberra",): {".": 1.0},
    ("I",): {"don't": 0.9, "think": 0.1},
    ("I", "don't"): {"know": 1.0},
}
greedy_path(AUS)                                 → ["Sydney", "."]
greedy_path_prob(AUS, ["Sydney", "."])           → 0.5
greedy_path_prob(AUS, ["Canberra", "."])         → 0.4
greedy_path_prob(AUS, ["I", "don't", "know"])    → 0.09
greedy_path_prob(AUS, ["Melbourne"])             → 0.0
greedy_path_prob(AUS, [])                        → 1.0

TRAP = {
    (): {"a": 0.6, "b": 0.4},
    ("a",): {"x": 0.5, "y": 0.5},
    ("b",): {"z": 1.0},
}
greedy_path(TRAP)                                → ["a", "x"]     probability 0.3
greedy_path_prob(TRAP, ["b", "z"])               → 0.4            higher, but greedy never sees it
```

## Constraints

- Tables have at most 100 prefixes; every path is finite (no cycles).
- Pure Python. Absolute tolerance 1e-9.

## Hints

1. For `path = ["I", "don't", "know"]`, which three prefixes do you look up, and in what order? How do you build `path[:i]` as a tuple key?
2. What should the running product start at, and what does that give for an empty path?
3. In `greedy_path`, which built-in picks the key with the largest value from a dict, and does it return the *first* maximal key on a tie?
4. When does the greedy loop stop? What tells you a sequence is finished in this table format?

## Explain-back

- In the AUS table, "Sydney." has the highest probability and is wrong. What did the training objective reward here, and why would more data about Australia (not a "truth check") be the fix?
- Once greedy emits "Sydney", every later token conditions on it. What is exposure bias, and why does an early wrong token rarely get corrected?
- The TRAP table shows greedy missing the best sequence. What does beam search change, and why does it still not solve hallucination?
- Someone says "just use temperature 0, then it won't make things up". Using the AUS table, explain what temperature 0 does and does not change.
