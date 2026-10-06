# Self-consistency score

Topic: 13. Hallucination, mechanically
Difficulty: 1 of 3

## Problem

A model that knows a fact gives the same answer every time you sample; a model that is guessing gives different answers. That is the idea behind sampling-based hallucination detection (SelfCheckGPT-style): ask the same question several times at temperature > 0 and measure agreement. Pure Python.

- `normalize(answer: str) -> str` strips leading and trailing whitespace, lowercases, and collapses any run of internal whitespace to one space, so `" Paris "` and `"paris"` count as the same answer.
- `consistency_score(samples: list[str]) -> float` normalizes every sample, finds the **majority answer** (the most frequent normalized string; ties go to the one that appears **first** in `samples`), and returns the fraction of samples equal to it. Raise `ValueError` on an empty list.
- `flag_inconsistent(samples: list[str], threshold: float) -> bool` returns `True` when `consistency_score(samples) < threshold` (strictly below), meaning the answer should be treated as unreliable. Raise `ValueError` if `threshold` is outside `[0, 1]`.

## Examples

```
consistency_score(["Paris", "paris", " Paris "])          → 1.0
consistency_score(["Paris", "Lyon", "Paris", "Marseille"]) → 0.5
consistency_score(["1905", "1915", "1921"])               → 0.3333...
consistency_score(["a", "b", "b", "a"])                   → 0.5      tie: "a" appears first, same score
consistency_score([])                                     → ValueError

flag_inconsistent(["Paris", "Lyon", "Paris", "Marseille"], 0.6) → True
flag_inconsistent(["Paris", "Lyon", "Paris", "Marseille"], 0.5) → False   0.5 is not below 0.5
```

## Constraints

- At most 1 000 samples, each at most 1 000 characters.
- Pure Python; `collections.Counter` is fine.

## Hints

1. Which string methods turn `"  The  Answer "` into `"the answer"`? What does `" ".join(s.split())` do to internal whitespace?
2. How do you count occurrences of each normalized answer, and how do you pick the largest count while still respecting "first seen wins" on ties?
3. Is the score "how many samples match the majority" or "how many distinct answers there are"? Which one goes down as the model gets less sure?
4. Should the flag fire when the score equals the threshold? Re-read the definition before choosing `<` or `<=`.

## Explain-back

- Why does agreement across samples say anything about truth? Give an example where the model is consistently wrong and the score is 1.0.
- Would sampling at temperature 0 make this check useful? What would every sample look like, and what does that say about "temperature 0 prevents hallucination"?
- The method costs N forward passes per question. When is that worth it, and what cheaper signals (token log-probs, calibrated confidence) trade against it?
- How does retrieval-backed checking differ from consistency checking? Which kinds of errors does each catch, and which does neither?
