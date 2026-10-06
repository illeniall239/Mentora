# Exact match and token F1

Topic: 13. Evaluating generative models
Difficulty: 2 of 3

## Problem

The two metrics behind every extractive QA leaderboard, and the first two you should reach for when your app has a right answer. Exact match asks "is this string the gold string, once both are tidied up?". Token F1 gives partial credit when the model said the right thing with extra words around it. Both live or die by the normalization step — without it, `"The answer."` and `"answer"` disagree.

Pure Python with `string` and `collections`; no numpy, no NLTK, no `evaluate`/`datasets` library.

- `normalize_answer(s: str) -> str` — the SQuAD normalizer, in this order:
  1. lowercase;
  2. **delete** every character in `string.punctuation` (delete, do not replace with a space, so `"don't"` becomes `"dont"` and `"co-op"` becomes `"coop"`);
  3. drop the whole tokens `a`, `an` and `the`;
  4. collapse any run of whitespace to one space and strip the ends.
- `exact_match(pred: str, gold: str) -> bool` — `normalize_answer(pred) == normalize_answer(gold)`.
- `token_f1(pred: str, gold: str) -> float` — split both normalized strings on whitespace, then:
  - if either token list is empty, return `1.0` when both are empty and `0.0` otherwise;
  - `common` = the size of the **multiset** intersection, i.e. `Σ_t min(count_pred(t), count_gold(t))` — a token the model repeats twice only counts twice if the gold has it twice;
  - if `common == 0`, return `0.0`;
  - `precision = common / len(pred_tokens)`, `recall = common / len(gold_tokens)`, and return `2 · precision · recall / (precision + recall)`.

Token F1 is a bag of words: word order is invisible to it, and it is symmetric in its two arguments. Both facts are tested, and both are reasons not to trust it on open-ended generation.

## Examples

```
normalize_answer("The Answer.")         → "answer"
normalize_answer("  AN  co-op   ")      → "coop"
normalize_answer("the")                 → ""

exact_match("The Answer.", "answer")    → True
exact_match("42", "forty-two")          → False
exact_match("Paris, France", "paris france") → True

token_f1("a red car", "red car")        → 1.0
token_f1("car red", "red car")          → 1.0        word order is invisible
token_f1("the quick brown fox", "a quick red fox") → 0.6666666...   2 shared of 3 and 3
token_f1("the cat the cat", "cat")      → 0.6666666...   two "cat"s against one, not a set overlap of 1.0
token_f1("blue car", "red bus")         → 0.0
token_f1("the", "cat")                  → 0.0          pred normalizes away to nothing
token_f1("", "")                        → 1.0
```

## Constraints

- Strings up to 10 000 characters. Absolute tolerance 1e-9 on F1.
- `collections.Counter` is allowed and is the short route to the multiset intersection.

## Hints

1. Build `normalize_answer` as four small steps and check each on `"The Answer."` before combining them. Does step 2 run before or after step 3, and what happens to `"The"` if you get that order wrong?
2. For the multiset overlap, what does `Counter(pred) & Counter(gold)` give you, and how is it different from `set(pred) & set(gold)` on `["cat", "cat"]` against `["cat"]`?
3. Precision and recall divide by different denominators. Which one punishes a model that answers with the whole sentence, and which punishes one that answers with a single word?
4. What should `token_f1("the", "cat")` be, and which branch of your code decides it — the empty check or the `common == 0` check? Does it matter which fires first?

## Explain-back

- A model answers "The capital is Paris" when the gold is "Paris". What do EM and F1 each say, and which one is right about whether the user got what they asked for?
- Token F1 scores "car red" and "red car" identically. Give a question where that is harmless and one where it inverts the meaning.
- These metrics need a gold string. Name two generation tasks where writing that gold string is impossible, and say what you would measure instead.
- Deleting punctuation makes `"don't"` and `"dont"` match. Which failures does that choice hide, and when would you want the stricter comparison back?
