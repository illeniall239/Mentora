# Bigram baseline

Topic: 6. The language-modelling objective and a tiny GPT
Difficulty: 1 of 3

## Problem

Before training a neural network, build the model every language-modelling project should beat first: a **bigram** table that predicts the next token from the current one alone. Pure Python and `math` only.

- `train_bigram(ids: list[int], vocab: int) -> list[list[float]]` counts every adjacent pair `(ids[t], ids[t + 1])` and returns a `vocab × vocab` table `P` where `P[a][b]` is the probability that token `b` follows token `a`, with **add-one (Laplace) smoothing**: `P[a][b] = (count(a, b) + 1) / (count(a, ·) + vocab)`, where `count(a, ·)` is the number of pairs that start with `a`. Every row sums to 1 and every entry is strictly positive, including rows for tokens that never appear. Raise `ValueError` if any id is outside `0 … vocab − 1`.
- `perplexity(model: list[list[float]], ids: list[int]) -> float` scores a sequence with the table: `exp` of the mean over `t = 1 … len(ids) − 1` of `−ln P[ids[t − 1]][ids[t]]`. Raise `ValueError` if `ids` has fewer than 2 tokens.

A uniform model has perplexity `vocab` on any sequence; a bigram table trained on a sequence with real structure must do better on that sequence and on a fresh sequence drawn from the same source.

## Examples

```
P = train_bigram([0, 1, 0, 1, 0], 2)
P[0]   → [1/4, 3/4]     0 is followed by 1 twice, never by 0: (0+1)/(2+2), (2+1)/(2+2)
P[1]   → [3/4, 1/4]
P = train_bigram([0, 0], 3)
P[2]   → [1/3, 1/3, 1/3]  token 2 never appears: smoothing alone

perplexity([[0.5, 0.5], [0.5, 0.5]], [0, 1, 1, 0])   → 2.0
perplexity(train_bigram([0, 1, 0, 1, 0], 2), [0, 1, 0, 1, 0])  → 4/3 ≈ 1.3333
```

## Constraints

- `ids` has at most 20 000 tokens; `vocab` is at most 64.
- Pure Python and `math` only: no numpy or torch.

## Hints

1. How many pairs does a sequence of `n` tokens contain, and which two tokens make up pair number `t`?
2. Without smoothing, what probability does the table give a pair it never saw, and what does `−ln` of that do to the perplexity of a new sequence?
3. After adding one to every count in a row, what must you add to the row's total so it still sums to 1?
4. Perplexity is `exp` of an average of `−ln P`. Should you average the probabilities or the log-probabilities, and why does it matter?

## Explain-back

- What does perplexity 4/3 on the training sequence mean in words ("the model is as unsure as choosing between … options")? Why is it below 2 even though the model was smoothed?
- The bigram table scored the training data well. Would you trust that number as a measure of the model? What sequence should you score instead, and what does a big gap between the two tell you?
- A GPT trained on the same data gets a lower perplexity than this table. What does it condition on that the bigram model cannot?
- Where does add-one smoothing hurt most: for a token seen ten thousand times or for one seen twice? What would a neural model do instead of counting?
