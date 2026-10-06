# Shifted windows and cross-entropy

Topic: 6. The language-modelling objective and a tiny GPT
Difficulty: 1 of 3

## Problem

A language model is trained to predict the **next** token at every position. Write the two pieces that turn a token sequence into that objective. Pure Python and `math` only.

- `make_xy(ids: list[int], block_size: int, i: int) -> tuple[list[int], list[int]]` returns the input window `x = ids[i : i + block_size]` and its target window `y = ids[i + 1 : i + block_size + 1]`, so that `y[t]` is the token that follows `x[t]`. Both have exactly `block_size` entries. Raise `ValueError` if `i < 0`, `block_size < 1`, or the target window would run past the end of `ids` (`i + block_size + 1 > len(ids)`).
- `cross_entropy(logits: list[list[float]], targets: list[int]) -> float` takes a batch of `B` rows of `V` logits and `B` target indices and returns the mean over the batch of `−log p(targets[b])`, where `p = softmax(logits[b])`. Compute it as `logsumexp(logits[b]) − logits[b][targets[b]]` using the max-subtraction trick, so rows like `[1000.0, 1001.0]` neither overflow nor return `inf`. Raise `ValueError` if `logits` is empty or the lengths of `logits` and `targets` differ.

## Examples

```
make_xy([10, 11, 12, 13, 14], 3, 0)  → ([10, 11, 12], [11, 12, 13])
make_xy([10, 11, 12, 13, 14], 3, 1)  → ([11, 12, 13], [12, 13, 14])
make_xy([10, 11, 12, 13, 14], 3, 2)  → ValueError   (target would need ids[5])

cross_entropy([[0.0, 0.0, 0.0, 0.0]], [2])            → ln 4 ≈ 1.3863
cross_entropy([[0.0, 100.0]], [1])                    → ≈ 0.0
cross_entropy([[1000.0, 1001.0]], [0])                → ln(1 + e) ≈ 1.3133
cross_entropy([[0.0, 0.0], [0.0, 100.0]], [0, 1])     → (ln 2 + 0) / 2 ≈ 0.3466
```

## Constraints

- `ids` has at most 10 000 tokens; `V` and `B` are at most 64.
- Logit values may be as large as ±1e4 in magnitude; the result must stay finite.
- Pure Python and `math` only: no numpy or torch.

## Hints

1. If the model reads `x[t]`, which token is it being asked to predict, and where in `ids` does that token sit relative to `x[t]`? What goes wrong if `y` starts at the same index as `x`?
2. How many tokens past `i` does the pair of windows need in total? Write the inequality that must hold before slicing.
3. `−log softmax(z)[k]` expands to `log Σ_j e^{z_j} − z_k`. Why does `math.exp(1000)` fail, and what can you subtract from every `z_j` without changing the softmax?
4. When every logit is equal, what is `p(target)`, and what does that make the loss for a vocabulary of `V`?

## Explain-back

- Why does `y` have to be `x` shifted by exactly one? What would a model trained with `y = x` learn to do, and why would its loss look wonderfully low?
- A freshly initialised GPT with a 50 257-token vocabulary shows a loss of about 10.8. Where does that number come from, and what would a loss of 3 at step 0 tell you?
- Perplexity is `exp(loss)`. What perplexity does the uniform model have, and how do you read "perplexity 20" in words?
- Is the model ever trained to produce a whole answer at once? What does one training example actually score, and how many predictions does a window of `block_size` tokens contribute?
