# Nearest neighbours and analogy

Topic: 2. Embeddings revisited: static, contextual and tied
Difficulty: 2 of 3

## Problem

Static word vectors are compared by direction, not by length. Write two numpy functions over an embedding table `table` of shape `(V, D)` and a `vocab` list of `V` words, where `table[i]` is the vector of `vocab[i]`.

- `nearest(query: np.ndarray, table: np.ndarray, k: int) -> list[int]` returns the indices of the `k` rows with the highest **cosine similarity** to `query` (shape `(D,)`), most similar first. Cosine similarity is `a·b / (‖a‖ ‖b‖)`; scaling `query` by any positive constant must not change the result. `1 ≤ k ≤ V`. No row and no query is the zero vector.
- `analogy(a: str, b: str, c: str, table: np.ndarray, vocab: list[str]) -> str` answers "`a` is to `b` as `c` is to ?" with the word whose vector is nearest (by cosine) to `table[b] − table[a] + table[c]`, **never returning `a`, `b` or `c`** even when one of them is the closest vector. Raise `ValueError` if any of the three words is not in `vocab`.

No loops over rows for the similarity itself: compute all `V` cosines with array ops.

## Examples

```
vocab = ["king", "queen", "man", "woman"]
table = [[1, 1, 0], [1, -1, 0], [0, 1, 0], [0, -1, 0]]
nearest([1, -0.9, 0], table, 2)                   → [1, 3]        queen, then woman
nearest([2, -1.8, 0], table, 2)                   → [1, 3]        same direction, same answer
analogy("man", "king", "woman", table, vocab)     → "queen"       king − man + woman = [1, -1, 0]
analogy("king", "king", "queen", table, vocab)    → "woman"       target is queen's own vector, but queen is an input
analogy("man", "king", "duke", table, vocab)      → ValueError
```

## Constraints

- `V ≤ 100`, `D ≤ 16`. float64 numpy.
- Ties do not occur in the tests.
- numpy only.

## Hints

1. Which two things does the raw dot product `table @ query` reward: agreement in direction, or also length? Which of the two do you want, and what do you divide by to remove the other?
2. `np.linalg.norm` has an `axis` argument. What shape must the per-row norms have so they divide `table @ query` row by row?
3. Which numpy function gives you the indices that would sort an array, and how do you get the *largest* `k` first rather than the smallest?
4. For `analogy`, you already have the ranked indices. Which ones do you skip, and why is skipping better than removing rows from `table` before ranking?

## Explain-back

- Why does every word-vector method use cosine rather than Euclidean distance? What information does a vector's length carry in word2vec-style training?
- Word2vec's "king − man + woman ≈ queen" is usually reported with the input words excluded. What would the answer often be without that exclusion, and what does that say about how impressive the result is?
- Are these static vectors what an LLM uses inside its layers? Which of the two would distinguish "bank" (river) from "bank" (money)?
- Could you read coordinate 0 of the table as "royalty" here? Why does a learned table almost never have interpretable axes, even when this hand-made one does?
