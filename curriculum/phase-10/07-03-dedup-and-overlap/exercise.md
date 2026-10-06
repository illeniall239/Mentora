# Deduplication and n-gram overlap

Topic: 7. Scaling and pretraining
Difficulty: 2 of 3

## Problem

A pretraining pipeline throws away duplicated text before training. Write the two simplest tools it uses. Pure Python (`hashlib` allowed).

- `dedup_exact(docs: list[str]) -> list[str]` returns the documents with exact duplicates removed, keeping the **first** occurrence of each and preserving order. Compare documents by a hash of their UTF-8 bytes (e.g. `hashlib.sha256`) rather than keeping every full text in a set: the whole point of hashing is that the deduplication table stays small when the corpus does not. Two documents are duplicates only if they are byte-for-byte identical; `"Hello"` and `"hello"` are different.
- `ngram_overlap(a: str, b: str, n: int) -> float` measures near-duplication as the **Jaccard similarity** of the two documents' sets of word `n`-grams: split each text on whitespace into words, form the set of every run of `n` consecutive words, and return `|A ∩ B| / |A ∪ B|`. Return `0.0` when either document has fewer than `n` words. Raise `ValueError` if `n < 1`.

## Examples

```
dedup_exact(["a b", "c", "a b", "d", "c"])   → ["a b", "c", "d"]
dedup_exact(["Hello", "hello"])              → ["Hello", "hello"]

ngram_overlap("the cat sat on the mat", "the cat sat on the mat", 2)  → 1.0
ngram_overlap("the cat sat on the mat", "the dog sat on the mat", 2)
    A = {the cat, cat sat, sat on, on the, the mat}
    B = {the dog, dog sat, sat on, on the, the mat}
    → 3 / 7 ≈ 0.4286
ngram_overlap("a b c", "d e f", 1)          → 0.0
ngram_overlap("a b", "a b c", 3)            → 0.0     (a has no 3-grams)
```

## Constraints

- Up to 10 000 documents of up to 5 000 characters; `n` is 1 to 10.
- Results are compared with `assertAlmostEqual` to 9 places.
- Pure Python only.

## Hints

1. What do you need to remember about each document you have kept so far to decide whether the next one is a repeat, and how much memory would that take for a web-scale corpus if you stored the text itself?
2. If the words are `w[0], …, w[k − 1]`, which slices are the `n`-grams, and how many are there? What happens to that count when `k < n`?
3. The n-grams must go into a `set`. Which Python type can a run of words be so it is hashable?
4. Jaccard is `|A ∩ B| / |A ∪ B|`. Which set operators give those, and when could the denominator be zero?

## Explain-back

- Two crawled pages differ by one changed word. Does `dedup_exact` catch that? What does `ngram_overlap` return for them, and how would you turn it into a keep/drop decision?
- Why does training twice on the same document hurt more than training on two different documents of the same quality? What does the model do with a memorised page at sampling time?
- What else besides deduplication does a real data pipeline do between "web crawl" and "training tokens"? Name two filters and the risk each one carries.
- After all that filtering, does the model "memorise the internet"? What happens to a fact that appears once in a trillion tokens?
