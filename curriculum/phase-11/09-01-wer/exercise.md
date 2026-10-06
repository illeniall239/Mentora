# Word error rate

Topic: 9. Speech recognition and Whisper
Difficulty: 2 of 3

## Problem

Every ASR paper reports WER, and it is nothing more than a word-level edit distance divided by the length of the reference. Write it in plain Python.

`wer(ref: str, hyp: str) -> tuple[int, int, int, int, float]` returns `(S, D, I, N, rate)`:

- Split both strings on whitespace (`str.split()`), so runs of spaces and newlines collapse and tokens are **whole words**. Comparison is exact: no lower-casing, no punctuation stripping. `N` is the number of words in the **reference**.
- Align the two word sequences with a Levenshtein DP where a substitution, a deletion and an insertion each cost 1, and a match costs 0. From the optimal alignment count:
  - `S` — reference words replaced by a different word,
  - `D` — reference words missing from the hypothesis,
  - `I` — extra words in the hypothesis that match nothing in the reference.
- `rate = (S + D + I) / N`.

`S + D + I` must equal the edit distance. When more than one optimal alignment exists, prefer a substitution (or match) over a deletion, and a deletion over an insertion, so the split of a given total is deterministic.

Raise `ValueError` if the reference has no words — you cannot divide by zero, and "the model transcribed silence correctly" is not a WER.

Note what the formula does **not** contain: nothing caps it at 1. A hypothesis longer than the reference can contribute unboundedly many insertions, so a hallucinating model can score a WER of 300%, and that is the number ASR engineers actually look for on silence and music.

Plain Python: no `jiwer`, no `Levenshtein`, no `difflib`, no numpy.

## Examples

```
wer("the cat sat on the mat", "the cat sat on the mat")  -> (0, 0, 0, 6, 0.0)
wer("the cat sat on the mat", "the cat sat on a mat")    -> (1, 0, 0, 6, 0.1666...)   substitution
wer("the cat sat on the mat", "the cat sat the mat")     -> (0, 1, 0, 6, 0.1666...)   deletion
wer("the cat sat on the mat", "the cat sat on on the mat") -> (0, 0, 1, 6, 0.1666...) insertion

wer("i love deep learning", "i loved deep learning a lot") -> (1, 0, 2, 4, 0.75)
wer("hello", "hello hello hello")  -> (0, 0, 2, 1, 2.0)    200%: the decoder ran away
wer("one two three", "")           -> (0, 3, 0, 3, 1.0)    nothing transcribed
wer("", "anything")                -> ValueError
```

## Constraints

- Plain Python; sequences up to about 300 words, so an O(len(ref) * len(hyp)) table is fine.
- `S`, `D`, `I`, `N` are `int`; `rate` is a `float`, compared to 1e-9.
- The rate is not clamped: values above 1.0 are correct and expected.

## Hints

1. Your DP table has `len(ref) + 1` rows and `len(hyp) + 1` columns. What do the first row and the first column mean before any word has been compared?
2. Three moves lead into cell `(i, j)`. Which one leaves the reference word unconsumed, and which one leaves the hypothesis word unconsumed? Name them against `D` and `I` before you write the code.
3. The minimum cost alone does not tell you *which* operations made it up. What do you have to store, or re-walk, to split the total into `S`, `D` and `I`?
4. `wer("hello", "hello hello hello")` is 2.0. Which part of the formula has no upper bound, and which part is fixed by the reference alone?

## Explain-back

- A transcript scores 0% WER after you lower-case both strings and strip punctuation, and 40% before. Which number would you put in a paper, and what has to be said alongside it?
- Whisper transcribes four seconds of silence as "Thank you for watching!". Compute the WER against an empty-ish reference and explain what in the architecture produces that sentence.
- Why is WER on characters (CER) used for Chinese and Japanese, and what does that change about the denominator?
- Two systems both score 15% WER: one makes 15 substitutions, the other 15 insertions in one burst. Why might you ship one and not the other?
