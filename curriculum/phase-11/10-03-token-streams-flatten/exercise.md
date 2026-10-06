# Flattening codec token streams

Topic: 10. Audio tokens and neural codecs
Difficulty: 2 of 3

## Problem

An RVQ codec does not emit one sequence, it emits `Q` parallel streams: one index per codebook per frame. A language model consumes a single sequence, so the streams have to be laid out in one line and taken apart again after generation. The simplest layout, and the one you implement here, is **frame-major interleaving**: all `Q` codebook indices of frame 0, then all `Q` of frame 1, and so on.

Pure Python lists, no numpy.

- `flatten_streams(indices: list[list[int]]) -> list[int]` — `indices[q][f]` is codebook `q`'s index for frame `f`. Every stream has the same number of frames `T`. Return a list of length `Q · T` with

  ```
  seq[f · Q + q] = indices[q][f]
  ```

  i.e. the inner loop walks the codebooks of one frame, the outer loop walks the frames.
- `unflatten_streams(seq: list[int], n_codebooks: int) -> list[list[int]]` — the exact inverse: return `n_codebooks` streams, each of length `len(seq) / n_codebooks`, such that `unflatten_streams(flatten_streams(x), len(x)) == x`.

Rules and errors:

- `T = 0` is allowed: `flatten_streams([[], [], []])` is `[]` and `unflatten_streams([], 3)` is `[[], [], []]`.
- `flatten_streams` raises `ValueError` if `indices` is empty (no codebooks) or the streams differ in length.
- `unflatten_streams` raises `ValueError` if `n_codebooks < 1` or `len(seq)` is not a multiple of `n_codebooks` — a sequence that does not divide evenly means a truncated generation, and guessing a split would silently shift every later frame into the wrong codebook.
- Do not modify the inputs; return new lists.

## Examples

```
flatten_streams([[1, 2, 3],
                 [10, 20, 30]])            → [1, 10, 2, 20, 3, 30]
      not [1, 2, 3, 10, 20, 30] — that is one stream after the other, which forces
      the model to predict all of codebook 1 before it has heard frame 0 at all

flatten_streams([[7, 8]])                  → [7, 8]
unflatten_streams([1, 10, 2, 20, 3, 30], 2) → [[1, 2, 3], [10, 20, 30]]
unflatten_streams([], 3)                    → [[], [], []]
unflatten_streams([1, 2, 3], 2)             → ValueError
flatten_streams([[1, 2], [3]])              → ValueError
```

## Constraints

- Up to 8 codebooks and 2000 frames. Values are plain `int`s, compared exactly.
- Time: O(Q · T).

## Hints

1. Given a position `n` in the flat sequence, which frame and which codebook does it belong to? Which of `divmod(n, Q)` and `divmod(n, T)` gives you that?
2. Write the flatten as two nested loops. Which one is outer if frame 0 must be fully emitted before frame 1 starts?
3. For `unflatten_streams`, you know `len(seq)` and `n_codebooks` but not `T`. Where does `T` come from, and what does it mean if the division leaves a remainder?
4. A model generates 100 tokens with `Q = 4` and you keep only the first 99. Which frames survive intact, and what does the last partial frame do to your decode if you keep it?

## Explain-back

- Flattening makes the LM sequence `Q` times longer than the frame rate. Give the token count for 10 seconds of 75 Hz audio at 8 codebooks, and say what that does to attention cost compared with 2 codebooks.
- Codebook 1 is coarse and codebook 8 is fine. What does frame-major order give the model when it predicts codebook 8 of frame `f` that a stream-after-stream layout would not?
- Real systems rarely use plain interleaving: they use a delay pattern, or a coarse model plus a fine model. What problem with one flat stream are they solving?
- If the decoder splits the sequence with the wrong `Q`, the indices are all still valid codebook entries and nothing raises. What does the output sound like, and what cheap check would have caught it?
