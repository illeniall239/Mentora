# Greedy CTC decoding

Topic: 9. Speech recognition and Whisper
Difficulty: 2 of 3

## Problem

A wav2vec2-style CTC encoder emits one token per frame — about 50 per second — so a one-second "hello" comes out as something like `hh---e--llll---ll--oo`. Two rules turn that back into text, and **their order is the whole exercise**. Write the decoder in plain Python.

`ctc_greedy_decode(frame_ids: list[int], blank: int) -> list[int]`

1. **Collapse runs of the same id first**: every maximal run of identical consecutive ids becomes a single id.
2. **Then remove every blank**.

That order is what lets a word contain a doubled letter. The blank is not "nothing happened" padding to be swept away first — it is the model's way of saying "a new token starts here, even though it is the same token again". `[l, blank, l]` collapses to `[l, blank, l]` and then to `[l, l]`, a real double letter. Removing blanks first would give `[l, l]` -> `[l]`, and the model would be unable to ever write "hello", "butter" or "ll" in any language.

- Return a new `list[int]`; do not modify `frame_ids`.
- An empty input returns `[]`. An all-blank input returns `[]` (that is how CTC transcribes silence).
- Raise `ValueError` if `blank < 0` or if any entry of `frame_ids` is negative: these are vocabulary indices.
- This is the *greedy* (best-path) decoder: it takes the argmax ids it is given. It does not sum probability over alignments and it has no language model, so you take ids in, ids out — no probabilities anywhere in the signature.

Plain Python; `itertools.groupby` is allowed. Banned: `torchaudio` decoders, `ctcdecode`, `pyctcdecode`.

## Examples

With the toy vocabulary `0 = blank, 5 = e, 8 = h, 12 = l, 15 = o`:

```
"hh-e-ll-ll-o"
ctc_greedy_decode([8,8,0,5,0,12,12,0,12,12,0,15], blank=0)
    collapse -> [8,0,5,0,12,0,12,0,15]
    deblank  -> [8,5,12,12,15]              = h e l l o

ctc_greedy_decode([12, 12], 0)        -> [12]      one l: a run is one token
ctc_greedy_decode([12, 0, 12], 0)     -> [12, 12]  two l: the blank separates them
ctc_greedy_decode([], 0)              -> []
ctc_greedy_decode([0, 0, 0, 0], 0)    -> []        silence
ctc_greedy_decode([3, 3, 3], 0)       -> [3]       no blanks anywhere
ctc_greedy_decode([1, 28, 1], 28)     -> [1, 1]    blank need not be 0
```

## Constraints

- Up to about 2000 frames; one pass is enough, O(len(frame_ids)).
- Pure Python lists of `int` in and out. Compared exactly.

## Hints

1. Walk `[12, 0, 12]` through both orders on paper: collapse-then-deblank, and deblank-then-collapse. Which one can still produce a double letter?
2. "Collapse a run" means comparing each id with which other id — the previous *frame*, or the previous id you kept in the output?
3. After collapsing, can the output still contain two equal adjacent ids? What must sit between them in the original frames?
4. The model emits a blank in about 80% of frames. What would be ambiguous about the frame sequence if the blank did not exist at all?

## Explain-back

- Someone drops the blanks first "to clean up", then collapses repeats. Which words break, and which still decode correctly — so why does this bug survive a quick test?
- CTC assumes the output tokens are conditionally independent given the audio. What does that let you do at decode time that Whisper's autoregressive decoder cannot, and what does it cost in fluency?
- Greedy decoding takes the argmax per frame. Why is that not the most probable *transcript*, and what does beam search with an external language model add?
- Whisper does not use CTC at all. What does its decoder emit instead of a per-frame token, and why does that make it hallucinate on silence while a CTC model emits blanks?
