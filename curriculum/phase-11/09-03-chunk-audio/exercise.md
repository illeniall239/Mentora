# Chunking long audio for Whisper

Topic: 9. Speech recognition and Whisper
Difficulty: 2 of 3

## Problem

Whisper's encoder takes exactly 30 seconds of audio: a one-hour podcast has to be cut into windows before it ever reaches the model. Cutting at exact 30-second boundaries slices words in half, so real pipelines overlap the windows and stitch the transcripts back together. Write the window arithmetic in plain Python.

`chunk_audio(n_samples: int, sr: int, chunk_s: float, overlap_s: float) -> list[tuple[int, int]]`

Return a list of `(start, end)` sample ranges, `end` **exclusive**, built like this:

```
chunk   = round(chunk_s * sr)      samples per window
overlap = round(overlap_s * sr)    samples shared with the previous window
stride  = chunk - overlap          samples between consecutive starts
```

- The first window starts at sample 0. Each next window starts `stride` samples after the previous one.
- A window's end is `min(start + chunk, n_samples)`, so the last one may be short — the audio is never padded here and never truncated.
- **Stop as soon as a window reaches the end of the audio.** Never emit a window that is contained in the one before it: once `end == n_samples`, that window is the last one.
- `n_samples = 0` returns `[]`. Audio shorter than one chunk returns exactly one window, `[(0, n_samples)]`.
- The windows must **cover every sample**: the union of the ranges is exactly `0 .. n_samples - 1`, with no gaps.

Raise `ValueError` if `n_samples < 0`, `sr < 1`, `chunk_s <= 0`, `overlap_s < 0`, or `overlap_s >= chunk_s` — a non-positive stride would never terminate.

Plain Python. No numpy, no `torchaudio`, no generators: return a real `list` of `tuple`s of `int`.

## Examples

```
70 seconds at 16 kHz, 30 s windows with 5 s overlap: chunk = 480000, stride = 400000

chunk_audio(1120000, 16000, 30, 5)
    -> [(0, 480000), (400000, 880000), (800000, 1120000)]
       the last window is clipped to the end and stops the loop

chunk_audio(160000, 16000, 30, 5)   -> [(0, 160000)]      10 s of audio: one short window
chunk_audio(0, 16000, 30, 5)        -> []
chunk_audio(1000, 16000, 0.025, 0.01)
    -> [(0, 400), (240, 640), (480, 880), (720, 1000)]    25 ms windows, 10 ms overlap

chunk_audio(1000, 16000, 0.025, 0.0)
    -> [(0, 400), (400, 800), (800, 1000)]                no overlap: plain tiles

chunk_audio(1000, 16000, 0.01, 0.01)  -> ValueError       stride 0
```

## Constraints

- Up to about 10 million samples and a few thousand windows; one loop, O(number of windows).
- `start` and `end` are `int`. `chunk_s` and `overlap_s` are seconds and may be fractional.
- Consecutive starts differ by exactly `stride`, and consecutive windows overlap by exactly `overlap` samples wherever the second one is not clipped.

## Hints

1. Write down the start of window `k` in closed form. What condition on that start means there is no audio left to cover?
2. The last window is special twice over: its end is clipped, and it ends the loop. Which of those two things prevents a redundant window at the very end?
3. Take `n_samples = 1000`, `chunk = 400`, `overlap = 160`. List the starts on paper and check that every sample from 0 to 999 falls in at least one window.
4. What happens to the loop if `overlap` equals `chunk`? Trace the second start and say why the check belongs at the top of the function.

## Explain-back

- The windows overlap by 5 seconds, so the same words are transcribed twice. What has to happen to the two transcripts afterwards, and why is that merge the hard part rather than the chunking?
- A sentence boundary falls at 29.8 s. What does a 30-second window with no overlap do to it, and what does the overlap buy you?
- Whisper is said to be "non-streaming". Your function cuts audio into windows, which sounds like streaming. What is still missing for a live caption feed?
- Audio shorter than 30 s is padded to 30 s before the encoder, but your function returns a short final window. Where should that padding live, and why not here?
