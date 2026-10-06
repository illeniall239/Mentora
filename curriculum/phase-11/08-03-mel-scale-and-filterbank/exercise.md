# The mel scale and a triangular filterbank

Topic: 8. Audio as data
Difficulty: 3 of 3

## Problem

A linear-frequency spectrum spends as many bins between 7 kHz and 8 kHz as between 0 and 1 kHz, although a human hears almost nothing in the first interval and almost everything in the second. The mel scale fixes that, and a mel filterbank is just a matrix that sums spectrum bins into mel bands. Write both in numpy.

- `hz_to_mel(hz: float) -> float` and `mel_to_hz(mel: float) -> float`, the **HTK** formulas (the ones Whisper's feature extractor uses):

  ```
  mel = 2595 * log10(1 + hz / 700)
  hz  = 700 * (10 ** (mel / 2595) - 1)
  ```

  They are exact inverses of each other. `hz_to_mel(0)` is exactly `0` — the mel scale is **not** a log scale: it is close to linear below about 1 kHz and only becomes logarithmic above it. By construction `hz_to_mel(1000)` is about 1000 (999.99 would need the constant 2595 tuned; 2595 gives 1000.0 to four digits). Raise `ValueError` on a negative argument.

- `mel_filterbank(n_fft: int, sr: int, n_mels: int) -> np.ndarray` returns a `(n_mels, n_fft // 2 + 1)` matrix of triangular weights, built exactly like this:

  1. The spectrum bins sit at frequencies `bin_hz = linspace(0, sr / 2, n_fft // 2 + 1)`.
  2. Take `n_mels + 2` points evenly spaced **in mel**, from `hz_to_mel(0)` to `hz_to_mel(sr / 2)`, and convert them back to hertz: `pts[0] .. pts[n_mels + 1]`.
  3. Filter `m` (for `m = 0 .. n_mels - 1`) is the triangle with left foot `pts[m]`, peak `pts[m + 1]` and right foot `pts[m + 2]`:

     ```
     w(f) = max(0, min( (f - pts[m]) / (pts[m+1] - pts[m]),
                        (pts[m+2] - f) / (pts[m+2] - pts[m+1]) ))
     ```

     evaluated at each `bin_hz`. Weights are 0 outside `(pts[m], pts[m + 2])`, rise linearly to 1 at the peak frequency and fall linearly back. No area normalisation: the triangle's height is 1, so a wide high-frequency filter weighs more total energy than a narrow low-frequency one.

  Neighbouring filters overlap: filter `m`'s peak is filter `m+1`'s left foot. Peaks move right as `m` grows, and because the points are even in mel and not in hertz, the filters get wider in hertz as `m` grows. Raise `ValueError` if `n_fft < 2`, `sr < 1` or `n_mels < 1`.

The filterbank is applied to the **magnitude or power spectrum, before any log**: `mel_spec = filterbank @ spectrum`. Taking the log first and then summing would be summing logarithms, which is multiplying the energies — a different and wrong quantity.

Banned: `librosa.filters.mel`, `torchaudio.functional.melscale_fbanks`, `scipy`. numpy arithmetic only.

## Examples

```
hz_to_mel(0)        -> 0.0
hz_to_mel(1000)     -> 999.98...        the scale is anchored near 1000 Hz = 1000 mel
hz_to_mel(8000)     -> 2840.02...       8x the frequency, under 3x the mel value
mel_to_hz(hz_to_mel(440)) -> 440.0      exact round trip

fb = mel_filterbank(n_fft=512, sr=16000, n_mels=10)
fb.shape                       -> (10, 257)
fb.min(), fb.max()             -> 0.0, 1.0 (or just under 1, if no bin sits on a peak)
(fb[0] > 0).sum()              -> a handful of bins
(fb[9] > 0).sum()              -> many more: high-frequency filters are wide
argmax(fb[m]) strictly increasing in m
fb[0, 0]                       -> 0.0   bin 0 sits on the first filter's left foot
```

## Constraints

- numpy; `n_fft` up to 2048, `n_mels` from 1 to 128, `sr` from 1000 to 48000.
- Tolerance 1e-6; weights are float, never negative, never above 1.
- `n_mels + 2` mel points are **evenly spaced in mel**, which is the whole point — do not space them evenly in hertz.

## Hints

1. Check the formulas against each other before anything else: does `mel_to_hz(hz_to_mel(x))` return `x` for 0, 100 and 8000?
2. You need `n_mels` triangles but `n_mels + 2` points. Which two points are used by only one triangle each, and what would go wrong with exactly `n_mels` of them?
3. The triangle is the lower of two straight lines, floored at zero. Write the rising line and the falling line separately — what is each one's value at the peak frequency?
4. If you spaced the points evenly in hertz instead, what would happen to the filter widths, and which part of the spectrum would get too much resolution?

## Explain-back

- Someone says "mel is just a log frequency scale". Your `hz_to_mel(0)` returns 0 while `log(0)` is undefined. What else is different about the scale below 1 kHz, and why was it built that way?
- The filterbank multiplies the spectrum and *then* you take the log. What happens if you swap those two steps, and what quantity do you end up having computed?
- 80 mel bands replace 201 linear bins in Whisper. What is thrown away, and why does an ASR model not miss it?
- The triangles overlap and have unit height, not unit area. What would change in the output if you normalised each filter to unit area instead?
