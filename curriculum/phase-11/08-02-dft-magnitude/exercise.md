# A DFT by hand

Topic: 8. Audio as data
Difficulty: 2 of 3

## Problem

A spectrogram is a stack of DFTs, one per frame. Write the DFT yourself, the slow way, so that the bin indices and the magnitude stop being magic.

`dft_magnitude(samples: np.ndarray) -> np.ndarray` takes a 1-D real signal of length `N` and returns the **one-sided magnitude spectrum**, a real array of length `N // 2 + 1`:

```
X[k] = sum over n = 0..N-1 of  samples[n] * exp(-2j * pi * k * n / N)
out[k] = abs(X[k])                 for k = 0 .. N // 2
```

- Only the first `N // 2 + 1` bins are returned. For a real signal the rest are mirror images (`|X[N-k]| = |X[k]|`), so they carry nothing new.
- `out` is real and non-negative, dtype float. The phase of `X[k]` is computed and then **thrown away** by `abs`; nothing downstream can recover it.
- Bin `k` corresponds to the frequency `k * sr / N` hertz, so bin 0 is DC (the mean, times `N`) and bin `N // 2` is the Nyquist frequency. A tone at `f` hertz lands exactly on bin `f * N / sr` when that is a whole number, and smears across neighbouring bins when it is not.
- Raise `ValueError` if `samples` is empty or not 1-D.

Write the sum yourself: an O(N^2) double loop, or the same thing as one matrix of complex exponentials times the signal. **Banned: `np.fft`, `torch.fft`, `scipy.fft`, `scipy.signal` and `librosa`** — the point is the sum, not the fast algorithm. numpy arithmetic, `np.exp`, `np.abs` and `np.arange` are all fine.

## Examples

```
dft_magnitude([1, 1, 1, 1])         -> [4, 0, 0]          DC only: 4 = N * mean
dft_magnitude([1, -1, 1, -1])       -> [0, 0, 4]          alternating = Nyquist
dft_magnitude([0, 1, 0, -1])        -> [0, 2, 0]          one cycle in 4 samples

N = 400 samples of a 1000 Hz sine at sr = 8000:
    peak bin = argmax = 1000 * 400 / 8000 = 50
    out[50]  ~ N / 2 = 200            an amplitude-1 sine splits its energy over +f and -f
    out[k]   ~ 0 elsewhere
```

## Constraints

- numpy, `N` up to about 1024 (an O(N^2) transform of that size is still well under a second).
- Tolerance 1e-6 relative to a reference transform.
- Input may be any real dtype; output length is exactly `N // 2 + 1` for both even and odd `N`.

## Hints

1. `X[k]` is a sum over all `N` samples, and there are about `N / 2` values of `k`. What does that make the total number of multiplications?
2. `exp(-2j * pi * k * n / N)` depends on the product `k * n`. What shape does `np.outer(k_indices, n_indices)` have, and what does multiplying that matrix of exponentials by the signal give you in one step?
3. For `N = 4`, write out the four complex exponentials for `k = 1` by hand. What are they, and why does `[0, 1, 0, -1]` give magnitude 2 rather than 4?
4. Two recordings of the same tone, one starting a quarter cycle later. What differs in `X[k]`, and what does `abs` do to that difference?

## Explain-back

- Your output has `N // 2 + 1` bins, not `N`. What is in the bins you dropped, and why is nothing lost for a real-valued signal?
- A 1000 Hz tone sampled at 8000 Hz for 400 samples peaks at bin 50. Where does a 1050 Hz tone peak, and what does the spectrum look like around it?
- The magnitude spectrum discards the phase. Why does that make a mel spectrogram impossible to invert exactly, and what has to be trained to fake the missing part?
- You want finer frequency resolution. Which do you change, `N` or `sr`, and what do you give up in time resolution?
