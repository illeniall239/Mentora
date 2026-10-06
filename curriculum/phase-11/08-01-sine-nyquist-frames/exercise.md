# Sine waves, Nyquist and frame counts

Topic: 8. Audio as data
Difficulty: 1 of 3

## Problem

Before any model sees audio it sees a list of numbers: amplitudes measured `sr` times per second. Three small functions set up everything later in this topic — the signal to analyse, the frequency ceiling, and how many analysis frames a clip yields. Write them in numpy.

- `sine_wave(freq: float, sr: int, dur: float) -> np.ndarray` — a 1-D float array of `n = round(sr * dur)` samples:

  ```
  x[i] = sin(2 * pi * freq * i / sr)        for i = 0 .. n-1
  ```

  Amplitude 1, zero phase, so `x[0]` is exactly 0. `dur = 0` gives an empty array. Raise `ValueError` if `sr < 1` or `dur < 0`.

- `nyquist(sr: int) -> float` — the highest frequency this sampling rate can represent, `sr / 2`, as a float. Raise `ValueError` if `sr < 1`.

- `frames(samples: int, win: int, hop: int) -> int` — how many **complete** analysis windows of length `win` fit in `samples` samples when each window starts `hop` samples after the previous one:

  ```
  0                           if samples < win
  1 + (samples - win) // hop  otherwise
  ```

  No padding, no partial final frame. Note that the frame count depends on `hop`, not on `win`, once the first window fits: with `win = 400` and `hop = 160`, consecutive windows overlap by 240 samples and a new frame appears every 160 samples. Raise `ValueError` if `samples < 0`, `win < 1` or `hop < 1`.

A sine above the Nyquist frequency does not simply vanish — it comes back as a different frequency. Your `sine_wave` will reproduce that on its own, with no extra code: feeding it `sr - f` gives the negated version of `f`, and feeding it `sr + f` gives `f` back exactly. That is aliasing, and it is why real recorders low-pass filter before sampling.

numpy only; no `scipy.signal`, no `librosa`, no `torchaudio`.

## Examples

```
nyquist(16000)                -> 8000.0        speech: everything above 8 kHz is gone
nyquist(44100)                -> 22050.0

len(sine_wave(440, 16000, 0.5))   -> 8000
sine_wave(1, 4, 1.0)              -> [0.0, 1.0, 0.0, -1.0]      one cycle in 4 samples
sine_wave(0, 16000, 0.1)          -> all zeros

frames(16000, 400, 160)       -> 98           1 s at 16 kHz, 25 ms window, 10 ms hop
frames(400, 400, 160)         -> 1            exactly one window fits
frames(399, 400, 160)         -> 0            not even one
frames(1000, 400, 400)        -> 2            no overlap: hop = win
```

## Constraints

- numpy arrays of float (float64 is fine); tolerance 1e-9.
- Clips up to 30 s at 48 kHz, so build the array vectorised rather than with a Python loop.
- `freq` may be 0, negative, or above the Nyquist frequency: the formula is applied as written either way.

## Hints

1. What is the index of the last sample, and what time in seconds does it sit at? Is the final sample at `dur` seconds or one step before it?
2. If you build the time axis with `np.arange`, what do you divide by to turn a sample index into seconds?
3. For `frames`, draw the start positions on a line for `samples = 1000`, `win = 400`, `hop = 160`. Which start is the last one whose window still ends inside the signal?
4. `frames(399, 400, 160)` and `frames(400, 400, 160)` differ by one sample. Does your formula handle both, or does it return something negative for the first?

## Explain-back

- A colleague wants to record speech at 48 kHz "for accuracy" before sending it to a 16 kHz ASR model. What does the model do with that file, and what did the extra samples buy?
- A 10 kHz tone is recorded at 16 kHz with no anti-alias filter. What frequency appears in the file, and why can no later processing remove it?
- Window 400 and hop 160 at 16 kHz: what are those in milliseconds, and why is the hop shorter than the window?
- Doubling the hop halves the number of frames. What does that do to the model's view of a fast consonant, and what does it do to compute?
