# Length regulator

Topic: 11. Text-to-speech
Difficulty: 2 of 3

## Problem

A non-autoregressive TTS model predicts one duration per phoneme and then **expands** the phoneme sequence to the frame rate of the mel spectrogram: a 50 ms phoneme at a 10 ms hop becomes 5 identical frames of conditioning. That expansion is the length regulator, and it is where TTS decides how long the utterance will be.

The trap is rounding. Rounding each duration on its own and adding up the results drifts away from the true total: five phonemes of 14 ms at a 10 ms hop are 70 ms of speech, but five independent roundings of 1.4 give 5 frames, not 7. So round the **cumulative** time instead, and take each phoneme's frame count as the difference between consecutive boundaries.

Pure Python with `math`, no numpy.

- `durations_to_frames(phonemes: list[str], durs: list[float], hop_s: float) -> list[str]` — `durs[i]` is phoneme `i`'s duration in seconds and `hop_s` is the frame hop in seconds. Let `cum[i]` be the running sum `durs[0] + … + durs[i]`, accumulated left to right with ordinary float addition, and

  ```
  boundary[i] = round_half_up(cum[i] / hop_s),   boundary[-1] = 0
  n_i         = boundary[i] − boundary[i − 1]
  ```

  where `round_half_up(x) = math.floor(x + 0.5)`. Return `phonemes[0]` repeated `n_0` times, then `phonemes[1]` repeated `n_1` times, and so on. `round_half_up` is not Python's `round`, which rounds 0.5 to 0 (banker's rounding) — use the floor form so a half-frame always rounds up.
- `total_frames(durs: list[float], hop_s: float) -> int` — `round_half_up(total / hop_s)` where `total` is the same left-to-right sum. By construction `len(durations_to_frames(p, d, hop)) == total_frames(d, hop)`: that is the invariant the cumulative rule buys you.

Notes and errors:

- `n_i` can be 0. A phoneme shorter than half a hop gets no frames at all and disappears from the output — that is a real failure mode, not a bug to patch over.
- Both raise `ValueError` if `hop_s <= 0`, if any duration is negative, or (for `durations_to_frames`) if `phonemes` and `durs` have different lengths.
- Empty input is fine: `durations_to_frames([], [], 0.01)` is `[]` and `total_frames([], 0.01)` is `0`.

## Examples

```
durations_to_frames(["HH", "AH", "L", "OW"], [0.05, 0.03, 0.04, 0.10], 0.01)
    → ["HH"]*5 + ["AH"]*3 + ["L"]*4 + ["OW"]*10          22 frames
total_frames([0.05, 0.03, 0.04, 0.10], 0.01)              → 22

durations_to_frames(["a", "b", "c", "d", "e"], [0.014]*5, 0.01)
    → ["a", "b", "b", "c", "d", "d", "e"]                 7 frames, not 5
    (rounding 1.4 five times on its own gives 1 frame each and loses 2 frames)

durations_to_frames(["a", "b"], [0.5, 1.0], 1.0)          → ["a", "b"]
    (boundaries 0.5 → 1 and 1.5 → 2; Python's round would give 0 → ["b", "b"])

durations_to_frames(["a", "b", "c"], [0.05, 0.001, 0.04], 0.01)
    → ["a"]*5 + ["c"]*4                                   "b" is too short to survive

durations_to_frames(["a"], [0.1], 0.0)                    → ValueError
```

## Constraints

- Up to 2000 phonemes. Frame counts compared exactly; `hop_s` and durations are ordinary floats.
- Time: O(number of output frames).

## Hints

1. Before you write any code: for `[0.014] * 5` at a 10 ms hop, write down the five cumulative times, divide each by the hop, and round. What are the five differences?
2. What do you have to carry from one loop iteration to the next so that phoneme `i` knows where phoneme `i − 1` ended?
3. `math.floor(x + 0.5)` and `round(x)` disagree on exactly which inputs? Try both on 0.5, 1.5 and 2.5 and decide which one the invariant needs.
4. If `n_i` comes out as 0, what has the model said about that phoneme, and what would silently clamping it to 1 do to the invariant between `durations_to_frames` and `total_frames`?

## Explain-back

- Why does the expanded sequence have to exist at all? What would the decoder be missing if you fed it one vector per phoneme instead of one per frame?
- Rounding each duration independently loses frames. Where does that error end up in the audio — a shorter utterance, a shifted one, or both?
- A phoneme predicted at 1 ms disappears here. What does that sound like, and would a mel-based system or an autoregressive one be more forgiving?
- Duration is only part of prosody. Name the other predicted quantities in a FastSpeech-style model, and say which one you would reach for to make a line sound like a question.
