# Codec bitrate

Topic: 10. Audio tokens and neural codecs
Difficulty: 1 of 3

## Problem

A neural codec turns a waveform into a small number of discrete tokens per second. EnCodec at 24 kHz emits 75 frames per second, and each frame carries one index from each of `n_codebooks` residual codebooks. Work out what that costs. Pure Python with `math`.

- `codec_bitrate(frame_rate: float, n_codebooks: int, codebook_size: int) -> float` — bits per second:

  ```
  bitrate = frame_rate · n_codebooks · log2(codebook_size)
  ```

  So 75 Hz × 8 codebooks × 1024 entries = 75 · 8 · 10 = 6000 bits/s (6 kbps). `codebook_size` need not be a power of two; use the real `log2` in that case.
- `tokens_per_second(frame_rate: float, n_codebooks: int) -> float` — how many tokens a language model must emit per second of audio: `frame_rate · n_codebooks`. This is the cost the bitrate figure hides: doubling the codebooks doubles the sequence length as well as the bitrate.
- `compression_ratio(sample_rate: int, bit_depth: int, bitrate_bps: float) -> float` — raw PCM bits per second (`sample_rate · bit_depth`, one channel) divided by `bitrate_bps`.

All three raise `ValueError` on nonsense inputs: `frame_rate <= 0`, `n_codebooks < 1`, `codebook_size < 2`, `sample_rate < 1`, `bit_depth < 1`, `bitrate_bps <= 0`. Return plain floats (bits per second, tokens per second, a ratio) — no rounding to kbps.

## Examples

```
codec_bitrate(75, 8, 1024)              → 6000.0      EnCodec 24 kHz, 6 kbps
codec_bitrate(75, 2, 1024)              → 1500.0      1.5 kbps
codec_bitrate(50, 4, 2048)              → 2200.0      log2(2048) = 11
codec_bitrate(75, 1, 1000)              → 747.33...   log2(1000) = 9.9657...
tokens_per_second(75, 8)                → 600.0
tokens_per_second(75, 1)                → 75.0
compression_ratio(24000, 16, 6000.0)    → 64.0        384 kbps of PCM into 6 kbps
codec_bitrate(75, 0, 1024)              → ValueError
```

## Constraints

- Pure Python with `math`. No numpy needed.
- Absolute tolerance 1e-9 on exact powers of two, 1e-6 elsewhere.
- `n_codebooks` up to 32, `codebook_size` up to 2^16.

## Hints

1. How many bits does it take to name one entry out of 1024? Out of 2048? What function turns "how many choices" into "how many bits"?
2. One frame carries one index per codebook. How many indices is that per second, and how many bits is each one?
3. A codec is quoted as "6 kbps". Which of the three numbers in the formula could you change to halve that, and what does each change cost you — quality, sequence length, or both?
4. PCM at 24 kHz and 16 bits is 384 000 bits per second. What is 384 000 divided by 6 000, and does that ratio say anything about how good the reconstruction sounds?

## Explain-back

- An LM generating speech at 75 Hz with 8 codebooks emits 600 tokens per second of audio. How does that compare with the number of text tokens in a second of speech, and why is "audio tokens are like text tokens, one per word" wrong?
- Adding codebooks lowers the reconstruction error. Name the two prices you pay, and say which one hurts an autoregressive model most.
- Codebook 1 of an RVQ stack carries most of the signal and codebook 8 the least. If you had to drop half the codebooks at decode time, which half would you drop and what happens to the audio?
- Semantic tokens from a self-supervised model and acoustic tokens from a codec both arrive at some frames per second. What does each one let you reconstruct, and why do speech LMs often use both?
