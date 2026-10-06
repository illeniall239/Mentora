# Latent shape and compression ratio

Topic: 6. Latent diffusion and text-to-image conditioning
Difficulty: 1 of 3

## Problem

Stable Diffusion does not run its UNet on pixels. A VAE encoder first shrinks a `512 × 512 × 3` image to a `4 × 64 × 64` latent, the whole reverse diffusion loop runs there, and the VAE decoder expands the final latent back to pixels. That one change is most of why text-to-image became affordable. Work out the arithmetic in plain Python (no numpy, no torch).

- `latent_shape(h: int, w: int, down: int, ch: int) -> tuple[int, int, int]` — the latent tensor's shape for an `h × w` RGB image, **channels first**: `(ch, h // down, w // down)`. `down` is the downsampling factor *per side* (8 for Stable Diffusion), `ch` the number of latent channels (4 for SD 1.x/2.x, 16 for SD3/Flux).
- `compression_ratio(h: int, w: int, down: int, ch: int) -> float` — how many numbers the pixel image holds divided by how many the latent holds:

  ```
  pixel_elements  = h * w * 3            the image is always RGB
  latent_elements = ch * (h // down) * (w // down)
  ratio           = pixel_elements / latent_elements
  ```

  Return a `float`, exact division (not integer division).

Both raise `ValueError` if `h` or `w` is not a multiple of `down`, or if any of `h`, `w`, `down`, `ch` is less than 1. Note that the ratio counts **numbers**, not bits or bytes: it is not a file-compression ratio, and the latent is not a small JPEG.

## Examples

```
latent_shape(512, 512, 8, 4)        → (4, 64, 64)          Stable Diffusion 1.5
latent_shape(768, 512, 8, 4)        → (4, 96, 64)          non-square, height first
latent_shape(1024, 1024, 8, 16)     → (16, 128, 128)       SD3-style 16-channel VAE
compression_ratio(512, 512, 8, 4)   → 48.0                 786432 pixels values / 16384 latents
compression_ratio(512, 512, 8, 16)  → 12.0                 more channels, less compression
compression_ratio(512, 512, 1, 3)   → 1.0                  no VAE at all
latent_shape(515, 512, 8, 4)        → ValueError
```

## Constraints

- Plain Python ints and floats only; no numpy, no torch.
- `h`, `w` up to 4096. The returned shape is a tuple of three `int`s, in the order `(channels, height, width)`.
- Floats are compared to 1e-9.

## Hints

1. The downsampling factor is given *per side*. If one side shrinks by 8, by how much does the number of spatial positions shrink?
2. The latent has fewer spatial positions but more channels than the 3 of RGB. Which of those two effects does the ratio divide by, and which does it multiply by?
3. Why must the division in `compression_ratio` be `/` and not `//`? Find an `h`, `w`, `down`, `ch` where the two disagree.
4. Both functions reject the same inputs. Where should that check live so you write it once?

## Explain-back

- A 512 × 512 image becomes a 4 × 64 × 64 latent. Self-attention in the UNet costs O(tokens²): how many times cheaper is one attention layer in the latent than at pixel resolution?
- The ratio is 48, yet a 512 × 512 JPEG is far smaller than 48× below its raw size. What is the latent actually storing, and why is "the latent is a compressed JPEG" wrong?
- SD3 moved from 4 latent channels to 16, which *lowers* the compression ratio. What does the model get in exchange, and what does it pay?
- Your function demands that the side be a multiple of `down`. What happens in a real pipeline when a user asks for a 515-pixel-wide image?
