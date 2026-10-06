# The image token budget

Topic: 12. Multimodal models
Difficulty: 1 of 3

## Problem

A vision-language model does not see pixels: it sees patch tokens, and they share the context window with the text. One 336×336 crop at patch 14 is 576 tokens — longer than most questions asked about it. High-resolution inputs are handled by **tiling**: the image is cut into tiles of `tile × tile` pixels, each tile is encoded on its own, and one extra "thumbnail" tile (the whole image squeezed into a single tile) is always prepended so the model keeps the global view it would otherwise lose.

Write the token budget. Pure Python with `math`.

`vlm_token_count(img_h: int, img_w: int, patch: int, tile: int, max_tiles: int) -> int`:

```
tokens_per_tile = (tile // patch)²
n_tiles         = min( ceil(img_h / tile) · ceil(img_w / tile),  max_tiles )
total           = (n_tiles + 1) · tokens_per_tile
```

- The grid is covered by whole tiles, so a partially filled tile still costs a full tile — use `ceil`, not integer division.
- `max_tiles` is a hard cap: beyond it the image is downscaled to fit, so the cost stops growing and the detail is what is lost instead.
- The `+ 1` is the thumbnail, and it is counted even when the image is one tile.
- Raise `ValueError` if `img_h < 1`, `img_w < 1`, `patch < 1`, `tile < 1`, `max_tiles < 1`, or `tile % patch != 0` (a tile that is not a whole number of patches has no well-defined token count).

## Examples

```
vlm_token_count(336, 336, 14, 336, 4)     → 1152     1 tile + thumbnail, 576 each
vlm_token_count(337, 336, 14, 336, 4)     → 1728     one pixel taller, a whole extra tile
vlm_token_count(1000, 1000, 14, 336, 4)   → 2880     3×3 = 9 tiles capped to 4, plus thumbnail
vlm_token_count(5000, 5000, 14, 336, 4)   → 2880     the cap, not the image, sets the cost
vlm_token_count(224, 224, 14, 224, 1)     → 512      256 per tile
vlm_token_count(224, 224, 7, 224, 1)      → 2048     half the patch size, four times the tokens
vlm_token_count(336, 336, 15, 336, 4)     → ValueError
```

## Constraints

- Sizes up to 10 000 pixels, `max_tiles` up to 64. Exact integer results.
- Pure Python with `math`. No numpy, no image library — nothing is loaded or resized here, you are only counting.

## Hints

1. How many patches fit along one side of a tile, and how many in the whole tile? Where does the square come from?
2. An image 337 pixels tall with 336-pixel tiles needs how many tile rows? Which of `//` and `ceil` gives that, and what would the other one leave uncovered?
3. The cap applies to the number of tiles, not to the total. Does the thumbnail fall inside or outside the cap in the formula above, and what would change if it fell inside?
4. Take the 1000×1000 case and work out the token count if `max_tiles` were 9 instead of 4. What did the cap buy, and what did it cost?

## Explain-back

- "The LLM sees pixels." Say what actually enters the LLM sequence here, and at which step the pixels stop existing.
- Two users send a 336×336 photo and a 4000×3000 photo with the same question. What do they each pay, and why is "images cost the same regardless of size" wrong in both directions?
- A video at 1 frame per second for 30 seconds, each frame costing one tile plus thumbnail: compute the tokens. Which trick would you reach for first to make that fit?
- The thumbnail tile is redundant with the detail tiles — they cover the same scene. What does the model lose without it, and why does cropping alone not give the model that information?
