# Reference solution: only used by scripts/verify-exercises.mjs. Shown in the Breakdown only after the Learner's own solution passes.
import math


def vlm_token_count(img_h: int, img_w: int, patch: int, tile: int, max_tiles: int) -> int:
    if img_h < 1 or img_w < 1 or patch < 1 or tile < 1 or max_tiles < 1:
        raise ValueError("sizes, patch, tile and max_tiles must all be at least 1")
    if tile % patch:
        raise ValueError("tile must be a whole number of patches")
    tokens_per_tile = (tile // patch) ** 2
    n_tiles = min(math.ceil(img_h / tile) * math.ceil(img_w / tile), max_tiles)
    return (n_tiles + 1) * tokens_per_tile
