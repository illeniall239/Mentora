def vlm_token_count(img_h: int, img_w: int, patch: int, tile: int, max_tiles: int) -> int:
    """(min(ceil(h/tile)*ceil(w/tile), max_tiles) + 1 thumbnail) * (tile//patch)^2 image tokens."""
    raise NotImplementedError
