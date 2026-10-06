def latent_shape(h: int, w: int, down: int, ch: int) -> tuple[int, int, int]:
    """Channels-first latent shape (ch, h // down, w // down) for an h x w RGB image."""
    raise NotImplementedError


def compression_ratio(h: int, w: int, down: int, ch: int) -> float:
    """Pixel element count divided by latent element count, as a float."""
    raise NotImplementedError
