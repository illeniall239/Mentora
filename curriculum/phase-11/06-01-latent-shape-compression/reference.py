# Reference solution: only used by scripts/verify-exercises.mjs. Shown in the Breakdown only after the Learner's own solution passes.
def _check(h: int, w: int, down: int, ch: int) -> None:
    if min(h, w, down, ch) < 1:
        raise ValueError("h, w, down and ch must all be at least 1")
    if h % down or w % down:
        raise ValueError("h and w must be multiples of down")


def latent_shape(h: int, w: int, down: int, ch: int) -> tuple[int, int, int]:
    _check(h, w, down, ch)
    return (ch, h // down, w // down)


def compression_ratio(h: int, w: int, down: int, ch: int) -> float:
    c, lh, lw = latent_shape(h, w, down, ch)
    return (h * w * 3) / (c * lh * lw)
