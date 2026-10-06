def quantize_absmax_int8(xs: list[float]) -> tuple[list[int], float]:
    """scale = max|x| / 127, q = round(x / scale) clamped to [-127, 127]; all-zero input gives scale 0.0."""
    raise NotImplementedError


def dequantize(q: list[int], scale: float) -> list[float]:
    """q_i * scale."""
    raise NotImplementedError


def quantize_groupwise(xs: list[float], group: int, bits: int) -> tuple[list[int], list[float]]:
    """Absmax-quantize each consecutive group of `group` entries with qmax = 2^(bits-1) - 1; return (flat q, scales)."""
    raise NotImplementedError


def dequantize_groupwise(q: list[int], scales: list[float], group: int) -> list[float]:
    """Undo quantize_groupwise: each q_i times the scale of its group."""
    raise NotImplementedError
