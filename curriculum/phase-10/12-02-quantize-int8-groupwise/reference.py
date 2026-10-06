# Reference solution: only used by scripts/verify-exercises.mjs. Shown in the Breakdown only after the Learner's own solution passes.
def _absmax_quantize(xs: list[float], qmax: int) -> tuple[list[int], float]:
    scale = max(abs(x) for x in xs) / qmax
    if scale == 0.0:
        return [0] * len(xs), 0.0
    return [max(-qmax, min(qmax, round(x / scale))) for x in xs], scale


def quantize_absmax_int8(xs: list[float]) -> tuple[list[int], float]:
    if not xs:
        raise ValueError("xs must not be empty")
    return _absmax_quantize(xs, 127)


def dequantize(q: list[int], scale: float) -> list[float]:
    return [v * scale for v in q]


def quantize_groupwise(xs: list[float], group: int, bits: int) -> tuple[list[int], list[float]]:
    if not xs or group < 1 or not 2 <= bits <= 8:
        raise ValueError("need non-empty xs, group >= 1 and 2 <= bits <= 8")
    qmax = 2 ** (bits - 1) - 1
    q: list[int] = []
    scales: list[float] = []
    for start in range(0, len(xs), group):
        chunk_q, scale = _absmax_quantize(xs[start:start + group], qmax)
        q.extend(chunk_q)
        scales.append(scale)
    return q, scales


def dequantize_groupwise(q: list[int], scales: list[float], group: int) -> list[float]:
    return [v * scales[i // group] for i, v in enumerate(q)]
