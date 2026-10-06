# Reference solution: only used by scripts/verify-exercises.mjs. Shown in the Breakdown only after the Learner's own solution passes.
import math


def _check_codebooks(codebooks: list[list[list[float]]], dim: int | None = None) -> int:
    if not codebooks or any(not cb for cb in codebooks):
        raise ValueError("need at least one codebook and no empty codebook")
    d = dim if dim is not None else len(codebooks[0][0])
    if d < 1:
        raise ValueError("vectors must be non-empty")
    for cb in codebooks:
        for entry in cb:
            if len(entry) != d:
                raise ValueError("every codebook entry must have the same length as the vector")
    return d


def rvq_encode(vec: list[float], codebooks: list[list[list[float]]]) -> list[int]:
    _check_codebooks(codebooks, len(vec))
    residual = list(vec)
    indices: list[int] = []
    for cb in codebooks:
        best, best_d = 0, None
        for i, entry in enumerate(cb):
            d = sum((r - e) ** 2 for r, e in zip(residual, entry))
            if best_d is None or d < best_d:
                best, best_d = i, d
        indices.append(best)
        residual = [r - e for r, e in zip(residual, cb[best])]
    return indices


def rvq_decode(indices: list[int], codebooks: list[list[list[float]]]) -> list[float]:
    dim = _check_codebooks(codebooks)
    if len(indices) != len(codebooks):
        raise ValueError("one index per codebook")
    out = [0.0] * dim
    for i, cb in zip(indices, codebooks):
        if not 0 <= i < len(cb):
            raise ValueError("index out of range for its codebook")
        out = [o + e for o, e in zip(out, cb[i])]
    return out


def rvq_errors(vec: list[float], codebooks: list[list[list[float]]]) -> list[float]:
    _check_codebooks(codebooks, len(vec))
    residual = list(vec)
    errors = [math.sqrt(sum(r * r for r in residual))]
    for cb in codebooks:
        i = rvq_encode(residual, [cb])[0]
        residual = [r - e for r, e in zip(residual, cb[i])]
        errors.append(math.sqrt(sum(r * r for r in residual)))
    return errors
