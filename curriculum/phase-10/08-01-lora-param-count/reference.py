# Reference solution: only used by scripts/verify-exercises.mjs. Shown in the Breakdown only after the Learner's own solution passes.
def full_param_count(d_in: int, d_out: int) -> int:
    if d_in < 1 or d_out < 1:
        raise ValueError("dimensions must be at least 1")
    return d_in * d_out


def lora_param_count(d_in: int, d_out: int, r: int) -> int:
    if d_in < 1 or d_out < 1:
        raise ValueError("dimensions must be at least 1")
    if r < 1 or r > min(d_in, d_out):
        raise ValueError("rank must be between 1 and min(d_in, d_out)")
    return r * (d_in + d_out)  # B is (d_in, r), A is (r, d_out); W stays frozen
