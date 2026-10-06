def full_param_count(d_in: int, d_out: int) -> int:
    """Entries of a (d_in, d_out) weight matrix; ValueError if a dimension is below 1."""
    raise NotImplementedError


def lora_param_count(d_in: int, d_out: int, r: int) -> int:
    """Trainable entries of B (d_in, r) and A (r, d_out); ValueError for bad dims or r outside 1..min(d_in, d_out)."""
    raise NotImplementedError
