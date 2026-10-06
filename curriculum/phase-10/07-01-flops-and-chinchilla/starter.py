def train_flops(n_params: int | float, n_tokens: int | float) -> float:
    """Approximate training compute 6 * N * D; ValueError if either is not positive."""
    raise NotImplementedError


def chinchilla_tokens(n_params: int | float) -> float:
    """Compute-optimal training tokens, about 20 * N; ValueError if N is not positive."""
    raise NotImplementedError
