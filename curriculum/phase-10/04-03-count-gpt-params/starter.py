def count_params(
    vocab: int,
    d_model: int,
    n_layers: int,
    n_heads: int,
    ff_mult: int,
    tied: bool,
    n_positions: int = 1024,
) -> int:
    """Exact number of learned scalars in a GPT-2-style decoder-only model."""
    raise NotImplementedError
