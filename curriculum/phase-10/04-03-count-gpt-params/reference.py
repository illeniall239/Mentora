# Reference solution: only used by scripts/verify-exercises.mjs. Shown in the Breakdown only after the Learner's own solution passes.
def count_params(
    vocab: int,
    d_model: int,
    n_layers: int,
    n_heads: int,
    ff_mult: int,
    tied: bool,
    n_positions: int = 1024,
) -> int:
    positives = {
        "vocab": vocab,
        "d_model": d_model,
        "n_heads": n_heads,
        "ff_mult": ff_mult,
        "n_positions": n_positions,
    }
    for name, value in positives.items():
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if not isinstance(n_layers, int) or isinstance(n_layers, bool) or n_layers < 0:
        raise ValueError("n_layers must be a non-negative integer")
    if d_model % n_heads != 0:
        raise ValueError("n_heads must divide d_model")

    d_ff = ff_mult * d_model
    embeddings = vocab * d_model + n_positions * d_model

    layer_norm = 2 * d_model
    qkv = 3 * (d_model * d_model + d_model)
    out_proj = d_model * d_model + d_model
    ffn = (d_model * d_ff + d_ff) + (d_ff * d_model + d_model)
    per_block = layer_norm + qkv + out_proj + layer_norm + ffn

    head = 0 if tied else vocab * d_model
    return embeddings + n_layers * per_block + layer_norm + head
