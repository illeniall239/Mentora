def kv_cache_bytes(layers: int, kv_heads: int, head_dim: int, seq_len: int, batch: int, bytes_per: int) -> int:
    """Bytes to cache K and V for every layer: 2 * layers * kv_heads * head_dim * seq_len * batch * bytes_per."""
    raise NotImplementedError


def moe_active_params(shared: int, per_expert: int, n_experts: int, top_k: int) -> int:
    """Parameters used per token: shared + top_k * per_expert; ValueError on invalid counts."""
    raise NotImplementedError
