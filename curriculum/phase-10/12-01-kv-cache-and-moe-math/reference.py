# Reference solution: only used by scripts/verify-exercises.mjs. Shown in the Breakdown only after the Learner's own solution passes.
def kv_cache_bytes(layers: int, kv_heads: int, head_dim: int, seq_len: int, batch: int, bytes_per: int) -> int:
    if min(layers, kv_heads, head_dim, seq_len, batch, bytes_per) < 1:
        raise ValueError("every argument must be at least 1")
    return 2 * layers * kv_heads * head_dim * seq_len * batch * bytes_per


def moe_active_params(shared: int, per_expert: int, n_experts: int, top_k: int) -> int:
    if n_experts < 1 or top_k < 1 or top_k > n_experts or shared < 0 or per_expert < 0:
        raise ValueError("invalid MoE configuration")
    return shared + top_k * per_expert
