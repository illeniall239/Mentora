def attention(
    Q: list[list[float]], K: list[list[float]], V: list[list[float]], causal: bool = False
) -> tuple[list[list[float]], list[list[float]]]:
    """softmax(Q K^T / sqrt(d_k)) V with an optional causal mask; return (output, weights)."""
    raise NotImplementedError
