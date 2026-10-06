import torch


def incremental_attention(
    cache: dict[str, torch.Tensor], q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Append k, v to the cache and return (softmax(q K^T / sqrt(d)) V, new cache)."""
    raise NotImplementedError
