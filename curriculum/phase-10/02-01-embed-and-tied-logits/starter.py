import torch


def embed(ids: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
    """Look up rows of table (V, D) for every id; result has shape ids.shape + (D,)."""
    raise NotImplementedError


def tied_logits(h: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
    """Logits h @ table.T, shape (..., V), using the embedding table as the output layer."""
    raise NotImplementedError
