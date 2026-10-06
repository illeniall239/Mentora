# Reference solution: only used by scripts/verify-exercises.mjs. Shown in the Breakdown only after the Learner's own solution passes.
import torch


def embed(ids: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
    return table[ids]


def tied_logits(h: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
    if h.shape[-1] != table.shape[1]:
        raise ValueError(f"h has {h.shape[-1]} features but table rows have {table.shape[1]}")
    return h @ table.T
