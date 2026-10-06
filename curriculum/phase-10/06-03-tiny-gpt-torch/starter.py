import torch
import torch.nn as nn


class TinyGPT(nn.Module):
    def __init__(self, vocab: int, block: int, d_model: int, n_heads: int, n_layers: int):
        """Token + learned position embeddings, n_layers pre-LN causal blocks, final LayerNorm, LM head."""
        super().__init__()
        raise NotImplementedError

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Map (B, T) ids to ((B, T, vocab) logits, cross-entropy loss or None); ValueError if T > block."""
        raise NotImplementedError

    def generate(self, ids: torch.Tensor, n: int) -> torch.Tensor:
        """Append n greedy (argmax) tokens, feeding only the last `block` tokens each step; return (B, T + n)."""
        raise NotImplementedError
