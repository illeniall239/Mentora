import numpy as np


def rms_norm(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Divide every row by the root-mean-square of its own features; no mean subtraction, no gain."""
    raise NotImplementedError


def multi_head_attention(x: np.ndarray, params: dict, n_heads: int, causal: bool = True) -> np.ndarray:
    """Project x to Q/K/V, attend in n_heads parallel heads, concatenate, apply the output projection."""
    raise NotImplementedError


def transformer_block(x: np.ndarray, params: dict) -> np.ndarray:
    """One pre-LN causal decoder block: x + attn(norm(x)), then h + ffn(norm(h))."""
    raise NotImplementedError
