# Reference solution: only used by scripts/verify-exercises.mjs. Shown in the Breakdown only after the Learner's own solution passes.
import numpy as np


def rms_norm(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    rms = np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + eps)
    return x / rms


def _softmax(scores: np.ndarray) -> np.ndarray:
    shifted = scores - np.max(scores, axis=-1, keepdims=True)
    exps = np.exp(shifted)
    return exps / np.sum(exps, axis=-1, keepdims=True)


def _gelu(z: np.ndarray) -> np.ndarray:
    inner = np.sqrt(2.0 / np.pi) * (z + 0.044715 * z**3)
    return 0.5 * z * (1.0 + np.tanh(inner))


def multi_head_attention(x: np.ndarray, params: dict, n_heads: int, causal: bool = True) -> np.ndarray:
    if x.ndim != 2:
        raise ValueError("x must be 2-D (T, d_model)")
    t, d_model = x.shape
    if n_heads < 1 or d_model % n_heads != 0:
        raise ValueError("n_heads must be at least 1 and divide d_model")
    d_k = d_model // n_heads

    def split(w: np.ndarray) -> np.ndarray:
        # (T, d_model) -> (T, n_heads, d_k) -> (n_heads, T, d_k)
        return (x @ w).reshape(t, n_heads, d_k).transpose(1, 0, 2)

    q, k, v = split(params["wq"]), split(params["wk"]), split(params["wv"])
    scores = q @ k.transpose(0, 2, 1) / np.sqrt(d_k)
    if causal:
        future = np.triu(np.ones((t, t), dtype=bool), k=1)
        scores = np.where(future, -np.inf, scores)
    heads = _softmax(scores) @ v  # (n_heads, T, d_k)
    merged = heads.transpose(1, 0, 2).reshape(t, d_model)
    return merged @ params["wo"]


def transformer_block(x: np.ndarray, params: dict) -> np.ndarray:
    attn_in = rms_norm(x) * params["ln1_g"]
    h = x + multi_head_attention(attn_in, params, params["n_heads"], causal=True)
    ffn_in = rms_norm(h) * params["ln2_g"]
    return h + _gelu(ffn_in @ params["w1"]) @ params["w2"]
