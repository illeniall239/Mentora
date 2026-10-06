import unittest

import numpy as np
import torch
import torch.nn.functional as F

from solution import multi_head_attention, rms_norm, transformer_block


def make_params(d_model: int, n_heads: int, seed: int, ff_mult: int = 4) -> dict:
    torch.manual_seed(seed)
    d_ff = ff_mult * d_model
    g = lambda *shape: torch.randn(*shape, dtype=torch.float64).numpy()
    return {
        "n_heads": n_heads,
        "ln1_g": g(d_model),
        "wq": g(d_model, d_model),
        "wk": g(d_model, d_model),
        "wv": g(d_model, d_model),
        "wo": g(d_model, d_model),
        "ln2_g": g(d_model),
        "w1": g(d_model, d_ff),
        "w2": g(d_ff, d_model),
    }


def make_x(t: int, d_model: int, seed: int) -> np.ndarray:
    torch.manual_seed(1000 + seed)
    return torch.randn(t, d_model, dtype=torch.float64).numpy()


def torch_rms(z: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return z * torch.rsqrt(z.pow(2).mean(-1, keepdim=True) + eps)


def torch_mha(x: np.ndarray, p: dict, n_heads: int, causal: bool) -> np.ndarray:
    """Expected multi-head attention, built from torch primitives on the same weights."""
    X = torch.tensor(x, dtype=torch.float64)
    t, d_model = X.shape
    d_k = d_model // n_heads
    heads = lambda key: (X @ torch.tensor(p[key])).view(t, n_heads, d_k).transpose(0, 1)
    out = F.scaled_dot_product_attention(heads("wq"), heads("wk"), heads("wv"), is_causal=causal)
    return (out.transpose(0, 1).reshape(t, d_model) @ torch.tensor(p["wo"])).numpy()


def torch_block(x: np.ndarray, p: dict) -> np.ndarray:
    """Expected pre-LN block, built from torch primitives on the same weights."""
    X = torch.tensor(x, dtype=torch.float64)
    attn_in = (torch_rms(X) * torch.tensor(p["ln1_g"])).numpy()
    h = X + torch.tensor(torch_mha(attn_in, p, p["n_heads"], True))
    ffn_in = torch_rms(h) * torch.tensor(p["ln2_g"])
    ffn = F.gelu(ffn_in @ torch.tensor(p["w1"]), approximate="tanh") @ torch.tensor(p["w2"])
    return (h + ffn).numpy()


class TestRmsNorm(unittest.TestCase):
    def test_does_not_subtract_the_mean(self):
        out = rms_norm(np.array([[1.0, 2.0, 3.0]]))
        self.assertTrue(np.allclose(out, [[0.4629100, 0.9258201, 1.3887301]], atol=1e-6))
        # A LayerNorm would leave a zero-mean row here; RMSNorm must not.
        self.assertGreater(float(out.mean()), 0.5)
        self.assertTrue(np.allclose(rms_norm(np.array([[5.0, 5.0]])), [[1.0, 1.0]], atol=1e-6))

    def test_scale_invariant_and_row_independent(self):
        x = make_x(5, 8, 0)
        out = rms_norm(x)
        self.assertEqual(out.shape, x.shape)
        self.assertTrue(np.allclose(rms_norm(10.0 * x), out, atol=1e-8))
        # Row 0 must not care which other rows are present.
        alone = rms_norm(x[:1])
        self.assertTrue(np.allclose(alone[0], out[0], atol=1e-12))
        shuffled = rms_norm(x[::-1])
        self.assertTrue(np.allclose(shuffled[-1], out[0], atol=1e-12))

    def test_zero_row_and_eps(self):
        out = rms_norm(np.zeros((2, 4)))
        self.assertTrue(np.all(np.isfinite(out)))
        self.assertTrue(np.allclose(out, 0.0))
        x = np.array([[3.0, 4.0]])
        self.assertTrue(np.allclose(rms_norm(x, eps=0.0), x / np.sqrt(12.5), atol=1e-12))


class TestMultiHeadAttention(unittest.TestCase):
    def test_matches_torch_for_every_head_count(self):
        x = make_x(7, 8, 1)
        for n_heads in (1, 2, 4, 8):
            p = make_params(8, n_heads, seed=n_heads)
            for causal in (True, False):
                got = multi_head_attention(x, p, n_heads, causal)
                want = torch_mha(x, p, n_heads, causal)
                self.assertEqual(got.shape, (7, 8))
                self.assertTrue(np.allclose(got, want, atol=1e-10), f"heads={n_heads} causal={causal}")

    def test_causal_row_zero_sees_only_itself(self):
        x = make_x(6, 8, 2)
        p = make_params(8, 2, seed=3)
        out = multi_head_attention(x, p, 2, True)
        self.assertTrue(np.allclose(out[0], (x[0] @ p["wv"]) @ p["wo"], atol=1e-10))
        # Rewriting the future must not move earlier rows.
        x2 = x.copy()
        x2[4:] += 50.0
        out2 = multi_head_attention(x2, p, 2, True)
        self.assertTrue(np.allclose(out[:4], out2[:4], atol=1e-10))
        self.assertFalse(np.allclose(out[5], out2[5], atol=1e-3))

    def test_non_causal_row_zero_does_see_the_future(self):
        x = make_x(6, 8, 4)
        p = make_params(8, 2, seed=5)
        out = multi_head_attention(x, p, 2, False)
        self.assertFalse(np.allclose(out[0], (x[0] @ p["wv"]) @ p["wo"], atol=1e-3))

    def test_head_count_changes_values_not_shape(self):
        x = make_x(5, 8, 6)
        p = make_params(8, 1, seed=7)
        one = multi_head_attention(x, p, 1, True)
        four = multi_head_attention(x, p, 4, True)
        self.assertEqual(one.shape, four.shape)
        self.assertFalse(np.allclose(one, four, atol=1e-3))

    def test_rejects_bad_head_counts_and_shapes(self):
        p = make_params(8, 2, seed=8)
        with self.assertRaises(ValueError):
            multi_head_attention(make_x(4, 8, 9), p, 3, True)
        with self.assertRaises(ValueError):
            multi_head_attention(make_x(4, 8, 9), p, 0, True)
        with self.assertRaises(ValueError):
            multi_head_attention(np.zeros((2, 4, 8)), p, 2, True)


class TestTransformerBlock(unittest.TestCase):
    def test_matches_the_torch_block(self):
        for d_model, n_heads, t, seed in ((8, 2, 6, 10), (16, 4, 9, 11), (4, 1, 3, 12)):
            p = make_params(d_model, n_heads, seed=seed)
            x = make_x(t, d_model, seed)
            got = transformer_block(x, p)
            self.assertEqual(got.shape, (t, d_model))
            self.assertTrue(np.allclose(got, torch_block(x, p), atol=1e-10), f"d_model={d_model}")

    def test_pre_ln_with_dead_sublayers_is_the_identity(self):
        p = make_params(8, 2, seed=13)
        p["wo"] = np.zeros_like(p["wo"])
        p["w2"] = np.zeros_like(p["w2"])
        x = make_x(5, 8, 13)
        # Post-LN would return a normalized x here, not x.
        self.assertTrue(np.allclose(transformer_block(x, p), x, atol=1e-12))

    def test_only_attention_mixes_tokens(self):
        p = make_params(8, 2, seed=14)
        p["wo"] = np.zeros_like(p["wo"])  # attention contributes nothing, FFN still runs
        x = make_x(5, 8, 14)
        out = transformer_block(x, p)
        x2 = x.copy()
        x2[1] += 7.0
        out2 = transformer_block(x2, p)
        self.assertTrue(np.allclose(out[0], out2[0], atol=1e-12))
        self.assertTrue(np.allclose(out[2:], out2[2:], atol=1e-12))
        self.assertFalse(np.allclose(out[1], out2[1], atol=1e-3))
        # And with attention alive, an early token does reach a later one.
        p["wo"] = make_params(8, 2, seed=15)["wo"]
        self.assertFalse(np.allclose(transformer_block(x, p)[4], transformer_block(x2, p)[4], atol=1e-6))

    def test_residual_stream_is_never_normalized(self):
        p = make_params(8, 2, seed=16)
        x = make_x(4, 8, 16) * 100.0
        out = transformer_block(x, p)
        # A post-LN block would squash the output to roughly unit RMS whatever the input scale.
        self.assertGreater(float(np.sqrt((out**2).mean())), 10.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
