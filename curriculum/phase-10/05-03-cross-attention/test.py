import unittest

import numpy as np
import torch
import torch.nn.functional as F

from solution import cross_attention

D_MODEL, D_K, D_V = 8, 6, 5


def make_params(seed: int) -> dict:
    torch.manual_seed(seed)
    g = lambda *shape: torch.randn(*shape, dtype=torch.float64).numpy()
    return {"wq": g(D_MODEL, D_K), "wk": g(D_MODEL, D_K), "wv": g(D_MODEL, D_V), "wo": g(D_V, D_MODEL)}


def make_x(rows: int, seed: int) -> np.ndarray:
    torch.manual_seed(500 + seed)
    return torch.randn(rows, D_MODEL, dtype=torch.float64).numpy()


def torch_cross(dec_x: np.ndarray, enc_out: np.ndarray, p: dict, enc_mask=None) -> np.ndarray:
    """Expected output, built from torch primitives: Q from the decoder, K and V from the encoder."""
    dec = torch.tensor(dec_x, dtype=torch.float64)
    enc = torch.tensor(enc_out, dtype=torch.float64)
    q = dec @ torch.tensor(p["wq"])
    k = enc @ torch.tensor(p["wk"])
    v = enc @ torch.tensor(p["wv"])
    attn_mask = None
    if enc_mask is not None:
        attn_mask = torch.tensor(enc_mask, dtype=torch.bool).view(1, -1).expand(dec.shape[0], -1)
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
    return (out @ torch.tensor(p["wo"])).numpy()


class TestCrossAttention(unittest.TestCase):
    def test_shapes_follow_the_decoder_length(self):
        p = make_params(0)
        dec_x, enc_out = make_x(3, 0), make_x(5, 1)
        out, weights = cross_attention(dec_x, enc_out, p)
        self.assertEqual(out.shape, (3, D_MODEL))
        self.assertEqual(weights.shape, (3, 5))
        # The same call with the arguments swapped must not produce the same thing.
        swapped, swapped_w = cross_attention(enc_out, dec_x, p)
        self.assertEqual(swapped.shape, (5, D_MODEL))
        self.assertEqual(swapped_w.shape, (5, 3))

    def test_weights_sum_to_one_over_encoder_positions(self):
        p = make_params(1)
        out, weights = cross_attention(make_x(4, 2), make_x(7, 3), p)
        self.assertEqual(weights.shape, (4, 7))
        self.assertTrue(np.allclose(weights.sum(axis=1), np.ones(4), atol=1e-12))
        self.assertTrue(np.all(weights >= 0.0))
        self.assertEqual(out.shape, (4, D_MODEL))

    def test_matches_the_torch_reference(self):
        for n, m, seed in ((3, 5, 4), (1, 9, 5), (6, 2, 6)):
            p = make_params(seed)
            dec_x, enc_out = make_x(n, seed), make_x(m, seed + 100)
            out, _ = cross_attention(dec_x, enc_out, p)
            self.assertTrue(np.allclose(out, torch_cross(dec_x, enc_out, p), atol=1e-10), f"{n}x{m}")

    def test_decoder_positions_do_not_mix(self):
        p = make_params(7)
        dec_x, enc_out = make_x(4, 7), make_x(5, 8)
        out, _ = cross_attention(dec_x, enc_out, p)
        bumped = dec_x.copy()
        bumped[2] += 3.0
        out2, _ = cross_attention(bumped, enc_out, p)
        for i in (0, 1, 3):
            self.assertTrue(np.allclose(out[i], out2[i], atol=1e-12), f"row {i} must not move")
        self.assertFalse(np.allclose(out[2], out2[2], atol=1e-6))

    def test_every_decoder_row_sees_the_whole_encoder(self):
        p = make_params(9)
        dec_x, enc_out = make_x(4, 9), make_x(5, 10)
        out, weights = cross_attention(dec_x, enc_out, p)
        # No causal mask here: even decoder row 0 attends to the last encoder position.
        self.assertTrue(np.all(weights > 0.0))
        bumped = enc_out.copy()
        bumped[-1] += 3.0
        out2, _ = cross_attention(dec_x, bumped, p)
        for i in range(4):
            self.assertFalse(np.allclose(out[i], out2[i], atol=1e-6), f"row {i} should see encoder position 4")

    def test_single_encoder_position(self):
        p = make_params(11)
        dec_x, enc_out = make_x(3, 11), make_x(1, 12)
        out, weights = cross_attention(dec_x, enc_out, p)
        self.assertTrue(np.allclose(weights, np.ones((3, 1)), atol=1e-12))
        want = (enc_out[0] @ p["wv"]) @ p["wo"]
        for row in out:
            self.assertTrue(np.allclose(row, want, atol=1e-10))

    def test_enc_mask_blocks_before_the_softmax(self):
        p = make_params(13)
        dec_x, enc_out = make_x(3, 13), make_x(5, 14)
        keep = [True, True, False, True, False]
        out, weights = cross_attention(dec_x, enc_out, p, keep)
        self.assertTrue(np.allclose(weights[:, 2], 0.0, atol=0.0))
        self.assertTrue(np.allclose(weights[:, 4], 0.0, atol=0.0))
        # Zeroing after the softmax would leave these rows short of 1.
        self.assertTrue(np.allclose(weights.sum(axis=1), np.ones(3), atol=1e-12))
        self.assertTrue(np.allclose(out, torch_cross(dec_x, enc_out, p, keep), atol=1e-10))
        # Blocking a position must equal never having had it.
        kept_rows = enc_out[[0, 1, 3]]
        sub_out, sub_w = cross_attention(dec_x, kept_rows, p)
        self.assertTrue(np.allclose(out, sub_out, atol=1e-10))
        self.assertTrue(np.allclose(weights[:, [0, 1, 3]], sub_w, atol=1e-10))

    def test_rejects_bad_shapes_and_empty_masks(self):
        p = make_params(15)
        dec_x, enc_out = make_x(3, 15), make_x(5, 16)
        with self.assertRaises(ValueError):
            cross_attention(dec_x, enc_out, p, [False] * 5)
        with self.assertRaises(ValueError):
            cross_attention(dec_x, enc_out, p, [True] * 4)
        with self.assertRaises(ValueError):
            cross_attention(dec_x, np.zeros((5, D_MODEL + 1)), p)
        with self.assertRaises(ValueError):
            cross_attention(np.zeros((2, 3, D_MODEL)), enc_out, p)


if __name__ == "__main__":
    unittest.main(verbosity=2)
