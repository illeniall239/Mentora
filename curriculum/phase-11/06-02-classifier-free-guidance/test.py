import unittest

import torch

from solution import cfg, cfg_negative


class TestClassifierFreeGuidance(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)
        self.u = torch.randn(1, 4, 8, 8)
        self.c = torch.randn(1, 4, 8, 8)

    def test_scale_one_is_exactly_the_conditional(self):
        out = cfg(self.u, self.c, 1.0)
        self.assertTrue(torch.allclose(out, self.c, atol=1e-6))
        self.assertFalse(torch.allclose(out, self.u, atol=1e-3))

    def test_scale_zero_is_exactly_the_unconditional(self):
        self.assertTrue(torch.allclose(cfg(self.u, self.c, 0.0), self.u, atol=1e-6))

    def test_hand_values(self):
        u = torch.tensor([[1.0, 0.0]])
        c = torch.tensor([[3.0, 4.0]])
        self.assertTrue(torch.allclose(cfg(u, c, 2.0), torch.tensor([[5.0, 8.0]]), atol=1e-6))
        self.assertTrue(torch.allclose(cfg(u, c, 7.5), torch.tensor([[16.0, 30.0]]), atol=1e-6))
        self.assertTrue(torch.allclose(cfg(u, c, -1.0), torch.tensor([[-1.0, -4.0]]), atol=1e-6))

    def test_extrapolates_and_does_not_interpolate_or_flip_direction(self):
        s = 7.5
        out = cfg(self.u, self.c, s)
        self.assertTrue(torch.allclose(out, self.u + s * (self.c - self.u), atol=1e-6))
        self.assertFalse(torch.allclose(out, self.c + s * (self.c - self.u), atol=1e-3))
        self.assertFalse(torch.allclose(out, self.u + s * (self.u - self.c), atol=1e-3))
        self.assertFalse(torch.allclose(out, (1 - s) * self.c + s * self.u, atol=1e-3))

    def test_step_length_grows_linearly_with_scale(self):
        base = (self.c - self.u).norm().item()
        for s in (0.5, 1.0, 3.0, 12.0):
            got = (cfg(self.u, self.c, s) - self.u).norm().item()
            self.assertAlmostEqual(got, s * base, places=4)

    def test_shape_dtype_preserved_and_inputs_untouched(self):
        u0, c0 = self.u.clone(), self.c.clone()
        out = cfg(self.u, self.c, 3.0)
        self.assertEqual(out.shape, self.u.shape)
        self.assertEqual(out.dtype, self.u.dtype)
        self.assertTrue(torch.equal(self.u, u0))
        self.assertTrue(torch.equal(self.c, c0))
        flat = cfg(torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0]), 2.0)
        self.assertTrue(torch.allclose(flat, torch.tensor([5.0, 6.0]), atol=1e-6))

    def test_negative_prompt_replaces_the_unconditional_branch(self):
        n = torch.randn(1, 4, 8, 8)
        self.assertTrue(torch.allclose(cfg_negative(n, self.c, 1.0), self.c, atol=1e-6))
        self.assertTrue(torch.allclose(cfg_negative(n, self.c, 0.0), n, atol=1e-6))
        self.assertTrue(torch.allclose(cfg_negative(n, self.c, 5.0), n + 5.0 * (self.c - n), atol=1e-6))
        self.assertTrue(torch.allclose(cfg_negative(self.u, self.c, 4.0), cfg(self.u, self.c, 4.0), atol=1e-6))
        self.assertFalse(torch.allclose(cfg_negative(n, self.c, 4.0), cfg(self.u, self.c, 4.0), atol=1e-3))

    def test_guidance_moves_away_from_the_negative_prediction(self):
        n = torch.randn(1, 4, 8, 8)
        d1 = (cfg_negative(n, self.c, 2.0) - n).norm().item()
        d2 = (cfg_negative(n, self.c, 8.0) - n).norm().item()
        self.assertGreater(d2, d1)

    def test_shape_mismatch_raises(self):
        with self.assertRaises(ValueError):
            cfg(torch.zeros(1, 4, 8, 8), torch.zeros(1, 4, 8, 4), 7.5)
        with self.assertRaises(ValueError):
            cfg_negative(torch.zeros(3), torch.zeros(4), 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
