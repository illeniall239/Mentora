import math
import unittest

import torch
from torch.distributions import Normal, kl_divergence

from solution import kl_gaussian, reparameterize


class TestReparameterize(unittest.TestCase):
    def test_uses_the_standard_deviation_not_the_variance(self):
        mu = torch.tensor([1.0, 2.0])
        log_var = torch.tensor([0.0, math.log(4.0)])  # sigma = [1, 2], variance = [1, 4]
        eps = torch.tensor([1.0, -1.0])
        got = reparameterize(mu, log_var, eps)
        self.assertTrue(torch.allclose(got, torch.tensor([2.0, 0.0]), atol=1e-6))

    def test_zero_eps_returns_the_mean_and_zero_log_var_adds_eps(self):
        torch.manual_seed(0)
        mu, log_var = torch.randn(4, 3), torch.randn(4, 3)
        self.assertTrue(torch.allclose(reparameterize(mu, log_var, torch.zeros_like(mu)), mu, atol=1e-7))
        eps = torch.randn(4, 3)
        self.assertTrue(torch.allclose(reparameterize(mu, torch.zeros_like(mu), eps), mu + eps, atol=1e-6))

    def test_sample_statistics_match_mu_and_sigma(self):
        torch.manual_seed(1)
        mu = torch.tensor([[-2.0, 5.0]])
        log_var = torch.tensor([[math.log(0.25), math.log(9.0)]])  # sigma = [0.5, 3]
        eps = torch.randn(40000, 2)
        z = reparameterize(mu.expand(40000, 2), log_var.expand(40000, 2), eps)
        self.assertTrue(torch.allclose(z.mean(0), mu[0], atol=0.05))
        self.assertTrue(torch.allclose(z.std(0), torch.tensor([0.5, 3.0]), atol=0.05))

    def test_gradients_reach_mu_and_log_var(self):
        mu = torch.zeros(2, 3, requires_grad=True)
        log_var = torch.zeros(2, 3, requires_grad=True)
        eps = torch.full((2, 3), 2.0)
        reparameterize(mu, log_var, eps).sum().backward()
        self.assertIsNotNone(mu.grad)
        self.assertIsNotNone(log_var.grad)
        self.assertTrue(torch.allclose(mu.grad, torch.ones(2, 3), atol=1e-6))
        # d/d log_var of exp(0.5 log_var) * 2 at log_var = 0 is 1.0
        self.assertTrue(torch.allclose(log_var.grad, torch.ones(2, 3), atol=1e-6))

    def test_shape_mismatch_raises(self):
        with self.assertRaises(ValueError):
            reparameterize(torch.zeros(2, 3), torch.zeros(2, 4), torch.zeros(2, 3))
        with self.assertRaises(ValueError):
            reparameterize(torch.zeros(2, 3), torch.zeros(2, 3), torch.zeros(3))


class TestKlGaussian(unittest.TestCase):
    def test_zero_at_the_prior(self):
        kl = kl_gaussian(torch.zeros(3, 5), torch.zeros(3, 5))
        self.assertEqual(tuple(kl.shape), (3,))
        self.assertTrue(torch.allclose(kl, torch.zeros(3), atol=1e-7))

    def test_hand_values(self):
        self.assertAlmostEqual(float(kl_gaussian(torch.tensor([[1.0, 0.0]]), torch.zeros(1, 2))), 0.5, places=5)
        self.assertAlmostEqual(
            float(kl_gaussian(torch.zeros(1, 1), torch.tensor([[1.0]]))), -0.5 * (2.0 - math.e), places=5
        )
        self.assertAlmostEqual(
            float(kl_gaussian(torch.zeros(1, 1), torch.tensor([[-1.0]]))),
            -0.5 * (1.0 - 1.0 - math.exp(-1.0)),
            places=5,
        )

    def test_matches_torch_distributions(self):
        torch.manual_seed(2)
        for b, d in ((1, 1), (4, 3), (16, 8)):
            mu, log_var = torch.randn(b, d), torch.randn(b, d)
            want = kl_divergence(Normal(mu, torch.exp(0.5 * log_var)), Normal(0.0, 1.0)).sum(-1)
            got = kl_gaussian(mu, log_var)
            self.assertEqual(tuple(got.shape), (b,))
            self.assertTrue(torch.allclose(got, want, atol=1e-5), f"b={b} d={d}")

    def test_non_negative_and_grows_away_from_the_prior(self):
        torch.manual_seed(3)
        kl = kl_gaussian(torch.randn(64, 6) * 3.0, torch.randn(64, 6))
        self.assertTrue(bool((kl >= 0).all()))
        self.assertGreater(float(kl.mean()), 1.0)
        far_mu = kl_gaussian(torch.full((1, 2), 4.0), torch.zeros(1, 2))
        near_mu = kl_gaussian(torch.full((1, 2), 0.5), torch.zeros(1, 2))
        self.assertGreater(float(far_mu), float(near_mu))
        wide = kl_gaussian(torch.zeros(1, 2), torch.full((1, 2), 2.0))
        narrow = kl_gaussian(torch.zeros(1, 2), torch.full((1, 2), -2.0))
        self.assertGreater(float(wide), 0.0)
        self.assertGreater(float(narrow), 0.0)

    def test_sums_over_the_latent_dimension_only(self):
        mu = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        kl = kl_gaussian(mu, torch.zeros(2, 3))
        self.assertEqual(tuple(kl.shape), (2,))
        self.assertTrue(torch.allclose(kl, torch.tensor([0.5, 0.0]), atol=1e-6))

    def test_gradients_flow(self):
        mu = torch.full((1, 2), 2.0, requires_grad=True)
        log_var = torch.zeros(1, 2, requires_grad=True)
        kl_gaussian(mu, log_var).sum().backward()
        self.assertTrue(torch.allclose(mu.grad, torch.full((1, 2), 2.0), atol=1e-6))

    def test_bad_shapes_raise(self):
        with self.assertRaises(ValueError):
            kl_gaussian(torch.zeros(2, 3), torch.zeros(2, 4))
        with self.assertRaises(ValueError):
            kl_gaussian(torch.zeros(5), torch.zeros(5))


if __name__ == "__main__":
    unittest.main(verbosity=2)
