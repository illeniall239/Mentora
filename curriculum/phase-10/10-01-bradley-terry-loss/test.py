import math
import unittest

import torch

from solution import bradley_terry_loss

LN2 = math.log(2.0)


def t(*xs):
    return torch.tensor(list(xs), dtype=torch.float32)


class TestBradleyTerryLoss(unittest.TestCase):
    def test_equal_rewards_give_ln_two(self):
        for n in (1, 3, 8):
            loss = bradley_terry_loss(torch.zeros(n), torch.zeros(n))
            self.assertEqual(loss.shape, torch.Size([]))
            self.assertAlmostEqual(loss.item(), LN2, places=6)
        loss = bradley_terry_loss(t(4.0, -2.0, 7.5), t(4.0, -2.0, 7.5))  # equal, but not zero
        self.assertAlmostEqual(loss.item(), LN2, places=6)

    def test_hand_computed_values_and_sign(self):
        self.assertAlmostEqual(bradley_terry_loss(t(1.0), t(0.0)).item(), 0.31326169, places=6)
        self.assertAlmostEqual(bradley_terry_loss(t(0.0), t(1.0)).item(), 1.31326169, places=6)
        self.assertAlmostEqual(bradley_terry_loss(t(5.0), t(0.0)).item(), 0.00671535, places=6)
        # a bigger chosen reward must cost less
        margins = [-3.0, -1.0, 0.0, 0.5, 2.0, 6.0]
        losses = [bradley_terry_loss(t(m), t(0.0)).item() for m in margins]
        self.assertEqual(losses, sorted(losses, reverse=True))
        self.assertTrue(all(x > 0 for x in losses))

    def test_mean_not_sum(self):
        single = bradley_terry_loss(t(3.0), t(2.0)).item()
        self.assertAlmostEqual(bradley_terry_loss(t(3.0, 3.0, 3.0), t(2.0, 2.0, 2.0)).item(), single, places=6)
        self.assertAlmostEqual(bradley_terry_loss(t(3.0, 0.0), t(2.0, 1.0)).item(), 0.81326169, places=6)

    def test_only_the_difference_matters(self):
        torch.manual_seed(0)
        r_c, r_r = torch.randn(6), torch.randn(6)
        base = bradley_terry_loss(r_c, r_r).item()
        for shift in (1.0, -4.0, 1000.0):
            self.assertAlmostEqual(bradley_terry_loss(r_c + shift, r_r + shift).item(), base, places=5)

    def test_swapping_the_pair_follows_the_exact_identity(self):
        torch.manual_seed(1)
        r_c, r_r = torch.randn(5) * 2, torch.randn(5) * 2
        forward = bradley_terry_loss(r_c, r_r).item()
        backward = bradley_terry_loss(r_r, r_c).item()
        self.assertAlmostEqual(backward, forward + (r_c - r_r).mean().item(), places=5)
        self.assertGreater(bradley_terry_loss(t(0.0), t(2.0)).item(), LN2)
        self.assertLess(bradley_terry_loss(t(2.0), t(0.0)).item(), LN2)

    def test_numerically_stable_at_a_large_negative_margin(self):
        loss = bradley_terry_loss(t(-100.0), t(100.0))
        self.assertTrue(torch.isfinite(loss), msg=f"got {loss}")
        self.assertAlmostEqual(loss.item(), 200.0, places=3)
        big = bradley_terry_loss(t(80.0), t(-80.0))
        self.assertTrue(torch.isfinite(big))
        self.assertLess(big.item(), 1e-6)

    def test_gradients_match_the_analytic_form(self):
        r_c = torch.tensor([1.0, -0.5, 3.0], requires_grad=True)
        r_r = torch.tensor([0.0, 0.5, 0.0], requires_grad=True)
        bradley_terry_loss(r_c, r_r).backward()
        want = -torch.sigmoid(r_r.detach() - r_c.detach()) / 3
        torch.testing.assert_close(r_c.grad, want, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(r_r.grad, -want, atol=1e-6, rtol=1e-5)

    def test_bad_shapes_raise(self):
        with self.assertRaises(ValueError):
            bradley_terry_loss(torch.zeros(3), torch.zeros(4))
        with self.assertRaises(ValueError):
            bradley_terry_loss(torch.zeros(2, 2), torch.zeros(2, 2))
        with self.assertRaises(ValueError):
            bradley_terry_loss(torch.zeros(0), torch.zeros(0))


if __name__ == "__main__":
    unittest.main(verbosity=2)
