import math
import unittest

import torch

from solution import dpo_loss

LN2 = math.log(2.0)


def t(*xs):
    return torch.tensor(list(xs), dtype=torch.float32)


class TestDpoLoss(unittest.TestCase):
    def test_policy_equal_to_reference_gives_ln_two(self):
        pi_c, pi_r = t(-1.0, -4.0), t(-2.0, -0.5)
        for beta in (0.1, 1.0, 5.0):
            loss = dpo_loss(pi_c, pi_r, pi_c.clone(), pi_r.clone(), beta)
            self.assertEqual(loss.shape, torch.Size([]))
            self.assertAlmostEqual(loss.item(), LN2, places=5, msg=f"beta={beta}")

    def test_beta_scales_inside_the_log_sigmoid(self):
        args = (t(-1.0), t(-2.0), t(-1.0), t(-1.0))  # ratio margin 1.0
        self.assertAlmostEqual(dpo_loss(*args, 0.5).item(), 0.47407698, places=5)
        self.assertAlmostEqual(dpo_loss(*args, 1.0).item(), 0.31326169, places=5)
        self.assertAlmostEqual(dpo_loss(*args, 2.0).item(), 0.12692801, places=5)
        half, full = dpo_loss(*args, 0.5).item(), dpo_loss(*args, 1.0).item()
        self.assertNotAlmostEqual(full, 2 * half, places=3)  # not a factor pulled outside

    def test_the_reference_log_probs_are_used(self):
        with_ref = dpo_loss(t(-1.0), t(-2.0), t(-0.5), t(-2.0), 0.5)
        self.assertAlmostEqual(with_ref.item(), 0.82593942, places=5)
        ignoring_ref = dpo_loss(t(-1.0), t(-2.0), t(-1.0), t(-1.0), 0.5)
        self.assertAlmostEqual(ignoring_ref.item(), 0.47407698, places=5)
        # a reference that already prefers the chosen response leaves less for the policy to learn
        torch.manual_seed(0)
        pi_c, pi_r = -torch.rand(4) * 2, -torch.rand(4) * 2
        ref = -torch.rand(4) * 2
        self.assertNotAlmostEqual(
            dpo_loss(pi_c, pi_r, ref, ref.clone(), 0.3).item(),
            dpo_loss(pi_c, pi_r, ref + 1.0, ref.clone(), 0.3).item(),
            places=4,
        )

    def test_a_growing_chosen_margin_lowers_the_loss(self):
        ref_c, ref_r = t(-2.0), t(-2.0)
        losses = [dpo_loss(t(-2.0 + m), t(-2.0), ref_c, ref_r, 0.4).item() for m in (-2.0, -0.5, 0.0, 0.5, 2.0, 5.0)]
        self.assertEqual(losses, sorted(losses, reverse=True))
        self.assertAlmostEqual(losses[2], LN2, places=5)
        self.assertTrue(all(x > 0 for x in losses))

    def test_mean_over_the_batch(self):
        single = dpo_loss(t(-1.0), t(-2.0), t(-1.0), t(-1.0), 1.0).item()
        batch = dpo_loss(t(-1.0, -1.0, -1.0), t(-2.0, -2.0, -2.0), t(-1.0, -1.0, -1.0), t(-1.0, -1.0, -1.0), 1.0)
        self.assertAlmostEqual(batch.item(), single, places=5)
        mixed = dpo_loss(t(-1.0, -2.0), t(-2.0, -1.0), t(-1.0, -2.0), t(-1.0, -2.0), 1.0)
        self.assertAlmostEqual(mixed.item(), (0.31326169 + 1.31326169) / 2, places=5)

    def test_numerically_stable_at_a_large_negative_logit(self):
        loss = dpo_loss(t(-100.0), t(0.0), t(0.0), t(0.0), 2.0)
        self.assertTrue(torch.isfinite(loss), msg=f"got {loss}")
        self.assertAlmostEqual(loss.item(), 200.0, places=2)
        tiny = dpo_loss(t(0.0), t(-100.0), t(0.0), t(0.0), 2.0)
        self.assertTrue(torch.isfinite(tiny))
        self.assertLess(tiny.item(), 1e-6)

    def test_gradients_match_the_analytic_form(self):
        pi_c = torch.tensor([-1.0, -3.0], requires_grad=True)
        pi_r = torch.tensor([-2.0, -1.0], requires_grad=True)
        ref_c, ref_r = t(-1.5, -2.5), t(-2.0, -2.0)
        beta = 0.4
        dpo_loss(pi_c, pi_r, ref_c, ref_r, beta).backward()
        logits = beta * ((pi_c.detach() - ref_c) - (pi_r.detach() - ref_r))
        want = -beta * torch.sigmoid(-logits) / 2
        torch.testing.assert_close(pi_c.grad, want, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(pi_r.grad, -want, atol=1e-6, rtol=1e-5)

    def test_bad_arguments_raise(self):
        a, b = t(-1.0, -2.0), t(-2.0, -3.0)
        with self.assertRaises(ValueError):
            dpo_loss(a, b, a, t(-1.0), 0.1)
        with self.assertRaises(ValueError):
            dpo_loss(a.reshape(2, 1), b.reshape(2, 1), a.reshape(2, 1), b.reshape(2, 1), 0.1)
        with self.assertRaises(ValueError):
            dpo_loss(torch.zeros(0), torch.zeros(0), torch.zeros(0), torch.zeros(0), 0.1)
        with self.assertRaises(ValueError):
            dpo_loss(a, b, a, b, 0.0)
        with self.assertRaises(ValueError):
            dpo_loss(a, b, a, b, -0.5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
