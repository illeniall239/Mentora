import unittest

import torch

from solution import kl_estimate, kl_penalized_reward


class TestKlPenalizedReward(unittest.TestCase):
    def test_no_drift_means_no_penalty(self):
        logp = torch.tensor([[-1.0, -2.0, -0.5]])
        r = torch.tensor([4.0])
        self.assertAlmostEqual(kl_estimate(logp, logp.clone()).item(), 0.0, places=6)
        out = kl_penalized_reward(r, logp, logp.clone(), 0.2)
        self.assertEqual(out.shape, torch.Size([1]))
        torch.testing.assert_close(out, r, atol=1e-5, rtol=0)

    def test_penalty_is_summed_over_tokens_and_subtracted(self):
        logp_policy = torch.full((1, 4), -1.0)
        logp_ref = torch.full((1, 4), -1.5)
        r = torch.tensor([4.0])
        self.assertAlmostEqual(kl_estimate(logp_policy, logp_ref).item(), 2.0, places=5)
        for beta, want in [(0.0, 4.0), (0.2, 3.6), (1.0, 2.0), (3.0, -2.0)]:
            got = kl_penalized_reward(r, logp_policy, logp_ref, beta)
            self.assertAlmostEqual(got.item(), want, places=5, msg=f"beta={beta}")

    def test_kl_estimate_averages_over_the_batch(self):
        logp_policy = torch.tensor([[-1.0, -1.0, -1.0, -1.0], [-2.0, -2.0, -2.0, -2.0]])
        logp_ref = torch.tensor([[-1.5, -1.5, -1.5, -1.5], [-2.0, -2.0, -2.0, -2.0]])
        est = kl_estimate(logp_policy, logp_ref)
        self.assertEqual(est.shape, torch.Size([]))
        self.assertAlmostEqual(est.item(), 1.0, places=5)  # mean of 2.0 and 0.0, not the sum
        out = kl_penalized_reward(torch.tensor([1.0, 1.0]), logp_policy, logp_ref, 0.5)
        torch.testing.assert_close(out, torch.tensor([0.0, 1.0]), atol=1e-5, rtol=0)

    def test_a_single_sample_estimate_may_be_negative(self):
        est = kl_estimate(torch.tensor([[-3.0]]), torch.tensor([[-1.0]]))
        self.assertAlmostEqual(est.item(), -2.0, places=5)
        out = kl_penalized_reward(torch.tensor([0.0]), torch.tensor([[-3.0]]), torch.tensor([[-1.0]]), 0.5)
        self.assertAlmostEqual(out.item(), 1.0, places=5)  # the penalty really did add here

    def test_matches_the_two_pieces_on_random_batches(self):
        torch.manual_seed(0)
        logp_policy = -torch.rand(5, 7) * 3
        logp_ref = -torch.rand(5, 7) * 3
        r = torch.randn(5)
        per_seq = (logp_policy - logp_ref).sum(dim=1)
        out = kl_penalized_reward(r, logp_policy, logp_ref, 0.3)
        self.assertEqual(out.shape, torch.Size([5]))
        torch.testing.assert_close(out, r - 0.3 * per_seq, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(kl_estimate(logp_policy, logp_ref), per_seq.mean(), atol=1e-5, rtol=1e-5)
        # drifting further can only lower the objective
        further = logp_policy + 0.4
        self.assertTrue(torch.all(kl_penalized_reward(r, further, logp_ref, 0.3) < out))

    def test_gradients_flow_through_the_penalty(self):
        logp_policy = torch.full((2, 3), -1.0, requires_grad=True)
        logp_ref = torch.full((2, 3), -1.0)
        r = torch.tensor([1.0, 2.0])
        kl_penalized_reward(r, logp_policy, logp_ref, 0.25).sum().backward()
        torch.testing.assert_close(logp_policy.grad, torch.full((2, 3), -0.25), atol=1e-6, rtol=0)
        logp_policy.grad = None
        kl_estimate(logp_policy, logp_ref).backward()
        torch.testing.assert_close(logp_policy.grad, torch.full((2, 3), 0.5), atol=1e-6, rtol=0)

    def test_inputs_are_not_modified(self):
        logp_policy = torch.full((2, 3), -1.0)
        logp_ref = torch.full((2, 3), -1.5)
        r = torch.tensor([1.0, 2.0])
        kl_penalized_reward(r, logp_policy, logp_ref, 0.5)
        kl_estimate(logp_policy, logp_ref)
        torch.testing.assert_close(logp_policy, torch.full((2, 3), -1.0), atol=0, rtol=0)
        torch.testing.assert_close(logp_ref, torch.full((2, 3), -1.5), atol=0, rtol=0)
        torch.testing.assert_close(r, torch.tensor([1.0, 2.0]), atol=0, rtol=0)

    def test_bad_arguments_raise(self):
        p, q = torch.full((2, 3), -1.0), torch.full((2, 3), -1.5)
        r = torch.tensor([1.0, 2.0])
        with self.assertRaises(ValueError):
            kl_estimate(p, torch.full((2, 4), -1.5))
        with self.assertRaises(ValueError):
            kl_estimate(torch.full((3,), -1.0), torch.full((3,), -1.5))
        with self.assertRaises(ValueError):
            kl_estimate(torch.zeros(0, 3), torch.zeros(0, 3))
        with self.assertRaises(ValueError):
            kl_penalized_reward(torch.tensor([1.0]), p, q, 0.5)
        with self.assertRaises(ValueError):
            kl_penalized_reward(r.reshape(2, 1), p, q, 0.5)
        with self.assertRaises(ValueError):
            kl_penalized_reward(r, p, q, -0.1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
