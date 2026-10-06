import unittest

import numpy as np

from solution import reinforce_grad


def objective(logits, actions, returns, baseline=0.0):
    """J = mean_i (R_i - b) * log softmax(logits_i)[a_i], computed independently of the solution."""
    logits = np.asarray(logits, dtype=float)
    n = logits.shape[0]
    shifted = logits - logits.max(axis=1, keepdims=True)
    logp = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    chosen = logp[np.arange(n), np.asarray(actions)]
    return float(np.mean((np.asarray(returns, dtype=float) - baseline) * chosen))


def numeric_grad(logits, actions, returns, baseline=0.0, eps=1e-5):
    logits = np.asarray(logits, dtype=float)
    out = np.zeros_like(logits)
    for i in range(logits.shape[0]):
        for j in range(logits.shape[1]):
            up, down = logits.copy(), logits.copy()
            up[i, j] += eps
            down[i, j] -= eps
            out[i, j] = (objective(up, actions, returns, baseline) - objective(down, actions, returns, baseline)) / (2 * eps)
    return out


class TestReinforceGrad(unittest.TestCase):
    def test_single_step_by_hand(self):
        g = reinforce_grad(np.zeros((1, 2)), np.array([0]), np.array([1.0]))
        np.testing.assert_allclose(g, [[0.5, -0.5]], atol=1e-6)
        g = reinforce_grad(np.zeros((1, 2)), np.array([0]), np.array([-1.0]))
        np.testing.assert_allclose(g, [[-0.5, 0.5]], atol=1e-6)
        g = reinforce_grad(np.array([[0.0, np.log(3.0)]]), np.array([1]), np.array([1.0]))
        np.testing.assert_allclose(g, [[-0.25, 0.25]], atol=1e-6)  # p = [0.25, 0.75]

    def test_mean_over_the_batch_and_rows_sum_to_zero(self):
        logits = np.zeros((2, 2))
        g = reinforce_grad(logits, np.array([0, 0]), np.array([1.0, 1.0]))
        np.testing.assert_allclose(g, [[0.25, -0.25], [0.25, -0.25]], atol=1e-6)
        rng = np.random.default_rng(0)
        g = reinforce_grad(rng.standard_normal((5, 4)), rng.integers(0, 4, 5), rng.standard_normal(5), 0.3)
        self.assertEqual(g.shape, (5, 4))
        np.testing.assert_allclose(g.sum(axis=1), np.zeros(5), atol=1e-9)

    def test_matches_a_numeric_gradient(self):
        rng = np.random.default_rng(7)
        for baseline in (0.0, 1.5, -2.0):
            logits = rng.standard_normal((6, 4)) * 1.5
            actions = rng.integers(0, 4, 6)
            returns = rng.standard_normal(6) * 3.0
            np.testing.assert_allclose(
                reinforce_grad(logits, actions, returns, baseline),
                numeric_grad(logits, actions, returns, baseline),
                atol=1e-6,
                err_msg=f"baseline={baseline}",
            )

    def test_ascent_direction_raises_the_sampled_action(self):
        rng = np.random.default_rng(3)
        logits = rng.standard_normal((4, 5))
        actions = rng.integers(0, 5, 4)
        returns = np.array([2.0, 2.0, 2.0, 2.0])
        g = reinforce_grad(logits, actions, returns)
        chosen = g[np.arange(4), actions]
        self.assertTrue(np.all(chosen > 0), msg=str(chosen))
        before = objective(logits, actions, returns)
        after = objective(logits + 0.05 * g, actions, returns)
        self.assertGreater(after, before)
        g_bad = reinforce_grad(logits, actions, -returns)
        self.assertTrue(np.all(g_bad[np.arange(4), actions] < 0))

    def test_baseline_shifts_the_return_not_the_log_prob(self):
        rng = np.random.default_rng(11)
        logits = rng.standard_normal((4, 3))
        actions = rng.integers(0, 3, 4)
        returns = rng.standard_normal(4) + 5.0
        np.testing.assert_allclose(
            reinforce_grad(logits, actions, returns, 5.0),
            reinforce_grad(logits, actions, returns - 5.0, 0.0),
            atol=1e-9,
        )
        flat = np.full(4, 2.5)
        np.testing.assert_allclose(reinforce_grad(logits, actions, flat, 2.5), np.zeros((4, 3)), atol=1e-9)

    def test_softmax_is_numerically_stable(self):
        big = np.array([[1000.0, 1001.0]])
        g = reinforce_grad(big, np.array([1]), np.array([1.0]))
        self.assertTrue(np.all(np.isfinite(g)))
        np.testing.assert_allclose(g, reinforce_grad(np.array([[0.0, 1.0]]), np.array([1]), np.array([1.0])), atol=1e-9)

    def test_inputs_are_not_mutated(self):
        logits = np.zeros((2, 3))
        actions = np.array([0, 1])
        returns = np.array([1.0, -1.0])
        reinforce_grad(logits, actions, returns, 0.5)
        np.testing.assert_allclose(logits, np.zeros((2, 3)), atol=0)
        np.testing.assert_array_equal(actions, [0, 1])
        np.testing.assert_allclose(returns, [1.0, -1.0], atol=0)

    def test_bad_shapes_raise(self):
        logits = np.zeros((2, 3))
        with self.assertRaises(ValueError):
            reinforce_grad(np.zeros(3), np.array([0]), np.array([1.0]))
        with self.assertRaises(ValueError):
            reinforce_grad(np.zeros((0, 3)), np.array([], dtype=int), np.array([]))
        with self.assertRaises(ValueError):
            reinforce_grad(logits, np.array([0]), np.array([1.0, 2.0]))
        with self.assertRaises(ValueError):
            reinforce_grad(logits, np.array([0, 1]), np.array([1.0]))
        with self.assertRaises(ValueError):
            reinforce_grad(logits, np.array([0, 3]), np.array([1.0, 2.0]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
