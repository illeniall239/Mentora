import unittest

import numpy as np

from solution import init_lora, lora_forward, merge_lora


class TestLoraForwardMerge(unittest.TestCase):
    def test_rank_one_by_hand(self):
        W = np.eye(2)
        B = np.array([[1.0], [0.0]])
        A = np.array([[0.0, 1.0]])
        x = np.array([[1.0, 2.0]])
        np.testing.assert_allclose(lora_forward(x, W, A, B, 8.0, 1), [[1.0, 10.0]], atol=1e-6)
        np.testing.assert_allclose(merge_lora(W, A, B, 8.0, 1), [[1.0, 8.0], [0.0, 1.0]], atol=1e-6)

    def test_alpha_over_r_scaling(self):
        W = np.eye(2)
        B = np.array([[1.0], [0.0]])
        A = np.array([[0.0, 1.0]])
        x = np.array([[1.0, 2.0]])
        np.testing.assert_allclose(lora_forward(x, W, A, B, 16.0, 1), [[1.0, 18.0]], atol=1e-6)
        rng = np.random.default_rng(1)
        W, x = rng.standard_normal((6, 5)), rng.standard_normal((3, 6))
        B, A = rng.standard_normal((6, 2)), rng.standard_normal((2, 5))
        delta_a = lora_forward(x, W, A, B, 8.0, 2) - x @ W  # scale 4
        delta_b = lora_forward(x, W, A, B, 2.0, 2) - x @ W  # scale 1
        np.testing.assert_allclose(delta_a, 4 * delta_b, atol=1e-6)
        np.testing.assert_allclose(delta_b, (x @ B) @ A, atol=1e-6)

    def test_init_shapes_and_zero_b(self):
        rng = np.random.default_rng(0)
        A, B = init_lora(4, 6, 2, rng)
        self.assertEqual(A.shape, (2, 6))
        self.assertEqual(B.shape, (4, 2))
        self.assertTrue(np.all(B == 0))
        self.assertGreater(np.abs(A).max(), 0)
        np.testing.assert_allclose(A, np.random.default_rng(0).standard_normal((2, 6)) * 0.02, atol=1e-12)

    def test_adapter_is_identity_at_init(self):
        rng = np.random.default_rng(2)
        W = rng.standard_normal((5, 7))
        x = rng.standard_normal((4, 5))
        A, B = init_lora(5, 7, 3, rng)
        np.testing.assert_allclose(lora_forward(x, W, A, B, 6.0, 3), x @ W, atol=1e-12)
        np.testing.assert_allclose(merge_lora(W, A, B, 6.0, 3), W, atol=1e-12)

    def test_merged_matrix_reproduces_the_unmerged_forward(self):
        rng = np.random.default_rng(3)
        W = rng.standard_normal((8, 6))
        B, A = rng.standard_normal((8, 4)), rng.standard_normal((4, 6))
        x = rng.standard_normal((10, 8))
        merged = merge_lora(W, A, B, 8.0, 4)
        self.assertEqual(merged.shape, W.shape)
        np.testing.assert_allclose(x @ merged, lora_forward(x, W, A, B, 8.0, 4), atol=1e-6)
        np.testing.assert_allclose(x @ merged, x @ (W + 2.0 * B @ A), atol=1e-6)

    def test_merge_does_not_touch_w_and_update_has_rank_r(self):
        rng = np.random.default_rng(4)
        W = rng.standard_normal((8, 8))
        W_copy = W.copy()
        B, A = rng.standard_normal((8, 2)), rng.standard_normal((2, 8))
        merged = merge_lora(W, A, B, 4.0, 2)
        np.testing.assert_allclose(W, W_copy, atol=0)
        self.assertEqual(np.linalg.matrix_rank(merged - W), 2)
        self.assertEqual(np.linalg.matrix_rank(merged), 8)

    def test_shape_mismatch_raises(self):
        W = np.eye(4)
        B, A = np.zeros((4, 2)), np.zeros((2, 4))
        for args in [(A, B, 3), (np.zeros((2, 5)), B, 2), (A, np.zeros((5, 2)), 2), (A, B, 0)]:
            a, b, r = args
            with self.assertRaises(ValueError):
                lora_forward(np.ones((1, 4)), W, a, b, 4.0, r)
            with self.assertRaises(ValueError):
                merge_lora(W, a, b, 4.0, r)


if __name__ == "__main__":
    unittest.main(verbosity=2)
