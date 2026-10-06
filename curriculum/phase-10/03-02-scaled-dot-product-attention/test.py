import unittest

import torch
import torch.nn.functional as F

from solution import attention

Q3 = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
V3 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]


def close(rows, want, atol=1e-6):
    return torch.allclose(torch.tensor(rows, dtype=torch.float64), torch.tensor(want, dtype=torch.float64), atol=atol)


def torch_attention(Q, K, V, causal):
    q, k, v = (torch.tensor(x, dtype=torch.float64) for x in (Q, K, V))
    return F.scaled_dot_product_attention(q, k, v, is_causal=causal).tolist()


class TestScaledDotProductAttention(unittest.TestCase):
    def test_three_tokens_by_hand_causal(self):
        O, W = attention(Q3, Q3, V3, causal=True)
        want = [[1.0, 0.0, 0.0], [0.3302, 0.6698, 0.0], [0.2482, 0.2482, 0.5035]]
        self.assertTrue(close(W, want, atol=1e-4))
        self.assertTrue(close(O, want, atol=1e-4))
        self.assertEqual(O[0], V3[0])
        self.assertEqual(W[0], [1.0, 0.0, 0.0])

    def test_three_tokens_non_causal_sees_everything(self):
        O, W = attention(Q3, Q3, V3, causal=False)
        self.assertTrue(close(W[1], [0.1978, 0.4011, 0.4011], atol=1e-4))
        self.assertTrue(close(O, torch_attention(Q3, Q3, V3, False)))

    def test_weights_rows_sum_to_one(self):
        torch.manual_seed(0)
        Q, K, V = (torch.randn(6, 4).tolist() for _ in range(3))
        for causal in (False, True):
            _, W = attention(Q, K, V, causal)
            self.assertEqual(len(W), 6)
            for row in W:
                self.assertEqual(len(row), 6)
                self.assertAlmostEqual(sum(row), 1.0, places=9)

    def test_causal_weights_are_zero_above_the_diagonal(self):
        torch.manual_seed(1)
        Q, K, V = (torch.randn(5, 3).tolist() for _ in range(3))
        O, W = attention(Q, K, V, causal=True)
        for i in range(5):
            for j in range(i + 1, 5):
                self.assertEqual(W[i][j], 0.0)
        self.assertTrue(close(O[0], V[0]))
        self.assertTrue(close(O, torch_attention(Q, K, V, True)))

    def test_matches_torch_with_scaling(self):
        torch.manual_seed(2)
        Q, K = (((torch.randn(4, 16)) * 3).tolist() for _ in range(2))
        V = torch.randn(4, 5).tolist()
        O, _ = attention(Q, K, V)
        self.assertTrue(close(O, torch_attention(Q, K, V, False)))

    def test_cross_shape_follows_the_queries(self):
        torch.manual_seed(3)
        Q = torch.randn(2, 3).tolist()
        K = torch.randn(7, 3).tolist()
        V = torch.randn(7, 4).tolist()
        O, W = attention(Q, K, V)
        self.assertEqual((len(O), len(O[0])), (2, 4))
        self.assertEqual((len(W), len(W[0])), (2, 7))
        self.assertTrue(close(O, torch_attention(Q, K, V, False)))

    def test_one_key_dominates(self):
        Q = [[10.0, 0.0]]
        K = [[10.0, 0.0], [-10.0, 0.0], [0.0, 10.0]]
        V = [[1.0], [2.0], [3.0]]
        O, W = attention(Q, K, V)
        self.assertAlmostEqual(W[0][0], 1.0, places=6)
        self.assertAlmostEqual(O[0][0], 1.0, places=6)

    def test_invalid_shapes_raise(self):
        with self.assertRaises(ValueError):
            attention([[1.0, 0.0]], [[1.0, 0.0], [0.0, 1.0]], [[1.0], [2.0], [3.0]])
        with self.assertRaises(ValueError):
            attention([[1.0, 0.0]], [[1.0, 0.0, 0.0]], [[1.0]])
        with self.assertRaises(ValueError):
            attention([[1.0, 0.0]], [[1.0, 0.0], [0.0, 1.0]], [[1.0], [2.0]], causal=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
