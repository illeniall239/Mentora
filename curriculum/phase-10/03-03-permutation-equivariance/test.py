import unittest

import numpy as np
import torch
import torch.nn.functional as F

from solution import is_permutation_equivariant, permute_rows, self_attention


def make(n=6, d=4, d_k=3, d_v=5, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    params = {"W_q": rng.normal(size=(d, d_k)), "W_k": rng.normal(size=(d, d_k)), "W_v": rng.normal(size=(d, d_v))}
    perm = [int(i) for i in rng.permutation(n)]
    return X, params, perm


def torch_self_attention(X, params, causal):
    q = torch.tensor(X @ params["W_q"])
    k = torch.tensor(X @ params["W_k"])
    v = torch.tensor(X @ params["W_v"])
    return F.scaled_dot_product_attention(q, k, v, is_causal=causal).numpy()


class TestPermutationEquivariance(unittest.TestCase):
    def test_self_attention_matches_torch(self):
        X, params, _ = make()
        for causal in (False, True):
            got = self_attention(X, params, causal)
            self.assertEqual(got.shape, (6, 5))
            self.assertTrue(np.allclose(got, torch_self_attention(X, params, causal), atol=1e-6), msg=f"causal={causal}")

    def test_causal_row_zero_is_its_own_value(self):
        X, params, _ = make(seed=1)
        out = self_attention(X, params, causal=True)
        self.assertTrue(np.allclose(out[0], X[0] @ params["W_v"], atol=1e-9))
        self.assertFalse(np.allclose(self_attention(X, params, causal=False)[0], out[0], atol=1e-6))

    def test_permute_rows(self):
        X = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
        out = permute_rows(X, [2, 0, 1])
        self.assertTrue(np.array_equal(out, np.array([[3.0, 3.0], [1.0, 1.0], [2.0, 2.0]])))
        out[0, 0] = 99.0
        self.assertEqual(X[2, 0], 3.0)
        with self.assertRaises(ValueError):
            permute_rows(X, [0, 0, 1])
        with self.assertRaises(ValueError):
            permute_rows(X, [0, 1])

    def test_harness_on_simple_functions(self):
        X, _, perm = make(seed=2)
        self.assertTrue(is_permutation_equivariant(lambda A: 2 * A, X, perm))
        self.assertTrue(is_permutation_equivariant(np.tanh, X, perm))
        self.assertFalse(is_permutation_equivariant(lambda A: np.cumsum(A, axis=0), X, perm))
        self.assertTrue(is_permutation_equivariant(lambda A: np.cumsum(A, axis=0), X, list(range(6))))

    def test_harness_distinguishes_invariance_from_equivariance(self):
        X, _, perm = make(seed=3)
        # Sorting each column is invariant (same output for any row order) but not equivariant.
        self.assertFalse(is_permutation_equivariant(lambda A: np.sort(A, axis=0), X, perm))

    def test_non_causal_self_attention_is_equivariant(self):
        X, params, perm = make(seed=4)
        fn = lambda A: self_attention(A, params)
        self.assertTrue(is_permutation_equivariant(fn, X, perm))
        self.assertTrue(is_permutation_equivariant(fn, X, [5, 4, 3, 2, 1, 0]))
        # Equivariant, not invariant: the outputs really did move.
        self.assertFalse(np.allclose(fn(permute_rows(X, perm)), fn(X), atol=1e-6))

    def test_causal_mask_breaks_equivariance(self):
        X, params, _ = make(seed=5)
        fn = lambda A: self_attention(A, params, causal=True)
        self.assertFalse(is_permutation_equivariant(fn, X, [5, 4, 3, 2, 1, 0]))
        self.assertTrue(is_permutation_equivariant(fn, X, [0, 1, 2, 3, 4, 5]))

    def test_positional_encoding_breaks_equivariance(self):
        X, params, perm = make(seed=6)
        P = np.random.default_rng(7).normal(size=X.shape)
        self.assertFalse(is_permutation_equivariant(lambda A: self_attention(A + P, params), X, perm))


if __name__ == "__main__":
    unittest.main(verbosity=2)
