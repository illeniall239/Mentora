import math
import unittest

import torch

from solution import incremental_attention


def empty_cache(d):
    return {"K": torch.zeros(0, d), "V": torch.zeros(0, d)}


def full_causal_attention(Q, K, V):
    T, d = Q.shape
    scores = Q @ K.T / math.sqrt(d)
    mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
    scores = scores.masked_fill(mask, float("-inf"))
    return torch.softmax(scores, dim=-1) @ V


class TestIncrementalAttention(unittest.TestCase):
    def test_first_token_attends_only_to_itself(self):
        out, cache = incremental_attention(empty_cache(2), torch.tensor([1.0, 0.0]), torch.tensor([1.0, 0.0]), torch.tensor([3.0, 4.0]))
        self.assertTrue(torch.allclose(out, torch.tensor([3.0, 4.0]), atol=1e-6))
        self.assertEqual(tuple(cache["K"].shape), (1, 2))
        self.assertEqual(tuple(cache["V"].shape), (1, 2))

    def test_second_token_by_hand(self):
        cache = {"K": torch.tensor([[1.0, 0.0]]), "V": torch.tensor([[3.0, 4.0]])}
        out, cache = incremental_attention(cache, torch.tensor([1.0, 0.0]), torch.tensor([-1.0, 0.0]), torch.tensor([5.0, 6.0]))
        s = 1 / math.sqrt(2)
        w0 = math.exp(s) / (math.exp(s) + math.exp(-s))
        want = torch.tensor([w0 * 3 + (1 - w0) * 5, w0 * 4 + (1 - w0) * 6])
        self.assertTrue(torch.allclose(out, want, atol=1e-5))
        self.assertTrue(torch.equal(cache["K"], torch.tensor([[1.0, 0.0], [-1.0, 0.0]])))
        self.assertTrue(torch.equal(cache["V"], torch.tensor([[3.0, 4.0], [5.0, 6.0]])))

    def test_matches_last_row_of_full_causal_attention_at_every_step(self):
        torch.manual_seed(0)
        T, d = 9, 8
        Q, K, V = torch.randn(T, d) * 3, torch.randn(T, d) * 3, torch.randn(T, d)
        full = full_causal_attention(Q, K, V)
        cache = empty_cache(d)
        for t in range(T):
            out, cache = incremental_attention(cache, Q[t], K[t], V[t])
            self.assertEqual(tuple(out.shape), (d,))
            self.assertTrue(torch.allclose(out, full[t], atol=1e-5), msg=f"step {t}")
        self.assertEqual(tuple(cache["K"].shape), (T, d))
        self.assertTrue(torch.allclose(cache["K"], K))
        self.assertTrue(torch.allclose(cache["V"], V))

    def test_scaling_by_sqrt_d_is_applied(self):
        torch.manual_seed(1)
        d = 16
        cache = {"K": torch.randn(3, d) * 4, "V": torch.randn(3, d)}
        q, k, v = torch.randn(d) * 4, torch.randn(d) * 4, torch.randn(d)
        out, _ = incremental_attention(cache, q, k, v)
        K = torch.cat([cache["K"], k[None]])
        V = torch.cat([cache["V"], v[None]])
        scaled = torch.softmax(K @ q / math.sqrt(d), dim=0) @ V
        unscaled = torch.softmax(K @ q, dim=0) @ V
        self.assertTrue(torch.allclose(out, scaled, atol=1e-5))
        self.assertFalse(torch.allclose(out, unscaled, atol=1e-3))

    def test_cache_holds_only_keys_and_values_and_is_not_mutated(self):
        torch.manual_seed(2)
        d = 4
        K0, V0 = torch.randn(2, d), torch.randn(2, d)
        cache = {"K": K0.clone(), "V": V0.clone()}
        q, k, v = torch.randn(d), torch.randn(d), torch.randn(d)
        _, new_cache = incremental_attention(cache, q, k, v)
        self.assertEqual(set(new_cache.keys()), {"K", "V"})
        self.assertTrue(torch.equal(cache["K"], K0))
        self.assertTrue(torch.equal(cache["V"], V0))
        self.assertEqual(tuple(cache["K"].shape), (2, d))
        self.assertEqual(tuple(new_cache["K"].shape), (3, d))
        self.assertTrue(torch.equal(new_cache["K"][-1], k))
        self.assertTrue(torch.equal(new_cache["V"][-1], v))

    def test_weights_sum_to_one_and_use_the_cache_contents(self):
        torch.manual_seed(3)
        d = 5
        row = torch.randn(d)
        cache = {"K": torch.randn(4, d), "V": row.repeat(4, 1)}
        out, _ = incremental_attention(cache, torch.randn(d), torch.randn(d), row)
        self.assertTrue(torch.allclose(out, row, atol=1e-5))
        q, k, v = torch.randn(d), torch.randn(d), torch.randn(d)
        a, _ = incremental_attention({"K": torch.randn(3, d), "V": torch.randn(3, d)}, q, k, v)
        b, _ = incremental_attention({"K": torch.randn(3, d), "V": torch.randn(3, d)}, q, k, v)
        self.assertFalse(torch.allclose(a, b, atol=1e-3))


if __name__ == "__main__":
    unittest.main(verbosity=2)
