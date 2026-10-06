import unittest

import torch

from solution import mean_pool


class TestMeanPool(unittest.TestCase):
    def test_hand_computed(self):
        hidden = torch.tensor([[[3.0, 0.0], [1.0, 2.0], [99.0, 99.0]]])
        out = mean_pool(hidden, torch.tensor([[1, 1, 0]]))
        self.assertTrue(torch.allclose(out, torch.tensor([[2.0, 1.0]]) / 5 ** 0.5, atol=1e-6))
        out = mean_pool(hidden, torch.tensor([[1, 0, 0]]))
        self.assertTrue(torch.allclose(out, torch.tensor([[1.0, 0.0]]), atol=1e-6))

    def test_padding_does_not_change_the_vector(self):
        torch.manual_seed(0)
        sentence = torch.randn(5, 8)
        short = torch.cat([sentence, torch.randn(2, 8) * 50]).unsqueeze(0)
        long = torch.cat([sentence, torch.randn(9, 8) * 50]).unsqueeze(0)
        a = mean_pool(short, torch.tensor([[1] * 5 + [0] * 2]))
        b = mean_pool(long, torch.tensor([[1] * 5 + [0] * 9]))
        self.assertEqual(tuple(a.shape), (1, 8))
        self.assertTrue(torch.allclose(a, b, atol=1e-6))

    def test_divides_by_token_count_not_by_length(self):
        torch.manual_seed(1)
        hidden = torch.randn(3, 6, 4)
        mask = torch.tensor([[1, 1, 1, 1, 1, 1], [1, 1, 1, 0, 0, 0], [1, 0, 0, 0, 0, 0]])
        out = mean_pool(hidden, mask)
        for b in range(3):
            n = int(mask[b].sum())
            want = hidden[b, :n].mean(dim=0)
            want = want / want.norm()
            self.assertTrue(torch.allclose(out[b], want, atol=1e-6), msg=f"row {b}")

    def test_every_row_has_unit_norm(self):
        torch.manual_seed(2)
        hidden = torch.randn(4, 7, 5) * 10
        mask = (torch.rand(4, 7) > 0.3).long()
        mask[:, 0] = 1
        out = mean_pool(hidden, mask)
        self.assertTrue(torch.allclose(out.norm(dim=1), torch.ones(4), atol=1e-5))

    def test_accepts_bool_and_float_masks(self):
        torch.manual_seed(3)
        hidden = torch.randn(2, 4, 3)
        mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]])
        a = mean_pool(hidden, mask.bool())
        b = mean_pool(hidden, mask.float())
        c = mean_pool(hidden, mask)
        self.assertTrue(torch.allclose(a, b, atol=1e-6))
        self.assertTrue(torch.allclose(a, c, atol=1e-6))

    def test_all_padding_row_raises(self):
        hidden = torch.randn(2, 3, 4)
        with self.assertRaises(ValueError):
            mean_pool(hidden, torch.tensor([[1, 1, 0], [0, 0, 0]]))

    def test_shape_mismatch_raises(self):
        with self.assertRaises(ValueError):
            mean_pool(torch.randn(2, 3, 4), torch.ones(2, 4))
        with self.assertRaises(ValueError):
            mean_pool(torch.randn(2, 3, 4), torch.ones(3, 3))

    def test_differentiable_only_through_real_tokens(self):
        torch.manual_seed(4)
        hidden = torch.randn(1, 4, 3, requires_grad=True)
        mean_pool(hidden, torch.tensor([[1, 1, 0, 0]])).sum().backward()
        self.assertIsNotNone(hidden.grad)
        self.assertGreater(hidden.grad[0, :2].abs().sum().item(), 0.0)
        self.assertEqual(hidden.grad[0, 2:].abs().sum().item(), 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
