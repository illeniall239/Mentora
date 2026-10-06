import unittest

import torch
import torch.nn.functional as F

from solution import embed, tied_logits

TABLE = torch.tensor([[1.0, 0.0], [0.0, 1.0], [2.0, 3.0]])


class TestEmbedAndTiedLogits(unittest.TestCase):
    def test_embed_hand_rows(self):
        out = embed(torch.tensor([2, 0]), TABLE)
        self.assertTrue(torch.equal(out, torch.tensor([[2.0, 3.0], [1.0, 0.0]])))

    def test_embed_equals_one_hot_matmul(self):
        torch.manual_seed(0)
        table = torch.randn(11, 5)
        ids = torch.randint(0, 11, (4, 6))
        want = F.one_hot(ids, 11).float() @ table
        got = embed(ids, table)
        self.assertEqual(tuple(got.shape), (4, 6, 5))
        self.assertTrue(torch.allclose(got, want, atol=1e-6))

    def test_embed_gradient_counts_repeated_ids(self):
        table = TABLE.clone().requires_grad_(True)
        embed(torch.tensor([1, 1, 2]), table).sum().backward()
        self.assertTrue(torch.equal(table.grad, torch.tensor([[0.0, 0.0], [2.0, 2.0], [1.0, 1.0]])))

    def test_embed_out_of_range_id_raises(self):
        with self.assertRaises(IndexError):
            embed(torch.tensor([3]), TABLE)

    def test_tied_logits_hand_values(self):
        out = tied_logits(torch.tensor([[2.0, 3.0]]), TABLE)
        self.assertTrue(torch.allclose(out, torch.tensor([[2.0, 3.0, 13.0]])))

    def test_tied_logits_shape_and_batch(self):
        torch.manual_seed(1)
        table = torch.randn(7, 3)
        h = torch.randn(2, 5, 3)
        out = tied_logits(h, table)
        self.assertEqual(tuple(out.shape), (2, 5, 7))
        self.assertTrue(torch.allclose(out, h @ table.T, atol=1e-6))

    def test_tied_logits_rejects_wrong_width(self):
        with self.assertRaises(ValueError):
            tied_logits(torch.tensor([[1.0, 2.0, 3.0]]), TABLE)

    def test_tied_round_trip_shares_one_table(self):
        torch.manual_seed(2)
        table = torch.randn(9, 4, requires_grad=True)
        ids = torch.tensor([3, 8])
        logits = tied_logits(embed(ids, table), table)
        self.assertTrue(torch.allclose(logits, F.one_hot(ids, 9).float() @ table @ table.T, atol=1e-5))
        logits[0, 0].backward()
        # Both the looked-up rows and every row the output layer scores against get gradient.
        self.assertGreater(table.grad.abs().sum().item(), 0.0)
        self.assertNotEqual(table.grad[0].abs().sum().item(), 0.0)
        self.assertNotEqual(table.grad[3].abs().sum().item(), 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
