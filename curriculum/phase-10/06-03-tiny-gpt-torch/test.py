import math
import unittest

import torch
import torch.nn.functional as F

from solution import TinyGPT

V, BLOCK = 8, 8
PATTERN = [0, 3, 5, 1, 6]


def windows():
    ids = torch.tensor((PATTERN * 20)[:64])
    X = torch.stack([ids[i : i + BLOCK] for i in range(64 - BLOCK)])
    Y = torch.stack([ids[i + 1 : i + BLOCK + 1] for i in range(64 - BLOCK)])
    return X, Y


def train(model, X, Y, steps, lr):
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    for _ in range(steps):
        _, loss = model(X, Y)
        opt.zero_grad()
        loss.backward()
        opt.step()


class TestTinyGPT(unittest.TestCase):
    def test_shapes_and_loss_none_without_targets(self):
        torch.manual_seed(0)
        model = TinyGPT(V, BLOCK, 32, 2, 1)
        logits, loss = model(torch.zeros(4, 6, dtype=torch.long))
        self.assertEqual(tuple(logits.shape), (4, 6, V))
        self.assertIsNone(loss)
        logits, _ = model(torch.zeros(2, BLOCK, dtype=torch.long))
        self.assertEqual(tuple(logits.shape), (2, BLOCK, V))

    def test_parameter_count_and_layers(self):
        torch.manual_seed(0)
        one = sum(p.numel() for p in TinyGPT(V, BLOCK, 32, 2, 1).parameters())
        self.assertEqual(one, 13544)
        two = sum(p.numel() for p in TinyGPT(V, BLOCK, 32, 2, 2).parameters())
        self.assertEqual(two - one, 12 * 32 * 32 + 9 * 32 + 4 * 32)  # one more block: qkv, proj, mlp, two LayerNorms
        self.assertEqual(sum(p.numel() for p in TinyGPT(V, BLOCK, 32, 4, 1).parameters()), one)  # heads are free
        with self.assertRaises(ValueError):
            TinyGPT(V, BLOCK, 30, 4, 1)

    def test_loss_is_cross_entropy_over_all_positions(self):
        torch.manual_seed(0)
        model = TinyGPT(V, BLOCK, 32, 2, 1)
        X, Y = windows()
        logits, loss = model(X, Y)
        want = F.cross_entropy(logits.reshape(-1, V), Y.reshape(-1))
        self.assertAlmostEqual(loss.item(), want.item(), places=6)
        self.assertTrue(loss.requires_grad)

    def test_initial_loss_is_near_ln_vocab(self):
        X, Y = windows()
        for seed in (0, 1):
            torch.manual_seed(seed)
            _, loss = TinyGPT(V, BLOCK, 32, 2, 1)(X, Y)
            self.assertLess(abs(loss.item() - math.log(V)), 0.6)

    def test_future_tokens_do_not_change_earlier_logits(self):
        torch.manual_seed(0)
        model = TinyGPT(V, BLOCK, 32, 2, 2)
        a = torch.tensor([[0, 3, 5, 1, 6, 0, 3, 5]])
        b = a.clone()
        b[0, 5:] = torch.tensor([7, 7, 7])
        la, _ = model(a)
        lb, _ = model(b)
        self.assertTrue(torch.allclose(la[0, :5], lb[0, :5], atol=1e-5))
        self.assertFalse(torch.allclose(la[0, 5:], lb[0, 5:], atol=1e-3))

    def test_position_matters(self):
        torch.manual_seed(0)
        model = TinyGPT(V, BLOCK, 32, 2, 1)
        la, _ = model(torch.tensor([[1, 2, 1, 2]]))
        lb, _ = model(torch.tensor([[2, 1, 2, 1]]))
        self.assertFalse(torch.allclose(la[0, -1], lb[0, -1], atol=1e-3))  # same last token, different history

    def test_too_long_input_raises(self):
        torch.manual_seed(0)
        model = TinyGPT(V, BLOCK, 32, 2, 1)
        with self.assertRaises(ValueError):
            model(torch.zeros(1, BLOCK + 1, dtype=torch.long))

    def test_generate_keeps_prefix_and_crops_to_block(self):
        torch.manual_seed(0)
        model = TinyGPT(V, BLOCK, 32, 2, 1)
        prefix = torch.tensor([[0, 3, 5], [1, 1, 1]])
        out = model.generate(prefix, 12)  # runs past block, so each step must crop
        self.assertEqual(tuple(out.shape), (2, 15))
        self.assertEqual(out.dtype, torch.long)
        self.assertTrue(torch.equal(out[:, :3], prefix))
        self.assertTrue(bool(((out >= 0) & (out < V)).all()))
        self.assertTrue(torch.equal(out, model.generate(prefix, 12)))  # greedy is deterministic

    def test_overfits_repeating_sequence_and_continues_it(self):
        X, Y = windows()
        torch.manual_seed(0)
        model = TinyGPT(V, BLOCK, 32, 2, 1)
        train(model, X, Y, 300, 3e-3)
        _, loss = model(X, Y)
        self.assertLess(loss.item(), 0.1)
        out = model.generate(torch.tensor([[0, 3]]), 10)
        self.assertEqual(out[0].tolist(), (PATTERN * 3)[:12])


if __name__ == "__main__":
    unittest.main(verbosity=2)
