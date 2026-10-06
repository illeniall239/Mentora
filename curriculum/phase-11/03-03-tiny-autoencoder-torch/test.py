import unittest

import torch
import torch.nn.functional as F

from solution import AE, train_ae


def manifold(seed: int = 0):
    """256 training points on a 3-D linear manifold in 8-D, plus held-out and off-manifold points."""
    g = torch.Generator().manual_seed(seed)
    basis = torch.randn(3, 8, generator=g)
    train = torch.randn(256, 3, generator=g) @ basis + 0.5
    held_out = torch.randn(64, 3, generator=g) @ basis + 0.5
    off = torch.randn(64, 8, generator=g) * train.std()
    return train, held_out, off


def mse(model: AE, x: torch.Tensor) -> float:
    with torch.no_grad():
        return float(F.mse_loss(model(x), x))


class TestAE(unittest.TestCase):
    def test_shapes_and_parameter_count(self):
        m = AE(8, 3)
        self.assertEqual(tuple(m.encode(torch.randn(5, 8)).shape), (5, 3))
        self.assertEqual(tuple(m.decode(torch.randn(5, 3)).shape), (5, 8))
        self.assertEqual(tuple(m(torch.randn(5, 8)).shape), (5, 8))
        self.assertEqual(sum(p.numel() for p in m.parameters()), 59)
        self.assertEqual(sum(p.numel() for p in AE(16, 2).parameters()), 16 * 2 + 2 + 2 * 16 + 16)

    def test_forward_is_decode_of_encode(self):
        torch.manual_seed(0)
        m = AE(6, 2)
        x = torch.randn(7, 6)
        with torch.no_grad():
            self.assertTrue(torch.allclose(m(x), m.decode(m.encode(x)), atol=1e-7))

    def test_bad_sizes_raise(self):
        with self.assertRaises(ValueError):
            AE(8, 0)
        with self.assertRaises(ValueError):
            AE(0, 3)


class TestTrainAE(unittest.TestCase):
    def test_matching_bottleneck_reaches_the_threshold(self):
        train, _, _ = manifold()
        model, loss = train_ae(train, 3)
        self.assertLess(loss, 1e-6)
        self.assertAlmostEqual(loss, mse(model, train), places=9)

    def test_too_narrow_bottleneck_stalls(self):
        train, _, _ = manifold()
        _, loss1 = train_ae(train, 1)
        _, loss3 = train_ae(train, 3)
        self.assertGreater(loss1, 0.5)
        self.assertGreater(loss1, loss3 * 1000.0)

    def test_generalizes_to_the_manifold_not_the_training_points(self):
        train, held_out, off = manifold()
        model, _ = train_ae(train, 3)
        self.assertLess(mse(model, held_out), 1e-6)
        self.assertGreater(mse(model, off), 0.5)

    def test_same_seed_is_reproducible(self):
        train, _, _ = manifold()
        _, a = train_ae(train, 2, steps=50)
        _, b = train_ae(train, 2, steps=50)
        self.assertEqual(a, b)
        _, c = train_ae(train, 2, steps=50, seed=1)
        self.assertNotEqual(a, c)

    def test_training_actually_descends(self):
        train, _, _ = manifold()
        _, few = train_ae(train, 3, steps=1)
        _, many = train_ae(train, 3, steps=300)
        self.assertLess(many, few)
        self.assertGreater(few, 0.1)

    def test_returns_a_trained_module_with_working_parts(self):
        train, _, _ = manifold()
        model, _ = train_ae(train, 3, steps=100)
        self.assertIsInstance(model, AE)
        z = model.encode(train)
        self.assertEqual(tuple(z.shape), (256, 3))
        with torch.no_grad():
            self.assertTrue(torch.allclose(model.decode(z), model(train), atol=1e-6))

    def test_bad_inputs_raise(self):
        train, _, _ = manifold()
        with self.assertRaises(ValueError):
            train_ae(torch.randn(8), 3)
        with self.assertRaises(ValueError):
            train_ae(train, 3, steps=0)
        with self.assertRaises(ValueError):
            train_ae(train, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
