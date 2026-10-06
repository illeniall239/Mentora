import math
import unittest

import numpy as np
import torch
import torch.nn.functional as F

from solution import clip_loss

I2 = np.array([[1.0, 0.0], [0.0, 1.0]])


def torch_clip_loss(img: np.ndarray, txt: np.ndarray, temperature: float) -> float:
    """Expected value, built from torch primitives."""
    a = torch.tensor(img, dtype=torch.float64)
    b = torch.tensor(txt, dtype=torch.float64)
    a = a / a.norm(dim=1, keepdim=True)
    b = b / b.norm(dim=1, keepdim=True)
    logits = a @ b.T / temperature
    labels = torch.arange(a.shape[0])
    return float(0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)))


def embs(n: int, d: int, seed: int, align: float = 0.0):
    rng = np.random.default_rng(seed)
    img = rng.normal(size=(n, d))
    txt = rng.normal(size=(n, d)) + align * img
    return img, txt


class TestClipLoss(unittest.TestCase):
    def test_closed_form_on_orthonormal_pairs(self):
        for temperature in (1.0, 0.5, 0.07):
            want = math.log(1.0 + math.exp(-1.0 / temperature))
            self.assertAlmostEqual(clip_loss(I2, I2, temperature), want, places=12)

    def test_matches_torch_cross_entropy_both_directions(self):
        for n, d, seed in ((2, 3, 0), (5, 8, 1), (16, 32, 2)):
            for temperature in (0.07, 1.0, 4.0):
                img, txt = embs(n, d, seed)
                self.assertAlmostEqual(
                    clip_loss(img, txt, temperature),
                    torch_clip_loss(img, txt, temperature),
                    places=10,
                    msg=f"n={n} temp={temperature}",
                )

    def test_one_direction_alone_is_not_enough(self):
        # On this batch the two halves differ, so averaging only one of them misses.
        img, txt = embs(6, 4, seed=3)
        a = torch.tensor(img) / torch.tensor(img).norm(dim=1, keepdim=True)
        b = torch.tensor(txt) / torch.tensor(txt).norm(dim=1, keepdim=True)
        logits = a @ b.T / 0.5
        labels = torch.arange(6)
        i2t = float(F.cross_entropy(logits, labels))
        t2i = float(F.cross_entropy(logits.T, labels))
        self.assertGreater(abs(i2t - t2i), 1e-3)
        got = clip_loss(img, txt, 0.5)
        self.assertAlmostEqual(got, 0.5 * (i2t + t2i), places=10)
        self.assertNotAlmostEqual(got, i2t, places=4)
        self.assertNotAlmostEqual(got, t2i, places=4)
        self.assertNotAlmostEqual(got, i2t + t2i, places=4)  # the 0.5 is part of the definition

    def test_normalization_kills_embedding_scale(self):
        img, txt = embs(8, 6, seed=4, align=1.5)
        base = clip_loss(img, txt, 0.3)
        self.assertAlmostEqual(clip_loss(img * 50.0, txt, 0.3), base, places=10)
        self.assertAlmostEqual(clip_loss(img, txt * 0.002, 0.3), base, places=10)
        scaled = img.copy()
        scaled[2] *= 100.0
        self.assertAlmostEqual(clip_loss(scaled, txt, 0.3), base, places=10)
        # Without normalization the scaled batch would not land on the same number.
        self.assertGreater(base, 0.0)

    def test_temperature_divides(self):
        # Aligned pairs: a sharper (smaller) temperature must lower the loss.
        img, txt = embs(8, 6, seed=5, align=4.0)
        losses = [clip_loss(img, txt, t) for t in (0.02, 0.1, 0.5, 2.0, 10.0)]
        self.assertEqual(losses, sorted(losses))
        self.assertLess(losses[0], 0.05)
        self.assertGreater(losses[-1], 1.5)

    def test_matched_pairs_beat_shuffled_pairs(self):
        img, txt = embs(10, 8, seed=6, align=3.0)
        matched = clip_loss(img, txt, 0.2)
        shuffled = clip_loss(img, np.roll(txt, 1, axis=0), 0.2)
        self.assertLess(matched, shuffled)
        self.assertGreater(shuffled - matched, 1.0)

    def test_single_pair_has_no_negatives(self):
        img, txt = embs(1, 5, seed=7)
        self.assertAlmostEqual(clip_loss(img, txt, 0.07), 0.0, places=12)
        self.assertAlmostEqual(clip_loss(np.array([[1.0, 0.0]]), np.array([[0.0, 1.0]]), 1.0), 0.0, places=12)

    def test_bad_inputs_raise(self):
        with self.assertRaises(ValueError):
            clip_loss(I2, I2, 0.0)
        with self.assertRaises(ValueError):
            clip_loss(I2, I2, -1.0)
        with self.assertRaises(ValueError):
            clip_loss(I2, np.zeros((2, 2)), 1.0)
        with self.assertRaises(ValueError):
            clip_loss(I2, np.ones((3, 2)), 1.0)
        with self.assertRaises(ValueError):
            clip_loss(np.array([1.0, 0.0]), np.array([1.0, 0.0]), 1.0)
        with self.assertRaises(ValueError):
            clip_loss(np.zeros((0, 4)), np.zeros((0, 4)), 1.0)

    def test_inputs_are_not_modified(self):
        img, txt = embs(4, 4, seed=8)
        img_copy, txt_copy = img.copy(), txt.copy()
        clip_loss(img, txt, 0.1)
        self.assertTrue(np.array_equal(img, img_copy))
        self.assertTrue(np.array_equal(txt, txt_copy))


if __name__ == "__main__":
    unittest.main(verbosity=2)
