import unittest

import numpy as np
import torch

from solution import vq_quantize

CB = np.array([[0.0, 0.0], [1.0, 1.0], [5.0, 5.0]])


class TestVqQuantize(unittest.TestCase):
    def test_hand_example(self):
        idx, quant = vq_quantize(np.array([[0.1, 0.1], [0.9, 1.2], [4.0, 4.0]]), CB)
        self.assertTrue(np.array_equal(idx, np.array([0, 1, 2])))
        self.assertTrue(np.allclose(quant, CB, atol=1e-12))
        self.assertEqual(idx.shape, (3,))
        self.assertEqual(quant.shape, (3, 2))

    def test_codebook_entries_quantize_to_themselves(self):
        idx, quant = vq_quantize(CB, CB)
        self.assertTrue(np.array_equal(idx, np.arange(3)))
        self.assertTrue(np.allclose(quant, CB, atol=1e-12))

    def test_distance_not_dot_product(self):
        idx, quant = vq_quantize(np.array([[1.0, 0.0]]), np.array([[1.0, 0.0], [5.0, 0.0]]))
        self.assertTrue(np.array_equal(idx, np.array([0])))
        self.assertTrue(np.allclose(quant, np.array([[1.0, 0.0]]), atol=1e-12))
        # A long entry in a nearby direction must not win just by being long.
        idx2, _ = vq_quantize(np.array([[0.0, 1.0]]), np.array([[0.0, 20.0], [0.0, 0.9]]))
        self.assertTrue(np.array_equal(idx2, np.array([1])))

    def test_ties_go_to_the_smallest_index(self):
        idx, quant = vq_quantize(np.array([[0.0, 0.0]]), np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]]))
        self.assertTrue(np.array_equal(idx, np.array([0])))
        self.assertTrue(np.allclose(quant, np.array([[1.0, 0.0]]), atol=1e-12))

    def test_matches_torch_cdist_on_random_data(self):
        rng = np.random.default_rng(0)
        for n, k, d in ((5, 3, 2), (40, 16, 8), (200, 64, 4)):
            vectors = rng.normal(size=(n, d))
            codebook = rng.normal(size=(k, d)) * rng.uniform(0.1, 5.0, size=(k, 1))
            want = torch.cdist(torch.tensor(vectors), torch.tensor(codebook)).argmin(dim=1).numpy()
            idx, quant = vq_quantize(vectors, codebook)
            self.assertTrue(np.array_equal(idx, want), f"n={n} k={k}")
            self.assertTrue(np.allclose(quant, codebook[want], atol=1e-12))

    def test_output_rows_are_exact_codebook_rows(self):
        rng = np.random.default_rng(1)
        codebook = rng.normal(size=(6, 3))
        idx, quant = vq_quantize(rng.normal(size=(50, 3)), codebook)
        for row, i in zip(quant, idx):
            self.assertTrue(np.array_equal(row, codebook[i]))
        self.assertLessEqual(len(set(map(tuple, quant.tolist()))), 6)

    def test_more_entries_never_increase_the_error(self):
        rng = np.random.default_rng(2)
        vectors = rng.normal(size=(300, 4))
        big = rng.normal(size=(32, 4))
        small = big[:4]
        err = lambda cb: float(np.mean((vectors - vq_quantize(vectors, cb)[1]) ** 2))
        self.assertLessEqual(err(big), err(small))
        self.assertLess(err(big), err(small))

    def test_bad_inputs_raise(self):
        with self.assertRaises(ValueError):
            vq_quantize(np.array([0.0, 0.0]), CB)
        with self.assertRaises(ValueError):
            vq_quantize(np.array([[0.0, 0.0]]), np.zeros(2))
        with self.assertRaises(ValueError):
            vq_quantize(np.array([[0.0, 0.0]]), np.zeros((3, 5)))
        with self.assertRaises(ValueError):
            vq_quantize(np.array([[0.0, 0.0]]), np.zeros((0, 2)))
        with self.assertRaises(ValueError):
            vq_quantize(np.zeros((0, 2)), CB)

    def test_inputs_are_not_modified(self):
        rng = np.random.default_rng(3)
        vectors, codebook = rng.normal(size=(10, 3)), rng.normal(size=(4, 3))
        v_copy, c_copy = vectors.copy(), codebook.copy()
        _, quant = vq_quantize(vectors, codebook)
        quant[0] += 1.0
        self.assertTrue(np.array_equal(vectors, v_copy))
        self.assertTrue(np.array_equal(codebook, c_copy))


if __name__ == "__main__":
    unittest.main(verbosity=2)
