import unittest

import numpy as np
import torch

from solution import gelu, project


def make_weights(seed=0, d_vision=5, d_hidden=7, d_llm=3, n=4):
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(n, d_vision))
    W1 = rng.normal(size=(d_vision, d_hidden)) * 0.5
    b1 = rng.normal(size=(d_hidden,)) * 0.1
    W2 = rng.normal(size=(d_hidden, d_llm)) * 0.5
    b2 = rng.normal(size=(d_llm,)) * 0.1
    return features, W1, b1, W2, b2


class TestProjector(unittest.TestCase):
    def test_gelu_values_and_shape(self):
        x = np.array([[1.0, -1.0], [0.0, 2.0]])
        want = np.array([[0.8411919906082768, -0.1588080093917233], [0.0, 1.954597694087775]])
        got = gelu(x)
        self.assertEqual(got.shape, (2, 2))
        np.testing.assert_allclose(got, want, atol=1e-12)

    def test_identity_projector_is_pure_gelu_not_relu(self):
        features = np.array([[1.0, -1.0]])
        eye, zero = np.eye(2), np.zeros(2)
        got = project(features, eye, zero, eye, zero)
        np.testing.assert_allclose(got, [[0.8411919906082768, -0.1588080093917233]], atol=1e-12)
        self.assertNotAlmostEqual(float(got[0, 1]), 0.0, places=6, msg="GELU damps negatives, ReLU zeroes them")

    def test_output_shape_is_n_by_d_llm(self):
        features, W1, b1, W2, b2 = make_weights(seed=1)
        out = project(features, W1, b1, W2, b2)
        self.assertEqual(out.shape, (4, 3))

    def test_matches_torch(self):
        features, W1, b1, W2, b2 = make_weights(seed=2, d_vision=6, d_hidden=9, d_llm=4, n=5)
        t = lambda a: torch.tensor(a, dtype=torch.float64)
        hidden = torch.nn.functional.gelu(t(features) @ t(W1) + t(b1), approximate="tanh")
        want = (hidden @ t(W2) + t(b2)).numpy()
        np.testing.assert_allclose(project(features, W1, b1, W2, b2), want, atol=1e-10)

    def test_each_patch_row_is_projected_independently(self):
        features, W1, b1, W2, b2 = make_weights(seed=3, n=6)
        out = project(features, W1, b1, W2, b2)
        perm = [5, 0, 3, 1, 4, 2]
        out_perm = project(features[perm], W1, b1, W2, b2)
        np.testing.assert_allclose(out_perm, out[perm], atol=1e-12)
        single = project(features[2:3], W1, b1, W2, b2)
        np.testing.assert_allclose(single, out[2:3], atol=1e-12)

    def test_biases_broadcast_over_the_token_axis(self):
        features, _, _, W2, b2 = make_weights(seed=4, n=6)
        W1 = np.zeros((5, 7))
        b1 = np.linspace(-1.0, 1.0, 7)
        out = project(features, W1, b1, W2, b2)
        want_row = gelu(b1) @ W2 + b2
        for i in range(out.shape[0]):
            np.testing.assert_allclose(out[i], want_row, atol=1e-12)

    def test_projector_is_not_affine(self):
        features, W1, b1, W2, b2 = make_weights(seed=5)
        b1_zero, b2_zero = np.zeros_like(b1), np.zeros_like(b2)
        once = project(features, W1, b1_zero, W2, b2_zero)
        twice = project(2.0 * features, W1, b1_zero, W2, b2_zero)
        self.assertFalse(np.allclose(twice, 2.0 * once, atol=1e-6), "a projector without GELU would be linear")

    def test_validation(self):
        features, W1, b1, W2, b2 = make_weights(seed=6)
        with self.assertRaises(ValueError):
            project(features, np.zeros((6, 7)), b1, W2, b2)
        with self.assertRaises(ValueError):
            project(features, W1, np.zeros(8), W2, b2)
        with self.assertRaises(ValueError):
            project(features, W1, b1, np.zeros((8, 3)), b2)
        with self.assertRaises(ValueError):
            project(features, W1, b1, W2, np.zeros(4))
        with self.assertRaises(ValueError):
            project(features[0], W1, b1, W2, b2)
        with self.assertRaises(ValueError):
            project(features, W1, b1.reshape(1, -1), W2, b2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
