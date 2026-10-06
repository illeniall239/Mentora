import unittest

import numpy as np

from solution import fid

A = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 2.0], [0.0, -2.0]])


def fid_2d_closed_form(a: np.ndarray, b: np.ndarray) -> float:
    """Independent expected value, valid only for d = 2.

    For a 2x2 matrix M with eigenvalues l1, l2 >= 0,
    (sqrt(l1) + sqrt(l2))^2 = tr(M) + 2*sqrt(det(M)), so the trace of its
    square root needs no eigendecomposition at all.
    """
    d_mu = a.mean(axis=0) - b.mean(axis=0)
    ca = np.cov(a, rowvar=False, ddof=1)
    cb = np.cov(b, rowvar=False, ddof=1)
    m = ca @ cb
    tr_sqrt = np.sqrt(np.trace(m) + 2.0 * np.sqrt(np.linalg.det(ca) * np.linalg.det(cb)))
    return float(d_mu @ d_mu + np.trace(ca) + np.trace(cb) - 2.0 * tr_sqrt)


class TestFid(unittest.TestCase):
    def test_identical_feature_sets_are_zero(self):
        self.assertAlmostEqual(fid(A, A.copy()), 0.0, places=6)
        rng = np.random.default_rng(0)
        x = rng.normal(size=(200, 8))
        self.assertAlmostEqual(fid(x, x.copy()), 0.0, places=6)

    def test_row_order_does_not_matter(self):
        rng = np.random.default_rng(1)
        x = rng.normal(size=(50, 4))
        self.assertAlmostEqual(fid(x, x[::-1].copy()), 0.0, places=6)
        self.assertAlmostEqual(fid(x, rng.permutation(x)), 0.0, places=6)

    def test_mean_shift_contributes_its_squared_norm(self):
        self.assertAlmostEqual(fid(A, A + 3.0), 18.0, places=6)
        self.assertAlmostEqual(fid(A, A + np.array([0.0, 5.0])), 25.0, places=6)

    def test_scaled_covariance_known_value_and_ddof_one(self):
        # C_a = diag(2/3, 8/3) with ddof=1; C_b = 9 C_a  ->  4*(2/3) + 4*(8/3) = 40/3.
        # Using ddof=0 covariances would give 10.0 instead.
        self.assertAlmostEqual(fid(A, 3.0 * A), 40.0 / 3.0, places=6)

    def test_matches_the_closed_form_on_non_commuting_covariances(self):
        rng = np.random.default_rng(2)
        for _ in range(5):
            a = rng.normal(size=(120, 2)) @ np.array([[2.0, 0.7], [0.0, 1.0]]) + 1.5
            b = rng.normal(size=(90, 2)) @ np.array([[0.4, 0.0], [1.3, 2.2]])
            ca, cb = np.cov(a, rowvar=False, ddof=1), np.cov(b, rowvar=False, ddof=1)
            self.assertGreater(np.abs(ca @ cb - cb @ ca).max(), 1e-3, "covariances should not commute")
            self.assertAlmostEqual(fid(a, b), fid_2d_closed_form(a, b), places=6)

    def test_the_square_root_of_the_product_is_not_the_product_of_square_roots(self):
        rng = np.random.default_rng(3)
        a = rng.normal(size=(150, 2)) @ np.array([[3.0, 0.0], [0.0, 0.3]])
        b = rng.normal(size=(150, 2)) @ np.array([[1.0, 1.4], [1.4, 1.0]])
        ca, cb = np.cov(a, rowvar=False, ddof=1), np.cov(b, rowvar=False, ddof=1)

        def root(c):
            w, v = np.linalg.eigh(c)
            return (v * np.sqrt(np.clip(w, 0, None))) @ v.T

        wrong = float(np.trace(ca) + np.trace(cb) - 2.0 * np.trace(root(ca) @ root(cb)))
        self.assertAlmostEqual(fid(a, b), fid_2d_closed_form(a, b), places=6)
        self.assertGreater(abs(fid(a, b) - wrong), 1e-2)

    def test_symmetric_non_negative_and_monotone_in_the_shift(self):
        rng = np.random.default_rng(4)
        a = rng.normal(size=(100, 5))
        b = rng.normal(size=(80, 5)) * 1.7
        self.assertAlmostEqual(fid(a, b), fid(b, a), places=6)
        self.assertGreaterEqual(fid(a, b), 0.0)
        shifts = [fid(a, b + s) for s in (0.0, 1.0, 2.0, 4.0)]
        self.assertEqual(shifts, sorted(shifts))
        self.assertIsInstance(fid(a, b), float)

    def test_few_samples_inflate_fid_for_the_same_distribution(self):
        rng = np.random.default_rng(5)
        big = fid(rng.normal(size=(4000, 4)), rng.normal(size=(4000, 4)))
        small = fid(rng.normal(size=(8, 4)), rng.normal(size=(8, 4)))
        self.assertGreater(small, 10 * big)

    def test_bad_shapes_raise(self):
        with self.assertRaises(ValueError):
            fid(np.zeros((10, 3)), np.zeros((10, 4)))
        with self.assertRaises(ValueError):
            fid(np.zeros((1, 3)), np.zeros((10, 3)))
        with self.assertRaises(ValueError):
            fid(np.zeros((10, 3)), np.zeros((1, 3)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
