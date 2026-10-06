import unittest

import numpy as np

from solution import mode_coverage

CENTERS = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0], [10.0, 10.0]])


def slow_coverage(samples: np.ndarray, centers: np.ndarray, radius: float) -> float:
    hits = 0
    for c in centers:
        if any(float(np.linalg.norm(s - c)) <= radius for s in samples):
            hits += 1
    return hits / len(centers)


class TestModeCoverage(unittest.TestCase):
    def test_every_mode_reached(self):
        self.assertEqual(mode_coverage(CENTERS, CENTERS, 0.0), 1.0)
        self.assertEqual(mode_coverage(CENTERS + 0.5, CENTERS, 1.0), 1.0)

    def test_mode_collapse(self):
        collapsed = np.tile(np.array([[0.0, 0.0]]), (500, 1))
        self.assertAlmostEqual(mode_coverage(collapsed, CENTERS, 1.0), 0.25, places=12)
        self.assertAlmostEqual(mode_coverage(collapsed, CENTERS, 0.0), 0.25, places=12)

    def test_denominator_is_modes_not_samples(self):
        # 51 samples, all within radius of some centre, but only two centres reached.
        samples = np.vstack([np.tile(np.array([[0.0, 0.0]]), (50, 1)), np.array([[10.0, 0.0]])])
        self.assertAlmostEqual(mode_coverage(samples, CENTERS, 1.0), 0.5, places=12)
        # Piling on more samples at an already-covered mode changes nothing.
        more = np.vstack([samples, np.tile(np.array([[0.0, 0.0]]), (9000, 1))])
        self.assertAlmostEqual(mode_coverage(more, CENTERS, 1.0), 0.5, places=12)

    def test_boundary_is_inclusive(self):
        s, c = np.array([[3.0, 4.0]]), np.array([[0.0, 0.0]])
        self.assertEqual(mode_coverage(s, c, 5.0), 1.0)
        self.assertEqual(mode_coverage(s, c, 4.999), 0.0)
        self.assertEqual(mode_coverage(s, c, 5.001), 1.0)

    def test_nothing_reached(self):
        self.assertEqual(mode_coverage(np.array([[50.0, 50.0]]), CENTERS, 1.0), 0.0)
        self.assertEqual(mode_coverage(np.array([[0.0, 0.0]]), CENTERS, -0.0), 0.25)

    def test_monotone_in_radius(self):
        rng = np.random.default_rng(0)
        samples = rng.normal(size=(200, 3)) * 4.0
        centers = rng.normal(size=(8, 3)) * 4.0
        values = [mode_coverage(samples, centers, r) for r in (0.0, 0.5, 1.0, 2.0, 5.0, 50.0)]
        self.assertEqual(values, sorted(values))
        self.assertEqual(values[0], 0.0)
        self.assertEqual(values[-1], 1.0)

    def test_matches_the_explicit_loop(self):
        rng = np.random.default_rng(1)
        for n, m, d in ((10, 3, 2), (120, 8, 4), (300, 16, 3)):
            samples = rng.normal(size=(n, d)) * 3.0
            centers = rng.normal(size=(m, d)) * 3.0
            for radius in (0.5, 1.5, 3.0):
                self.assertAlmostEqual(
                    mode_coverage(samples, centers, radius),
                    slow_coverage(samples, centers, radius),
                    places=12,
                    msg=f"n={n} m={m} r={radius}",
                )

    def test_bad_inputs_raise(self):
        with self.assertRaises(ValueError):
            mode_coverage(CENTERS, CENTERS, -1.0)
        with self.assertRaises(ValueError):
            mode_coverage(np.array([0.0, 0.0]), CENTERS, 1.0)
        with self.assertRaises(ValueError):
            mode_coverage(CENTERS, np.zeros((2, 5)), 1.0)
        with self.assertRaises(ValueError):
            mode_coverage(np.zeros((0, 2)), CENTERS, 1.0)
        with self.assertRaises(ValueError):
            mode_coverage(CENTERS, np.zeros((0, 2)), 1.0)

    def test_inputs_are_not_modified(self):
        samples = CENTERS + 0.25
        samples_copy, centers_copy = samples.copy(), CENTERS.copy()
        mode_coverage(samples, CENTERS, 1.0)
        self.assertTrue(np.array_equal(samples, samples_copy))
        self.assertTrue(np.array_equal(CENTERS, centers_copy))


if __name__ == "__main__":
    unittest.main(verbosity=2)
