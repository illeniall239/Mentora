import math
import random
import unittest

from solution import apply_rope, sinusoidal_pe


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def norm(a):
    return math.sqrt(dot(a, a))


class TestSinusoidalPE(unittest.TestCase):
    def test_position_zero_is_sine_then_cosine(self):
        self.assertEqual(sinusoidal_pe(0, 8), [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
        self.assertEqual(sinusoidal_pe(0, 2), [0.0, 1.0])

    def test_hand_computed_values(self):
        got = sinusoidal_pe(1, 4)
        want = [math.sin(1.0), math.cos(1.0), math.sin(0.01), math.cos(0.01)]
        for g, w in zip(got, want):
            self.assertAlmostEqual(g, w, delta=1e-9)
        for g, w in zip(sinusoidal_pe(2, 2), [math.sin(2.0), math.cos(2.0)]):
            self.assertAlmostEqual(g, w, delta=1e-9)

    def test_frequencies_use_pair_index_over_d_model(self):
        # Pair i of a d_model=16 encoding turns at 1 / 10000**(2i/16).
        pe = sinusoidal_pe(100, 16)
        self.assertEqual(len(pe), 16)
        for i in range(8):
            angle = 100 * (1.0 / (10000.0 ** (2 * i / 16)))
            self.assertAlmostEqual(pe[2 * i], math.sin(angle), delta=1e-9)
            self.assertAlmostEqual(pe[2 * i + 1], math.cos(angle), delta=1e-9)

    def test_bounded_and_rejects_bad_shapes(self):
        for pos in (0, 1, 7, 10000):
            pe = sinusoidal_pe(pos, 32)
            self.assertEqual(len(pe), 32)
            self.assertTrue(all(-1.0 <= v <= 1.0 for v in pe))
        with self.assertRaises(ValueError):
            sinusoidal_pe(1, 3)
        with self.assertRaises(ValueError):
            sinusoidal_pe(1, 0)
        with self.assertRaises(ValueError):
            sinusoidal_pe(-1, 4)


class TestRope(unittest.TestCase):
    def test_position_zero_is_the_identity(self):
        v = [1.0, 0.0, -2.5, 3.0]
        self.assertEqual(apply_rope(v, 0), v)
        self.assertEqual(v, [1.0, 0.0, -2.5, 3.0])  # not mutated

    def test_rotation_direction_and_exact_values(self):
        got = apply_rope([1.0, 0.0], 1)
        self.assertAlmostEqual(got[0], math.cos(1.0), delta=1e-9)
        self.assertAlmostEqual(got[1], math.sin(1.0), delta=1e-9)
        got = apply_rope([3.0, 4.0], 2)
        c, s = math.cos(2.0), math.sin(2.0)
        self.assertAlmostEqual(got[0], 3.0 * c - 4.0 * s, delta=1e-9)
        self.assertAlmostEqual(got[1], 3.0 * s + 4.0 * c, delta=1e-9)

    def test_rotation_preserves_length(self):
        rng = random.Random(0)
        v = [rng.uniform(-2, 2) for _ in range(8)]
        for pos in (0, 1, 5, 97, 10000):
            self.assertAlmostEqual(norm(apply_rope(v, pos)), norm(v), delta=1e-9)

    def test_score_depends_only_on_relative_distance(self):
        rng = random.Random(1)
        q = [rng.uniform(-1, 1) for _ in range(8)]
        k = [rng.uniform(-1, 1) for _ in range(8)]
        for gap in (0, 1, 4):
            base = dot(apply_rope(q, gap), apply_rope(k, 0))
            for shift in (1, 3, 50):
                shifted = dot(apply_rope(q, gap + shift), apply_rope(k, shift))
                self.assertAlmostEqual(shifted, base, delta=1e-9)

    def test_different_distances_give_different_scores(self):
        rng = random.Random(2)
        q = [rng.uniform(-1, 1) for _ in range(4)]
        k = [rng.uniform(-1, 1) for _ in range(4)]
        scores = [dot(apply_rope(q, m), apply_rope(k, 0)) for m in range(5)]
        self.assertGreater(max(scores) - min(scores), 1e-3)

    def test_each_pair_rotates_in_its_own_plane(self):
        # Zeroing the second pair must leave the first pair's output untouched.
        a = apply_rope([1.0, 2.0, 3.0, 4.0], 11)
        b = apply_rope([1.0, 2.0, 0.0, 0.0], 11)
        self.assertAlmostEqual(a[0], b[0], delta=1e-9)
        self.assertAlmostEqual(a[1], b[1], delta=1e-9)
        self.assertAlmostEqual(b[2], 0.0, delta=1e-12)
        self.assertAlmostEqual(b[3], 0.0, delta=1e-12)

    def test_rejects_bad_shapes(self):
        with self.assertRaises(ValueError):
            apply_rope([1.0, 2.0, 3.0], 1)
        with self.assertRaises(ValueError):
            apply_rope([], 1)
        with self.assertRaises(ValueError):
            apply_rope([1.0, 2.0], -3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
