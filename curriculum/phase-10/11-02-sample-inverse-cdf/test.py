import random
import unittest
from collections import Counter

from solution import sample


class FixedU:
    def __init__(self, u):
        self.u = u
        self.calls = 0

    def random(self):
        self.calls += 1
        return self.u


class TestSampleInverseCdf(unittest.TestCase):
    def test_empirical_frequencies_match_probabilities(self):
        probs = [0.1, 0.2, 0.3, 0.4]
        rng = random.Random(1234)
        counts = Counter(sample(probs, rng) for _ in range(10_000))
        for i, p in enumerate(probs):
            self.assertAlmostEqual(counts[i] / 10_000, p, delta=0.02, msg=f"token {i}")

    def test_same_seed_replays_the_same_tokens(self):
        probs = [0.25, 0.25, 0.5]
        rng1, rng2 = random.Random(42), random.Random(42)
        seq1 = [sample(probs, rng1) for _ in range(50)]
        seq2 = [sample(probs, rng2) for _ in range(50)]
        self.assertEqual(seq1, seq2)
        self.assertGreater(len(set(seq1)), 1)

    def test_boundaries_use_strictly_greater(self):
        probs = [0.2, 0.3, 0.5]
        self.assertEqual(sample(probs, FixedU(0.0)), 0)
        self.assertEqual(sample(probs, FixedU(0.19999)), 0)
        self.assertEqual(sample(probs, FixedU(0.2)), 1)
        self.assertEqual(sample(probs, FixedU(0.5)), 2)
        self.assertEqual(sample(probs, FixedU(0.999)), 2)

    def test_zero_probability_tokens_are_never_sampled(self):
        self.assertEqual(sample([0.0, 1.0, 0.0], FixedU(0.0)), 1)
        self.assertEqual(sample([0.0, 1.0, 0.0], FixedU(0.999999)), 1)
        rng = random.Random(3)
        draws = {sample([0.0, 0.5, 0.0, 0.5], rng) for _ in range(500)}
        self.assertEqual(draws, {1, 3})

    def test_rounding_falls_back_to_last_nonzero_index(self):
        third = 1 / 3
        probs = [third, third, third, 0.0]  # sums to 1 - 1e-16-ish
        self.assertEqual(sample(probs, FixedU(0.9999999999999999)), 2)
        self.assertEqual(sample([0.5, 0.5, 0.0], FixedU(0.9999999999999)), 1)

    def test_exactly_one_call_to_rng(self):
        stub = FixedU(0.4)
        sample([0.25, 0.25, 0.25, 0.25], stub)
        self.assertEqual(stub.calls, 1)

    def test_rejects_invalid_distributions(self):
        rng = random.Random(0)
        for bad in ([], [0.6, 0.6], [0.5, 0.4], [1.2, -0.2]):
            with self.assertRaises(ValueError, msg=str(bad)):
                sample(bad, rng)


if __name__ == "__main__":
    unittest.main(verbosity=2)
