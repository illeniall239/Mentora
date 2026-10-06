import itertools
import math
import random
import unittest

from solution import durations_to_frames, total_frames


class TestLengthRegulator(unittest.TestCase):
    def test_hand_example(self):
        got = durations_to_frames(["HH", "AH", "L", "OW"], [0.05, 0.03, 0.04, 0.10], 0.01)
        self.assertEqual(got, ["HH"] * 5 + ["AH"] * 3 + ["L"] * 4 + ["OW"] * 10)
        self.assertEqual(len(got), 22)
        self.assertEqual(total_frames([0.05, 0.03, 0.04, 0.10], 0.01), 22)

    def test_cumulative_rounding_does_not_drift(self):
        got = durations_to_frames(["a", "b", "c", "d", "e"], [0.014] * 5, 0.01)
        self.assertEqual(got, ["a", "b", "b", "c", "d", "d", "e"])
        self.assertEqual(len(got), 7)
        self.assertNotEqual(len(got), 5, "rounding each duration on its own loses two frames")
        self.assertEqual(total_frames([0.014] * 5, 0.01), 7)

    def test_half_rounds_up_not_to_even(self):
        self.assertEqual(durations_to_frames(["a", "b"], [0.5, 1.0], 1.0), ["a", "b"])
        self.assertEqual(total_frames([0.5], 1.0), 1)
        self.assertEqual(total_frames([2.5], 1.0), 3)
        self.assertEqual(total_frames([1.5], 1.0), 2)

    def test_too_short_phoneme_gets_no_frames(self):
        got = durations_to_frames(["a", "b", "c"], [0.05, 0.001, 0.04], 0.01)
        self.assertEqual(got, ["a"] * 5 + ["c"] * 4)
        self.assertNotIn("b", got)
        self.assertEqual(len(got), total_frames([0.05, 0.001, 0.04], 0.01))

    def test_empty_and_zero_durations(self):
        self.assertEqual(durations_to_frames([], [], 0.01), [])
        self.assertEqual(total_frames([], 0.01), 0)
        self.assertEqual(durations_to_frames(["a"], [0.0], 0.01), [])

    def test_invariant_and_order_on_a_seeded_sequence(self):
        rng = random.Random(42)
        phonemes = [f"p{i}" for i in range(50)]
        durs = [rng.uniform(0.0, 0.08) for _ in range(50)]
        hop = 0.0125
        frames = durations_to_frames(phonemes, durs, hop)
        self.assertEqual(len(frames), total_frames(durs, hop))
        self.assertEqual(total_frames(durs, hop), math.floor(sum(durs) / hop + 0.5))
        # Each phoneme appears as one contiguous run, and the runs keep the input order.
        runs = [key for key, _ in itertools.groupby(frames)]
        self.assertEqual(runs, sorted(set(runs), key=lambda p: int(p[1:])))
        self.assertEqual(len(runs), len(set(runs)))

    def test_boundaries_match_cumulative_time(self):
        durs = [0.021, 0.019, 0.04]
        frames = durations_to_frames(["x", "y", "z"], durs, 0.01)
        self.assertEqual(frames.count("x"), 2)
        self.assertEqual(frames.count("y"), 2)
        self.assertEqual(frames.count("z"), 4)

    def test_validation(self):
        with self.assertRaises(ValueError):
            durations_to_frames(["a"], [0.1], 0.0)
        with self.assertRaises(ValueError):
            durations_to_frames(["a"], [0.1], -0.01)
        with self.assertRaises(ValueError):
            durations_to_frames(["a", "b"], [0.1], 0.01)
        with self.assertRaises(ValueError):
            durations_to_frames(["a"], [-0.1], 0.01)
        with self.assertRaises(ValueError):
            total_frames([0.1], 0.0)
        with self.assertRaises(ValueError):
            total_frames([-0.1], 0.01)


if __name__ == "__main__":
    unittest.main(verbosity=2)
