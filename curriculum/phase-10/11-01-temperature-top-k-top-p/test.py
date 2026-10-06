import math
import unittest

from solution import apply_temperature, top_k_filter, top_p_filter


def close(a, b, tol=1e-6):
    return len(a) == len(b) and all(abs(x - y) <= tol for x, y in zip(a, b))


class TestTemperatureTopKTopP(unittest.TestCase):
    def test_temperature_one_is_softmax(self):
        self.assertTrue(close(apply_temperature([0.0, math.log(3)], 1.0), [0.25, 0.75]))
        self.assertTrue(close(apply_temperature([2.0, 2.0, 2.0], 1.0), [1 / 3] * 3))

    def test_temperature_divides_logits_not_probabilities(self):
        # softmax(x / 2) differs from softmax(x) / 2 renormalized (which would be softmax(x) again).
        got = apply_temperature([0.0, math.log(3)], 2.0)
        r = math.sqrt(3)
        self.assertTrue(close(got, [1 / (1 + r), r / (1 + r)]))
        self.assertFalse(close(got, [0.25, 0.75], 1e-3))
        self.assertAlmostEqual(sum(got), 1.0, places=9)

    def test_low_temperature_approaches_greedy_and_high_approaches_uniform(self):
        cold = apply_temperature([1.0, 2.0, 0.5], 0.01)
        self.assertGreater(cold[1], 0.999999)
        hot = apply_temperature([1.0, 2.0, 0.5], 1000.0)
        self.assertTrue(close(hot, [1 / 3] * 3, 1e-3))

    def test_temperature_is_numerically_stable_and_rejects_nonpositive(self):
        got = apply_temperature([1000.0, 1001.0], 1.0)
        self.assertTrue(all(math.isfinite(x) for x in got))
        self.assertTrue(close(got, [1 / (1 + math.e), math.e / (1 + math.e)]))
        with self.assertRaises(ValueError):
            apply_temperature([1.0, 2.0], 0.0)
        with self.assertRaises(ValueError):
            apply_temperature([1.0, 2.0], -1.0)

    def test_top_k_keeps_positions_and_breaks_ties_by_index(self):
        got = top_k_filter([0.1, 0.5, 0.2, 0.2], 2)
        self.assertTrue(close(got, [0.0, 0.5 / 0.7, 0.2 / 0.7, 0.0]))
        self.assertTrue(close(top_k_filter([0.25, 0.25, 0.25, 0.25], 1), [1.0, 0.0, 0.0, 0.0]))
        self.assertTrue(close(top_k_filter([0.1, 0.5, 0.4], 3), [0.1, 0.5, 0.4]))
        self.assertTrue(close(top_k_filter([0.1, 0.5, 0.4], 10), [0.1, 0.5, 0.4]))
        with self.assertRaises(ValueError):
            top_k_filter([0.5, 0.5], 0)

    def test_top_k_on_unsorted_input_is_by_value_not_position(self):
        got = top_k_filter([0.05, 0.1, 0.6, 0.25], 1)
        self.assertTrue(close(got, [0.0, 0.0, 1.0, 0.0]))

    def test_top_p_includes_the_entry_that_crosses_p(self):
        probs = [0.5, 0.25, 0.125, 0.125]
        self.assertTrue(close(top_p_filter(probs, 0.8), [0.5 / 0.875, 0.25 / 0.875, 0.125 / 0.875, 0.0]))
        self.assertTrue(close(top_p_filter(probs, 0.6), [2 / 3, 1 / 3, 0.0, 0.0]))
        self.assertTrue(close(top_p_filter(probs, 0.1), [1.0, 0.0, 0.0, 0.0]))

    def test_top_p_exact_boundary_and_p_equal_one(self):
        probs = [0.5, 0.25, 0.125, 0.125]
        self.assertTrue(close(top_p_filter(probs, 0.75), [2 / 3, 1 / 3, 0.0, 0.0]))
        self.assertTrue(close(top_p_filter(probs, 0.5), [1.0, 0.0, 0.0, 0.0]))
        self.assertTrue(close(top_p_filter(probs, 1.0), probs))
        self.assertTrue(close(top_p_filter([0.125, 0.5, 0.375], 0.5), [0.0, 1.0, 0.0]))

    def test_top_p_validates_and_does_not_mutate(self):
        probs = [0.25, 0.5, 0.25]
        top_p_filter(probs, 0.5)
        top_k_filter(probs, 1)
        self.assertEqual(probs, [0.25, 0.5, 0.25])
        for bad in (0.0, -0.5, 1.5):
            with self.assertRaises(ValueError):
                top_p_filter(probs, bad)


if __name__ == "__main__":
    unittest.main(verbosity=2)
