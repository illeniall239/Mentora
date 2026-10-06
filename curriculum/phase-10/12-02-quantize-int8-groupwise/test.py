import random
import unittest

from solution import dequantize, dequantize_groupwise, quantize_absmax_int8, quantize_groupwise


def max_abs_error(xs, ys):
    return max(abs(x - y) for x, y in zip(xs, ys))


class TestQuantizeInt8Groupwise(unittest.TestCase):
    def test_int8_by_hand(self):
        q, scale = quantize_absmax_int8([0.0, 1.0, -5.0, 2.0])
        self.assertEqual(q, [0, 25, -127, 51])
        self.assertAlmostEqual(scale, 5 / 127, places=9)
        self.assertTrue(all(type(v) is int for v in q))

    def test_rounds_to_nearest_not_truncation(self):
        q, scale = quantize_absmax_int8([1.0, 0.99])
        self.assertEqual(q, [127, 126])
        self.assertLessEqual(max_abs_error(dequantize(q, scale), [1.0, 0.99]), scale / 2 + 1e-12)

    def test_error_bound_and_zeros_preserved_on_random_vector(self):
        rng = random.Random(0)
        xs = [rng.uniform(-3.0, 3.0) for _ in range(60)] + [0.0, 0.0]
        q, scale = quantize_absmax_int8(xs)
        back = dequantize(q, scale)
        self.assertEqual(len(back), len(xs))
        self.assertLessEqual(max_abs_error(back, xs), scale / 2 + 1e-12)
        self.assertEqual(back[-2:], [0.0, 0.0])
        self.assertTrue(all(-127 <= v <= 127 for v in q))
        self.assertIn(127, [abs(v) for v in q])

    def test_all_zeros_and_empty(self):
        self.assertEqual(quantize_absmax_int8([0.0, 0.0, 0.0]), ([0, 0, 0], 0.0))
        self.assertEqual(dequantize([0, 0, 0], 0.0), [0.0, 0.0, 0.0])
        with self.assertRaises(ValueError):
            quantize_absmax_int8([])

    def test_groupwise_by_hand(self):
        xs = [100.0, 0.1, -0.2, 0.3, 0.1, -0.2, 0.3, 0.05]
        q, scales = quantize_groupwise(xs, 4, 8)
        self.assertEqual(q, [127, 0, 0, 0, 42, -85, 127, 21])
        self.assertEqual(len(scales), 2)
        self.assertAlmostEqual(scales[0], 100 / 127, places=9)
        self.assertAlmostEqual(scales[1], 0.3 / 127, places=9)

    def test_outlier_in_one_group_does_not_hurt_the_others(self):
        xs = [100.0, 0.1, -0.2, 0.3, 0.1, -0.2, 0.3, 0.05]
        q, scales = quantize_groupwise(xs, 4, 8)
        back = dequantize_groupwise(q, scales, 4)
        self.assertLess(max_abs_error(back[4:], xs[4:]), 0.002)
        q8, s8 = quantize_absmax_int8(xs)
        self.assertGreater(max_abs_error(dequantize(q8, s8)[4:], xs[4:]), 0.05)

    def test_four_bit_range_and_error_bound(self):
        q, scales = quantize_groupwise([1.0, -0.6, 0.25], 3, 4)
        self.assertEqual(q, [7, -4, 2])
        self.assertAlmostEqual(scales[0], 1 / 7, places=9)
        rng = random.Random(1)
        xs = [rng.uniform(-1.0, 1.0) for _ in range(40)]
        q, scales = quantize_groupwise(xs, 8, 4)
        self.assertTrue(all(-7 <= v <= 7 for v in q))
        back = dequantize_groupwise(q, scales, 8)
        for i, (x, y) in enumerate(zip(xs, back)):
            self.assertLessEqual(abs(x - y), scales[i // 8] / 2 + 1e-12, msg=f"index {i}")

    def test_ragged_last_group_and_single_group_matches_int8(self):
        xs = [0.5, -1.0, 0.25, 2.0, 0.75]
        q, scales = quantize_groupwise(xs, 2, 8)
        self.assertEqual(len(q), 5)
        self.assertEqual(len(scales), 3)
        self.assertAlmostEqual(scales[2], 0.75 / 127, places=9)
        self.assertEqual(q[4], 127)
        q_one, scales_one = quantize_groupwise(xs, 100, 8)
        q8, s8 = quantize_absmax_int8(xs)
        self.assertEqual(q_one, q8)
        self.assertEqual(len(scales_one), 1)
        self.assertAlmostEqual(scales_one[0], s8, places=12)

    def test_groupwise_validation(self):
        with self.assertRaises(ValueError):
            quantize_groupwise([], 4, 8)
        with self.assertRaises(ValueError):
            quantize_groupwise([1.0], 0, 8)
        with self.assertRaises(ValueError):
            quantize_groupwise([1.0], 4, 1)
        with self.assertRaises(ValueError):
            quantize_groupwise([1.0], 4, 9)


if __name__ == "__main__":
    unittest.main(verbosity=2)
