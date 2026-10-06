import math
import unittest

from solution import softmax

INF = math.inf


class TestStableSoftmaxRows(unittest.TestCase):
    def assertRowsClose(self, got, want, places=9):
        self.assertEqual(len(got), len(want))
        for g, w in zip(got, want):
            self.assertEqual(len(g), len(w))
            for a, b in zip(g, w):
                self.assertAlmostEqual(a, b, places=places)

    def test_small_values_by_hand(self):
        s = 1 / (1 + math.e)
        self.assertRowsClose(softmax([[1.0, 2.0]]), [[s, 1 - s]])
        self.assertRowsClose(softmax([[0.0, 0.0, 0.0, 0.0]]), [[0.25] * 4])
        self.assertRowsClose(softmax([[5.0], [3.0, 3.0]]), [[1.0], [0.5, 0.5]])

    def test_huge_values_do_not_overflow(self):
        s = 1 / (1 + math.e)
        self.assertRowsClose(softmax([[1000.0, 1001.0]]), [[s, 1 - s]])
        self.assertRowsClose(softmax([[1e5, 1e5 + 3.0, 1e5]]), [[1 / (2 + math.e ** 3), math.e ** 3 / (2 + math.e ** 3), 1 / (2 + math.e ** 3)]])

    def test_very_negative_values_do_not_underflow(self):
        s = 1 / (1 + math.e)
        self.assertRowsClose(softmax([[-1000.0, -1001.0]]), [[1 - s, s]])

    def test_shift_invariance(self):
        row = [0.3, -1.2, 2.5, 0.0]
        self.assertRowsClose(softmax([row]), softmax([[x + 123.456 for x in row]]))

    def test_minus_inf_gets_exactly_zero(self):
        out = softmax([[1.0, 2.0, -INF], [-INF, 0.0, -INF]])
        self.assertEqual(out[0][2], 0.0)
        self.assertEqual(out[1], [0.0, 1.0, 0.0])
        s = 1 / (1 + math.e)
        self.assertRowsClose([out[0][:2]], [[s, 1 - s]])

    def test_rows_sum_to_one_and_are_independent(self):
        xs = [[0.5, -3.0, 4.0, -INF, 1.0], [1e3, -1e3, 0.0, 0.0, 0.0], [7.0]]
        out = softmax(xs)
        for row in out:
            self.assertAlmostEqual(sum(row), 1.0, places=9)
        self.assertRowsClose([out[0]], softmax([xs[0]]))
        self.assertRowsClose([out[2]], [[1.0]])

    def test_input_is_not_modified(self):
        xs = [[3.0, 1.0], [-INF, 2.0]]
        softmax(xs)
        self.assertEqual(xs, [[3.0, 1.0], [-INF, 2.0]])

    def test_empty_or_fully_masked_row_raises(self):
        with self.assertRaises(ValueError):
            softmax([[-INF, -INF]])
        with self.assertRaises(ValueError):
            softmax([[1.0, 2.0], []])


if __name__ == "__main__":
    unittest.main(verbosity=2)
