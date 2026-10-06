import unittest

import torch

from solution import make_mask

T, F_ = True, False


class TestMakeMask(unittest.TestCase):
    def test_bidirectional_is_all_true(self):
        self.assertEqual(make_mask(3, "bidirectional"), [[T, T, T], [T, T, T], [T, T, T]])
        self.assertEqual(make_mask(1, "bidirectional"), [[T]])

    def test_causal_keeps_the_diagonal(self):
        self.assertEqual(
            make_mask(4, "causal"),
            [
                [T, F_, F_, F_],
                [T, T, F_, F_],
                [T, T, T, F_],
                [T, T, T, T],
            ],
        )
        # torch.tril of ones is the same grid: lower triangle INCLUDING the diagonal.
        want = torch.ones(6, 6).tril().bool().tolist()
        self.assertEqual(make_mask(6, "causal"), want)
        self.assertEqual(make_mask(1, "causal"), [[T]])

    def test_causal_row_sums_and_no_empty_row(self):
        m = make_mask(8, "causal")
        self.assertEqual([sum(row) for row in m], [1, 2, 3, 4, 5, 6, 7, 8])
        for row in m:
            self.assertTrue(any(row), "an all-blocked row would make its softmax undefined")

    def test_prefix_grid(self):
        self.assertEqual(
            make_mask(4, "prefix", 2),
            [
                [T, T, F_, F_],
                [T, T, F_, F_],
                [T, T, T, F_],
                [T, T, T, T],
            ],
        )
        self.assertEqual(
            make_mask(5, "prefix", 3),
            [
                [T, T, T, F_, F_],
                [T, T, T, F_, F_],
                [T, T, T, F_, F_],
                [T, T, T, T, F_],
                [T, T, T, T, T],
            ],
        )

    def test_prefix_collapses_at_both_extremes(self):
        for n in (1, 4, 7):
            self.assertEqual(make_mask(n, "prefix", 0), make_mask(n, "causal"))
            self.assertEqual(make_mask(n, "prefix", n), make_mask(n, "bidirectional"))

    def test_entries_are_real_bools_and_rows_are_distinct(self):
        for kind, extra in (("bidirectional", None), ("causal", None), ("prefix", 2)):
            m = make_mask(4, kind, extra)
            self.assertEqual(len(m), 4)
            for row in m:
                self.assertEqual(len(row), 4)
                for cell in row:
                    self.assertIsInstance(cell, bool)
            m[0][3] = not m[0][3]
            self.assertNotEqual(m[0], make_mask(4, kind, extra)[0])
            self.assertEqual(m[1], make_mask(4, kind, extra)[1])

    def test_rejects_bad_arguments(self):
        with self.assertRaises(ValueError):
            make_mask(4, "prefix")
        with self.assertRaises(ValueError):
            make_mask(4, "masked-lm")
        with self.assertRaises(ValueError):
            make_mask(4, "Causal")
        with self.assertRaises(ValueError):
            make_mask(0, "causal")
        with self.assertRaises(ValueError):
            make_mask(4, "prefix", 5)
        with self.assertRaises(ValueError):
            make_mask(4, "prefix", -1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
