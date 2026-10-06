import unittest

from solution import get_pair_counts, merge


class TestPairCountsAndMerge(unittest.TestCase):
    def test_counts_on_the_canonical_example(self):
        self.assertEqual(get_pair_counts([1, 2, 1, 2, 3]), {(1, 2): 2, (2, 1): 1, (2, 3): 1})

    def test_counts_keys_are_in_first_seen_order(self):
        self.assertEqual(list(get_pair_counts([5, 6, 7, 5, 6])), [(5, 6), (6, 7), (7, 5)])

    def test_counts_of_short_lists(self):
        self.assertEqual(get_pair_counts([]), {})
        self.assertEqual(get_pair_counts([7]), {})
        self.assertEqual(get_pair_counts([7, 7]), {(7, 7): 1})

    def test_merge_replaces_every_occurrence(self):
        self.assertEqual(merge([1, 2, 1, 2, 3], (1, 2), 4), [4, 4, 3])

    def test_merge_does_not_overlap(self):
        self.assertEqual(merge([1, 1, 1], (1, 1), 9), [9, 1])
        self.assertEqual(merge([1, 1, 1, 1], (1, 1), 9), [9, 9])

    def test_merge_keeps_unmatched_tail_and_head(self):
        self.assertEqual(merge([3, 1, 2, 1], (1, 2), 4), [3, 4, 1])
        self.assertEqual(merge([1, 2], (2, 1), 4), [1, 2])

    def test_neither_function_changes_the_input(self):
        ids = [1, 2, 1, 2, 3]
        get_pair_counts(ids)
        merge(ids, (1, 2), 4)
        self.assertEqual(ids, [1, 2, 1, 2, 3])

    def test_linear_time_on_100000_ids(self):
        ids = [i % 3 for i in range(100000)]
        counts = get_pair_counts(ids)
        self.assertEqual(counts[(0, 1)], 33333)
        merged = merge(ids, (0, 1), 3)
        self.assertEqual(len(merged), 100000 - 33333)


if __name__ == "__main__":
    unittest.main(verbosity=2)
