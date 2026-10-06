import unittest

from solution import consistency_score, flag_inconsistent, normalize


class TestConsistencyScore(unittest.TestCase):
    def test_normalize(self):
        self.assertEqual(normalize("  The  Answer "), "the answer")
        self.assertEqual(normalize("Paris\n"), "paris")
        self.assertEqual(normalize("a\t b   c"), "a b c")
        self.assertEqual(normalize(""), "")

    def test_all_agree_after_normalization(self):
        self.assertEqual(consistency_score(["Paris", "paris", " Paris ", "PARIS\n"]), 1.0)
        self.assertEqual(consistency_score(["only one"]), 1.0)

    def test_fraction_agreeing_with_majority(self):
        self.assertAlmostEqual(consistency_score(["Paris", "Lyon", "Paris", "Marseille"]), 0.5)
        self.assertAlmostEqual(consistency_score(["1905", "1915", "1921"]), 1 / 3)
        self.assertAlmostEqual(consistency_score(["x", "y", "y", "y", "z"]), 0.6)

    def test_majority_is_by_count_not_first_sample(self):
        self.assertAlmostEqual(consistency_score(["Lyon", "Paris", "Paris", "Paris"]), 0.75)
        self.assertAlmostEqual(consistency_score(["a", "b", "b", "a"]), 0.5)

    def test_score_is_not_one_over_distinct_answers(self):
        # 2 distinct answers but the majority holds 5 of 6 samples.
        self.assertAlmostEqual(consistency_score(["k", "k", "k", "k", "k", "j"]), 5 / 6)

    def test_empty_samples_raise(self):
        with self.assertRaises(ValueError):
            consistency_score([])

    def test_flag_is_strictly_below_threshold(self):
        samples = ["Paris", "Lyon", "Paris", "Marseille"]
        self.assertTrue(flag_inconsistent(samples, 0.6))
        self.assertFalse(flag_inconsistent(samples, 0.5))
        self.assertFalse(flag_inconsistent(["a", "a", "a"], 1.0))
        self.assertTrue(flag_inconsistent(["a", "b", "c"], 0.34))

    def test_flag_validates_threshold(self):
        with self.assertRaises(ValueError):
            flag_inconsistent(["a"], 1.5)
        with self.assertRaises(ValueError):
            flag_inconsistent(["a"], -0.1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
