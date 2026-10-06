import unittest

from solution import exact_match, normalize_answer, token_f1


class TestExactMatchAndTokenF1(unittest.TestCase):
    def test_normalize_answer(self):
        self.assertEqual(normalize_answer("The Answer."), "answer")
        self.assertEqual(normalize_answer("  AN  co-op   "), "coop")
        self.assertEqual(normalize_answer("the"), "")
        self.assertEqual(normalize_answer(""), "")
        self.assertEqual(normalize_answer("Paris, France"), "paris france")
        self.assertEqual(normalize_answer("don't"), "dont")
        # Articles are dropped as whole tokens only.
        self.assertEqual(normalize_answer("a an theatre"), "theatre")

    def test_exact_match(self):
        self.assertTrue(exact_match("The Answer.", "answer"))
        self.assertTrue(exact_match("Paris, France", "paris france"))
        self.assertTrue(exact_match("", ""))
        self.assertFalse(exact_match("42", "forty-two"))
        self.assertFalse(exact_match("red car", "car red"))

    def test_token_f1_full_and_partial_credit(self):
        self.assertAlmostEqual(token_f1("a red car", "red car"), 1.0, places=9)
        self.assertAlmostEqual(token_f1("the quick brown fox", "a quick red fox"), 2 / 3, places=9)
        self.assertAlmostEqual(token_f1("the capital is Paris", "Paris"), 2 * (1 / 3) * 1 / (1 / 3 + 1), places=9)
        self.assertAlmostEqual(token_f1("blue car", "red bus"), 0.0, places=9)

    def test_overlap_is_a_multiset_not_a_set(self):
        got = token_f1("the cat the cat", "cat")
        self.assertAlmostEqual(got, 2 / 3, places=9)
        self.assertNotAlmostEqual(got, 1.0, places=3)
        self.assertAlmostEqual(token_f1("cat cat", "cat cat"), 1.0, places=9)
        self.assertAlmostEqual(token_f1("cat cat cat", "cat"), 0.5, places=9)

    def test_token_f1_ignores_word_order_and_is_symmetric(self):
        self.assertAlmostEqual(token_f1("car red", "red car"), 1.0, places=9)
        for a, b in [("the quick brown fox", "a quick red fox"), ("the capital is Paris", "Paris"), ("x y", "y z w")]:
            self.assertAlmostEqual(token_f1(a, b), token_f1(b, a), places=9)

    def test_empty_after_normalization(self):
        self.assertAlmostEqual(token_f1("", ""), 1.0, places=9)
        self.assertAlmostEqual(token_f1("the a an", "  "), 1.0, places=9)
        self.assertAlmostEqual(token_f1("the", "cat"), 0.0, places=9)
        self.assertAlmostEqual(token_f1("cat", ""), 0.0, places=9)

    def test_exact_match_implies_f1_of_one(self):
        pairs = [("The Answer.", "answer"), ("Paris, France", "paris france"), ("a red car", "red car")]
        for pred, gold in pairs:
            self.assertTrue(exact_match(pred, gold))
            self.assertAlmostEqual(token_f1(pred, gold), 1.0, places=9)
        # ... but not the other way round: order changes EM and leaves F1 alone.
        self.assertFalse(exact_match("car red", "red car"))
        self.assertAlmostEqual(token_f1("car red", "red car"), 1.0, places=9)


if __name__ == "__main__":
    unittest.main(verbosity=2)
