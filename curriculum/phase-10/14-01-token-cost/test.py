import unittest

from solution import token_cost

# "th", "the", " the", "at", "on" — a tiny table of the kind BPE learns from English text.
MERGES = {(116, 104): 256, (256, 101): 257, (32, 257): 258, (97, 116): 259, (111, 110): 260}

ENGLISH = "the cat sat on the mat"
RUSSIAN = "кот сидел на коврике"


class TestTokenCost(unittest.TestCase):
    def test_counts_bytes_not_characters(self):
        self.assertEqual(token_cost("héllo", MERGES), 6)
        self.assertEqual(token_cost("🌍", MERGES), 4)
        self.assertEqual(token_cost("héllo", {}), 6)
        self.assertNotEqual(token_cost("héllo", MERGES), len("héllo"))

    def test_merges_shorten_the_count(self):
        self.assertEqual(token_cost("the", {}), 3)
        self.assertEqual(token_cost("the", MERGES), 1)
        self.assertEqual(token_cost("hello", MERGES), 5)

    def test_a_merge_feeds_a_later_merge(self):
        self.assertEqual(token_cost("the the", MERGES), 2)
        self.assertEqual(token_cost("on", MERGES), 1)
        self.assertEqual(token_cost("cat", MERGES), 2)

    def test_known_english_sentence(self):
        self.assertEqual(token_cost(ENGLISH, MERGES), 13)
        self.assertLess(token_cost(ENGLISH, MERGES), len(ENGLISH))

    def test_non_latin_costs_more_despite_fewer_characters(self):
        self.assertEqual(token_cost(RUSSIAN, MERGES), 37)
        self.assertLess(len(RUSSIAN), len(ENGLISH))
        self.assertGreater(token_cost(RUSSIAN, MERGES), 2 * token_cost(ENGLISH, MERGES))

    def test_empty_text_and_empty_merge_table(self):
        self.assertEqual(token_cost("", MERGES), 0)
        self.assertEqual(token_cost("", {}), 0)
        self.assertEqual(token_cost("abc", {}), 3)

    def test_returns_a_plain_int(self):
        result = token_cost(ENGLISH, MERGES)
        self.assertIsInstance(result, int)
        self.assertNotIsInstance(result, bool)

    def test_merge_table_is_not_modified(self):
        before = dict(MERGES)
        token_cost(ENGLISH, MERGES)
        self.assertEqual(MERGES, before)


if __name__ == "__main__":
    unittest.main(verbosity=2)
