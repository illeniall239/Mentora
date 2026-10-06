import unittest

from solution import split_digits, token_count_delta

# "t1", "12", "34", "1234", "56" — a table that happily merges across digit boundaries.
MERGES = {(116, 49): 256, (49, 50): 257, (51, 52): 258, (257, 258): 259, (53, 54): 260}


class TestSplitDigits(unittest.TestCase):
    def test_groups_are_counted_from_the_right(self):
        self.assertEqual(split_digits("1234567"), ["1", "234", "567"])
        self.assertEqual(split_digits("1000000"), ["1", "000", "000"])
        self.assertEqual(split_digits("12345"), ["12", "345"])
        self.assertEqual(split_digits("123456"), ["123", "456"])
        self.assertEqual(split_digits("007"), ["007"])
        self.assertEqual(split_digits("1"), ["1"])
        self.assertEqual(split_digits("42"), ["42"])

    def test_non_digit_runs_are_single_untouched_chunks(self):
        self.assertEqual(split_digits("abc123"), ["abc", "123"])
        self.assertEqual(split_digits("no digits here"), ["no digits here"])
        self.assertEqual(split_digits("a1b22c333d4444"), ["a", "1", "b", "22", "c", "333", "d", "4", "444"])
        self.assertEqual(split_digits("2024-01-02"), ["2", "024", "-", "01", "-", "02"])

    def test_empty_text_and_the_join_invariant(self):
        self.assertEqual(split_digits(""), [])
        for text in ["", "1234567", "a1b22c333d4444", "2024-01-02", "héllo 🌍 99999", "price: 1250 €"]:
            self.assertEqual("".join(split_digits(text)), text, msg=text)

    def test_only_ascii_digits_count(self):
        self.assertEqual(split_digits("²³"), ["²³"])
        self.assertEqual(split_digits("١٢٣٤"), ["١٢٣٤"])
        self.assertEqual(split_digits("x²1234"), ["x²", "1", "234"])


class TestTokenCountDelta(unittest.TestCase):
    def test_split_costs_tokens_when_a_merge_spanned_a_boundary(self):
        self.assertEqual(token_count_delta("1234567", MERGES), 2)

    def test_split_stops_a_letter_merging_into_a_number(self):
        self.assertEqual(token_count_delta("t1234", MERGES), 1)

    def test_delta_is_zero_when_nothing_spanned_a_boundary(self):
        self.assertEqual(token_count_delta("abc123", MERGES), 0)
        self.assertEqual(token_count_delta("no digits here", MERGES), 0)
        self.assertEqual(token_count_delta("", MERGES), 0)
        self.assertEqual(token_count_delta("12", MERGES), 0)

    def test_delta_is_zero_with_an_empty_merge_table(self):
        for text in ["1234567", "t1234", "a1b22c333d4444", "héllo 🌍"]:
            self.assertEqual(token_count_delta(text, {}), 0, msg=text)

    def test_returns_an_int_and_leaves_the_merge_table_alone(self):
        before = dict(MERGES)
        result = token_count_delta("1234567", MERGES)
        self.assertIsInstance(result, int)
        self.assertNotIsInstance(result, bool)
        self.assertEqual(MERGES, before)


if __name__ == "__main__":
    unittest.main(verbosity=2)
