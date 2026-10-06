import unittest

from solution import normalize_text, number_to_words


class TestNormalizeText(unittest.TestCase):
    def test_number_to_words_table(self):
        self.assertEqual(number_to_words(0), "zero")
        self.assertEqual(number_to_words(7), "seven")
        self.assertEqual(number_to_words(13), "thirteen")
        self.assertEqual(number_to_words(19), "nineteen")
        self.assertEqual(number_to_words(20), "twenty")
        self.assertEqual(number_to_words(21), "twenty-one")
        self.assertEqual(number_to_words(42), "forty-two")
        self.assertEqual(number_to_words(90), "ninety")
        self.assertEqual(number_to_words(99), "ninety-nine")

    def test_number_to_words_hundreds_have_no_and(self):
        self.assertEqual(number_to_words(100), "one hundred")
        self.assertEqual(number_to_words(101), "one hundred one")
        self.assertEqual(number_to_words(115), "one hundred fifteen")
        self.assertEqual(number_to_words(342), "three hundred forty-two")
        self.assertEqual(number_to_words(900), "nine hundred")
        self.assertEqual(number_to_words(999), "nine hundred ninety-nine")

    def test_number_to_words_validation(self):
        with self.assertRaises(ValueError):
            number_to_words(-1)
        with self.assertRaises(ValueError):
            number_to_words(1000)

    def test_abbreviations_and_digits_together(self):
        self.assertEqual(normalize_text("Dr. 3"), "Doctor three")
        self.assertEqual(normalize_text("Mr. Smith has 21 cats."), "Mister Smith has twenty-one cats.")
        self.assertEqual(normalize_text("St. Ives etc."), "Saint Ives et cetera")
        self.assertEqual(normalize_text("Prof. Mrs. Ms. Jr. vs."), "Professor Missus Miss Junior versus")

    def test_trailing_punctuation_stays_on_the_number(self):
        self.assertEqual(normalize_text("No. 115, approx. 7 km"), "Number one hundred fifteen, approximately seven km")
        self.assertEqual(normalize_text("I have 3."), "I have three.")
        self.assertEqual(normalize_text("Really? 12!"), "Really? twelve!")

    def test_whitespace_is_collapsed(self):
        self.assertEqual(normalize_text("  wide   spacing\n here "), "wide spacing here")
        self.assertEqual(normalize_text(""), "")
        self.assertEqual(normalize_text("   "), "")

    def test_whole_token_match_not_substring_replacement(self):
        self.assertEqual(normalize_text("13 vs. 3rd"), "thirteen versus 3rd")
        self.assertEqual(normalize_text("A4 h3llo 3-4"), "A4 h3llo 3-4")
        self.assertEqual(normalize_text("dr. Dr.J Drive"), "dr. Dr.J Drive")

    def test_out_of_range_number_token_raises(self):
        with self.assertRaises(ValueError):
            normalize_text("1000 cats")
        with self.assertRaises(ValueError):
            normalize_text("in 1996")

    def test_normalizing_twice_changes_nothing(self):
        for text in ["Dr. 3", "No. 115, approx. 7 km", "13 vs. 3rd"]:
            once = normalize_text(text)
            self.assertEqual(normalize_text(once), once)


if __name__ == "__main__":
    unittest.main(verbosity=2)
