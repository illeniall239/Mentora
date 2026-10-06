import random
import unittest

from solution import wer


def edit_distance(a: list[str], b: list[str]) -> int:
    """Plain Levenshtein distance, used only to check S + D + I."""
    prev = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        cur = [i] + [0] * len(b)
        for j in range(1, len(b) + 1):
            cur[j] = min(prev[j - 1] + (a[i - 1] != b[j - 1]), prev[j] + 1, cur[j - 1] + 1)
        prev = cur
    return prev[len(b)]


class TestWer(unittest.TestCase):
    def test_perfect_transcript(self):
        self.assertEqual(wer("the cat sat on the mat", "the cat sat on the mat"), (0, 0, 0, 6, 0.0))

    def test_one_substitution(self):
        s, d, i, n, rate = wer("the cat sat on the mat", "the cat sat on a mat")
        self.assertEqual((s, d, i, n), (1, 0, 0, 6))
        self.assertAlmostEqual(rate, 1 / 6, places=9)

    def test_one_deletion(self):
        s, d, i, n, rate = wer("the cat sat on the mat", "the cat sat the mat")
        self.assertEqual((s, d, i, n), (0, 1, 0, 6))
        self.assertAlmostEqual(rate, 1 / 6, places=9)

    def test_one_insertion(self):
        s, d, i, n, rate = wer("the cat sat on the mat", "the cat sat on on the mat")
        self.assertEqual((s, d, i, n), (0, 0, 1, 6))
        self.assertAlmostEqual(rate, 1 / 6, places=9)

    def test_mixed_errors(self):
        s, d, i, n, rate = wer("i love deep learning", "i loved deep learning a lot")
        self.assertEqual((s, d, i, n), (1, 0, 2, 4))
        self.assertAlmostEqual(rate, 0.75, places=9)

    def test_rate_can_exceed_one(self):
        s, d, i, n, rate = wer("hello", "hello hello hello")
        self.assertEqual((s, d, i, n), (0, 0, 2, 1))
        self.assertAlmostEqual(rate, 2.0, places=9)
        s, d, i, n, rate = wer("ok then", "thank you for watching this video")
        self.assertEqual(n, 2)
        self.assertGreater(rate, 1.0)
        self.assertAlmostEqual(rate, (s + d + i) / n, places=9)

    def test_empty_hypothesis_is_all_deletions(self):
        self.assertEqual(wer("one two three", ""), (0, 3, 0, 3, 1.0))
        self.assertEqual(wer("one two three", "   \n  "), (0, 3, 0, 3, 1.0))

    def test_tokens_are_words_not_characters_and_whitespace_collapses(self):
        s, d, i, n, rate = wer("the cat", "the bat")
        self.assertEqual((s, d, i, n), (1, 0, 0, 2))
        self.assertAlmostEqual(rate, 0.5, places=9)
        self.assertEqual(wer("  the   cat \n sat ", "the cat sat"), (0, 0, 0, 3, 0.0))
        self.assertEqual(wer("a b c", "abc")[3], 3)
        self.assertEqual(wer("The cat", "the cat")[0], 1)

    def test_counts_sum_to_the_edit_distance_on_random_pairs(self):
        rng = random.Random(0)
        vocab = ["a", "bee", "cat", "dog", "egg", "fox"]
        for _ in range(300):
            r = [rng.choice(vocab) for _ in range(rng.randint(1, 8))]
            h = [rng.choice(vocab) for _ in range(rng.randint(0, 8))]
            s, d, i, n, rate = wer(" ".join(r), " ".join(h))
            self.assertEqual(n, len(r))
            self.assertEqual(s + d + i, edit_distance(r, h), msg=f"{r} vs {h}")
            self.assertAlmostEqual(rate, (s + d + i) / n, places=9)
            self.assertEqual(len(h) - len(r), i - d, msg=f"{r} vs {h}")

    def test_empty_reference_raises(self):
        with self.assertRaises(ValueError):
            wer("", "anything")
        with self.assertRaises(ValueError):
            wer("   ", "")


if __name__ == "__main__":
    unittest.main(verbosity=2)
