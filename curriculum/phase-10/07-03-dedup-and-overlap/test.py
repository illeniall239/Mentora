import unittest

from solution import dedup_exact, ngram_overlap

CAT = "the cat sat on the mat"
DOG = "the dog sat on the mat"


class TestDedupAndOverlap(unittest.TestCase):
    def test_dedup_keeps_first_occurrence_in_order(self):
        self.assertEqual(dedup_exact(["a b", "c", "a b", "d", "c"]), ["a b", "c", "d"])
        self.assertEqual(dedup_exact([]), [])
        self.assertEqual(dedup_exact(["x", "x", "x"]), ["x"])

    def test_dedup_is_exact_not_normalised(self):
        self.assertEqual(dedup_exact(["Hello", "hello", "Hello "]), ["Hello", "hello", "Hello "])
        self.assertEqual(dedup_exact(["héllo", "hello", "héllo"]), ["héllo", "hello"])

    def test_dedup_on_a_larger_synthetic_corpus(self):
        docs = [f"doc {i % 37} says {(i * 7) % 11}" for i in range(2000)]
        kept = dedup_exact(docs)
        self.assertEqual(kept, list(dict.fromkeys(docs)))
        self.assertEqual(len(kept), len(set(docs)))

    def test_identical_and_disjoint_documents(self):
        self.assertAlmostEqual(ngram_overlap(CAT, CAT, 2), 1.0, places=9)
        self.assertAlmostEqual(ngram_overlap(CAT, CAT, 1), 1.0, places=9)
        self.assertAlmostEqual(ngram_overlap("a b c", "d e f", 1), 0.0, places=9)
        self.assertAlmostEqual(ngram_overlap("a b c d", "d c b a", 2), 0.0, places=9)  # order matters

    def test_one_changed_word_by_hand(self):
        self.assertAlmostEqual(ngram_overlap(CAT, DOG, 2), 3 / 7, places=9)
        # unigrams: {the, cat, sat, on, mat} vs {the, dog, sat, on, mat}: 4 shared of 6
        self.assertAlmostEqual(ngram_overlap(CAT, DOG, 1), 4 / 6, places=9)
        # trigrams: {the cat sat, cat sat on, sat on the, on the mat} vs {the dog sat, dog sat on, ...}: 2 of 6
        self.assertAlmostEqual(ngram_overlap(CAT, DOG, 3), 2 / 6, places=9)

    def test_overlap_is_symmetric_and_bounded(self):
        a = "one two three four five six"
        b = "zero two three four seven"
        self.assertAlmostEqual(ngram_overlap(a, b, 2), ngram_overlap(b, a, 2), places=9)
        self.assertAlmostEqual(ngram_overlap(a, b, 2), 2 / 7, places=9)
        for n in (1, 2, 3):
            self.assertTrue(0.0 <= ngram_overlap(a, b, n) <= 1.0)

    def test_too_short_documents_and_bad_n(self):
        self.assertEqual(ngram_overlap("a b", "a b c", 3), 0.0)
        self.assertEqual(ngram_overlap("", "a b c", 1), 0.0)
        self.assertEqual(ngram_overlap("", "", 1), 0.0)
        with self.assertRaises(ValueError):
            ngram_overlap("a b", "a b", 0)

    def test_repeated_ngrams_count_once(self):
        # sets, not multisets: "a a a a" has the single bigram (a, a)
        self.assertAlmostEqual(ngram_overlap("a a a a", "a a", 2), 1.0, places=9)
        self.assertAlmostEqual(ngram_overlap("a a a b", "a a", 2), 0.5, places=9)


if __name__ == "__main__":
    unittest.main(verbosity=2)
