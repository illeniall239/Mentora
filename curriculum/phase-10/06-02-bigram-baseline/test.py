import math
import random
import unittest

from solution import perplexity, train_bigram


def markov_sequence(rng, n, vocab):
    """A synthetic token stream where each token is usually followed by the next one (mod vocab)."""
    ids = [0]
    for _ in range(n - 1):
        prev = ids[-1]
        ids.append((prev + 1) % vocab if rng.random() < 0.8 else rng.randrange(vocab))
    return ids


class TestBigramBaseline(unittest.TestCase):
    def test_counts_by_hand_with_add_one_smoothing(self):
        P = train_bigram([0, 1, 0, 1, 0], 2)
        self.assertEqual(len(P), 2)
        self.assertAlmostEqual(P[0][0], 1 / 4, places=9)
        self.assertAlmostEqual(P[0][1], 3 / 4, places=9)
        self.assertAlmostEqual(P[1][0], 3 / 4, places=9)
        self.assertAlmostEqual(P[1][1], 1 / 4, places=9)

    def test_unseen_tokens_get_uniform_rows(self):
        P = train_bigram([0, 0], 3)
        self.assertEqual(len(P), 3)
        self.assertEqual(len(P[2]), 3)
        for b in range(3):
            self.assertAlmostEqual(P[2][b], 1 / 3, places=9)
        self.assertAlmostEqual(P[0][0], 2 / 4, places=9)
        self.assertAlmostEqual(P[0][1], 1 / 4, places=9)

    def test_rows_sum_to_one_and_are_strictly_positive(self):
        rng = random.Random(0)
        ids = markov_sequence(rng, 2000, 8)
        P = train_bigram(ids, 8)
        for row in P:
            self.assertAlmostEqual(sum(row), 1.0, places=9)
            self.assertTrue(all(p > 0 for p in row))

    def test_out_of_vocab_id_raises(self):
        with self.assertRaises(ValueError):
            train_bigram([0, 1, 5], 3)
        with self.assertRaises(ValueError):
            train_bigram([0, -1], 3)

    def test_uniform_model_has_perplexity_vocab(self):
        uniform = [[0.5, 0.5], [0.5, 0.5]]
        self.assertAlmostEqual(perplexity(uniform, [0, 1, 1, 0]), 2.0, places=9)
        uniform5 = [[0.2] * 5 for _ in range(5)]
        self.assertAlmostEqual(perplexity(uniform5, [0, 4, 2, 2, 3, 1]), 5.0, places=9)

    def test_perplexity_by_hand(self):
        P = train_bigram([0, 1, 0, 1, 0], 2)
        self.assertAlmostEqual(perplexity(P, [0, 1, 0, 1, 0]), 4 / 3, places=9)
        # geometric, not arithmetic, mean of the probabilities: (0.75 * 0.25)^(-1/2)
        self.assertAlmostEqual(perplexity(P, [0, 1, 1]), 1 / math.sqrt(0.75 * 0.25), places=9)

    def test_too_short_sequence_raises(self):
        with self.assertRaises(ValueError):
            perplexity([[1.0]], [0])

    def test_beats_uniform_on_train_and_fresh_data(self):
        rng = random.Random(0)
        vocab = 8
        train = markov_sequence(rng, 4000, vocab)
        fresh = markov_sequence(rng, 1000, vocab)
        P = train_bigram(train, vocab)
        self.assertLess(perplexity(P, train), 3.0)
        self.assertLess(perplexity(P, fresh), 3.0)
        self.assertGreater(perplexity(P, [rng.randrange(vocab) for _ in range(1000)]), 5.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
