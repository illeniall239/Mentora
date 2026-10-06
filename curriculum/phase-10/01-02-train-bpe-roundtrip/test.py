import unittest

from solution import decode, encode, train_bpe

PARAGRAPH = (
    "the cat sat on the mat. the dog sat on the log. "
    "héllo wörld, ¿qué tal? مرحبا بالعالم 🌍🌍 the end."
)


class TestTrainBpeRoundtrip(unittest.TestCase):
    def test_first_merges_on_the_canonical_string(self):
        merges, vocab = train_bpe("aaabdaaabac", 259)
        self.assertEqual(merges, {(97, 97): 256, (256, 97): 257, (257, 98): 258})
        self.assertEqual(vocab[256], b"aa")
        self.assertEqual(vocab[258], b"aaab")

    def test_vocab_contains_all_256_bytes(self):
        _, vocab = train_bpe("abc", 256)
        self.assertEqual(len(vocab), 256)
        self.assertEqual(vocab[0], b"\x00")
        self.assertEqual(vocab[255], b"\xff")

    def test_encode_and_decode_the_canonical_string(self):
        merges, vocab = train_bpe("aaabdaaabac", 259)
        ids = encode("aaabdaaabac", merges)
        self.assertEqual(ids, [258, 100, 258, 97, 99])
        self.assertEqual(decode(ids, vocab), "aaabdaaabac")

    def test_roundtrip_non_ascii_and_emoji(self):
        merges, vocab = train_bpe(PARAGRAPH, 270)
        for text in [PARAGRAPH, "🌍", "wörld", "مرحبا", "unseen text 123"]:
            self.assertEqual(decode(encode(text, merges), vocab), text)

    def test_merges_shorten_the_sequence_and_ids_stay_in_vocab(self):
        merges, vocab = train_bpe(PARAGRAPH, 270)
        ids = encode(PARAGRAPH, merges)
        self.assertLess(len(ids), len(PARAGRAPH.encode("utf-8")))
        self.assertTrue(all(0 <= i < 270 for i in ids))
        self.assertEqual(len(vocab), 270)

    def test_training_is_deterministic(self):
        a = train_bpe(PARAGRAPH, 270)
        b = train_bpe(PARAGRAPH, 270)
        self.assertEqual(a, b)
        self.assertEqual(list(a[0].values()), list(range(256, 270)))

    def test_stops_early_when_no_pair_repeats(self):
        merges, vocab = train_bpe("abc", 300)
        self.assertEqual(merges, {})
        self.assertEqual(len(vocab), 256)

    def test_decode_replaces_invalid_utf8(self):
        _, vocab = train_bpe("x", 256)
        self.assertEqual(decode([0xE2, 0x9C], vocab), "�")
        self.assertEqual(decode([], vocab), "")


if __name__ == "__main__":
    unittest.main(verbosity=2)
