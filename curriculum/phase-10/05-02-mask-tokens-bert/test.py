import random
import unittest

from solution import mask_tokens

VOCAB = 1000
MASK = 4
SPECIAL = (0, 1, 2, 3)


def plain_ids(n: int, seed: int = 99) -> list[int]:
    """n ids, none of them special and none of them the mask id."""
    r = random.Random(seed)
    return [r.randrange(10, 900) for _ in range(n)]


def buckets(ids, out, labels):
    masked = kept = randomized = 0
    for original, got, label in zip(ids, out, labels):
        if label == -100:
            continue
        if got == MASK:
            masked += 1
        elif got == original:
            kept += 1
        else:
            randomized += 1
    return masked, kept, randomized


class TestMaskTokens(unittest.TestCase):
    def test_labels_only_where_the_input_changed_hands(self):
        ids = plain_ids(200)
        out, labels = mask_tokens(ids, 0.3, MASK, random.Random(0), VOCAB, SPECIAL)
        self.assertEqual(len(out), len(ids))
        self.assertEqual(len(labels), len(ids))
        for i, label in enumerate(labels):
            if label == -100:
                self.assertEqual(out[i], ids[i], "an unselected position must be untouched")
            else:
                self.assertEqual(label, ids[i], "a label is the ORIGINAL id, not the corrupted one")
        self.assertGreater(sum(1 for lab in labels if lab != -100), 0)

    def test_rate_zero_copies_and_rate_one_selects_everything(self):
        ids = plain_ids(50)
        out, labels = mask_tokens(ids, 0.0, MASK, random.Random(1), VOCAB, SPECIAL)
        self.assertEqual(out, ids)
        self.assertIsNot(out, ids)
        self.assertEqual(labels, [-100] * 50)

        out, labels = mask_tokens(ids, 1.0, MASK, random.Random(2), VOCAB, SPECIAL)
        self.assertEqual(labels, ids)
        self.assertEqual(ids, plain_ids(50), "ids must not be mutated")

    def test_selected_fraction_tracks_the_rate(self):
        ids = plain_ids(4000)
        _, labels = mask_tokens(ids, 0.15, MASK, random.Random(3), VOCAB, SPECIAL)
        frac = sum(1 for lab in labels if lab != -100) / len(ids)
        self.assertGreater(frac, 0.125)
        self.assertLess(frac, 0.175)

    def test_eighty_ten_ten_split(self):
        ids = plain_ids(4000)
        out, labels = mask_tokens(ids, 0.15, MASK, random.Random(4), VOCAB, SPECIAL)
        masked, kept, randomized = buckets(ids, out, labels)
        n = masked + kept + randomized
        self.assertGreater(n, 400)
        # Always writing [MASK] would put this at 1.0; the other two tenths must exist.
        self.assertGreater(masked / n, 0.72)
        self.assertLess(masked / n, 0.88)
        self.assertGreater(kept / n, 0.04)
        self.assertLess(kept / n, 0.17)
        self.assertGreater(randomized / n, 0.04)
        self.assertLess(randomized / n, 0.17)

    def test_random_replacements_are_legal_ids(self):
        ids = plain_ids(2000)
        out, labels = mask_tokens(ids, 0.5, MASK, random.Random(5), VOCAB, SPECIAL)
        self.assertTrue(all(0 <= token < VOCAB for token in out))
        changed = [t for t, original, lab in zip(out, ids, labels) if lab != -100 and t != MASK and t != original]
        self.assertGreater(len(changed), 20)

    def test_special_tokens_are_never_selected(self):
        ids = [0, 1, 2, 3] * 500 + plain_ids(500)
        out, labels = mask_tokens(ids, 1.0, MASK, random.Random(6), VOCAB, SPECIAL)
        for i, token in enumerate(ids):
            if token in SPECIAL:
                self.assertEqual(labels[i], -100, "padding and friends must carry no label")
                self.assertEqual(out[i], token, "padding and friends must not be corrupted")
        self.assertEqual(sum(1 for lab in labels if lab != -100), 500)

    def test_seeded_and_reproducible(self):
        ids = plain_ids(500)
        a = mask_tokens(ids, 0.3, MASK, random.Random(7), VOCAB, SPECIAL)
        b = mask_tokens(ids, 0.3, MASK, random.Random(7), VOCAB, SPECIAL)
        c = mask_tokens(ids, 0.3, MASK, random.Random(8), VOCAB, SPECIAL)
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)

    def test_rejects_bad_arguments(self):
        with self.assertRaises(ValueError):
            mask_tokens([7], 1.5, MASK, random.Random(0), VOCAB, SPECIAL)
        with self.assertRaises(ValueError):
            mask_tokens([7], -0.1, MASK, random.Random(0), VOCAB, SPECIAL)
        with self.assertRaises(ValueError):
            mask_tokens([7, 2000], 0.1, MASK, random.Random(0), VOCAB, SPECIAL)
        with self.assertRaises(ValueError):
            mask_tokens([7], 0.1, 5000, random.Random(0), VOCAB, SPECIAL)
        with self.assertRaises(ValueError):
            mask_tokens([7], 0.1, MASK, random.Random(0), 0, SPECIAL)


if __name__ == "__main__":
    unittest.main(verbosity=2)
