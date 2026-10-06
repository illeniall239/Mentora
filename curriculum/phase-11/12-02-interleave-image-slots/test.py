import copy
import unittest

from solution import interleave

IMG = -200


class TestInterleaveImageSlots(unittest.TestCase):
    def test_single_image_in_the_middle(self):
        seq, spans = interleave([5, 6, 7], [1], IMG, 4)
        self.assertEqual(seq, [5, IMG, IMG, IMG, IMG, 6, 7])
        self.assertEqual(spans, [(1, 5)])

    def test_two_images_at_the_front_and_the_end(self):
        seq, spans = interleave([5, 6, 7], [0, 3], IMG, 2)
        self.assertEqual(seq, [IMG, IMG, 5, 6, 7, IMG, IMG])
        self.assertEqual(spans, [(0, 2), (5, 7)])

    def test_two_images_in_the_same_slot_are_adjacent(self):
        seq, spans = interleave([5, 6], [1, 1], IMG, 1)
        self.assertEqual(seq, [5, IMG, IMG, 6])
        self.assertEqual(spans, [(1, 2), (2, 3)])

    def test_no_text_and_no_images(self):
        self.assertEqual(interleave([], [0], IMG, 3), ([IMG, IMG, IMG], [(0, 3)]))
        self.assertEqual(interleave([5, 6, 7], [], IMG, 2), ([5, 6, 7], []))
        self.assertEqual(interleave([], [], IMG, 2), ([], []))

    def test_spans_point_at_placeholder_blocks(self):
        text = list(range(1, 21))
        slots = [0, 5, 5, 20]
        seq, spans = interleave(text, slots, IMG, 3)
        self.assertEqual(len(spans), 4)
        self.assertEqual(len(seq), len(text) + 4 * 3)
        for start, end in spans:
            self.assertEqual(end - start, 3)
            self.assertEqual(seq[start:end], [IMG] * 3)
        self.assertEqual([s for s, _ in spans], sorted(s for s, _ in spans))
        for (_, end), (next_start, _) in zip(spans, spans[1:]):
            self.assertLessEqual(end, next_start)

    def test_text_survives_and_lands_at_the_shifted_positions(self):
        text = list(range(1, 21))
        slots = [0, 5, 5, 20]
        n = 3
        seq, _ = interleave(text, slots, IMG, n)
        self.assertEqual([t for t in seq if t != IMG], text)
        for i, token in enumerate(text):
            shift = sum(1 for s in slots if s <= i) * n
            self.assertEqual(seq[i + shift], token, msg=f"text index {i}")

    def test_validation(self):
        with self.assertRaises(ValueError):
            interleave([5, 6, 7], [3, 1], IMG, 2)
        with self.assertRaises(ValueError):
            interleave([5, 6, 7], [4], IMG, 2)
        with self.assertRaises(ValueError):
            interleave([5, 6, 7], [-1], IMG, 2)
        with self.assertRaises(ValueError):
            interleave([5, IMG, 7], [1], IMG, 2)
        with self.assertRaises(ValueError):
            interleave([5, 6, 7], [1], IMG, 0)

    def test_inputs_are_not_modified(self):
        text, slots = [5, 6, 7], [1, 3]
        text_snapshot, slots_snapshot = copy.deepcopy(text), copy.deepcopy(slots)
        interleave(text, slots, IMG, 2)
        self.assertEqual(text, text_snapshot)
        self.assertEqual(slots, slots_snapshot)


if __name__ == "__main__":
    unittest.main(verbosity=2)
