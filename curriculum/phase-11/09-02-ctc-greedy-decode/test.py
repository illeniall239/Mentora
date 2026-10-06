import random
import unittest

from solution import ctc_greedy_decode

BLANK, E, H, L, O = 0, 5, 8, 12, 15


def deblank_then_collapse(frame_ids, blank):
    """The common bug: removing blanks before collapsing runs."""
    out = []
    for i in [x for x in frame_ids if x != blank]:
        if not out or out[-1] != i:
            out.append(i)
    return out


class TestCtcGreedyDecode(unittest.TestCase):
    def test_hello(self):
        frames = [H, H, BLANK, E, BLANK, L, L, BLANK, L, L, BLANK, O]
        self.assertEqual(ctc_greedy_decode(frames, BLANK), [H, E, L, L, O])

    def test_a_blank_between_repeats_keeps_the_double_letter(self):
        self.assertEqual(ctc_greedy_decode([L, L], BLANK), [L])
        self.assertEqual(ctc_greedy_decode([L, BLANK, L], BLANK), [L, L])
        self.assertEqual(ctc_greedy_decode([L, BLANK, BLANK, L, L, BLANK, L], BLANK), [L, L, L])

    def test_collapsing_after_deblanking_is_wrong(self):
        frames = [L, L, BLANK, L, BLANK, BLANK, L, L]
        self.assertEqual(ctc_greedy_decode(frames, BLANK), [L, L, L])
        self.assertNotEqual(ctc_greedy_decode(frames, BLANK), deblank_then_collapse(frames, BLANK))

    def test_empty_silence_and_no_blanks(self):
        self.assertEqual(ctc_greedy_decode([], BLANK), [])
        self.assertEqual(ctc_greedy_decode([BLANK, BLANK, BLANK, BLANK], BLANK), [])
        self.assertEqual(ctc_greedy_decode([3, 3, 3], BLANK), [3])
        self.assertEqual(ctc_greedy_decode([1, 2, 3], BLANK), [1, 2, 3])

    def test_blank_id_is_not_hardcoded_to_zero(self):
        self.assertEqual(ctc_greedy_decode([1, 28, 1], 28), [1, 1])
        self.assertEqual(ctc_greedy_decode([0, 0, 28, 0, 28], 28), [0, 0])
        self.assertEqual(ctc_greedy_decode([28, 28], 28), [])

    def test_returns_a_new_list_and_does_not_mutate_the_input(self):
        frames = [H, H, BLANK, E]
        snapshot = list(frames)
        out = ctc_greedy_decode(frames, BLANK)
        self.assertIsInstance(out, list)
        self.assertEqual(frames, snapshot)
        self.assertIsNot(out, frames)
        self.assertTrue(all(isinstance(x, int) for x in out))

    def test_properties_on_random_frame_sequences(self):
        rng = random.Random(0)
        for _ in range(300):
            frames = [rng.choice([BLANK, BLANK, 1, 2, 3]) for _ in range(rng.randint(0, 40))]
            out = ctc_greedy_decode(frames, BLANK)
            self.assertNotIn(BLANK, out)
            # every output token must come from a run boundary in the frames
            expected = []
            for k, f in enumerate(frames):
                if f != BLANK and (k == 0 or frames[k - 1] != f):
                    expected.append(f)
            self.assertEqual(out, expected, msg=f"{frames}")
            self.assertLessEqual(len(out), len(frames))

    def test_negative_ids_raise(self):
        with self.assertRaises(ValueError):
            ctc_greedy_decode([1, 2], -1)
        with self.assertRaises(ValueError):
            ctc_greedy_decode([1, -2, 3], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
