import unittest

from solution import fits_context, trim_history

MESSAGES = ["sys", "u1", "a1", "u2", "a2"]
COUNTS = [10, 30, 40, 20, 25]


class TestFitsContext(unittest.TestCase):
    def test_the_reserve_counts_against_the_limit(self):
        self.assertTrue(fits_context([80], 100, 20))
        self.assertFalse(fits_context([81], 100, 20))
        self.assertFalse(fits_context([100], 100, 20))
        self.assertTrue(fits_context([100], 100, 0))

    def test_sums_every_message(self):
        self.assertFalse(fits_context(COUNTS, 100, 20))
        self.assertTrue(fits_context([10, 20, 25], 100, 20))
        self.assertTrue(fits_context([], 100, 20))
        self.assertFalse(fits_context([], 10, 20))

    def test_rejects_negative_arguments(self):
        for args in ([[10], -1, 0], [[10], 100, -1], [[10, -1], 100, 0]):
            with self.assertRaises(ValueError, msg=str(args)):
                fits_context(*args)


class TestTrimHistory(unittest.TestCase):
    def test_drops_the_oldest_turns_and_keeps_the_newest(self):
        self.assertEqual(trim_history(MESSAGES, COUNTS, 100, 20), ["sys", "u2", "a2"])
        self.assertEqual(trim_history(MESSAGES, COUNTS, 40, 5), ["sys", "a2"])
        self.assertEqual(trim_history(MESSAGES, COUNTS, 10, 0), ["sys"])

    def test_keeps_everything_when_it_already_fits(self):
        self.assertEqual(trim_history(MESSAGES, COUNTS, 200, 20), MESSAGES)
        self.assertEqual(trim_history(MESSAGES, COUNTS, 145, 20), MESSAGES)
        self.assertEqual(trim_history(MESSAGES, COUNTS, 144, 20), ["sys", "a1", "u2", "a2"])

    def test_the_result_always_fits_and_the_system_prompt_survives(self):
        for limit in range(15, 150, 7):
            kept = trim_history(MESSAGES, COUNTS, limit, 5)
            self.assertEqual(kept[0], "sys", msg=str(limit))
            kept_counts = [COUNTS[MESSAGES.index(m)] for m in kept]
            self.assertTrue(fits_context(kept_counts, limit, 5), msg=str(limit))
            self.assertEqual(kept, [m for m in MESSAGES if m in kept], msg=str(limit))

    def test_trimming_is_minimal(self):
        # Dropping one more turn than needed is not allowed.
        self.assertEqual(trim_history(MESSAGES, COUNTS, 115, 20), ["sys", "a1", "u2", "a2"])

    def test_inputs_are_not_modified(self):
        messages, counts = list(MESSAGES), list(COUNTS)
        result = trim_history(messages, counts, 100, 20)
        self.assertEqual(messages, MESSAGES)
        self.assertEqual(counts, COUNTS)
        self.assertIsNot(result, messages)

    def test_rejects_impossible_and_malformed_inputs(self):
        for args in (
            ([], [], 100, 20),
            (["sys"], [10, 20], 100, 20),
            (["sys"], [10], 100, 95),
            (["sys"], [10], 9, 0),
            (["sys", "u1"], [10, 20], 100, -1),
            (["sys", "u1"], [10, -20], 100, 0),
        ):
            with self.assertRaises(ValueError, msg=str(args)):
                trim_history(*args)


if __name__ == "__main__":
    unittest.main(verbosity=2)
