import math
import unittest

from solution import constrained_mask, repetition_penalty, softmax


def close(a, b, tol=1e-6):
    return len(a) == len(b) and all(abs(x - y) <= tol for x, y in zip(a, b))


class TestConstrainedAndPenalized(unittest.TestCase):
    def test_mask_sets_forbidden_to_neg_inf_and_keeps_allowed(self):
        got = constrained_mask([1.0, 2.0, 3.0], {0, 2})
        self.assertEqual(got, [1.0, -math.inf, 3.0])
        self.assertEqual(constrained_mask([1.0, 2.0, 3.0], {1}), [-math.inf, 2.0, -math.inf])
        self.assertEqual(constrained_mask([1.0, 2.0, 3.0], {0, 1, 2}), [1.0, 2.0, 3.0])

    def test_masked_tokens_have_zero_probability(self):
        probs = softmax(constrained_mask([1.0, 2.0, 3.0], {0, 2}))
        self.assertEqual(probs[1], 0.0)
        self.assertTrue(close(probs, [1 / (1 + math.e ** 2), 0.0, math.e ** 2 / (1 + math.e ** 2)]))
        self.assertAlmostEqual(sum(probs), 1.0, places=9)
        only = softmax(constrained_mask([5.0, -3.0, 100.0], {1}))
        self.assertEqual(only, [0.0, 1.0, 0.0])

    def test_softmax_is_stable(self):
        got = softmax([1000.0, 1001.0])
        self.assertTrue(close(got, [1 / (1 + math.e), math.e / (1 + math.e)]))
        self.assertTrue(close(softmax([-math.inf, 0.0, 0.0]), [0.0, 0.5, 0.5]))

    def test_mask_validation(self):
        with self.assertRaises(ValueError):
            constrained_mask([1.0, 2.0], set())
        with self.assertRaises(ValueError):
            constrained_mask([1.0, 2.0], {5})
        with self.assertRaises(ValueError):
            constrained_mask([1.0, 2.0], {0, -1})

    def test_penalty_divides_positive_and_multiplies_negative(self):
        got = repetition_penalty([2.0, -1.0, 0.5], [0, 1], 2.0)
        self.assertTrue(close(got, [1.0, -2.0, 0.5]))
        # Both moves lower the token's probability.
        before = softmax([2.0, -1.0, 0.5])
        after = softmax(got)
        self.assertLess(after[0], before[0])
        self.assertLess(after[1], before[1])
        self.assertGreater(after[2], before[2])

    def test_zero_logit_and_repeats_penalized_once(self):
        got = repetition_penalty([0.0, 3.0, -0.5], [0, 0, 1, 1, 1], 3.0)
        self.assertTrue(close(got, [0.0, 1.0, -0.5]))
        got = repetition_penalty([1.0, 4.0], [1, 1, 1, 1], 2.0)
        self.assertTrue(close(got, [1.0, 2.0]))

    def test_penalty_identity_and_out_of_range_history(self):
        self.assertTrue(close(repetition_penalty([2.0, -1.0, 0.5], [0, 2], 1.0), [2.0, -1.0, 0.5]))
        self.assertTrue(close(repetition_penalty([2.0, -1.0, 0.5], [], 2.0), [2.0, -1.0, 0.5]))
        self.assertTrue(close(repetition_penalty([2.0, -1.0], [7, -1], 2.0), [2.0, -1.0]))
        with self.assertRaises(ValueError):
            repetition_penalty([1.0], [0], 0.0)

    def test_inputs_are_not_mutated(self):
        logits = [2.0, -1.0, 0.5]
        constrained_mask(logits, {0})
        repetition_penalty(logits, [0, 1, 2], 2.0)
        softmax(logits)
        self.assertEqual(logits, [2.0, -1.0, 0.5])


if __name__ == "__main__":
    unittest.main(verbosity=2)
