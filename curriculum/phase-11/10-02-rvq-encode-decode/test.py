import copy
import math
import random
import unittest

from solution import rvq_decode, rvq_encode, rvq_errors


def norm(v):
    return math.sqrt(sum(x * x for x in v))


def random_rvq(seed, dim=4, stages=3, entries=6):
    """Codebooks with shrinking scale; entry 0 of each stage is the zero vector ("add nothing")."""
    rng = random.Random(seed)
    vec = [rng.uniform(-1.0, 1.0) for _ in range(dim)]
    codebooks = []
    for k in range(stages):
        scale = 0.5**k
        cb = [[0.0] * dim]
        cb += [[rng.uniform(-scale, scale) for _ in range(dim)] for _ in range(entries - 1)]
        codebooks.append(cb)
    return vec, codebooks


class TestResidualVectorQuantization(unittest.TestCase):
    def test_hand_example_two_stages(self):
        vec = [0.9, 0.2]
        cbs = [[[1.0, 0.0], [0.0, 1.0]], [[-0.1, 0.2], [0.9, 0.2]]]
        self.assertEqual(rvq_encode(vec, cbs), [0, 0])
        out = rvq_decode([0, 0], cbs)
        self.assertAlmostEqual(out[0], 0.9, places=9)
        self.assertAlmostEqual(out[1], 0.2, places=9)
        errs = rvq_errors(vec, cbs)
        self.assertEqual(len(errs), 3)
        self.assertAlmostEqual(errs[0], math.sqrt(0.85), places=9)
        self.assertAlmostEqual(errs[1], math.sqrt(0.05), places=9)
        self.assertAlmostEqual(errs[2], 0.0, places=9)

    def test_each_stage_quantizes_the_running_residual_not_the_input(self):
        vec = [0.9, 0.2]
        cbs = [[[1.0, 0.0], [0.0, 1.0]], [[-0.1, 0.2], [0.9, 0.2]]]
        got = rvq_encode(vec, cbs)
        self.assertEqual(got, [0, 0])
        self.assertNotEqual(got, [0, 1], "stage 2 must quantize the residual, not vec again")
        # Three stages: the residual keeps shrinking, so later stages pick small entries.
        cbs3 = cbs + [[[0.0, 0.0], [0.9, 0.2]]]
        self.assertEqual(rvq_encode(vec, cbs3), [0, 0, 0])

    def test_nearest_entry_is_squared_l2_not_largest_dot_product(self):
        self.assertEqual(rvq_encode([1.0, 0.0], [[[2.0, 0.0], [1.0, 0.0]]]), [1])
        self.assertEqual(rvq_encode([1.0, 1.0], [[[3.0, 0.0], [1.0, 1.0]]]), [1])
        self.assertEqual(rvq_errors([1.0, 0.0], [[[2.0, 0.0], [1.0, 0.0]]])[-1], 0.0)

    def test_ties_go_to_the_smallest_index(self):
        self.assertEqual(rvq_encode([1.0, 0.0], [[[0.0, 0.0], [2.0, 0.0]]]), [0])
        self.assertEqual(rvq_encode([0.0, 0.0], [[[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]]]), [0])

    def test_decode_pairs_each_index_with_its_own_codebook(self):
        cbs = [[[100.0, 0.0], [0.0, 100.0]], [[1.0, 2.0], [3.0, 4.0]]]
        self.assertEqual(rvq_decode([1, 0], cbs), [1.0, 102.0])
        self.assertEqual(rvq_decode([0, 1], cbs), [103.0, 4.0])
        # Decoding the same indices against the stages in the wrong order is a different vector.
        self.assertNotEqual(rvq_decode([1, 0], list(reversed(cbs))), rvq_decode([1, 0], cbs))

    def test_error_shrinks_with_every_added_codebook(self):
        vec, cbs = random_rvq(seed=7)
        errs = rvq_errors(vec, cbs)
        self.assertEqual(len(errs), len(cbs) + 1)
        self.assertAlmostEqual(errs[0], norm(vec), places=9)
        for a, b in zip(errs, errs[1:]):
            self.assertLessEqual(b, a + 1e-12)
        self.assertLess(errs[-1], errs[0])
        self.assertLess(errs[-1], errs[1])

    def test_errors_agree_with_decoding_the_first_k_streams(self):
        vec, cbs = random_rvq(seed=11, dim=5, stages=4, entries=8)
        indices = rvq_encode(vec, cbs)
        self.assertEqual(len(indices), 4)
        errs = rvq_errors(vec, cbs)
        for k in range(1, len(cbs) + 1):
            approx = rvq_decode(indices[:k], cbs[:k])
            residual = [v - a for v, a in zip(vec, approx)]
            self.assertAlmostEqual(errs[k], norm(residual), places=9)

    def test_validation(self):
        cbs = [[[1.0, 0.0], [0.0, 1.0]]]
        with self.assertRaises(ValueError):
            rvq_encode([1.0, 0.0], [])
        with self.assertRaises(ValueError):
            rvq_encode([1.0, 0.0], [[]])
        with self.assertRaises(ValueError):
            rvq_encode([1.0, 0.0], [[[1.0, 0.0, 0.0]]])
        with self.assertRaises(ValueError):
            rvq_decode([0], [])
        with self.assertRaises(ValueError):
            rvq_decode([0, 0], cbs)
        with self.assertRaises(ValueError):
            rvq_decode([2], cbs)
        with self.assertRaises(ValueError):
            rvq_decode([-1], cbs)
        with self.assertRaises(ValueError):
            rvq_errors([1.0, 0.0], [])

    def test_inputs_are_not_modified(self):
        vec, cbs = random_rvq(seed=3)
        vec_snapshot, cbs_snapshot = copy.deepcopy(vec), copy.deepcopy(cbs)
        indices = rvq_encode(vec, cbs)
        rvq_decode(indices, cbs)
        rvq_errors(vec, cbs)
        self.assertEqual(vec, vec_snapshot)
        self.assertEqual(cbs, cbs_snapshot)


if __name__ == "__main__":
    unittest.main(verbosity=2)
