import copy
import random
import unittest

from solution import flatten_streams, unflatten_streams


def make_streams(q, t, seed):
    rng = random.Random(seed)
    return [[rng.randrange(1024) for _ in range(t)] for _ in range(q)]


class TestTokenStreamsFlatten(unittest.TestCase):
    def test_hand_example_is_frame_major(self):
        self.assertEqual(flatten_streams([[1, 2, 3], [10, 20, 30]]), [1, 10, 2, 20, 3, 30])
        self.assertEqual(flatten_streams([[7, 8]]), [7, 8])

    def test_streams_are_interleaved_not_concatenated(self):
        got = flatten_streams([[1, 2, 3], [10, 20, 30]])
        self.assertNotEqual(got, [1, 2, 3, 10, 20, 30], "one stream after the other is the wrong layout")
        got3 = flatten_streams([[0, 1], [2, 3], [4, 5]])
        self.assertEqual(got3, [0, 2, 4, 1, 3, 5])

    def test_position_formula_holds(self):
        q, t = 4, 7
        streams = make_streams(q, t, seed=5)
        seq = flatten_streams(streams)
        self.assertEqual(len(seq), q * t)
        for f in range(t):
            for c in range(q):
                self.assertEqual(seq[f * q + c], streams[c][f], msg=f"frame {f}, codebook {c}")

    def test_unflatten_inverts_flatten(self):
        for q, t in [(1, 5), (2, 3), (3, 1), (8, 11)]:
            streams = make_streams(q, t, seed=q * 100 + t)
            self.assertEqual(unflatten_streams(flatten_streams(streams), q), streams)
        self.assertEqual(unflatten_streams([1, 10, 2, 20, 3, 30], 2), [[1, 2, 3], [10, 20, 30]])

    def test_flatten_inverts_unflatten(self):
        seq = [rnd for rnd in range(12)]
        for q in (1, 2, 3, 4, 6, 12):
            self.assertEqual(flatten_streams(unflatten_streams(seq, q)), seq)

    def test_wrong_codebook_count_reshuffles_rather_than_failing(self):
        streams = make_streams(4, 5, seed=2)
        seq = flatten_streams(streams)
        self.assertNotEqual(unflatten_streams(seq, 2), unflatten_streams(seq, 4))
        self.assertEqual(unflatten_streams(seq, 4), streams)

    def test_zero_frames(self):
        self.assertEqual(flatten_streams([[], [], []]), [])
        self.assertEqual(unflatten_streams([], 3), [[], [], []])
        self.assertEqual(unflatten_streams([], 1), [[]])

    def test_validation(self):
        with self.assertRaises(ValueError):
            flatten_streams([])
        with self.assertRaises(ValueError):
            flatten_streams([[1, 2], [3]])
        with self.assertRaises(ValueError):
            unflatten_streams([1, 2, 3], 2)
        with self.assertRaises(ValueError):
            unflatten_streams([1, 2, 3], 0)
        with self.assertRaises(ValueError):
            unflatten_streams([1, 2, 3, 4], -1)

    def test_inputs_are_not_modified(self):
        streams = make_streams(3, 4, seed=9)
        snapshot = copy.deepcopy(streams)
        seq = flatten_streams(streams)
        seq_snapshot = list(seq)
        unflatten_streams(seq, 3)
        self.assertEqual(streams, snapshot)
        self.assertEqual(seq, seq_snapshot)


if __name__ == "__main__":
    unittest.main(verbosity=2)
