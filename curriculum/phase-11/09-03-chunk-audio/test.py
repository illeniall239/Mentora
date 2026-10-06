import random
import unittest

from solution import chunk_audio


class TestChunkAudio(unittest.TestCase):
    def test_whisper_style_windows(self):
        got = chunk_audio(1120000, 16000, 30, 5)
        self.assertEqual(got, [(0, 480000), (400000, 880000), (800000, 1120000)])

    def test_small_hand_cases(self):
        self.assertEqual(chunk_audio(1000, 16000, 0.025, 0.01), [(0, 400), (240, 640), (480, 880), (720, 1000)])
        self.assertEqual(chunk_audio(1000, 16000, 0.025, 0.0), [(0, 400), (400, 800), (800, 1000)])

    def test_short_and_empty_audio(self):
        self.assertEqual(chunk_audio(160000, 16000, 30, 5), [(0, 160000)])
        self.assertEqual(chunk_audio(480000, 16000, 30, 5), [(0, 480000)])
        self.assertEqual(chunk_audio(1, 16000, 30, 5), [(0, 1)])
        self.assertEqual(chunk_audio(0, 16000, 30, 5), [])

    def test_windows_cover_every_sample_without_gaps(self):
        rng = random.Random(0)
        for _ in range(200):
            n = rng.randint(0, 5000)
            chunk_s = rng.choice([0.01, 0.025, 0.05, 0.125])
            overlap_s = rng.choice([0.0, 0.005, 0.01]) * (chunk_s > 0.01)
            got = chunk_audio(n, 8000, chunk_s, overlap_s)
            if n == 0:
                self.assertEqual(got, [])
                continue
            self.assertEqual(got[0][0], 0)
            self.assertEqual(got[-1][1], n)
            covered = set()
            for start, end in got:
                self.assertLess(start, end)
                covered.update(range(start, end))
            self.assertEqual(len(covered), n, msg=f"n={n} chunk_s={chunk_s} overlap_s={overlap_s}")

    def test_starts_advance_by_the_stride_and_windows_overlap_by_the_overlap(self):
        sr, chunk_s, overlap_s = 16000, 30, 5
        chunk, overlap = 480000, 80000
        stride = chunk - overlap
        got = chunk_audio(16000 * 300, sr, chunk_s, overlap_s)
        starts = [s for s, _ in got]
        self.assertEqual(starts, [k * stride for k in range(len(got))])
        for (s1, e1), (s2, _) in zip(got, got[1:]):
            self.assertEqual(e1 - s1, chunk)
            self.assertEqual(e1 - s2, overlap)

    def test_no_window_is_contained_in_the_previous_one(self):
        rng = random.Random(1)
        for _ in range(200):
            n = rng.randint(1, 3000)
            got = chunk_audio(n, 8000, 0.05, rng.choice([0.0, 0.01, 0.04, 0.045]))
            for (s1, e1), (s2, e2) in zip(got, got[1:]):
                self.assertGreater(s2, s1)
                self.assertGreater(e2, e1, msg=f"n={n} {got}")

    def test_zero_overlap_tiles_exactly(self):
        got = chunk_audio(1600, 16000, 0.025, 0)
        self.assertEqual(got, [(0, 400), (400, 800), (800, 1200), (1200, 1600)])
        self.assertEqual(len(chunk_audio(16000 * 90, 16000, 30, 0)), 3)

    def test_return_types(self):
        got = chunk_audio(1000, 16000, 0.025, 0.01)
        self.assertIsInstance(got, list)
        self.assertTrue(all(isinstance(w, tuple) and len(w) == 2 for w in got))
        self.assertTrue(all(isinstance(x, int) for w in got for x in w))

    def test_invalid_arguments_raise(self):
        for args in [(-1, 16000, 30, 5), (1000, 0, 30, 5), (1000, 16000, 0, 5), (1000, 16000, 30, -1),
                     (1000, 16000, 0.01, 0.01), (1000, 16000, 5, 30)]:
            with self.assertRaises(ValueError, msg=f"chunk_audio{args}"):
                chunk_audio(*args)


if __name__ == "__main__":
    unittest.main(verbosity=2)
