import unittest

import numpy as np

from solution import zero_shot_classify


class TestZeroShotClassify(unittest.TestCase):
    def test_picks_the_aligned_label(self):
        labels = np.array([[0.0, 1.0], [1.0, 0.0]])
        self.assertEqual(zero_shot_classify(np.array([1.0, 0.0]), labels), 1)
        self.assertEqual(zero_shot_classify(np.array([0.0, 5.0]), labels), 0)

    def test_normalization_beats_the_raw_dot_product(self):
        # Label 0 is long but points elsewhere: dot 9.0 vs 1.0, cosine 0.9939 vs 1.0.
        img = np.array([1.0, 0.0])
        labels = np.array([[9.0, 1.0], [1.0, 0.0]])
        self.assertEqual(zero_shot_classify(img, labels), 1)
        # Making it 100x longer must still not win.
        labels2 = labels.copy()
        labels2[0] *= 100.0
        self.assertEqual(zero_shot_classify(img, labels2), 1)

    def test_invariant_to_positive_scaling(self):
        rng = np.random.default_rng(0)
        img = rng.normal(size=16)
        labels = rng.normal(size=(7, 16))
        want = zero_shot_classify(img, labels)
        self.assertEqual(zero_shot_classify(img * 37.0, labels), want)
        scaled = labels.copy()
        scaled[3] *= 0.01
        scaled[5] *= 250.0
        self.assertEqual(zero_shot_classify(img, scaled), want)

    def test_matches_explicit_cosine_scores(self):
        rng = np.random.default_rng(1)
        for k, d in ((3, 4), (20, 8), (64, 32)):
            img = rng.normal(size=d)
            labels = rng.normal(size=(k, d))
            cos = [
                float(img @ labels[i] / (np.linalg.norm(img) * np.linalg.norm(labels[i]))) for i in range(k)
            ]
            self.assertEqual(zero_shot_classify(img, labels), int(np.argmax(cos)))

    def test_all_scores_negative_still_returns_the_argmax(self):
        img = np.array([1.0, 0.0])
        labels = np.array([[-1.0, 0.0], [-1.0, -1.0]])
        self.assertEqual(zero_shot_classify(img, labels), 1)  # -1.0 vs -0.7071

    def test_ties_go_to_the_lowest_index(self):
        img = np.array([1.0, 1.0])
        self.assertEqual(zero_shot_classify(img, np.array([[1.0, 1.0], [1.0, 1.0]])), 0)
        self.assertEqual(zero_shot_classify(img, np.array([[0.0, 1.0], [2.0, 2.0], [2.0, 2.0]])), 1)

    def test_bad_inputs_raise(self):
        img = np.array([1.0, 0.0])
        labels = np.array([[1.0, 0.0], [0.0, 1.0]])
        with self.assertRaises(ValueError):
            zero_shot_classify(np.array([[1.0, 0.0]]), labels)
        with self.assertRaises(ValueError):
            zero_shot_classify(img, np.array([1.0, 0.0]))
        with self.assertRaises(ValueError):
            zero_shot_classify(img, np.zeros((0, 2)))
        with self.assertRaises(ValueError):
            zero_shot_classify(img, np.ones((3, 5)))
        with self.assertRaises(ValueError):
            zero_shot_classify(img, np.array([[0.0, 0.0], [1.0, 0.0]]))
        with self.assertRaises(ValueError):
            zero_shot_classify(np.zeros(2), labels)

    def test_inputs_are_not_modified(self):
        rng = np.random.default_rng(2)
        img = rng.normal(size=8)
        labels = rng.normal(size=(5, 8))
        img_copy, labels_copy = img.copy(), labels.copy()
        zero_shot_classify(img, labels)
        self.assertTrue(np.array_equal(img, img_copy))
        self.assertTrue(np.array_equal(labels, labels_copy))


if __name__ == "__main__":
    unittest.main(verbosity=2)
