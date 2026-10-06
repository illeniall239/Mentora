import unittest

import numpy as np
from numpy.testing import assert_allclose

from solution import conv2d_valid


class TestConv2dValid(unittest.TestCase):
    def setUp(self):
        self.img = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])

    def test_hand_result_with_diagonal_kernel(self):
        out = conv2d_valid(self.img, np.array([[1.0, 0.0], [0.0, 1.0]]))
        assert_allclose(out, [[6.0, 8.0], [12.0, 14.0]])

    def test_1x1_identity_kernel_returns_the_image(self):
        assert_allclose(conv2d_valid(self.img, np.array([[1.0]])), self.img)

    def test_full_size_kernel_gives_a_single_number(self):
        out = conv2d_valid(self.img, np.ones((3, 3)))
        self.assertEqual(out.shape, (1, 1))
        assert_allclose(out, [[45.0]])

    def test_output_shape_for_rectangular_inputs(self):
        rng = np.random.default_rng(0)
        out = conv2d_valid(rng.normal(size=(10, 7)), rng.normal(size=(3, 2)))
        self.assertEqual(out.shape, (8, 6))

    def test_kernel_is_not_flipped(self):
        top_left = conv2d_valid(self.img, np.array([[1.0, 0.0], [0.0, 0.0]]))
        assert_allclose(top_left, [[1.0, 2.0], [4.0, 5.0]])
        bottom_right = conv2d_valid(self.img, np.array([[0.0, 0.0], [0.0, 1.0]]))
        assert_allclose(bottom_right, [[5.0, 6.0], [8.0, 9.0]])

    def test_box_filter_matches_window_means(self):
        rng = np.random.default_rng(1)
        img = rng.normal(size=(6, 6))
        out = conv2d_valid(img, np.full((3, 3), 1.0 / 9.0))
        expected = np.array([[img[i : i + 3, j : j + 3].mean() for j in range(4)] for i in range(4)])
        assert_allclose(out, expected, atol=1e-9)

    def test_kernel_larger_than_image_raises(self):
        with self.assertRaises(ValueError):
            conv2d_valid(self.img, np.ones((4, 2)))

    def test_inputs_are_not_modified(self):
        img = self.img.copy()
        kernel = np.array([[1.0, -1.0], [0.5, 2.0]])
        conv2d_valid(img, kernel)
        assert_allclose(img, self.img)
        assert_allclose(kernel, [[1.0, -1.0], [0.5, 2.0]])


if __name__ == "__main__":
    unittest.main(verbosity=2)
