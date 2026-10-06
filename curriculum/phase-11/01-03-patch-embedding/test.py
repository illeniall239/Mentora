import unittest

import numpy as np

from solution import patch_embed


def make(h: int, w: int, c: int, p: int, d: int, seed: int):
    rng = np.random.default_rng(seed)
    img = rng.normal(size=(h, w, c))
    W = rng.normal(size=(p * p * c, d))
    n = (h // p) * (w // p)
    return img, W, rng.normal(size=(n + 1, d)), rng.normal(size=(d,))


def slow_patches(img: np.ndarray, p: int) -> np.ndarray:
    """Patch matrix built with explicit loops, in the order the problem specifies."""
    h, w, c = img.shape
    rows = []
    for pr in range(h // p):
        for pc in range(w // p):
            rows.append(
                [
                    img[r, col, ch]
                    for r in range(pr * p, pr * p + p)
                    for col in range(pc * p, pc * p + p)
                    for ch in range(c)
                ]
            )
    return np.array(rows, dtype=float)


class TestPatchEmbed(unittest.TestCase):
    def test_shape_is_n_plus_one_by_d(self):
        img, W, pos, cls_vec = make(4, 6, 3, 2, 8, seed=0)
        self.assertEqual(patch_embed(img, 2, W, pos, cls_vec).shape, (7, 8))
        img, W, pos, cls_vec = make(8, 8, 1, 4, 5, seed=1)
        self.assertEqual(patch_embed(img, 4, W, pos, cls_vec).shape, (5, 5))

    def test_matches_the_explicit_construction(self):
        for h, w, c, p, d, seed in ((4, 4, 3, 2, 8, 2), (6, 9, 2, 3, 4, 3), (8, 8, 1, 8, 6, 4)):
            img, W, pos, cls_vec = make(h, w, c, p, d, seed)
            want = np.concatenate([cls_vec[None, :], slow_patches(img, p) @ W], axis=0) + pos
            got = patch_embed(img, p, W, pos, cls_vec)
            self.assertTrue(np.allclose(got, want, atol=1e-12), f"{h}x{w}x{c} p={p}")

    def test_cls_row_is_first_and_takes_pos_zero(self):
        img, W, pos, cls_vec = make(4, 4, 2, 2, 6, seed=5)
        out = patch_embed(img, 2, W, pos, cls_vec)
        self.assertTrue(np.allclose(out[0], cls_vec + pos[0], atol=1e-12))
        # Patch 0 must take pos[1], not pos[0].
        self.assertTrue(np.allclose(out[1] - pos[1], slow_patches(img, 2)[0] @ W, atol=1e-12))

    def test_positions_are_added_not_dropped(self):
        img, W, pos, cls_vec = make(4, 4, 3, 2, 8, seed=6)
        out = patch_embed(img, 2, W, pos, cls_vec)
        self.assertEqual(out.shape[1], 8)
        zero_pos = patch_embed(img, 2, W, np.zeros_like(pos), cls_vec)
        self.assertTrue(np.allclose(out - zero_pos, pos, atol=1e-12))
        self.assertFalse(np.allclose(out, zero_pos, atol=1e-6))

    def test_projection_is_shared_across_patches(self):
        # Identical pixel blocks in two places give identical tokens once positions are removed.
        img = np.zeros((4, 4, 1))
        img[0:2, 0:2, 0] = [[1.0, 2.0], [3.0, 4.0]]
        img[2:4, 2:4, 0] = [[1.0, 2.0], [3.0, 4.0]]
        _, W, pos, cls_vec = make(4, 4, 1, 2, 5, seed=7)
        out = patch_embed(img, 2, W, pos, cls_vec)
        self.assertTrue(np.allclose(out[1] - pos[1], out[4] - pos[4], atol=1e-12))

    def test_patch_order_is_row_major_and_flatten_is_row_col_channel(self):
        img = np.arange(16, dtype=float).reshape(4, 4, 1)
        W = np.zeros((4, 2))
        W[0, 0] = 1.0  # component 0 reads the patch's first flattened value
        W[1, 1] = 1.0  # component 1 reads the second
        out = patch_embed(img, 2, W, np.zeros((5, 2)), np.zeros(2))
        self.assertTrue(np.allclose(out[1:, 0], [0.0, 2.0, 8.0, 10.0], atol=1e-12))
        # The second flattened value is the pixel to the RIGHT, not the one below.
        self.assertTrue(np.allclose(out[1:, 1], [1.0, 3.0, 9.0, 11.0], atol=1e-12))

    def test_bad_shapes_raise(self):
        img, W, pos, cls_vec = make(4, 4, 3, 2, 8, seed=8)
        with self.assertRaises(ValueError):
            patch_embed(np.zeros((5, 4, 3)), 2, W, pos, cls_vec)
        with self.assertRaises(ValueError):
            patch_embed(np.zeros((4, 4)), 2, W, pos, cls_vec)
        with self.assertRaises(ValueError):
            patch_embed(img, 2, np.zeros((11, 8)), pos, cls_vec)
        with self.assertRaises(ValueError):
            patch_embed(img, 2, W, np.zeros((4, 8)), cls_vec)
        with self.assertRaises(ValueError):
            patch_embed(img, 2, W, pos, np.zeros(7))

    def test_inputs_are_not_modified(self):
        img, W, pos, cls_vec = make(4, 4, 2, 2, 6, seed=9)
        copies = [a.copy() for a in (img, W, pos, cls_vec)]
        patch_embed(img, 2, W, pos, cls_vec)
        for before, after in zip(copies, (img, W, pos, cls_vec)):
            self.assertTrue(np.array_equal(before, after))


if __name__ == "__main__":
    unittest.main(verbosity=2)
