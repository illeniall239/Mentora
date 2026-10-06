import copy
import random
import unittest

from solution import num_patch_tokens, patchify, unpatchify


def make_img(h: int, w: int, c: int, seed: int) -> list[list[list[float]]]:
    rng = random.Random(seed)
    return [[[rng.randint(0, 255) for _ in range(c)] for _ in range(w)] for _ in range(h)]


class TestPatchify(unittest.TestCase):
    def test_single_channel_hand_example(self):
        img = [[[1], [2], [3], [4]], [[5], [6], [7], [8]]]
        self.assertEqual(patchify(img, 2), [[1, 2, 5, 6], [3, 4, 7, 8]])

    def test_two_channel_flatten_order_is_row_col_channel(self):
        img = [[[1, 10], [2, 20]], [[3, 30], [4, 40]]]
        self.assertEqual(patchify(img, 2), [[1, 10, 2, 20, 3, 30, 4, 40]])

    def test_patches_are_ordered_row_major(self):
        img = [[[r * 10 + c] for c in range(4)] for r in range(4)]
        patches = patchify(img, 2)
        self.assertEqual([p[0] for p in patches], [0, 2, 20, 22])

    def test_patch_count_and_length(self):
        img = make_img(12, 8, 3, seed=1)
        patches = patchify(img, 4)
        self.assertEqual(len(patches), 6)
        self.assertTrue(all(len(p) == 4 * 4 * 3 for p in patches))

    def test_unpatchify_inverts_patchify(self):
        for h, w, c, p in [(4, 4, 1, 2), (6, 9, 3, 3), (8, 8, 2, 8), (5, 5, 1, 1)]:
            img = make_img(h, w, c, seed=h * w)
            self.assertEqual(unpatchify(patchify(img, p), h, w, p, c), img)

    def test_num_patch_tokens_for_vit_base(self):
        self.assertEqual(num_patch_tokens(224, 224, 16, True), 197)
        self.assertEqual(num_patch_tokens(224, 224, 16, False), 196)
        self.assertEqual(num_patch_tokens(32, 64, 8, True), 33)

    def test_non_multiple_sizes_raise(self):
        with self.assertRaises(ValueError):
            num_patch_tokens(225, 224, 16, False)
        with self.assertRaises(ValueError):
            patchify(make_img(5, 4, 1, seed=3), 2)
        with self.assertRaises(ValueError):
            unpatchify([[1, 2, 3, 4]], 3, 2, 2, 1)

    def test_inputs_are_not_modified(self):
        img = make_img(4, 4, 2, seed=7)
        snapshot = copy.deepcopy(img)
        patches = patchify(img, 2)
        snapshot_patches = copy.deepcopy(patches)
        unpatchify(patches, 4, 4, 2, 2)
        self.assertEqual(img, snapshot)
        self.assertEqual(patches, snapshot_patches)


if __name__ == "__main__":
    unittest.main(verbosity=2)
