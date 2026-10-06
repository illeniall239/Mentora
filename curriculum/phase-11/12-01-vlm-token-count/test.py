import unittest

from solution import vlm_token_count


class TestVlmTokenCount(unittest.TestCase):
    def test_hand_examples(self):
        self.assertEqual(vlm_token_count(336, 336, 14, 336, 4), 1152)
        self.assertEqual(vlm_token_count(1000, 1000, 14, 336, 4), 2880)
        self.assertEqual(vlm_token_count(224, 224, 14, 224, 1), 512)

    def test_tokens_per_tile_is_patches_squared(self):
        self.assertEqual(vlm_token_count(224, 224, 7, 224, 1), 2048)
        self.assertEqual(vlm_token_count(224, 224, 28, 224, 1), 128)
        # Halving the patch size quadruples the cost of every tile.
        self.assertEqual(vlm_token_count(224, 224, 7, 224, 1), 4 * vlm_token_count(224, 224, 14, 224, 1))

    def test_partial_tiles_cost_a_whole_tile(self):
        self.assertEqual(vlm_token_count(337, 336, 14, 336, 4), 1728)
        self.assertEqual(vlm_token_count(336, 337, 14, 336, 4), 1728)
        self.assertEqual(vlm_token_count(1, 1, 14, 336, 4), 1152)
        self.assertEqual(vlm_token_count(672, 336, 14, 336, 4), 1728)

    def test_max_tiles_caps_the_cost(self):
        self.assertEqual(vlm_token_count(5000, 5000, 14, 336, 4), vlm_token_count(1000, 1000, 14, 336, 4))
        self.assertEqual(vlm_token_count(1000, 1000, 14, 336, 9), 5760)
        self.assertEqual(vlm_token_count(5000, 5000, 14, 336, 1), 1152)

    def test_thumbnail_is_always_counted(self):
        one_tile = vlm_token_count(336, 336, 14, 336, 4)
        self.assertEqual(one_tile, 2 * 576)
        self.assertNotEqual(one_tile, 576, "the global thumbnail tile is charged even for a single-tile image")

    def test_cost_grows_with_resolution_until_the_cap(self):
        sizes = [336, 672, 1008]
        counts = [vlm_token_count(s, s, 14, 336, 16) for s in sizes]
        self.assertEqual(counts, [1152, 2880, 5760])
        self.assertEqual(sorted(counts), counts)

    def test_validation(self):
        with self.assertRaises(ValueError):
            vlm_token_count(336, 336, 15, 336, 4)
        with self.assertRaises(ValueError):
            vlm_token_count(0, 336, 14, 336, 4)
        with self.assertRaises(ValueError):
            vlm_token_count(336, 0, 14, 336, 4)
        with self.assertRaises(ValueError):
            vlm_token_count(336, 336, 0, 336, 4)
        with self.assertRaises(ValueError):
            vlm_token_count(336, 336, 14, 0, 4)
        with self.assertRaises(ValueError):
            vlm_token_count(336, 336, 14, 336, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
