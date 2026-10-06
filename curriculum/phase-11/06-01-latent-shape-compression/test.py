import unittest

from solution import compression_ratio, latent_shape


class TestLatentShape(unittest.TestCase):
    def test_stable_diffusion_default(self):
        self.assertEqual(latent_shape(512, 512, 8, 4), (4, 64, 64))
        self.assertEqual(latent_shape(1024, 1024, 8, 16), (16, 128, 128))

    def test_shape_is_channels_first_and_height_before_width(self):
        self.assertEqual(latent_shape(768, 512, 8, 4), (4, 96, 64))
        self.assertEqual(latent_shape(512, 768, 8, 4), (4, 64, 96))

    def test_shape_is_a_tuple_of_three_ints(self):
        shape = latent_shape(256, 128, 4, 3)
        self.assertIsInstance(shape, tuple)
        self.assertEqual(len(shape), 3)
        self.assertTrue(all(isinstance(x, int) and not isinstance(x, bool) for x in shape))
        self.assertEqual(shape, (3, 64, 32))

    def test_compression_ratio_known_values(self):
        self.assertAlmostEqual(compression_ratio(512, 512, 8, 4), 48.0, places=9)
        self.assertAlmostEqual(compression_ratio(512, 512, 8, 16), 12.0, places=9)
        self.assertAlmostEqual(compression_ratio(768, 512, 8, 4), 48.0, places=9)

    def test_no_downsampling_and_no_extra_channels_means_no_compression(self):
        self.assertAlmostEqual(compression_ratio(64, 64, 1, 3), 1.0, places=9)
        self.assertAlmostEqual(compression_ratio(64, 64, 1, 6), 0.5, places=9)

    def test_ratio_scales_with_the_square_of_down_not_with_down(self):
        self.assertAlmostEqual(compression_ratio(512, 512, 4, 3), 16.0, places=9)
        self.assertAlmostEqual(compression_ratio(512, 512, 8, 3), 64.0, places=9)

    def test_ratio_is_a_true_float_division(self):
        value = compression_ratio(256, 256, 8, 7)
        self.assertIsInstance(value, float)
        self.assertAlmostEqual(value, (256 * 256 * 3) / (7 * 32 * 32), places=9)
        self.assertNotAlmostEqual(value, float((256 * 256 * 3) // (7 * 32 * 32)), places=6)

    def test_invalid_inputs_raise(self):
        for args in [(515, 512, 8, 4), (512, 515, 8, 4), (512, 512, 0, 4), (512, 512, 8, 0), (-8, 512, 8, 4)]:
            with self.assertRaises(ValueError, msg=f"latent_shape{args}"):
                latent_shape(*args)
            with self.assertRaises(ValueError, msg=f"compression_ratio{args}"):
                compression_ratio(*args)


if __name__ == "__main__":
    unittest.main(verbosity=2)
