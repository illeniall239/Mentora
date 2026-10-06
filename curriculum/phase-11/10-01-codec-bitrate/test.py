import math
import unittest

from solution import codec_bitrate, compression_ratio, tokens_per_second


class TestCodecBitrate(unittest.TestCase):
    def test_encodec_24khz_six_kbps(self):
        self.assertAlmostEqual(codec_bitrate(75, 8, 1024), 6000.0, places=9)
        self.assertAlmostEqual(codec_bitrate(75, 2, 1024), 1500.0, places=9)
        self.assertAlmostEqual(codec_bitrate(50, 4, 2048), 2200.0, places=9)

    def test_codebook_size_enters_as_log2_not_linearly(self):
        self.assertAlmostEqual(codec_bitrate(75, 1, 1024), 750.0, places=9)
        # Doubling the codebook size adds one bit per token; it does not double the bitrate.
        self.assertAlmostEqual(codec_bitrate(75, 1, 2048) - codec_bitrate(75, 1, 1024), 75.0, places=9)
        self.assertAlmostEqual(codec_bitrate(75, 1, 1000), 75 * math.log2(1000), places=6)

    def test_bitrate_is_linear_in_frame_rate_and_codebooks(self):
        self.assertAlmostEqual(codec_bitrate(150, 8, 1024), 2 * codec_bitrate(75, 8, 1024), places=9)
        self.assertAlmostEqual(codec_bitrate(75, 16, 1024), 2 * codec_bitrate(75, 8, 1024), places=9)
        self.assertAlmostEqual(codec_bitrate(75, 32, 2), 2400.0, places=9)

    def test_tokens_per_second(self):
        self.assertAlmostEqual(tokens_per_second(75, 8), 600.0, places=9)
        self.assertAlmostEqual(tokens_per_second(75, 1), 75.0, places=9)
        # The codebook size changes the bits per token, never the number of tokens.
        self.assertAlmostEqual(tokens_per_second(50, 4), 200.0, places=9)

    def test_compression_ratio_against_pcm(self):
        self.assertAlmostEqual(compression_ratio(24000, 16, 6000.0), 64.0, places=9)
        self.assertAlmostEqual(compression_ratio(24000, 16, codec_bitrate(75, 8, 1024)), 64.0, places=9)
        self.assertAlmostEqual(compression_ratio(48000, 16, 24000.0), 32.0, places=9)
        self.assertAlmostEqual(compression_ratio(16000, 8, 128000.0), 1.0, places=9)

    def test_validation(self):
        with self.assertRaises(ValueError):
            codec_bitrate(0, 8, 1024)
        with self.assertRaises(ValueError):
            codec_bitrate(75, 0, 1024)
        with self.assertRaises(ValueError):
            codec_bitrate(75, 8, 1)
        with self.assertRaises(ValueError):
            tokens_per_second(75, 0)
        with self.assertRaises(ValueError):
            compression_ratio(24000, 16, 0.0)
        with self.assertRaises(ValueError):
            compression_ratio(0, 16, 6000.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
