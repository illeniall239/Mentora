import unittest

from solution import count_params

GPT2_SMALL = dict(vocab=50257, d_model=768, n_layers=12, n_heads=12, ff_mult=4, tied=True)


class TestCountGptParams(unittest.TestCase):
    def test_gpt2_small_lands_on_124m(self):
        total = count_params(**GPT2_SMALL)
        self.assertIsInstance(total, int)
        self.assertEqual(total, 124_439_808)

    def test_hand_countable_tiny_model(self):
        # 10*4 token + 5*4 pos + (8 + 60 + 20 + 8 + 80 + 68) block + 8 final LN
        self.assertEqual(count_params(10, 4, 1, 2, 4, True, n_positions=5), 312)
        self.assertEqual(count_params(10, 4, 0, 2, 4, True, n_positions=5), 68)
        self.assertEqual(count_params(10, 4, 2, 2, 4, True, n_positions=5), 312 + 244)

    def test_heads_are_free(self):
        for n_heads in (1, 2, 3, 4, 6, 12, 16, 768):
            self.assertEqual(count_params(**{**GPT2_SMALL, "n_heads": n_heads}), 124_439_808)

    def test_untying_adds_exactly_one_embedding_matrix(self):
        tied = count_params(**GPT2_SMALL)
        untied = count_params(**{**GPT2_SMALL, "tied": False})
        self.assertEqual(untied - tied, 50257 * 768)
        self.assertEqual(untied, 163_037_184)
        self.assertEqual(count_params(10, 4, 1, 2, 4, False, n_positions=5), 312 + 40)

    def test_per_block_cost_includes_the_output_projection_and_every_bias(self):
        d, ff_mult = 64, 4
        one = count_params(100, d, 1, 8, ff_mult, True, n_positions=16)
        two = count_params(100, d, 2, 8, ff_mult, True, n_positions=16)
        per_block = two - one
        d_ff = ff_mult * d
        attn = 4 * (d * d + d)  # q, k, v AND the output projection, each with a bias
        ffn = (d * d_ff + d_ff) + (d_ff * d + d)
        self.assertEqual(per_block, 2 * (2 * d) + attn + ffn)
        # Dropping W_O, or dropping the biases, would each change this number.
        self.assertNotEqual(per_block, 2 * (2 * d) + 3 * (d * d + d) + ffn)
        self.assertNotEqual(per_block, 2 * (2 * d) + 4 * d * d + 2 * d * d_ff)

    def test_ffn_holds_about_two_thirds_of_a_block(self):
        d, n = 768, 12
        block = (count_params(1, d, n + 1, 12, 4, True, n_positions=1) - count_params(1, d, n, 12, 4, True, n_positions=1))
        ffn = (d * 4 * d + 4 * d) + (4 * d * d + d)
        self.assertAlmostEqual(ffn / block, 2 / 3, places=2)

    def test_position_embedding_scales_with_the_context_limit(self):
        base = count_params(**GPT2_SMALL)
        self.assertEqual(count_params(**GPT2_SMALL, n_positions=2048) - base, 1024 * 768)

    def test_rejects_illegal_configurations(self):
        with self.assertRaises(ValueError):
            count_params(50257, 768, 12, 5, 4, True)
        with self.assertRaises(ValueError):
            count_params(50257, 768, 12, 0, 4, True)
        with self.assertRaises(ValueError):
            count_params(0, 768, 12, 12, 4, True)
        with self.assertRaises(ValueError):
            count_params(50257, 768, -1, 12, 4, True)
        with self.assertRaises(ValueError):
            count_params(50257, 768, 12, 12, 4, True, n_positions=0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
