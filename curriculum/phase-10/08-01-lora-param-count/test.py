import unittest

from solution import full_param_count, lora_param_count


class TestLoraParamCount(unittest.TestCase):
    def test_full_matrix(self):
        self.assertEqual(full_param_count(4096, 4096), 16_777_216)
        self.assertEqual(full_param_count(4096, 11008), 45_088_768)
        self.assertEqual(full_param_count(1, 1), 1)

    def test_square_4096_rank_8(self):
        self.assertEqual(lora_param_count(4096, 4096, 8), 65_536)
        self.assertLess(lora_param_count(4096, 4096, 8) / full_param_count(4096, 4096), 0.005)

    def test_non_square_counts_both_factors(self):
        self.assertEqual(lora_param_count(4096, 11008, 8), 8 * (4096 + 11008))
        self.assertEqual(lora_param_count(11008, 4096, 8), 8 * (4096 + 11008))
        self.assertEqual(lora_param_count(3, 5, 1), 8)

    def test_linear_in_rank(self):
        self.assertEqual(lora_param_count(4096, 4096, 16), 2 * lora_param_count(4096, 4096, 8))
        self.assertEqual(lora_param_count(4096, 4096, 64), 8 * lora_param_count(4096, 4096, 8))

    def test_frozen_weights_are_not_counted(self):
        self.assertEqual(lora_param_count(4096, 4096, 8), 8 * (4096 + 4096))
        self.assertLess(lora_param_count(4096, 4096, 8), full_param_count(4096, 4096))

    def test_invalid_inputs_raise(self):
        with self.assertRaises(ValueError):
            full_param_count(0, 5)
        with self.assertRaises(ValueError):
            full_param_count(5, -1)
        for d_in, d_out, r in [(4096, 4096, 0), (4096, 4096, 5000), (4, 8, 5), (0, 4, 1), (4, 0, 1)]:
            with self.assertRaises(ValueError):
                lora_param_count(d_in, d_out, r)
        self.assertEqual(lora_param_count(4, 8, 4), 48)  # r == min(d_in, d_out) is still allowed


if __name__ == "__main__":
    unittest.main(verbosity=2)
