import unittest

from solution import kv_cache_bytes, moe_active_params

GIB = 1024 ** 3


class TestKvCacheAndMoeMath(unittest.TestCase):
    def test_llama2_7b_fp16_is_two_gib(self):
        self.assertEqual(kv_cache_bytes(32, 32, 128, 4096, 1, 2), 2 * GIB)

    def test_gqa_shrinks_the_cache_by_the_head_ratio(self):
        mha = kv_cache_bytes(32, 32, 128, 4096, 1, 2)
        gqa = kv_cache_bytes(32, 8, 128, 4096, 1, 2)
        self.assertEqual(gqa, 512 * 1024 ** 2)
        self.assertEqual(mha // gqa, 4)

    def test_keys_and_values_both_counted(self):
        self.assertEqual(kv_cache_bytes(1, 1, 1, 1, 1, 1), 2)
        self.assertEqual(kv_cache_bytes(2, 3, 5, 7, 11, 4), 2 * 2 * 3 * 5 * 7 * 11 * 4)

    def test_linear_in_sequence_length_batch_and_precision(self):
        base = kv_cache_bytes(4, 4, 64, 100, 1, 2)
        self.assertEqual(kv_cache_bytes(4, 4, 64, 200, 1, 2), 2 * base)
        self.assertEqual(kv_cache_bytes(4, 4, 64, 100, 8, 2), 8 * base)
        self.assertEqual(kv_cache_bytes(4, 4, 64, 100, 1, 4), 2 * base)
        self.assertEqual(kv_cache_bytes(4, 4, 64, 100, 1, 1), base // 2)

    def test_kv_cache_rejects_nonpositive_arguments(self):
        for args in [(0, 8, 128, 4096, 1, 2), (32, 8, 128, 0, 1, 2), (32, 8, 128, 4096, 1, 0), (32, -1, 128, 4096, 1, 2)]:
            with self.assertRaises(ValueError, msg=str(args)):
                kv_cache_bytes(*args)

    def test_moe_active_is_shared_plus_top_k_experts(self):
        shared, per_expert = 1_300_000_000, 5_600_000_000
        active = moe_active_params(shared, per_expert, 8, 2)
        total = shared + 8 * per_expert
        self.assertEqual(active, 12_500_000_000)
        self.assertLess(active, total)
        self.assertEqual(total, 46_100_000_000)

    def test_moe_dense_when_every_expert_is_active(self):
        self.assertEqual(moe_active_params(100, 10, 4, 4), 140)
        self.assertEqual(moe_active_params(100, 10, 4, 1), 110)
        self.assertEqual(moe_active_params(0, 10, 4, 2), 20)

    def test_moe_rejects_bad_configurations(self):
        for args in [(100, 10, 4, 5), (100, 10, 4, 0), (100, 10, 0, 0), (-1, 10, 4, 2), (100, -10, 4, 2)]:
            with self.assertRaises(ValueError, msg=str(args)):
                moe_active_params(*args)


if __name__ == "__main__":
    unittest.main(verbosity=2)
