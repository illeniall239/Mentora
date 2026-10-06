import math
import unittest

from solution import gpu_days

CHINCHILLA = 5.88e23


class TestGpuDays(unittest.TestCase):
    def test_one_flop_per_second_for_one_day(self):
        self.assertTrue(math.isclose(gpu_days(86_400, 1, 1.0, 1), 1.0, rel_tol=1e-9))
        self.assertTrue(math.isclose(gpu_days(2 * 86_400, 1, 1.0, 1), 2.0, rel_tol=1e-9))

    def test_chinchilla_on_a_thousand_gpus(self):
        want = CHINCHILLA / (1e15 * 0.4 * 1000 * 86_400)
        self.assertTrue(math.isclose(gpu_days(CHINCHILLA, 1e15, 0.4, 1000), want, rel_tol=1e-9))
        self.assertTrue(math.isclose(want, 17.0138888, rel_tol=1e-6))

    def test_more_gpus_means_fewer_days(self):
        base = gpu_days(CHINCHILLA, 1e15, 0.4, 1000)
        self.assertTrue(math.isclose(gpu_days(CHINCHILLA, 1e15, 0.4, 2000), base / 2, rel_tol=1e-9))
        self.assertTrue(math.isclose(gpu_days(CHINCHILLA, 1e15, 0.4, 250), base * 4, rel_tol=1e-9))

    def test_utilisation_scales_inversely(self):
        base = gpu_days(CHINCHILLA, 1e15, 0.4, 1000)
        self.assertTrue(math.isclose(gpu_days(CHINCHILLA, 1e15, 0.2, 1000), base * 2, rel_tol=1e-9))
        self.assertTrue(math.isclose(gpu_days(CHINCHILLA, 1e15, 1.0, 1000), base * 0.4, rel_tol=1e-9))

    def test_faster_gpus_and_more_flops(self):
        base = gpu_days(1e21, 1e15, 0.5, 8)
        self.assertTrue(math.isclose(gpu_days(1e21, 2e15, 0.5, 8), base / 2, rel_tol=1e-9))
        self.assertTrue(math.isclose(gpu_days(3e21, 1e15, 0.5, 8), base * 3, rel_tol=1e-9))

    def test_invalid_inputs_raise(self):
        for args in [(0, 1e15, 0.4, 8), (1e20, 0, 0.4, 8), (1e20, 1e15, 0.4, 0), (1e20, 1e15, 0.0, 8), (1e20, 1e15, 1.5, 8), (1e20, 1e15, -0.4, 8), (-1e20, 1e15, 0.4, 8)]:
            with self.assertRaises(ValueError):
                gpu_days(*args)
        gpu_days(1e20, 1e15, 1.0, 8)  # exactly 100 % MFU is allowed


if __name__ == "__main__":
    unittest.main(verbosity=2)
