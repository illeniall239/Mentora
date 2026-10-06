import random
import unittest

from solution import elo_update


class TestEloUpdate(unittest.TestCase):
    def test_even_match(self):
        a, b = elo_update(1500, 1500, 1.0, 32)
        self.assertAlmostEqual(a, 1516.0, places=9)
        self.assertAlmostEqual(b, 1484.0, places=9)
        a, b = elo_update(1500, 1500, 0.0, 32)
        self.assertAlmostEqual(a, 1484.0, places=9)
        self.assertAlmostEqual(b, 1516.0, places=9)

    def test_tie_between_equals_moves_nothing(self):
        a, b = elo_update(1500, 1500, 0.5, 32)
        self.assertAlmostEqual(a, 1500.0, places=9)
        self.assertAlmostEqual(b, 1500.0, places=9)

    def test_expected_score_uses_the_right_sign_of_the_gap(self):
        # 400-point favourite expects 10/11 of a point: winning gains only ~2.9
        a, b = elo_update(1600, 1200, 1.0, 32)
        self.assertAlmostEqual(a, 1600 + 32 * (1 - 1 / 1.1), places=7)
        self.assertAlmostEqual(b, 1200 - 32 * (1 - 1 / 1.1), places=7)
        self.assertLess(a - 1600, 3.0)
        a, b = elo_update(1600, 1200, 0.0, 32)
        self.assertAlmostEqual(a, 1600 - 32 / 1.1, places=7)
        self.assertAlmostEqual(b, 1200 + 32 / 1.1, places=7)

    def test_upset_moves_more_than_an_expected_win(self):
        fav_win, _ = elo_update(1600, 1200, 1.0, 32)
        _, underdog_win = elo_update(1600, 1200, 0.0, 32)
        self.assertAlmostEqual(underdog_win - 1200, 10 * (fav_win - 1600), places=6)
        self.assertGreater(underdog_win - 1200, 25.0)

    def test_zero_sum_for_arbitrary_matches(self):
        rng = random.Random(3)
        for _ in range(200):
            ra, rb = rng.uniform(600, 2600), rng.uniform(600, 2600)
            outcome, k = rng.choice([0.0, 0.5, 1.0, 0.25]), rng.uniform(4, 64)
            a, b = elo_update(ra, rb, outcome, k)
            self.assertAlmostEqual(a + b, ra + rb, places=6)

    def test_symmetric_under_swapping_the_players(self):
        rng = random.Random(4)
        for _ in range(100):
            ra, rb = rng.uniform(600, 2600), rng.uniform(600, 2600)
            outcome, k = rng.uniform(0, 1), 24.0
            a, b = elo_update(ra, rb, outcome, k)
            b2, a2 = elo_update(rb, ra, 1 - outcome, k)
            self.assertAlmostEqual(a, a2, places=6)
            self.assertAlmostEqual(b, b2, places=6)

    def test_k_scales_the_change_linearly(self):
        base = elo_update(1700, 1540, 1.0, 10)[0] - 1700
        for k in (20.0, 55.0):
            got = elo_update(1700, 1540, 1.0, k)[0] - 1700
            self.assertAlmostEqual(got, base * k / 10, places=7)

    def test_returns_a_tuple_of_two_floats(self):
        result = elo_update(1500, 1480, 0.5, 16)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        self.assertTrue(all(isinstance(x, float) for x in result))

    def test_invalid_arguments_raise(self):
        for args in [(1500, 1500, 1.5, 32), (1500, 1500, -0.1, 32), (1500, 1500, 1.0, 0), (1500, 1500, 1.0, -5)]:
            with self.assertRaises(ValueError, msg=f"elo_update{args}"):
                elo_update(*args)


if __name__ == "__main__":
    unittest.main(verbosity=2)
