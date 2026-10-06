import random
import unittest

from solution import bradley_terry_ratings, judge_position_swap_agreement, judge_win_rate

TWO_PLAYER = [("A", "B"), ("A", "B"), ("A", "B"), ("B", "A")]
TRANSITIVE = [("A", "B"), ("A", "B"), ("B", "C"), ("B", "C"), ("C", "A")]


class TestArenaRatingsAndJudgeBias(unittest.TestCase):
    def test_two_players_match_the_empirical_win_rate(self):
        ratings = bradley_terry_ratings(TWO_PLAYER, 50)
        self.assertEqual(set(ratings), {"A", "B"})
        self.assertAlmostEqual(ratings["A"], 0.75, places=9)
        self.assertAlmostEqual(ratings["B"], 0.25, places=9)
        p_a_beats_b = ratings["A"] / (ratings["A"] + ratings["B"])
        self.assertAlmostEqual(p_a_beats_b, 3 / 4, places=9)

    def test_ratings_are_normalized_and_order_independent(self):
        ratings = bradley_terry_ratings(TRANSITIVE, 200)
        self.assertAlmostEqual(sum(ratings.values()), 1.0, places=9)
        shuffled = list(TRANSITIVE)
        random.Random(3).shuffle(shuffled)
        other = bradley_terry_ratings(shuffled, 200)
        for name in ratings:
            self.assertAlmostEqual(ratings[name], other[name], places=9)

    def test_transitive_tournament_orders_the_players(self):
        ratings = bradley_terry_ratings(TRANSITIVE, 200)
        self.assertGreater(ratings["A"], ratings["B"])
        self.assertGreater(ratings["B"], ratings["C"])
        self.assertAlmostEqual(ratings["A"], 0.5161117449569453, places=6)
        self.assertAlmostEqual(ratings["B"], 0.3043792304401380, places=6)
        self.assertAlmostEqual(ratings["C"], 0.1795090246029168, places=6)
        # Converged: more rounds do not move it.
        more = bradley_terry_ratings(TRANSITIVE, 500)
        for name in ratings:
            self.assertAlmostEqual(ratings[name], more[name], places=9)

    def test_cyclic_tournament_is_a_three_way_tie(self):
        ratings = bradley_terry_ratings([("A", "B"), ("B", "C"), ("C", "A")], 200)
        for name in "ABC":
            self.assertAlmostEqual(ratings[name], 1 / 3, places=9)

    def test_ratings_validation(self):
        with self.assertRaises(ValueError):
            bradley_terry_ratings([], 10)
        with self.assertRaises(ValueError):
            bradley_terry_ratings(TWO_PLAYER, 0)
        with self.assertRaises(ValueError):
            bradley_terry_ratings([("A", "A")], 10)
        with self.assertRaises(ValueError):
            bradley_terry_ratings([("A", "B")], 10)
        with self.assertRaises(ValueError):
            bradley_terry_ratings([("A", "B"), ("A", "B"), ("B", "C")], 10)

    def test_position_swap_agreement(self):
        self.assertAlmostEqual(judge_position_swap_agreement(["A"] * 4, ["B"] * 4), 0.0, places=9)
        self.assertAlmostEqual(judge_position_swap_agreement(["A"] * 4, ["A"] * 4), 1.0, places=9)
        self.assertAlmostEqual(judge_position_swap_agreement(["tie", "A"], ["tie", "B"]), 0.5, places=9)
        self.assertAlmostEqual(judge_position_swap_agreement(["A", "B", "tie"], ["A", "A", "tie"]), 2 / 3, places=9)

    def test_win_rate_counts_ties_as_half(self):
        self.assertAlmostEqual(judge_win_rate(["tie", "A"], ["tie", "A"]), 0.75, places=9)
        self.assertNotAlmostEqual(judge_win_rate(["tie", "A"], ["tie", "A"]), 1.0, places=3)
        self.assertAlmostEqual(judge_win_rate(["tie"] * 3, ["tie"] * 3), 0.5, places=9)
        self.assertAlmostEqual(judge_win_rate(["A", "B"], ["A", "B"]), 0.5, places=9)

    def test_win_rate_averages_both_orderings(self):
        self.assertAlmostEqual(judge_win_rate(["A", "A"], ["B", "A"]), 0.75, places=9)
        self.assertNotAlmostEqual(judge_win_rate(["A", "A"], ["B", "A"]), 1.0, places=3)
        # A judge that always picks the first candidate looks perfectly even, and the agreement exposes it.
        self.assertAlmostEqual(judge_win_rate(["A"] * 4, ["B"] * 4), 0.5, places=9)
        self.assertAlmostEqual(judge_position_swap_agreement(["A"] * 4, ["B"] * 4), 0.0, places=9)

    def test_verdict_validation(self):
        for fn in (judge_position_swap_agreement, judge_win_rate):
            with self.assertRaises(ValueError):
                fn([], [])
            with self.assertRaises(ValueError):
                fn(["A"], ["A", "B"])
            with self.assertRaises(ValueError):
                fn(["A"], ["draw"])
            with self.assertRaises(ValueError):
                fn(["a"], ["A"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
