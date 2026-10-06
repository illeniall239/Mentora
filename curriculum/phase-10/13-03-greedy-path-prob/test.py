import unittest

from solution import greedy_path, greedy_path_prob

AUS = {
    (): {"Sydney": 0.5, "Canberra": 0.4, "I": 0.1},
    ("Sydney",): {".": 1.0},
    ("Canberra",): {".": 1.0},
    ("I",): {"don't": 0.9, "think": 0.1},
    ("I", "don't"): {"know": 1.0},
}

TRAP = {
    (): {"a": 0.6, "b": 0.4},
    ("a",): {"x": 0.5, "y": 0.5},
    ("b",): {"z": 1.0},
}


class TestGreedyPathProb(unittest.TestCase):
    def test_path_probability_is_the_product_along_the_prefixes(self):
        self.assertAlmostEqual(greedy_path_prob(AUS, ["Sydney", "."]), 0.5, places=9)
        self.assertAlmostEqual(greedy_path_prob(AUS, ["Canberra", "."]), 0.4, places=9)
        self.assertAlmostEqual(greedy_path_prob(AUS, ["I", "don't", "know"]), 0.09, places=9)
        self.assertAlmostEqual(greedy_path_prob(AUS, ["I", "think"]), 0.01, places=9)

    def test_prefix_of_a_path_and_empty_path(self):
        self.assertAlmostEqual(greedy_path_prob(AUS, ["I"]), 0.1, places=9)
        self.assertAlmostEqual(greedy_path_prob(AUS, []), 1.0, places=9)

    def test_missing_token_or_prefix_gives_zero(self):
        self.assertEqual(greedy_path_prob(AUS, ["Melbourne"]), 0.0)
        self.assertEqual(greedy_path_prob(AUS, ["Sydney", ".", "!"]), 0.0)
        self.assertEqual(greedy_path_prob(AUS, ["Sydney", "?"]), 0.0)

    def test_probability_is_conditional_not_marginal(self):
        # "know" has probability 1.0 in its own row; the path still pays for "I" and "don't".
        self.assertLess(greedy_path_prob(AUS, ["I", "don't", "know"]), 0.1)

    def test_greedy_picks_the_fluent_wrong_answer(self):
        path = greedy_path(AUS)
        self.assertEqual(path, ["Sydney", "."])
        self.assertGreater(greedy_path_prob(AUS, path), greedy_path_prob(AUS, ["Canberra", "."]))

    def test_greedy_is_not_the_most_probable_sequence(self):
        path = greedy_path(TRAP)
        self.assertEqual(path, ["a", "x"])
        self.assertAlmostEqual(greedy_path_prob(TRAP, path), 0.3, places=9)
        self.assertGreater(greedy_path_prob(TRAP, ["b", "z"]), greedy_path_prob(TRAP, path))

    def test_ties_go_to_the_first_listed_token(self):
        table = {(): {"p": 0.5, "q": 0.5}, ("p",): {"r": 0.3, "s": 0.7}}
        self.assertEqual(greedy_path(table), ["p", "s"])
        table = {(): {"q": 0.5, "p": 0.5}}
        self.assertEqual(greedy_path(table), ["q"])

    def test_greedy_stops_at_a_prefix_without_entry_and_validates_root(self):
        self.assertEqual(greedy_path({(): {"end": 1.0}}), ["end"])
        with self.assertRaises(ValueError):
            greedy_path({("a",): {"b": 1.0}})


if __name__ == "__main__":
    unittest.main(verbosity=2)
