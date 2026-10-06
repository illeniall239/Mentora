import unittest

from solution import value_iteration


def corridor():
    """c0 -R-> c1 -R-> c2 -R-> goal, reward 1.0 only on entering goal; L steps back, L at c0 hits a wall."""
    cells = ["c0", "c1", "c2"]
    transitions = {}
    for i, s in enumerate(cells):
        transitions[(s, "L")] = [(1.0, cells[i - 1] if i > 0 else s, 0.0)]
        nxt = cells[i + 1] if i + 1 < len(cells) else "goal"
        transitions[(s, "R")] = [(1.0, nxt, 1.0 if nxt == "goal" else 0.0)]
    return {
        "states": cells + ["goal"],
        "terminal": ["goal"],
        "actions": {s: ["L", "R"] for s in cells},
        "transitions": transitions,
    }


TRAP = {  # the goal is terminal, yet the dict also offers it a paying self-loop
    "states": ["start", "goal"],
    "terminal": ["goal"],
    "actions": {"start": ["go"], "goal": ["stay"]},
    "transitions": {("start", "go"): [(1.0, "goal", 1.0)], ("goal", "stay"): [(1.0, "goal", 1.0)]},
}

FORK = {
    "states": ["fork", "win", "lose", "ok"],
    "terminal": ["win", "lose", "ok"],
    "actions": {"fork": ["safe", "risky"]},
    "transitions": {
        ("fork", "safe"): [(1.0, "ok", 1.0)],
        ("fork", "risky"): [(0.5, "win", 10.0), (0.5, "lose", -6.0)],
    },
}

CHAIN = {  # reward sits at the head of the list: a in-place sweep would carry it to b and c at once
    "states": ["a", "b", "c", "end"],
    "terminal": ["end"],
    "actions": {"a": ["go"], "b": ["go"], "c": ["go"]},
    "transitions": {
        ("a", "go"): [(1.0, "end", 1.0)],
        ("b", "go"): [(1.0, "a", 0.0)],
        ("c", "go"): [(1.0, "b", 0.0)],
    },
}


class TestValueIteration(unittest.TestCase):
    def assertValues(self, V, want):
        self.assertEqual(set(V), set(want))
        for s, v in want.items():
            self.assertAlmostEqual(V[s], v, places=6, msg=f"V[{s!r}]")

    def test_corridor_values_and_policy(self):
        V, policy = value_iteration(corridor(), 0.9, 1e-12)
        self.assertValues(V, {"c0": 0.81, "c1": 0.9, "c2": 1.0, "goal": 0.0})
        self.assertEqual(policy, {"c0": "R", "c1": "R", "c2": "R"})

    def test_discounting_actually_discounts(self):
        V, _ = value_iteration(corridor(), 0.5, 1e-12)
        self.assertValues(V, {"c0": 0.25, "c1": 0.5, "c2": 1.0, "goal": 0.0})
        V0, _ = value_iteration(corridor(), 0.0, 1e-12)
        self.assertValues(V0, {"c0": 0.0, "c1": 0.0, "c2": 1.0, "goal": 0.0})

    def test_terminal_states_are_never_backed_up(self):
        V, policy = value_iteration(TRAP, 0.9, 1e-12)
        self.assertValues(V, {"start": 1.0, "goal": 0.0})
        self.assertEqual(policy, {"start": "go"})  # no entry for the terminal state

    def test_stochastic_transitions_are_weighted_by_probability(self):
        V, policy = value_iteration(FORK, 0.9, 1e-12)
        self.assertAlmostEqual(V["fork"], 2.0, places=6)
        self.assertEqual(policy, {"fork": "risky"})

    def test_one_synchronous_sweep_only(self):
        V, _ = value_iteration(CHAIN, 1.0, 2.0)  # delta after sweep 1 is 1.0 < 2.0, so it stops there
        self.assertValues(V, {"a": 1.0, "b": 0.0, "c": 0.0, "end": 0.0})
        converged, policy = value_iteration(CHAIN, 1.0, 1e-12)
        self.assertValues(converged, {"a": 1.0, "b": 1.0, "c": 1.0, "end": 0.0})
        self.assertEqual(policy, {"a": "go", "b": "go", "c": "go"})

    def test_ties_go_to_the_first_action_listed(self):
        tie = {
            "states": ["s", "end"],
            "terminal": ["end"],
            "actions": {"s": ["west", "east"]},
            "transitions": {("s", "west"): [(1.0, "end", 3.0)], ("s", "east"): [(1.0, "end", 3.0)]},
        }
        _, policy = value_iteration(tie, 0.9, 1e-12)
        self.assertEqual(policy, {"s": "west"})
        tie["actions"]["s"] = ["east", "west"]
        _, policy = value_iteration(tie, 0.9, 1e-12)
        self.assertEqual(policy, {"s": "east"})

    def test_the_mdp_is_not_mutated(self):
        mdp = corridor()
        before = {k: repr(v) for k, v in mdp.items()}
        value_iteration(mdp, 0.9, 1e-12)
        self.assertEqual({k: repr(v) for k, v in mdp.items()}, before)

    def test_bad_arguments_raise(self):
        for gamma, tol in [(1.5, 1e-6), (-0.1, 1e-6), (0.9, 0.0), (0.9, -1.0)]:
            with self.assertRaises(ValueError, msg=f"gamma={gamma} tol={tol}"):
                value_iteration(corridor(), gamma, tol)


if __name__ == "__main__":
    unittest.main(verbosity=2)
