import random
import unittest

from solution import EpsilonGreedy

MEANS = [0.1, 0.5, 0.9, 0.3]


def run(agent, means, steps, env_seed, noise=1.0):
    env = random.Random(env_seed)
    picks = [0] * len(means)
    for _ in range(steps):
        arm = agent.select()
        picks[arm] += 1
        agent.update(arm, means[arm] + env.gauss(0.0, noise))
    return picks


class TestEpsilonGreedy(unittest.TestCase):
    def test_initial_state_and_validation(self):
        agent = EpsilonGreedy(3, 0.1, random.Random(0))
        self.assertEqual(agent.q, [0.0, 0.0, 0.0])
        self.assertEqual(agent.counts, [0, 0, 0])
        with self.assertRaises(ValueError):
            EpsilonGreedy(0, 0.1, random.Random(0))
        with self.assertRaises(ValueError):
            EpsilonGreedy(3, 1.5, random.Random(0))

    def test_running_mean_by_hand(self):
        agent = EpsilonGreedy(3, 0.0, random.Random(0))
        agent.update(0, 1.0)
        self.assertEqual(agent.q, [1.0, 0.0, 0.0])
        agent.update(0, 0.0)
        self.assertAlmostEqual(agent.q[0], 0.5, places=9)
        self.assertEqual(agent.counts, [2, 0, 0])
        agent.update(0, 2.0)
        self.assertAlmostEqual(agent.q[0], 1.0, places=9)
        agent.update(2, 0.7)
        self.assertAlmostEqual(agent.q[2], 0.7, places=9)
        self.assertEqual(agent.counts, [3, 0, 1])

    def test_running_mean_matches_the_true_mean_over_many_rewards(self):
        rng = random.Random(5)
        rewards = [rng.uniform(-3, 3) for _ in range(500)]
        agent = EpsilonGreedy(2, 0.0, random.Random(0))
        for r in rewards:
            agent.update(1, r)
        self.assertAlmostEqual(agent.q[1], sum(rewards) / len(rewards), places=9)
        self.assertEqual(agent.counts, [0, 500])

    def test_greedy_select_picks_highest_q_lowest_index_on_ties(self):
        agent = EpsilonGreedy(3, 0.0, random.Random(0))
        self.assertEqual(agent.select(), 0)
        agent.update(2, 0.7)
        self.assertEqual(agent.select(), 2)
        agent.update(1, 0.7)
        self.assertEqual(agent.select(), 1)  # tie between 1 and 2 goes to 1
        agent.update(0, 5.0)
        self.assertEqual(agent.select(), 0)

    def test_exploration_uses_the_rng_as_specified(self):
        agent = EpsilonGreedy(4, 0.3, random.Random(42))
        mirror = random.Random(42)
        agent.update(3, 10.0)  # greedy arm is 3
        for _ in range(200):
            u = mirror.random()
            want = mirror.randrange(4) if u < 0.3 else 3
            self.assertEqual(agent.select(), want)

    def test_always_exploring_visits_every_arm(self):
        agent = EpsilonGreedy(4, 1.0, random.Random(0))
        agent.update(3, 100.0)
        picks = [0] * 4
        for _ in range(400):
            picks[agent.select()] += 1
        self.assertTrue(all(p > 50 for p in picks))

    def test_epsilon_greedy_finds_the_best_arm(self):
        for seed in (0, 1, 2):
            agent = EpsilonGreedy(4, 0.1, random.Random(seed))
            picks = run(agent, MEANS, 2000, env_seed=seed + 100)
            self.assertGreater(picks[2] / 2000, 0.75, msg=f"seed {seed}: {picks}")
            self.assertEqual(max(range(4), key=lambda a: agent.q[a]), 2)
            self.assertEqual(sum(agent.counts), 2000)

    def test_pure_greedy_gets_stuck_on_the_first_arm(self):
        agent = EpsilonGreedy(2, 0.0, random.Random(0))
        picks = run(agent, [0.2, 0.9], 300, env_seed=0, noise=0.0)
        self.assertEqual(picks, [300, 0])
        self.assertEqual(agent.counts[1], 0)
        explorer = EpsilonGreedy(2, 0.1, random.Random(0))
        picks = run(explorer, [0.2, 0.9], 300, env_seed=0, noise=0.0)
        self.assertGreater(picks[1], 200)


if __name__ == "__main__":
    unittest.main(verbosity=2)
