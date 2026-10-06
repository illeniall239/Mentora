# Reference solution: only used by scripts/verify-exercises.mjs. Shown in the Breakdown only after the Learner's own solution passes.
import random


class EpsilonGreedy:
    def __init__(self, n_arms: int, eps: float, rng: random.Random):
        if n_arms < 1 or not 0.0 <= eps <= 1.0:
            raise ValueError("need at least one arm and eps in [0, 1]")
        self.n_arms = n_arms
        self.eps = eps
        self.rng = rng
        self.q = [0.0] * n_arms
        self.counts = [0] * n_arms

    def select(self) -> int:
        if self.rng.random() < self.eps:
            return self.rng.randrange(self.n_arms)  # explore
        return max(range(self.n_arms), key=lambda a: (self.q[a], -a))  # exploit; ties -> lowest index

    def update(self, arm: int, reward: float) -> None:
        self.counts[arm] += 1
        self.q[arm] += (reward - self.q[arm]) / self.counts[arm]  # running mean without storing rewards
