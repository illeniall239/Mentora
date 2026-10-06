import random


class EpsilonGreedy:
    def __init__(self, n_arms: int, eps: float, rng: random.Random):
        """Running-mean estimates q (floats) and counts (ints) for n_arms arms; ValueError on bad inputs."""
        raise NotImplementedError

    def select(self) -> int:
        """rng.random() < eps -> rng.randrange(n_arms); else the arm with the highest q (lowest index on ties)."""
        raise NotImplementedError

    def update(self, arm: int, reward: float) -> None:
        """Increment counts[arm] and move q[arm] to the running mean by q += (reward - q) / counts."""
        raise NotImplementedError
