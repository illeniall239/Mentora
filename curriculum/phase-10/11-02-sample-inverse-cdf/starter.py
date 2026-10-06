import random


def sample(probs: list[float], rng: random.Random) -> int:
    """Inverse-CDF draw: one rng.random() call, first index whose cumulative probability exceeds u."""
    raise NotImplementedError
