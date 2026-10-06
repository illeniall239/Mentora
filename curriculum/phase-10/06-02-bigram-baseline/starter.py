def train_bigram(ids: list[int], vocab: int) -> list[list[float]]:
    """vocab x vocab table of P[a][b] = (count(a, b) + 1) / (count(a, .) + vocab)."""
    raise NotImplementedError


def perplexity(model: list[list[float]], ids: list[int]) -> float:
    """exp of the mean of -ln model[ids[t-1]][ids[t]] over t = 1 .. len(ids)-1."""
    raise NotImplementedError
