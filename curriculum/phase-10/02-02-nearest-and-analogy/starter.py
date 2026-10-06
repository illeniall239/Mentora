import numpy as np


def nearest(query: np.ndarray, table: np.ndarray, k: int) -> list[int]:
    """Indices of the k rows of table most cosine-similar to query, most similar first."""
    raise NotImplementedError


def analogy(a: str, b: str, c: str, table: np.ndarray, vocab: list[str]) -> str:
    """Word nearest to table[b] - table[a] + table[c] by cosine, never a, b or c."""
    raise NotImplementedError
