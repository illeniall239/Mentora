def dedup_exact(docs: list[str]) -> list[str]:
    """Drop byte-for-byte duplicate documents (compared by hash), keeping the first occurrence, in order."""
    raise NotImplementedError


def ngram_overlap(a: str, b: str, n: int) -> float:
    """Jaccard similarity of the word n-gram sets of a and b; 0.0 if either has fewer than n words."""
    raise NotImplementedError
