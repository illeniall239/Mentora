def normalize_answer(s: str) -> str:
    """Lowercase, delete punctuation, drop a/an/the, collapse whitespace."""
    raise NotImplementedError


def exact_match(pred: str, gold: str) -> bool:
    """True when the two answers are identical after normalization."""
    raise NotImplementedError


def token_f1(pred: str, gold: str) -> float:
    """SQuAD token F1 over the multiset overlap of the normalized token lists."""
    raise NotImplementedError
