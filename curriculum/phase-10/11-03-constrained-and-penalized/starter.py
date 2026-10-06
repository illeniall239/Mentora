import math


def softmax(xs: list[float]) -> list[float]:
    """Max-subtracted softmax; -inf entries get probability 0."""
    raise NotImplementedError


def constrained_mask(logits: list[float], allowed_ids: set[int]) -> list[float]:
    """Copy of logits with -inf everywhere except allowed_ids; ValueError if empty or out of range."""
    raise NotImplementedError


def repetition_penalty(logits: list[float], history: list[int], penalty: float) -> list[float]:
    """Divide positive logits of history tokens by penalty, multiply non-positive ones; ValueError if penalty <= 0."""
    raise NotImplementedError
