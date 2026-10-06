# Reference solution: only used by scripts/verify-exercises.mjs. Shown in the Breakdown only after the Learner's own solution passes.
import random

IGNORE_INDEX = -100


def mask_tokens(
    ids: list[int],
    rate: float,
    mask_id: int,
    rng: random.Random,
    vocab_size: int = 1000,
    special_ids: tuple[int, ...] = (0,),
) -> tuple[list[int], list[int]]:
    if not 0.0 <= rate <= 1.0:
        raise ValueError("rate must be between 0.0 and 1.0")
    if vocab_size < 1:
        raise ValueError("vocab_size must be at least 1")
    if not 0 <= mask_id < vocab_size:
        raise ValueError("mask_id must be inside range(vocab_size)")
    if any(not 0 <= i < vocab_size for i in ids):
        raise ValueError("every id must be inside range(vocab_size)")

    special = set(special_ids)
    out = list(ids)
    labels = [IGNORE_INDEX] * len(ids)

    for i, token in enumerate(ids):
        if token in special or rng.random() >= rate:
            continue
        labels[i] = token
        roll = rng.random()
        if roll < 0.8:
            out[i] = mask_id
        elif roll < 0.9:
            out[i] = rng.randrange(vocab_size)
        # else: leave out[i] as the original token
    return out, labels
