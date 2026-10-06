import random


def mask_tokens(
    ids: list[int],
    rate: float,
    mask_id: int,
    rng: random.Random,
    vocab_size: int = 1000,
    special_ids: tuple[int, ...] = (0,),
) -> tuple[list[int], list[int]]:
    """Corrupt ids for the masked-LM objective (80/10/10) and return (corrupted_ids, labels)."""
    raise NotImplementedError
