def train_bpe(text: str, vocab_size: int) -> tuple[dict[tuple[int, int], int], dict[int, bytes]]:
    """Train byte-level BPE; return (merges: pair -> new id in learned order, vocab: id -> bytes)."""
    raise NotImplementedError


def encode(text: str, merges: dict[tuple[int, int], int]) -> list[int]:
    """UTF-8 bytes of text with every merge applied in the order it was learned."""
    raise NotImplementedError


def decode(ids: list[int], vocab: dict[int, bytes]) -> str:
    """Concatenate the bytes of each id and decode as UTF-8 with errors="replace"."""
    raise NotImplementedError
