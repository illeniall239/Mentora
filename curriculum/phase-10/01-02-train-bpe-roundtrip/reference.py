# Reference solution: only used by scripts/verify-exercises.mjs. Shown in the Breakdown only after the Learner's own solution passes.
def _pair_counts(ids: list[int]) -> dict[tuple[int, int], int]:
    counts: dict[tuple[int, int], int] = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def _merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    out: list[int] = []
    i = 0
    while i < len(ids):
        if i + 1 < len(ids) and (ids[i], ids[i + 1]) == pair:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


def train_bpe(text: str, vocab_size: int) -> tuple[dict[tuple[int, int], int], dict[int, bytes]]:
    ids = list(text.encode("utf-8"))
    merges: dict[tuple[int, int], int] = {}
    vocab = {i: bytes([i]) for i in range(256)}
    for new_id in range(256, vocab_size):
        counts = _pair_counts(ids)
        if not counts:
            break
        pair = max(counts, key=counts.get)
        if counts[pair] < 2:
            break
        ids = _merge(ids, pair, new_id)
        merges[pair] = new_id
        vocab[new_id] = vocab[pair[0]] + vocab[pair[1]]
    return merges, vocab


def encode(text: str, merges: dict[tuple[int, int], int]) -> list[int]:
    ids = list(text.encode("utf-8"))
    for pair, new_id in merges.items():
        ids = _merge(ids, pair, new_id)
    return ids


def decode(ids: list[int], vocab: dict[int, bytes]) -> str:
    return b"".join(vocab[i] for i in ids).decode("utf-8", errors="replace")
