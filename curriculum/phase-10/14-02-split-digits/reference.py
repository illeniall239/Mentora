# Reference solution: only used by scripts/verify-exercises.mjs. Shown in the Breakdown only after the Learner's own solution passes.
DIGITS = "0123456789"


def split_digits(text: str) -> list[str]:
    chunks: list[str] = []
    i = 0
    while i < len(text):
        j = i
        if text[i] in DIGITS:
            while j < len(text) and text[j] in DIGITS:
                j += 1
            run = text[i:j]
            head = len(run) % 3 or 3
            chunks.append(run[:head])
            chunks.extend(run[k:k + 3] for k in range(head, len(run), 3))
        else:
            while j < len(text) and text[j] not in DIGITS:
                j += 1
            chunks.append(text[i:j])
        i = j
    return chunks


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


def _token_count(text: str, merges: dict[tuple[int, int], int]) -> int:
    ids = list(text.encode("utf-8"))
    for pair, new_id in merges.items():
        ids = _merge(ids, pair, new_id)
    return len(ids)


def token_count_delta(text: str, merges: dict[tuple[int, int], int]) -> int:
    with_split = sum(_token_count(chunk, merges) for chunk in split_digits(text))
    return with_split - _token_count(text, merges)
