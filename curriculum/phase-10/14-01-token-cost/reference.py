# Reference solution: only used by scripts/verify-exercises.mjs. Shown in the Breakdown only after the Learner's own solution passes.
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


def token_cost(text: str, merges: dict[tuple[int, int], int]) -> int:
    ids = list(text.encode("utf-8"))
    for pair, new_id in merges.items():
        ids = _merge(ids, pair, new_id)
    return len(ids)
