# Reference solution: only used by scripts/verify-exercises.mjs. Shown in the Breakdown only after the Learner's own solution passes.
def merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
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


def encode(text: str, merges: dict[tuple[int, int], int]) -> list[int]:
    rank = {pair: r for r, pair in enumerate(merges)}
    ids = list(text.encode("utf-8"))
    while len(ids) >= 2:
        pairs = set(zip(ids, ids[1:]))
        best = min(pairs, key=lambda p: rank.get(p, float("inf")))
        if best not in rank:
            break
        ids = merge(ids, best, merges[best])
    return ids
