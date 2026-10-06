# Reference solution: only used by scripts/verify-exercises.mjs. Shown in the Breakdown only after the Learner's own solution passes.
def wer(ref: str, hyp: str) -> tuple[int, int, int, int, float]:
    r, h = ref.split(), hyp.split()
    n = len(r)
    if n == 0:
        raise ValueError("the reference must contain at least one word")

    # cell (i, j) = (cost, substitutions, deletions, insertions) for r[:i] vs h[:j]
    row = [(j, 0, 0, j) for j in range(len(h) + 1)]
    for i in range(1, n + 1):
        prev, row = row, [(i, 0, i, 0)] + [None] * len(h)
        for j in range(1, len(h) + 1):
            c, s, d, ins = prev[j - 1]
            same = r[i - 1] == h[j - 1]
            diag = (c + (0 if same else 1), s + (0 if same else 1), d, ins)
            c, s, d, ins = prev[j]
            delete = (c + 1, s, d + 1, ins)
            c, s, d, ins = row[j - 1]
            insert = (c + 1, s, d, ins + 1)
            row[j] = min((diag, delete, insert), key=lambda cell: cell[0])

    _, s, d, i = row[len(h)]
    return s, d, i, n, (s + d + i) / n
