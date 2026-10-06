# Reference solution: only used by scripts/verify-exercises.mjs. Shown in the Breakdown only after the Learner's own solution passes.
def _check(h: int, w: int, p: int) -> None:
    if p <= 0 or h % p or w % p:
        raise ValueError("image size must be a multiple of the patch size")


def patchify(img: list[list[list[float]]], p: int) -> list[list[float]]:
    h, w = len(img), len(img[0])
    _check(h, w, p)
    patches = []
    for pr in range(h // p):
        for pc in range(w // p):
            patches.append(
                [ch for r in range(pr * p, pr * p + p) for c in range(pc * p, pc * p + p) for ch in img[r][c]]
            )
    return patches


def unpatchify(patches: list[list[float]], h: int, w: int, p: int, c: int) -> list[list[list[float]]]:
    _check(h, w, p)
    img = [[[0.0] * c for _ in range(w)] for _ in range(h)]
    cols = w // p
    for n, patch in enumerate(patches):
        pr, pc = divmod(n, cols)
        k = 0
        for r in range(pr * p, pr * p + p):
            for col in range(pc * p, pc * p + p):
                img[r][col] = list(patch[k : k + c])
                k += c
    return img


def num_patch_tokens(h: int, w: int, p: int, cls: bool) -> int:
    _check(h, w, p)
    return (h // p) * (w // p) + (1 if cls else 0)
