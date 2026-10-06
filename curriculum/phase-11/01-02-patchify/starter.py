def patchify(img: list[list[list[float]]], p: int) -> list[list[float]]:
    """Cut an H×W×C nested-list image into N row-major patches, each flattened row, col, channel."""
    raise NotImplementedError


def unpatchify(patches: list[list[float]], h: int, w: int, p: int, c: int) -> list[list[list[float]]]:
    """Inverse of patchify: rebuild the H×W×C nested-list image."""
    raise NotImplementedError


def num_patch_tokens(h: int, w: int, p: int, cls: bool) -> int:
    """Number of tokens entering the encoder: (H/p)·(W/p), plus one for [CLS] if cls."""
    raise NotImplementedError
