import numpy as np


def patch_embed(img: np.ndarray, p: int, W: np.ndarray, pos: np.ndarray, cls_vec: np.ndarray) -> np.ndarray:
    """Patchify an (H, W, C) image, project each patch with the shared W, prepend [CLS], add positions."""
    raise NotImplementedError
