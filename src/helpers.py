import hashlib
import os
import numpy as np
from numba import njit, prange  # type: ignore
from src.perf_utils import time_function
from src.core.validation import ensure_image


@njit(parallel=True)
def _get_luminance_jit(img: np.ndarray) -> np.ndarray:
    """
    Fast JIT luminance calculation.
    """
    h, w, _ = img.shape
    res = np.empty((h, w), dtype=np.float32)
    for y in prange(h):
        for x in range(w):
            res[y, x] = (
                0.2126 * img[y, x, 0] + 0.7152 * img[y, x, 1] + 0.0722 * img[y, x, 2]
            )
    return res


def ensure_rgb(img: np.ndarray) -> np.ndarray:
    """
    Ensures the input image is a 3-channel RGB array.
    """
    if img.ndim == 2:
        res_2d: np.ndarray = np.stack([img] * 3, axis=-1)
        return res_2d
    if img.ndim == 3 and img.shape[2] == 1:
        res_1ch: np.ndarray = np.concatenate([img] * 3, axis=-1)
        return res_1ch
    return img


@time_function
def get_luminance(img: np.ndarray) -> np.ndarray:
    """
    Calculates relative luminance using Rec. 709 coefficients.
    Supports both 3D (H, W, 3) and 2D (N, 3) arrays.
    """
    if img.ndim == 3:
        return ensure_image(_get_luminance_jit(img.astype(np.float32)))

    # Fallback for 2D (flattened) arrays
    res: np.ndarray = 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]
    return res


def calculate_file_hash(file_path: str) -> str:
    """
    Generates a fast fingerprint of a RAW file.
    Hashes the first 1MB, last 1MB, and total file size.
    """
    try:
        file_size = os.path.getsize(file_path)
        hasher = hashlib.sha256()
        hasher.update(str(file_size).encode())

        with open(file_path, "rb") as f:
            # Hash first 1MB
            hasher.update(f.read(1024 * 1024))

            # Hash last 1MB (if file is large enough)
            if file_size > 2 * 1024 * 1024:
                f.seek(-1024 * 1024, os.SEEK_END)
                hasher.update(f.read(1024 * 1024))

        return hasher.hexdigest()
    except Exception:
        import uuid

        return f"err_{uuid.uuid4()}"
