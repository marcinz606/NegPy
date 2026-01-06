import hashlib
import os
from typing import Any
import numpy as np


def ensure_array(val: Any) -> np.ndarray:
    """
    Proves to the type checker that a value is a numpy array at runtime.
    """
    if not isinstance(val, np.ndarray):
        raise TypeError(f"Expected numpy.ndarray, got {type(val)}")
    return val


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


def get_luminance(img: np.ndarray) -> np.ndarray:
    """
    Calculates relative luminance using Rec. 709 coefficients.
    Supports both 3D (H, W, 3) and 2D (N, 3) arrays.
    """
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
