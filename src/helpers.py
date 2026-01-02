import hashlib
import io
import os
import numpy as np
from typing import Any, Tuple, cast
from src.backend.raw_handlers import load_special_raw


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
        return cast(np.ndarray, np.stack([img] * 3, axis=-1))
    if img.ndim == 3 and img.shape[2] == 1:
        return cast(np.ndarray, np.concatenate([img] * 3, axis=-1))
    return img


def get_luminance(img: np.ndarray) -> np.ndarray:
    """
    Calculates relative luminance using Rec. 709 coefficients.
    Supports both 3D (H, W, 3) and 2D (N, 3) arrays.
    """
    return cast(
        np.ndarray, 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]
    )


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
        # Fallback to a UUID if hashing fails, though it shouldn't
        import uuid

        return f"err_{uuid.uuid4()}"


def imread_raw(file_path: str) -> Any:
    """
    Safely opens a RAW file from disk into a BytesIO buffer to bypass
    LibRaw's path-length or file-size processing limits.
    Falls back to specialized loaders (e.g., Pakon) if standard RAW parsing fails.
    Should be used as a context manager: with imread_raw(path) as raw: ...
    """
    import rawpy

    try:
        with open(file_path, "rb") as f:
            return rawpy.imread(io.BytesIO(f.read()))
    except Exception:
        # Fallback for headerless/custom formats via the registry
        special_raw = load_special_raw(file_path)
        if special_raw:
            return special_raw
        raise


def transform_point(
    x: float,
    y: float,
    params: Any,  # Use Any to avoid circular import with ProcessingParams
    raw_w: int,
    raw_h: int,
    inverse: bool = False,
) -> Tuple[float, float]:
    """
    Transforms a normalized (0..1) point between Raw Space and Display Space.
    inverse=True: Display -> Raw (for saving clicks)
    inverse=False: Raw -> Display (for visualization)
    """
    rotation = params.get("rotation", 0) % 4

    if not inverse:
        # Raw -> Display (Forward rotation)
        if rotation == 0:
            return x, y
        if rotation == 1:
            return 1.0 - y, x
        if rotation == 2:
            return 1.0 - x, 1.0 - y
        if rotation == 3:
            return y, 1.0 - x
    else:
        # Display -> Raw (Inverse rotation)
        if rotation == 0:
            return x, y
        if rotation == 1:
            return y, 1.0 - x
        if rotation == 2:
            return 1.0 - x, 1.0 - y
        if rotation == 3:
            return 1.0 - y, x

    return x, y
