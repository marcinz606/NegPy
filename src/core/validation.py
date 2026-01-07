from typing import Any, TypeVar, cast
import numpy as np
from src.core.types import ImageBuffer

T = TypeVar("T")


def ensure_image(arr: Any) -> ImageBuffer:
    """
    Ensures the input is a float32 numpy array and returns it as an ImageBuffer.
    """
    if not isinstance(arr, np.ndarray):
        raise TypeError(f"Expected numpy.ndarray, got {type(arr)}")

    if arr.dtype != np.float32:
        arr = arr.astype(np.float32)

    return cast(ImageBuffer, arr)


def validate_float(val: Any, default: float = 0.0) -> float:
    """Ensures a value is a float, providing a default if None."""
    if val is None:
        return default
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def validate_int(val: Any, default: int = 0) -> int:
    """Ensures a value is an int, providing a default if None."""
    if val is None:
        return default
    try:
        return int(val)
    except (TypeError, ValueError):
        return default


def validate_bool(val: Any, default: bool = False) -> bool:
    """Ensures a value is a bool."""
    if val is None:
        return default
    return bool(val)
