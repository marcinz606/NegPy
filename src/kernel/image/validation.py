from typing import Any, TypeVar, cast
import numpy as np
from src.domain.types import ImageBuffer

T = TypeVar("T")


def ensure_image(arr: Any) -> ImageBuffer:
    """
    Casts to float32 ndarray.
    """
    if not isinstance(arr, np.ndarray):
        raise TypeError(f"Expected numpy.ndarray, got {type(arr)}")

    if arr.dtype != np.float32:
        arr = arr.astype(np.float32)

    return cast(ImageBuffer, arr)
