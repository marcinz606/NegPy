import sys
from typing import Any

from numba import njit as _njit


def njit(*args: Any, **kwargs: Any) -> Any:
    """Keep frozen applications independent of Numba's disk-cache availability."""
    if getattr(sys, "frozen", False) and kwargs.get("cache"):
        kwargs["cache"] = False
    return _njit(*args, **kwargs)
