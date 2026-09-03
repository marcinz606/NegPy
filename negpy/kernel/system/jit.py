import sys
from typing import Any

from numba import njit as _njit


def njit(*args: Any, **kwargs: Any) -> Any:
    """Compile with disk caching only when Python source files are available."""
    if getattr(sys, "frozen", False) and kwargs.get("cache"):
        kwargs["cache"] = False
    return _njit(*args, **kwargs)
