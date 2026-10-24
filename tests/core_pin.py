import hashlib
import platform
import sys

import numpy as np
import pytest

# Pins hold the bytes recorded on Linux x86-64. Some pinned paths run NumPy steps whose
# transcendentals differ by CPU; test_core_same_bits holds the cases every OS must match.
linux_pin = pytest.mark.skipif(
    not (sys.platform == "linux" and platform.machine() == "x86_64"),
    reason="pins are recorded on Linux x86-64",
)


def wave(t):
    """A smooth periodic pattern in -1..1, period 2, from exact arithmetic. Pin inputs avoid
    np.sin and the like: NumPy computes them through SVML on some CPUs and libm on others."""
    tri = np.abs(np.mod(t, 2.0) - 1.0)
    return tri * tri * (3.0 - 2.0 * tri) * 2.0 - 1.0


def digest(*parts) -> str:
    """SHA-256 of the bytes of each part in order: arrays as C-contiguous memory, numbers as
    float64, so a pinned kernel output changes its digest when a single bit moves."""
    h = hashlib.sha256()
    for part in parts:
        if isinstance(part, (list, tuple)):
            h.update(digest(*part).encode())
        elif part is None:
            h.update(b"None")
        else:
            h.update(np.ascontiguousarray(np.asarray(part, dtype=None if isinstance(part, np.ndarray) else np.float64)).tobytes())
    return h.hexdigest()
