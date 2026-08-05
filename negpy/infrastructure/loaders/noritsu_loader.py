import os
from typing import Any, ContextManager, Optional, Tuple

import numpy as np

from negpy.domain.interfaces import IImageLoader
from negpy.infrastructure.loaders.helpers import NonStandardFileWrapper
from negpy.infrastructure.loaders.pakon_loader import PakonLoader
from negpy.kernel.image.logic import uint16_to_float32

KNOWN_NORITSU_DIMS: list[tuple[int, int]] = [
    (3711, 5028),
    (4937, 5028),
    (6079, 5028),
    (7317, 5028),
    (5010, 5028),
    (4036, 5028),
    (5185, 5028),
    (5256, 5028),
    (9972, 5028),
    (10379, 5028),
    (3859, 5028),
    (7158, 4502),
    (3551, 4502),
    (12681, 4502),
    (11348, 4502),
    (4042, 6391),
]

KNOWN_NORITSU_HEIGHTS: list[int] = sorted({h for _, h in KNOWN_NORITSU_DIMS})


_MIN_SCAN_WIDTH = 2000
_MAX_SCAN_WIDTH = 15000
_MIN_ASPECT = 1.0
_MAX_ASPECT = 4.5


def detect_noritsu_dims(file_path: str) -> Optional[tuple[int, int]]:
    """Tiered dimension detection for a headerless Noritsu RAW file.

    Tier 1: exact match against KNOWN_NORITSU_DIMS.
    Tier 2: try each known height — if exactly one divides the file size
    evenly, use it to solve for width.
    Tier 3: open divisor search — enumerate all (w, h) where w*h*6 == size,
    both dimensions in a sane scanner range, and aspect ratio is film-plausible.
    Accept only if exactly one candidate survives.

    Returns (width, height) or None if ambiguous/unknown.
    """
    try:
        size = os.path.getsize(file_path)
    except OSError:
        return None
    for w, h in KNOWN_NORITSU_DIMS:
        if w * h * 6 == size:
            return (w, h)
    height_matches: list[tuple[int, int]] = []
    for h in KNOWN_NORITSU_HEIGHTS:
        stride = h * 6
        if size % stride == 0:
            height_matches.append((size // stride, h))
    if len(height_matches) == 1:
        return height_matches[0]

    total_pixels = size // 6
    if total_pixels * 6 != size:
        return None
    candidates: list[tuple[int, int]] = []
    for w in range(_MIN_SCAN_WIDTH, _MAX_SCAN_WIDTH + 1):
        if total_pixels % w == 0:
            h = total_pixels // w
            if h < _MIN_SCAN_WIDTH:
                continue
            aspect = max(w, h) / min(w, h)
            if _MIN_ASPECT <= aspect <= _MAX_ASPECT:
                candidates.append((w, h))
    if len(candidates) == 1:
        return candidates[0]
    return None


def is_noritsu_raw(file_path: str) -> bool:
    """True if this is a headerless Noritsu EZController RAW file.

    Detection: .raw extension + file size resolves to dimensions via the
    known-dims table or a known-height match + NOT a Pakon spec.
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext != ".raw":
        return False
    if PakonLoader.can_handle(file_path):
        return False
    return detect_noritsu_dims(file_path) is not None


class NoritsuLoader(IImageLoader):
    """Loader for headerless Noritsu EZController RAW files.

    Format: flat BGR16 little-endian, chunky, no header, no compression.
    12-bit sensor data in 16-bit container (max value 4095).
    Dimensions resolved from file size via tiered detection.
    """

    def load(self, file_path: str) -> Tuple[ContextManager[Any], dict]:
        dims = detect_noritsu_dims(file_path)
        if dims is None:
            raise ValueError(f"Unknown Noritsu dimensions for {file_path}")
        w, h = dims
        expected_pixels = w * h * 3

        with open(file_path, "rb") as f:
            data = np.fromfile(f, dtype="<u2", count=expected_pixels)

        if len(data) < expected_pixels:
            raise ValueError(f"File too small: expected {expected_pixels} samples, got {len(data)}")

        arr = data.reshape(h, w, 3)[:, :, ::-1]  # BGR → RGB
        f32 = uint16_to_float32(np.ascontiguousarray(arr))

        metadata = {"orientation": 0, "ir": None}
        return NonStandardFileWrapper(f32), metadata
