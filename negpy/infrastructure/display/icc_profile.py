"""Low-level ICC profile parsing for matrix/TRC detection and primaries extraction.

Reads the tag table and specific tag payloads using pure struct parsing —
no dependency on PIL.ImageCms or lcms2 for the introspection itself.
"""

import struct
from typing import Optional

import numpy as np

_D65_XYZ = np.array([0.95047, 1.00000, 1.08883], dtype=np.float64)
_WHITEPOINT_TOLERANCE = 0.005


def _read_tag_table(data: bytes) -> dict[bytes, tuple[int, int]]:
    """Parse the ICC tag table into {signature: (offset, size)}."""
    if len(data) < 132:
        return {}
    tag_count = struct.unpack_from(">I", data, 128)[0]
    tags: dict[bytes, tuple[int, int]] = {}
    for i in range(tag_count):
        base = 132 + i * 12
        if base + 12 > len(data):
            break
        sig = data[base : base + 4]
        offset, size = struct.unpack_from(">II", data, base + 4)
        tags[sig] = (offset, size)
    return tags


def _read_xyz_tag(data: bytes, offset: int, size: int) -> Optional[np.ndarray]:
    """Read an XYZType tag (ICC spec §10.31) → (3,) float64 array."""
    if size < 20:
        return None
    x = struct.unpack_from(">i", data, offset + 8)[0] / 65536.0
    y = struct.unpack_from(">i", data, offset + 12)[0] / 65536.0
    z = struct.unpack_from(">i", data, offset + 16)[0] / 65536.0
    return np.array([x, y, z], dtype=np.float64)


def is_matrix_trc_profile(data: bytes) -> bool:
    """True when the profile is a matrix/TRC (shaper-matrix) type.

    Requires rXYZ/gXYZ/bXYZ colorant tags and at least one TRC tag,
    and must NOT have A2B0/B2A0 LUT tags.
    """
    tags = _read_tag_table(data)
    has_colorants = all(sig in tags for sig in (b"rXYZ", b"gXYZ", b"bXYZ"))
    has_trc = any(sig in tags for sig in (b"rTRC", b"gTRC", b"bTRC"))
    has_lut = any(sig in tags for sig in (b"A2B0", b"B2A0"))
    return has_colorants and has_trc and not has_lut


def extract_primaries_matrix(data: bytes) -> Optional[np.ndarray]:
    """Extract the 3x3 RGB→XYZ matrix from rXYZ/gXYZ/bXYZ colorant tags.

    Returns a (3, 3) float64 array where each column is one primary's
    XYZ tristimulus, or None if the tags are missing/malformed.
    """
    tags = _read_tag_table(data)
    cols = []
    for sig in (b"rXYZ", b"gXYZ", b"bXYZ"):
        entry = tags.get(sig)
        if entry is None:
            return None
        xyz = _read_xyz_tag(data, *entry)
        if xyz is None:
            return None
        cols.append(xyz)
    return np.column_stack(cols)


def extract_whitepoint(data: bytes) -> Optional[np.ndarray]:
    """Read the profile's media white point (wtpt tag) as (3,) float64."""
    tags = _read_tag_table(data)
    entry = tags.get(b"wtpt")
    if entry is None:
        return None
    return _read_xyz_tag(data, *entry)


def is_d65_whitepoint(data: bytes) -> bool:
    """True when the profile's declared white point matches D65."""
    wp = extract_whitepoint(data)
    if wp is None:
        return False
    return bool(np.all(np.abs(wp - _D65_XYZ) < _WHITEPOINT_TOLERANCE))
