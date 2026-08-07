"""Low-level ICC profile parsing for matrix/TRC detection and primaries extraction.

Reads the tag table and specific tag payloads using pure struct parsing —
no dependency on PIL.ImageCms or lcms2 for the introspection itself.
"""

import struct
from typing import Optional

import numpy as np

# ICC's PCS (profile connection space) is always D50-relative, so every conformant
# matrix/TRC profile's rXYZ/gXYZ/bXYZ colorant tags are D50-adapted regardless of the
# profile's actual native illuminant (D65 for sRGB/Adobe RGB, D60 for ACES, D50 for
# ProPhoto, ...). Recovering D65-referenced primaries — needed to combine with this
# codebase's D65-referenced working-space math — is therefore always a two-step
# chromatic adaptation: PCS(D50) -> native white -> D65. `chad`, when present, gives
# the exact native white (inverting it alone only reaches D65 by coincidence, for a
# D65-native profile); its absence means the native white is unknown and D50 is the
# only assumption available.
_D50_XYZ = np.array([0.9642, 1.0000, 0.8249], dtype=np.float64)
_D65_XYZ = np.array([0.95047, 1.00000, 1.08883], dtype=np.float64)

# Bradford cone-response matrix (Lindbloom), used to build a chromatic adaptation
# transform between any two reference whites.
_BRADFORD_CONE = np.array(
    [
        [0.8951000, 0.2664000, -0.1614000],
        [-0.7502000, 1.7135000, 0.0367000],
        [0.0389000, -0.0685000, 1.0296000],
    ],
    dtype=np.float64,
)


def _bradford_adaptation(src_white: np.ndarray, dst_white: np.ndarray) -> np.ndarray:
    """3x3 Bradford chromatic adaptation matrix mapping XYZ relative to ``src_white``
    to XYZ relative to ``dst_white``."""
    lms_src = _BRADFORD_CONE @ src_white
    lms_dst = _BRADFORD_CONE @ dst_white
    scale = np.diag(lms_dst / lms_src)
    return np.linalg.inv(_BRADFORD_CONE) @ scale @ _BRADFORD_CONE


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


def _read_chad_tag(data: bytes, offset: int, size: int) -> Optional[np.ndarray]:
    """Read a chromaticAdaptationTag (s15Fixed16ArrayType, ICC spec §10.8) → (3, 3) float64."""
    if size < 8 + 9 * 4:
        return None
    vals = struct.unpack_from(">9i", data, offset + 8)
    return np.array(vals, dtype=np.float64).reshape(3, 3) / 65536.0


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
    """Extract the 3x3 D65-referenced RGB→XYZ matrix from rXYZ/gXYZ/bXYZ colorant tags.

    The raw tag values are PCS-relative (D50-adapted, per the ICC spec). `chad`
    records the profile's native-white -> PCS(D50) adaptation actually used, so
    `inv(chad)` recovers the *native* reference — D65 only by coincidence, for a
    D65-native profile (sRGB, Adobe RGB, ...). For anything else (ACES/ACEScg: D60,
    ProPhoto: D50, ...) that native reference then needs its own adaptation to D65.
    Without `chad` (typical of v2 profiles) the native white is unknown, so D50 is
    assumed. Either way the result always lands on D65 — see
    `test_extracted_primaries_are_d65_referenced` for the profiles this matters for.

    Returns a (3, 3) float64 array where each column is one primary's XYZ
    tristimulus, or **None if the colorant tags are missing** (malformed size, or
    absent entirely — e.g. a GRAY-space profile, which has no rXYZ/gXYZ/bXYZ to
    extract). This is a load-bearing part of the contract, not incidental: callers
    (`ImageProcessor._try_matrix_bypass`) rely on `None` here to fall through to the
    full-CMS path rather than crash or fabricate a matrix — `is_matrix_trc_profile`
    is expected to have already filtered out non-RGB profiles first, but this
    function must stay safe to call regardless.
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
    m_pcs = np.column_stack(cols)

    m_native, native_white = m_pcs, _D50_XYZ
    chad_entry = tags.get(b"chad")
    if chad_entry is not None:
        chad = _read_chad_tag(data, *chad_entry)
        if chad is not None:
            try:
                chad_inv = np.linalg.inv(chad)
                m_native, native_white = chad_inv @ m_pcs, chad_inv @ _D50_XYZ
            except np.linalg.LinAlgError:
                pass
    return _bradford_adaptation(native_white, _D65_XYZ) @ m_native
