"""Positive decode of a TIFF by Source Encoding: the samples' tone curve to linear,
then the embedded profile's primaries to the working space (Adobe RGB 1998)."""

import struct
from dataclasses import dataclass
from typing import Optional

import negpy_core
import numpy as np

from negpy.domain.models import ColorSpace
from negpy.features.process.models import SourceEncoding
from negpy.infrastructure.loaders.helpers import identify_color_space_from_icc
from negpy.kernel.system.logging import get_logger

logger = get_logger(__name__)

# What ran, in the loader metadata, when Auto read the file's profile or color tag.
PROFILE = "profile"

# Adobe RGB (1998) colorants in the D50 PCS, as the bundled AdobeCompat-v4.icc stores them
# (tests/test_source_encoding.py reads them back from that file).
_WORKING_COLORANTS_D50 = np.array(
    [
        [0.60974, 0.20528, 0.14919],
        [0.31111, 0.62567, 0.06322],
        [0.01947, 0.06087, 0.74457],
    ],
    dtype=np.float64,
)

# sRGB colorants in the D50 PCS, as the bundled sRGB-v4.icc stores them.
_SRGB_COLORANTS_D50 = np.array(
    [
        [0.43604, 0.38510, 0.14307],
        [0.22244, 0.71693, 0.06062],
        [0.01390, 0.09708, 0.71393],
    ],
    dtype=np.float64,
)

# A source matrix this close to identity is the working space itself; skipping it keeps an
# Adobe RGB file's samples exact.
_IDENTITY_TOLERANCE = 2e-3


@dataclass(frozen=True)
class ProfileDecode:
    """A matrix/TRC profile: three tone curves sampled on [0, 1], and source-to-working
    primaries (None when the profile is the working space or has no colorants)."""

    curves: tuple[np.ndarray, ...]
    to_working: Optional[np.ndarray]
    # Linear samples already in the working primaries: a uint16 source passes through as is.
    identity: bool = False


def _s15f16(raw: bytes) -> float:
    return struct.unpack(">i", raw)[0] / 65536.0


def _tag_table(icc: bytes) -> dict[bytes, bytes]:
    count = struct.unpack(">I", icc[128:132])[0]
    tags: dict[bytes, bytes] = {}
    for i in range(count):
        sig, offset, size = struct.unpack(">4sII", icc[132 + 12 * i : 144 + 12 * i])
        tags[sig] = icc[offset : offset + size]
    return tags


def _curve(tag: bytes, x: np.ndarray) -> Optional[np.ndarray]:
    """Encoded [0, 1] -> linear through an ICC `curv` or `para` tag; None for another type."""
    kind = tag[:4]
    if kind == b"curv":
        n = struct.unpack(">I", tag[8:12])[0]
        if n == 0:
            return x.copy()
        if n == 1:
            return np.power(x, struct.unpack(">H", tag[12:14])[0] / 256.0)
        table = np.frombuffer(tag[12 : 12 + 2 * n], dtype=">u2").astype(np.float64) / 65535.0
        return np.interp(x, np.linspace(0.0, 1.0, n), table)
    if kind == b"para":
        fn = struct.unpack(">H", tag[8:10])[0]
        counts = {0: 1, 1: 3, 2: 4, 3: 5, 4: 7}
        if fn not in counts:
            return None
        p = [_s15f16(tag[12 + 4 * i : 16 + 4 * i]) for i in range(counts[fn])]
        g = p[0]
        if fn == 0:
            return np.power(x, g)
        a, b = p[1], p[2]
        lin = np.maximum(a * x + b, 0.0)
        if fn == 1:
            return np.where(x >= -b / a, np.power(lin, g), 0.0)
        if fn == 2:
            return np.where(x >= -b / a, np.power(lin, g) + p[3], p[3])
        c, d = p[3], p[4]
        if fn == 3:
            return np.where(x >= d, np.power(lin, g), c * x)
        return np.where(x >= d, np.power(lin, g) + p[5], c * x + p[6])
    return None


def parse_profile(icc: Optional[bytes], samples: int) -> Optional[ProfileDecode]:
    """The decode a matrix/TRC (or gray TRC) profile describes, sampled at *samples* points.
    None for a missing, unreadable or LUT-only profile."""
    if not icc or len(icc) < 132:
        return None
    try:
        tags = _tag_table(icc)
        x = np.linspace(0.0, 1.0, samples)
        if b"rTRC" in tags and b"gTRC" in tags and b"bTRC" in tags:
            curves = tuple(_curve(tags[sig], x) for sig in (b"rTRC", b"gTRC", b"bTRC"))
        elif b"kTRC" in tags:
            gray = _curve(tags[b"kTRC"], x)
            curves = (gray, gray, gray)
        else:
            return None
        if any(c is None for c in curves):
            return None
        to_working = None
        if all(sig in tags for sig in (b"rXYZ", b"gXYZ", b"bXYZ")):
            cols = [[_s15f16(tags[sig][8 + 4 * i : 12 + 4 * i]) for i in range(3)] for sig in (b"rXYZ", b"gXYZ", b"bXYZ")]
            to_working = _to_working(np.array(cols, dtype=np.float64).T)
        return ProfileDecode(tuple(np.clip(c, 0.0, 1.0).astype(np.float32) for c in curves if c is not None), to_working)
    except Exception as e:
        logger.warning(f"Could not read the embedded ICC profile's tone curves: {e}")
        return None


def _apply_curves(img: np.ndarray, curves: tuple[np.ndarray, ...]) -> np.ndarray:
    """Per-channel lookup: exact for integer samples, interpolated for float ones."""
    out = np.empty(img.shape[:2] + (3,), dtype=np.float32)
    if img.dtype in (np.uint8, np.uint16):
        for ch in range(3):
            out[..., ch] = curves[ch][img[..., ch]]
        return out
    grid = np.linspace(0.0, 1.0, curves[0].shape[0], dtype=np.float32)
    top = float(np.iinfo(img.dtype).max) if np.issubdtype(img.dtype, np.integer) else 1.0
    flat = np.clip(img.astype(np.float32) / np.float32(top), 0.0, 1.0)
    for ch in range(3):
        out[..., ch] = np.interp(flat[..., ch], grid, curves[ch])
    return out


def _to_working(source_d50: np.ndarray) -> Optional[np.ndarray]:
    """Source-to-working primaries from D50 colorants; None when they are the working space."""
    m = np.linalg.solve(_WORKING_COLORANTS_D50, source_d50)
    if not np.all(np.isfinite(m)) or np.max(np.abs(m - np.eye(3))) <= _IDENTITY_TOLERANCE:
        return None
    return m.astype(np.float32)


_BRADFORD = np.array([[0.8951, 0.2664, -0.1614], [-0.7502, 1.7135, 0.0367], [0.0389, -0.0685, 1.0296]])
_D50 = np.array([0.9642, 1.0, 0.8249])


def _power(gamma: float, samples: int, to_working: Optional[np.ndarray] = None) -> ProfileDecode:
    curve = np.power(np.linspace(0.0, 1.0, samples), gamma).astype(np.float32)
    return ProfileDecode((curve,) * 3, to_working, identity=gamma == 1.0 and to_working is None)


def _srgb_curve(samples: int) -> np.ndarray:
    x = np.linspace(0.0, 1.0, samples)
    return np.where(x <= 0.04045, x / 12.92, np.power((x + 0.055) / 1.055, 2.4)).astype(np.float32)


def _srgb(samples: int) -> ProfileDecode:
    return ProfileDecode((_srgb_curve(samples),) * 3, _to_working(_SRGB_COLORANTS_D50))


def is_device_profile(icc: Optional[bytes]) -> bool:
    """A scanner's own input profile (class scnr) describes the device, not how the file is
    encoded: SilverFast embeds its IT8 profile in linear raw output. Input ICC applies one."""
    return bool(icc) and len(icc) >= 16 and icc[12:16] == b"scnr"


def resolve_decode(encoding: str, icc: Optional[bytes], dtype: np.dtype) -> tuple[str, ProfileDecode]:
    """(what ran, the decode) for a Positive source. Auto reads the ICC profile, unless it is a
    device profile, and else takes the untagged rule;
    "srgb" names the 8-bit untagged decode."""
    samples = 256 if dtype == np.uint8 else 65536
    enc = SourceEncoding(encoding)
    if enc is SourceEncoding.LINEAR:
        return SourceEncoding.LINEAR.value, _power(1.0, samples)
    if enc is SourceEncoding.GAMMA_22:
        return SourceEncoding.GAMMA_22.value, _power(2.2, samples)
    if is_device_profile(icc):
        icc = None
    profile = parse_profile(icc, samples)
    if profile is not None:
        return PROFILE, profile
    # A LUT-only or malformed profile: its name, when it is one of the two this decode knows.
    named = identify_color_space_from_icc(icc)
    if named == ColorSpace.ADOBE_RGB.value:
        return PROFILE, _power(563.0 / 256.0, samples)
    if named == ColorSpace.SRGB.value:
        return PROFILE, _srgb(samples)
    if dtype == np.uint8:
        return "srgb", _srgb(samples)
    return SourceEncoding.LINEAR.value, _power(1.0, samples)


def _into_gamut(rgb: np.ndarray) -> np.ndarray:
    """Pull each out-of-range pixel toward its own luminance until it fits [0, 1], which keeps its
    hue where a per-channel clip would turn it. An in-range image passes through."""
    if rgb.min() >= 0.0 and rgb.max() <= 1.0:
        return rgb
    y = np.clip(rgb @ _WORKING_COLORANTS_D50[1].astype(np.float32), 0.0, 1.0)[..., None]
    lo = rgb.min(axis=-1, keepdims=True)
    hi = rgb.max(axis=-1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.minimum(np.where(lo < 0.0, y / (y - lo), 1.0), np.where(hi > 1.0, (1.0 - y) / (hi - y), 1.0))
    t = np.clip(np.nan_to_num(t, nan=0.0), 0.0, 1.0)
    return np.clip(y + t * (rgb - y), 0.0, 1.0)


def apply_decode(img: np.ndarray, decode: ProfileDecode) -> np.ndarray:
    """Linear working-space float32 RGB in [0, 1] from a Positive source's samples."""
    linear = _apply_curves(img, decode.curves)
    if decode.to_working is not None:
        linear = negpy_core.matmul_3x3(np.ascontiguousarray(linear), np.ascontiguousarray(decode.to_working))
        # The rest of the decode stores samples in [0, 1].
        linear = _into_gamut(linear)
    return np.ascontiguousarray(linear, dtype=np.float32)
