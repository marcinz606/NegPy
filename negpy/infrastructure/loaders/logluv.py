# LogLuv32/24 decoder: SGI LogLuv (Greg Ward) to CIE XYZ to linear sRGB.
#
# Ported from flexcolor-tool (MIT License):
#   Copyright (c) 2026 flexcolor-tool contributors
#   https://github.com/rohanpandula/flexcolor-tool
#
# The decode math mirrors libtiff's tif_luv.c reference implementation. The constants
# (UVSCALE, U_NEU, V_NEU, uv_row table) are from the published LogLuv spec and libtiff's
# uvcode.h (Greg Ward, v1.0).

from typing import Tuple

import numpy as np

_M_LN2 = 0.69314718055994530942
_UVSCALE = 410.0
_U_NEU = 0.210526316
_V_NEU = 0.473684211

COMPRESSION_SGILOG = 34676
COMPRESSION_SGILOG24 = 34677


def _logl16_to_y(ptop: np.ndarray) -> np.ndarray:
    """LogL16 top-16 bits → luminance Y. Matches libtiff LogL16toY."""
    top = ptop.astype(np.int32)
    le = top & 0x7FFF
    sign = (top & 0x8000) != 0
    y = np.exp(_M_LN2 / 256.0 * (le.astype(np.float64) + 0.5) - _M_LN2 * 64.0)
    y = np.where(sign, -y, y)
    y = np.where(le != 0, y, 0.0)
    return y


def logluv32_to_xyz(packed: np.ndarray) -> np.ndarray:
    """LogLuv32 uint32 → XYZ float64 (..., 3)."""
    p = np.asarray(packed, dtype=np.uint32)
    out = np.empty(p.shape + (3,), dtype=np.float64)
    top = p.view(np.int32) >> 16
    L = _logl16_to_y(top)
    u = ((p >> 8) & 0xFF).astype(np.float64) + 0.5
    v = (p & 0xFF).astype(np.float64) + 0.5
    u = u / _UVSCALE
    v = v / _UVSCALE
    s = 1.0 / (6.0 * u - 16.0 * v + 12.0)
    x = 9.0 * u * s
    yv = 4.0 * v * s
    out[..., 0] = x / yv * L
    out[..., 1] = L
    out[..., 2] = (1.0 - x - yv) / yv * L
    ok = L > 0.0
    out[~ok] = 0.0
    return out


def decode_strip_logluv32(compressed: bytes, npixels: int) -> np.ndarray:
    """Decode a LogLuv32 RLE strip → npixels uint32 values.

    Four byte-planes (shifts 24,16,8,0) are run-length coded into one
    stream; runs ≥128 are count+value, else literals.
    """
    buf = np.frombuffer(compressed, dtype=np.uint8)
    n = buf.shape[0]
    pixels = np.zeros(npixels, dtype=np.uint32)
    pos = 0
    for shft in (24, 16, 8, 0):
        i = 0
        while i < npixels and pos < n:
            cnt = buf[pos]
            if cnt >= 128:
                if pos + 1 >= n:
                    break
                rc = int(cnt) - 126
                val = np.uint32(buf[pos + 1]) << shft
                pos += 2
                seg = min(rc, npixels - i)
                pixels[i : i + seg] |= val
                i += seg
            else:
                pos += 1
                seg = min(int(cnt), npixels - i, n - pos)
                pixels[i : i + seg] |= buf[pos : pos + seg].astype(np.uint32) << shft
                pos += seg
                i += seg
    return pixels


# LogLuv24: 10-bit log luminance + 14-bit uv index (libtiff uvcode.h table)
_UV_SQSIZ = 0.003500
_UV_VSTART = 0.016940
_UV_NVS = 163
_UV_NDIVS = 16289

_uv_row = [
    (0.247663, 4, 0),
    (0.243779, 6, 4),
    (0.241684, 7, 10),
    (0.237874, 9, 17),
    (0.235906, 10, 26),
    (0.232153, 12, 36),
    (0.228352, 14, 48),
    (0.226259, 15, 62),
    (0.222371, 17, 77),
    (0.220410, 18, 94),
    (0.214710, 21, 112),
    (0.212714, 22, 133),
    (0.210721, 23, 155),
    (0.204976, 26, 178),
    (0.202986, 27, 204),
    (0.199245, 29, 231),
    (0.195525, 31, 260),
    (0.193560, 32, 291),
    (0.189878, 34, 323),
    (0.186216, 36, 357),
    (0.186216, 36, 393),
    (0.182592, 38, 429),
    (0.179003, 40, 467),
    (0.175466, 42, 507),
    (0.172001, 44, 549),
    (0.172001, 44, 593),
    (0.168612, 46, 637),
    (0.168612, 46, 683),
    (0.163575, 49, 729),
    (0.158642, 52, 778),
    (0.158642, 52, 830),
    (0.158642, 52, 882),
    (0.153815, 55, 934),
    (0.153815, 55, 989),
    (0.149097, 58, 1044),
    (0.149097, 58, 1102),
    (0.142746, 62, 1160),
    (0.142746, 62, 1222),
    (0.142746, 62, 1284),
    (0.138270, 65, 1346),
    (0.138270, 65, 1411),
    (0.138270, 65, 1476),
    (0.132166, 69, 1541),
    (0.132166, 69, 1610),
    (0.126204, 73, 1679),
    (0.126204, 73, 1752),
    (0.126204, 73, 1825),
    (0.120381, 77, 1898),
    (0.120381, 77, 1975),
    (0.120381, 77, 2052),
    (0.120381, 77, 2129),
    (0.112962, 82, 2206),
    (0.112962, 82, 2288),
    (0.112962, 82, 2370),
    (0.107450, 86, 2452),
    (0.107450, 86, 2538),
    (0.107450, 86, 2624),
    (0.107450, 86, 2710),
    (0.100343, 91, 2796),
    (0.100343, 91, 2887),
    (0.100343, 91, 2978),
    (0.095126, 95, 3069),
    (0.095126, 95, 3164),
    (0.095126, 95, 3259),
    (0.095126, 95, 3354),
    (0.088276, 100, 3449),
    (0.088276, 100, 3549),
    (0.088276, 100, 3649),
    (0.088276, 100, 3749),
    (0.081523, 105, 3849),
    (0.081523, 105, 3954),
    (0.081523, 105, 4059),
    (0.081523, 105, 4164),
    (0.074861, 110, 4269),
    (0.074861, 110, 4379),
    (0.074861, 110, 4489),
    (0.074861, 110, 4599),
    (0.068290, 115, 4709),
    (0.068290, 115, 4824),
    (0.068290, 115, 4939),
    (0.068290, 115, 5054),
    (0.063573, 119, 5169),
    (0.063573, 119, 5288),
    (0.063573, 119, 5407),
    (0.063573, 119, 5526),
    (0.057219, 124, 5645),
    (0.057219, 124, 5769),
    (0.057219, 124, 5893),
    (0.057219, 124, 6017),
    (0.050985, 129, 6141),
    (0.050985, 129, 6270),
    (0.050985, 129, 6399),
    (0.050985, 129, 6528),
    (0.050985, 129, 6657),
    (0.044859, 134, 6786),
    (0.044859, 134, 6920),
    (0.044859, 134, 7054),
    (0.044859, 134, 7188),
    (0.040571, 138, 7322),
    (0.040571, 138, 7460),
    (0.040571, 138, 7598),
    (0.040571, 138, 7736),
    (0.036339, 142, 7874),
    (0.036339, 142, 8016),
    (0.036339, 142, 8158),
    (0.036339, 142, 8300),
    (0.032139, 146, 8442),
    (0.032139, 146, 8588),
    (0.032139, 146, 8734),
    (0.032139, 146, 8880),
    (0.027947, 150, 9026),
    (0.027947, 150, 9176),
    (0.027947, 150, 9326),
    (0.023739, 154, 9476),
    (0.023739, 154, 9630),
    (0.023739, 154, 9784),
    (0.023739, 154, 9938),
    (0.019504, 158, 10092),
    (0.019504, 158, 10250),
    (0.019504, 158, 10408),
    (0.016976, 161, 10566),
    (0.016976, 161, 10727),
    (0.016976, 161, 10888),
    (0.016976, 161, 11049),
    (0.012639, 165, 11210),
    (0.012639, 165, 11375),
    (0.012639, 165, 11540),
    (0.009991, 168, 11705),
    (0.009991, 168, 11873),
    (0.009991, 168, 12041),
    (0.009016, 170, 12209),
    (0.009016, 170, 12379),
    (0.009016, 170, 12549),
    (0.006217, 173, 12719),
    (0.006217, 173, 12892),
    (0.005097, 175, 13065),
    (0.005097, 175, 13240),
    (0.005097, 175, 13415),
    (0.003909, 177, 13590),
    (0.003909, 177, 13767),
    (0.002340, 177, 13944),
    (0.002389, 170, 14121),
    (0.001068, 164, 14291),
    (0.001653, 157, 14455),
    (0.000717, 150, 14612),
    (0.001614, 143, 14762),
    (0.000270, 136, 14905),
    (0.000484, 129, 15041),
    (0.001103, 123, 15170),
    (0.001242, 115, 15293),
    (0.001188, 109, 15408),
    (0.001011, 103, 15517),
    (0.000709, 97, 15620),
    (0.000301, 89, 15717),
    (0.002416, 82, 15806),
    (0.003251, 76, 15888),
    (0.003246, 69, 15964),
    (0.004141, 62, 16033),
    (0.005963, 55, 16095),
    (0.008839, 47, 16150),
    (0.010490, 40, 16197),
    (0.016994, 31, 16237),
    (0.023659, 21, 16268),
]
_u_start = np.array([r[0] for r in _uv_row], dtype=np.float64)
_u_nus = np.array([r[1] for r in _uv_row], dtype=np.int64)
_u_ncum = np.array([r[2] for r in _uv_row], dtype=np.int64)


def _logl10_to_y(p10: np.ndarray) -> np.ndarray:
    """LogL10 10-bit → luminance Y. Matches libtiff LogL10toY."""
    p10 = np.asarray(p10, dtype=np.int64)
    y = np.exp(_M_LN2 / 64.0 * (p10 + 0.5) - _M_LN2 * 12.0)
    return np.where(p10 != 0, y, 0.0)


def _uv_decode_index(ce: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """UV index (0..16288) → (u', v'). Vectorized libtiff uv_decode."""
    ce = np.asarray(ce, dtype=np.int64)
    vi = np.searchsorted(_u_ncum, ce, side="right") - 1
    vi = np.clip(vi, 0, _UV_NVS - 1)
    ui = ce - _u_ncum[vi]
    u = _u_start[vi] + (ui + 0.5) * _UV_SQSIZ
    v = _UV_VSTART + (vi + 0.5) * _UV_SQSIZ
    bad = (ce < 0) | (ce >= _UV_NDIVS)
    u = np.where(bad, _U_NEU, u)
    v = np.where(bad, _V_NEU, v)
    return u, v


def logluv24_to_xyz(packed: np.ndarray) -> np.ndarray:
    """LogLuv24 uint32 (24-bit in low bits) → XYZ float64 (..., 3)."""
    p = np.asarray(packed, dtype=np.uint32)
    le10 = (p >> 14) & 0x3FF
    ce = p & 0x3FFF
    L = _logl10_to_y(le10)
    u, v = _uv_decode_index(ce)
    s = 1.0 / (6.0 * u - 16.0 * v + 12.0)
    x = 9.0 * u * s
    yv = 4.0 * v * s
    out = np.empty(p.shape + (3,), dtype=np.float64)
    out[..., 0] = x / yv * L
    out[..., 1] = L
    out[..., 2] = (1.0 - x - yv) / yv * L
    ok = L > 0.0
    out[~ok] = 0.0
    return out


# XYZ to linear sRGB (Rec.709/sRGB primaries, D65): libtiff's XYZtoRGB24 matrix
_XYZ_TO_LRGB = np.array(
    [[2.690, -1.276, -0.414], [-1.022, 1.978, 0.044], [0.061, -0.224, 1.163]],
    dtype=np.float64,
)


def normalize_linear(lin: np.ndarray) -> np.ndarray:
    """Per-channel percentile black/white normalization.

    Maps raw HDR linear RGB into [0, 1] using the 0.2th/99.8th percentile
    per channel, correcting the per-channel black/gain offset inherent in
    Flextight CCD data (worst on red — weak CCD response, shallow C-41 cyan).
    """
    flat = lin.reshape(-1, lin.shape[-1])
    lo = np.percentile(flat, 0.2, axis=0)
    hi = np.percentile(flat, 99.8, axis=0)
    denom = hi - lo
    denom = np.where(denom > 0, denom, 1.0)
    return np.clip((lin - lo) / denom, 0.0, 1.0)


def xyz_to_linear_rgb(xyz: np.ndarray) -> np.ndarray:
    """CIE XYZ (..., 3) → linear sRGB float64 (..., 3), unbounded."""
    xyz = np.asarray(xyz, dtype=np.float64)
    v = np.moveaxis(xyz, -1, 0).reshape(3, -1)
    rgb = _XYZ_TO_LRGB @ v
    return np.moveaxis(rgb.reshape(3, *xyz.shape[:-1]), 0, -1)


def decode_logluv_strips(
    pages: list,
    width: int,
    height: int,
    data: bytes,
    byte_order: str,
) -> np.ndarray:
    """Decode LogLuv32/24 strips from raw TIFF pages → float32 RGB (H, W, 3).

    Reads strip offsets/byte counts from the TIFF page tags, decompresses,
    converts LogLuv → XYZ → linear sRGB, and normalizes to [0, 1].

    Parameters
    ----------
    pages : list of tifffile pages with LogLuv compression
    width, height : image dimensions
    data : raw file bytes
    byte_order : '<' for little-endian, '>' for big-endian

    Returns float32 array in [0, 1], clipped.
    """
    page = pages[0]
    tags = page.tags
    comp_val = int(tags["Compression"].value)

    if comp_val == COMPRESSION_SGILOG:
        px_to_xyz = logluv32_to_xyz
        bpp = 4
        use_rle = True
    elif comp_val == COMPRESSION_SGILOG24:
        px_to_xyz = logluv24_to_xyz
        bpp = 3
        use_rle = False
    else:
        raise ValueError(f"Not a LogLuv compression: {comp_val}")

    rps_tag = tags.get("RowsPerStrip")
    rps = int(rps_tag.value) if rps_tag else height

    offsets = _tag_to_list(tags.get("StripOffsets"))
    counts = _tag_to_list(tags.get("StripByteCounts"))
    if not offsets:
        raise ValueError("No StripOffsets in LogLuv IFD")

    npixels_total = width * height
    packed = np.empty(npixels_total, dtype=np.uint32)
    filled = 0
    n_strips = len(offsets)

    for si, (off, cnt) in enumerate(zip(offsets, counts)):
        rows = rps if si < n_strips - 1 else (height - (n_strips - 1) * rps)
        npix = rows * width
        chunk = data[off : off + cnt]
        if len(chunk) < cnt:
            raise ValueError(f"Strip {si} underrun (need {cnt}, got {len(chunk)})")

        if use_rle:
            packed[filled : filled + npix] = decode_strip_logluv32(chunk, npix)
        else:
            b = np.frombuffer(chunk, dtype=np.uint8, count=npix * bpp)
            p = b.reshape(npix, 3).astype(np.uint32)
            packed[filled : filled + npix] = (p[:, 0] << 16) | (p[:, 1] << 8) | p[:, 2]
        filled += npix

    packed = packed[:npixels_total].reshape(height, width)
    xyz = px_to_xyz(packed)
    lin = xyz_to_linear_rgb(xyz)
    normed = normalize_linear(lin)
    return normed.astype(np.float32)


def _tag_to_list(tag) -> list:
    if tag is None:
        return []
    v = tag.value
    if hasattr(v, "__len__") and not isinstance(v, (str, bytes)):
        return [int(x) for x in v]
    return [int(v)]
