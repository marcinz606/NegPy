"""The LogLuv decode kernels against the NumPy code they replaced, and a pin."""

import numpy as np
import pytest
from core_pin import digest, linux_pin

from negpy.infrastructure.loaders import logluv as ll


def _decode_numpy(compressed, npixels, width):
    buf = np.frombuffer(compressed, dtype=np.uint8)
    n = buf.shape[0]
    pixels = np.zeros(npixels, dtype=np.uint32)
    pos = 0
    for start in range(0, npixels, width):
        row = pixels[start : start + width]
        for shft in (24, 16, 8, 0):
            i = 0
            while i < row.shape[0] and pos < n:
                cnt = buf[pos]
                if cnt >= 128:
                    if pos + 1 >= n:
                        break
                    seg = min(int(cnt) - 126, row.shape[0] - i)
                    row[i : i + seg] |= np.uint32(buf[pos + 1]) << shft
                    pos += 2
                    i += seg
                else:
                    pos += 1
                    seg = min(int(cnt), row.shape[0] - i, n - pos)
                    row[i : i + seg] |= buf[pos : pos + seg].astype(np.uint32) << shft
                    pos += seg
                    i += seg
    return pixels


def _chroma_xyz(u, v, L, shape):
    s = 1.0 / (6.0 * u - 16.0 * v + 12.0)
    x = 9.0 * u * s
    yv = 4.0 * v * s
    out = np.empty(shape + (3,), dtype=np.float64)
    out[..., 0] = x / yv * L
    out[..., 1] = L
    out[..., 2] = (1.0 - x - yv) / yv * L
    out[~(L > 0.0)] = 0.0
    return out


def _logluv32_numpy(p):
    L = ll._logl16_to_y(p.view(np.int32) >> 16)
    u = (((p >> 8) & 0xFF).astype(np.float64) + 0.5) / ll._UVSCALE
    v = ((p & 0xFF).astype(np.float64) + 0.5) / ll._UVSCALE
    return _chroma_xyz(u, v, L, p.shape)


def _logluv24_numpy(p):
    L = ll._logl10_to_y((p >> 14) & 0x3FF)
    u, v = ll._uv_decode_index(p & 0x3FFF)
    return _chroma_xyz(u, v, L, p.shape)


def _packets(rng, npixels, width):
    """Random run and literal packets for every byte plane of every scanline."""
    out = bytearray()
    for start in range(0, npixels, width):
        n = min(width, npixels - start)
        for _ in range(4):
            i = 0
            while i < n:
                if rng.random() < 0.5:
                    run = int(rng.integers(2, 130))
                    out += bytes([126 + run, int(rng.integers(0, 256))])
                    i += run
                else:
                    lit = int(rng.integers(0, 128))
                    out.append(lit)
                    out += rng.integers(0, 256, lit, dtype=np.uint8).tobytes()
                    i += lit
    return bytes(out)


@pytest.mark.parametrize("width,rows", [(1, 7), (37, 5), (300, 3), (1000, 1)])
def test_rle_strip_matches_numpy(width, rows):
    rng = np.random.default_rng(width)
    npix = width * rows
    strip = _packets(rng, npix, width)
    want = _decode_numpy(strip, npix, width)
    np.testing.assert_array_equal(ll.decode_strip_logluv32(strip, npix, width), want)
    for cut in (0, 1, len(strip) // 3, len(strip) - 1):
        np.testing.assert_array_equal(ll.decode_strip_logluv32(strip[:cut], npix, width), _decode_numpy(strip[:cut], npix, width))


@pytest.mark.parametrize("seed", range(4))
def test_rle_corrupt_strip_matches_numpy(seed):
    rng = np.random.default_rng(seed)
    garbage = rng.integers(0, 256, 5000, dtype=np.uint8).tobytes()
    for npix, width in ((997, 31), (4000, 4000), (64, 7)):
        np.testing.assert_array_equal(ll.decode_strip_logluv32(garbage, npix, width), _decode_numpy(garbage, npix, width))
    np.testing.assert_array_equal(ll.decode_strip_logluv32(garbage, 50, None), _decode_numpy(garbage, 50, 50))


def _codes32(rng):
    p = rng.integers(0, 1 << 32, (311, 643), dtype=np.uint32)
    p[0, :4] = (0, 0x80000000, 0xFFFFFFFF, 0x7FFFFFFF)
    p[1, :2] = (0x00010000 | 0xFF00, 0x80010000)
    return p


def _rgb_numpy(xyz):
    """The matrix product in float64, summed left to right, rounded to float32 once."""
    m, (x, y, z) = ll._XYZ_TO_LRGB, np.moveaxis(xyz, -1, 0)
    return np.stack([m[r, 0] * x + m[r, 1] * y + m[r, 2] * z for r in range(3)], axis=-1).astype(np.float32)


def test_logluv32_rgb_matches_numpy():
    p = _codes32(np.random.default_rng(1))
    np.testing.assert_array_equal(ll.logluv32_to_rgb(p), _rgb_numpy(_logluv32_numpy(p)))


def test_logluv24_rgb_matches_numpy_for_every_code():
    p = np.arange(1 << 24, dtype=np.uint32).reshape(4096, 4096)
    np.testing.assert_array_equal(ll.logluv24_to_rgb(p), _rgb_numpy(_logluv24_numpy(p)))


def _normalize_numpy(lin):
    out = np.empty(lin.shape, dtype=np.float32)
    for ch in range(lin.shape[-1]):
        v = lin[..., ch].astype(np.float64)
        lo, hi = np.percentile(v, (0.2, 99.8))
        out[..., ch] = np.clip((v - lo) / (hi - lo if hi > lo else 1.0), 0.0, 1.0)
    return out


@pytest.mark.parametrize("shape", [(1, 1, 3), (7, 5, 3), (311, 517, 3), (40, 30, 4)])
def test_normalize_linear_matches_numpy(shape):
    rng = np.random.default_rng(sum(shape))
    lin = (rng.lognormal(0.0, 1.5, shape) - 0.5).astype(np.float32)
    lin[..., -1] = 0.25 if shape[0] == 7 else lin[..., -1]
    np.testing.assert_array_equal(ll.normalize_linear(lin), _normalize_numpy(lin))


class _Tag:
    def __init__(self, value):
        self.value = value


class _Page:
    def __init__(self, **tags):
        self.tags = {k: _Tag(v) for k, v in tags.items()}


def _strips(compression, height, width, rps, seed):
    rng = np.random.default_rng(seed)
    data, offsets, counts = bytearray(b"\0" * 5), [], []
    for top in range(0, height, rps):
        npix = min(rps, height - top) * width
        if compression == ll.COMPRESSION_SGILOG:
            strip = _packets(rng, npix, width)
        else:
            strip = rng.integers(0, 256, npix * 3, dtype=np.uint8).tobytes()
        offsets.append(len(data))
        counts.append(len(strip))
        data += strip
    page = _Page(Compression=compression, RowsPerStrip=rps, StripOffsets=offsets, StripByteCounts=counts)
    return page, bytes(data)


def _decode_strips_numpy(page, width, height, data):
    rps, offsets, counts = (page.tags[k].value for k in ("RowsPerStrip", "StripOffsets", "StripByteCounts"))
    parts = []
    for si, (off, cnt) in enumerate(zip(offsets, counts)):
        npix = min(rps, height - si * rps) * width
        chunk = data[off : off + cnt]
        if page.tags["Compression"].value == ll.COMPRESSION_SGILOG:
            parts.append(_decode_numpy(chunk, npix, width))
        else:
            b = np.frombuffer(chunk, dtype=np.uint8).reshape(npix, 3).astype(np.uint32)
            parts.append((b[:, 0] << 16) | (b[:, 1] << 8) | b[:, 2])
    packed = np.concatenate(parts).reshape(height, width)
    to_xyz = _logluv32_numpy if page.tags["Compression"].value == ll.COMPRESSION_SGILOG else _logluv24_numpy
    return to_xyz(packed)


def _normalize_f64(lin):
    lo, hi = np.percentile(lin, (0.2, 99.8), axis=(0, 1))
    return np.clip((lin - lo) / np.where(hi > lo, hi - lo, 1.0), 0.0, 1.0)


@pytest.mark.parametrize("compression", [ll.COMPRESSION_SGILOG, ll.COMPRESSION_SGILOG24])
def test_decode_logluv_strips_matches_numpy(compression):
    page, data = _strips(compression, 23, 41, 5, compression)
    got = ll.decode_logluv_strips([page], 41, 23, data, "<")
    xyz = _decode_strips_numpy(page, 41, 23, data)
    np.testing.assert_array_equal(got, _normalize_numpy(_rgb_numpy(xyz)))
    # Float32 RGB moves the percentile bounds, so the result stays within float32 steps of 1.
    exact = _normalize_f64(xyz @ ll._XYZ_TO_LRGB.T)
    np.testing.assert_allclose(got, exact, rtol=0, atol=2**-22)


def test_rle_pin():
    rng = np.random.default_rng(7)
    strip = _packets(rng, 37 * 9, 37)
    assert digest(ll.decode_strip_logluv32(strip, 37 * 9, 37)) == "66c494002ee00602decf8f02b24d19081ed92b955ebda184bcd6ede4273ddfd2"


@linux_pin
def test_logluv_decode_pin():
    got = [
        ll.decode_logluv_strips([page], 41, 23, data, "<")
        for page, data in (_strips(c, 23, 41, 5, 3) for c in (ll.COMPRESSION_SGILOG, ll.COMPRESSION_SGILOG24))
    ]
    assert digest(got) == "693905c8de55fd7a3cf091cb42b06b6f426b8f30c31d6374d5b46e53b03bac3f"
