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


def test_logluv32_xyz_matches_numpy():
    p = _codes32(np.random.default_rng(1))
    np.testing.assert_array_equal(ll.logluv32_to_xyz(p), _logluv32_numpy(p))


def test_logluv24_xyz_matches_numpy_for_every_code():
    p = np.arange(1 << 24, dtype=np.uint32).reshape(4096, 4096)
    np.testing.assert_array_equal(ll.logluv24_to_xyz(p), _logluv24_numpy(p))


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
    return ll.normalize_linear(ll.xyz_to_linear_rgb(to_xyz(packed))).astype(np.float32)


@pytest.mark.parametrize("compression", [ll.COMPRESSION_SGILOG, ll.COMPRESSION_SGILOG24])
def test_decode_logluv_strips_matches_numpy(compression):
    page, data = _strips(compression, 23, 41, 5, compression)
    got = ll.decode_logluv_strips([page], 41, 23, data, "<")
    np.testing.assert_array_equal(got, _decode_strips_numpy(page, 41, 23, data))


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
    assert digest(got) == "ffecfa3abf64675b5b5697b7335af788b9feca8a6e3aa01f074c1edfd4a918a4"
