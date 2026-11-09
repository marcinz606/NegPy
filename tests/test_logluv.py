"""Tests for the LogLuv decoder (ported from flexcolor-tool)."""

import numpy as np

from negpy.infrastructure.loaders.logluv import (
    decode_strip_logluv32,
    _XYZ_TO_LRGB,
    logluv24_to_rgb,
    logluv32_to_rgb,
    normalize_linear,
)


_M_LN2 = 0.69314718055994530942
_UVSCALE = 410.0
_U_NEU = 0.210526316
_V_NEU = 0.473684211


def _logl16_from_y(y):
    y = np.asarray(y, dtype=np.float64)
    le = np.floor(256.0 * (np.log2(np.abs(y)) + 64.0)).astype(np.int64)
    le = np.clip(le, 0, 0x7FFF)
    le = np.where(y <= 0, 0, le)
    return le.astype(np.uint32)


def _pack_logluv32(luminance, u_prime=_U_NEU, v_prime=_V_NEU):
    le = _logl16_from_y(luminance)
    ue = np.clip(np.trunc(_UVSCALE * u_prime), 0, 255).astype(np.uint32)
    ve = np.clip(np.trunc(_UVSCALE * v_prime), 0, 255).astype(np.uint32)
    return (le << 16) | (ue << 8) | ve


class TestLogLuv32:
    def test_roundtrip_luminance(self):
        Y_in = np.array([0.001, 0.01, 0.1, 0.5, 1.0, 5.0])
        packed = _pack_logluv32(Y_in)
        xyz = logluv32_to_rgb(packed) @ np.linalg.inv(_XYZ_TO_LRGB).T
        Y_out = xyz[..., 1]
        np.testing.assert_allclose(Y_out, Y_in, rtol=0.02)

    def test_black_pixel(self):
        packed = np.array([0], dtype=np.uint32)
        assert np.all(logluv32_to_rgb(packed) == 0.0)

    def test_rgb_shape(self):
        rgb = logluv32_to_rgb(np.zeros((10, 20), dtype=np.uint32))
        assert rgb.shape == (10, 20, 3)
        assert rgb.dtype == np.float32

    def test_decode_strip_roundtrip(self):
        packed = _pack_logluv32(np.array([0.5, 1.0, 0.1, 2.0]))

        def encode_strip_simple(pixels):
            pixels = np.asarray(pixels, dtype=np.uint32).ravel()
            n = pixels.size
            out = bytearray()
            for shft in (24, 16, 8, 0):
                plane = ((pixels >> shft) & 0xFF).astype(np.uint8)
                for start in range(0, n, 127):
                    chunk = plane[start : start + 127]
                    out.append(len(chunk))
                    out += bytes(chunk)
            return bytes(out)

        compressed = encode_strip_simple(packed)
        decoded = decode_strip_logluv32(compressed, len(packed))
        np.testing.assert_array_equal(decoded, packed)

        # libtiff codes each scanline on its own, so a strip of several rows is the rows'
        # streams one after another.
        rows = _pack_logluv32(np.linspace(0.1, 2.0, 12)).reshape(3, 4)
        strip = b"".join(encode_strip_simple(row) for row in rows)
        np.testing.assert_array_equal(decode_strip_logluv32(strip, 12, width=4), rows.ravel())


class TestLogLuv24:
    def test_zero_luminance(self):
        packed = np.array([0], dtype=np.uint32)
        assert np.all(logluv24_to_rgb(packed) == 0.0)

    def test_nonzero_produces_xyz(self):
        le10 = 500
        ce = 8000
        packed = np.array([(le10 << 14) | ce], dtype=np.uint32)
        assert logluv24_to_rgb(packed)[0].sum() > 0.0


class TestNormalizeLinear:
    def test_hdr_values_normalized_to_unit_range(self):
        lin = np.array(
            [[[0.1, 0.05, 0.2], [0.5, 0.4, 0.6], [2.0, 1.8, 3.0], [4.0, 3.5, 5.0]]],
            dtype=np.float64,
        )
        normed = normalize_linear(lin)
        assert normed.min() >= 0.0
        assert normed.max() <= 1.0

    def test_per_channel_offset_correction(self):
        rng = np.random.default_rng(42)
        n = 1000
        r = rng.uniform(0.5, 2.0, n)
        g = rng.uniform(0.1, 1.5, n)
        b = rng.uniform(0.3, 1.8, n)
        lin = np.stack([r, g, b], axis=-1).reshape(1, n, 3)
        normed = normalize_linear(lin)
        for ch in range(3):
            vals = normed[0, :, ch]
            assert vals.min() >= 0.0
            assert vals.max() <= 1.0
            assert vals.max() > 0.9

    def test_uniform_channel_not_divided_by_zero(self):
        lin = np.full((1, 10, 3), 0.5, dtype=np.float64)
        normed = normalize_linear(lin)
        assert np.all(np.isfinite(normed))


class TestLogLuvToRgb:
    def test_d65_white_maps_neutral(self):
        rgb = logluv32_to_rgb(_pack_logluv32(np.array([1.0]), 0.1978, 0.4683))
        np.testing.assert_allclose(rgb, 1.0, atol=0.2)
        assert rgb.max() - rgb.min() < 0.3
