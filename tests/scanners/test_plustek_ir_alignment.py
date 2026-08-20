"""Whole-pixel IR registration for Plustek USB two-pass colour+IR scans."""

import cv2
import numpy as np
import pytest

pytest.importorskip("pyopticfilm")

from pyopticfilm.ir_align import align_ir_to_rgb  # noqa: E402


def _texture(h=128, w=128, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.random((h, w), dtype=np.float32)
    return cv2.GaussianBlur(base, (0, 0), 1.5)


def _shift(img, dx, dy):
    m = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
    return cv2.warpAffine(
        img,
        m,
        (img.shape[1], img.shape[0]),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REFLECT,
    )


def test_plustek_align_recovers_whole_pixel_offset_exactly():
    base = _texture()
    rgb = np.stack([base, base, base], axis=-1)
    ir = np.roll(np.roll(base, 2, axis=1), -3, axis=0)

    aligned = align_ir_to_rgb(rgb, ir)

    sl = (slice(8, -8), slice(8, -8))
    np.testing.assert_allclose(aligned[sl], base[sl], atol=1e-6)


def test_plustek_align_never_interpolates():
    base = _texture()
    rgb = np.stack([base, base, base], axis=-1)
    ir = _shift(base, 1.5, -2.0)

    aligned = align_ir_to_rgb(rgb, ir)

    assert np.isin(aligned, ir).all()


def test_plustek_align_corrects_carriage_offset():
    base = _texture()
    rgb = np.stack([base, base, base], axis=-1)
    ir = _shift(base, 1.5, -2.0)

    aligned = align_ir_to_rgb(rgb, ir)

    sl = (slice(8, -8), slice(8, -8))
    err_aligned = np.abs(aligned[sl] - base[sl]).mean()
    err_raw = np.abs(ir[sl] - base[sl]).mean()
    assert err_aligned < err_raw * 0.5


def test_plustek_align_preserves_dtype_and_shape():
    base = _texture()
    rgb = np.stack([base, base, base], axis=-1)
    ir = (_shift(base, 1.0, 1.0) * 65535).astype(np.uint16)

    aligned = align_ir_to_rgb((rgb * 65535).astype(np.uint16), ir)

    assert aligned.dtype == ir.dtype
    assert aligned.shape == ir.shape


def test_plustek_align_shape_mismatch_is_noop():
    rgb = np.zeros((10, 10, 3), np.uint16)
    ir = np.zeros((12, 10), np.uint16)
    assert align_ir_to_rgb(rgb, ir) is ir


def test_plustek_align_zero_shift_is_noop_identity():
    base = _texture()
    rgb = np.stack([base, base, base], axis=-1)
    assert align_ir_to_rgb(rgb, base) is base


def test_plustek_align_recovers_vertical_only_offset():
    """USB Plustek at 1800 PPI often shows ~10 px Y and ~0 X."""
    base = _texture(h=256, w=256)
    rgb = np.stack([base, base, base], axis=-1)
    ir = np.roll(base, 11, axis=0)

    aligned = align_ir_to_rgb(rgb, ir)

    sl = (slice(16, -16), slice(16, -16))
    np.testing.assert_allclose(aligned[sl], base[sl], atol=1e-6)
