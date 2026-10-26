import math

import negpy_core
import numpy as np
import pytest

import negpy.features.retouch.logic as retouch
from negpy.infrastructure.loaders.ir_planes import normalize_ir_to_float32


def _numpy_rows(horizontal, output, top, bottom, source_top, scale_y, h):
    """The NumPy vertical pass negpy_core.area_rows_into replaces, as the oracle."""
    for row in range(top, bottom):
        start = row * scale_y
        end = start + scale_y
        first, last = math.floor(start), min(h, math.ceil(end))
        indices = np.arange(first, last)
        overlap = np.minimum(indices + 1, end) - np.maximum(indices, start)
        weights = (np.where(overlap > 1e-3, overlap, 0.0) / min(scale_y, h - start)).astype(np.float32)
        values = horizontal[first - source_top : last - source_top]
        output[row] = np.sum(values * weights.reshape((-1,) + (1,) * (values.ndim - 1)), axis=0)


@pytest.mark.parametrize("shape,dims", [((997, 641), (233, 150)), ((1200, 800, 3), (301, 200)), ((640, 999), (128, 81))])
def test_area_rows_match_numpy(shape, dims):
    rng = np.random.default_rng(len(shape) + dims[0])
    plane = rng.random(shape, dtype=np.float32)
    h = shape[0]
    dw, dh = dims
    scale_y = h / dh
    for top, bottom in [(0, dh // 3), (dh // 3, dh - 7), (dh - 7, dh)]:
        source_top = math.floor(top * scale_y)
        source_bottom = min(h, math.ceil((bottom - 1) * scale_y + scale_y))
        horizontal = negpy_core.resize_area(np.ascontiguousarray(plane[source_top:source_bottom]), (dw, source_bottom - source_top))
        expected = np.zeros((dh, dw) + shape[2:], np.float32)
        got = np.zeros_like(expected)
        _numpy_rows(horizontal, expected, top, bottom, source_top, scale_y, h)
        negpy_core.area_rows_into(horizontal, got[top:bottom], top, source_top, scale_y, h)
        assert np.array_equal(got[top:bottom], expected[top:bottom])


def test_bounded_downsample_matches_whole_plane(monkeypatch):
    rng = np.random.default_rng(7)
    plane = rng.random((1500, 1001), dtype=np.float32)
    whole = retouch.downsample_ir(plane, 400)
    monkeypatch.setattr(retouch, "_IR_DOWNSAMPLE_WORK_BYTES", 1 << 18)
    blocks = retouch.downsample_ir(plane, 400)
    assert blocks.shape == whole.shape
    assert np.allclose(blocks, whole, atol=1e-6)


def test_area_rows_rejects_rows_outside_the_block():
    src = np.zeros((4, 10), np.float32)
    with pytest.raises(ValueError):
        negpy_core.area_rows_into(src, np.zeros((5, 10), np.float32), 0, 0, 3.0, 30)


@pytest.mark.parametrize("dtype,scale", [(np.uint16, 1.0 / 65535.0), (np.uint8, 1.0 / 255.0)])
def test_ir_normalize_matches_numpy(dtype, scale):
    ir = np.random.default_rng(3).integers(0, np.iinfo(dtype).max + 1, (311, 257), dtype=dtype)
    assert np.array_equal(normalize_ir_to_float32(ir), ir.astype(np.float32) * scale)
    assert np.array_equal(normalize_ir_to_float32(ir[:, :, None].repeat(2, axis=2)), ir.astype(np.float32) * scale)
