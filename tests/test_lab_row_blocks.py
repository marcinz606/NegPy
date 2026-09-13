import numpy as np
import pytest

from negpy.features.lab.logic import (
    apply_output_sharpening,
    apply_saturation,
    _output_sharpening_block,
    _saturation_block,
)
from negpy.kernel.system import parallel


@pytest.mark.parametrize("radius,mask", [(0.1, 0.0), (1.0, 0.0), (3.0, 0.6)])
def test_sharpen_blocks_match_full_frame(radius, mask):
    image = np.random.default_rng(3).uniform(0.01, 0.9, (1100, 1024, 3)).astype(np.float32)
    original = image.copy()
    full = _output_sharpening_block(image, 0.25, radius, mask)
    blocks = apply_output_sharpening(image, 0.25, radius, mask)
    np.testing.assert_array_equal(blocks, full)
    np.testing.assert_array_equal(image, original)


@pytest.mark.parametrize("parallel_enabled", [False, True])
@pytest.mark.parametrize("tail_rows", [1, 21, 22])
@pytest.mark.parametrize("saturation,skin", [(1.0, 1.0), (0.6, 0.7)])
def test_saturation_short_tail_matches_full_frame(monkeypatch, parallel_enabled, tail_rows, saturation, skin):
    monkeypatch.setattr(parallel, "_parallel_enabled", parallel_enabled)
    image = np.random.default_rng(991).uniform(0.001, 0.999, (3072 + tail_rows, 1024, 3)).astype(np.float32)
    image[::7] = 0.0
    image[1::11] = 1.0
    image = image[:, ::-1, :]
    original = image.copy()
    full = _saturation_block(image, saturation, skin)
    blocks = apply_saturation(image, saturation, skin)
    np.testing.assert_array_equal(blocks, full)
    assert blocks.tobytes() == full.tobytes()
    np.testing.assert_array_equal(image, original)


@pytest.mark.parametrize("saturation,skin", [(1.0, 0.5), (0.7, 0.0), (1.3, 0.8)])
def test_saturation_blocks_match_full_frame(saturation, skin):
    image = np.random.default_rng(4).uniform(0.01, 0.9, (1100, 1024, 3)).astype(np.float32)
    original = image.copy()
    full = _saturation_block(image, saturation, skin)
    blocks = apply_saturation(image, saturation, skin)
    np.testing.assert_array_equal(blocks, full)
    np.testing.assert_array_equal(image, original)
