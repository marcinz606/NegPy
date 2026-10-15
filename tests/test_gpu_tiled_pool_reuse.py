"""A memory-bounded tiled render keeps one tile shape pooled, not one tile."""

from unittest.mock import patch

import numpy as np
import pytest

from negpy.domain.models import WorkspaceConfig
from negpy.infrastructure.gpu.device import GPUDevice

pytestmark = pytest.mark.skipif(not GPUDevice.get().is_available, reason="GPU not available")


def test_tiles_of_one_shape_share_the_pool_and_the_pixels_do_not_move():
    from negpy.services.rendering import gpu_engine
    from negpy.services.rendering.gpu_engine import GPUEngine

    img = np.random.default_rng(0).uniform(0.2, 0.6, (300, 1000, 3)).astype(np.float32)
    engine = GPUEngine()
    try:
        with patch.object(gpu_engine, "TILE_SIZE_LOW_VRAM", 128), patch.object(gpu_engine, "TILE_SIZE", 128):
            reference, _ = engine._process_tiled(img, WorkspaceConfig(), scale_factor=1.0)
            with patch.object(GPUEngine, "_release_texture_pool", autospec=True, side_effect=GPUEngine._release_texture_pool) as release:
                bounded, _ = engine._process_tiled(img, WorkspaceConfig(), scale_factor=1.0, memory_bounded=True)
    finally:
        engine.destroy_all()
    tiles = 3 * 8
    assert release.call_count < tiles // 2
    np.testing.assert_array_equal(bounded, reference)
