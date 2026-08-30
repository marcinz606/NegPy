"""Integrated GPUs get a smaller tile and no readback pipelining in
`_process_tiled` (see gpu_engine.TILE_SIZE_INTEGRATED_GPU): a full 2048px tile's
live intermediate textures, doubled up by the one-tile-ahead readback overlap,
can exceed an iGPU's tight, unqueryable VRAM budget and abort the process via
wgpu-native's panic-on-device-lost (see issue #738). This must not change the
tiled export's output, only how much GPU memory is live at once getting there.
"""

import unittest
from dataclasses import replace
from unittest.mock import PropertyMock, patch

import numpy as np

from negpy.domain.models import WorkspaceConfig
from negpy.features.process.models import ProcessMode
from negpy.infrastructure.gpu.device import GPUDevice
from negpy.services.rendering.gpu_engine import GPUEngine, TILE_SIZE, TILE_SIZE_INTEGRATED_GPU


def _negative(h: int, w: int) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    ramp = (xx / w) * 0.5 + (yy / h) * 0.25
    blobs = 0.08 * np.sin(xx / 90.0) * np.cos(yy / 40.0) + 0.05 * np.sin((xx + yy) / 160.0)
    img = np.empty((h, w, 3), dtype=np.float32)
    img[..., 0] = 0.55 + ramp * 0.30 + blobs
    img[..., 1] = 0.35 + ramp * 0.25 + blobs * 0.7
    img[..., 2] = 0.15 + ramp * 0.20 + blobs * 0.4
    return np.clip(img, 1e-4, 1.0)


def _base() -> WorkspaceConfig:
    s = WorkspaceConfig()
    return replace(
        s,
        process=replace(s.process, process_mode=ProcessMode.C41),
        export=replace(s.export, export_resolution_mode="original"),
    )


@unittest.skipUnless(GPUDevice.get().is_available, "GPU not available")
class TestGpuTiledIntegratedGpu(unittest.TestCase):
    def setUp(self):
        self.engine = GPUEngine()
        # Wider than either tile size, so both branches actually span multiple tiles.
        self.img = _negative(300, 2400)

    def tearDown(self):
        self.engine.destroy_all()

    def _tile_count(self, is_integrated: bool) -> int:
        calls = 0
        real = self.engine.process_to_texture

        def counting(*args, **kwargs):
            nonlocal calls
            calls += 1
            return real(*args, **kwargs)

        with patch.object(GPUDevice, "is_integrated", new_callable=PropertyMock) as mock_integrated:
            mock_integrated.return_value = is_integrated
            with patch.object(self.engine, "process_to_texture", side_effect=counting):
                self.engine._process_tiled(self.img, _base(), scale_factor=1.0)
        return calls

    def test_integrated_gpu_uses_smaller_tile(self):
        # One extra call is the preview-sized metering pass shared by both branches;
        # what should differ is how many tiles that leaves for the crop itself.
        discrete_tiles = self._tile_count(is_integrated=False) - 1
        integrated_tiles = self._tile_count(is_integrated=True) - 1
        self.assertGreater(
            integrated_tiles,
            discrete_tiles,
            "Integrated GPU did not split the export into more, smaller tiles",
        )
        self.assertEqual(TILE_SIZE, 2048)
        self.assertLess(TILE_SIZE_INTEGRATED_GPU, TILE_SIZE)

    def test_integrated_gpu_output_matches_discrete_tiling(self):
        """Smaller tiles and no pipelining must not change what gets exported."""
        settings = _base()
        with patch.object(GPUDevice, "is_integrated", new_callable=PropertyMock) as mock_integrated:
            mock_integrated.return_value = False
            discrete, _ = self.engine._process_tiled(self.img, settings, scale_factor=1.0)
            mock_integrated.return_value = True
            integrated, _ = self.engine._process_tiled(self.img, settings, scale_factor=1.0)
        self.assertEqual(discrete.shape, integrated.shape)
        self.assertLess(float(np.abs(discrete - integrated).mean()), 0.0005)


if __name__ == "__main__":
    unittest.main()
