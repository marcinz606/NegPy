"""The GPU print layout matches PrintService.apply_layout: same content size, area-filtered."""

from dataclasses import replace

import numpy as np
import pytest

from negpy.domain.models import ExportResolutionMode, WorkspaceConfig
from negpy.infrastructure.gpu.device import GPUDevice
from negpy.services.export.print import PrintService

pytestmark = pytest.mark.skipif(not GPUDevice.get().is_available, reason="GPU not available")


@pytest.mark.parametrize("shape", [(900, 1200), (1200, 900), (777, 1311)])
def test_gpu_layout_matches_the_cpu_layout(shape):
    from negpy.services.rendering.gpu_engine import GPUEngine

    rng = np.random.default_rng(3)
    xx = np.mgrid[0 : shape[0], 0 : shape[1]][1] / float(shape[1])
    img = np.clip(0.15 + 0.5 * xx[..., None] + rng.normal(0, 0.02, (*shape, 3)), 0.02, 0.95).astype(np.float32)
    cfg = WorkspaceConfig()
    cfg = replace(
        cfg,
        geometry=replace(cfg.geometry, autocrop_offset=0),
        export=replace(cfg.export, export_resolution_mode=ExportResolutionMode.TARGET_PX.value, export_target_long_edge_px=400),
        finish=replace(cfg.finish, border_size=0.3),
    )
    engine = GPUEngine()
    try:
        tex, _ = engine.process_to_texture(img, cfg, scale_factor=1.0, readback_metrics=False)
        gpu = engine._readback_downsampled(tex)[:, :, :3].astype(np.float64)
        plain_tex, _ = engine.process_to_texture(img, cfg, scale_factor=1.0, readback_metrics=False, apply_layout=False)
        plain = np.ascontiguousarray(engine._readback_downsampled(plain_tex)[:, :, :3])
    finally:
        engine.destroy_all()
    cpu, _ = PrintService.apply_layout(plain, cfg.export, border_size=cfg.finish.border_size, finish=cfg.finish)

    assert gpu.shape == cpu.shape
    diff = np.abs(gpu - cpu)
    assert float(diff.mean()) < 0.005
    assert float(diff.max()) < 0.05  # no mat line where the content falls a pixel short
