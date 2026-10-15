"""The mat around a bordered print carries the picked color on every engine."""

from dataclasses import replace

import numpy as np
import pytest

from negpy.domain.models import WorkspaceConfig
from negpy.infrastructure.gpu.device import GPUDevice
from negpy.services.export.print import PrintService

pytestmark = pytest.mark.skipif(not GPUDevice.get().is_available, reason="GPU not available")


def _bordered(**finish) -> WorkspaceConfig:
    cfg = WorkspaceConfig()
    return replace(cfg, finish=replace(cfg.finish, border_size=1.0, **finish))


def _img() -> np.ndarray:
    return np.random.default_rng(0).uniform(0.2, 0.6, (96, 128, 3)).astype(np.float32)


def test_the_direct_gpu_path_paints_the_picked_border_color():
    from negpy.services.rendering.gpu_engine import GPUEngine

    engine = GPUEngine()
    try:
        tex, _ = engine.process_to_texture(_img(), _bordered(border_color="#808080"), readback_metrics=False)
        corner = engine._readback_downsampled(tex)[0, 0, :3]
    finally:
        engine.destroy_all()
    np.testing.assert_allclose(corner, 128 / 255, atol=2e-3)


def test_tiled_export_matches_the_paper_white():
    from negpy.services.rendering.gpu_engine import GPUEngine

    cfg = _bordered(border_match_paper=True)
    cfg = replace(cfg, toning=replace(cfg.toning, highlight_tint_strength=0.8, highlight_tint_hue=60.0))
    expected = PrintService.effective_border_color(cfg.finish, cfg.toning).lstrip("#")
    engine = GPUEngine()
    try:
        result, _ = engine._process_tiled(_img(), cfg, scale_factor=1.0)
    finally:
        engine.destroy_all()
    np.testing.assert_allclose(result[0, 0, :3], [int(expected[i : i + 2], 16) / 255 for i in (0, 2, 4)], atol=2e-3)
