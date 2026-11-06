"""The transparency path meters the picture box (`meter_rect`) of a frame with no crop, on both engines."""

from dataclasses import replace

import numpy as np
import pytest

from negpy.domain.interfaces import PipelineContext
from negpy.domain.models import WorkspaceConfig
from negpy.features.process.models import ProcessMode, with_positive_source, with_process_mode
from negpy.services.rendering.engine import base_processor

_H, _W = 240, 360
#: (x1, y1, x2, y2) of the picture box, normalized.
_BOX = (0.25, 0.25, 0.75, 0.75)


def _wedge(h: int, w: int, cast: tuple) -> np.ndarray:
    v = np.geomspace(5e-4, 0.9, h * w).astype(np.float32).reshape(h, w)
    return np.stack([v * cast[0], v * cast[1], v * cast[2]], axis=-1)


def _framed_slide() -> np.ndarray:
    """A warm-cast wedge inside a surround with the opposite cast: the axis says which one was metered."""
    rng = np.random.default_rng(3)
    img = _wedge(_H, _W, (0.55, 0.8, 1.0))
    y1, y2, x1, x2 = int(_BOX[1] * _H), int(_BOX[3] * _H), int(_BOX[0] * _W), int(_BOX[2] * _W)
    img[y1:y2, x1:x2] = _wedge(y2 - y1, x2 - x1, (1.0, 0.82, 0.62))
    # Clipped red outside the box and clipped blue inside it: the clip fractions say which region was read.
    img[: _H // 8, : _W // 8, 0] = 1.0
    img[y1 : y1 + _H // 16, x1 : x1 + _W // 16, 2] = 1.0
    return np.ascontiguousarray(img + rng.uniform(0, 1e-4, img.shape).astype(np.float32))


def _config(meter_rect) -> WorkspaceConfig:
    cfg = with_positive_source(with_process_mode(WorkspaceConfig(), ProcessMode.E6), True)
    return replace(
        cfg,
        geometry=replace(cfg.geometry, autocrop_offset=0),
        exposure=replace(cfg.exposure, cast_removal_strength=1.0),
        process=replace(cfg.process, meter_rect=meter_rect),
    )


def _cpu_metrics(img: np.ndarray, cfg: WorkspaceConfig) -> dict:
    ctx = PipelineContext(scale_factor=1.0, original_size=img.shape[:2], process_mode=cfg.process.process_mode)
    base_processor(cfg).process(img, ctx)
    return ctx.metrics


def test_the_cpu_axis_reads_only_the_picture_box():
    img = _framed_slide()
    boxed = _cpu_metrics(img, _config(_BOX))["neutral_axis_refs"]
    whole = _cpu_metrics(img, _config(None))["neutral_axis_refs"]
    assert boxed is not None and whole is not None
    assert not np.allclose(boxed[0], whole[0], atol=1e-3)


def test_the_cpu_clip_fractions_read_only_the_picture_box():
    img = _framed_slide()
    boxed = _cpu_metrics(img, _config(_BOX))["scan_clip_fractions"]
    whole = _cpu_metrics(img, _config(None))["scan_clip_fractions"]
    assert boxed[0] == 0.0 and boxed[2] > 0.0
    assert whole[0] > 0.0


def test_the_gpu_meters_the_same_axis_as_the_cpu():
    from negpy.infrastructure.gpu.device import GPUDevice

    if not GPUDevice.get().is_available:
        pytest.skip("GPU not available")
    from negpy.kernel.system.config import render_scale_factor
    from negpy.services.rendering.gpu_engine import GPUEngine

    img = _framed_slide()
    cfg = _config(_BOX)
    engine = GPUEngine()
    try:
        _, gpu = engine.process_to_texture(img, cfg, scale_factor=render_scale_factor(img.shape), apply_layout=False, readback_metrics=True)
    finally:
        engine.destroy_all()
    cpu = _cpu_metrics(img, cfg)
    for ref in (0, 1):
        np.testing.assert_allclose(gpu["neutral_axis_refs"][ref], cpu["neutral_axis_refs"][ref], atol=1e-5)
    np.testing.assert_allclose(gpu["scan_clip_fractions"], cpu["scan_clip_fractions"], atol=1e-6)
