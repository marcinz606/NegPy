"""
Wall-clock metrics for a preview *render frame* (synthetic buffers, real GPU).

`test_preview_load` covers decode; this covers the render itself.

A frame is timed as render-thread wall time **plus the GPU time it still owes**:
`submit()` only queues work, so timing the call alone lets a change that defers work
look like a speed-up.

Recorded under ``render.frame.*``; the regression check in ``test_preview_load``
compares every metric in the session snapshot against ``NEGPY_METRICS_BASELINE``.
"""

from __future__ import annotations

import dataclasses
import time

import numpy as np
import pytest

from negpy.domain.models import WorkspaceConfig
from negpy.infrastructure.gpu.device import GPUDevice

from . import recorder

pytestmark = [pytest.mark.slow, pytest.mark.metrics]

# Big enough that per-frame cost dominates fixed overhead, small enough to stay quick.
_W, _H = 1600, 1066
_FRAMES = 8


def _gpu_available() -> bool:
    return GPUDevice.get().is_available


@pytest.fixture(scope="module")
def rig():
    if not _gpu_available():
        pytest.skip("GPU not available")
    import wgpu

    from negpy.services.rendering.image_processor import ImageProcessor

    rng = np.random.default_rng(0)
    img = rng.random((_H, _W, 3), dtype=np.float32) * 0.5 + 0.2
    proc = ImageProcessor()
    engine = proc.engine_gpu
    device = engine.gpu.device
    sync = device.create_buffer(size=256, usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ)

    def drain() -> float:
        """Seconds the GPU still owed when the render call returned."""
        enc = device.create_command_encoder()
        enc.copy_buffer_to_buffer(engine._buffers["metrics"].buffer, 0, sync, 0, 256)
        device.queue.submit([enc.finish()])
        t0 = time.perf_counter()
        sync.map_sync(wgpu.MapMode.READ)
        sync.unmap()
        return time.perf_counter() - t0

    try:
        yield proc, img, drain
    finally:
        proc.destroy_all()


def _frame_seconds(rig, mutate, *, settle: bool) -> float:
    """Median wall time of one frame, GPU included."""
    proc, img, drain = rig
    base = WorkspaceConfig()

    def render(i):
        cfg = mutate(base, i)
        t0 = time.perf_counter()
        proc.run_pipeline(img, cfg, "metrics-frame", render_size_ref=float(_W), prefer_gpu=True, readback_metrics=settle)
        cpu = time.perf_counter() - t0
        return cpu + drain()

    render(0)  # warm the shaders, the analysis cache and the source upload
    return float(np.median([render(i) for i in range(1, _FRAMES + 1)]))


def _density(base, i):
    return dataclasses.replace(base, exposure=dataclasses.replace(base.exposure, density=0.005 * (i + 1)))


def _sepia(base, i):
    return dataclasses.replace(base, toning=dataclasses.replace(base.toning, sepia_strength=0.02 * (i + 1)))


def _white_point(base, i):
    return dataclasses.replace(base, process=dataclasses.replace(base.process, white_point_offset=0.001 * (i + 1)))


def _luma_clip(base, i):
    return dataclasses.replace(base, process=dataclasses.replace(base.process, luma_range_clip=0.001 * (i + 1)))


def test_drag_frame(rig) -> None:
    """An exposure slider step: resumes from the exposure stage, no metrics readback."""
    elapsed = _frame_seconds(rig, _density, settle=False)
    recorder.record("render.frame.drag_s", elapsed)
    assert elapsed < 0.5, f"a {_W}px drag frame took {elapsed * 1000:.0f}ms"


def test_late_stage_drag_frame(rig) -> None:
    """A toning step resumes far later in the chain, so it must not cost more."""
    elapsed = _frame_seconds(rig, _sepia, settle=False)
    recorder.record("render.frame.drag_late_stage_s", elapsed)
    assert elapsed < 0.5, f"a toning-only frame took {elapsed * 1000:.0f}ms"


def test_white_point_drag_frame(rig) -> None:
    """A white-point step applies as a uniform offset: it must reuse the analysis
    cache and cost no more than a creative-slider frame."""
    elapsed = _frame_seconds(rig, _white_point, settle=False)
    recorder.record("render.frame.drag_offset_s", elapsed)
    assert elapsed < 0.5, f"a white-point drag frame took {elapsed * 1000:.0f}ms"


def test_clip_drag_frame(rig) -> None:
    """A clip step re-meters by design, but reuses the prefiltered log grid; only
    the percentile analysis re-runs."""
    elapsed = _frame_seconds(rig, _luma_clip, settle=False)
    recorder.record("render.frame.drag_remeter_s", elapsed)
    assert elapsed < 0.5, f"a clip drag frame took {elapsed * 1000:.0f}ms"


def test_settle_frame(rig) -> None:
    """Release: the same render plus the metrics/histogram readback."""
    elapsed = _frame_seconds(rig, _density, settle=True)
    recorder.record("render.frame.settle_s", elapsed)
    assert elapsed < 1.0, f"a settle frame took {elapsed * 1000:.0f}ms"


def test_stage_skipping_is_actually_engaged(rig) -> None:
    """The guard the wall-clock budgets cannot give.

    A render that stopped resuming would still land inside the budgets at this
    resolution, so assert the resume itself: a toning-only change must not re-run
    the geometry stage.
    """
    proc, img, _ = rig
    engine = proc.engine_gpu
    base = WorkspaceConfig()
    proc.run_pipeline(img, base, "skip-probe", render_size_ref=float(_W), prefer_gpu=True, readback_metrics=False)

    dispatched: list[str] = []
    real = engine._dispatch_pass

    def spy(enc, name, bindings, w, h):
        dispatched.append(name)
        return real(enc, name, bindings, w, h)

    engine._dispatch_pass = spy
    try:
        proc.run_pipeline(img, _sepia(base, 1), "skip-probe", render_size_ref=float(_W), prefer_gpu=True, readback_metrics=False)
    finally:
        engine._dispatch_pass = real

    assert "toning" in dispatched, "the changed stage must run"
    assert "geometry" not in dispatched, f"stage skipping is not engaged — geometry re-ran for a toning change: {dispatched}"
