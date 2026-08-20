"""
Wall-clock metrics for the export path: full-res render and encode, measured
separately (real GPU when available, synthetic source on disk).

Recorded under ``export.*``; the regression check in ``test_preview_load``
compares every metric in the session snapshot against ``NEGPY_METRICS_BASELINE``.
"""

from __future__ import annotations

import time

import numpy as np
import pytest
import tifffile

from negpy.domain.models import WorkspaceConfig
from negpy.infrastructure.gpu.device import GPUDevice

from . import recorder

pytestmark = [pytest.mark.slow, pytest.mark.metrics]

# Above TILING_THRESHOLD_PX, so the measured path is the tiled one real scans take.
_W, _H = 4600, 3200


@pytest.fixture(scope="module")
def rig(tmp_path_factory):
    if not GPUDevice.get().is_available:
        pytest.skip("GPU not available")
    from negpy.services.rendering.image_processor import ImageProcessor

    rng = np.random.default_rng(1)
    arr = (rng.random((_H, _W, 3)) * 40000 + 8000).astype(np.uint16)
    path = tmp_path_factory.mktemp("export") / "metric_source.tif"
    tifffile.imwrite(path, arr, photometric="rgb")

    proc = ImageProcessor()
    try:
        yield proc, str(path)
    finally:
        proc.destroy_all()


def test_export_render_and_encode(rig) -> None:
    proc, path = rig
    cfg = WorkspaceConfig()

    # Warm run; measure steady state.
    fbuf, cs = proc._render_export_buffer(path, cfg, cfg.export, "metrics-export")
    proc._encode_export(fbuf, cfg.export, cs)

    t0 = time.perf_counter()
    fbuf, cs = proc._render_export_buffer(path, cfg, cfg.export, "metrics-export-2")
    render_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    bits, _ = proc._encode_export(fbuf, cfg.export, cs)
    encode_s = time.perf_counter() - t0

    assert bits
    recorder.record("export.render_s", render_s)
    recorder.record("export.encode_s", encode_s)
    assert render_s < 30.0, f"export render took {render_s:.1f}s"
    assert encode_s < 10.0, f"export encode took {encode_s:.1f}s"
