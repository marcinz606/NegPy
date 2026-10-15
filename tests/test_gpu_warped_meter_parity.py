"""The GPU meters the region the print shows when a crop is combined with a warp, and
its meter caches follow the warp sliders."""

from dataclasses import replace

import numpy as np
import pytest

from negpy.domain.models import WorkspaceConfig
from negpy.infrastructure.gpu.device import GPUDevice
from negpy.services.rendering.engine import DarkroomEngine

pytestmark = pytest.mark.skipif(not GPUDevice.get().is_available, reason="GPU not available")


def _scene() -> np.ndarray:
    """Dense left half, thin right half: a crop near the edge meters very differently
    depending on which way it is rotated."""
    rng = np.random.default_rng(1)
    img = np.empty((256, 256, 3), np.float32)
    x = np.linspace(0.0, 1.0, 256, dtype=np.float32)
    img[:] = (0.05 + 0.6 * x**2)[None, :, None]
    img *= np.array([1.0, 0.8, 0.6], np.float32)
    return np.clip(img + rng.normal(0, 0.004, img.shape).astype(np.float32), 0.0, 1.0)


def _warped(**geo) -> WorkspaceConfig:
    cfg = WorkspaceConfig()
    return replace(cfg, geometry=replace(cfg.geometry, crop_rect=(0.55, 0.2, 0.95, 0.8), autocrop_offset=0, **geo))


def _gpu(engine, img, cfg, source_hash=None):
    tex, metrics = engine.process_to_texture(
        img, cfg, scale_factor=1.0, apply_layout=False, readback_metrics=False, source_hash=source_hash, analysis_source_hash=source_hash
    )
    return engine._readback_downsampled(tex), metrics


@pytest.mark.parametrize("geo", [{"fine_rotation": 3.0}, {"converge_v": 8.0}, {"distortion_k1": 0.15}])
def test_a_warped_crop_meters_like_the_cpu(geo):
    from negpy.services.rendering.gpu_engine import GPUEngine

    img = _scene()
    cfg = _warped(**geo)
    engine = GPUEngine()
    try:
        gpu, _ = _gpu(engine, img, cfg)
    finally:
        engine.destroy_all()
    cpu = DarkroomEngine().process(img, cfg, "warped")
    assert cpu.shape == gpu.shape
    assert float(np.mean(np.abs(cpu - gpu))) < 0.001


def test_a_fine_rotation_drag_re_meters():
    from negpy.services.rendering.gpu_engine import GPUEngine

    img = _scene()
    cached = GPUEngine()
    fresh = GPUEngine()
    try:
        _gpu(cached, img, _warped(), source_hash="s")
        after_drag, _ = _gpu(cached, img, _warped(fine_rotation=-3.0), source_hash="s")
        expected, _ = _gpu(fresh, img, _warped(fine_rotation=-3.0), source_hash="s")
    finally:
        cached.destroy_all()
        fresh.destroy_all()
    np.testing.assert_allclose(after_drag, expected, atol=1e-4)


def test_the_contrast_mask_follows_a_fine_rotation_drag():
    from negpy.services.rendering.gpu_engine import GPUEngine

    img = _scene()

    def masked(**geo):
        cfg = _warped(**geo)
        return replace(cfg, exposure=replace(cfg.exposure, contrast_mask=0.6))

    cached, fresh = GPUEngine(), GPUEngine()
    try:
        _gpu(cached, img, masked(), source_hash="s")
        after_drag, _ = _gpu(cached, img, masked(fine_rotation=-3.0), source_hash="s")
        expected, _ = _gpu(fresh, img, masked(fine_rotation=-3.0), source_hash="s")
    finally:
        cached.destroy_all()
        fresh.destroy_all()
    np.testing.assert_allclose(after_drag, expected, atol=1e-4)


def test_contrast_mask_with_point_trims_matches_the_cpu():
    from negpy.services.rendering.image_processor import ImageProcessor

    img = _scene()
    cfg = WorkspaceConfig()
    cfg = replace(
        cfg,
        geometry=replace(cfg.geometry, autocrop_offset=0),
        exposure=replace(cfg.exposure, contrast_mask=0.8),
        process=replace(cfg.process, white_point_offset=0.1, black_point_offset=-0.1),
    )
    processor = ImageProcessor()
    out = []
    for gpu in (False, True):
        result, _ = processor.run_pipeline(img, cfg, f"cm-{gpu}", render_size_ref=256.0, prefer_gpu=gpu, readback_metrics=False)
        arr = np.asarray(result.readback() if hasattr(result, "readback") else result)[:, :, :3]
        out.append(arr.astype(np.float64))
    assert float(np.mean(np.abs(out[0] - out[1]))) < 0.005
