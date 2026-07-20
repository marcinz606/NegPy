from dataclasses import replace

import numpy as np
from PyQt6.QtGui import QColor

from negpy.desktop.session import AppState
from negpy.desktop.view.canvas.overlay import CanvasOverlay
from negpy.domain.models import WorkspaceConfig
from negpy.features.retouch.models import RetouchConfig
from negpy.services.rendering.image_processor import ImageProcessor


def _speck_image():
    rng = np.random.default_rng(42)
    img = (np.full((160, 160, 3), 0.18) * (1.0 + rng.normal(0, 0.02, (160, 160, 3)))).astype(np.float32)
    img[80:83, 80:83] = 0.005
    return img


def test_augment_retouch_returns_luma_strokes():
    service = ImageProcessor()
    cfg = replace(WorkspaceConfig(), retouch=RetouchConfig(dust_remove=True, dust_threshold=0.5, dust_size=4))
    settings, detected, _ = service._augment_retouch(cfg, _speck_image(), "s")

    assert detected is not None and set(detected) == {"luma"}
    assert len(detected["luma"]) >= 1
    # The merged strokes still reach the render-local config (auto flag cleared).
    assert settings.retouch.dust_remove is False
    assert len(settings.retouch.manual_heal_strokes) >= 1


def test_ir_bake_repairs_defects_in_place():
    """IR defects never become strokes — _ir_bake rebuilds them in the source buffer."""
    h, w = 80, 80
    rng = np.random.default_rng(17)
    img = np.clip(np.full((h, w, 3), 0.5) + rng.normal(0, 0.01, (h, w, 3)), 0, 1).astype(np.float32)
    img[39:42, 39:42] = 0.05
    ir = np.full((h, w), 0.9, dtype=np.float32)
    ir[39:42, 39:42] = 0.05

    service = ImageProcessor()
    cfg = replace(WorkspaceConfig(), retouch=RetouchConfig(ir_dust_remove=True, ir_threshold=0.5))
    baked, corrected_mask, degenerate, routed = service._ir_bake(img, ir, cfg, "s")
    assert not degenerate and routed is None
    assert corrected_mask is not None and corrected_mask[40, 40]
    assert float(np.asarray(baked)[40, 40].min()) > 0.4, "the speck is rebuilt in the bake"

    _, detected, _ = service._augment_retouch(cfg, baked, "s")
    assert detected is None, "IR-only config synthesizes no strokes"


def test_augment_retouch_returns_none_when_detection_off():
    service = ImageProcessor()
    cfg = replace(WorkspaceConfig(), retouch=RetouchConfig(dust_remove=False, ir_dust_remove=False))
    settings, detected, hair = service._augment_retouch(cfg, _speck_image(), "s")
    assert detected is None and hair == []
    assert settings is cfg  # untouched


def test_run_pipeline_surfaces_detected_dust_to_metrics(monkeypatch):
    service = ImageProcessor()
    monkeypatch.setattr(service.engine_cpu, "process", lambda img, settings, sh, ctx: img)

    cfg = replace(WorkspaceConfig(), retouch=RetouchConfig(dust_remove=True, dust_threshold=0.5, dust_size=4))
    _, metrics = service.run_pipeline(_speck_image(), cfg, "h", render_size_ref=512, prefer_gpu=False, readback_metrics=False)
    assert len(metrics["detected_dust_luma"]) >= 1

    cfg_off = replace(WorkspaceConfig(), retouch=RetouchConfig(dust_remove=False))
    _, metrics_off = service.run_pipeline(_speck_image(), cfg_off, "h2", render_size_ref=512, prefer_gpu=False, readback_metrics=False)
    assert "detected_dust_luma" not in metrics_off
    assert "detected_dust_ir" not in metrics_off


def _identity_uv(h, w):
    u, v = np.meshgrid(np.linspace(0, 1, w, dtype=np.float32), np.linspace(0, 1, h, dtype=np.float32))
    return np.ascontiguousarray(np.stack([u, v], axis=-1))


def test_ir_layer_identity_roundtrip_and_cache():
    overlay = CanvasOverlay(AppState())
    h, w = 20, 30
    ir = np.linspace(0, 1, h * w, dtype=np.float32).reshape(h, w)
    overlay.state.preview_ir = ir
    with overlay.state.metrics_lock:
        overlay.state.last_metrics["uv_grid"] = _identity_uv(h, w)

    img = overlay._ir_layer_qimage()
    assert img is not None and img.width() == w and img.height() == h
    # Identity remap preserves the gradient: bottom-right brightest, top-left darkest.
    assert QColor(img.pixel(w - 1, h - 1)).red() > QColor(img.pixel(0, 0)).red()
    # Same uv_grid + preview_ir identity → cache hit (no rebuild).
    assert overlay._ir_layer_qimage() is img


def test_ir_layer_follows_uv_grid_geometry():
    overlay = CanvasOverlay(AppState())
    h, w = 10, 10
    ir = np.zeros((h, w), np.float32)
    ir[:, w - 1] = 1.0  # bright right edge in source
    overlay.state.preview_ir = ir
    with overlay.state.metrics_lock:
        overlay.state.last_metrics["uv_grid"] = np.ascontiguousarray(_identity_uv(h, w)[:, ::-1])  # horizontal flip

    img = overlay._ir_layer_qimage()
    # After the flip the bright edge appears on the LEFT of the displayed frame.
    assert QColor(img.pixel(0, 5)).red() > QColor(img.pixel(w - 1, 5)).red()


def test_ir_layer_none_without_ir_or_uv():
    overlay = CanvasOverlay(AppState())
    assert overlay._ir_layer_qimage() is None  # no preview_ir
    overlay.state.preview_ir = np.zeros((8, 8), np.float32)
    assert overlay._ir_layer_qimage() is None  # no uv_grid in metrics
