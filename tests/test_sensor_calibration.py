from dataclasses import replace

import numpy as np
import pytest

from negpy.domain.models import WorkspaceConfig
from negpy.features.process.sensor import (
    apply_sensor_correction,
    build_sensor_matrix,
    measure_capture,
    sensor_token,
)
from negpy.features.rgbscan.models import RgbScanConfig, is_rgb_triplet
from negpy.kernel.system.config import APP_CONFIG

_RGB_R = (0.9, 0.1, 0.03)
_RGB_G = (0.05, 0.5, 0.15)  # green leaks 30% into blue after own-channel normalization
_RGB_B = (0.04, 0.3, 1.0)


def _s_norm():
    s = np.column_stack([_RGB_R, _RGB_G, _RGB_B])
    return s / np.diag(s)


def test_pure_captures_give_identity():
    m = np.array(build_sensor_matrix((1, 0, 0), (0, 1, 0), (0, 0, 1))).reshape(3, 3)
    assert np.allclose(m, np.eye(3), atol=1e-12)


def test_matrix_inverts_normalized_mixing():
    m = np.array(build_sensor_matrix(_RGB_R, _RGB_G, _RGB_B)).reshape(3, 3)
    assert np.allclose(m @ _s_norm(), np.eye(3), atol=1e-9)


def test_matrix_invariant_to_per_capture_exposure():
    scaled = build_sensor_matrix(tuple(v * 3.7 for v in _RGB_R), tuple(v * 0.4 for v in _RGB_G), tuple(v * 1.9 for v in _RGB_B))
    assert np.allclose(scaled, build_sensor_matrix(_RGB_R, _RGB_G, _RGB_B), atol=1e-9)


def test_rejects_zero_own_channel_and_singular():
    with pytest.raises(ValueError):
        build_sensor_matrix((0.0, 0.1, 0.03), _RGB_G, _RGB_B)
    with pytest.raises(ValueError):
        build_sensor_matrix((1, 1, 1), (1, 1, 1), (1, 1, 1))


def test_apply_none_is_same_object():
    img = np.random.default_rng(0).random((8, 8, 3)).astype(np.float32)
    assert apply_sensor_correction(img, None) is img


def test_apply_unmixes_clips_and_keeps_float32():
    clean = np.array([[[0.6, 0.2, 0.1]]], dtype=np.float32)
    mixed = np.einsum("ck,hwk->hwc", _s_norm().astype(np.float32), clean)
    out = apply_sensor_correction(mixed, build_sensor_matrix(_RGB_R, _RGB_G, _RGB_B))
    assert np.allclose(out[0, 0], clean[0, 0], atol=1e-5)
    assert out.dtype == np.float32
    assert np.all(out >= 0.0)


def test_measure_capture_reads_central_crop_only():
    img = np.zeros((100, 100, 3), dtype=np.float32)
    img[:, :] = (0.8, 0.3, 0.05)
    img[:10, :, :] = 5.0  # poisoned border a centre crop must ignore
    img[:, -10:, :] = 5.0
    assert measure_capture(img) == pytest.approx((0.8, 0.3, 0.05), abs=1e-5)


def test_sensor_token_tracks_matrix():
    off = WorkspaceConfig().process
    on = replace(off, sensor_matrix=(1, 0, 0, 0, 1, 0, 0, 0, 1))
    on2 = replace(off, sensor_matrix=(1, -0.1, 0, 0, 1, 0, 0, 0, 1))
    assert sensor_token(off) == ""
    assert sensor_token(on) != sensor_token(on2) != ""


def test_is_rgb_triplet_truth_table():
    assert not is_rgb_triplet(RgbScanConfig())
    assert not is_rgb_triplet(RgbScanConfig(enabled=True))
    assert not is_rgb_triplet(RgbScanConfig(enabled=True, green_path="g"))
    assert not is_rgb_triplet(RgbScanConfig(enabled=False, green_path="g", blue_path="b"))
    assert is_rgb_triplet(RgbScanConfig(enabled=True, green_path="g", blue_path="b"))


def test_config_roundtrips_sensor_fields():
    matrix = (1.0, -0.09, -0.01, 0.05, 1.12, -0.33, 0.11, -0.34, 1.1)
    cfg = WorkspaceConfig()
    cfg = replace(cfg, process=replace(cfg.process, sensor_matrix=matrix, sensor_profile="My Sensor"))
    restored = WorkspaceConfig.from_flat_dict(cfg.to_dict())
    assert restored.process.sensor_matrix == matrix
    assert restored.process.sensor_profile == "My Sensor"
    # JSON round-trip delivers lists; __post_init__ must coerce back to tuple.
    from_list = WorkspaceConfig.from_flat_dict({**cfg.to_dict(), "sensor_matrix": list(matrix)})
    assert from_list.process.sensor_matrix == matrix


def test_run_pipeline_gates_triplets(monkeypatch):
    from negpy.services.rendering import image_processor as ip_mod

    calls = []

    def _recorder(img, matrix):
        calls.append(matrix)
        return img

    monkeypatch.setattr(ip_mod, "apply_sensor_correction", _recorder)
    ip = ip_mod.ImageProcessor.__new__(ip_mod.ImageProcessor)
    ip.engine_gpu = None
    ip.engine_cpu = __import__("unittest.mock", fromlist=["MagicMock"]).MagicMock()
    ip.engine_cpu.process.return_value = np.zeros((8, 8, 3), dtype=np.float32)
    ip._augment_retouch = lambda settings, img, key: (settings, None, [])
    ip._ir_bake = lambda img, ir, settings, key: (img, None, None, None)
    ip._is_flat = lambda settings: False

    img = np.full((8, 8, 3), 0.5, dtype=np.float32)
    matrix = (1.0, -0.1, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    cfg = WorkspaceConfig()
    cfg = replace(cfg, process=replace(cfg.process, sensor_matrix=matrix))

    ip.run_pipeline(img, cfg, "h", render_size_ref=float(APP_CONFIG.preview_render_size), prefer_gpu=False)
    assert calls == [matrix]

    calls.clear()
    triplet_cfg = replace(cfg, rgbscan=RgbScanConfig(enabled=True, green_path="g", blue_path="b"))
    ip.run_pipeline(img, triplet_cfg, "h", render_size_ref=float(APP_CONFIG.preview_render_size), prefer_gpu=False)
    assert calls == []

    calls.clear()
    ip.run_pipeline(img, cfg, "h", render_size_ref=float(APP_CONFIG.preview_render_size), prefer_gpu=False, skip_flatfield=True)
    assert calls == []  # skip_flatfield buffers were corrected at decode already
