from dataclasses import replace

import numpy as np
import pytest

from negpy.domain.models import WorkspaceConfig
from negpy.features.process.sensor import (
    apply_sensor_correction,
    build_sensor_matrix,
    effective_sensor_matrix,
    measure_capture,
    sensor_token,
    unmix_block_reason,
)
from negpy.features.process.models import ProcessMode
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
    # linear_raw is explicit: the unmix is gated on it (test_effective_matrix_needs_linear_raw).
    off = replace(WorkspaceConfig().process, linear_raw=True)
    on = replace(off, sensor_matrix=(1, 0, 0, 0, 1, 0, 0, 0, 1))
    on2 = replace(off, sensor_matrix=(1, -0.1, 0, 0, 1, 0, 0, 0, 1))
    assert sensor_token(off) == ""
    assert sensor_token(on) != sensor_token(on2) != ""


def test_effective_matrix_needs_linear_raw():
    matrix = (1.0, -0.1, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    base = WorkspaceConfig().process
    assert effective_sensor_matrix(base) is None  # no matrix baked
    on = replace(base, sensor_matrix=matrix, linear_raw=True)
    assert effective_sensor_matrix(on) == matrix
    off = replace(on, linear_raw=False)
    assert effective_sensor_matrix(off) is None
    # The token must agree, or the render cache would key two identical renders apart.
    assert sensor_token(off) == ""


def test_a_transparency_refuses_the_unmix():
    """The matrix only means anything for a capture made under narrowband light, and
    narrowband is not used for slides. A profile is sticky, so it follows a negative rig's
    settings onto a slide unasked — which is the bug: it corrected for a light the frame
    was never shot under, and the panel that would have shown it was hidden."""
    matrix = (1.0, -0.1, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    on = replace(WorkspaceConfig().process, sensor_matrix=matrix, linear_raw=True)
    assert effective_sensor_matrix(on) == matrix

    for normalize in (True, False):
        slide = replace(on, process_mode=ProcessMode.E6, e6_normalize=normalize)
        assert effective_sensor_matrix(slide) is None, f"e6_normalize={normalize}"
        assert unmix_block_reason(slide) == "transparency"
        # The token must agree, or the render cache would key two identical renders apart.
        assert sensor_token(slide) == ""

    # Baked, not cleared: switching the frame back to a negative restores the selection.
    assert effective_sensor_matrix(replace(on, process_mode=ProcessMode.C41)) == matrix


def test_the_block_reason_names_the_nearer_cause():
    """Both blocks can hold at once; the panel shows one hint, and a slide is the reason
    the user cannot act on — turning Linear RAW on would not bring the unmix back."""
    both = replace(WorkspaceConfig().process, linear_raw=False, process_mode=ProcessMode.E6)
    assert unmix_block_reason(both) == "transparency"
    assert unmix_block_reason(replace(both, process_mode=ProcessMode.C41)) == "linear_raw"
    assert unmix_block_reason(replace(both, process_mode=ProcessMode.C41, linear_raw=True)) == ""


def test_camera_wb_basis_breaks_the_unmix():
    """Why the Linear RAW gate exists: a WB gain does not commute with the unmix."""
    s_true = np.array([[1.0, 0.06, 0.02], [0.08, 1.0, 0.22], [0.02, 0.18, 1.0]])
    wb = np.diag([2.1, 1.0, 1.55])  # as-shot camera multipliers (use_camera_wb=True)

    def cols(m):
        return (tuple(m[:, 0]), tuple(m[:, 1]), tuple(m[:, 2]))

    def residual(matrix, buffer_basis):
        out = np.array(matrix).reshape(3, 3) @ buffer_basis
        norm = out / np.diag(out)[:, None]  # a leftover diagonal is absorbed downstream
        return np.abs(norm - np.diag(np.diag(norm))).max()

    neutral = build_sensor_matrix(*cols(s_true))
    assert residual(neutral, s_true) < 1e-9  # matched basis: exact
    # Neutral-calibrated matrix on a camera-WB buffer keeps most of the 22% leak.
    assert residual(neutral, wb @ s_true) > 0.1
    # Column normalization makes the matrix transform by similarity, so calibrating
    # in the same WB basis would be exact — the gate, not the math, is the blocker.
    assert residual(build_sensor_matrix(*cols(wb @ s_true)), wb @ s_true) < 1e-9


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
    ip._precorrect_key = None
    ip._precorrect_value = None
    ip._prepare_gate = __import__("threading").Lock()
    ip._augment_retouch = lambda settings, img, key: (settings, None, [])
    ip._ir_bake = lambda img, ir, settings, key: (img, None, None, None)
    ip._is_flat = lambda settings: False

    img = np.full((8, 8, 3), 0.5, dtype=np.float32)
    matrix = (1.0, -0.1, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    cfg = WorkspaceConfig()
    cfg = replace(cfg, process=replace(cfg.process, sensor_matrix=matrix, linear_raw=True))

    ip.run_pipeline(img, cfg, "h", render_size_ref=float(APP_CONFIG.preview_render_size), prefer_gpu=False)
    assert calls == [matrix]

    calls.clear()
    triplet_cfg = replace(cfg, rgbscan=RgbScanConfig(enabled=True, green_path="g", blue_path="b"))
    ip.run_pipeline(img, triplet_cfg, "h", render_size_ref=float(APP_CONFIG.preview_render_size), prefer_gpu=False)
    assert calls == []

    calls.clear()
    ip.run_pipeline(img, cfg, "h", render_size_ref=float(APP_CONFIG.preview_render_size), prefer_gpu=False, skip_flatfield=True)
    assert calls == []  # skip_flatfield buffers were corrected at decode already

    calls.clear()
    wb_cfg = replace(cfg, process=replace(cfg.process, linear_raw=False))
    ip.run_pipeline(img, wb_cfg, "h", render_size_ref=float(APP_CONFIG.preview_render_size), prefer_gpu=False)
    assert calls == [None]  # camera-WB buffer: wrong basis for a neutral-WB matrix
