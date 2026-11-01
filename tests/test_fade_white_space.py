"""A camera RAW's fade white: tagged with the decode space it was measured in, and the zero point the
restoration scales density from."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from negpy.desktop.controller import AppController
from negpy.domain.interfaces import PipelineContext
from negpy.domain.models import WorkspaceConfig
from negpy.features.exposure.normalization import prefilter_log_grid, unmix_log_image
from negpy.features.process.capture_color import source_cam_xyz
from negpy.features.process.fade import (
    EstimateMethod,
    estimate_fade,
    fade_white_space,
    fade_white_stale,
    fade_white_target,
    resolve_fade_matrix,
    unmix_with_fade,
)
from negpy.features.process.models import ProcessConfig, ProcessMode
from negpy.features.transparency.processor import TransparencyBaseProcessor
from negpy.infrastructure.gpu.device import GPUDevice
from tests.test_fade_estimation import _synthetic_neutral_slide

_CAMERA = [[0.6722, -0.0635, -0.0963], [-0.4287, 1.246, 0.2046], [-0.0908, 0.2268, 0.5762]]
_WB = [2.0, 1.0, 1.5]
_SPACE = fade_white_space(_CAMERA, _WB)
_FADE = dict(fade_ratio_g=0.8, fade_ratio_b=0.6)
# Clear film two stops under the decoder's white, as a camera captures a slide.
_RAW_WHITE = (-0.6, -0.55, -0.65)


def _slide(**kw) -> ProcessConfig:
    return ProcessConfig(process_mode=ProcessMode.E6, **kw)


def _white(white=_RAW_WHITE, space=_SPACE) -> dict:
    return {"fade_white_r": white[0], "fade_white_g": white[1], "fade_white_b": white[2], "fade_white_space": space}


def test_the_space_names_a_camera_and_its_white_balance():
    assert _SPACE and _SPACE == fade_white_space(_CAMERA, list(_WB))
    assert fade_white_space(_CAMERA, [1.8, 1.0, 1.5]) != _SPACE
    assert fade_white_space(_CAMERA, None) != _SPACE
    assert fade_white_space(None, _WB) == ""
    assert fade_white_space(source_cam_xyz(np.eye(3)), None) == ""


def test_a_camera_raw_scales_density_above_its_white_and_keeps_the_white():
    process = _slide(**_FADE, **_white())
    matrix = unmix_with_fade(process, ProcessMode.E6, _SPACE)
    density = np.array([-0.3, -0.24, -0.18])
    pixel = (np.asarray(_RAW_WHITE) + density)[None, None, :].astype(np.float32)
    expected = np.asarray(_RAW_WHITE) + resolve_fade_matrix(1.0, 0.8, 0.6) @ density
    np.testing.assert_allclose(unmix_log_image(pixel, matrix)[0, 0], expected, atol=1e-6)
    white = np.asarray(_RAW_WHITE, dtype=np.float32)[None, None, :]
    np.testing.assert_allclose(unmix_log_image(white, matrix)[0, 0], _RAW_WHITE, atol=1e-6)
    assert fade_white_target(process, _SPACE) is None


def test_the_render_is_the_same_at_any_exposure():
    """A stop more light moves every channel's log by the same amount, white included."""
    process = _slide(**_FADE, **_white())
    shifted = _slide(**_FADE, **_white(tuple(w + 0.3 for w in _RAW_WHITE)))
    pixel = np.array([[[-0.9, -0.8, -0.75]]], dtype=np.float32)
    out = unmix_log_image(pixel, unmix_with_fade(process, ProcessMode.E6, _SPACE))
    brighter = unmix_log_image(pixel + 0.3, unmix_with_fade(shifted, ProcessMode.E6, _SPACE))
    np.testing.assert_allclose(brighter - out, 0.3, atol=1e-6)


def test_remove_stain_still_maps_the_white_to_neutral():
    process = _slide(**_FADE, fade_white_anchor=True, **_white())
    white = np.asarray(_RAW_WHITE, dtype=np.float32)[None, None, :]
    np.testing.assert_allclose(unmix_log_image(white, unmix_with_fade(process, ProcessMode.E6, _SPACE))[0, 0], -0.55, atol=1e-6)
    assert fade_white_target(process, _SPACE) == pytest.approx(-0.55)


@pytest.mark.parametrize("stored, render", [("", _SPACE), (_SPACE, ""), (_SPACE, fade_white_space(_CAMERA, None))])
def test_a_white_from_another_space_does_not_apply(stored, render):
    anchored = _slide(**_FADE, fade_white_anchor=True, **_white(space=stored))
    np.testing.assert_array_equal(unmix_with_fade(anchored, ProcessMode.E6, render), resolve_fade_matrix(1.0, 0.8, 0.6))
    assert fade_white_target(anchored, render) is None


def test_a_file_without_a_camera_matrix_keeps_the_full_scale_zero():
    """A JPEG or scanner TIFF has its white at full scale, so only Remove Stain moves the zero."""
    process = _slide(**_FADE, **_white(space=""))
    np.testing.assert_array_equal(unmix_with_fade(process, ProcessMode.E6, ""), resolve_fade_matrix(1.0, 0.8, 0.6))


def test_the_white_goes_stale_only_where_it_acts():
    assert fade_white_stale(_slide(**_FADE), _SPACE)
    assert fade_white_stale(_slide(fade_white_anchor=True), _SPACE)
    assert fade_white_stale(_slide(**_FADE, **_white(space="old")), _SPACE)
    assert not fade_white_stale(_slide(**_FADE, **_white()), _SPACE)
    assert not fade_white_stale(_slide(), _SPACE)
    assert not fade_white_stale(_slide(**_FADE), "")
    assert not fade_white_stale(ProcessConfig(process_mode=ProcessMode.C41, **_FADE), _SPACE)


def test_a_camera_raw_estimate_ignores_the_exposure():
    grid = prefilter_log_grid(_synthetic_neutral_slide(0.8, 0.6), None, 0.0)
    under = grid - np.float32(0.3)
    for method in EstimateMethod:
        a = estimate_fade(grid, camera_raw=True)[method]
        b = estimate_fade(under, camera_raw=True)[method]
        assert (a.green, a.blue) == pytest.approx((b.green, b.blue), abs=1e-3)


def test_the_processor_reads_the_space_from_the_capture():
    """The CPU render tags the white with the context's camera, so a stale white falls back to no zero shift."""
    img = np.random.default_rng(1).uniform(0.05, 0.25, (32, 48, 3)).astype(np.float32)

    def restored(process):
        ctx = PipelineContext(
            original_size=img.shape[:2], scale_factor=1.0, process_mode=ProcessMode.E6, cam_xyz=_CAMERA, camera_wb=_WB, wants_uv_grid=False
        )
        TransparencyBaseProcessor(process).process(img, ctx)
        return ctx.metrics["normalized_log"]

    plain = restored(_slide(linear_raw=True, **_FADE))
    stale = restored(_slide(linear_raw=True, **_FADE, **_white(space="old")))
    tagged = restored(_slide(linear_raw=True, **_FADE, **_white()))
    np.testing.assert_array_equal(stale, plain)
    assert not np.allclose(tagged, plain)


def test_a_stale_white_is_measured_again_without_a_roll_lock(monkeypatch):
    import negpy.desktop.workers.chart_fit as chart_fit

    grid = np.full((8, 8, 3), -0.7, dtype=np.float32)
    grid[:2] = (-0.5, -0.45, -0.6)
    monkeypatch.setattr(chart_fit, "fade_estimate_grid", lambda source: grid)
    applied = []
    cfg = WorkspaceConfig()
    fake = SimpleNamespace(
        state=SimpleNamespace(config=replace(cfg, process=_slide(**_FADE))),
        _effective_cam_xyz=lambda: (_CAMERA, _WB),
        chart_fit_source=lambda: object(),
        apply_config=lambda config, persist: applied.append((config.process, persist)),
    )
    AppController._refresh_stale_fade_white(fake)
    ((process, persist),) = applied
    assert persist and process.fade_white_space == _SPACE
    assert process.fade_white_r == pytest.approx(-0.5, abs=0.01) and process.fade_white_b == pytest.approx(-0.6, abs=0.01)


@pytest.mark.skipif(not GPUDevice.get().is_available, reason="GPU not available")
@pytest.mark.parametrize("anchor", [False, True])
def test_gpu_matches_cpu_on_a_camera_raw(anchor):
    from negpy.services.rendering.image_processor import ImageProcessor

    processor = ImageProcessor()
    if processor.engine_gpu is None:
        pytest.skip("GPU engine not initialised")
    cfg = WorkspaceConfig()
    settings = replace(cfg, process=_slide(linear_raw=True, fade_white_anchor=anchor, **_FADE, **_white()))
    img = np.random.default_rng(2).uniform(0.03, 0.25, (96, 128, 3)).astype(np.float32)

    def render(prefer_gpu):
        out, _ = processor.run_pipeline(
            img.copy(),
            settings,
            "raw-white-parity",
            render_size_ref=128.0,
            prefer_gpu=prefer_gpu,
            readback_metrics=False,
            cam_xyz=_CAMERA,
            camera_wb=_WB,
        )
        arr = out.readback() if hasattr(out, "readback") else out
        return np.asarray(arr)[:, :, :3].astype(np.float64)

    cpu, gpu = render(False), render(True)
    assert float(np.mean(np.abs(cpu - gpu))) < 0.005
    assert float(np.max(np.abs(cpu - gpu))) < 0.04
