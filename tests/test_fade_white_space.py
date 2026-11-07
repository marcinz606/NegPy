"""A camera RAW's fade white: tagged with the decode space it was measured in, and the zero point the
restoration scales density from."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from fade_crossover_scene import estimate_fade
from negpy.desktop.controller import AppController
from negpy.domain.interfaces import PipelineContext
from negpy.domain.models import WorkspaceConfig
from negpy.features.exposure.normalization import effective_crosstalk_matrix, prefilter_log_grid, unmix_log_image
from negpy.features.process.capture_color import source_cam_xyz, wb_only_cam_xyz
from negpy.features.process.fade import (
    fade_white_space,
    fade_white_stale,
    fade_white_target,
    render_white_space,
    resolve_fade_matrix,
    fade_density,
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


def test_an_input_icc_keeps_the_space():
    """The white is read before the primaries, so a camera profile that swaps them leaves it valid."""
    assert fade_white_space(wb_only_cam_xyz(_CAMERA), _WB) == _SPACE


def test_a_camera_raw_scales_density_above_a_neutral_level_at_its_white():
    """Without Remove Stain the anchor is the white's cleanest channel on all three, so the white keeps its color."""
    process = _slide(**_FADE, **_white())
    matrix = fade_density(process, ProcessMode.E6, _SPACE).matrix
    level = max(_RAW_WHITE)
    density = np.array([-0.3, -0.24, -0.18])
    pixel = (level + density)[None, None, :].astype(np.float32)
    expected = level + resolve_fade_matrix(1.0, 0.8, 0.6) @ density
    np.testing.assert_allclose(unmix_log_image(pixel, matrix)[0, 0], expected, atol=1e-6)
    anchor = np.full((1, 1, 3), level, dtype=np.float32)
    np.testing.assert_allclose(unmix_log_image(anchor, matrix)[0, 0], level, atol=1e-6)
    assert fade_white_target(process, _SPACE) is None


def test_estimate_then_render_keeps_grays_neutral_with_a_colored_white():
    """A slide with no clear film: its white is the brightest picture color, here a blue sky. The
    survivals Estimate finds against the render's own anchor restore the grays to neutral."""
    rng = np.random.default_rng(3)
    true_log = np.empty((120, 180, 3))
    true_log[:] = np.linspace(-2.2, -0.35, 180)[None, :, None]
    true_log[:30, 20:150] = np.log10((0.15, 0.32, 0.55))
    true_log += rng.normal(0, 0.01, true_log.shape)
    faded = true_log @ np.linalg.inv(resolve_fade_matrix(0.7, 1.4, 1.25)).T - 0.6
    grid = faded[30:].astype(np.float32)
    whole = faded.astype(np.float32)
    for remove_stain in (False, True):
        est = estimate_fade(whole, white_anchor=remove_stain, camera_raw=True)
        stored = _white(est.white)
        process = _slide(fade_ratio_r=est.red, fade_ratio_g=est.green, fade_ratio_b=est.blue, fade_white_anchor=remove_stain, **stored)
        out = unmix_log_image(grid, fade_density(process, ProcessMode.E6, _SPACE).matrix)[20:80, 5:175].reshape(-1, 3)
        cast = out - out[:, 1:2]
        assert np.all(np.abs(cast.mean(axis=0)) < 0.03), (remove_stain, cast.mean(axis=0))


def test_the_render_is_the_same_at_any_exposure():
    """A stop more light moves every channel's log by the same amount, white included."""
    process = _slide(**_FADE, **_white())
    shifted = _slide(**_FADE, **_white(tuple(w + 0.3 for w in _RAW_WHITE)))
    pixel = np.array([[[-0.9, -0.8, -0.75]]], dtype=np.float32)
    out = unmix_log_image(pixel, fade_density(process, ProcessMode.E6, _SPACE).matrix)
    brighter = unmix_log_image(pixel + 0.3, fade_density(shifted, ProcessMode.E6, _SPACE).matrix)
    np.testing.assert_allclose(brighter - out, 0.3, atol=1e-6)


def test_remove_stain_still_maps_the_white_to_neutral():
    process = _slide(**_FADE, fade_white_anchor=True, **_white())
    white = np.asarray(_RAW_WHITE, dtype=np.float32)[None, None, :]
    np.testing.assert_allclose(unmix_log_image(white, fade_density(process, ProcessMode.E6, _SPACE).matrix)[0, 0], -0.55, atol=1e-6)
    assert fade_white_target(process, _SPACE) == pytest.approx(-0.55)


@pytest.mark.parametrize("stored, render", [("", _SPACE), (_SPACE, ""), (_SPACE, fade_white_space(_CAMERA, None))])
def test_a_white_from_another_space_does_not_apply(stored, render):
    anchored = _slide(**_FADE, fade_white_anchor=True, **_white(space=stored))
    np.testing.assert_array_equal(fade_density(anchored, ProcessMode.E6, render).matrix, resolve_fade_matrix(1.0, 0.8, 0.6))
    assert fade_white_target(anchored, render) is None


def test_a_file_without_a_camera_matrix_keeps_the_full_scale_zero():
    """A JPEG or scanner TIFF has its white at full scale, so only Remove Stain moves the zero."""
    process = _slide(**_FADE, **_white(space=""))
    np.testing.assert_array_equal(fade_density(process, ProcessMode.E6, "").matrix, resolve_fade_matrix(1.0, 0.8, 0.6))


def test_remove_stain_on_a_scanner_file_restores_its_highlights_too():
    """A scanner file's white is a percentile of its clear film: the light above it is film, not light
    past the film, so the restoration acts on it."""
    img = np.random.default_rng(5).uniform(0.03, 1.0, (40, 60, 3)).astype(np.float32)
    config = _slide(fade_white_anchor=True, **_FADE, **_white((-0.05, -0.08, -0.12), space=""))
    assert fade_density(config, ProcessMode.E6, "").anchor is None
    ctx = PipelineContext(original_size=img.shape[:2], scale_factor=1.0, process_mode=ProcessMode.E6)
    TransparencyBaseProcessor(config).process(img, ctx)
    b = ctx.metrics["log_bounds"]
    restored = ctx.metrics["normalized_log"] * (np.asarray(b.ceils) - np.asarray(b.floors)) + np.asarray(b.floors)
    expected = unmix_log_image(np.log10(img), fade_density(config, ProcessMode.E6, "").matrix)
    np.testing.assert_allclose(restored, expected, atol=1e-4)


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
    a = estimate_fade(grid, camera_raw=True)
    b = estimate_fade(under, camera_raw=True)
    assert (a.green, a.blue) == pytest.approx((b.green, b.blue), abs=1e-3)


def _restored(process, img):
    ctx = PipelineContext(
        original_size=img.shape[:2], scale_factor=1.0, process_mode=ProcessMode.E6, cam_xyz=_CAMERA, camera_wb=_WB, wants_uv_grid=False
    )
    TransparencyBaseProcessor(process).process(img, ctx)
    return ctx.metrics["normalized_log"], ctx


@pytest.mark.parametrize("stored", ["", "old"])
def test_a_render_measures_a_stale_white_for_itself(stored):
    """Export and thumbnails read no desktop state, so the render measures the white it needs."""
    img = np.random.default_rng(1).uniform(0.05, 0.25, (32, 48, 3)).astype(np.float32)
    stale, ctx = _restored(_slide(linear_raw=True, **_FADE, **_white(space=stored)), img)
    r, g, b, space = ctx.metrics["fade_white"]
    assert space == _SPACE
    tagged, ctx = _restored(_slide(linear_raw=True, **_FADE, **_white((r, g, b))), img)
    np.testing.assert_array_equal(stale, tagged)
    assert "fade_white" not in ctx.metrics


def test_a_measured_white_is_stored_with_the_bounds():
    """The render's white joins the measured-bounds write: no render, no undo step, no roll lock."""
    cfg = WorkspaceConfig()
    fake = SimpleNamespace(state=SimpleNamespace(config=replace(cfg, process=_slide(**_FADE))), _effective_cam_xyz=lambda: (_CAMERA, _WB))
    measured = (-0.5, -0.45, -0.6, _SPACE)
    assert AppController._measured_fade_white(fake, {"fade_white": measured}) == dict(_white(measured[:3]))
    # Another frame's white, or one already stored, is left alone.
    assert AppController._measured_fade_white(fake, {"fade_white": (-0.5, -0.45, -0.6, "other")}) == {}
    fake.state.config = replace(cfg, process=_slide(**_FADE, **_white()))
    assert AppController._measured_fade_white(fake, {"fade_white": measured}) == {}


def _crosstalk(**kw) -> ProcessConfig:
    unmix = ((1.0, -0.12, -0.03), (-0.08, 1.0, -0.15), (-0.02, -0.1, 1.0))
    return _slide(crosstalk_matrix=unmix, crosstalk_strength=1.0, crosstalk_process=ProcessMode.E6, **kw)


def _crosstalk_space(process: ProcessConfig) -> str:
    return render_white_space(process, ProcessMode.E6, _CAMERA, _WB)


def test_crosstalk_alone_takes_no_white():
    """Crosstalk keeps a neutral neutral, so it renders the same whatever white is stored."""
    space = _crosstalk_space(_crosstalk())
    assert not fade_white_stale(_crosstalk(), space)
    assert fade_density(_crosstalk(**_white(space=space)), ProcessMode.E6, space).anchor is None


def test_a_crosstalk_change_makes_the_white_stale():
    """w is read after the unmix, so a new unmix names a new space."""
    space = _crosstalk_space(_crosstalk())
    assert space not in ("", _SPACE)
    stronger = _crosstalk(**_FADE, **_white(space=space))
    stronger = replace(stronger, crosstalk_strength=0.5)
    assert fade_white_stale(stronger, _crosstalk_space(stronger))


@pytest.mark.parametrize("remove_stain", [False, True])
def test_with_crosstalk_the_white_maps_to_its_target(remove_stain):
    """w is read after the crosstalk unmix, so the restoration scales about the point the unmix maps to w."""
    space = _crosstalk_space(_crosstalk())
    process = _crosstalk(fade_white_anchor=remove_stain, **_FADE, **_white(space=space))
    matrix, anchor = fade_density(process, ProcessMode.E6, space)
    unmix = effective_crosstalk_matrix(process, ProcessMode.E6)
    white = np.asarray(_RAW_WHITE) if remove_stain else np.full(3, max(_RAW_WHITE))
    np.testing.assert_allclose(unmix @ np.asarray(anchor), white, atol=1e-9)
    target = fade_white_target(process, space) if remove_stain else max(_RAW_WHITE)
    np.testing.assert_allclose(matrix[:, :3] @ np.asarray(anchor) + matrix[:, 3], target, atol=1e-9)


@pytest.mark.skipif(not GPUDevice.get().is_available, reason="GPU not available")
@pytest.mark.parametrize("stored", [_SPACE, "old"])
@pytest.mark.parametrize("anchor", [False, True])
def test_gpu_matches_cpu_on_a_camera_raw(anchor, stored):
    from negpy.services.rendering.image_processor import ImageProcessor

    processor = ImageProcessor()
    if processor.engine_gpu is None:
        pytest.skip("GPU engine not initialised")
    cfg = WorkspaceConfig()
    settings = replace(cfg, process=_slide(linear_raw=True, fade_white_anchor=anchor, **_FADE, **_white(space=stored)))
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


@pytest.mark.parametrize("positive", [False, True])
@pytest.mark.parametrize("remove_stain", [False, True])
def test_a_reconstructed_highlight_stays_white_through_the_fade(positive, remove_stain):
    """Highlight Reconstruction decodes a blown highlight as neutral at full scale. It is light past the
    film, above the fade's anchor in every channel, so it passes the density matrix unscaled."""
    img = np.random.default_rng(4).uniform(0.03, 0.4, (40, 60, 3)).astype(np.float32)
    img[:10, :20] = 1.0
    white = (-0.42, -0.45, -0.5)
    config = _slide(
        positive_source=positive, linear_raw=False, highlight_reconstruction=5, fade_white_anchor=remove_stain, **_FADE, **_white(white)
    )
    ctx = PipelineContext(original_size=img.shape[:2], scale_factor=1.0, process_mode=ProcessMode.E6, cam_xyz=_CAMERA, camera_wb=_WB)
    TransparencyBaseProcessor(config).process(img, ctx)
    b = ctx.metrics["log_bounds"]
    restored = ctx.metrics["normalized_log"] * (np.asarray(b.ceils) - np.asarray(b.floors)) + np.asarray(b.floors)
    np.testing.assert_allclose(restored[:10, :20].reshape(-1, 3), 0.0, atol=1e-4)
    # Below the anchor the fade still acts.
    assert not np.allclose(restored[20:, 30:], np.log10(img[20:, 30:]), atol=1e-3)


def test_estimate_reads_the_grid_without_the_stored_white():
    """The grid a new white is read from takes no fade and no white, so a stored white does not move it."""
    from negpy.desktop.workers import chart_fit

    img = (_synthetic_neutral_slide(0.8, 0.6, size=96) * np.float32((0.22, 0.25, 0.18))).astype(np.float32)

    def grid(**white):
        cfg = replace(WorkspaceConfig(), process=_slide(linear_raw=True, fade_white_anchor=True, **_FADE, **white))
        return chart_fit.fade_estimate_grid(chart_fit.ChartSource(img, cfg, "h", _CAMERA, _WB, (), 1))

    stored = grid(**_white(white=(-0.9, -0.95, -1.0)))
    assert stored is not None
    np.testing.assert_array_equal(stored, grid(**_white(white=(0.0, 0.0, 0.0))))


def test_an_export_renders_a_stale_white_as_the_measured_one():
    """The exposure stage reads the white the base stage measured, as Remove Stain's target sets Cast Removal."""
    from negpy.services.rendering.image_processor import ImageProcessor

    # A faded gray card under a stained white, two stops below full scale.
    img = (_synthetic_neutral_slide(0.8, 0.6, size=96) * np.float32((0.22, 0.25, 0.18))).astype(np.float32)
    base = WorkspaceConfig()
    stale = replace(base, process=_slide(linear_raw=True, fade_white_anchor=True, **_FADE, **_white(space="old")))
    stale = replace(stale, exposure=replace(stale.exposure, cast_removal_strength=1.0))

    def render(settings):
        return ImageProcessor(use_gpu=False).run_pipeline(
            img.copy(), settings, "stale-export", render_size_ref=96.0, prefer_gpu=False, cache_stages=False, cam_xyz=_CAMERA, camera_wb=_WB
        )

    out, metrics = render(stale)
    r, g, b, _space = metrics["fade_white"]
    stored, _ = render(replace(stale, process=replace(stale.process, **_white((r, g, b)))))
    np.testing.assert_array_equal(np.asarray(out), np.asarray(stored))
    assert metrics["neutral_axis_refs"] is not None


@pytest.mark.skipif(not GPUDevice.get().is_available, reason="GPU not available")
def test_the_gpu_measures_a_stale_white_once_per_frame(monkeypatch):
    """Drag ticks render with the white still stale until a settled render stores it."""
    import negpy.services.rendering.gpu_engine as gpu_engine

    calls = []
    real = gpu_engine.resolve_fade_white
    monkeypatch.setattr(gpu_engine, "resolve_fade_white", lambda *a: calls.append(1) or real(*a))
    engine = gpu_engine.GPUEngine()
    try:
        img = np.random.default_rng(2).uniform(0.03, 0.25, (64, 96, 3)).astype(np.float32)
        settings = replace(WorkspaceConfig(), process=_slide(linear_raw=True, **_FADE, **_white(space="old")))
        for strength in (0.0, 0.5):
            moved = replace(settings, exposure=replace(settings.exposure, cast_removal_strength=strength))
            _, metrics = engine.process_to_texture(
                img, moved, scale_factor=1.0, apply_layout=False, cam_xyz=_CAMERA, camera_wb=_WB, analysis_source_hash="frame"
            )
            assert metrics["fade_white"][3] == _SPACE
        assert len(calls) == 1
    finally:
        engine.destroy_all()


@pytest.mark.parametrize("level", [2, 5])
def test_highlight_reconstruction_makes_a_stored_white_stale(level):
    """Highlight Reconstruction rebuilds the clipped pixels the white is read from, so a new level measures it again."""
    process = _slide(**_FADE, **_white(space=render_white_space(_slide(), ProcessMode.E6, _CAMERA, _WB)))
    assert not fade_white_stale(process, render_white_space(process, ProcessMode.E6, _CAMERA, _WB))
    rebuilt = replace(process, highlight_reconstruction=level)
    assert fade_white_stale(rebuilt, render_white_space(rebuilt, ProcessMode.E6, _CAMERA, _WB))


def test_a_negative_and_a_scanner_source_take_no_white_space():
    negative = ProcessConfig(process_mode=ProcessMode.C41, highlight_reconstruction=5)
    assert render_white_space(negative, ProcessMode.C41, _CAMERA, _WB) == ""
    assert render_white_space(_slide(highlight_reconstruction=5), ProcessMode.E6, None, None) == ""
