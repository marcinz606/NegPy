from dataclasses import replace

import negpy_core
import numpy as np
import pytest

from negpy.domain.models import WorkspaceConfig
from negpy.features.exposure.bw_filter import BW_FILTER_WEIGHTS, collapse_matrix, collapse_weights, plane_weights
from negpy.features.exposure.models import BwFilter, ExposureConfig
from negpy.features.exposure.normalization import LogNegativeBounds, diffusion_grid, diffusion_plane
from negpy.features.process.models import ProcessConfig, ProcessMode
from negpy.infrastructure.gpu.device import GPUDevice


def _settings(bw_filter: BwFilter, source_color, mode: ProcessMode = ProcessMode.BW) -> WorkspaceConfig:
    s = WorkspaceConfig()
    return replace(
        s,
        process=replace(s.process, process_mode=mode, source_color=source_color),
        exposure=replace(s.exposure, bw_filter=bw_filter),
    )


def _color_negative() -> np.ndarray:
    rng = np.random.default_rng(4)
    h, w = 48, 64
    x = np.linspace(0.05, 0.9, w, dtype=np.float32)[None, :].repeat(h, 0)
    y = np.linspace(0.1, 0.8, h, dtype=np.float32)[:, None].repeat(w, 1)
    img = np.stack([0.2 + 0.6 * x, 0.15 + 0.5 * y, 0.1 + 0.3 * (1.0 - x) * y], axis=-1)
    return np.ascontiguousarray(img + rng.uniform(0, 0.01, img.shape).astype(np.float32))


def _render(settings: WorkspaceConfig, prefer_gpu: bool = False) -> np.ndarray:
    from negpy.services.rendering.image_processor import ImageProcessor

    processor = ImageProcessor()
    img = _color_negative()
    result, _ = processor.run_pipeline(
        img, settings, "bw-filter-src", render_size_ref=float(max(img.shape[:2])), prefer_gpu=prefer_gpu, readback_metrics=False
    )
    arr = result.readback() if hasattr(result, "readback") else result
    return np.asarray(arr)[:, :, :3].astype(np.float64)


@pytest.mark.parametrize("bw_filter", list(BwFilter))
def test_every_row_sums_to_one(bw_filter):
    assert sum(BW_FILTER_WEIGHTS[bw_filter]) == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("bw_filter", list(BwFilter))
def test_a_gray_pixel_does_not_move(bw_filter):
    gray = np.repeat(np.linspace(-0.2, 1.2, 50, dtype=np.float32)[:, None], 3, axis=1)
    out = negpy_core.matmul_3x3(gray, collapse_matrix(BW_FILTER_WEIGHTS[bw_filter]))
    np.testing.assert_allclose(out, gray, atol=1e-6)


def test_an_unknown_saved_filter_falls_back_to_none():
    assert ExposureConfig(bw_filter="deep_red_29").bw_filter is BwFilter.NONE


def test_the_filter_acts_only_on_a_color_source_in_bw():
    red = ExposureConfig(bw_filter=BwFilter.RED_25)
    none = BW_FILTER_WEIGHTS[BwFilter.NONE]
    assert collapse_weights(ProcessConfig(process_mode=ProcessMode.BW, source_color=True), red) == BW_FILTER_WEIGHTS[BwFilter.RED_25]
    assert collapse_weights(ProcessConfig(process_mode=ProcessMode.BW, source_color=False), red) == none
    assert collapse_weights(ProcessConfig(process_mode=ProcessMode.BW, source_color=None), red) == none
    assert collapse_weights(ProcessConfig(process_mode=ProcessMode.C41, source_color=True), red) == none


def test_planes_collapse_only_in_bw():
    red = ExposureConfig(bw_filter=BwFilter.RED_25)
    assert plane_weights(ProcessConfig(process_mode=ProcessMode.C41, source_color=True), red) is None
    assert plane_weights(ProcessConfig(process_mode=ProcessMode.BW, source_color=True), red) == BW_FILTER_WEIGHTS[BwFilter.RED_25]


def test_the_diffusion_plane_collapses_to_the_weighted_value():
    weights = BW_FILTER_WEIGHTS[BwFilter.ORANGE_21]
    bounds = LogNegativeBounds((-2.0, -2.0, -2.0), (0.0, 0.0, 0.0))
    plane = diffusion_plane(diffusion_grid(_color_negative(), None), bounds, weights)
    color = diffusion_plane(diffusion_grid(_color_negative(), None), bounds)
    assert plane is not None and color is not None
    np.testing.assert_allclose(plane[..., 0], color @ np.asarray(weights, dtype=np.float32), atol=1e-5)
    np.testing.assert_array_equal(plane[..., 0], plane[..., 2])


def test_a_filter_on_a_bw_source_renders_as_none():
    np.testing.assert_array_equal(_render(_settings(BwFilter.RED_25, False)), _render(_settings(BwFilter.NONE, False)))


def test_a_filter_on_a_color_source_changes_the_print():
    assert np.abs(_render(_settings(BwFilter.RED_25, True)) - _render(_settings(BwFilter.NONE, True))).max() > 0.01


@pytest.mark.skipif(not GPUDevice.get().is_available, reason="GPU not available")
@pytest.mark.parametrize("bw_filter", list(BwFilter))
def test_cpu_gpu_parity_per_filter(bw_filter):
    settings = _settings(bw_filter, True)
    settings = replace(settings, exposure=replace(settings.exposure, contrast_mask_strength=0.5, diffusion=0.4))
    cpu = _render(settings, prefer_gpu=False)
    gpu = _render(settings, prefer_gpu=True)
    assert cpu.shape == gpu.shape
    assert float(np.mean(np.abs(cpu - gpu))) < 0.01
    assert float(np.max(np.abs(cpu - gpu))) < 0.04


@pytest.mark.parametrize("prefer_gpu", [False, True])
def test_a_bw_render_clears_the_cast_axis_from_merged_metrics(prefer_gpu):
    """The curve plot reads merged metrics: a B&W render must overwrite a color render's axis."""
    from negpy.features.exposure.logic import curve_params_from_metrics
    from negpy.services.rendering.image_processor import ImageProcessor

    if prefer_gpu and not GPUDevice.get().is_available:
        pytest.skip("GPU not available")
    processor = ImageProcessor()
    img = _color_negative()
    merged: dict = {}
    for settings in (_settings(BwFilter.NONE, True, ProcessMode.C41), _settings(BwFilter.NONE, True)):
        _, metrics = processor.run_pipeline(img, settings, "axis-src", render_size_ref=64.0, prefer_gpu=prefer_gpu, readback_metrics=True)
        merged.update(metrics)
    slopes, pivots, _ = curve_params_from_metrics(settings.exposure, ProcessMode.BW, merged)
    assert len(set(slopes)) == 1 and len(set(pivots)) == 1


@pytest.mark.parametrize("prefer_gpu", [False, True])
@pytest.mark.parametrize("field", [f"{z}_{d}" for z in ("wb", "shadow", "highlight") for d in ("cyan", "magenta", "yellow")])
def test_hidden_color_filtration_leaves_a_bw_print_alone(field, prefer_gpu):
    if prefer_gpu and not GPUDevice.get().is_available:
        pytest.skip("GPU not available")
    base = _settings(BwFilter.NONE, True)
    moved = replace(base, exposure=replace(base.exposure, **{field: 0.6}))
    np.testing.assert_array_equal(_render(moved, prefer_gpu), _render(base, prefer_gpu))


def test_color_filtration_still_moves_a_color_print():
    base = _settings(BwFilter.NONE, True, ProcessMode.C41)
    moved = replace(base, exposure=replace(base.exposure, wb_cyan=0.6))
    assert np.abs(_render(moved) - _render(base)).max() > 0.01
