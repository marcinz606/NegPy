"""Dmax, Dmed and Dmin above the clear base: read in the rebate, shown on the Negative row's hover."""

from __future__ import annotations

import numpy as np

from negpy.domain.interfaces import PipelineContext
from negpy.domain.models import WorkspaceConfig
from negpy.features.exposure.normalization import LogNegativeBounds, density_above_base, measure_base_log
from negpy.features.exposure.stats import StatRow, negative_statistics
from negpy.features.process.models import ProcessMode
from negpy.infrastructure.simulated.images import film


def _negative() -> np.ndarray:
    """The simulated scan: a picture inside a clear-base band, as the camera sees it."""
    return np.ascontiguousarray(film(600, 900, 3, 24)[0], dtype=np.float32)


def test_the_base_is_the_rebate_of_the_simulated_scan():
    img = _negative()

    base = measure_base_log(img)

    assert base is not None
    assert all(v <= 0.0 for v in base)  # log10 of a transmittance
    picture = img[150:450, 225:675].reshape(-1, 3).mean(axis=1)
    assert 10 ** np.mean(base) > float(np.median(picture))  # a negative's base is its thinnest film


def _band_frame(fill: list[tuple[float, float, float]]) -> tuple[np.ndarray, np.ndarray]:
    """A 160 x 320 frame whose band (the outer 32 rows) is filled column run by column run from
    `fill`: (value, noise, share); the picture inside is mid-gray."""
    rng = np.random.default_rng(0)
    det = np.full((160, 320, 3), 0.30, dtype=np.float32)
    band = np.zeros((160, 320), bool)
    band[:32, :] = True
    band[-32:, :] = True
    x = 0
    for value, noise, share in fill:
        width = int(320 * share)
        for rows in (slice(0, 32), slice(128, 160)):
            det[rows, x : x + width] = value + rng.normal(0, noise, (32, width, 3)).astype(np.float32)
        x += width
    return det, band


def test_the_largest_flat_group_is_the_base():
    """A flat sliver of bare light, brighter than the rebate, is the smaller group."""
    from negpy.features.exposure.normalization import _base_blocks

    det, band = _band_frame([(0.84, 0.002, 0.30), (0.47, 0.004, 0.70)])

    base = _base_blocks(det, band)

    assert base is not None
    assert abs(float(np.median(base)) - 0.47) < 0.02


def test_clipped_light_is_never_the_base():
    from negpy.features.exposure.normalization import _base_blocks

    det, band = _band_frame([(0.995, 0.001, 0.70), (0.60, 0.004, 0.30)])  # holes at sensor white

    base = _base_blocks(det, band)

    assert base is not None
    assert abs(float(np.median(base)) - 0.60) < 0.02


def test_a_band_with_no_flat_film_reads_nothing():
    from negpy.features.exposure.normalization import _base_blocks

    det, band = _band_frame([(0.5, 0.2, 1.0)])  # texture everywhere, no flat block

    assert _base_blocks(det, band) is None


def test_a_render_that_reads_no_metrics_does_not_measure_the_base():
    """A splash, a thumbnail or a test strip: the Analysis panel never sees their metrics."""
    context = PipelineContext(scale_factor=1.0, original_size=(900, 600), process_mode=ProcessMode.C41, readback_metrics=False)

    _cpu_render(_negative(), WorkspaceConfig(), context)

    assert context.metrics["density_above_base"] is None


def test_an_export_does_not_measure_the_base():
    context = PipelineContext(scale_factor=1.0, original_size=(900, 600), process_mode=ProcessMode.C41, cache_stages=False)

    _cpu_render(_negative(), WorkspaceConfig(), context)

    assert context.metrics["density_above_base"] is None


def test_a_frame_with_no_film_box_has_no_base():
    assert measure_base_log(np.full((600, 900, 3), 0.5, dtype=np.float32)) is None


def test_the_three_densities_read_above_the_base():
    base = (-0.1, -0.1, -0.1)
    bounds = LogNegativeBounds(floors=(-1.5, -1.5, -1.5), ceils=(-0.3, -0.3, -0.3))
    prefiltered = np.full((50, 3), -0.8, dtype=np.float32)

    dmin, dmed, dmax = density_above_base(base, bounds, prefiltered, None)

    assert abs(dmin - 0.2) < 1e-6
    assert abs(dmed - 0.7) < 1e-6
    assert abs(dmax - 1.4) < 1e-6
    assert density_above_base(None, bounds, prefiltered, None) is None


def _cpu_render(img, cfg, context, key="density"):
    from negpy.services.rendering.engine import DarkroomEngine

    engine = DarkroomEngine()
    engine.process(img, cfg, key, context=context)
    return engine


def test_the_base_stage_records_the_metric():
    context = PipelineContext(scale_factor=1.0, original_size=(900, 600), process_mode=ProcessMode.C41)

    _cpu_render(_negative(), WorkspaceConfig(), context)

    dmin, dmed, dmax = context.metrics["density_above_base"]
    assert -0.1 < dmin < dmed < dmax
    assert dmax > 0.5


def test_a_process_change_reuses_the_base(monkeypatch):
    """The base depends on the source and its geometry only, so a Crosstalk drag pays nothing."""
    from dataclasses import replace

    from negpy.services.rendering import base_read
    from negpy.services.rendering.engine import DarkroomEngine

    calls: list[int] = []
    real = base_read.measure_base_log
    monkeypatch.setattr(base_read, "measure_base_log", lambda image: calls.append(1) or real(image))
    img, cfg = _negative(), WorkspaceConfig()
    engine = DarkroomEngine()
    first = PipelineContext(scale_factor=1.0, original_size=(900, 600), process_mode=ProcessMode.C41)
    engine.process(img, cfg, "reuse", context=first)
    second = PipelineContext(scale_factor=1.0, original_size=(900, 600), process_mode=ProcessMode.C41)
    engine.process(
        img, replace(cfg, process=replace(cfg.process, analysis_buffer=cfg.process.analysis_buffer + 0.05)), "reuse", context=second
    )

    third = PipelineContext(scale_factor=1.0, original_size=(900, 600), process_mode=ProcessMode.C41)
    engine.process(
        img, replace(cfg, geometry=replace(cfg.geometry, crop_rect=(0.1, 0.1, 0.9, 0.9), fine_rotation=2.0)), "reuse", context=third
    )

    assert calls == [1]  # a Process change, a crop edit and a warp all reuse the base
    assert second.metrics["density_above_base"] is not None
    assert third.metrics["density_above_base"] is not None


def test_the_negative_row_carries_the_densities_on_hover():
    rows = negative_statistics(1.26, 0.5, 0.0, 0.0, density_above_base=(0.22, 0.71, 1.42), base_source="read")

    assert rows[0].name == "Negative"
    assert rows[0].value == "Δ1.26"
    assert rows[0].detail.startswith("Normal negative. Above the clear base read from the rebate: Dmax 1.42 · Dmed 0.71 · Dmin 0.22")
    assert "Film Base tool pins it" in rows[0].detail


def test_a_picked_base_is_named_as_such_without_the_reader_caveat():
    rows = negative_statistics(1.26, 0.5, 0.0, 0.0, density_above_base=(0.22, 0.71, 1.42), base_source="picked")

    assert "Above the film base set on the roll: Dmax 1.42" in rows[0].detail
    assert "Film Base tool pins it" not in rows[0].detail


def test_a_scan_with_no_base_says_so():
    rows = negative_statistics(1.26, 0.5, 0.0, 0.0, density_above_base=None)

    assert "Dmax, Dmed and Dmin need the roll's film base." in rows[0].detail


def test_the_hover_shows_the_detail_under_the_fixed_text(qapp):
    from negpy.desktop.view.widgets.stats import NegativeStatsWidget

    widget = NegativeStatsWidget()
    widget.update_stats([StatRow("Negative", "1.26 · normal", detail="Above the clear base: Dmax 1.42 · Dmed 0.71 · Dmin 0.22.")])

    tip = widget._values[0].toolTip()
    assert tip.index("How contrasty the negative is") < tip.index("Dmax 1.42")


def test_both_engines_record_the_same_densities():
    from negpy.infrastructure.gpu.device import GPUDevice

    if not GPUDevice.get().is_available:
        import pytest

        pytest.skip("GPU not available")
    from negpy.services.rendering.engine import DarkroomEngine
    from negpy.services.rendering.gpu_engine import GPUEngine

    img = _negative()
    cfg = WorkspaceConfig()
    engine = GPUEngine()
    try:
        _tex, gpu_metrics = engine.process_to_texture(
            img, cfg, scale_factor=1.0, apply_layout=False, readback_metrics=True, source_hash="density", analysis_source_hash="density"
        )
    finally:
        engine.destroy_all()
    context = PipelineContext(scale_factor=1.0, original_size=(900, 600), process_mode=ProcessMode.C41)
    DarkroomEngine().process(img, cfg, "density", context=context)

    cpu, gpu = context.metrics["density_above_base"], gpu_metrics["density_above_base"]
    assert gpu is not None and cpu is not None
    assert all(abs(c - g) < 0.02 for c, g in zip(cpu, gpu)), (cpu, gpu)


def test_a_release_after_a_drag_fills_the_densities():
    """A drag renders without metrics; the release that reads them must not inherit the drag's None."""
    from negpy.infrastructure.gpu.device import GPUDevice

    if not GPUDevice.get().is_available:
        import pytest

        pytest.skip("GPU not available")
    from negpy.services.rendering.gpu_engine import GPUEngine

    img = _negative()
    cfg = WorkspaceConfig()
    engine = GPUEngine()
    try:
        _tex, drag = engine.process_to_texture(
            img, cfg, scale_factor=1.0, apply_layout=False, readback_metrics=False, source_hash="drag", analysis_source_hash="drag"
        )
        _tex, release = engine.process_to_texture(
            img, cfg, scale_factor=1.0, apply_layout=False, readback_metrics=True, source_hash="drag", analysis_source_hash="drag"
        )
    finally:
        engine.destroy_all()

    assert drag["density_above_base"] is None
    assert release["density_above_base"] is not None


def test_a_pass_with_no_source_identity_still_reports_the_densities():
    """The tiled reference pass carries no analysis hash; it measures per render instead of caching."""
    from negpy.infrastructure.gpu.device import GPUDevice

    if not GPUDevice.get().is_available:
        import pytest

        pytest.skip("GPU not available")
    from negpy.services.rendering.gpu_engine import GPUEngine

    engine = GPUEngine()
    try:
        _tex, metrics = engine.process_to_texture(
            _negative(), WorkspaceConfig(), scale_factor=1.0, apply_layout=False, readback_metrics=True
        )
    finally:
        engine.destroy_all()

    assert metrics["density_above_base"] is not None


def test_a_creative_slider_render_keeps_the_densities():
    """The second render fills its analysis from the cache and skips the meter grid; the
    densities ride that cache instead of dropping to None."""
    from dataclasses import replace

    from negpy.infrastructure.gpu.device import GPUDevice

    if not GPUDevice.get().is_available:
        import pytest

        pytest.skip("GPU not available")
    from negpy.services.rendering.gpu_engine import GPUEngine

    img = _negative()
    cfg = WorkspaceConfig()
    engine = GPUEngine()
    try:
        _tex, first = engine.process_to_texture(
            img, cfg, scale_factor=1.0, apply_layout=False, readback_metrics=True, source_hash="cache", analysis_source_hash="cache"
        )
        darker = replace(cfg, exposure=replace(cfg.exposure, density=cfg.exposure.density + 0.3))
        _tex, second = engine.process_to_texture(
            img, darker, scale_factor=1.0, apply_layout=False, readback_metrics=True, source_hash="cache", analysis_source_hash="cache"
        )
    finally:
        engine.destroy_all()

    assert first["density_above_base"] is not None
    assert second["density_above_base"] == first["density_above_base"]
