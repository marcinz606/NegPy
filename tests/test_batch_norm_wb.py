"""Regression: batch normalization must decode each file with the same white
balance the render path uses (use_camera_wb = not linear_raw). Analysing in a
different WB space shifts per-channel bounds and produces a color cast (the
roll-average "everything goes red" bug).
"""

from dataclasses import replace

import numpy as np

from negpy.desktop.workers.render import NormalizationInput, NormalizationTask, NormalizationWorker
from negpy.domain.models import WorkspaceConfig
from negpy.features.lens.models import LensCorrections


class _FakePreviewService:
    """Records the use_camera_wb flag each file is decoded with."""

    def __init__(self) -> None:
        self.calls: dict[str, bool] = {}

    def load_linear_preview(
        self,
        path,
        color_space,
        use_camera_wb,
        full_resolution,
        file_hash,
        demosaic="Auto",
        positive_source=False,
        highlight_mode=0,
        bake_camera_wb=False,
        lens_corrections=LensCorrections(),
        lens_flatfield=None,
    ):
        self.calls[file_hash] = use_camera_wb
        raw = np.full((8, 8, 3), 0.5, dtype=np.float32)
        return raw, (8, 8), {}


def _frames(settings: dict[str, WorkspaceConfig]) -> list[NormalizationInput]:
    return [NormalizationInput(file_info={"path": f"/{h}.dng", "hash": h, "name": h}, config=cfg) for h, cfg in settings.items()]


def test_batch_analysis_decodes_in_render_wb(qapp):
    base = WorkspaceConfig()
    settings = {
        "h_cam": replace(base, process=replace(base.process, linear_raw=False)),
        "h_flat": replace(base, process=replace(base.process, linear_raw=True)),
    }
    preview = _FakePreviewService()
    worker = NormalizationWorker(preview)

    task = NormalizationTask(
        frames=_frames(settings),
        workspace_color_space="sRGB",
        override_analysis_buffer=base.process.analysis_buffer,
        override_luma_range_clip=base.process.luma_range_clip,
        override_color_range_clip=base.process.color_range_clip,
    )

    captured: list[tuple] = []
    worker.finished.connect(lambda f, c, o: captured.append((f, c, o)))

    worker.process(task)

    # use_camera_wb must equal (not linear_raw) for each file.
    assert preview.calls["h_cam"] is True  # linear_raw=False -> camera WB (matches render)
    assert preview.calls["h_flat"] is False  # linear_raw=True  -> flat WB

    # Sanity: analysis completed and emitted floors/ceils.
    assert len(captured) == 1
    floors, ceils, _outliers = captured[0]
    assert len(floors) == 3 and len(ceils) == 3


def test_batch_analysis_applies_roll_wide_buffer_and_luma_range(qapp, monkeypatch):
    """The current image's analysis_buffer / luma_range_clip override every file's own
    saved value, so the whole roll is analyzed with one setting before averaging."""
    import negpy.features.exposure.normalization as norm_mod

    captured_kwargs: list[dict] = []

    class _Bounds:
        floors = (0.0, 0.0, 0.0)
        ceils = (1.0, 1.0, 1.0)

    def _spy(transformed, **kwargs):
        captured_kwargs.append(kwargs)
        return _Bounds()

    monkeypatch.setattr(norm_mod, "analyze_log_exposure_bounds", _spy)

    base = WorkspaceConfig()
    # Files carry DIFFERENT saved buffer/luma bounds — must be ignored in favor of override.
    settings = {
        "h1": replace(base, process=replace(base.process, analysis_buffer=0.20, luma_range_clip=5.0)),
        "h2": replace(base, process=replace(base.process, analysis_buffer=0.01, luma_range_clip=-2.0)),
    }
    worker = NormalizationWorker(_FakePreviewService())

    task = NormalizationTask(
        frames=_frames(settings),
        workspace_color_space="sRGB",
        override_analysis_buffer=0.12,
        override_luma_range_clip=3.5,
        override_color_range_clip=0.0,
    )

    worker.process(task)

    assert len(captured_kwargs) == 2
    for kw in captured_kwargs:
        assert kw["analysis_buffer"] == 0.12
        assert kw["percentile_clip"] == 3.5


def test_batch_analysis_meters_each_frames_own_region(qapp, monkeypatch):
    """A frame with a freehand region meters that region, as its render does; the rest
    keep the roll-wide buffer."""
    import negpy.features.exposure.normalization as norm_mod

    captured_kwargs: list[dict] = []

    class _Bounds:
        floors = (0.0, 0.0, 0.0)
        ceils = (1.0, 1.0, 1.0)

    def _spy(transformed, **kwargs):
        captured_kwargs.append(kwargs)
        return _Bounds()

    monkeypatch.setattr(norm_mod, "analyze_log_exposure_bounds", _spy)

    base = WorkspaceConfig()
    settings = {
        "h_region": replace(base, process=replace(base.process, analysis_rect=(0.25, 0.25, 0.75, 0.75))),
        "h_plain": base,
    }
    task = NormalizationTask(
        frames=_frames(settings),
        workspace_color_space="sRGB",
        override_analysis_buffer=0.12,
        override_luma_range_clip=0.0,
        override_color_range_clip=0.0,
    )

    NormalizationWorker(_FakePreviewService()).process(task)

    by_roi = {kw["roi"]: kw["analysis_buffer"] for kw in captured_kwargs}
    assert by_roi == {(2, 6, 2, 6): 0.0, None: 0.12}


class _VaryingPreviewService(_FakePreviewService):
    """Decodes each file to a flat color driven by `fills`, so a mocked analysis
    function can derive deterministic, per-file bounds straight from the pixel data."""

    def __init__(self, fills: dict[str, tuple[float, float, float]]) -> None:
        super().__init__()
        self.fills = fills

    def load_linear_preview(
        self,
        path,
        color_space,
        use_camera_wb,
        full_resolution,
        file_hash,
        demosaic="Auto",
        positive_source=False,
        highlight_mode=0,
        bake_camera_wb=False,
        lens_corrections=LensCorrections(),
        lens_flatfield=None,
    ):
        self.calls[file_hash] = use_camera_wb
        raw = np.full((8, 8, 3), self.fills[file_hash], dtype=np.float32)
        return raw, (8, 8), {}


def _bounds_from_pixels(transformed, **_kwargs):
    """A fake analyze_log_exposure_bounds: floors/ceils bracket the fake decode's own
    flat fill color, so each frame's synthetic pixel value drives its own bounds."""

    class _Bounds:
        pass

    mean = transformed.reshape(-1, 3).mean(axis=0)
    b = _Bounds()
    b.floors = tuple(mean - 0.1)
    b.ceils = tuple(mean + 0.1)
    return b


def test_batch_analysis_flags_a_frame_whose_own_bounds_are_outside_the_pooled_band(qapp, monkeypatch):
    """A frame far from the roll on one channel gets trimmed out of that channel's
    average -- which is exactly why the baseline about to be forced onto it is a bad
    match. That mismatch must be reported back as an outlier."""
    import negpy.features.exposure.normalization as norm_mod

    monkeypatch.setattr(norm_mod, "analyze_log_exposure_bounds", _bounds_from_pixels)

    # 5 normal frames (flat 0.5 in every channel) + 1 outlier, blue channel only.
    fills = {f"h{i}": (0.5, 0.5, 0.5) for i in range(5)}
    fills["h_outlier"] = (0.5, 0.5, 0.9)
    base = WorkspaceConfig()
    settings = {h: base for h in fills}
    preview = _VaryingPreviewService(fills)
    worker = NormalizationWorker(preview)

    task = NormalizationTask(
        frames=_frames(settings),
        workspace_color_space="sRGB",
        override_analysis_buffer=base.process.analysis_buffer,
        override_luma_range_clip=base.process.luma_range_clip,
        override_color_range_clip=base.process.color_range_clip,
    )

    captured: list[tuple] = []
    worker.finished.connect(lambda f, c, o: captured.append((f, c, o)))

    worker.process(task)

    assert len(captured) == 1
    _floors, _ceils, outliers = captured[0]
    assert outliers == ["/h_outlier.dng"]


def test_batch_analysis_reports_no_outliers_when_the_roll_is_uniform(qapp, monkeypatch):
    import negpy.features.exposure.normalization as norm_mod

    monkeypatch.setattr(norm_mod, "analyze_log_exposure_bounds", _bounds_from_pixels)

    fills = {f"h{i}": (0.5, 0.5, 0.5) for i in range(6)}
    base = WorkspaceConfig()
    preview = _VaryingPreviewService(fills)
    worker = NormalizationWorker(preview)

    task = NormalizationTask(
        frames=_frames({h: base for h in fills}),
        workspace_color_space="sRGB",
        override_analysis_buffer=base.process.analysis_buffer,
        override_luma_range_clip=base.process.luma_range_clip,
        override_color_range_clip=base.process.color_range_clip,
    )

    captured: list[tuple] = []
    worker.finished.connect(lambda f, c, o: captured.append((f, c, o)))

    worker.process(task)

    assert captured[0][2] == []


def test_batch_analysis_decodes_a_triplet_as_a_composite(qapp):
    """A triplet's bounds must come from the assembled three-band source. Measured on the
    lone red exposure, green and blue hold sensor leak alone, and the roll baseline then
    puts every frame's real G/B above their ceils — a solid red roll."""
    from negpy.features.rgbscan.models import RgbScanConfig

    class _TripletPreviewService(_FakePreviewService):
        def __init__(self) -> None:
            super().__init__()
            self.merged: list[tuple[str, str, str]] = []

        def load_linear_preview_rgb(self, red_path, rgbscan, color_space, **kw):
            self.merged.append((red_path, rgbscan.green_path, rgbscan.blue_path))
            return np.full((8, 8, 3), 0.5, dtype=np.float32), (8, 8), {}

    base = WorkspaceConfig()
    triplet = replace(base, rgbscan=RgbScanConfig(enabled=True, green_path="/g.dng", blue_path="/b.dng"))
    preview = _TripletPreviewService()
    worker = NormalizationWorker(preview)

    worker.process(
        NormalizationTask(
            frames=[NormalizationInput(file_info={"path": "/r.dng", "hash": "h_r", "name": "r"}, config=triplet)],
            workspace_color_space="sRGB",
            override_analysis_buffer=base.process.analysis_buffer,
            override_luma_range_clip=base.process.luma_range_clip,
            override_color_range_clip=base.process.color_range_clip,
        )
    )

    assert preview.merged == [("/r.dng", "/g.dng", "/b.dng")]
    assert preview.calls == {}  # never the lone red exposure
