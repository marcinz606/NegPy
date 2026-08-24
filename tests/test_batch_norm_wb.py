"""Regression: batch normalization must decode each file with the same white
balance the render path uses (use_camera_wb = not linear_raw). Analysing in a
different WB space shifts per-channel bounds and produces a color cast (the
roll-average "everything goes red" bug).
"""

from dataclasses import replace

import numpy as np

from negpy.desktop.workers.render import NormalizationInput, NormalizationTask, NormalizationWorker
from negpy.domain.models import WorkspaceConfig


class _FakePreviewService:
    """Records the use_camera_wb flag each file is decoded with."""

    def __init__(self) -> None:
        self.calls: dict[str, bool] = {}

    def load_linear_preview(self, path, color_space, use_camera_wb, full_resolution, file_hash):
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
    worker.finished.connect(lambda f, c: captured.append((f, c)))

    worker.process(task)

    # use_camera_wb must equal (not linear_raw) for each file.
    assert preview.calls["h_cam"] is True  # linear_raw=False -> camera WB (matches render)
    assert preview.calls["h_flat"] is False  # linear_raw=True  -> flat WB

    # Sanity: analysis completed and emitted floors/ceils.
    assert len(captured) == 1
    floors, ceils = captured[0]
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
