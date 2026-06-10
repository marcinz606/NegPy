"""Regression: batch normalization must decode each file with the same white
balance the render path uses (use_camera_wb = not linear_raw). Analysing in a
different WB space shifts per-channel bounds and produces a color cast (the
roll-average "everything goes red" bug).
"""

from dataclasses import replace

import numpy as np

from negpy.desktop.workers.render import NormalizationTask, NormalizationWorker
from negpy.domain.models import WorkspaceConfig


class _FakePreviewService:
    """Records the use_camera_wb flag each file is decoded with."""

    def __init__(self) -> None:
        self.calls: dict[str, bool] = {}

    def load_linear_preview(self, path, color_space, use_camera_wb, full_resolution, file_hash):
        self.calls[file_hash] = use_camera_wb
        raw = np.full((8, 8, 3), 0.5, dtype=np.float32)
        return raw, (8, 8), {}


class _FakeRepo:
    def __init__(self, settings: dict[str, WorkspaceConfig]) -> None:
        self._settings = settings

    def load_file_settings(self, file_hash):
        return self._settings.get(file_hash)


def test_batch_analysis_decodes_in_render_wb(qapp):
    base = WorkspaceConfig()
    settings = {
        "h_cam": replace(base, exposure=replace(base.exposure, linear_raw=False)),
        "h_flat": replace(base, exposure=replace(base.exposure, linear_raw=True)),
    }
    preview = _FakePreviewService()
    worker = NormalizationWorker(preview, _FakeRepo(settings))

    task = NormalizationTask(
        files=[
            {"path": "/a.dng", "hash": "h_cam", "name": "a"},
            {"path": "/b.dng", "hash": "h_flat", "name": "b"},
        ],
        workspace_color_space="sRGB",
        override_analysis_buffer=base.process.analysis_buffer,
        override_drange_clip=base.process.drange_clip,
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


def test_batch_analysis_applies_roll_wide_buffer_and_drange(qapp, monkeypatch):
    """The current image's analysis_buffer / drange_clip override every file's own
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
    # Files carry DIFFERENT saved buffer/d-range — must be ignored in favor of override.
    settings = {
        "h1": replace(base, process=replace(base.process, analysis_buffer=0.20, drange_clip=5.0)),
        "h2": replace(base, process=replace(base.process, analysis_buffer=0.01, drange_clip=-2.0)),
    }
    worker = NormalizationWorker(_FakePreviewService(), _FakeRepo(settings))

    task = NormalizationTask(
        files=[
            {"path": "/a.dng", "hash": "h1", "name": "a"},
            {"path": "/b.dng", "hash": "h2", "name": "b"},
        ],
        workspace_color_space="sRGB",
        override_analysis_buffer=0.12,
        override_drange_clip=3.5,
    )

    worker.process(task)

    assert len(captured_kwargs) == 2
    for kw in captured_kwargs:
        assert kw["analysis_buffer"] == 0.12
        assert kw["percentile_clip"] == 3.5
