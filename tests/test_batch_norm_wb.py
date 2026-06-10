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
