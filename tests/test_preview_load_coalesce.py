"""Preview decode coalescing during rapid filmstrip navigation."""

from unittest.mock import MagicMock

import numpy as np

from negpy.desktop.workers.render import PreviewLoadTask, PreviewLoadWorker


def _task(file_path: str, load_seq: int) -> PreviewLoadTask:
    return PreviewLoadTask(
        file_path=file_path,
        workspace_color_space="Adobe RGB",
        use_camera_wb=True,
        use_splash=False,
        load_seq=load_seq,
    )


def test_superseded_preview_decode_skips_io(qapp) -> None:
    from negpy.services.rendering.preview_manager import PreviewManager

    worker = PreviewLoadWorker(MagicMock(spec=PreviewManager))
    worker.set_latest_preview_seq(2)

    finished = MagicMock()
    worker.finished.connect(finished)

    worker.process(_task("/tmp/old.dng", 1))

    finished.assert_not_called()


def test_current_preview_decode_emits_finished(qapp) -> None:
    from negpy.services.rendering.preview_manager import PreviewManager

    preview_service = MagicMock(spec=PreviewManager)
    preview_service.load_linear_preview.return_value = (
        np.zeros((4, 4, 3), dtype=np.float32),
        (4, 4),
        {"color_space": "Adobe RGB"},
    )

    worker = PreviewLoadWorker(preview_service)
    worker.set_latest_preview_seq(3)

    finished = MagicMock()
    worker.finished.connect(finished)

    worker.process(_task("/tmp/current.dng", 3))

    finished.assert_called_once()
    preview_service.load_linear_preview.assert_called_once()
