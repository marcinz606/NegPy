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


def _warm_task(file_path: str, prefetch_gen: int) -> PreviewLoadTask:
    return PreviewLoadTask(
        file_path=file_path,
        workspace_color_space="Adobe RGB",
        use_camera_wb=True,
        use_splash=False,
        for_cache_warm=True,
        prefetch_gen=prefetch_gen,
    )


def test_stale_prefetch_task_skips_io(qapp) -> None:
    from negpy.services.rendering.preview_manager import PreviewManager

    preview_service = MagicMock(spec=PreviewManager)
    worker = PreviewLoadWorker(preview_service)
    worker.set_latest_prefetch_gen(5)  # user has navigated on

    worker.process(_warm_task("/tmp/old_neighbor.dng", prefetch_gen=3))

    preview_service.load_linear_preview.assert_not_called()


def test_current_prefetch_task_warms_cache(qapp) -> None:
    from negpy.services.rendering.preview_manager import PreviewManager

    preview_service = MagicMock(spec=PreviewManager)
    worker = PreviewLoadWorker(preview_service)
    worker.set_latest_prefetch_gen(5)

    worker.process(_warm_task("/tmp/neighbor.dng", prefetch_gen=5))

    preview_service.load_linear_preview.assert_called_once()


def test_splash_emitted_before_linear_decode_completes(qapp) -> None:
    """The embedded splash paints via the on_splash callback (fired mid-decode), not
    only after load_splash_and_linear returns."""
    from negpy.services.rendering.preview_manager import PreviewManager

    events: list[str] = []
    splash_buf = np.zeros((2, 2, 3), dtype=np.float32)

    def fake_load(*args, on_splash=None, **kwargs):
        if on_splash is not None:
            on_splash(splash_buf, (2, 2))  # early paint, before the decode "finishes"
        events.append("linear_done")
        return (splash_buf, (2, 2)), (np.zeros((4, 4, 3), np.float32), (4, 4), {"color_space": "Adobe RGB"})

    preview_service = MagicMock(spec=PreviewManager)
    preview_service.load_splash_and_linear.side_effect = fake_load

    worker = PreviewLoadWorker(preview_service)
    worker.set_latest_preview_seq(1)
    worker.splash.connect(lambda *_: events.append("splash"))
    worker.finished.connect(lambda *_: events.append("finished"))

    task = PreviewLoadTask(
        file_path="/tmp/cold.dng",
        workspace_color_space="Adobe RGB",
        use_camera_wb=True,
        use_splash=True,
        load_seq=1,
    )
    worker.process(task)

    assert events == ["splash", "linear_done", "finished"]
