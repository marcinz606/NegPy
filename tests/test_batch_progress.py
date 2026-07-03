from unittest.mock import MagicMock

from PyQt6.QtCore import Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication

from negpy.desktop.view.widgets.progress_dialog import ProgressDialog
from negpy.desktop.workers.export import ExportTask, ExportWorker
from negpy.desktop.workers.render import NormalizationTask, NormalizationWorker
from negpy.kernel.system.config import DEFAULT_WORKSPACE_CONFIG


def _export_task(name: str) -> ExportTask:
    return ExportTask(
        file_info={"name": name, "path": f"/tmp/{name}", "hash": name},
        params=DEFAULT_WORKSPACE_CONFIG,
        export_settings=DEFAULT_WORKSPACE_CONFIG.export,
    )


def test_progress_dialog_updates_and_abort() -> None:
    dlg = ProgressDialog()
    aborted: list[bool] = []
    dlg.abort_requested.connect(lambda: aborted.append(True))

    dlg.start("Exporting", abortable=True)
    assert dlg.isVisible()
    assert dlg._abort.isVisible()

    dlg.set_progress(2, 5, "frame.cr2")
    QApplication.processEvents()
    assert dlg._count.text() == "2/5"
    assert dlg._file_label.text() == "frame.cr2"
    assert dlg._bar.value() == 2

    QTest.mouseClick(dlg._abort, Qt.MouseButton.LeftButton)
    assert aborted == [True]
    assert not dlg._abort.isEnabled()

    dlg.finish()
    assert not dlg.isVisible()


def test_progress_dialog_hides_abort_when_not_abortable() -> None:
    dlg = ProgressDialog()
    dlg.start("Generating thumbnails", abortable=False)
    assert not dlg._abort.isVisible()


def test_export_worker_cancel_stops_batch_keeps_partial() -> None:
    worker = ExportWorker()
    proc = MagicMock()

    def _process_export(*_a, **_k):
        worker.cancel()  # abort requested mid-batch, after first file
        return (b"", None)  # empty bits => nothing written

    proc.process_export.side_effect = _process_export
    worker._processor = proc

    finished: list[bool] = []
    cancelled: list[bool] = []
    worker.finished.connect(lambda: finished.append(True))
    worker.cancelled.connect(lambda: cancelled.append(True))

    worker.run_batch([_export_task("a.cr2"), _export_task("b.cr2")])

    assert cancelled == [True]
    assert finished == []
    assert proc.process_export.call_count == 1  # second task skipped


def test_export_batch_keeps_source_cache_for_consecutive_same_file(tmp_path) -> None:
    from unittest.mock import call

    from negpy.desktop.workers.export import _same_decode_source
    from negpy.domain.models import preset_from_export_config

    worker = ExportWorker()
    proc = MagicMock()
    proc.process_export.return_value = (b"bits", None)
    worker._processor = proc

    # Real batches carry ExportPresets (SAME_AS_SOURCE output), like the controller builds.
    preset = preset_from_export_config(DEFAULT_WORKSPACE_CONFIG.export)

    def _task(name: str) -> ExportTask:
        return ExportTask(
            file_info={"name": name, "path": str(tmp_path / name), "hash": name},
            params=DEFAULT_WORKSPACE_CONFIG,
            export_settings=preset,
        )

    # a.cr2 exported twice (multi-format preset), then b.cr2.
    worker.run_batch([_task("a.cr2"), _task("a.cr2"), _task("b.cr2")])

    assert proc.cleanup.call_args_list == [
        call(release_source_cache=False, collect=False),  # next task = same source
        call(release_source_cache=True, collect=False),
        call(release_source_cache=True, collect=False),  # last task
    ]
    assert proc.process_export.call_count == 3

    # _same_decode_source mirrors the _load_source_f32 cache key fields.
    from dataclasses import replace

    a, b = _task("a.cr2"), _task("b.cr2")
    assert _same_decode_source(a, _task("a.cr2"))
    assert not _same_decode_source(a, b)
    flat = replace(DEFAULT_WORKSPACE_CONFIG, flatfield=replace(DEFAULT_WORKSPACE_CONFIG.flatfield, apply=True, reference_path="/f.dng"))
    a_flat = ExportTask(file_info=a.file_info, params=flat, export_settings=preset)
    assert not _same_decode_source(a, a_flat)


def test_normalization_worker_cancel_emits_cancelled_no_baseline() -> None:
    preview = MagicMock()
    repo = MagicMock()
    repo.load_file_settings.return_value = None
    worker = NormalizationWorker(preview, repo)

    def _load(*_a, **_k):
        worker.cancel()
        raise RuntimeError("aborted")

    preview.load_linear_preview.side_effect = _load

    finished: list[tuple] = []
    cancelled: list[bool] = []
    worker.finished.connect(lambda f, c: finished.append((f, c)))
    worker.cancelled.connect(lambda: cancelled.append(True))

    task = NormalizationTask(
        files=[{"name": "a.cr2", "path": "/tmp/a.cr2", "hash": "a"}],
        workspace_color_space="sRGB",
        override_analysis_buffer=0.0,
        override_luma_range_clip=0.0,
        override_color_range_clip=0.0,
    )
    worker.process(task)

    assert cancelled == [True]
    assert finished == []
