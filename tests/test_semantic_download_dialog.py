from unittest.mock import MagicMock

from negpy.desktop.view.widgets import semantic_download_dialog as dialog_module
from negpy.desktop.view.widgets.semantic_download_dialog import ClipDownloadDialog


def test_the_download_progress_reads_in_megabytes(qapp):
    dlg = ClipDownloadDialog()

    dlg._on_progress(1_048_576, 150 * 1_048_576)

    assert dlg.bar.maximum() == 150 * 1_048_576
    assert "1 MB of 150 MB" in dlg.status.text()


def test_clicking_download_starts_a_worker_and_disables_the_button(qapp, monkeypatch):
    started = []
    monkeypatch.setattr(dialog_module.ClipDownloadWorker, "start", lambda self: started.append(self))

    dlg = ClipDownloadDialog()
    dlg.download_button.click()

    assert started
    assert not dlg.download_button.isEnabled()
    assert not dlg.bar.isHidden()


def test_a_completed_download_accepts_the_dialog(qapp):
    dlg = ClipDownloadDialog()
    accepted = []
    dlg.accept = lambda: accepted.append(True)

    dlg._on_ready()

    assert accepted == [True]


def test_a_failed_download_reports_the_message_and_re_enables_the_button(qapp):
    dlg = ClipDownloadDialog()
    dlg.download_button.setEnabled(False)
    dlg.bar.setVisible(True)

    dlg._on_failed("Download failed: connection reset")

    assert dlg.status.text() == "Download failed: connection reset"
    assert dlg.download_button.isEnabled()
    assert dlg.bar.isHidden()


def test_cancel_stops_a_running_worker_before_rejecting(qapp):
    dlg = ClipDownloadDialog()
    dlg._worker = MagicMock()
    dlg._worker.isRunning.return_value = True

    dlg.reject()

    dlg._worker.cancel.assert_called_once_with()


def test_reject_with_no_worker_started_does_not_raise(qapp):
    dlg = ClipDownloadDialog()
    dlg.reject()  # no worker yet -- Cancel before Download is clicked
