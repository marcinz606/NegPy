import threading
import time
from pathlib import Path

import pytest

from negpy.desktop.view.widgets import update_dialog as dialog_module
from negpy.desktop.view.widgets.update_dialog import UpdateDialog
from negpy.kernel.system.updater import UpdateError, UpdateInfo


def _info(**overrides) -> UpdateInfo:
    fields = dict(
        version="9.9.9",
        notes="## What's new\n\n- A thing",
        page_url="https://example.test/release",
        asset_name="NegPy-9.9.9-x86_64.AppImage",
        download_url="https://example.test/asset",
        size=2 * 1_048_576,
    )
    return UpdateInfo(**{**fields, **overrides})


@pytest.fixture
def installable(monkeypatch):
    monkeypatch.setattr("negpy.kernel.system.updater.install_kind", lambda: "appimage")


def test_the_notes_and_the_size_are_shown(qapp, installable):
    dlg = UpdateDialog(_info())

    assert "What's new" in dlg.notes.toMarkdown()
    assert "2 MB" in dlg.subtitle.text()
    assert dlg.install_button.text() == "Install Update"


def test_a_release_without_a_build_for_this_install_only_offers_the_page(qapp, monkeypatch):
    monkeypatch.setattr("negpy.kernel.system.updater.install_kind", lambda: "")
    opened = []
    monkeypatch.setattr(dialog_module.QDesktopServices, "openUrl", lambda url: opened.append(url.toString()))

    dlg = UpdateDialog(_info(download_url="", asset_name=""))
    assert dlg.install_button.text() == "Open Releases Page"

    dlg.install_button.click()

    assert opened == ["https://example.test/release"]


def test_the_download_progress_reads_in_megabytes(qapp, installable):
    dlg = UpdateDialog(_info())

    dlg._on_progress(1_048_576, 4 * 1_048_576)

    assert dlg.bar.maximum() == 4 * 1_048_576
    assert "1 MB of 4 MB" in dlg.status.text()


def test_a_downloaded_asset_is_handed_to_the_installer_and_the_app_closes(qapp, installable, monkeypatch, tmp_path):
    applied = []
    quit_calls = []
    monkeypatch.setattr(dialog_module, "apply_update", lambda path, info: applied.append((path, info.version)))
    monkeypatch.setattr(dialog_module.QApplication, "closeAllWindows", lambda *a: quit_calls.append("close"))
    monkeypatch.setattr(dialog_module.QApplication, "quit", lambda *a: quit_calls.append("quit"))
    dlg = UpdateDialog(_info())

    dlg._on_ready(tmp_path / "NegPy-9.9.9-x86_64.AppImage")

    assert applied == [(tmp_path / "NegPy-9.9.9-x86_64.AppImage", "9.9.9")]
    assert quit_calls == ["close", "quit"]


def test_a_refused_installer_leaves_the_app_running(qapp, installable, monkeypatch):
    quit_calls = []

    def refuse(path, info):
        raise UpdateError("Windows refused to start the installer")

    monkeypatch.setattr(dialog_module, "apply_update", refuse)
    monkeypatch.setattr(dialog_module.QApplication, "quit", lambda *a: quit_calls.append("quit"))
    dlg = UpdateDialog(_info())

    dlg._on_ready(Path("/tmp/NegPy.AppImage"))

    assert not quit_calls
    assert "refused" in dlg.status.text()
    assert dlg.install_button.isEnabled()  # the user can try again


def test_the_module_owns_the_check_thread_not_the_caller(qapp, monkeypatch):
    """A QThread destroyed while it still runs aborts the process, and a stalled
    network call outlives the panel that asked for it."""
    started, release = threading.Event(), threading.Event()

    def slow(*_args, **_kwargs):
        started.set()
        release.wait(5)
        return None

    monkeypatch.setattr(dialog_module, "find_update", slow)
    before = len(dialog_module._RUNNING)
    try:
        dialog_module.start_update_check(lambda info: None)

        assert started.wait(2)
        assert len(dialog_module._RUNNING) == before + 1
    finally:
        release.set()
        dialog_module._wait_for_checks()
        dialog_module._RUNNING.clear()


def test_closing_the_window_stops_the_download_without_waiting_on_it(qapp, installable, monkeypatch):
    running, cancelled = threading.Event(), threading.Event()

    def blocking_download(info, dest_dir, on_progress=None, is_cancelled=None):
        running.set()
        while not is_cancelled():
            time.sleep(0.01)
        cancelled.set()
        raise UpdateError("Download cancelled.")

    monkeypatch.setattr(dialog_module, "download_asset", blocking_download)
    before = len(dialog_module._RUNNING)
    dlg = UpdateDialog(_info())
    try:
        dlg.install_button.click()
        assert running.wait(2)
        assert len(dialog_module._RUNNING) == before + 1  # not the dialog's to destroy

        dlg.reject()

        assert cancelled.wait(2)
    finally:
        dialog_module._wait_for_checks()
        dialog_module._RUNNING.clear()


def test_a_failed_download_points_back_at_the_releases_page(qapp, installable):
    dlg = UpdateDialog(_info())

    dlg._on_failed("Download failed: connection reset")

    assert not dlg.bar.isVisibleTo(dlg)
    assert "releases page" in dlg.status.text()
