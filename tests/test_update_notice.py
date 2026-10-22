from unittest.mock import MagicMock

import pytest
from PyQt6.QtWidgets import QWidget

from negpy.desktop.view.update_notice import UpdateNotice
from negpy.kernel.system.updater import UpdateInfo


def _update(**overrides) -> UpdateInfo:
    fields = dict(
        version="9.9.9",
        notes="## What's new",
        page_url="https://example.test/release",
        asset_name="NegPy-9.9.9-x86_64.AppImage",
        download_url="https://example.test/asset",
        size=1,
    )
    return UpdateInfo(**{**fields, **overrides})


@pytest.fixture
def notice(qapp, monkeypatch):
    monkeypatch.setattr("negpy.desktop.view.update_notice.start_update_check", lambda cb: None)
    return UpdateNotice(QWidget(), MagicMock())


def test_nothing_is_announced_until_a_newer_release_turns_up(notice):
    seen = []
    notice.update_found.connect(seen.append)

    notice._on_update_checked(None)

    assert notice.update_info is None and seen == []


def test_a_found_update_is_announced_with_its_version(notice):
    seen = []
    notice.update_found.connect(seen.append)

    notice._on_update_checked(_update())

    assert seen == ["9.9.9"]


def test_the_update_window_opens_once_an_update_is_found(notice, monkeypatch):
    opened = []
    monkeypatch.setattr(
        "negpy.desktop.view.update_notice.UpdateDialog", lambda info, parent, repo=None: MagicMock(exec=lambda: opened.append(info))
    )
    notice._on_update_checked(_update())

    notice.check_for_updates()

    assert opened and opened[0].version == "9.9.9"


def test_the_window_stays_shut_while_the_running_version_is_current(notice, monkeypatch):
    monkeypatch.setattr("negpy.desktop.view.update_notice.UpdateDialog", lambda *a, **k: pytest.fail("no update to show"))

    notice.show_update_dialog()


def test_a_manual_check_runs_one_thread_at_a_time(notice, monkeypatch):
    started = []
    monkeypatch.setattr("negpy.desktop.view.update_notice.start_update_check", lambda cb: started.append(cb))

    notice.check_for_updates()
    notice.check_for_updates()

    assert len(started) == 1


def test_a_manual_check_can_run_again_once_nothing_is_new(notice, monkeypatch):
    started = []
    monkeypatch.setattr("PyQt6.QtWidgets.QMessageBox.information", lambda *a, **k: None)
    monkeypatch.setattr("negpy.desktop.view.update_notice.start_update_check", lambda cb: started.append(cb))
    notice.check_for_updates()

    notice._on_manual_check(None)
    notice.check_for_updates()

    assert len(started) == 2
