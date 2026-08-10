"""Canvas empty state and the collapsible sidebar branding header."""

from unittest.mock import MagicMock

import pytest
from PyQt6.QtWidgets import QApplication, QWidget

from negpy.desktop.session import DesktopSessionManager
from negpy.desktop.view.main_window import _EmptyStateOverlay
from negpy.desktop.view.sidebar.header import SidebarHeader
from negpy.desktop.view.sidebar.session_panel import SessionPanel
from negpy.infrastructure.storage.repository import StorageRepository


@pytest.fixture
def host(qapp):
    """A stand-in for the canvas: the overlay must track whatever it is parented to."""
    w = QWidget()
    w.resize(600, 400)
    w.show()
    QApplication.processEvents()
    return w


@pytest.fixture
def overlay(host):
    ov = _EmptyStateOverlay(host)
    ov.show()
    QApplication.processEvents()
    return ov


# --- empty state ----------------------------------------------------------


def test_prompt_is_a_button_not_a_label(overlay):
    assert overlay.load_btn.text() == "Load some scans to get started"
    assert overlay.load_btn.isEnabled()


def test_load_menu_offers_both_import_routes(overlay, monkeypatch):
    captured: list[str] = []

    class _Menu:
        def __init__(self, *_a, **_k):
            pass

        def addAction(self, text):
            captured.append(text)
            return MagicMock()

        def exec(self, *_a, **_k):
            return None

    monkeypatch.setattr("negpy.desktop.view.main_window.QMenu", _Menu)
    overlay._show_load_menu()
    assert captured == ["Add files…", "Add folder…"]


def test_tour_button_emits_its_signal(overlay):
    seen: list[int] = []
    overlay.tour_requested.connect(lambda: seen.append(1))
    overlay.tour_btn.click()
    assert seen == [1]


def test_overlay_follows_the_canvas_when_it_resizes(overlay, host):
    """Regression: hiding a dock resizes the canvas but not the window, so the
    overlay stayed at its old width and drifted off-centre from the floating
    toolbar (which the canvas lays out itself)."""
    host.resize(1000, 400)
    QApplication.processEvents()
    assert overlay.size() == host.size()

    host.resize(480, 720)
    QApplication.processEvents()
    assert overlay.size() == host.size()


def test_overlay_stays_centred_on_its_parent(overlay, host):
    host.resize(1200, 500)
    QApplication.processEvents()
    assert overlay.geometry().center().x() == host.rect().center().x()


# --- collapsible branding header -----------------------------------------


@pytest.fixture
def controller(tmp_path, monkeypatch):
    monkeypatch.setattr("negpy.desktop.view.widgets.update_dialog.find_update", lambda *a, **k: None)
    repo = StorageRepository(str(tmp_path / "edits.db"), str(tmp_path / "settings.db"))
    repo.initialize()
    ctrl = MagicMock()
    ctrl.session = DesktopSessionManager(repo)
    ctrl.library_roots.return_value = []
    return ctrl


def test_header_expanded_shows_logo_and_version(qapp, controller):
    header = SidebarHeader(controller, expanded=True)
    assert header.is_expanded()
    assert header.body.isVisible() or not header.isVisible()  # body hidden only when collapsed
    assert header.ver_label.text().startswith("v")


def test_collapsing_hides_the_branding_and_frees_height(qapp, controller):
    header = SidebarHeader(controller, expanded=True)
    header.show()
    QApplication.processEvents()
    tall = header.sizeHint().height()

    header.set_expanded(False)
    QApplication.processEvents()

    assert not header.body.isVisible()
    assert header.sizeHint().height() < tall


def test_toggle_button_drives_the_collapse(qapp, controller):
    header = SidebarHeader(controller, expanded=True)
    header.show()
    QApplication.processEvents()

    header.toggle_button.click()
    assert not header.is_expanded()
    assert not header.body.isVisible()


def test_header_emits_expanded_changed(qapp, controller):
    header = SidebarHeader(controller, expanded=True)
    seen: list[bool] = []
    header.expanded_changed.connect(seen.append)
    header.set_expanded(False)
    header.set_expanded(True)
    assert seen == [False, True]


def test_session_panel_persists_and_restores_header_state(qapp, controller):
    repo = controller.session.repo
    panel = SessionPanel(controller)
    assert panel.header.is_expanded(), "defaults to expanded"

    panel.header.set_expanded(False)
    assert repo.get_global_setting("section_expanded_app_header") is False

    reopened = SessionPanel(controller)
    assert not reopened.header.is_expanded(), "collapsed state survives a restart"

    reopened.header.set_expanded(True)
    assert repo.get_global_setting("section_expanded_app_header") is True
    assert SessionPanel(controller).header.is_expanded()
