"""Canvas empty state."""

from unittest.mock import MagicMock

import pytest
from PyQt6.QtWidgets import QApplication, QWidget

from negpy.desktop.view.main_window import _EmptyStateOverlay


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
    assert captured == ["Import Folder as a Roll…", "Add Files…"]


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
