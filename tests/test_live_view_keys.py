"""Space takes the shot from anywhere in the Live View window.

A focused button takes Space for itself, so a click on Retake or on one of the view
toggles would otherwise leave Space working that control.
"""

from __future__ import annotations

import pytest
from PyQt6.QtCore import QEvent, Qt
from PyQt6.QtGui import QKeyEvent
from PyQt6.QtWidgets import QApplication

from negpy.desktop.view.sidebar.live_view_window import LiveViewWindow

_NO_MODS = Qt.KeyboardModifier.NoModifier


def _send(app, window, key: int, repeats: int = 1) -> None:
    target = QApplication.focusWidget() or window
    for i in range(repeats):
        QApplication.sendEvent(target, QKeyEvent(QEvent.Type.KeyPress, key, _NO_MODS, "", i > 0))
    QApplication.sendEvent(target, QKeyEvent(QEvent.Type.KeyRelease, key, _NO_MODS))
    app.processEvents()


@pytest.fixture
def window(qapp):
    win = LiveViewWindow()
    win.show()
    qapp.processEvents()
    yield win
    win.close()


def _spy(window, *names: str) -> list[str]:
    fired: list[str] = []
    for name in names:
        getattr(window, name).clicked.connect(lambda _=False, n=name: fired.append(n))
    return fired


@pytest.mark.parametrize("focused", [None, "histogram_btn", "zebra_btn", "retake_btn", "meter_btn"])
def test_space_scans_whatever_toolbar_button_holds_the_focus(qapp, window, focused):
    fired = _spy(window, "scan_btn", "retake_btn", "meter_btn", "histogram_btn", "zebra_btn")
    if focused is not None:
        getattr(window, focused).setFocus(Qt.FocusReason.MouseFocusReason)
        qapp.processEvents()

    _send(qapp, window, Qt.Key.Key_Space)

    assert fired == ["scan_btn"]


def test_a_held_space_scans_once(qapp, window):
    fired = _spy(window, "scan_btn")
    window.histogram_btn.setFocus(Qt.FocusReason.MouseFocusReason)
    qapp.processEvents()

    _send(qapp, window, Qt.Key.Key_Space, repeats=4)

    assert fired == ["scan_btn"]


def test_a_focused_stepper_keeps_space_and_steps_on_the_arrow_keys(qapp, window):
    """Space on a focused stepper never scans; the arrow keys step it."""
    fired = _spy(window, "scan_btn")
    window.iso_stepper.addItem("100", "100")
    window.iso_stepper.addItem("200", "200")
    window.iso_stepper.setFocus(Qt.FocusReason.TabFocusReason)
    qapp.processEvents()

    _send(qapp, window, Qt.Key.Key_Space)
    assert fired == []
    assert window.iso_stepper.currentIndex() == 0

    _send(qapp, window, Qt.Key.Key_Down)
    assert window.iso_stepper.currentIndex() == 1


def test_letter_keys_keep_their_own_buttons(qapp, window):
    fired = _spy(window, "scan_btn", "retake_btn", "histogram_btn")
    window.histogram_btn.setFocus(Qt.FocusReason.MouseFocusReason)
    qapp.processEvents()

    for key in (Qt.Key.Key_H, Qt.Key.Key_R, Qt.Key.Key_S):
        _send(qapp, window, key)

    assert fired == ["histogram_btn", "retake_btn", "scan_btn"]
