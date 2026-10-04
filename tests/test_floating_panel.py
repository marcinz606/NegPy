import sys
from pathlib import Path

import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QKeySequence, QShortcut
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QDialog, QMainWindow, QPushButton, QSlider, QVBoxLayout

from negpy.desktop.view.widgets.floating_panel import float_over_app
from negpy.desktop.view.widgets.progress_dialog import ProgressDialog

# Qt::Tool shares the Window and Dialog bits, so only the masked windowType() tells
# a panel from a plain dialog — `flags & Tool` is truthy for both.


def test_float_over_app_makes_a_panel_on_macos() -> None:
    dlg = QDialog()
    assert dlg.windowType() == Qt.WindowType.Dialog
    float_over_app(dlg, platform="darwin")
    assert dlg.windowType() == Qt.WindowType.Tool
    assert dlg.testAttribute(Qt.WidgetAttribute.WA_MacAlwaysShowToolWindow)


def test_float_over_app_leaves_other_platforms_alone() -> None:
    for platform in ("win32", "linux"):
        dlg = QDialog()
        before = dlg.windowFlags()
        float_over_app(dlg, platform=platform)
        assert dlg.windowFlags() == before
        assert dlg.windowType() == Qt.WindowType.Dialog
        assert not dlg.testAttribute(Qt.WidgetAttribute.WA_MacAlwaysShowToolWindow)


def test_float_over_app_keeps_the_other_flags() -> None:
    dlg = QDialog()
    dlg.setWindowFlags(dlg.windowFlags() | Qt.WindowType.WindowCloseButtonHint)
    float_over_app(dlg, platform="darwin")
    assert dlg.windowType() == Qt.WindowType.Tool
    assert dlg.windowFlags() & Qt.WindowType.WindowCloseButtonHint


def test_progress_dialog_floats_and_stays_modeless() -> None:
    dlg = ProgressDialog()
    assert not dlg.isModal()
    is_panel = dlg.windowType() == Qt.WindowType.Tool
    assert is_panel == (sys.platform == "darwin")


@pytest.fixture
def panel_over_main():
    main = QMainWindow()
    fired: list[str] = []
    for key in ("Esc", "Left", "Right", "K"):
        QShortcut(QKeySequence(key), main).activated.connect(lambda key=key: fired.append(key))
    dlg = QDialog(main)
    float_over_app(dlg, platform="darwin")
    main.show()
    yield main, dlg, fired
    dlg.close()
    main.close()


def _focus(dlg: QDialog, widget) -> None:
    dlg.show()
    dlg.activateWindow()
    assert QTest.qWaitForWindowActive(dlg, 2000)
    widget.setFocus()


def test_esc_closes_a_panel_instead_of_reaching_the_main_window(panel_over_main) -> None:
    _, dlg, fired = panel_over_main
    btn = QPushButton("x", dlg)
    QVBoxLayout(dlg).addWidget(btn)
    _focus(dlg, btn)
    QTest.keyClick(btn, Qt.Key.Key_Escape)
    assert fired == []
    assert not dlg.isVisible()


def test_arrows_move_a_panel_slider_and_letters_stay_in_the_panel(panel_over_main) -> None:
    _, dlg, fired = panel_over_main
    slider = QSlider(Qt.Orientation.Horizontal, dlg)
    slider.setRange(0, 10)
    slider.setValue(5)
    QVBoxLayout(dlg).addWidget(slider)
    _focus(dlg, slider)
    QTest.keyClick(slider, Qt.Key.Key_Right)
    QTest.keyClick(slider, Qt.Key.Key_K)
    assert slider.value() == 6
    assert fired == []


def test_no_panel_binds_a_qshortcut() -> None:
    root = Path(__file__).resolve().parents[1] / "negpy"
    offenders = [
        str(path.relative_to(root))
        for path in root.rglob("*.py")
        if path.name != "floating_panel.py" and "float_over_app(" in (text := path.read_text(encoding="utf-8")) and "QShortcut(" in text
    ]
    assert offenders == []


def test_command_chords_pass_through_a_panel(panel_over_main) -> None:
    main, dlg, _ = panel_over_main
    fired: list[str] = []
    QShortcut(QKeySequence("Ctrl+E"), main).activated.connect(lambda: fired.append("export"))
    btn = QPushButton("x", dlg)
    QVBoxLayout(dlg).addWidget(btn)
    _focus(dlg, btn)
    QTest.keyClick(btn, Qt.Key.Key_E, Qt.KeyboardModifier.ControlModifier)
    assert fired == ["export"]


@pytest.mark.parametrize("window", ["live_view", "calibration"])
def test_esc_and_close_both_end_a_scan_window_session(panel_over_main, window) -> None:
    from negpy.desktop.view.sidebar.calibration_window import CalibrationWindow
    from negpy.desktop.view.sidebar.live_view_window import LiveViewWindow

    main, _, _ = panel_over_main
    win = (LiveViewWindow if window == "live_view" else CalibrationWindow)(main)
    float_over_app(win, platform="darwin")
    closed: list[str] = []
    win.closed.connect(lambda: closed.append("closed"))
    _focus(win, win)
    QTest.keyClick(win, Qt.Key.Key_Escape)
    assert closed == ["closed"]
    assert not win.isVisible()
    win.show()
    win.close()
    assert closed == ["closed", "closed"]


def test_esc_leaves_a_running_job_visible(panel_over_main) -> None:
    main, _, _ = panel_over_main
    dlg = ProgressDialog(main)
    aborts: list[str] = []
    dlg.abort_requested.connect(lambda: aborts.append("abort"))
    dlg.start("Exporting", abortable=True)
    _focus(dlg, dlg)
    QTest.keyClick(dlg, Qt.Key.Key_Escape)
    assert dlg.isVisible()
    assert aborts == []
    dlg.close()
    assert not dlg.isVisible()


def test_progress_leaves_main_window_keys_live(panel_over_main) -> None:
    main, _, fired = panel_over_main
    dlg = ProgressDialog(main)
    float_over_app(dlg, platform="darwin", claim_keys=False)
    dlg.start("Exporting", abortable=True)
    _focus(dlg, dlg)
    QTest.keyClick(dlg, Qt.Key.Key_Right)
    assert fired == ["Right"]
    dlg.finish()


def test_esc_leaves_a_running_calibration_open(panel_over_main) -> None:
    from negpy.desktop.view.sidebar.calibration_window import CalibrationWindow

    main, _, _ = panel_over_main
    win = CalibrationWindow(main)
    float_over_app(win, platform="darwin")
    closed: list[str] = []
    win.closed.connect(lambda: closed.append("closed"))
    win.set_inputs_locked(True)
    _focus(win, win)
    QTest.keyClick(win, Qt.Key.Key_Escape)
    assert win.isVisible()
    assert closed == []
    win.close()
    assert closed == ["closed"]
