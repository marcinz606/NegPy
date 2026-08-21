import sys

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QDialog

from negpy.desktop.view.widgets.floating_panel import float_over_app
from negpy.desktop.view.widgets.progress_dialog import ProgressDialog


def test_float_over_app_makes_a_panel_on_macos() -> None:
    dlg = QDialog()
    float_over_app(dlg, platform="darwin")
    assert dlg.windowFlags() & Qt.WindowType.Tool == Qt.WindowType.Tool
    assert dlg.testAttribute(Qt.WidgetAttribute.WA_MacAlwaysShowToolWindow)


def test_float_over_app_leaves_other_platforms_alone() -> None:
    for platform in ("win32", "linux"):
        dlg = QDialog()
        before = dlg.windowFlags()
        float_over_app(dlg, platform=platform)
        assert dlg.windowFlags() == before
        assert not dlg.testAttribute(Qt.WidgetAttribute.WA_MacAlwaysShowToolWindow)


def test_progress_dialog_floats_and_stays_modeless() -> None:
    dlg = ProgressDialog()
    assert not dlg.isModal()
    is_panel = bool(dlg.windowFlags() & Qt.WindowType.Tool)
    assert is_panel == (sys.platform == "darwin")
