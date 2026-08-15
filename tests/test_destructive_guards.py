from unittest.mock import MagicMock, patch

from PyQt6.QtCore import QPoint, QPointF, Qt
from PyQt6.QtGui import QWheelEvent
from PyQt6.QtWidgets import QComboBox

from negpy.desktop.session import AppState
from negpy.desktop.view.sidebar.history import HistoryPanel
from negpy.desktop.view.sidebar.scan import ScanSidebar


def _panel(cls):
    controller = MagicMock()
    controller.state = AppState()
    return controller, cls(controller)


def _wheel(widget) -> QWheelEvent:
    return QWheelEvent(
        QPointF(widget.rect().center()),
        QPointF(widget.mapToGlobal(widget.rect().center())),
        QPoint(0, 0),
        QPoint(0, -120),
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.NoScrollPhase,
        False,
    )


def test_scan_panel_combos_ignore_the_wheel_unless_focused(qapp):
    """ScanSidebar is not a BaseSidebar, so it never got the guard — scrolling the panel
    changed scanner settings under the pointer."""
    _, sidebar = _panel(ScanSidebar)
    combos = sidebar.findChildren(QComboBox)
    assert combos, "the scan panel should have combo boxes to guard"

    exercised = 0
    for combo in combos:
        if combo.count() < 2:
            continue
        exercised += 1
        combo.setCurrentIndex(0)
        combo.clearFocus()
        combo.wheelEvent(_wheel(combo))
        assert combo.currentIndex() == 0

    assert exercised, "no combo had enough items to scroll — the test proved nothing"


def test_blank_rename_leaves_the_work_print_alone(qapp):
    """QInputDialog returns ok=True on an emptied field, which used to rename the work
    print to an empty string."""
    controller, panel = _panel(HistoryPanel)
    panel.work_prints.addItem("keeper")

    menu = MagicMock()
    rename_action = object()
    menu.addAction.side_effect = lambda *_: rename_action
    menu.exec.return_value = rename_action

    with patch("negpy.desktop.view.sidebar.history.QMenu", return_value=menu):
        with patch("negpy.desktop.view.sidebar.history.QInputDialog.getText", return_value=("   ", True)):
            panel._on_work_print_menu(QPoint(0, 0))

    controller.session.rename_work_print.assert_not_called()


def test_deleting_a_work_print_asks_first(qapp):
    controller, panel = _panel(HistoryPanel)
    panel.work_prints.addItem("keeper")

    menu = MagicMock()
    delete_action = object()
    menu.addAction.side_effect = [object(), object(), delete_action]
    menu.exec.return_value = delete_action

    with patch("negpy.desktop.view.sidebar.history.QMenu", return_value=menu):
        with patch("negpy.desktop.view.sidebar.history.confirm_delete_named", return_value=False) as confirm:
            panel._on_work_print_menu(QPoint(0, 0))

    confirm.assert_called_once()
    controller.session.delete_work_print.assert_not_called()
