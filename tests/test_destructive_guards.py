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


def test_dropdowns_and_number_boxes_ignore_the_wheel(qapp):
    """Scrolling a panel never changes a setting under the pointer, focused or not; the
    ignored event goes on to the panel."""
    from negpy.desktop.main import WheelScrollsPanel
    from negpy.desktop.view.widgets.sliders import CompactSlider

    _, sidebar = _panel(ScanSidebar)
    sidebar.device_combo.addItem("Second scanner", "second")
    boxes = [c for c in sidebar.findChildren(QComboBox) if c.count() > 1]
    assert boxes, "no combo had enough items to scroll — the test proved nothing"
    boxes.append(_spin())
    slider = CompactSlider("Density", 0.0, 2.0, 1.0)
    boxes.append(slider.spin)

    guard = WheelScrollsPanel(qapp)
    qapp.installEventFilter(guard)
    try:
        for box in boxes:
            before = box.currentIndex() if isinstance(box, QComboBox) else box.value()
            event = _wheel(box)
            qapp.sendEvent(box, event)
            after = box.currentIndex() if isinstance(box, QComboBox) else box.value()
            assert after == before
            assert not event.isAccepted()
    finally:
        qapp.removeEventFilter(guard)


def _spin():
    from PyQt6.QtWidgets import QSpinBox

    spin = QSpinBox()
    spin.setRange(0, 100)
    spin.setValue(50)
    return spin


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
