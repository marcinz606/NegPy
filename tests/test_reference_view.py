from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
from PyQt6.QtGui import QColor, QImage
from PyQt6.QtWidgets import QSplitter, QWidget

from negpy.desktop.view.canvas.reference_pane import ReferencePane
from negpy.desktop.view.main_window import MainWindow
from negpy.infrastructure.display.color_spaces import WORKING_COLOR_SPACE


def _window(metrics, path="/roll/frame_07.tif"):
    pane = ReferencePane(lambda: QColor("black"))
    splitter = QSplitter()
    splitter.addWidget(pane)
    splitter.addWidget(QWidget())
    splitter.resize(800, 400)
    splitter.show()
    pane.hide()
    win = SimpleNamespace(
        reference_pane=pane,
        central_splitter=splitter,
        state=SimpleNamespace(current_file_path=path, last_metrics=metrics),
        controller=MagicMock(),
        canvas=MagicMock(),
    )
    win.controller.display_transform_params.return_value = (WORKING_COLOR_SPACE, None, None)
    win._reference_snapshot = lambda: MainWindow._reference_snapshot(win)
    win.close_reference = lambda: MainWindow.close_reference(win)
    return win


def test_pinning_shows_the_frame_beside_the_canvas_and_a_second_press_closes_it(qapp):
    win = _window({"base_positive": np.full((30, 40, 3), 0.5, dtype=np.float32)})

    MainWindow.toggle_reference(win)
    assert win.reference_pane.isVisible()
    assert (win.reference_pane.image.width(), win.reference_pane.image.height()) == (40, 30)

    MainWindow.toggle_reference(win)
    assert not win.reference_pane.isVisible()
    assert win.reference_pane.image is None


def test_nothing_to_pin_without_a_rendered_frame(qapp):
    win = _window({}, path="")

    MainWindow.toggle_reference(win)

    assert not win.reference_pane.isVisible()
    win.canvas.hud.showMessage.assert_called_once()


def test_the_pane_paints_its_frame_and_closes_from_its_button(qapp):
    pane = ReferencePane(lambda: QColor("black"))
    image = QImage(40, 30, QImage.Format.Format_RGB888)
    image.fill(QColor("white"))
    pane.set_reference(image, "frame_07.tif")
    pane.resize(300, 200)
    closed = []
    pane.closed.connect(lambda: closed.append(True))

    shot = pane.grab().toImage()
    pane.close_btn.click()

    assert shot.pixelColor(150, 100) == QColor("white")
    assert closed == [True]
