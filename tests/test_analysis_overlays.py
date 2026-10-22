"""Clipping Overlay: the masks it washes over the frame, the zone strip's hover that names a
zone, and the wash itself."""

import numpy as np

from negpy.features.exposure.analysis import clip_masks


def test_clip_masks_use_the_histograms_end_bins() -> None:
    rgb = np.full((2, 3, 3), 0.5, dtype=np.float32)
    rgb[0, 0] = (0.001, 0.5, 0.5)  # one channel in the black bin
    rgb[0, 1] = (0.5, 0.999, 0.5)  # one channel in the white bin
    rgb[1, 2] = (1.0 / 256, 0.5, 0.5)  # first value past the black bin

    low, high = clip_masks(rgb)

    assert low.tolist() == [[True, False, False], [False, False, False]]
    assert high.tolist() == [[False, True, False], [False, False, False]]


def test_masks_are_strided_to_the_edge_cap() -> None:
    rgb = np.zeros((300, 1000, 3), dtype=np.float32)

    low, _high = clip_masks(rgb, max_edge=256)

    assert max(low.shape) <= 256
    assert low.all()


def test_zone_strip_hover_names_the_cell_and_leaving_clears_it(qapp) -> None:
    from PyQt6.QtCore import QEvent, QPointF, Qt
    from PyQt6.QtGui import QMouseEvent

    from negpy.desktop.view.widgets.charts import ZoneStripWidget

    strip = ZoneStripWidget()
    strip.resize(110, 24)
    strip.update_data(np.full(11, 0.1))
    hovered: list = []
    strip.zone_hovered.connect(hovered.append)

    def move(x: float) -> None:
        strip.mouseMoveEvent(
            QMouseEvent(
                QEvent.Type.MouseMove, QPointF(x, 12.0), Qt.MouseButton.NoButton, Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier
            )
        )

    move(55.0)
    move(56.0)  # same cell: no repeat
    move(105.0)
    strip.leaveEvent(QEvent(QEvent.Type.Leave))

    assert hovered == [5, 10, None]


def test_tint_layers_paint_their_color_and_leave_the_rest_clear(qapp) -> None:
    from PyQt6.QtGui import QColor

    from negpy.desktop.view.canvas.overlay import _tint_qimage

    low = np.array([[True, False, False]])
    high = np.array([[False, True, False]])

    img = _tint_qimage([(low, QColor(0, 0, 255)), (high, QColor(255, 0, 0))], 255)

    assert QColor(img.pixel(0, 0)).getRgb() == (0, 0, 255, 255)
    assert QColor(img.pixel(1, 0)).getRgb() == (255, 0, 0, 255)
    assert QColor.fromRgba(img.pixel(2, 0)).alpha() == 0


def test_clip_marks_take_the_color_vision_pair(qapp) -> None:
    from PyQt6.QtGui import QColor

    from negpy.desktop.view.sidebar.roi_image import zebra_image
    from negpy.desktop.view.styles.color_vision import palette_for
    from negpy.desktop.view.widgets.charts import MiniHistogramWidget, ZoneStripWidget

    shadow, highlight = palette_for("tritan").pair

    def hue_near(color: QColor, token: str) -> bool:
        return abs(color.hue() - QColor(token).hue()) <= 12

    hist = MiniHistogramWidget()
    hist.set_clip_colors(shadow, highlight)
    buf = np.zeros((4, 256))
    buf[3, [0, 128, 255]] = 100
    hist.update_data(buf)
    hist.resize(200, 20)
    image = hist.grab().toImage()
    assert hue_near(image.pixelColor(1, 10), shadow) and hue_near(image.pixelColor(198, 10), highlight)

    strip = ZoneStripWidget()
    strip.set_clip_colors(shadow, highlight)
    strip.update_data(np.zeros(11), (True, True))
    strip.resize(220, 26)
    image = strip.grab().toImage()
    assert hue_near(image.pixelColor(2, 2), shadow) and hue_near(image.pixelColor(217, 2), highlight)

    zebra = zebra_image(np.ones((8, 8), bool), np.zeros((8, 8), bool), (shadow, highlight))
    colors = {zebra.pixelColor(x, y).rgb() & 0xFFFFFF for x in range(8) for y in range(8) if zebra.pixelColor(x, y).alpha()}
    assert colors == {QColor(highlight).rgb() & 0xFFFFFF}
