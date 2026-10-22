"""Keep and reject marks sit on the composite badge's neutral chip, with a white glyph."""

import pytest
from PyQt6.QtCore import QRect
from PyQt6.QtGui import QColor, QImage, QPainter

from negpy.desktop.view.styles.theme import THEME
from negpy.desktop.view.widgets.marks import draw_mark_badge


@pytest.mark.parametrize("check", [True, False])
def test_a_mark_is_a_white_glyph_on_a_neutral_chip(qapp, check):
    image = QImage(60, 40, QImage.Format.Format_RGB32)
    image.fill(QColor(128, 128, 128))
    painter = QPainter(image)
    draw_mark_badge(painter, QRect(0, 0, 60, 40), check)
    painter.end()

    colors = [image.pixelColor(x, y) for x in range(30, 60) for y in range(12, 40)]
    assert not any(c.red() > c.green() + 30 for c in colors), "no red on the badge"
    assert QColor(THEME.text_primary) in colors
