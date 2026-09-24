"""A thumbnail still loading (or that failed to decode) still shows its selection border."""

from unittest.mock import MagicMock

from PyQt6.QtCore import QRect, Qt
from PyQt6.QtGui import QColor, QPainter, QPixmap
from PyQt6.QtWidgets import QStyle, QStyleOptionViewItem

from negpy.desktop.view.sidebar.files import _ThumbnailDelegate
from negpy.desktop.view.styles.theme import THEME


def _paint_placeholder(selected: bool) -> QPixmap:
    index = MagicMock()
    index.data.side_effect = lambda role: {"path": "/a.nef"} if role == Qt.ItemDataRole.UserRole else None
    option = QStyleOptionViewItem()
    option.rect = QRect(0, 0, 40, 40)
    option.state = QStyle.StateFlag.State_Selected if selected else QStyle.StateFlag.State_None
    target = QPixmap(40, 40)
    target.fill(QColor(0, 0, 0))
    painter = QPainter(target)
    _ThumbnailDelegate().paint(painter, option, index)
    painter.end()
    return target


def _left_edge_colour(pix: QPixmap) -> QColor:
    img = pix.toImage()
    # On the selection ring, which sits outside the placeholder's edge, mid-height.
    return QColor(img.pixel(_ThumbnailDelegate._MARGIN - _ThumbnailDelegate._SELECTION_OUTSET + 1, 20))


def test_selected_placeholder_shows_the_accent_border(qapp):
    assert _left_edge_colour(_paint_placeholder(selected=True)).name().upper() == THEME.accent_primary.upper()


def test_unselected_placeholder_does_not(qapp):
    assert _left_edge_colour(_paint_placeholder(selected=False)).name().upper() != THEME.accent_primary.upper()
