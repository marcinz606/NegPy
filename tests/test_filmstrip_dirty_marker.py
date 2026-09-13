"""The film strip marks the one frame whose edits are not yet on disk."""

from types import SimpleNamespace
from unittest.mock import MagicMock

from PyQt6.QtCore import QRect, Qt
from PyQt6.QtGui import QColor, QIcon, QPainter, QPixmap
from PyQt6.QtWidgets import QStyleOptionViewItem

from negpy.desktop.view.sidebar.files import _ThumbnailDelegate
from negpy.desktop.view.styles.theme import THEME


def _paint(delegate, path: str) -> QPixmap:
    thumb = QPixmap(60, 40)
    thumb.fill(QColor(120, 120, 120))
    index = MagicMock()
    index.data.side_effect = lambda role: {"path": path} if role == Qt.ItemDataRole.UserRole else QIcon(thumb)
    option = QStyleOptionViewItem()
    option.rect = QRect(0, 0, 66, 46)
    target = QPixmap(66, 46)
    target.fill(QColor(0, 0, 0))
    painter = QPainter(target)
    delegate.paint(painter, option, index)
    painter.end()
    return target


def _bottom_edge_colour(pix: QPixmap) -> QColor:
    img = pix.toImage()
    # One px above the rounded frame line, mid-width, inside the image rect (margin 3).
    return QColor(img.pixel(33, 46 - 3 - 2))


def test_active_dirty_frame_gets_an_accent_line(qapp):
    state = SimpleNamespace(is_dirty=True, current_file_path="/a.nef")
    dirty = _bottom_edge_colour(_paint(_ThumbnailDelegate(state=state), "/a.nef"))
    assert dirty.name().upper() == THEME.accent_primary.upper()


def test_other_frames_and_clean_frames_do_not(qapp):
    state = SimpleNamespace(is_dirty=True, current_file_path="/a.nef")
    other = _bottom_edge_colour(_paint(_ThumbnailDelegate(state=state), "/b.nef"))
    assert other.name().upper() != THEME.accent_primary.upper()
    clean = SimpleNamespace(is_dirty=False, current_file_path="/a.nef")
    same = _bottom_edge_colour(_paint(_ThumbnailDelegate(state=clean), "/a.nef"))
    assert same.name().upper() != THEME.accent_primary.upper()
