"""The open frame on the film strip: found by its hash, and its picture carries no line for unsaved edits."""

from types import SimpleNamespace
from unittest.mock import MagicMock

from PyQt6.QtCore import QRect, Qt
from PyQt6.QtGui import QColor, QIcon, QPainter, QPixmap
from PyQt6.QtWidgets import QStyle, QStyleOptionViewItem

from negpy.desktop.view.sidebar.files import _ThumbnailDelegate


def _paint(delegate, file_hash: str) -> QPixmap:
    thumb = QPixmap(60, 40)
    thumb.fill(QColor(120, 120, 120))
    index = MagicMock()
    index.data.side_effect = lambda role: {"path": "/a.nef", "hash": file_hash} if role == Qt.ItemDataRole.UserRole else QIcon(thumb)
    option = QStyleOptionViewItem()
    option.rect = QRect(0, 0, 66, 46)
    option.state = QStyle.StateFlag.State_Selected
    target = QPixmap(66, 46)
    target.fill(QColor(0, 0, 0))
    painter = QPainter(target)
    delegate.paint(painter, option, index)
    painter.end()
    return target


def test_the_open_frame_with_unsaved_edits_carries_no_line(qapp):
    state = SimpleNamespace(is_dirty=True, current_file_path="/a.nef", current_file_hash="h1", stale_thumbnails=set())
    image = _paint(_ThumbnailDelegate(state=state), "h1").toImage()

    # The image's bottom row, mid-width: the 60x40 thumbnail fills the 66x46 cell down to y=44.
    assert image.pixelColor(33, 44) == image.pixelColor(33, 23) == QColor(120, 120, 120)


def test_only_the_open_half_of_a_half_frame_scan_is_the_open_frame(qapp):
    delegate = _ThumbnailDelegate(state=SimpleNamespace(current_file_hash="ha#1"))

    assert delegate.is_active({"path": "/a.nef", "hash": "ha#1"})
    assert not delegate.is_active({"path": "/a.nef", "hash": "ha#2"})
