"""A frame's picture edge takes its scene's color while Show Scenes is on, with the
selection ring outside it; decode failure owns top-left."""

from types import SimpleNamespace
from unittest.mock import MagicMock

from PyQt6.QtCore import QRect, Qt
from PyQt6.QtGui import QColor, QIcon, QPainter, QPixmap
from PyQt6.QtWidgets import QStyle, QStyleOptionViewItem

from negpy.desktop.view.sidebar.files import _ThumbnailDelegate
from negpy.desktop.view.styles.theme import THEME, scene_color
from negpy.services.assets.thumbnails import asset_thumbnail_key
from tests.test_filmstrip_stale_thumbnail import _paint, image_rect


def _state(stale=()):
    return SimpleNamespace(is_dirty=False, current_file_path=None, stale_thumbnails=set(stale))


def _pixel(pix, xy) -> str:
    return QColor(pix.toImage().pixel(*xy)).name().upper()


def _paint_selected(delegate, file_info: dict) -> QPixmap:
    thumb = QPixmap(60, 40)
    thumb.fill(QColor(120, 120, 120))
    index = MagicMock()
    index.data.side_effect = lambda role: file_info if role == Qt.ItemDataRole.UserRole else QIcon(thumb)
    option = QStyleOptionViewItem()
    option.rect = QRect(0, 0, 66, 46)
    option.state = QStyle.StateFlag.State_Selected
    target = QPixmap(66, 46)
    target.fill(QColor(0, 0, 0))
    painter = QPainter(target)
    delegate.paint(painter, option, index)
    painter.end()
    return target


def _edge():
    r = image_rect()
    return (r.left() + 1, r.center().y())


def test_scene_edge_is_drawn_in_the_scene_color_when_shown(qapp):
    delegate = _ThumbnailDelegate(state=_state())
    delegate.set_show_scenes(True)
    pix = _paint(delegate, {"path": "/a.nef", "hash": "h1", "scene": (2, "s2", "Night")})
    assert _pixel(pix, _edge()) == scene_color(2).upper()


def test_scene_edge_is_hidden_when_toggle_is_off(qapp):
    pix = _paint(_ThumbnailDelegate(state=_state()), {"path": "/a.nef", "hash": "h1", "scene": (1, "s1", "Beach")})
    assert _pixel(pix, _edge()) != scene_color(1).upper()


def test_the_selection_ring_sits_outside_the_scene_edge(qapp):
    delegate = _ThumbnailDelegate(state=_state())
    delegate.set_show_scenes(True)
    pix = _paint_selected(delegate, {"path": "/a.nef", "hash": "h1", "scene": (1, "s1", "Beach")})
    x, y = _edge()
    assert _pixel(pix, (x, y)) == scene_color(1).upper()
    assert _pixel(pix, (image_rect().left() - _ThumbnailDelegate._SELECTION_OUTSET + 1, y)) == THEME.accent_primary.upper()


def test_decode_failure_takes_top_left_over_the_stale_dot(qapp):
    info = {"path": "/a.nef", "hash": "h1", "decode_failed": "bad"}
    pix = _paint(_ThumbnailDelegate(state=_state({asset_thumbnail_key(info)})), info)
    r = image_rect()
    assert _pixel(pix, (r.left() + 4 + 4, r.top() + 4 + 4)) == THEME.error.upper()


def test_palette_cycles_and_never_uses_the_selection_red():
    assert scene_color(1) == scene_color(7)
    assert THEME.accent_primary not in {scene_color(i) for i in range(1, 7)}
