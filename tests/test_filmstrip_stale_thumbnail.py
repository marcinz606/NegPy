"""The film strip flags a thumbnail whose bitmap predates a settings write that
reached the file without a render (a bulk apply, not the active canvas)."""

from types import SimpleNamespace
from unittest.mock import MagicMock

from PyQt6.QtCore import QRect, QSize, Qt
from PyQt6.QtGui import QColor, QIcon, QPainter, QPixmap
from PyQt6.QtWidgets import QStyleOptionViewItem

from negpy.desktop.view.sidebar.files import _ThumbnailDelegate
from negpy.desktop.view.styles.theme import THEME
from negpy.services.assets.thumbnails import asset_thumbnail_key


def _paint(delegate, file_info: dict) -> QPixmap:
    thumb = QPixmap(60, 40)
    thumb.fill(QColor(120, 120, 120))
    index = MagicMock()
    index.data.side_effect = lambda role: file_info if role == Qt.ItemDataRole.UserRole else QIcon(thumb)
    option = QStyleOptionViewItem()
    option.rect = QRect(0, 0, 66, 46)
    target = QPixmap(66, 46)
    target.fill(QColor(0, 0, 0))
    painter = QPainter(target)
    delegate.paint(painter, option, index)
    painter.end()
    return target


def image_rect() -> QRect:
    """Where _paint's 60x40 thumbnail lands in its 66x46 cell."""
    m = _ThumbnailDelegate._MARGIN
    area = QRect(m, m, 66 - 2 * m, 46 - 2 * m)
    return _ThumbnailDelegate._fit_rect(area, QSize(60, 40).scaled(area.size(), Qt.AspectRatioMode.KeepAspectRatio))


def _top_left_colour(pix: QPixmap) -> QColor:
    # Dot center: radius 4, offset by 4px from the image's top-left corner.
    r = image_rect()
    return QColor(pix.toImage().pixel(r.left() + 4 + 4, r.top() + 4 + 4))


def test_stale_frame_gets_a_dot(qapp):
    # Keyed like push_external_history keys it: asset_thumbnail_key, not the bare hash --
    # a triplet's thumbnail cache key differs from its plain hash.
    state = SimpleNamespace(is_dirty=False, current_file_path=None, stale_thumbnails={asset_thumbnail_key({"hash": "h1"})})
    dot = _top_left_colour(_paint(_ThumbnailDelegate(state=state), {"path": "/a.nef", "hash": "h1"}))
    assert dot.name().upper() == THEME.warn_amber.upper()


def test_fresh_frame_does_not(qapp):
    state = SimpleNamespace(is_dirty=False, current_file_path=None, stale_thumbnails={asset_thumbnail_key({"hash": "h1"})})
    clean = _top_left_colour(_paint(_ThumbnailDelegate(state=state), {"path": "/b.nef", "hash": "h2"}))
    assert clean.name().upper() != THEME.warn_amber.upper()


def test_stale_key_is_the_thumbnail_cache_key_not_the_bare_hash(qapp):
    """A regression guard for the mismatch this indicator originally shipped with:
    push_external_history adds asset_thumbnail_key(asset) (hash plus a cache-version
    suffix), never the bare hash, so the read side must key the same way."""
    state = SimpleNamespace(is_dirty=False, current_file_path=None, stale_thumbnails={"h1"})
    clean = _top_left_colour(_paint(_ThumbnailDelegate(state=state), {"path": "/a.nef", "hash": "h1"}))
    assert clean.name().upper() != THEME.warn_amber.upper()


def test_no_state_never_stale(qapp):
    clean = _top_left_colour(_paint(_ThumbnailDelegate(state=None), {"path": "/a.nef", "hash": "h1"}))
    assert clean.name().upper() != THEME.warn_amber.upper()
