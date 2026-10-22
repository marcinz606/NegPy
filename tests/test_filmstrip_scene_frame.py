"""Decode failure owns top-left; the scene palette never takes the selection red."""

from types import SimpleNamespace

from PyQt6.QtGui import QColor

from negpy.desktop.view.sidebar.files import _ThumbnailDelegate
from negpy.desktop.view.styles.theme import THEME, scene_color
from negpy.services.assets.thumbnails import asset_thumbnail_key
from tests.test_filmstrip_stale_thumbnail import _paint, image_rect


def _state(stale=()):
    return SimpleNamespace(is_dirty=False, current_file_path=None, stale_thumbnails=set(stale))


def _pixel(pix, xy) -> str:
    return QColor(pix.toImage().pixel(*xy)).name().upper()


def test_decode_failure_takes_top_left_over_the_stale_dot(qapp):
    info = {"path": "/a.nef", "hash": "h1", "decode_failed": "bad"}
    pix = _paint(_ThumbnailDelegate(state=_state({asset_thumbnail_key(info)})), info)
    r = image_rect()
    assert _pixel(pix, (r.left() + 4 + 4, r.top() + 4 + 4)) == THEME.error.upper()


def test_palette_cycles_and_never_uses_the_selection_red():
    assert scene_color(1) == scene_color(7)
    assert not {THEME.border_checked, THEME.border_checked_hover} & {scene_color(i) for i in range(1, 7)}
