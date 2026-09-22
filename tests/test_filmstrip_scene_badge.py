"""Scene chip at top-right while Show Scenes is on; decode failure owns top-left."""

from types import SimpleNamespace

from PyQt6.QtGui import QColor

from negpy.desktop.view.sidebar.files import _ThumbnailDelegate
from negpy.desktop.view.styles.theme import THEME, scene_color
from negpy.services.assets.thumbnails import asset_thumbnail_key
from tests.test_filmstrip_stale_thumbnail import _paint

# Image rect spans x 3..62, y 3..42; chips have radius 9 and sit 4px in from the edge.
_TOP_RIGHT = (62 - 9 - 4 - 6, 3 + 9 + 4)
_TOP_LEFT_DOT = (3 + 4 + 4, 3 + 4 + 4)


def _state(stale=()):
    return SimpleNamespace(is_dirty=False, current_file_path=None, stale_thumbnails=set(stale))


def _pixel(pix, xy) -> str:
    return QColor(pix.toImage().pixel(*xy)).name().upper()


def test_scene_chip_is_drawn_in_the_scene_color_when_shown(qapp):
    delegate = _ThumbnailDelegate(state=_state())
    delegate.set_show_scenes(True)
    pix = _paint(delegate, {"path": "/a.nef", "hash": "h1", "scene": (2, "s2", "Night")})
    assert _pixel(pix, _TOP_RIGHT) == scene_color(2).upper()


def test_scene_chip_is_hidden_when_toggle_is_off(qapp):
    pix = _paint(_ThumbnailDelegate(state=_state()), {"path": "/a.nef", "hash": "h1", "scene": (1, "s1", "Beach")})
    assert _pixel(pix, _TOP_RIGHT) != scene_color(1).upper()


def test_decode_failure_takes_top_left_over_the_stale_dot(qapp):
    info = {"path": "/a.nef", "hash": "h1", "decode_failed": "bad"}
    pix = _paint(_ThumbnailDelegate(state=_state({asset_thumbnail_key(info)})), info)
    assert _pixel(pix, _TOP_LEFT_DOT) == THEME.error.upper()


def test_palette_cycles_and_never_uses_the_selection_red():
    assert scene_color(1) == scene_color(7)
    assert THEME.accent_primary not in {scene_color(i) for i in range(1, 7)}
