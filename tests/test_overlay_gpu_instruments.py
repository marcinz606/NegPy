"""Canvas instruments that measure pixels, on the GPU display path.

The GPU path hands the canvas a texture instead of host pixels, so the zone grid, the
grain loupe and the notes sheet lost their input. They read the texture back themselves,
once, and only when one of them is actually on.
"""

import numpy as np
from PyQt6.QtCore import QPointF, QRectF
from PyQt6.QtGui import QPainter, QPixmap

from negpy.desktop.session import AppState
from negpy.desktop.view.canvas.overlay import CanvasOverlay

W, H = 400, 300


class _FakeTexture:
    def __init__(self, value: float = 0.5):
        self.width, self.height = W, H
        self._array = np.full((H, W, 4), value, dtype=np.float32)
        self.readbacks = 0

    def readback(self):
        self.readbacks += 1
        return self._array


def _overlay(tex: _FakeTexture) -> CanvasOverlay:
    overlay = CanvasOverlay(AppState())
    overlay.update_buffer(None, "Adobe RGB", gpu_size=(W, H), gpu_texture=tex)
    overlay._view_rect = QRectF(0, 0, W, H)
    overlay._mouse_pos = QPointF(W / 2, H / 2)
    return overlay


def test_host_pixels_come_off_the_texture_once():
    tex = _FakeTexture()
    overlay = _overlay(tex)
    buf = overlay._host_buffer()

    assert buf is not None and buf.shape == (H, W, 3)
    assert overlay._host_buffer() is buf
    assert tex.readbacks == 1


def test_a_new_frame_invalidates_the_host_copy():
    overlay = _overlay(_FakeTexture(0.25))
    first = overlay._host_buffer()
    overlay.update_buffer(None, "Adobe RGB", gpu_size=(W, H), gpu_texture=_FakeTexture(0.75))

    assert float(overlay._host_buffer().mean()) != float(first.mean())


def test_the_zone_grid_builds_from_the_texture():
    """Regression: the zone map went blank on the GPU path — _display_buffer was None."""
    overlay = _overlay(_FakeTexture())
    overlay.state.zones_overlay = True
    pixmap = QPixmap(W, H)
    painter = QPainter(pixmap)
    overlay._draw_zone_grid(painter)
    painter.end()

    assert overlay._zone_cells is not None


def test_the_loupe_gets_an_image_and_caches_it():
    overlay = _overlay(_FakeTexture())
    img = overlay._host_qimage()

    assert img is not None and (img.width(), img.height()) == (W, H)
    assert overlay._host_qimage() is img


def test_the_notes_sheet_has_pixels_to_annotate():
    overlay = _overlay(_FakeTexture())
    assert overlay.printing_notes_sheet() is not None


def test_nothing_is_read_back_while_no_instrument_asks():
    """The readback is what the GPU display path exists to avoid."""
    tex = _FakeTexture()
    overlay = _overlay(tex)
    pixmap = QPixmap(W, H)
    painter = QPainter(pixmap)
    overlay._draw_ui(painter)
    painter.end()

    assert tex.readbacks == 0


def test_dropping_the_texture_stops_the_instruments():
    overlay = _overlay(_FakeTexture())
    overlay.drop_gpu_texture()

    assert overlay._host_buffer() is None
    assert overlay._host_qimage() is None


def test_a_failed_readback_does_not_break_the_paint():
    tex = _FakeTexture()
    tex.readback = lambda: (_ for _ in ()).throw(RuntimeError("device lost"))
    overlay = _overlay(tex)

    assert overlay._host_buffer() is None


def test_the_cpu_path_keeps_using_its_own_buffer():
    overlay = CanvasOverlay(AppState())
    buf = np.full((H, W, 3), 0.5, dtype=np.float32)
    overlay.update_buffer(buf, "Adobe RGB")

    assert overlay._host_buffer() is buf
    assert overlay._host_qimage() is overlay._qimage
