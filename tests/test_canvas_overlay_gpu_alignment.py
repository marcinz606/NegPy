"""The overlay places the image where the GPU shader draws it, at every zoom and pan.

The shader (gpu_widget.vs_main) fits the image into the area above the toolbar reserve,
then zooms about the widget center and pans. The overlay paints the before/after
baseline, the crop box and masks into its own view rect, so any disagreement shows as an
offset between those and the image once the reserve is non-zero and zoom is not 1.
"""

import sys

import pytest
from PyQt6.QtWidgets import QApplication

from negpy.desktop.session import AppState
from negpy.desktop.view.canvas.overlay import CanvasOverlay

if not QApplication.instance():
    _app = QApplication(sys.argv)

W, H, IMG_W, IMG_H = 1000, 800, 1600, 1064


def _shader_rect(reserve: float, zoom: float, pan_x: float, pan_y: float) -> tuple:
    """Logical (x, y, w, h) the shader draws, from _draw_frame's uniforms and vs_main."""
    fit_h = H - reserve
    r = min(W / IMG_W, fit_h / IMG_H)
    nw, nh = IMG_W * r, IMG_H * r
    left, top = (1.0 - nw / W) - 1.0, 1.0 - ((fit_h - nh) / 2.0 / H) * 2.0
    ndc_left = left * zoom + pan_x * 2.0
    ndc_top = top * zoom - pan_y * 2.0
    x = (ndc_left + 1.0) / 2.0 * W
    y = (1.0 - ndc_top) / 2.0 * H
    return x, y, nw * zoom, nh * zoom


@pytest.mark.parametrize("reserve", [0.0, 56.0])
@pytest.mark.parametrize("zoom", [0.5, 1.0, 1.5, 3.0])
@pytest.mark.parametrize("pan", [(0.0, 0.0), (0.1, -0.2)])
def test_overlay_view_rect_matches_the_shader(reserve, zoom, pan):
    overlay = CanvasOverlay(AppState())
    overlay.resize(W, H)
    overlay._current_size = (IMG_W, IMG_H)
    overlay.fit_height_reserve = reserve
    overlay.set_transform(zoom, *pan)
    rect = overlay._view_rect
    expected = _shader_rect(reserve, zoom, *pan)
    assert (rect.x(), rect.y(), rect.width(), rect.height()) == pytest.approx(expected, abs=1e-6)
