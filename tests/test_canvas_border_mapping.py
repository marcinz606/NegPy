"""Tool coordinates must be content-normalized when the preview is padded
with the print border/mat (finish.border_size / paper aspect layout)."""

from types import SimpleNamespace

import numpy as np
from PyQt6.QtCore import QEvent, QPointF, QRectF, Qt
from PyQt6.QtGui import QMouseEvent

from negpy.desktop.session import AppState, ToolMode
from negpy.desktop.view.canvas.overlay import CanvasOverlay
from negpy.features.retouch.models import HEAL_SIZE_REF

# Padded display buffer 200x160 shown 1:1; image content inset by the border.
_DISPLAY = (200, 160)
_CONTENT = (20, 16, 160, 128)


def _mouse_event(kind: QEvent.Type, pos: QPointF, buttons=Qt.MouseButton.LeftButton) -> QMouseEvent:
    return QMouseEvent(kind, pos, Qt.MouseButton.LeftButton, buttons, Qt.KeyboardModifier.NoModifier)


def _identity_uv(h: int, w: int) -> np.ndarray:
    u, v = np.meshgrid(np.linspace(0, 1, w, dtype=np.float32), np.linspace(0, 1, h, dtype=np.float32))
    return np.ascontiguousarray(np.stack([u, v], axis=-1))


def _bordered_overlay(with_parent: bool = False) -> CanvasOverlay:
    if with_parent:
        from PyQt6.QtWidgets import QWidget

        parent = QWidget()
        parent._is_panning = False
        overlay = CanvasOverlay(AppState(), parent)
        overlay._test_parent = parent
    else:
        overlay = CanvasOverlay(AppState())
    overlay._view_rect = QRectF(0, 0, *_DISPLAY)
    overlay._current_size = _DISPLAY
    overlay._content_rect = _CONTENT
    return overlay


def test_heal_click_on_bordered_preview_emits_content_coords() -> None:
    overlay = _bordered_overlay(with_parent=True)
    overlay.set_tool_mode(ToolMode.DUST_PICK)
    clicks: list = []
    overlay.clicked.connect(lambda x, y: clicks.append((x, y)))

    overlay.mousePressEvent(_mouse_event(QEvent.Type.MouseButtonPress, QPointF(60, 48)))
    overlay.mouseReleaseEvent(_mouse_event(QEvent.Type.MouseButtonRelease, QPointF(60, 48), Qt.MouseButton.NoButton))

    assert len(clicks) == 1
    assert abs(clicks[0][0] - 0.25) < 1e-6 and abs(clicks[0][1] - 0.25) < 1e-6


def test_click_on_border_mat_is_ignored() -> None:
    overlay = _bordered_overlay(with_parent=True)
    overlay.set_tool_mode(ToolMode.DUST_PICK)
    clicks: list = []
    strokes: list = []
    overlay.clicked.connect(lambda x, y: clicks.append((x, y)))
    overlay.scratch_completed.connect(strokes.append)

    overlay.mousePressEvent(_mouse_event(QEvent.Type.MouseButtonPress, QPointF(10, 8)))
    overlay.mouseReleaseEvent(_mouse_event(QEvent.Type.MouseButtonRelease, QPointF(10, 8), Qt.MouseButton.NoButton))

    assert clicks == []
    assert strokes == []


def test_raw_to_screen_lands_inside_content() -> None:
    overlay = _bordered_overlay()
    uv = _identity_uv(64, 80)  # one uv cell = 2px on screen

    pt = overlay._raw_to_screen(0.25, 0.25, uv)

    assert abs(pt.x() - 60.0) <= 2.5
    assert abs(pt.y() - 48.0) <= 2.5


def test_forward_reverse_roundtrip_with_border() -> None:
    # Consistency guard: marker must draw back at the click point.
    overlay = _bordered_overlay(with_parent=True)
    overlay.set_tool_mode(ToolMode.DUST_PICK)
    clicks: list = []
    overlay.clicked.connect(lambda x, y: clicks.append((x, y)))

    start = QPointF(108, 100)
    overlay.mousePressEvent(_mouse_event(QEvent.Type.MouseButtonPress, start))
    overlay.mouseReleaseEvent(_mouse_event(QEvent.Type.MouseButtonRelease, start, Qt.MouseButton.NoButton))

    assert len(clicks) == 1
    back = overlay._raw_to_screen(clicks[0][0], clicks[0][1], _identity_uv(64, 80))
    assert abs(back.x() - start.x()) <= 2.5
    assert abs(back.y() - start.y()) <= 2.5


def test_lasso_vertices_content_normalized() -> None:
    overlay = _bordered_overlay()
    overlay.set_tool_mode(ToolMode.LOCAL_DRAW)
    overlay._lasso_drawing = True
    overlay._lasso_pts = [QPointF(20, 16), QPointF(180, 16), QPointF(100, 144)]
    emitted: list = []
    overlay.local_mask_created.connect(lambda _shape, pts: emitted.append(pts))

    overlay._finish_lasso()

    assert len(emitted) == 1
    expected = [(0.0, 0.0), (1.0, 0.0), (0.5, 1.0)]
    for (nx, ny), (ex, ey) in zip(emitted[0], expected):
        assert abs(nx - ex) < 1e-6 and abs(ny - ey) < 1e-6


def test_mask_vertex_release_content_normalized() -> None:
    overlay = _bordered_overlay(with_parent=True)
    overlay.state.local_selected_mask = 0
    overlay._local_drag_vertex = 0
    overlay._local_edit_verts = [QPointF(60, 48), QPointF(100, 48), QPointF(80, 80)]
    emitted: list = []
    overlay.local_mask_edited.connect(lambda i, vp: emitted.append((i, vp)))

    overlay.mouseReleaseEvent(_mouse_event(QEvent.Type.MouseButtonRelease, QPointF(60, 48), Qt.MouseButton.NoButton))

    assert len(emitted) == 1
    index, vp = emitted[0]
    assert index == 0
    assert abs(vp[0][0] - 0.25) < 1e-6 and abs(vp[0][1] - 0.25) < 1e-6


def test_brush_radius_scales_with_content() -> None:
    overlay = _bordered_overlay()
    assert abs(overlay._brush_screen_radius(12.0) - 12.0 / (2.0 * HEAL_SIZE_REF) * 160.0) < 1e-6


def test_densitometer_not_double_compensated() -> None:
    # Overlay emits content-normalized coords; the controller must not subtract
    # content_rect a second time.
    from negpy.desktop.controller import AppController

    nl = np.zeros((80, 100, 3), dtype=np.float32)
    nl[20, 25] = 0.7
    bounds = SimpleNamespace(floors=(0.0, 0.0, 0.0), ceils=(1.0, 1.0, 1.0))
    stub = SimpleNamespace(
        state=SimpleNamespace(
            last_metrics={"normalized_log": nl, "final_bounds": bounds},
            active_tool=ToolMode.NONE,
        ),
        canvas=SimpleNamespace(
            display_size=lambda: _DISPLAY,
            content_rect=lambda: _CONTENT,
        ),
    )
    stub._sample_normalized_log = AppController._sample_normalized_log.__get__(stub)

    reading = AppController._compute_densitometer_reading(stub, 0.25, 0.25, (0.5, 0.5, 0.5))

    assert reading is not None
    assert all(abs(d - 0.7) < 1e-6 for d in reading.dd_rgb)


def test_no_border_fallback_unchanged() -> None:
    overlay = _bordered_overlay()
    overlay._content_rect = None

    assert overlay._map_to_image_coords(QPointF(50, 40)) == (0.25, 0.25)
