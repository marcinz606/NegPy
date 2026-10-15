import pytest
from dataclasses import replace

from PyQt6.QtCore import QEvent, QPointF, QRectF, Qt
from PyQt6.QtGui import QMouseEvent

from negpy.desktop.session import AppState, ToolMode
from negpy.desktop.view.canvas.overlay import CanvasOverlay
from negpy.features.local.models import LocalAdjustmentsConfig, LocalMask, MaskShape
from negpy.features.process.models import ProcessMode

_TRIANGLE = [QPointF(20, 20), QPointF(80, 20), QPointF(50, 80)]


def _move(pos: QPointF) -> QMouseEvent:
    return QMouseEvent(QEvent.Type.MouseMove, pos, Qt.MouseButton.NoButton, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)


def _release(pos: QPointF) -> QMouseEvent:
    return QMouseEvent(
        QEvent.Type.MouseButtonRelease, pos, Qt.MouseButton.LeftButton, Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier
    )


def _overlay_with_mask(tool: ToolMode = ToolMode.LOCAL_DRAW, shape: MaskShape = MaskShape.POLYGON) -> CanvasOverlay:
    from PyQt6.QtWidgets import QWidget

    parent = QWidget()  # The move path reads parent()._is_panning.
    parent._is_panning = False
    overlay = CanvasOverlay(AppState(), parent)
    overlay._test_parent = parent  # Keep the parent alive with the overlay.
    overlay._view_rect = QRectF(0, 0, 100, 100)
    overlay.set_tool_mode(tool)
    overlay.state.local_selected_mask = 0
    mask = LocalMask(vertices=tuple((p.x() / 100.0, p.y() / 100.0) for p in _TRIANGLE), shape=shape)
    overlay.state.config = replace(overlay.state.config, local=LocalAdjustmentsConfig(masks=(mask,)))
    # Normally set during paint.
    overlay._local_mask_screen_polys = [list(_TRIANGLE)]
    overlay._local_mask_screen_ctrl = [list(_TRIANGLE)]
    return overlay


def test_selected_mask_editable_without_draw_tool() -> None:
    # No Draw Mask tool active: pressing a vertex of the selected mask still edits it.
    overlay = _overlay_with_mask(ToolMode.NONE)
    ev = QMouseEvent(
        QEvent.Type.MouseButtonPress,
        QPointF(20, 20),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    overlay.mousePressEvent(ev)
    assert overlay._local_drag_vertex == 0
    assert ev.isAccepted()


def test_a_slide_mask_cannot_be_edited_on_the_canvas() -> None:
    """A slide prints no masks and grays the panel, so the canvas must not edit them either."""
    overlay = _overlay_with_mask(ToolMode.NONE)
    cfg = overlay.state.config
    overlay.state.config = replace(cfg, process=replace(cfg.process, process_mode=ProcessMode.E6))
    ev = QMouseEvent(
        QEvent.Type.MouseButtonPress,
        QPointF(20, 20),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    overlay.mousePressEvent(ev)

    assert overlay._local_drag_vertex is None
    assert overlay.try_delete_local_vertex(QPointF(80, 20)) is False


def test_press_grabs_mask_vertex() -> None:
    overlay = _overlay_with_mask()
    overlay._handle_lasso_press(QPointF(20, 20))  # on vertex 0
    assert overlay._local_drag_vertex == 0
    assert overlay._local_edit_verts is not None and len(overlay._local_edit_verts) == 3
    assert overlay._lasso_drawing is False  # did not start a fresh shape


def test_press_on_edge_midpoint_inserts_point() -> None:
    overlay = _overlay_with_mask()
    overlay._handle_lasso_press(QPointF(50, 20))  # midpoint of edge 0->1
    assert overlay._local_edit_verts is not None and len(overlay._local_edit_verts) == 4
    assert overlay._local_drag_vertex == 1  # inserted right after vertex 0


def test_right_click_deletes_vertex() -> None:
    overlay = _overlay_with_mask()
    emitted: list = []
    overlay.local_vertex_deleted.connect(lambda i, v: emitted.append((i, v)))
    assert overlay.try_delete_local_vertex(QPointF(80, 20)) is True  # vertex 1
    assert emitted == [(0, 1)]
    assert overlay.try_delete_local_vertex(QPointF(5, 5)) is False  # empty space


def test_press_selects_mask_when_off_handles() -> None:
    overlay = _overlay_with_mask()
    selected: list = []
    overlay.local_mask_selected.connect(selected.append)
    overlay._handle_lasso_press(QPointF(50, 45))  # inside, clear of vertices/midpoints
    assert selected == [0]
    assert overlay._local_drag_vertex is None


def test_fixed_arity_shapes_refuse_point_edits() -> None:
    # An oval has 3 points. No point insert and no point delete are possible.
    overlay = _overlay_with_mask(shape=MaskShape.OVAL)
    overlay._handle_lasso_press(QPointF(50, 20))  # midpoint of edge 0->1
    assert overlay._local_edit_verts is None
    assert overlay.try_delete_local_vertex(QPointF(80, 20)) is False


def test_dragging_an_ovals_centre_carries_its_axes() -> None:
    overlay = _overlay_with_mask(shape=MaskShape.OVAL)
    overlay._handle_lasso_press(QPointF(20, 20))  # the centre handle
    assert overlay._local_drag_anchor is not None

    overlay.mouseMoveEvent(
        QMouseEvent(
            QEvent.Type.MouseMove,
            QPointF(30, 25),
            Qt.MouseButton.NoButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
    )
    assert overlay._local_edit_verts == [QPointF(30, 25), QPointF(90, 25), QPointF(60, 85)]


def test_dragging_out_an_oval_emits_three_control_points() -> None:
    overlay = _overlay_with_mask(ToolMode.LOCAL_OVAL)
    emitted: list = []
    overlay.local_mask_created.connect(lambda shape, pts: emitted.append((shape, pts)))

    overlay._handle_shape_press(QPointF(90, 90))  # Away from the existing mask.
    overlay._finish_shape_draw(QPointF(50, 50))

    assert len(emitted) == 1
    shape, pts = emitted[0]
    assert shape == "oval"
    assert [(round(x, 3), round(y, 3)) for x, y in pts] == [(0.7, 0.7), (0.5, 0.7), (0.7, 0.5)]


def test_a_card_edge_can_be_drawn_off_the_frame() -> None:
    """The start of a tilted card edge must go past the corner it burns."""
    overlay = _overlay_with_mask(ToolMode.LOCAL_GRADIENT)
    emitted: list = []
    overlay.local_mask_created.connect(lambda shape, pts: emitted.append(pts))

    overlay._handle_shape_press(QPointF(130, -20))  # Outside the 100x100 content rect.
    overlay._finish_shape_draw(QPointF(60, 40))

    assert len(emitted) == 1
    assert [(round(x, 2), round(y, 2)) for x, y in emitted[0]] == [(1.3, -0.2), (0.6, 0.4)]


def test_a_dragged_handle_is_not_held_inside_the_frame() -> None:
    overlay = _overlay_with_mask(shape=MaskShape.OVAL)
    edits: list = []
    overlay.local_mask_edited.connect(lambda i, pts: edits.append(pts))

    overlay._handle_lasso_press(QPointF(80, 20))  # An axis handle.
    overlay.mouseMoveEvent(_move(QPointF(150, -30)))
    overlay.mouseReleaseEvent(_release(QPointF(150, -30)))

    assert len(edits) == 1
    assert edits[0][1] == (1.5, -0.3)


def test_a_shape_click_without_travel_draws_nothing() -> None:
    overlay = _overlay_with_mask(ToolMode.LOCAL_GRADIENT)
    emitted: list = []
    overlay.local_mask_created.connect(lambda shape, pts: emitted.append(pts))

    overlay._handle_shape_press(QPointF(90, 90))
    overlay._finish_shape_draw(QPointF(92, 91))

    assert emitted == []
    assert overlay._shape_draw_p1 is None


def test_hidden_masks_leave_no_handle_to_grab(qapp) -> None:
    """Off the Dodge & Burn tab the masks are not drawn, so a press there pans, never edits."""
    overlay = _overlay_with_mask(ToolMode.NONE)
    overlay.state.local_masks_shown = False
    from PyQt6.QtGui import QImage, QPainter

    image = QImage(100, 100, QImage.Format.Format_ARGB32)
    painter = QPainter(image)
    overlay._draw_ui(painter)  # A paint drops the handles of masks it does not draw.
    painter.end()

    ev = QMouseEvent(
        QEvent.Type.MouseButtonPress,
        QPointF(20, 20),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    overlay.mousePressEvent(ev)

    assert overlay._local_drag_vertex is None
    assert overlay.try_delete_local_vertex(QPointF(80, 20)) is False


def _uv(u0: float, u1: float, v0: float, v1: float, h: int = 50, w: int = 100):
    import numpy as np

    u, v = np.meshgrid(np.linspace(u0, u1, w, dtype=np.float32), np.linspace(v0, v1, h, dtype=np.float32))
    return np.stack([u, v], axis=-1)


def test_the_brush_ring_scales_with_the_whole_frame_not_the_crop(qapp) -> None:
    """A heal's size is a fraction of the source frame, so a 50% crop shows it twice as big."""
    overlay = _overlay_with_mask(ToolMode.NONE)
    overlay.state.original_res = (200, 400)
    overlay.state.last_metrics["uv_grid"] = _uv(0.0, 1.0, 0.0, 1.0)
    whole = overlay._brush_screen_radius(10.0)
    overlay.state.last_metrics["uv_grid"] = _uv(0.25, 0.75, 0.25, 0.75)
    cropped = overlay._brush_screen_radius(10.0)
    assert cropped == pytest.approx(2.0 * whole, rel=0.03)
