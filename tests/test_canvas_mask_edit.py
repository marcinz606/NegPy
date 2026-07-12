from PyQt6.QtCore import QEvent, QPointF, QRectF, Qt
from PyQt6.QtGui import QMouseEvent

from negpy.desktop.session import AppState, ToolMode
from negpy.desktop.view.canvas.overlay import CanvasOverlay

_TRIANGLE = [QPointF(20, 20), QPointF(80, 20), QPointF(50, 80)]


def _overlay_with_mask(tool: ToolMode = ToolMode.LOCAL_DRAW) -> CanvasOverlay:
    overlay = CanvasOverlay(AppState())
    overlay._view_rect = QRectF(0, 0, 100, 100)
    overlay.set_tool_mode(tool)
    overlay.state.local_selected_mask = 0
    overlay._local_mask_screen_polys = [list(_TRIANGLE)]  # normally set during paint
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
