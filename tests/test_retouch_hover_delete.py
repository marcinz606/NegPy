from dataclasses import replace
from unittest.mock import MagicMock, patch

import numpy as np
from PyQt6.QtCore import QEvent, QPointF, QRectF, Qt
from PyQt6.QtGui import QEnterEvent, QImage, QKeyEvent, QMouseEvent, QPainter
from PyQt6.QtWidgets import QApplication, QDockWidget, QDoubleSpinBox, QLineEdit, QMainWindow, QSlider, QVBoxLayout, QWidget

from negpy.desktop.session import RETOUCH_TOOLS, AppState, ToolMode
from negpy.desktop.view.canvas.overlay import CanvasOverlay
from negpy.desktop.view.canvas.widget import ImageCanvas
from negpy.desktop.view.keyboard_shortcuts import RetouchDeleteKeyFilter
from negpy.desktop.view.sidebar.files import FileBrowser


def _uv_grid(h: int = 60, w: int = 80) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    return np.stack([xx / (w - 1), yy / (h - 1)], -1)


def _move_event(pos: QPointF) -> QMouseEvent:
    return QMouseEvent(
        QEvent.Type.MouseMove,
        pos,
        Qt.MouseButton.NoButton,
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
    )


def _overlay(**retouch_fields) -> CanvasOverlay:
    parent = QWidget()
    parent._is_panning = False  # type: ignore[attr-defined]
    parent._space_pan_held = False  # type: ignore[attr-defined]
    state = AppState()
    state.config = replace(state.config, retouch=replace(state.config.retouch, **retouch_fields))
    state.last_metrics["uv_grid"] = _uv_grid()
    overlay = CanvasOverlay(state, parent)
    overlay.resize(100, 100)
    overlay._view_rect = QRectF(0, 0, 100, 100)
    overlay._test_parent = parent  # type: ignore[attr-defined]
    return overlay


def _paint(overlay: CanvasOverlay) -> QImage:
    img = QImage(100, 100, QImage.Format.Format_ARGB32_Premultiplied)
    img.fill(0)
    painter = QPainter(img)
    overlay._draw_placed_heals(painter)
    painter.end()
    return img


def _paint_ui(overlay: CanvasOverlay) -> QImage:
    img = QImage(100, 100, QImage.Format.Format_ARGB32_Premultiplied)
    img.fill(0)
    painter = QPainter(img)
    overlay._draw_ui(painter)
    painter.end()
    return img


def test_hover_highlights_repair_clears_on_leave_and_restores_on_enter() -> None:
    stroke = ([[0.3, 0.3], [0.5, 0.3], [0.7, 0.3]], 12.0, 0.0, 0.0)
    overlay = _overlay(manual_heal_strokes=[stroke])
    overlay.set_tool_mode(ToolMode.DUST_PICK)

    at_rest = _paint(overlay)
    assert overlay._hovered_heal is None

    overlay.mouseMoveEvent(_move_event(QPointF(50.0, 30.0)))
    assert overlay._hovered_heal == ("stroke", 0)
    hovered = _paint(overlay)
    assert hovered != at_rest

    overlay.leaveEvent(QEvent(QEvent.Type.Leave))
    assert overlay._hovered_heal is None
    assert _paint(overlay) == at_rest

    enter = QEnterEvent(QPointF(50.0, 30.0), QPointF(50.0, 30.0), QPointF(50.0, 30.0))
    overlay.enterEvent(enter)
    assert overlay._hovered_heal == ("stroke", 0)


def test_menu_pin_preserves_hover_across_leave_event() -> None:
    spot = ([[0.4, 0.4]], 10.0, 0.0, 0.0)
    overlay = _overlay(manual_heal_strokes=[spot])
    overlay.set_tool_mode(ToolMode.DUST_PICK)

    overlay.set_menu_pos(QPointF(40.0, 40.0))
    overlay.leaveEvent(QEvent(QEvent.Type.Leave))
    assert overlay._hovered_heal == ("stroke", 0)
    assert overlay.delete_hovered_heal() is False

    overlay.set_menu_pos(None)
    assert overlay._hovered_heal is None


def test_uv_flip_with_unchanged_retouch_config_refreshes_hover_target() -> None:
    left = ([[0.25, 0.5]], 10.0, 0.0, 0.0)
    right = ([[0.75, 0.5]], 10.0, 0.0, 0.0)
    overlay = _overlay(manual_heal_strokes=[left, right])
    overlay.set_tool_mode(ToolMode.DUST_PICK)

    overlay.mouseMoveEvent(_move_event(QPointF(25.0, 50.0)))
    assert overlay._hovered_heal == ("stroke", 0)
    _paint(overlay)

    # Flip uv_grid horizontally without changing RetouchConfig or _view_rect.
    overlay.state.last_metrics["uv_grid"] = np.ascontiguousarray(_uv_grid()[:, ::-1])
    _paint(overlay)
    assert overlay._hovered_heal == ("stroke", 1)


def test_curved_catmull_rom_stroke_hits_rendered_arc_not_raw_chord() -> None:
    # Control points form a cap whose middle span bows well below y=50 under Catmull-Rom.
    curved = ([[0.0, 0.0], [0.3, 0.5], [0.7, 0.5], [1.0, 0.0]], 4.0, 0.0, 0.0)
    overlay = _overlay(manual_heal_strokes=[curved])
    overlay.resize(300, 300)
    overlay._view_rect = QRectF(0, 0, 300, 300)
    overlay.set_tool_mode(ToolMode.DUST_PICK)

    entries = overlay._placed_heal_geometry(overlay.state.config.retouch, overlay.state.last_metrics["uv_grid"])
    pts = entries[0][3]
    apex_y = max(p.y() for p in pts)
    assert apex_y > 165.0
    sample = next(p for p in pts if 110.0 <= p.x() <= 130.0)
    assert overlay.heal_hit_test(sample) == ("stroke", 0)
    assert overlay.heal_hit_test(QPointF(sample.x(), 140.0)) is None


def test_spots_and_strokes_share_uniform_pin_at_rest_and_reveal_capsule_on_hover() -> None:
    spot = ([[0.5, 0.5]], 24.0, 0.0, 0.0)
    stroke = ([[0.1, 0.5], [0.5, 0.5], [0.9, 0.5]], 12.0, 0.0, 0.0)
    clone = ([[0.5, 0.5]], 24.0, 0.1, 0.1, 1.0, 0.5, True)
    spot_overlay = _overlay(manual_heal_strokes=[spot])
    stroke_overlay = _overlay(manual_heal_strokes=[stroke])
    clone_overlay = _overlay(clone_strokes=[clone])
    for ov in (spot_overlay, stroke_overlay, clone_overlay):
        ov.state.last_metrics["uv_grid"] = _uv_grid(61, 81)
        ov.set_tool_mode(ToolMode.DUST_PICK)

    spot_rest = _paint(spot_overlay)
    stroke_rest = _paint(stroke_overlay)
    clone_rest = _paint(clone_overlay)
    assert spot_rest == stroke_rest
    assert clone_rest != spot_rest
    assert stroke_rest.pixel(15, 50) == 0
    assert stroke_rest.pixel(50, 50) != 0
    # Standalone -45° nib has no surrounding circle on the orthogonal (-6, -6) diagonal.
    assert stroke_rest.pixel(44, 44) == 0
    assert clone_rest.pixel(44, 44) == 0

    stroke_overlay.mouseMoveEvent(_move_event(QPointF(15.0, 50.0)))
    assert stroke_overlay._hovered_heal == ("stroke", 0)
    stroke_hovered = _paint(stroke_overlay)
    assert stroke_hovered.pixel(15, 50) != 0

    # Resting pins hide while painting a stroke.
    stroke_overlay._heal_drag_pts = [QPointF(10.0, 10.0), QPointF(30.0, 10.0)]
    assert _paint(stroke_overlay).pixel(50, 50) == 0


def test_pin_hit_radius_prioritizes_small_spot_over_wide_overlapping_stroke() -> None:
    wide = ([[0.1, 0.5], [0.9, 0.5]], 64.0, 0.0, 0.0)
    tiny_spot = ([[0.3, 0.52]], 2.0, 0.0, 0.0)
    overlay = _overlay(manual_heal_strokes=[wide, tiny_spot])
    overlay.set_tool_mode(ToolMode.DUST_PICK)

    assert overlay.heal_hit_test(QPointF(30.0, 52.0)) == ("stroke", 1)
    assert overlay.heal_hit_test(QPointF(75.0, 50.0)) == ("stroke", 0)


def test_repairs_hide_and_refuse_delete_under_peeks_strips_and_compare_before() -> None:
    left = ([[0.25, 0.5]], 10.0, 0.0, 0.0)
    right = ([[0.75, 0.5]], 10.0, 0.0, 0.0)
    overlay = _overlay(manual_heal_strokes=[left, right])
    overlay.set_tool_mode(ToolMode.DUST_PICK)
    overlay.mouseMoveEvent(_move_event(QPointF(25.0, 50.0)))
    assert overlay._hovered_heal == ("stroke", 0)

    for flag in ("test_strip", "compare_hold", "negative_peek", "flat_peek", "flatfield_peek", "embedded_peek"):
        setattr(overlay.state, flag, True)
        assert overlay.retouch_hit_at(QPointF(25.0, 50.0)) is None
        assert overlay.delete_hovered_heal() is False
        setattr(overlay.state, flag, False)

    # Compare split: left half (x < 50) is BEFORE, right half (x >= 50) is AFTER.
    overlay.state.compare_mode = True
    overlay.state.compare_split = 0.5
    overlay.state.compare_before = np.zeros((10, 10, 3), dtype=np.float32)
    assert overlay.retouch_hit_at(QPointF(25.0, 50.0)) is None
    assert overlay.delete_hovered_heal() is False

    overlay.mouseMoveEvent(_move_event(QPointF(75.0, 50.0)))
    assert overlay._hovered_heal == ("stroke", 1)
    overlay.state.compare_mode = False

    # IR dust overlay draws underneath placed heals so hover highlight remains visible.
    overlay.state.dust_overlay_mode = "ir"
    overlay.state.preview_ir = np.full((20, 20), 0.5, dtype=np.float32)
    overlay.mouseMoveEvent(_move_event(QPointF(25.0, 50.0)))
    with_heals = _paint_ui(overlay)
    overlay.state.config = replace(overlay.state.config, retouch=replace(overlay.state.config.retouch, manual_heal_strokes=[]))
    ir_only = _paint_ui(overlay)
    assert with_heals != ir_only


def test_hit_test_and_paint_clip_to_content_rect_inside_print_border() -> None:
    edge_spot = ([[0.02, 0.5]], 24.0, 0.0, 0.0)
    overlay = _overlay(manual_heal_strokes=[edge_spot])
    overlay._content_rect = (20, 20, 60, 60)
    overlay._current_size = (100, 100)
    overlay.set_tool_mode(ToolMode.DUST_PICK)

    # Border area outside _content_view_rect() neither hits nor receives painted bleed.
    assert overlay.heal_hit_test(QPointF(18.0, 50.0)) is None
    painted = _paint(overlay)
    assert painted.pixel(15, 50) == 0


def test_space_pan_suppresses_hover_and_delete() -> None:
    spot = ([[0.5, 0.5]], 10.0, 0.0, 0.0)
    overlay = _overlay(manual_heal_strokes=[spot])
    overlay.set_tool_mode(ToolMode.DUST_PICK)
    overlay.mouseMoveEvent(_move_event(QPointF(50.0, 50.0)))
    assert overlay._hovered_heal == ("stroke", 0)

    overlay._test_parent._space_pan_held = True  # type: ignore[attr-defined]
    overlay.sync_retouch_hover()
    assert overlay._hovered_heal is None
    assert overlay.delete_hovered_heal() is False


def test_delete_and_backspace_remove_hovered_repairs_in_stack_order() -> None:
    first = ([[0.4, 0.4]], 10.0, 0.0, 0.0)
    second = ([[0.4, 0.4]], 10.0, 0.0, 0.0)
    overlay = _overlay(manual_heal_strokes=[first, second])
    overlay.set_tool_mode(ToolMode.DUST_PICK)

    deleted: list[tuple[str, int]] = []

    def _apply_delete(kind: str, index: int) -> None:
        deleted.append((kind, index))
        strokes = list(overlay.state.config.retouch.manual_heal_strokes)
        strokes.pop(index)
        overlay.state.config = replace(
            overlay.state.config,
            retouch=replace(overlay.state.config.retouch, manual_heal_strokes=strokes),
        )

    overlay.heal_delete_requested.connect(_apply_delete)
    overlay.mouseMoveEvent(_move_event(QPointF(40.0, 40.0)))
    assert overlay._hovered_heal == ("stroke", 1)

    assert overlay.delete_hovered_heal() is True
    assert deleted == [("stroke", 1)]
    assert overlay._hovered_heal == ("stroke", 0)

    overlay.handle_retouch_backspace()
    assert deleted == [("stroke", 1), ("stroke", 0)]
    assert overlay._hovered_heal is None
    assert overlay.delete_hovered_heal() is False


def test_backspace_undoes_scratch_point_and_suppresses_menu_hit_while_drawing() -> None:
    spot = ([[0.3, 0.3]], 10.0, 0.0, 0.0)
    overlay = _overlay(manual_heal_strokes=[spot])
    overlay.set_tool_mode(ToolMode.SCRATCH_PICK)
    deleted: list[tuple[str, int]] = []
    overlay.heal_delete_requested.connect(lambda k, i: deleted.append((k, i)))

    overlay.mouseMoveEvent(_move_event(QPointF(30.0, 30.0)))
    assert overlay._hovered_heal == ("stroke", 0)

    overlay._scratch_pts = [QPointF(10.0, 10.0), QPointF(30.0, 30.0)]
    assert overlay.retouch_hit_at(QPointF(30.0, 30.0)) is None
    overlay.handle_retouch_backspace()
    assert len(overlay._scratch_pts) == 1
    assert deleted == []

    overlay.handle_retouch_backspace()
    assert overlay._scratch_pts == []
    assert deleted == []
    assert overlay._hovered_heal == ("stroke", 0)

    overlay.handle_retouch_backspace()
    assert deleted == [("stroke", 0)]


def test_key_filter_routes_from_floating_dock_blocks_autorepeat_and_respects_inputs(qapp) -> None:
    win = QMainWindow()
    state = AppState()
    spot1 = ([[0.4, 0.4]], 10.0, 0.0, 0.0)
    spot2 = ([[0.4, 0.4]], 10.0, 0.0, 0.0)
    state.config = replace(state.config, retouch=replace(state.config.retouch, manual_heal_strokes=[spot1, spot2]))
    state.last_metrics["uv_grid"] = _uv_grid()
    state.active_tool = ToolMode.DUST_PICK
    win.state = state  # type: ignore[attr-defined]

    central = QWidget(win)
    layout = QVBoxLayout(central)
    canvas = ImageCanvas(state, central)
    canvas.resize(100, 100)
    canvas.overlay.resize(100, 100)
    canvas.overlay._view_rect = QRectF(0, 0, 100, 100)
    canvas.overlay.set_tool_mode(ToolMode.DUST_PICK)
    slider = QSlider(Qt.Orientation.Horizontal, central)
    line_edit = QLineEdit(central)
    spin = QDoubleSpinBox(central)
    for w in (canvas, slider, line_edit, spin):
        layout.addWidget(w)
    win.setCentralWidget(central)
    win.canvas = canvas  # type: ignore[attr-defined]

    dock = QDockWidget("Controls", win)
    dock.setFloating(True)
    win.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)
    win.show()
    dock.show()
    qapp.processEvents()
    canvas.overlay._current_size = (100, 100)
    canvas.overlay.resize(100, 100)
    canvas.overlay._view_rect = QRectF(0, 0, 100, 100)

    deleted: list[tuple[str, int]] = []
    canvas.overlay.heal_delete_requested.connect(lambda k, i: deleted.append((k, i)))
    key_filter = RetouchDeleteKeyFilter(win)
    try:
        with patch.object(QApplication, "activeWindow", return_value=dock), patch.object(QApplication, "focusWidget", return_value=slider):
            # Moving over the canvas does not steal focus from slider, and floating dock still routes Delete.
            canvas.overlay.mouseMoveEvent(_move_event(QPointF(40.0, 40.0)))
            assert canvas.overlay._hovered_heal == ("stroke", 1)

            override = QKeyEvent(QEvent.Type.ShortcutOverride, Qt.Key.Key_Delete, Qt.KeyboardModifier.NoModifier)
            assert key_filter.eventFilter(slider, override) is True
            assert override.isAccepted()

            # Auto-repeat press is swallowed without firing delete.
            repeat = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Delete, Qt.KeyboardModifier.NoModifier, "", True, 1)
            assert key_filter.eventFilter(slider, repeat) is True
            assert deleted == []

            press = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Delete, Qt.KeyboardModifier.NoModifier, "", False, 1)
            assert key_filter.eventFilter(slider, press) is True
            assert deleted == [("stroke", 1)]

        # Focused text and spin inputs keep Delete/Backspace and suppress hover highlight.
        for editor in (line_edit, spin):
            with (
                patch.object(QApplication, "activeWindow", return_value=win),
                patch.object(QApplication, "focusWidget", return_value=editor),
            ):
                canvas.overlay.sync_retouch_hover()
                assert canvas.overlay._hovered_heal is None
                press = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Backspace, Qt.KeyboardModifier.NoModifier)
                assert key_filter.eventFilter(editor, press) is False
    finally:
        key_filter.uninstall()
        dock.close()
        win.close()


def test_hover_delete_works_across_lines_and_clones_only_in_retouch_tools() -> None:
    clone = ([[0.2, 0.2], [0.3, 0.2]], 10.0, 0.1, 0.1, 1.0, 0.5, True)
    line = (0.1, 0.7, 0.9, 0.7, 8.0)
    overlay = _overlay(clone_strokes=[clone], scratch_lines=[line])
    deleted: list[tuple[str, int]] = []
    overlay.heal_delete_requested.connect(lambda k, i: deleted.append((k, i)))

    overlay.set_tool_mode(ToolMode.NONE)
    overlay.mouseMoveEvent(_move_event(QPointF(25.0, 20.0)))
    assert overlay._hovered_heal is None
    assert overlay.delete_hovered_heal() is False

    overlay.set_tool_mode(ToolMode.CLONE)
    overlay.mouseMoveEvent(_move_event(QPointF(25.0, 20.0)))
    assert overlay._hovered_heal == ("clone", 0)
    assert overlay.delete_hovered_heal() is True
    assert deleted == [("clone", 0)]

    overlay.set_tool_mode(ToolMode.SCRATCH_LINE)
    overlay.mouseMoveEvent(_move_event(QPointF(50.0, 70.0)))
    assert overlay._hovered_heal == ("line", 0)
    assert overlay.delete_hovered_heal() is True
    assert deleted == [("clone", 0), ("line", 0)]


def test_canvas_forwards_hover_delete_and_line_menu_to_controller(qapp) -> None:
    state = AppState()
    state.selected_file_idx = 0
    state.last_metrics["uv_grid"] = _uv_grid()
    canvas = ImageCanvas(state)
    controller = MagicMock()
    canvas.set_controller(controller)

    canvas.overlay.heal_delete_requested.emit("stroke", 2)
    controller.delete_heal.assert_called_once_with("stroke", 2)

    canvas.overlay.heal_delete_requested.emit("clone", 1)
    controller.delete_clone.assert_called_once_with(1)

    state.active_tool = ToolMode.SCRATCH_LINE
    called: list = []
    canvas._exec_retouch_menu = lambda pos, gp: called.append((pos, gp))  # type: ignore[method-assign]
    canvas.show_canvas_menu(QPointF(10.0, 10.0), QPointF(10.0, 10.0).toPoint())
    assert len(called) == 1

    # Test strip suppresses the canvas right-click menu.
    state.test_strip = True
    canvas.show_canvas_menu(QPointF(10.0, 10.0), QPointF(10.0, 10.0).toPoint())
    assert len(called) == 1


def test_film_strip_delete_key_never_unloads_frame_during_retouch() -> None:
    dummy = MagicMock(spec=FileBrowser)
    dummy.session = MagicMock()
    state = AppState()
    state.uploaded_files = [{"path": "frame01.tif"}]
    state.selected_file_idx = 0
    dummy.session.state = state

    for mode in RETOUCH_TOOLS:
        state.active_tool = mode
        FileBrowser._on_delete_key(dummy)
        dummy._on_remove_from_menu.assert_not_called()

    state.active_tool = ToolMode.NONE
    FileBrowser._on_delete_key(dummy)
    dummy._on_remove_from_menu.assert_called_once()


def test_hover_recovers_after_full_split_sweep_and_popup_close() -> None:
    right = ([[0.75, 0.5]], 10.0, 0.0, 0.0)
    overlay = _overlay(manual_heal_strokes=[right])
    overlay.set_tool_mode(ToolMode.DUST_PICK)
    overlay.mouseMoveEvent(_move_event(QPointF(75.0, 50.0)))

    overlay.state.compare_mode = True
    overlay.state.compare_before = np.zeros((10, 10, 3), dtype=np.float32)
    overlay.state.compare_split = 0.5
    _paint(overlay)
    assert overlay._hovered_heal == ("stroke", 0)

    overlay.state.compare_split = 1.0
    _paint(overlay)
    assert overlay._hovered_heal is None

    overlay.state.compare_split = 0.5
    _paint(overlay)
    assert overlay._hovered_heal == ("stroke", 0)

    popup = QWidget()
    with patch.object(QApplication, "activePopupWidget", return_value=popup):
        _paint(overlay)
        assert overlay._hovered_heal is None
    _paint(overlay)
    assert overlay._hovered_heal == ("stroke", 0)


def test_ir_overlay_draws_under_crop_tool_and_peeks_block_exclusion_drag() -> None:
    overlay = _overlay(dust_remove=True)
    overlay.state.dust_overlay_mode = "ir"
    overlay.state.preview_ir = np.full((20, 20), 0.5, dtype=np.float32)
    overlay.set_tool_mode(ToolMode.NONE)
    ir_only = _paint_ui(overlay)

    overlay.set_tool_mode(ToolMode.CROP_MANUAL)
    overlay._crop_rect_norm = (0.2, 0.2, 0.8, 0.8)
    with_crop = _paint_ui(overlay)
    assert with_crop != ir_only

    overlay.set_tool_mode(ToolMode.DUST_PICK)
    overlay.state.negative_peek = True
    right_press = QMouseEvent(
        QEvent.Type.MouseButtonPress,
        QPointF(50.0, 50.0),
        Qt.MouseButton.RightButton,
        Qt.MouseButton.RightButton,
        Qt.KeyboardModifier.NoModifier,
    )
    overlay.mousePressEvent(right_press)
    assert overlay._exclude_drag_pts == []


def test_key_filter_uninstalls_cleanly_when_overlay_is_deleted_without_close_event(qapp) -> None:
    from PyQt6 import sip

    win = QMainWindow()
    state = AppState()
    win.state = state  # type: ignore[attr-defined]
    canvas = ImageCanvas(state, win)
    win.setCentralWidget(canvas)
    win.canvas = canvas  # type: ignore[attr-defined]
    key_filter = RetouchDeleteKeyFilter(win)

    sip.delete(canvas)
    # Must not raise RuntimeError on deleted C++ overlay.
    key_filter._on_focus_changed(None, None)
    press = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Delete, Qt.KeyboardModifier.NoModifier)
    assert key_filter.eventFilter(win, press) is False
    sip.delete(win)


def test_brush_circle_contact_triggers_highlight_and_prefers_contained_repair() -> None:
    spot = ([[0.5, 0.5]], 20.0, 0.0, 0.0)
    overlay = _overlay(manual_heal_strokes=[spot], manual_dust_size=12)
    overlay.resize(1600, 1600)
    overlay._view_rect = QRectF(0, 0, 1600, 1600)
    overlay.set_tool_mode(ToolMode.DUST_PICK)

    _kind, _i, center, _pts, repaired_r = overlay._placed_heal_geometry(
        overlay.state.config.retouch, overlay.state.last_metrics["uv_grid"]
    )[0]
    small_brush_r = overlay._brush_screen_radius(12)
    large_brush_r = overlay._brush_screen_radius(36)
    assert repaired_r == 10.0
    assert small_brush_r == 6.0
    assert large_brush_r == 18.0

    # At d = 25 (gap = 15 > small_brush_r = 6): outside brush circle contact.
    overlay.mouseMoveEvent(_move_event(QPointF(center.x() + 25.0, center.y())))
    assert overlay._hovered_heal is None

    # Growing brush size to 36 (large_brush_r = 18 >= gap = 15) lights up on repaint without moving mouse.
    overlay.state.config = replace(
        overlay.state.config,
        retouch=replace(overlay.state.config.retouch, manual_dust_size=36),
    )
    _paint(overlay)
    assert overlay._hovered_heal == ("stroke", 0)

    # Just outside repaired_r (10) + large_brush_r (18) = 28 px clears again; just inside hits.
    overlay.mouseMoveEvent(_move_event(QPointF(center.x() + 28.5, center.y())))
    assert overlay._hovered_heal is None
    overlay.mouseMoveEvent(_move_event(QPointF(center.x() + 27.5, center.y())))
    assert overlay._hovered_heal == ("stroke", 0)

    # When cursor center sits inside a wide stroke while the brush rim grazes a nearby spot,
    # the repair directly under the cursor center wins over the grazed neighbor.
    wide = ([[0.2, 0.5], [0.5, 0.5]], 40.0, 0.0, 0.0)  # radius = 20 px
    nearby = ([[0.52, 0.5]], 4.0, 0.0, 0.0)  # radius = 2 px
    overlay.state.config = replace(
        overlay.state.config,
        retouch=replace(overlay.state.config.retouch, manual_heal_strokes=[wide, nearby], manual_dust_size=40),
    )
    entries = overlay._placed_heal_geometry(overlay.state.config.retouch, overlay.state.last_metrics["uv_grid"])
    wide_end = entries[0][3][-1]
    nearby_pin = entries[1][2]
    probe = QPointF((wide_end.x() + nearby_pin.x()) * 0.5, wide_end.y())
    assert overlay.heal_hit_test(probe) == ("stroke", 0)
