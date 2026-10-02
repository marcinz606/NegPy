import pytest
from types import SimpleNamespace

from PyQt6.QtCore import QEvent, QPointF, QRectF, Qt
from PyQt6.QtGui import QKeyEvent, QMouseEvent
from PyQt6.QtWidgets import QLineEdit, QPushButton, QWidget

from negpy.desktop.session import AppState, ToolMode
from negpy.desktop.view.keyboard_shortcuts import SpacePanKeyFilter
from negpy.desktop.view.canvas.overlay import (
    _AUTO_PAN_MAX_SPEED_PX_S,
    CanvasOverlay,
)
from negpy.desktop.view.canvas.widget import ImageCanvas

_SIZE = (200, 160)
_INITIAL_RECT = (0.2, 0.2, 0.8, 0.8)


def _mouse_event(kind: QEvent.Type, pos: QPointF, buttons=Qt.MouseButton.LeftButton) -> QMouseEvent:
    return QMouseEvent(kind, pos, Qt.MouseButton.LeftButton, buttons, Qt.KeyboardModifier.NoModifier)


class _Canvas(QWidget):
    def __init__(self):
        super().__init__()
        self.setFixedSize(*_SIZE)
        self._is_panning = False
        self._space_pan_held = False
        self.pan_offset = QPointF()
        self.zoom_level = 1.0
        self.pan_deltas = []
        self.overlay = None

    def pan_by_viewport_delta(self, dx: float, dy: float) -> None:
        self.pan_deltas.append((dx, dy))
        self.pan_offset += QPointF(dx / self.width(), dy / self.height())
        self._sync_transform()

    def _sync_transform(self) -> None:
        if self.overlay is not None:
            self.overlay.set_transform(self.zoom_level, self.pan_offset.x(), self.pan_offset.y())

    def reset_tool_cursor(self) -> None:
        pass


def _overlay(tool=ToolMode.CROP_MANUAL, crop_rect=_INITIAL_RECT) -> CanvasOverlay:
    parent = _Canvas()
    overlay = CanvasOverlay(AppState(), parent)
    parent.overlay = overlay
    overlay.pan_requested.connect(parent.pan_by_viewport_delta)
    overlay.setFixedSize(*_SIZE)
    overlay._current_size = _SIZE
    overlay._recalc_view_rect()
    overlay.set_tool_mode(tool)
    overlay._crop_rect_norm = crop_rect
    overlay._test_parent = parent
    return overlay


def _clipped_image(overlay: CanvasOverlay) -> CanvasOverlay:
    overlay.zoom_level = 2.0
    overlay.parent().zoom_level = 2.0
    overlay._recalc_view_rect()
    return overlay


@pytest.mark.parametrize(
    ("pos", "expected_axis", "sign"),
    [
        (QPointF(199, 80), "x", -1),
        (QPointF(1, 80), "x", 1),
        (QPointF(100, 159), "y", -1),
        (QPointF(100, 1), "y", 1),
    ],
)
def test_auto_pan_starts_inside_the_clipped_image_edge_zone(pos, expected_axis, sign):
    overlay = _clipped_image(_overlay())
    overlay._begin_auto_pan(pos)
    velocity = overlay._auto_pan_velocity()

    if expected_axis == "x":
        assert velocity.x() * sign > 0
        assert velocity.y() == 0
    else:
        assert velocity.y() * sign > 0
        assert velocity.x() == 0
    overlay._stop_auto_pan()


def test_pointer_inside_buffer_does_not_pan():
    overlay = _clipped_image(_overlay())
    overlay._begin_auto_pan(QPointF(4, 80))

    assert overlay._auto_pan_velocity().isNull()
    assert not overlay._auto_pan_timer.isActive()
    overlay._track_auto_pan_pointer(QPointF(3, 80))
    assert overlay._auto_pan_timer.isActive()
    assert overlay._auto_pan_velocity().x() > 0
    overlay._track_auto_pan_pointer(QPointF(4, 80))
    assert overlay._auto_pan_velocity().isNull()
    assert not overlay._auto_pan_timer.isActive()
    overlay._track_auto_pan_pointer(QPointF(3, 80))
    assert overlay._auto_pan_timer.isActive()
    overlay._stop_auto_pan()


def test_fully_visible_image_does_not_pan_into_letterbox():
    overlay = _overlay()
    overlay._view_rect = QRectF(40, 20, 120, 120)
    overlay._begin_auto_pan(QPointF(1, 80))

    assert overlay._auto_pan_velocity().isNull()
    overlay._stop_auto_pan()


def test_speed_is_linear_then_capped():
    overlay = _clipped_image(_overlay())
    overlay._begin_auto_pan(QPointF(3, 80))
    slow = abs(overlay._auto_pan_velocity().x())
    overlay._track_auto_pan_pointer(QPointF(1, 80))
    fast = abs(overlay._auto_pan_velocity().x())
    overlay._track_auto_pan_pointer(QPointF(-100, 80))
    capped = abs(overlay._auto_pan_velocity().x())

    assert slow == pytest.approx(_AUTO_PAN_MAX_SPEED_PX_S * 0.005 * 4.0)
    assert fast == pytest.approx(_AUTO_PAN_MAX_SPEED_PX_S * 0.015 * 4.0)
    assert fast > slow
    assert capped == _AUTO_PAN_MAX_SPEED_PX_S
    overlay._stop_auto_pan()


def test_corner_overflow_selects_axis_with_greater_normalized_excess():
    overlay = _clipped_image(_overlay())
    overlay._begin_auto_pan(QPointF(1, 0))

    velocity = overlay._auto_pan_velocity()

    assert velocity.x() == 0
    assert velocity.y() > 0
    overlay._stop_auto_pan()


def test_timer_tick_pans_and_updates_stationary_straighten_pointer():
    overlay = _overlay(ToolMode.STRAIGHTEN, crop_rect=None)
    _clipped_image(overlay)
    overlay._straighten_p1 = QPointF(50, 50)
    overlay._straighten_p2 = QPointF(1, 50)
    overlay._begin_auto_pan(QPointF(1, 50))
    overlay._auto_pan_last_tick = 1.0

    overlay._auto_pan_tick(now=1.02)

    parent = overlay.parent()
    assert parent.pan_deltas[0][0] > 0
    assert overlay._straighten_p1.x() > 50
    assert overlay._straighten_p2 == QPointF(1, 50)
    assert overlay._auto_pan_timer.isActive()
    overlay._stop_auto_pan()


def test_delayed_timer_tick_clamps_elapsed_time():
    overlay = _overlay(ToolMode.STRAIGHTEN, crop_rect=None)
    _clipped_image(overlay)
    overlay._straighten_p1 = QPointF(50, 50)
    overlay._begin_auto_pan(QPointF(400, 50))
    overlay._auto_pan_last_tick = 1.0

    overlay._auto_pan_tick(now=2.0)

    dx, dy = overlay.parent().pan_deltas[0]
    assert dx == pytest.approx(-_AUTO_PAN_MAX_SPEED_PX_S * 0.05)
    assert dy == 0
    overlay._stop_auto_pan()


def test_crop_corner_anchor_tracks_pan_and_crop_move_uses_press_anchor():
    overlay = _overlay()
    overlay.mousePressEvent(_mouse_event(QEvent.Type.MouseButtonPress, QPointF(40, 32)))
    assert overlay._crop_drag_mode == "corner"
    old_anchor = QPointF(overlay._crop_anchor_screen)
    overlay.parent().pan_by_viewport_delta(-20, 0)
    assert overlay._crop_anchor_screen.x() == pytest.approx(old_anchor.x() - 20)
    overlay._end_crop_drag()

    overlay = _overlay()
    overlay.mousePressEvent(_mouse_event(QEvent.Type.MouseButtonPress, QPointF(100, 80)))
    assert overlay._crop_drag_mode == "move"
    overlay.parent().pan_by_viewport_delta(-20, 0)
    overlay._update_tracked_gesture(QPointF(100, 80), Qt.KeyboardModifier.NoModifier)
    first = overlay._crop_rect_norm
    overlay.parent().pan_by_viewport_delta(-20, 0)
    overlay._update_tracked_gesture(QPointF(100, 80), Qt.KeyboardModifier.NoModifier)

    assert first[0] == pytest.approx(0.3)
    assert overlay._crop_rect_norm[0] == pytest.approx(0.4)
    overlay._end_crop_drag()


def test_crop_draw_anchor_tracks_pan():
    overlay = _overlay(crop_rect=None)
    overlay.mousePressEvent(_mouse_event(QEvent.Type.MouseButtonPress, QPointF(50, 50)))
    assert overlay._crop_drag_mode == "draw"
    overlay.parent().pan_by_viewport_delta(-20, 0)
    assert overlay._crop_draw_p1.x() == pytest.approx(30)
    overlay._update_tracked_gesture(QPointF(100, 100), Qt.KeyboardModifier.NoModifier)
    assert overlay._crop_draw_p2 == QPointF(100, 100)
    overlay._end_crop_drag()


def test_auto_pan_stops_when_crop_drag_ends_or_tool_changes():
    overlay = _clipped_image(_overlay())
    overlay._begin_auto_pan(QPointF(1, 80))
    assert overlay._auto_pan_timer.isActive()
    overlay._end_crop_drag()
    assert not overlay._auto_pan_timer.isActive()

    overlay._begin_auto_pan(QPointF(1, 80))
    overlay.set_tool_mode(ToolMode.NONE)
    assert not overlay._auto_pan_timer.isActive()


def test_crop_release_commits_and_stops_auto_pan():
    overlay = _overlay()
    _clipped_image(overlay)
    emitted = []
    overlay.crop_rect_changed.connect(lambda *args: emitted.append(args))
    overlay.mousePressEvent(_mouse_event(QEvent.Type.MouseButtonPress, QPointF(40, 80)))
    overlay.mouseMoveEvent(_mouse_event(QEvent.Type.MouseMove, QPointF(220, 80)))
    assert overlay._auto_pan_timer.isActive()

    release = QMouseEvent(
        QEvent.Type.MouseButtonRelease,
        QPointF(220, 80),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
    )
    overlay.mouseReleaseEvent(release)

    assert emitted[-1][-1] is True
    assert not overlay._auto_pan_timer.isActive()


def test_auto_pan_stops_on_straighten_release_and_cancel():
    overlay = _overlay(ToolMode.STRAIGHTEN, crop_rect=None)
    _clipped_image(overlay)
    overlay._straighten_p1 = QPointF(40, 50)
    overlay._straighten_p2 = QPointF(100, 50)
    overlay._begin_auto_pan(QPointF(1, 50))
    assert overlay._auto_pan_timer.isActive()
    completed = []
    overlay.straighten_completed.connect(completed.append)

    release = QMouseEvent(
        QEvent.Type.MouseButtonRelease,
        QPointF(1, 50),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
    )
    overlay.mouseReleaseEvent(release)

    assert completed
    assert not overlay._auto_pan_timer.isActive()

    overlay._straighten_p1 = QPointF(40, 50)
    overlay._straighten_p2 = QPointF(100, 50)
    overlay._begin_auto_pan(QPointF(1, 50))
    assert overlay._auto_pan_timer.isActive()
    assert overlay.cancel_in_progress()
    assert not overlay._auto_pan_timer.isActive()


def test_canvas_pan_method_scales_by_viewport_size_and_syncs():
    class _Harness:
        pan_by_viewport_delta = ImageCanvas.pan_by_viewport_delta

        def __init__(self):
            self.pan_offset = QPointF()
            self.sync_count = 0

        def width(self):
            return 200

        def height(self):
            return 160

        def _sync_transform(self):
            self.sync_count += 1

    canvas = _Harness()
    canvas.pan_by_viewport_delta(-20, 8)

    assert canvas.pan_offset == QPointF(-0.1, 0.05)
    assert canvas.sync_count == 1


def test_space_pan_enters_existing_canvas_pan_state():
    class _Harness:
        zoom_level = 2.0
        state = SimpleNamespace(active_tool=ToolMode.CROP_MANUAL)
        _space_pan_held = True
        _is_panning = False
        _last_mouse_pos = QPointF()

        def setCursor(self, _cursor):
            pass

    canvas = _Harness()
    event = _mouse_event(QEvent.Type.MouseButtonPress, QPointF(25, 30))
    ImageCanvas.mousePressEvent(canvas, event)

    assert canvas._is_panning
    assert canvas._last_mouse_pos == QPointF(25, 30)
    assert event.isAccepted()


def test_space_and_pointer_motion_without_left_press_does_not_pan():
    overlay = _clipped_image(_overlay())
    overlay.parent()._space_pan_held = True
    event = _mouse_event(QEvent.Type.MouseMove, QPointF(120, 90), buttons=Qt.MouseButton.NoButton)

    overlay.mouseMoveEvent(event)

    assert not overlay.parent()._is_panning
    assert overlay.parent().pan_offset == QPointF()


def test_crop_rotation_handle_does_not_start_auto_pan():
    overlay = _overlay()
    _clipped_image(overlay)
    handle = overlay._crop_rotation_handle_points()["top"]
    overlay.mousePressEvent(_mouse_event(QEvent.Type.MouseButtonPress, handle))

    assert overlay._crop_drag_mode == "rotate"
    assert not overlay._auto_pan_active
    assert not overlay._auto_pan_timer.isActive()
    overlay._end_crop_drag()


class _ActiveWindow(QWidget):
    def __init__(self, active_tool=ToolMode.CROP_MANUAL):
        super().__init__()
        self._active = True
        self.state = AppState(active_tool=active_tool)
        self.crop_button = QPushButton(self)
        self.controls_panel = SimpleNamespace(geometry_sidebar=SimpleNamespace(manual_crop_btn=self.crop_button))

    def isActiveWindow(self):
        return self._active


def _key_event(kind, *, autorepeat=False):
    return QKeyEvent(kind, Qt.Key.Key_Space, Qt.KeyboardModifier.NoModifier, " ", autorepeat, 2 if autorepeat else 1)


def test_space_filter_tracks_key_state_and_ignores_auto_repeat(qapp):
    window = _ActiveWindow()
    event_filter = SpacePanKeyFilter(window)
    states = []
    event_filter.space_held_changed.connect(states.append)
    target = QWidget(window)

    assert event_filter.eventFilter(target, _key_event(QEvent.Type.KeyPress)) is True
    assert event_filter.eventFilter(target, _key_event(QEvent.Type.KeyPress, autorepeat=True)) is False
    assert not event_filter._space_down
    event_filter._on_space_timeout()
    assert event_filter._space_down
    assert event_filter.eventFilter(target, _key_event(QEvent.Type.KeyRelease)) is True
    assert not event_filter._space_down
    assert states == [True, False]
    event_filter.uninstall()


def test_space_pan_cursor_updates_immediately_when_keyboard_hold_detected(qapp):
    canvas = ImageCanvas(AppState(active_tool=ToolMode.CROP_MANUAL))
    canvas.resize(200, 160)
    canvas.show()
    qapp.processEvents()
    canvas.underMouse = lambda: True

    canvas.set_space_pan_held(True)
    assert canvas.cursor().shape() == Qt.CursorShape.ClosedHandCursor

    canvas.set_space_pan_held(False)
    assert canvas.cursor().shape() == Qt.CursorShape.CrossCursor


def test_space_filter_preserves_text_input_and_replays_short_press(qapp):
    window = _ActiveWindow()
    event_filter = SpacePanKeyFilter(window)
    text_input = QLineEdit(window)
    button = window.crop_button
    button.setCheckable(True)
    button.setChecked(False)
    window.setFixedSize(160, 80)
    button.setGeometry(0, 0, 120, 30)
    text_input.setGeometry(0, 40, 120, 30)
    window.show()
    button.show()
    text_input.show()
    text_input.setFocus()
    qapp.processEvents()

    assert qapp.focusWidget() is text_input
    press_event = _key_event(QEvent.Type.KeyPress)
    release_event = _key_event(QEvent.Type.KeyRelease)
    assert event_filter.eventFilter(text_input, press_event) is True
    assert not event_filter._space_down
    qapp.sendEvent(text_input, press_event)
    qapp.sendEvent(text_input, release_event)
    assert text_input.text() == " "

    button.setFocus()
    qapp.processEvents()
    assert qapp.focusWidget() is button
    press_event = _key_event(QEvent.Type.KeyPress)
    assert event_filter.eventFilter(button, press_event) is True
    assert not event_filter._space_down
    assert event_filter.eventFilter(button, _key_event(QEvent.Type.KeyRelease)) is True
    assert not event_filter._space_down
    assert button.isChecked()
    event_filter.uninstall()


def test_space_filter_clears_on_window_exit_and_deactivation(qapp):
    window = _ActiveWindow()
    event_filter = SpacePanKeyFilter(window)
    inside = QWidget(window)
    outside = QWidget()
    states = []
    event_filter.space_held_changed.connect(states.append)

    assert not event_filter.eventFilter(outside, _key_event(QEvent.Type.KeyPress))
    assert not event_filter._space_down
    assert event_filter.eventFilter(inside, _key_event(QEvent.Type.KeyPress)) is True
    event_filter._on_focus_changed(inside, QWidget(window))
    assert not event_filter._space_down
    event_filter._on_focus_changed(inside, outside)
    assert not event_filter._space_down

    event_filter.eventFilter(inside, _key_event(QEvent.Type.KeyPress))
    window._active = False
    event_filter.eventFilter(window, QEvent(QEvent.Type.WindowDeactivate))
    assert not event_filter._space_down
    assert states == []
    event_filter.uninstall()


def test_space_filter_does_not_consume_zone_place_without_tool_button(qapp):
    window = _ActiveWindow(ToolMode.ZONE_PLACE)
    event_filter = SpacePanKeyFilter(window)
    zone_strip = QWidget(window)

    assert event_filter.eventFilter(zone_strip, _key_event(QEvent.Type.KeyPress)) is True
    event_filter._on_space_timeout()
    assert event_filter._space_down
    assert event_filter.eventFilter(zone_strip, _key_event(QEvent.Type.KeyRelease)) is True
    assert not event_filter._space_down
    event_filter.uninstall()


@pytest.mark.parametrize("tool", [tool for tool in ToolMode if tool != ToolMode.NONE])
def test_space_left_drag_pans_without_starting_active_tool(tool):
    overlay = _overlay(tool, crop_rect=_INITIAL_RECT)
    _clipped_image(overlay)
    overlay.parent()._space_pan_held = True
    before_crop = overlay._crop_rect_norm
    before_straighten = (overlay._straighten_p1, overlay._straighten_p2)
    before_keystone = (overlay._keystone_draw_p1, overlay._keystone_draw_p2)
    overlay.mousePressEvent(_mouse_event(QEvent.Type.MouseButtonPress, QPointF(100, 80)))

    assert overlay.parent()._is_panning
    assert overlay._crop_drag_mode is None
    assert overlay._straighten_p1 is None
    assert overlay._keystone_draw_p1 is None
    overlay.mouseMoveEvent(_mouse_event(QEvent.Type.MouseMove, QPointF(120, 90)))
    assert overlay.parent().pan_offset == QPointF(0.1, 0.0625)
    assert overlay._crop_rect_norm == before_crop
    assert (overlay._straighten_p1, overlay._straighten_p2) == before_straighten
    assert (overlay._keystone_draw_p1, overlay._keystone_draw_p2) == before_keystone

    release = QMouseEvent(
        QEvent.Type.MouseButtonRelease,
        QPointF(120, 90),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
    )
    overlay.mouseReleaseEvent(release)
    assert not overlay.parent()._is_panning
