"""Sizing the heal/scratch/exclusion brush from the canvas.

Alt is the brush modifier the keyboard already uses, so the plain wheel keeps zooming.
A pinch sizes the brush only while one is live; everywhere else it still zooms, so no
context is left without a way to zoom.
"""

from __future__ import annotations

from dataclasses import replace
from unittest.mock import MagicMock

from PyQt6.QtCore import QPoint, QPointF, Qt
from PyQt6.QtGui import QNativeGestureEvent, QPointingDevice, QWheelEvent

from negpy.desktop.session import AppState, ToolMode
from negpy.desktop.view.canvas.widget import ImageCanvas
from negpy.features.retouch.models import HEAL_SIZE_MAX, HEAL_SIZE_MIN


def _canvas(qapp, *, tool=ToolMode.NONE, dust_remove=False, right_click_excludes=False):
    state = AppState()
    state.active_tool = tool
    state.right_click_excludes = right_click_excludes
    state.config = replace(state.config, retouch=replace(state.config.retouch, dust_remove=dust_remove))
    canvas = ImageCanvas(state)
    canvas._controller = MagicMock()
    return state, canvas


def _wheel(angle_y: int, alt: bool) -> QWheelEvent:
    return QWheelEvent(
        QPointF(10, 10),
        QPointF(10, 10),
        QPoint(0, 0),
        QPoint(0, angle_y),
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.AltModifier if alt else Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.NoScrollPhase,
        False,
    )


def _pinch(canvas: ImageCanvas, value: float) -> None:
    canvas.event(
        QNativeGestureEvent(
            Qt.NativeGestureType.ZoomNativeGesture,
            QPointingDevice.primaryPointingDevice(),
            2,
            QPointF(10.0, 10.0),
            QPointF(10.0, 10.0),
            QPointF(10.0, 10.0),
            value,
            QPointF(0.0, 0.0),
            0,
        )
    )


def test_a_plain_wheel_still_zooms(qapp):
    _state, canvas = _canvas(qapp)
    before = canvas.zoom_level

    canvas.wheelEvent(_wheel(-120, alt=False))

    canvas._controller.adjust_brush_size.assert_not_called()
    assert canvas.zoom_level != before, "the wheel is still the zoom"


def test_alt_wheel_sizes_the_brush_instead_of_zooming(qapp):
    _state, canvas = _canvas(qapp)
    before = canvas.zoom_level

    canvas.wheelEvent(_wheel(-120, alt=True))

    canvas._controller.adjust_brush_size.assert_called_once()
    assert canvas.zoom_level == before, "Alt takes the wheel off the zoom"


def test_alt_wheel_sizes_both_ways(qapp):
    _state, canvas = _canvas(qapp)

    canvas.wheelEvent(_wheel(-120, alt=True))
    up = canvas._controller.adjust_brush_size.call_args[0][0]
    canvas.wheelEvent(_wheel(120, alt=True))
    down = canvas._controller.adjust_brush_size.call_args[0][0]

    assert up > 0 > down


def test_the_size_clamps_to_the_sliders_range(qapp):
    """The canvas and the slider read one range, so neither can leave the other's bounds."""
    from negpy.desktop.controller import AppController

    state = AppState()
    controller = MagicMock()
    controller.state = state
    controller.session = MagicMock()

    state.config = replace(state.config, retouch=replace(state.config.retouch, manual_dust_size=int(HEAL_SIZE_MAX)))
    AppController.adjust_brush_size(controller, 10.0)
    controller.session.update_config.assert_not_called()

    state.config = replace(state.config, retouch=replace(state.config.retouch, manual_dust_size=int(HEAL_SIZE_MIN)))
    AppController.adjust_brush_size(controller, -10.0)
    controller.session.update_config.assert_not_called()

    AppController.adjust_brush_size(controller, 3.0)
    applied = controller.session.update_config.call_args[0][0]
    assert applied.retouch.manual_dust_size == int(HEAL_SIZE_MIN) + 3


def test_reversed_scroll_reverses_the_brush_too(qapp):
    """One scroll direction means "more" for both, or the zoom and the brush disagree for
    anyone who flipped the preference."""
    state, canvas = _canvas(qapp)
    canvas.wheelEvent(_wheel(-120, alt=True))
    plain = canvas._controller.adjust_brush_size.call_args[0][0]

    state.invert_zoom_scroll = True
    canvas.wheelEvent(_wheel(-120, alt=True))
    inverted = canvas._controller.adjust_brush_size.call_args[0][0]

    assert plain == -inverted


def test_pinch_zooms_when_no_brush_is_live(qapp):
    _state, canvas = _canvas(qapp)

    assert canvas._pinch_sizes_brush() is False
    _pinch(canvas, 0.5)

    canvas._controller.adjust_brush_size.assert_not_called()


def test_pinch_sizes_the_brush_while_a_heal_tool_is_live(qapp):
    _state, canvas = _canvas(qapp, tool=ToolMode.DUST_PICK)

    assert canvas._pinch_sizes_brush() is True
    for _ in range(8):
        _pinch(canvas, 0.5)

    assert canvas._controller.adjust_brush_size.called, "the pinch reached the brush"


def test_pinch_sizes_the_brush_while_a_right_click_excludes(qapp):
    _state, canvas = _canvas(qapp, dust_remove=True, right_click_excludes=True)
    assert canvas._pinch_sizes_brush() is True


def test_an_armed_right_click_needs_optical_removal_on(qapp):
    _state, canvas = _canvas(qapp, dust_remove=False, right_click_excludes=True)
    assert canvas._pinch_sizes_brush() is False


def test_a_slow_pinch_is_held_back_until_it_is_worth_a_pixel(qapp):
    """A pinch reports a fraction per event; dropping them would leave the brush stuck."""
    _state, canvas = _canvas(qapp, tool=ToolMode.DUST_PICK)

    canvas._pinch_brush_step(1.001)
    canvas._controller.adjust_brush_size.assert_not_called()

    for _ in range(200):
        canvas._pinch_brush_step(1.001)
    assert canvas._controller.adjust_brush_size.called, "the fractions accumulated"


def test_a_trackpad_scroll_sizes_the_brush_too(qapp):
    """wheel_notch_delta reads pixelDelta when there is no angleDelta, so Alt plus a
    two-finger scroll works like Alt plus a wheel."""
    _state, canvas = _canvas(qapp)
    event = QWheelEvent(
        QPointF(10, 10),
        QPointF(10, 10),
        QPoint(0, -64),
        QPoint(0, 0),
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.AltModifier,
        Qt.ScrollPhase.NoScrollPhase,
        False,
    )

    canvas.wheelEvent(event)

    canvas._controller.adjust_brush_size.assert_called_once()


def test_the_brush_circle_shows_only_where_a_right_click_would_paint(qapp):
    """The pinch sizes a brush that nothing else draws: no tool is active. It must not
    follow the cursor through ordinary editing, so every other state stays clean."""
    _state, armed = _canvas(qapp, dust_remove=True, right_click_excludes=True)
    assert armed.overlay._draws_exclusion_brush() is True

    _state, unarmed = _canvas(qapp, dust_remove=True, right_click_excludes=False)
    assert unarmed.overlay._draws_exclusion_brush() is False

    _state, off = _canvas(qapp, dust_remove=False, right_click_excludes=True)
    assert off.overlay._draws_exclusion_brush() is False, "Optical Removal is off"

    _state, healing = _canvas(qapp, tool=ToolMode.DUST_PICK, dust_remove=True, right_click_excludes=True)
    healing.overlay.set_tool_mode(ToolMode.DUST_PICK)
    assert healing.overlay._draws_exclusion_brush() is False, "the heal tool draws its own"
