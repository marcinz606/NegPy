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


def test_a_touch_pinch_anchors_on_its_screen_centre_mapped_into_the_canvas(qapp, monkeypatch):
    from unittest.mock import patch

    from PyQt6.QtCore import QPoint, QPointF
    from PyQt6.QtWidgets import QGestureEvent, QPinchGesture

    monkeypatch.setattr("sys.platform", "linux")
    _state, canvas = _canvas(qapp)
    canvas.move(300, 200)
    canvas.show()
    gesture = QPinchGesture()
    centre = canvas.mapToGlobal(QPoint(40, 30))
    gesture.setCenterPoint(QPointF(centre))
    gesture.setLastScaleFactor(1.5)
    event = QGestureEvent([gesture])
    with (
        patch.object(QPinchGesture, "state", return_value=Qt.GestureState.GestureUpdated),
        patch.object(canvas, "_apply_scale_at", return_value=True) as apply,
    ):
        canvas._try_pinch_gesture(event)
    anchor = apply.call_args.args[1]
    assert (round(anchor.x()), round(anchor.y())) == (40, 30)


def test_a_trackpad_pinch_on_macos_anchors_on_the_pointer(qapp, monkeypatch):
    from unittest.mock import patch

    from PyQt6.QtCore import QPoint, QPointF
    from PyQt6.QtWidgets import QGestureEvent, QPinchGesture

    monkeypatch.setattr("sys.platform", "darwin")
    _state, canvas = _canvas(qapp)
    canvas.move(300, 200)
    canvas.show()
    gesture = QPinchGesture()
    # The Cocoa recognizer fills the center with widget coordinates.
    gesture.setCenterPoint(QPointF(100, 50))
    gesture.setLastScaleFactor(1.5)
    event = QGestureEvent([gesture])
    with (
        patch.object(QPinchGesture, "state", return_value=Qt.GestureState.GestureUpdated),
        patch("negpy.desktop.view.canvas.widget.QCursor.pos", return_value=canvas.mapToGlobal(QPoint(40, 30))),
        patch.object(canvas, "_apply_scale_at", return_value=True) as apply,
    ):
        canvas._try_pinch_gesture(event)
    anchor = apply.call_args.args[1]
    assert (round(anchor.x()), round(anchor.y())) == (40, 30)


def test_a_trackpad_pinch_on_macos_with_the_pointer_off_the_canvas_anchors_on_its_centre(qapp, monkeypatch):
    from unittest.mock import patch

    from PyQt6.QtCore import QPoint, QPointF
    from PyQt6.QtWidgets import QGestureEvent, QPinchGesture

    monkeypatch.setattr("sys.platform", "darwin")
    _state, canvas = _canvas(qapp)
    canvas.move(300, 200)
    canvas.show()
    gesture = QPinchGesture()
    gesture.setCenterPoint(QPointF(100, 50))
    gesture.setLastScaleFactor(1.5)
    event = QGestureEvent([gesture])
    with (
        patch.object(QPinchGesture, "state", return_value=Qt.GestureState.GestureUpdated),
        patch("negpy.desktop.view.canvas.widget.QCursor.pos", return_value=canvas.mapToGlobal(QPoint(-200, 30))),
        patch.object(canvas, "_apply_scale_at", return_value=True) as apply,
    ):
        canvas._try_pinch_gesture(event)
    anchor = apply.call_args.args[1]
    assert (round(anchor.x()), round(anchor.y())) == (100, 50)


def test_fractional_wheel_deltas_accumulate_and_reverse_cleanly(qapp, tmp_path):
    """Smooth wheels and trackpads emit <0.5 notch steps per event; accumulate them instead of
    rounding each event back to zero, persist once when the single-shot timer expires, and
    isolate interleaved strokes or slider drags."""
    from types import SimpleNamespace

    from PyQt6.QtCore import QTimer
    from PyQt6.QtTest import QTest

    from negpy.desktop.controller import AppController
    from negpy.desktop.session import BRUSH_SIZE_KEY, DesktopSessionManager
    from negpy.infrastructure.storage.repository import StorageRepository

    repo = StorageRepository(str(tmp_path / "edits.db"), str(tmp_path / "settings.db"))
    repo.initialize()
    session = DesktopSessionManager(repo)
    state = session.state
    state.uploaded_files = [{"name": "f0.arw", "path": "/tmp/f0.arw", "hash": "h0"}]
    state.selected_file_idx = 0
    state.current_file_hash = "h0"
    state.current_file_path = "/tmp/f0.arw"
    state.config = replace(state.config, retouch=replace(state.config.retouch, manual_dust_size=10))

    timer = QTimer()
    timer.setSingleShot(True)
    timer.setInterval(30)
    overlay = MagicMock()
    stub = SimpleNamespace(
        state=state,
        session=session,
        _brush_size_accum=0.0,
        _brush_size_dir=0,
        _brush_persist_timer=timer,
        canvas=SimpleNamespace(overlay=overlay),
        brush_size_changed=MagicMock(),
        set_status=MagicMock(),
    )
    timer.timeout.connect(lambda: AppController._flush_brush_size(stub))

    for _ in range(5):
        AppController.adjust_brush_size(stub, 0.2)
    assert state.config.retouch.manual_dust_size == 11
    # In-memory only during scroll: DB has no saved row yet.
    assert repo.load_file_settings("h0") is None
    overlay.refresh_brush_cursor.assert_called_once_with(10)
    stub.brush_size_changed.emit.assert_called_once_with(11)
    assert timer.isActive()

    # Reversing scroll direction drops any opposite-sign remainder immediately.
    AppController.adjust_brush_size(stub, 0.4)
    for _ in range(3):
        AppController.adjust_brush_size(stub, -0.2)
    assert state.config.retouch.manual_dust_size == 10

    # Scroll up to 14 and let the real Qt single-shot timer expire (isActive() == False inside slot).
    AppController.adjust_brush_size(stub, 4.0)
    assert state.config.retouch.manual_dust_size == 14
    QTest.qWait(80)
    assert not timer.isActive()
    assert repo.load_file_settings("h0").retouch.manual_dust_size == 14
    assert repo.get_global_setting(BRUSH_SIZE_KEY) == 14
    assert state.undo_index == 1

    # Painting a stroke within the debounce window records two separate undo steps (brush then stroke).
    AppController.adjust_brush_size(stub, 6.0)
    assert state.config.retouch.manual_dust_size == 20
    assert timer.isActive()
    stroke_cfg = replace(state.config, retouch=replace(state.config.retouch, manual_dust_spots=[(0.5, 0.5, 20.0)]))
    session.update_config(stroke_cfg, persist=True, render=False)
    assert state.undo_index == 3
    session.undo()
    assert state.config.retouch.manual_dust_spots == []
    assert state.config.retouch.manual_dust_size == 20, "undoing the stroke keeps the brush size"
    # Timer firing later does not clobber anything.
    QTest.qWait(80)
    assert state.config.retouch.manual_dust_size == 20

    # Background auto-measured writes (record_history=False) during a wheel scroll rebase the
    # pending baseline instead of splitting the scroll into two undo steps or persisting mid-scroll.
    AppController.adjust_brush_size(stub, 3.0)
    assert state.config.retouch.manual_dust_size == 23
    measured_cfg = replace(
        state.config,
        geometry=replace(state.config.geometry, crop_detect_key="auto-k1"),
    )
    session.update_config(measured_cfg, persist=True, render=False, record_history=False)
    assert state.undo_index == 2, "record_history=False must not split an active wheel gesture"
    assert repo.get_global_setting(BRUSH_SIZE_KEY) == 20, "uncommitted preview size is not written mid-scroll"
    AppController.adjust_brush_size(stub, 2.0)
    assert state.config.retouch.manual_dust_size == 25
    QTest.qWait(80)
    assert state.undo_index == 3
    assert repo.get_global_setting(BRUSH_SIZE_KEY) == 25
    assert repo.load_file_settings("h0").geometry.crop_detect_key == "auto-k1"
    session.undo()
    assert state.config.retouch.manual_dust_size == 20
    assert state.config.geometry.crop_detect_key == "auto-k1"

    # Unloading the sole open frame within the debounce window flushes the pending brush size first.
    AppController.adjust_brush_size(stub, 8.0)
    assert state.config.retouch.manual_dust_size == 28
    session.remove_current_file()
    assert repo.get_global_setting(BRUSH_SIZE_KEY) == 28
    assert repo.load_file_settings("h0").retouch.manual_dust_size == 28
    assert state.config.retouch.manual_dust_size == 28


def test_toolbar_refreshes_undo_on_history_changed_and_syncs_tool_slider_on_undo(qapp, tmp_path):
    from negpy.desktop.session import DesktopSessionManager
    from negpy.desktop.view.canvas.toolbar import ActionToolbar
    from negpy.infrastructure.storage.repository import StorageRepository

    repo = StorageRepository(str(tmp_path / "edits.db"), str(tmp_path / "settings.db"))
    repo.initialize()
    session = DesktopSessionManager(repo)
    session.state.uploaded_files = [{"name": "f0.arw", "path": "/tmp/f0.arw", "hash": "h0"}]
    session.state.selected_file_idx = 0
    session.state.current_file_hash = "h0"
    session.state.current_file_path = "/tmp/f0.arw"
    session.state.config = replace(session.state.config, retouch=replace(session.state.config.retouch, manual_dust_size=10))

    ctrl = MagicMock()
    ctrl.session = session
    ctrl.state = session.state
    ctrl.measuring_baseline.return_value = False
    ctrl.average_scopes.return_value = {}
    tb = ActionToolbar(ctrl)
    assert not tb.btn_undo.isEnabled()

    # Debounced wheel commit emits history_changed (render=False -> no state_changed): btn_undo must enable.
    session.preview_brush_size(22)
    session.commit_brush_size()
    assert tb.btn_undo.isEnabled()

    # Undoing a brush-size step updates the mirrored tool slider via _sync_tool_row.
    fake_slider = MagicMock()
    fake_source = MagicMock()
    fake_source.isEnabled.return_value = True
    tb._tool_slider = fake_slider
    tb._tool_slider_source = fake_source
    tb._shown_tool = ToolMode.DUST_PICK
    session.state.active_tool = ToolMode.DUST_PICK
    session.undo()
    assert session.state.config.retouch.manual_dust_size == 10
    fake_slider.setValue.assert_called_with(10.0)
