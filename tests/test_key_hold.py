"""Hold-to behavior: a tap of a held-aware key latches, a hold lasts only while the key is down."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
from PyQt6.QtCore import QEvent, Qt
from PyQt6.QtGui import QKeyEvent
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QWidget

from negpy.desktop.controller import AppController
from negpy.desktop.session import AppState, ToolMode
from negpy.desktop.view.keyboard_shortcuts import KeyHold, ShortcutManager

KEY = Qt.Key.Key_N


def _key_event(kind: QEvent.Type, auto_repeat: bool = False) -> QKeyEvent:
    return QKeyEvent(kind, KEY, Qt.KeyboardModifier.NoModifier, "n", auto_repeat, 1)


def _hold(qapp):
    window = QWidget()
    hold = KeyHold(window)
    hold._timer.setInterval(1)
    calls: list[str] = []
    callbacks = dict(on_tap=lambda: calls.append("tap"), on_hold=lambda: calls.append("hold"), on_end=lambda: calls.append("end"))
    return window, hold, calls, callbacks


def test_quick_release_is_a_tap(qapp):
    window, hold, calls, callbacks = _hold(qapp)
    hold._timer.setInterval(10_000)
    hold.press(KEY, **callbacks)
    hold.eventFilter(window, _key_event(QEvent.Type.KeyRelease))
    assert calls == ["tap"]


def test_long_press_holds_until_release(qapp):
    window, hold, calls, callbacks = _hold(qapp)
    hold.press(KEY, **callbacks)
    QTest.qWait(30)
    assert calls == ["hold"]
    hold.eventFilter(window, _key_event(QEvent.Type.KeyRelease))
    assert calls == ["hold", "end"]


def test_autorepeat_is_swallowed_and_does_not_release(qapp):
    window, hold, calls, callbacks = _hold(qapp)
    hold.press(KEY, **callbacks)
    QTest.qWait(30)
    assert hold.eventFilter(window, _key_event(QEvent.Type.KeyPress, auto_repeat=True)) is True
    assert hold.eventFilter(window, _key_event(QEvent.Type.KeyRelease, auto_repeat=True)) is True
    assert calls == ["hold"]


def test_leaving_the_app_ends_the_hold(qapp):
    window, hold, calls, callbacks = _hold(qapp)
    hold.press(KEY, **callbacks)
    QTest.qWait(30)
    hold.eventFilter(window, QEvent(QEvent.Type.ApplicationDeactivate))
    assert calls == ["hold", "end"]


def test_a_second_press_ends_the_first(qapp):
    _window, hold, calls, callbacks = _hold(qapp)
    hold.press(KEY, **callbacks)
    QTest.qWait(30)
    hold.press(Qt.Key.Key_M)
    assert calls == ["hold", "end"]


def _spring(flag_value: bool, refuse: bool = False):
    state = SimpleNamespace(negative_peek=flag_value)
    toggles: list = []

    def toggle(force=None):
        toggles.append(force)
        if not refuse:
            state.negative_peek = (not state.negative_peek) if force is None else force

    pressed: list = []
    manager = SimpleNamespace(
        window=SimpleNamespace(controller=SimpleNamespace(state=state)),
        key_hold=SimpleNamespace(press=lambda key, **kw: pressed.append(kw)),
    )
    return ShortcutManager._spring_loaded(manager, "negative_peek", toggle), state, toggles, pressed


def test_peek_key_flips_on_press_and_a_hold_flips_it_back():
    press, state, toggles, pressed = _spring(False)
    press(KEY)
    assert state.negative_peek is True
    pressed[0]["on_end"]()
    assert state.negative_peek is False
    assert toggles == [None, False]


def test_refused_peek_leaves_nothing_to_undo():
    press, state, _toggles, pressed = _spring(False, refuse=True)
    press(KEY)
    assert state.negative_peek is False
    assert pressed == []


def _controller(**state) -> MagicMock:
    ctrl = MagicMock()
    ctrl.state = AppState(preview_raw=np.zeros((2, 2, 3), np.float32), **state)
    ctrl._compare_before_key.return_value = "k"
    return ctrl


def test_hold_asks_for_a_baseline_when_none_is_kept():
    ctrl = _controller()
    AppController.begin_compare_hold(ctrl)
    assert ctrl.state.compare_hold is True
    ctrl._request_compare_baseline.assert_called_once()
    AppController.end_compare_hold(ctrl)
    assert ctrl.state.compare_hold is False


def test_hold_reuses_a_kept_baseline():
    ctrl = _controller(compare_before=np.zeros((2, 2, 3), np.float32), compare_before_key="k")
    AppController.begin_compare_hold(ctrl)
    assert ctrl.state.compare_hold is True
    ctrl._request_compare_baseline.assert_not_called()


def test_no_hold_over_a_peek_or_an_uncropped_tool():
    for state in ({"negative_peek": True}, {"active_tool": ToolMode.CROP_MANUAL}):
        ctrl = _controller(**state)
        AppController.begin_compare_hold(ctrl)
        assert ctrl.state.compare_hold is False
        ctrl._request_compare_baseline.assert_not_called()
