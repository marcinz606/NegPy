"""Offline tests for the map widget: tiles are stubbed, nothing reaches the network."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import sys

import pytest
from PyQt6.QtCore import QPoint, QPointF, Qt
from PyQt6.QtGui import QMouseEvent, QWheelEvent
from PyQt6.QtWidgets import QApplication

from negpy.desktop.view.widgets import slippy_map
from negpy.desktop.view.widgets.slippy_map import SlippyMapWidget

if not QApplication.instance():
    _app = QApplication(sys.argv)


@pytest.fixture
def widget(monkeypatch) -> SlippyMapWidget:
    monkeypatch.setattr(slippy_map, "fetch_tile", lambda *a, **k: None)
    map_widget = SlippyMapWidget()
    map_widget.resize(512, 384)
    monkeypatch.setattr(map_widget._pool, "start", lambda job, *args: job.run())
    return map_widget


def _click(widget: SlippyMapWidget, x: int, y: int) -> None:
    for event_type in (QMouseEvent.Type.MouseButtonPress, QMouseEvent.Type.MouseButtonRelease):
        widget.event(
            QMouseEvent(
                event_type,
                QPointF(x, y),
                Qt.MouseButton.LeftButton,
                Qt.MouseButton.LeftButton,
                Qt.KeyboardModifier.NoModifier,
            )
        )


def test_paints_with_no_tiles_available(widget: SlippyMapWidget) -> None:
    """A missing tile must paint a placeholder, not raise."""
    widget.set_pin(50.0614, 19.9366)
    assert not widget.grab().isNull()


def test_click_sets_the_pin_at_the_cursor(widget: SlippyMapWidget) -> None:
    expected = widget.latlon_at(100, 80)
    _click(widget, 100, 80)
    assert widget.pin() == pytest.approx(expected)


def test_centre_of_the_widget_is_the_centre_of_the_view(widget: SlippyMapWidget) -> None:
    widget.set_pin(50.0614, 19.9366)
    x, y = widget._pixel_at(50.0614, 19.9366)
    assert (x, y) == pytest.approx((widget.width() / 2.0, widget.height() / 2.0))


def test_drag_pans_without_moving_the_pin(widget: SlippyMapWidget) -> None:
    widget.set_pin(50.0, 19.0)
    press = QMouseEvent(
        QMouseEvent.Type.MouseButtonPress,
        QPointF(200, 200),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    move = QMouseEvent(
        QMouseEvent.Type.MouseMove,
        QPointF(260, 200),
        Qt.MouseButton.NoButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    release = QMouseEvent(
        QMouseEvent.Type.MouseButtonRelease,
        QPointF(260, 200),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    for event in (press, move, release):
        widget.event(event)

    assert widget.pin() == (50.0, 19.0)
    # Dragging right shows what is west of the old centre.
    assert widget._pixel_at(50.0, 19.0)[0] > widget.width() / 2.0


def test_wheel_zoom_keeps_the_position_under_the_cursor(widget: SlippyMapWidget) -> None:
    anchor = widget.latlon_at(120.0, 90.0)
    widget.event(
        QWheelEvent(
            QPointF(120.0, 90.0),
            QPointF(120.0, 90.0),
            QPoint(0, 0),
            QPoint(0, 120),
            Qt.MouseButton.NoButton,
            Qt.KeyboardModifier.NoModifier,
            Qt.ScrollPhase.NoScrollPhase,
            False,
        )
    )
    assert widget._zoom == 5
    assert widget.latlon_at(120.0, 90.0) == pytest.approx(anchor, abs=1e-6)


def test_zoom_clamps_to_the_supported_range(widget: SlippyMapWidget) -> None:
    widget.set_zoom(99)
    assert widget._zoom == slippy_map.MAX_ZOOM
    widget.set_zoom(0)
    assert widget._zoom == slippy_map.MIN_ZOOM


def test_pending_tiles_are_capped(monkeypatch) -> None:
    """The pool joins its queue on close, so the queue must stay small."""
    monkeypatch.setattr(slippy_map, "fetch_tile", lambda *a, **k: None)
    map_widget = SlippyMapWidget()
    map_widget.resize(512, 384)
    started: list[tuple] = []
    monkeypatch.setattr(map_widget._pool, "start", lambda job, *args: started.append(job))

    for x in range(200):
        map_widget._request((4, x, 4))

    assert len(started) == slippy_map._MAX_PENDING_TILES


def test_shutdown_stops_queued_fetches(monkeypatch) -> None:
    fetched: list[tuple] = []
    monkeypatch.setattr(slippy_map, "fetch_tile", lambda *a, **k: fetched.append(a))
    map_widget = SlippyMapWidget()
    queued: list = []
    monkeypatch.setattr(map_widget._pool, "start", lambda job, *args: queued.append(job))
    map_widget._request((4, 1, 1))

    map_widget.shutdown()
    for job in queued:
        job.run()

    assert fetched == []


def test_shutdown_joins_running_fetches_instead_of_the_destructor(monkeypatch) -> None:
    """The pool's destructor waits with the GIL held, which would hang the GUI for good."""
    waited: list[int] = []
    monkeypatch.setattr(slippy_map, "fetch_tile", lambda *a, **k: None)
    map_widget = SlippyMapWidget()
    monkeypatch.setattr(map_widget._pool, "waitForDone", lambda ms: waited.append(ms) or True)

    map_widget.shutdown()

    assert waited == [slippy_map._SHUTDOWN_WAIT_MS]
