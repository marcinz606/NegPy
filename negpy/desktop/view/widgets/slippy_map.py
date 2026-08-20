"""A pan/zoom OpenStreetMap tile view with one draggable pin."""

from __future__ import annotations

import math
from typing import Optional

import qtawesome as qta
from PyQt6.QtCore import QObject, QPoint, QPointF, QRunnable, QSize, Qt, QThreadPool, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QMouseEvent, QNativeGestureEvent, QPainter, QPixmap, QWheelEvent
from PyQt6.QtWidgets import QToolButton, QWidget

from negpy.desktop.view.styles.theme import THEME
from negpy.features.metadata.capture import deg2tile, tile2deg
from negpy.services.maps import MAX_ZOOM, MIN_ZOOM, TILE_SIZE, fetch_tile

_ATTRIBUTION = "© OpenStreetMap contributors"
_DRAG_SLOP_PX = 4
_MAX_CONCURRENT_TILES = 4
# Panning enqueues tiles faster than they arrive, and the pool joins its queue when the view
# closes. Cap the queue so that join is short, and so stale requests cannot pile up.
_MAX_PENDING_TILES = 24
_SHUTDOWN_WAIT_MS = 6000
# One mouse notch is 120 units. A trackpad sends many smaller deltas, so the wheel must sum
# to a notch instead of taking a zoom level per event.
_WHEEL_NOTCH = 120.0
# A trackpad pinch reports a scale delta per event; this much of one is a zoom level.
_PINCH_STEP = 0.35
_ZOOM_BUTTON_PX = 26


class _TileSignals(QObject):
    ready = pyqtSignal(int, int, int, object)

    def __init__(self):
        super().__init__()
        self.stopped = False


class _TileJob(QRunnable):
    def __init__(self, signals: _TileSignals, z: int, x: int, y: int):
        super().__init__()
        self._signals = signals
        self._key = (z, x, y)

    def run(self) -> None:
        if self._signals.stopped:
            return
        data = fetch_tile(*self._key)
        if self._signals.stopped:
            return
        self._signals.ready.emit(*self._key, data)


class SlippyMapWidget(QWidget):
    """Tiles are fetched off the GUI thread; a missing tile paints flat and nothing blocks."""

    pin_moved = pyqtSignal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(420, 300)
        self.setCursor(Qt.CursorShape.CrossCursor)

        self._zoom = 4
        self._center = (50.0, 15.0)
        self._pin: Optional[tuple[float, float]] = None

        self._tiles: dict[tuple[int, int, int], QPixmap] = {}
        self._requested: set[tuple[int, int, int]] = set()
        self._missing: set[tuple[int, int, int]] = set()

        self._pool = QThreadPool(self)
        self._pool.setMaxThreadCount(_MAX_CONCURRENT_TILES)
        self._signals = _TileSignals()
        self._signals.ready.connect(self._on_tile_ready)

        self._drag_origin: Optional[QPoint] = None
        self._dragged = False
        self._wheel_accum = 0.0
        self._pinch_accum = 0.0

        self._zoom_in_btn = self._zoom_button("fa5s.plus", "Zoom in", 1)
        self._zoom_out_btn = self._zoom_button("fa5s.minus", "Zoom out", -1)

    def _zoom_button(self, icon: str, tip: str, step: int) -> QToolButton:
        button = QToolButton(self)
        button.setIcon(qta.icon(icon, color=THEME.text_primary))
        button.setIconSize(QSize(10, 10))
        button.setFixedSize(_ZOOM_BUTTON_PX, _ZOOM_BUTTON_PX)
        button.setToolTip(tip)
        button.setCursor(Qt.CursorShape.ArrowCursor)
        button.setAutoRepeat(True)
        button.setStyleSheet(
            f"QToolButton {{ background: {THEME.surface_overlay}; border: 1px solid {THEME.border_color};"
            f" border-radius: {THEME.radius_sm}px; }}"
            f"QToolButton:hover {{ background: {THEME.surface_overlay_hover}; }}"
        )
        button.clicked.connect(lambda: self.zoom_by(step))
        return button

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt override
        self._zoom_in_btn.move(THEME.space_lg, THEME.space_lg)
        self._zoom_out_btn.move(THEME.space_lg, THEME.space_lg + _ZOOM_BUTTON_PX + THEME.space_xs)
        super().resizeEvent(event)

    # ── state ────────────────────────────────────────────────────────────

    def pin(self) -> Optional[tuple[float, float]]:
        return self._pin

    def set_pin(self, lat: float, lon: float, *, recenter: bool = True) -> None:
        self._pin = (lat, lon)
        if recenter:
            self._center = (lat, lon)
        self.update()

    def set_center(self, lat: float, lon: float) -> None:
        self._center = (lat, lon)
        self.update()

    def clear_pin(self) -> None:
        self._pin = None
        self.update()

    def set_zoom(self, zoom: int) -> None:
        self._zoom = max(MIN_ZOOM, min(MAX_ZOOM, zoom))
        self.update()

    # ── coordinate helpers ───────────────────────────────────────────────

    def _tile_at(self, x_px: float, y_px: float) -> tuple[float, float]:
        cx, cy = deg2tile(*self._center, self._zoom)
        return (
            cx + (x_px - self.width() / 2.0) / TILE_SIZE,
            cy + (y_px - self.height() / 2.0) / TILE_SIZE,
        )

    def _pixel_at(self, lat: float, lon: float) -> tuple[float, float]:
        cx, cy = deg2tile(*self._center, self._zoom)
        tx, ty = deg2tile(lat, lon, self._zoom)
        return (
            self.width() / 2.0 + (tx - cx) * TILE_SIZE,
            self.height() / 2.0 + (ty - cy) * TILE_SIZE,
        )

    def latlon_at(self, x_px: float, y_px: float) -> tuple[float, float]:
        return tile2deg(*self._tile_at(x_px, y_px), self._zoom)

    # ── tiles ────────────────────────────────────────────────────────────

    def _request(self, key: tuple[int, int, int]) -> None:
        if self._signals.stopped or len(self._requested) >= _MAX_PENDING_TILES:
            return
        if key in self._tiles or key in self._requested or key in self._missing:
            return
        self._requested.add(key)
        self._pool.start(_TileJob(self._signals, *key))

    def shutdown(self) -> None:
        """
        Drop pending tiles and join the running ones here, not in the pool's destructor: that
        destructor waits while holding the GIL, so a fetch thread could never finish and the
        GUI would hang for good. waitForDone releases the GIL, so the wait is one fetch long.
        """
        self._signals.stopped = True
        self._pool.clear()
        self._pool.waitForDone(_SHUTDOWN_WAIT_MS)

    def hideEvent(self, event) -> None:  # noqa: N802 - Qt override
        self.shutdown()
        super().hideEvent(event)

    def showEvent(self, event) -> None:  # noqa: N802 - Qt override
        self._signals.stopped = False
        super().showEvent(event)

    def _on_tile_ready(self, z: int, x: int, y: int, data: object) -> None:
        if self._signals.stopped:
            return
        key = (z, x, y)
        self._requested.discard(key)
        pixmap = QPixmap()
        if isinstance(data, (bytes, bytearray)) and pixmap.loadFromData(bytes(data)):
            self._tiles[key] = pixmap
        else:
            self._missing.add(key)
        self.update()

    # ── painting ─────────────────────────────────────────────────────────

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt override
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(THEME.canvas_bg_dark_grey))

        span = 2**self._zoom
        left, top = self._tile_at(0.0, 0.0)
        first_x, first_y = math.floor(left), math.floor(top)
        offset_x = (first_x - left) * TILE_SIZE
        offset_y = (first_y - top) * TILE_SIZE

        columns = int(self.width() / TILE_SIZE) + 2
        rows = int(self.height() / TILE_SIZE) + 2

        for col in range(columns):
            for row in range(rows):
                tile_x, tile_y = first_x + col, first_y + row
                if not 0 <= tile_y < span:
                    continue
                key = (self._zoom, tile_x % span, tile_y)
                px = int(offset_x + col * TILE_SIZE)
                py = int(offset_y + row * TILE_SIZE)
                pixmap = self._tiles.get(key)
                if pixmap is None:
                    self._request(key)
                    painter.fillRect(px, py, TILE_SIZE, TILE_SIZE, QColor(THEME.canvas_bg_mid_grey))
                    continue
                painter.drawPixmap(px, py, pixmap)

        self._paint_pin(painter)
        self._paint_attribution(painter)
        painter.end()

    def _paint_pin(self, painter: QPainter) -> None:
        if self._pin is None:
            return
        x, y = self._pixel_at(*self._pin)
        painter.setPen(QColor(THEME.accent_secondary))
        painter.setBrush(QColor(THEME.accent_primary))
        painter.drawEllipse(int(x) - 5, int(y) - 5, 10, 10)
        painter.drawLine(int(x), int(y) - 14, int(x), int(y) - 5)

    def _paint_attribution(self, painter: QPainter) -> None:
        font = QFont(painter.font())
        font.setPointSize(8)
        painter.setFont(font)
        metrics = painter.fontMetrics()
        width = metrics.horizontalAdvance(_ATTRIBUTION) + 8
        height = metrics.height() + 2
        painter.fillRect(self.width() - width, self.height() - height, width, height, QColor(0, 0, 0, 150))
        painter.setPen(QColor(THEME.text_primary))
        painter.drawText(self.width() - width + 4, self.height() - 3 - metrics.descent(), _ATTRIBUTION)

    # ── interaction ──────────────────────────────────────────────────────

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt override
        if event.button() != Qt.MouseButton.LeftButton:
            return
        self._drag_origin = event.pos()
        self._dragged = False

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt override
        if self._drag_origin is None:
            return
        delta = event.pos() - self._drag_origin
        if not self._dragged and delta.manhattanLength() < _DRAG_SLOP_PX:
            return
        self._dragged = True
        self._drag_origin = event.pos()
        cx, cy = deg2tile(*self._center, self._zoom)
        self._center = tile2deg(cx - delta.x() / TILE_SIZE, cy - delta.y() / TILE_SIZE, self._zoom)
        self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt override
        if self._drag_origin is None:
            return
        was_drag = self._dragged
        self._drag_origin = None
        self._dragged = False
        if was_drag:
            return
        lat, lon = self.latlon_at(event.pos().x(), event.pos().y())
        self.set_pin(lat, lon, recenter=False)
        self.pin_moved.emit(lat, lon)

    def zoom_by(self, step: int, anchor: Optional[QPointF] = None) -> None:
        """Zoom whole levels, holding the point under `anchor` fixed (the view centre if None)."""
        zoom = max(MIN_ZOOM, min(MAX_ZOOM, self._zoom + step))
        if zoom == self._zoom:
            return
        if anchor is None:
            self._zoom = zoom
            self.update()
            return
        latlon = self.latlon_at(anchor.x(), anchor.y())
        self._zoom = zoom
        ax, ay = deg2tile(*latlon, zoom)
        self._center = tile2deg(
            ax - (anchor.x() - self.width() / 2.0) / TILE_SIZE,
            ay - (anchor.y() - self.height() / 2.0) / TILE_SIZE,
            zoom,
        )
        self.update()

    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802 - Qt override
        if event.phase() in (Qt.ScrollPhase.ScrollBegin, Qt.ScrollPhase.ScrollEnd):
            self._wheel_accum = 0.0
        notches = event.angleDelta().y()
        if not notches:
            return
        if notches * self._wheel_accum < 0:
            self._wheel_accum = 0.0
        self._wheel_accum += notches
        step = int(self._wheel_accum / _WHEEL_NOTCH)
        if not step:
            return
        self._wheel_accum -= step * _WHEEL_NOTCH
        self.zoom_by(step, event.position())

    def event(self, event) -> bool:
        # A trackpad pinch arrives as a native gesture, not as a wheel, and only where the
        # platform synthesizes one.
        if isinstance(event, QNativeGestureEvent):
            gesture = event.gestureType()
            if gesture == Qt.NativeGestureType.BeginNativeGesture:
                self._pinch_accum = 0.0
            elif gesture == Qt.NativeGestureType.ZoomNativeGesture:
                self._pinch_accum += event.value()
                step = int(self._pinch_accum / _PINCH_STEP)
                if step:
                    self._pinch_accum -= step * _PINCH_STEP
                    self.zoom_by(step, event.position())
                return True
        return super().event(event)
