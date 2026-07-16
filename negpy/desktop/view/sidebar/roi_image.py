"""A QLabel that shows a live frame and lets the user pick a point/region on it.

Coordinates are reported in fractions of the full image (0..1) so a click maps
onto the calibration RAWs (and the camera's focus-magnifier grid) regardless of
preview vs. sensor resolution. In the calibration window a click drops a small
base-sampling patch (or a drag draws a box); in the scan pop-up a click just
emits its fractional position to aim the hardware focus magnifier.
"""

from typing import Optional

from PyQt6.QtCore import QPoint, QRect, QRectF, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QMouseEvent, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import QLabel, QSizePolicy

from negpy.services.capture.calibration import Roi

_CLICK_SLOP = 5  # px: a release within this of the press counts as a click (scan pop-up magnifier)
_CROSSHAIR_FRAC = 0.012  # a click samples a patch this wide (fraction of frame) — the rebate is narrow
_CROSSHAIR_ASPECT = 2.5  # patch height : width in pixels → a vertical strip that fits the rebate bar


class RoiImageLabel(QLabel):
    """Live frame; click to drop a small base-sampling crosshair (calib) or aim the magnifier (scan)."""

    roiChanged = pyqtSignal()
    clicked = pyqtSignal(float, float)  # (fx, fy) in full-frame fractions — toggles the magnifier

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(160)
        # Grow with the (resizable) pop-up — the frame is custom-painted to the
        # current widget size, so the policy must let the label fill the window.
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # roi_mode=True drops the calibration crosshair patch on click (calib window);
        # False just emits `clicked` (scan pop-up → aim the focus magnifier).
        self.roi_mode = True
        self._roi_locked = False  # true while a calibration runs → clicks must not move the patch
        self._pixmap: Optional[QPixmap] = None
        self._roi: Optional[tuple[float, float, float, float]] = None
        self._drag_start: Optional[QPoint] = None
        # Video-player-style buffering spinner drawn on the black frame while a stream spins up.
        self._loading = False
        self._spin_angle = 0
        self._spin_timer = QTimer(self)
        self._spin_timer.setInterval(70)
        self._spin_timer.timeout.connect(self._advance_spinner)

    # ── public API ────────────────────────────────────────────────────

    def set_frame(self, pixmap: QPixmap) -> None:
        self._pixmap = pixmap
        if self._loading:
            self.set_loading(False)  # first real frame arrived → drop the buffering spinner
        self.update()

    def clear_frame(self) -> None:
        """Drop the current frame → the widget goes black (e.g. while a new stream starts)."""
        self._pixmap = None
        self.update()

    def set_loading(self, on: bool) -> None:
        """Show/hide a buffering spinner over the black frame while a stream is starting."""
        if on == self._loading:
            return
        self._loading = on
        self._spin_timer.start() if on else self._spin_timer.stop()
        self.update()

    def _advance_spinner(self) -> None:
        self._spin_angle = (self._spin_angle + 30) % 360
        self.update()

    def set_roi_locked(self, locked: bool) -> None:
        """Freeze the base ROI while a calibration is metering it: a click no longer moves the
        sampling patch. The cursor drops the crosshair so it's clear the target is fixed."""
        self._roi_locked = locked
        self.setCursor(Qt.CursorShape.ArrowCursor if locked else Qt.CursorShape.CrossCursor)

    def clear_roi(self) -> None:
        self._roi = None
        self.update()
        self.roiChanged.emit()

    def _set_crosshair(self, cx: float, cy: float) -> None:
        """Centre a small vertical sampling strip on (cx, cy), clamped inside the frame.

        The clear film rebate is a narrow vertical bar, so the patch is a vertical rectangle:
        narrow width, `_CROSSHAIR_ASPECT`× taller in *pixels*. The height fraction is scaled by
        the frame aspect so the pixel shape holds on non-square frames (screen + sensor alike)."""
        w = _CROSSHAIR_FRAC
        aspect = 1.0
        if self._pixmap is not None and not self._pixmap.isNull() and self._pixmap.height() > 0:
            aspect = self._pixmap.width() / self._pixmap.height()
        h = min(_CROSSHAIR_ASPECT * w * aspect, 1.0)
        x = min(max(cx - w / 2, 0.0), 1.0 - w)
        y = min(max(cy - h / 2, 0.0), 1.0 - h)
        self._roi = (x, y, w, h)
        self.update()
        self.roiChanged.emit()

    def roi(self) -> Optional[Roi]:
        if self._roi is None:
            return None
        return Roi(*self._roi)

    # ── geometry ──────────────────────────────────────────────────────

    def _display(self) -> Optional[QRect]:
        """Widget-px rect the full frame is drawn into (letterboxed & centred); None when no frame."""
        if self._pixmap is None or self._pixmap.isNull():
            return None
        pw, ph = self._pixmap.width(), self._pixmap.height()
        if pw <= 0 or ph <= 0:
            return None
        scale = min(self.width() / pw, self.height() / ph)
        dw, dh = int(pw * scale), int(ph * scale)
        return QRect((self.width() - dw) // 2, (self.height() - dh) // 2, dw, dh)

    @staticmethod
    def _to_fraction(p: QPoint, draw_rect: QRect) -> tuple[float, float]:
        fx = min(1.0, max(0.0, (p.x() - draw_rect.x()) / max(1, draw_rect.width())))
        fy = min(1.0, max(0.0, (p.y() - draw_rect.y()) / max(1, draw_rect.height())))
        return fx, fy

    # ── mouse ─────────────────────────────────────────────────────────

    def mousePressEvent(self, ev: QMouseEvent) -> None:
        if self._display() is not None:
            self._drag_start = ev.pos()
            self.update()

    def mouseReleaseEvent(self, ev: QMouseEvent) -> None:
        draw_rect = self._display()
        if self._drag_start is not None and draw_rect is not None:
            cx, cy = self._to_fraction(ev.pos(), draw_rect)
            if self.roi_mode:
                if not self._roi_locked:  # locked while a calibration meters the patch
                    self._set_crosshair(cx, cy)  # calibration: drop the small base-sampling patch (no drag-box)
            elif (ev.pos() - self._drag_start).manhattanLength() < _CLICK_SLOP:
                self.clicked.emit(cx, cy)  # scan pop-up: a click toggles the focus magnifier
        self._drag_start = None
        self.update()

    # ── paint ─────────────────────────────────────────────────────────

    def paintEvent(self, _ev) -> None:
        painter = QPainter(self)
        draw_rect = self._display()
        if draw_rect is not None and self._pixmap is not None:
            painter.drawPixmap(draw_rect, self._pixmap)
            if self._roi is not None:  # just the box outline — no centre cross (cleaner)
                painter.setPen(QPen(QColor("#1D9E75"), 2))
                painter.drawRect(self._roi_in_widget(self._roi, draw_rect))
        else:
            painter.fillRect(self.rect(), QColor("#0D0D0F"))  # black while there's no frame (e.g. stream starting)
            if self._loading:  # stream starting → video-player-style buffering hint
                self._paint_spinner(painter)
            elif self.roi_mode:  # calibration window: guide the user to pick the film base
                painter.setPen(QColor("#888780"))
                painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Live View → click the clear film base")
        painter.end()

    def _paint_spinner(self, painter: QPainter) -> None:
        """A rotating arc + label on the black frame — signals the stream is buffering."""
        r = 16.0
        cx = self.width() / 2.0
        cy = self.height() / 2.0 - 12
        pen = QPen(QColor("#B4B2A9"), 3)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        # drawArc angles are 1/16°; sweep a 90° arc rotating with _spin_angle.
        painter.drawArc(QRectF(cx - r, cy - r, 2 * r, 2 * r), -self._spin_angle * 16, 90 * 16)
        painter.setPen(QColor("#888780"))
        painter.drawText(QRect(0, int(cy + r + 6), self.width(), 22), Qt.AlignmentFlag.AlignHCenter, "Loading live view…")

    @staticmethod
    def _roi_in_widget(roi, draw_rect: QRect) -> QRect:
        rx, ry, rw, rh = roi
        x0 = draw_rect.x() + int(rx * draw_rect.width())
        y0 = draw_rect.y() + int(ry * draw_rect.height())
        x1 = draw_rect.x() + int((rx + rw) * draw_rect.width())
        y1 = draw_rect.y() + int((ry + rh) * draw_rect.height())
        return QRect(QPoint(x0, y0), QPoint(x1, y1)).normalized()
