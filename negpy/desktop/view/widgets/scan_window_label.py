"""QLabel showing a scan preview with one draggable/resizable window rect (0..1, crop convention).

Rect math lives in ``scan_window_geometry``; scaffolding mirrors ``sidebar/roi_image.py``.
"""

from typing import Optional

from PyQt6.QtCore import QPoint, QRect, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QMouseEvent, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import QLabel, QSizePolicy

from negpy.desktop.view.widgets.scan_window_geometry import (
    Rect,
    hit_corner,
    normalize_rect,
    resize_corner,
)
from negpy.features.geometry.logic import translate_manual_crop_rect

_HANDLE_TOL = 0.03  # corner grab radius, fraction of frame
_HANDLE_PX = 5  # drawn handle half-size, widget px


class ScanWindowLabel(QLabel):
    """Preview frame with one draggable/resizable scan-window rectangle."""

    windowChanged = pyqtSignal(object)  # (x1, y1, x2, y2) in 0..1, or None

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(220)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self._pixmap: Optional[QPixmap] = None
        self._coverage: Optional[tuple[float, float]] = None
        self._rect: Optional[Rect] = None
        self._mode: Optional[str] = None  # "draw" | "move" | "resize"
        self._active_corner: Optional[int] = None
        self._press_frac: Optional[tuple[float, float]] = None
        self._rect_at_press: Optional[Rect] = None
        self._offset_indicators: list[tuple[float, str]] = []  # (frac 0..1, "left" | "right")

    # ── public API ────────────────────────────────────────────────────

    def set_frame(self, pixmap: QPixmap, coverage: Optional[tuple[float, float]] = None) -> None:
        """Show a preview. `coverage` is the (start, end) span this pixmap occupies along
        x, in fractions of the frame the widget represents — the *next scan's* raster, so
        fractions read off this widget (crop rects, offset lines) are exactly the window
        fractions the backend applies. `start` may be negative: a raster previewed at a
        lower offset than the slider now reads slides left and clips off the edge."""
        self._pixmap = pixmap
        self._coverage = coverage
        self.update()

    def set_coverage(self, coverage: Optional[tuple[float, float]]) -> None:
        """Re-place the kept pixmap (offset slider moved); no new scan needed."""
        self._coverage = coverage
        self.update()

    def set_window(self, rect: Optional[Rect]) -> None:
        self._rect = tuple(rect) if rect is not None else None  # type: ignore[assignment]
        self.update()

    def window(self) -> Optional[Rect]:
        return self._rect

    def set_offset_indicators(self, indicators: list[tuple[float, str]]) -> None:
        """Live cut lines, each (frac 0..1, edge "left"/"right"): a shaded band
        grows from that edge to a dashed line at frac."""
        self._offset_indicators = [(max(0.0, min(1.0, frac)), edge) for frac, edge in indicators]
        self.update()

    def clear_window(self) -> None:
        self._rect = None
        self.update()
        self.windowChanged.emit(None)

    def has_frame(self) -> bool:
        return self._pixmap is not None and not self._pixmap.isNull()

    # ── geometry ──────────────────────────────────────────────────────

    def _display(self) -> Optional[QRect]:
        """Widget-px rect the full frame is drawn into (letterboxed & centred)."""
        if self._pixmap is None or self._pixmap.isNull():
            return None
        pw, ph = self._pixmap.width(), self._pixmap.height()
        if pw <= 0 or ph <= 0:
            return None
        span = self._coverage_span()
        pw = pw / span  # the frame is wider than the raster when the scan started past 0
        scale = min(self.width() / pw, self.height() / ph)
        dw, dh = int(pw * scale), int(ph * scale)
        return QRect((self.width() - dw) // 2, (self.height() - dh) // 2, dw, dh)

    def _coverage_span(self) -> float:
        if self._coverage is None:
            return 1.0
        start, end = self._coverage
        return max(1e-3, end - start)

    def _content_rect(self, draw_rect: QRect) -> QRect:
        """Sub-rect the pixmap occupies; may overflow draw_rect (painting clips)."""
        if self._coverage is None:
            return draw_rect
        x = draw_rect.left() + int(self._coverage[0] * draw_rect.width())
        return QRect(x, draw_rect.top(), max(1, int(self._coverage_span() * draw_rect.width())), draw_rect.height())

    @staticmethod
    def _to_fraction(p: QPoint, draw_rect: QRect) -> tuple[float, float]:
        fx = min(1.0, max(0.0, (p.x() - draw_rect.x()) / max(1, draw_rect.width())))
        fy = min(1.0, max(0.0, (p.y() - draw_rect.y()) / max(1, draw_rect.height())))
        return fx, fy

    @staticmethod
    def _rect_in_widget(rect: Rect, draw_rect: QRect) -> QRect:
        x1, y1, x2, y2 = rect
        ax = draw_rect.x() + int(x1 * draw_rect.width())
        ay = draw_rect.y() + int(y1 * draw_rect.height())
        bx = draw_rect.x() + int(x2 * draw_rect.width())
        by = draw_rect.y() + int(y2 * draw_rect.height())
        return QRect(QPoint(ax, ay), QPoint(bx, by)).normalized()

    # ── mouse ─────────────────────────────────────────────────────────

    def mousePressEvent(self, ev: QMouseEvent) -> None:
        draw_rect = self._display()
        if draw_rect is None:
            return
        fx, fy = self._to_fraction(ev.pos(), draw_rect)
        self._press_frac = (fx, fy)
        self._rect_at_press = self._rect
        if self._rect is not None:
            corner = hit_corner(self._rect, fx, fy, _HANDLE_TOL)
            if corner is not None:
                self._mode, self._active_corner = "resize", corner
                return
            x1, y1, x2, y2 = self._rect
            if x1 <= fx <= x2 and y1 <= fy <= y2:
                self._mode = "move"
                return
        self._mode = "draw"
        self._rect = (fx, fy, fx, fy)
        self.update()

    def mouseMoveEvent(self, ev: QMouseEvent) -> None:
        draw_rect = self._display()
        if draw_rect is None or self._mode is None or self._press_frac is None:
            return
        fx, fy = self._to_fraction(ev.pos(), draw_rect)
        if self._mode == "draw":
            px, py = self._press_frac
            self._rect = (px, py, fx, fy)
        elif self._mode == "resize" and self._rect is not None and self._active_corner is not None:
            self._rect = resize_corner(self._rect, self._active_corner, fx, fy)
        elif self._mode == "move" and self._rect_at_press is not None:
            self._rect = translate_manual_crop_rect(self._rect_at_press, fx - self._press_frac[0], fy - self._press_frac[1])
        self.update()

    def mouseReleaseEvent(self, _ev: QMouseEvent) -> None:
        if self._mode is None:
            return
        if self._rect is not None:
            self._rect = normalize_rect(self._rect)
        self._mode = None
        self._active_corner = None
        self._press_frac = None
        self._rect_at_press = None
        self.update()
        self.windowChanged.emit(self._rect)

    # ── paint ─────────────────────────────────────────────────────────

    def paintEvent(self, _ev) -> None:
        painter = QPainter(self)
        draw_rect = self._display()
        if draw_rect is not None and self._pixmap is not None:
            content = self._content_rect(draw_rect)
            if content != draw_rect:
                painter.fillRect(draw_rect, QColor("#0D0D0F"))
            painter.save()
            painter.setClipRect(draw_rect)
            painter.drawPixmap(content, self._pixmap)
            painter.restore()
            if self._rect is not None:
                wr = self._rect_in_widget(self._rect, draw_rect)
                painter.setPen(QPen(QColor("#1D9E75"), 2))
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawRect(wr)
                painter.setBrush(QColor("#1D9E75"))
                painter.setPen(Qt.PenStyle.NoPen)
                for corner in (wr.topLeft(), wr.topRight(), wr.bottomRight(), wr.bottomLeft()):
                    painter.drawRect(QRect(corner.x() - _HANDLE_PX, corner.y() - _HANDLE_PX, 2 * _HANDLE_PX, 2 * _HANDLE_PX))
            for frac, edge in self._offset_indicators:
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QColor(0, 0, 0, 110))
                pen = QPen(QColor("#E0A83C"), 2)
                pen.setStyle(Qt.PenStyle.DashLine)
                # Line ≥1 px inside the frame — an edge-pinned indicator must stay visible.
                if edge == "left":
                    x = min(draw_rect.right() - 1, max(draw_rect.left() + 1, draw_rect.left() + int(frac * draw_rect.width())))
                    painter.drawRect(QRect(draw_rect.left(), draw_rect.top(), max(0, x - draw_rect.left()), draw_rect.height()))
                else:
                    x = min(draw_rect.right() - 1, max(draw_rect.left() + 1, draw_rect.right() - int(frac * draw_rect.width())))
                    painter.drawRect(QRect(x, draw_rect.top(), max(0, draw_rect.right() - x), draw_rect.height()))
                painter.setPen(pen)
                painter.drawLine(x, draw_rect.top(), x, draw_rect.bottom())
        else:
            painter.fillRect(self.rect(), QColor("#0D0D0F"))
        painter.end()
