"""Modal pop-up: position a crop rectangle and a split centerline for half-frame scans.

The widget shows a positive preview of one scan. A draggable/resizable rectangle
defines what is kept (everything outside is discarded). A vertical centerline
inside the rectangle marks the split between the two halves; its thickness
discards a band centered on it (the physical black separator between exposures).

Read after ``exec()`` via ``crop_rect()``, ``split_x()`` and ``gutter_thickness()``.
"""

from typing import Optional

import numpy as np
import qtawesome as qta
from PyQt6.QtCore import QPoint, QRect, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QMouseEvent, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
)

from negpy.desktop.view.styles.theme import THEME

_HANDLE_TOL = 0.04
_HANDLE_PX = 5
_MIN_RECT_W = 0.1
_MIN_RECT_H = 0.1
_SPLIT_TOL = 0.04  # grab radius for the centerline, fraction of width


def _preview_positive(rgb: np.ndarray) -> np.ndarray:
    """Cheap negative->positive preview: per-channel invert + auto-level."""
    a = rgb.astype(np.float32)
    if a.ndim == 2:
        a = a[:, :, None]
    out = np.empty_like(a)
    for c in range(a.shape[2]):
        ch = a[..., c]
        lo, hi = np.percentile(ch, 1), np.percentile(ch, 99)
        out[..., c] = 0.0 if hi <= lo else np.clip((hi - ch) / (hi - lo), 0.0, 1.0) * 255.0
    return out.astype(np.uint8)


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


class _HalfFrameLabel(QLabel):
    """Preview with one draggable/resizable crop rect and a draggable split line."""

    changed = pyqtSignal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(300)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self._pixmap: Optional[QPixmap] = None
        self._rect: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0)
        # split_x is relative to the cropped rect width
        self._split_x: float = 0.5
        self._gutter: float = 0.0
        self._mode: Optional[str] = None  # "draw" | "move" | "resize" | "split"
        self._active_corner: Optional[int] = None
        self._press_frac: Optional[tuple[float, float]] = None
        self._rect_at_press: Optional[tuple[float, float, float, float]] = None
        self._split_at_press: Optional[float] = None

    # ── public API ────────────────────────────────────────────────────

    def set_frame(self, pixmap: QPixmap) -> None:
        self._pixmap = pixmap
        self.update()

    def set_rect(self, rect: tuple[float, float, float, float]) -> None:
        self._rect = rect
        self.update()

    def set_split(self, split_x: float) -> None:
        self._split_x = _clamp01(split_x)
        self.update()

    def set_gutter(self, gutter: float) -> None:
        self._gutter = _clamp01(gutter)
        self.update()

    def rect_value(self) -> tuple[float, float, float, float]:
        x1, y1, x2, y2 = self._rect
        return (_clamp01(min(x1, x2)), _clamp01(min(y1, y2)), _clamp01(max(x1, x2)), _clamp01(max(y1, y2)))

    def split_value(self) -> float:
        return self._split_x

    def gutter_value(self) -> float:
        return self._gutter

    # ── geometry ──────────────────────────────────────────────────────

    def _display(self) -> Optional[QRect]:
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
        fx = _clamp01((p.x() - draw_rect.x()) / max(1, draw_rect.width()))
        fy = _clamp01((p.y() - draw_rect.y()) / max(1, draw_rect.height()))
        return fx, fy

    @staticmethod
    def _rect_in_widget(rect, draw_rect: QRect) -> QRect:
        x1, y1, x2, y2 = rect
        ax = draw_rect.x() + int(x1 * draw_rect.width())
        ay = draw_rect.y() + int(y1 * draw_rect.height())
        bx = draw_rect.x() + int(x2 * draw_rect.width())
        by = draw_rect.y() + int(y2 * draw_rect.height())
        return QRect(QPoint(ax, ay), QPoint(bx, by)).normalized()

    def _split_in_widget(self, draw_rect: QRect) -> int:
        x1, _, x2, _ = self.rect_value()
        cx = x1 + self._split_x * (x2 - x1)
        return draw_rect.x() + int(cx * draw_rect.width())

    @staticmethod
    def _hit_corner(rect, fx: float, fy: float, tol: float) -> Optional[int]:
        x1, y1, x2, y2 = rect
        corners = ((0, x1, y1), (1, x2, y1), (2, x2, y2), (3, x1, y2))
        for idx, cx, cy in corners:
            if abs(fx - cx) <= tol and abs(fy - cy) <= tol:
                return idx
        return None

    # ── mouse ─────────────────────────────────────────────────────────

    def mousePressEvent(self, ev: QMouseEvent) -> None:
        draw_rect = self._display()
        if draw_rect is None:
            return
        fx, fy = self._to_fraction(ev.pos(), draw_rect)
        self._press_frac = (fx, fy)
        self._rect_at_press = self._rect
        self._split_at_press = self._split_x
        r = self.rect_value()
        corner = self._hit_corner(r, fx, fy, _HANDLE_TOL)
        if corner is not None:
            self._mode, self._active_corner = "resize", corner
            return
        # Split line grab: inside the rect, near the centerline
        x1, y1, x2, y2 = r
        cx = x1 + self._split_x * (x2 - x1)
        if y1 <= fy <= y2 and abs(fx - cx) <= _SPLIT_TOL:
            self._mode = "split"
            return
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
        elif self._mode == "resize" and self._active_corner is not None:
            self._rect = self._resize_corner(self._rect_at_press, self._active_corner, fx, fy)
        elif self._mode == "move" and self._rect_at_press is not None:
            dx = fx - self._press_frac[0]
            dy = fy - self._press_frac[1]
            x1, y1, x2, y2 = self._rect_at_press
            w, h = x2 - x1, y2 - y1
            nx1 = _clamp01(x1 + dx)
            ny1 = _clamp01(y1 + dy)
            nx2 = nx1 + w
            ny2 = ny1 + h
            if nx2 > 1.0:
                nx1 -= nx2 - 1.0
                nx2 = 1.0
            if ny2 > 1.0:
                ny1 -= ny2 - 1.0
                ny2 = 1.0
            self._rect = (nx1, ny1, nx2, ny2)
        elif self._mode == "split":
            x1, _, x2, _ = self.rect_value()
            span = max(1e-6, x2 - x1)
            self._split_x = _clamp01((fx - x1) / span)
        self.update()

    def mouseReleaseEvent(self, _ev: QMouseEvent) -> None:
        if self._mode is None:
            return
        r = self._rect
        x1, y1, x2, y2 = r
        if x2 - x1 < _MIN_RECT_W:
            x2 = min(1.0, x1 + _MIN_RECT_W)
        if y2 - y1 < _MIN_RECT_H:
            y2 = min(1.0, y1 + _MIN_RECT_H)
        self._rect = (_clamp01(min(x1, x2)), _clamp01(min(y1, y2)), _clamp01(max(x1, x2)), _clamp01(max(y1, y2)))
        self._mode = None
        self._active_corner = None
        self._press_frac = None
        self._rect_at_press = None
        self._split_at_press = None
        self.update()
        self.changed.emit()

    @staticmethod
    def _resize_corner(rect, corner: int, fx: float, fy: float):
        x1, y1, x2, y2 = rect
        if corner == 0:
            return (_clamp01(fx), _clamp01(fy), x2, y2)
        if corner == 1:
            return (x1, _clamp01(fy), _clamp01(fx), y2)
        if corner == 2:
            return (x1, y1, _clamp01(fx), _clamp01(fy))
        return (_clamp01(fx), y1, x2, _clamp01(fy))

    # ── paint ──────────────────────────────────────────────────────────

    def paintEvent(self, _ev) -> None:
        painter = QPainter(self)
        draw_rect = self._display()
        if draw_rect is not None and self._pixmap is not None and not self._pixmap.isNull():
            painter.save()
            painter.setClipRect(draw_rect)
            painter.drawPixmap(draw_rect, self._pixmap)
            painter.restore()
            r = self.rect_value()
            wr = self._rect_in_widget(r, draw_rect)
            # Dim outside the crop rect
            outside = [
                QRect(draw_rect.topLeft(), QPoint(wr.left() - 1, draw_rect.bottom())),
                QRect(QPoint(wr.right() + 1, draw_rect.top()), draw_rect.bottomRight()),
                QRect(QPoint(wr.left(), draw_rect.top()), QPoint(wr.right(), wr.top() - 1)),
                QRect(QPoint(wr.left(), wr.bottom() + 1), QPoint(wr.right(), draw_rect.bottom())),
            ]
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(0, 0, 0, 130))
            for o in outside:
                if o.width() > 0 and o.height() > 0:
                    painter.drawRect(o)
            # Crop rect
            painter.setPen(QPen(QColor("#1D9E75"), 2))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(wr)
            painter.setBrush(QColor("#1D9E75"))
            painter.setPen(Qt.PenStyle.NoPen)
            for corner in (wr.topLeft(), wr.topRight(), wr.bottomRight(), wr.bottomLeft()):
                painter.drawRect(QRect(corner.x() - _HANDLE_PX, corner.y() - _HANDLE_PX, 2 * _HANDLE_PX, 2 * _HANDLE_PX))
            # Split centerline + gutter band
            x1, _, x2, _ = r
            span = max(1e-6, x2 - x1)
            cx = self._split_in_widget(draw_rect)
            gw = max(0, int(self._gutter * span * draw_rect.width()))
            if gw > 0:
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QColor(224, 168, 60, 120))
                painter.drawRect(QRect(cx - gw // 2, wr.top(), gw, wr.height()))
            pen = QPen(QColor("#E0A83C"), 2)
            pen.setStyle(Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.drawLine(cx, wr.top(), cx, wr.bottom())
        else:
            painter.fillRect(self.rect(), QColor("#0D0D0F"))
            painter.setPen(QColor(THEME.text_muted))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No preview")
        painter.end()


class HalfFrameDialog(QDialog):
    """Pick a crop rectangle and a split centerline for half-frame scans.

    Returns a ``(crop_rect, split_x, gutter_thickness)`` triple; all normalized
    fractions. ``split_x`` is relative to the cropped rect width.
    """

    def __init__(
        self,
        preview_rgb: np.ndarray,
        initial_rect: Optional[tuple[float, float, float, float]] = None,
        initial_split: Optional[float] = None,
        initial_gutter: Optional[float] = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Half Frame — split & crop")
        self.setModal(True)
        self.resize(720, 560)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        self._label = _HalfFrameLabel()
        layout.addWidget(self._label, 1)

        hint = QLabel(
            "Drag the green rectangle to crop. Drag the orange line to set the split. Use the slider to thicken the cut band (the physical black separator)."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet(f"color: {THEME.text_muted};")
        layout.addWidget(hint)

        gutter_row = QHBoxLayout()
        gutter_row.addWidget(QLabel("Cut thickness"))
        self._gutter_slider = QSlider(Qt.Orientation.Horizontal)
        self._gutter_slider.setRange(0, 100)
        self._gutter_slider.setValue(int((initial_gutter or 0.0) * 1000))
        self._gutter_slider.valueChanged.connect(self._on_gutter)
        self._gutter_label = QLabel()
        gutter_row.addWidget(self._gutter_slider, 1)
        gutter_row.addWidget(self._gutter_label)
        layout.addLayout(gutter_row)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self._auto_btn = QPushButton("Auto-detect")
        self._auto_btn.setIcon(qta.icon("fa5s.magic", color=THEME.text_primary))
        self._auto_btn.clicked.connect(self._on_auto)
        btn_row.addWidget(self._auto_btn)
        self._reset_btn = QPushButton("Reset")
        self._reset_btn.setIcon(qta.icon("fa5s.undo", color=THEME.text_primary))
        self._reset_btn.clicked.connect(self._on_reset)
        btn_row.addWidget(self._reset_btn)
        self._ok_btn = QPushButton("Apply")
        self._ok_btn.setObjectName("scan_btn")
        self._ok_btn.setIcon(qta.icon("fa5s.check", color=THEME.text_primary))
        self._ok_btn.clicked.connect(self.accept)
        btn_row.addWidget(self._ok_btn)
        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(self._cancel_btn)
        layout.addLayout(btn_row)

        self._preview_rgb = preview_rgb
        self._set_preview(preview_rgb)
        self._label.set_rect(initial_rect or (0.0, 0.0, 1.0, 1.0))
        self._label.set_split(initial_split if initial_split is not None else 0.5)
        self._label.set_gutter(initial_gutter or 0.0)
        self._label.changed.connect(self._update_gutter_label)
        self._update_gutter_label()

    def _set_preview(self, rgb: np.ndarray) -> None:
        from PyQt6.QtGui import QImage

        pos = _preview_positive(rgb)
        h, w = pos.shape[:2]
        max_dim = 1024
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            pos = pos[:: max(1, int(1 / scale)), :: max(1, int(1 / scale))]
        h2, w2 = pos.shape[:2]
        qimg = QImage(bytes(pos.tobytes()), w2, h2, w2 * 3, QImage.Format.Format_RGB888).copy()
        self._label.set_frame(QPixmap.fromImage(qimg))

    def _on_gutter(self, value: int) -> None:
        self._label.set_gutter(value / 1000.0)
        self._update_gutter_label()

    def _update_gutter_label(self) -> None:
        g = self._label.gutter_value()
        self._gutter_label.setText(f"{g * 100:.1f}%")

    def _on_auto(self) -> None:
        from negpy.services.assets.half_frame import detect_split_x

        sx = detect_split_x(self._preview_rgb)
        self._label.set_split(sx)
        self._update_gutter_label()

    def _on_reset(self) -> None:
        self._label.set_rect((0.0, 0.0, 1.0, 1.0))
        self._label.set_split(0.5)
        self._label.set_gutter(0.0)
        self._gutter_slider.setValue(0)
        self._update_gutter_label()

    # ── results ───────────────────────────────────────────────────────

    def crop_rect(self) -> tuple[float, float, float, float]:
        return self._label.rect_value()

    def split_x(self) -> float:
        return self._label.split_value()

    def gutter_thickness(self) -> float:
        return self._label.gutter_value()
