from typing import Any

import numpy as np
from PyQt6.QtCore import QPointF, Qt
from PyQt6.QtGui import (
    QBrush,
    QColor,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPen,
)
from PyQt6.QtWidgets import QSizePolicy, QWidget

from negpy.kernel.image.logic import get_luminance

_CLIP_THRESH = 0.005  # fraction of pixels considered "clipping"


class HistogramWidget(QWidget):
    """
    Native high-performance histogram using QPainter.
    Offers additive blending-like visuals and reliable updates.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumHeight(40)
        self._data_r: list = []
        self._data_g: list = []
        self._data_b: list = []
        self._data_l: list = []
        self._clip_low: dict[str, bool] = {}
        self._clip_high: dict[str, bool] = {}

    def update_data(self, buffer: Any) -> None:
        """Calculates histograms and triggers repaint."""
        if buffer is None:
            self._data_r = []
            self._data_g = []
            self._data_b = []
            self._data_l = []
            self._clip_low = {}
            self._clip_high = {}
            self.update()
            return

        if isinstance(buffer, np.ndarray) and buffer.shape == (4, 256):
            self._data_r = self._normalize(buffer[0])
            self._data_g = self._normalize(buffer[1])
            self._data_b = self._normalize(buffer[2])
            self._data_l = self._normalize(buffer[3])
            totals = [max(1.0, float(buffer[c].sum())) for c in range(3)]
            self._clip_low = {
                "r": buffer[0][0] / totals[0] > _CLIP_THRESH,
                "g": buffer[1][0] / totals[1] > _CLIP_THRESH,
                "b": buffer[2][0] / totals[2] > _CLIP_THRESH,
            }
            self._clip_high = {
                "r": buffer[0][255] / totals[0] > _CLIP_THRESH,
                "g": buffer[1][255] / totals[1] > _CLIP_THRESH,
                "b": buffer[2][255] / totals[2] > _CLIP_THRESH,
            }
            self.update()
            return

        if not isinstance(buffer, np.ndarray):
            return

        if buffer.shape[0] > 500:
            buffer = buffer[::4, ::4]

        lum = get_luminance(buffer)
        self._data_r = self._calc_hist(buffer[..., 0])
        self._data_g = self._calc_hist(buffer[..., 1])
        self._data_b = self._calc_hist(buffer[..., 2])
        self._data_l = self._calc_hist(lum)

        n = max(1, buffer.shape[0] * buffer.shape[1])
        self._clip_low = {
            "r": float(np.sum(buffer[..., 0] <= 0.002)) / n > _CLIP_THRESH,
            "g": float(np.sum(buffer[..., 1] <= 0.002)) / n > _CLIP_THRESH,
            "b": float(np.sum(buffer[..., 2] <= 0.002)) / n > _CLIP_THRESH,
        }
        self._clip_high = {
            "r": float(np.sum(buffer[..., 0] >= 0.998)) / n > _CLIP_THRESH,
            "g": float(np.sum(buffer[..., 1] >= 0.998)) / n > _CLIP_THRESH,
            "b": float(np.sum(buffer[..., 2] >= 0.998)) / n > _CLIP_THRESH,
        }
        self.update()

    def _normalize(self, counts: np.ndarray) -> list:
        max_val = float(np.max(counts))
        if max_val <= 0:
            return []
        return (counts.astype(float) / max_val).tolist()

    def _calc_hist(self, data: np.ndarray) -> list:
        hist, _ = np.histogram(data, bins=256, range=(0, 1))
        max_val = hist.max()
        if max_val <= 0:
            return []
        return (hist.astype(float) / max_val).tolist()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w = self.width()
        h = self.height()

        # Background and border
        rect = self.rect().adjusted(0, 0, -1, -1)
        painter.fillRect(rect, QColor("#050505"))
        painter.setPen(QPen(QColor("#262626"), 1))
        painter.drawRect(rect)

        # Quarter-tone grid lines
        painter.setPen(QPen(QColor("#1A1A1A"), 1))
        for i in range(1, 4):
            x = int(w * i / 4)
            painter.drawLine(x, 0, x, h)
            y = int(h * i / 4)
            painter.drawLine(0, y, w, y)

        # Channels
        self._draw_channel(painter, self._data_l, "#D4D4D4", 30, 150, w, h)
        self._draw_channel(painter, self._data_r, "#D32F2F", 80, 200, w, h)
        self._draw_channel(painter, self._data_g, "#388E3C", 80, 200, w, h)
        self._draw_channel(painter, self._data_b, "#1976D2", 80, 200, w, h)

        # H2: Zone tick marks at 0.1 intervals along the bottom
        painter.setPen(QPen(QColor("#3A3A3A"), 1))
        for i in range(1, 10):
            x = int(w * i * 0.1)
            painter.drawLine(x, h - 5, x, h - 1)

        # H1: Per-channel clipping indicators
        self._draw_clip_indicators(painter, w, h)

    def _draw_clip_indicators(self, painter: QPainter, w: int, h: int) -> None:
        channels = [("r", "#D32F2F"), ("g", "#388E3C"), ("b", "#1976D2")]
        size = 5
        gap = size + 2

        painter.setPen(Qt.PenStyle.NoPen)
        for i, (ch, color) in enumerate(channels):
            y = 4 + i * gap
            c = QColor(color)

            if self._clip_low.get(ch):
                # Right-pointing triangle → shadows clipping to black
                tri = QPainterPath()
                tri.moveTo(3.0, float(y))
                tri.lineTo(3.0, float(y + size))
                tri.lineTo(3.0 + size, float(y + size / 2))
                tri.closeSubpath()
                painter.fillPath(tri, QBrush(c))

            if self._clip_high.get(ch):
                # Left-pointing triangle ← highlights clipping to white
                tri = QPainterPath()
                tri.moveTo(float(w - 3), float(y))
                tri.lineTo(float(w - 3), float(y + size))
                tri.lineTo(float(w - 3 - size), float(y + size / 2))
                tri.closeSubpath()
                painter.fillPath(tri, QBrush(c))

    def _draw_channel(
        self,
        painter: QPainter,
        data: list,
        color_hex: str,
        alpha_fill: int,
        alpha_line: int,
        w: int,
        h: int,
    ) -> None:
        if len(data) < 2:
            return

        path = QPainterPath()
        path.moveTo(0, h)

        step = w / (len(data) - 1)
        for i, val in enumerate(data):
            path.lineTo(i * step, h - val * h)

        path.lineTo(w, h)
        path.closeSubpath()

        c_fill = QColor(color_hex)
        c_fill.setAlpha(alpha_fill)
        painter.setBrush(QBrush(c_fill))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawPath(path)

        path_line = QPainterPath()
        path_line.moveTo(0, h - data[0] * h)
        for i, val in enumerate(data):
            path_line.lineTo(i * step, h - val * h)

        c_line = QColor(color_hex)
        c_line.setAlpha(alpha_line)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(c_line, 1.5))
        painter.drawPath(path_line)


class PhotometricCurveWidget(QWidget):
    """
    H&D sigmoid curve visualization using native QPainter.
    Annotates the pivot point, toe/shoulder zones, gradient fill, and zone ticks.
    """

    # Data coordinate ranges
    _X_MIN, _X_MAX = -0.1, 1.1   # plt_x domain
    _Y_MIN, _Y_MAX = -0.05, 1.05  # output domain

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumHeight(40)
        self._curve_pts: list[tuple[float, float]] = []
        self._pivot_pt: tuple[float, float] | None = None
        self._toe_mask: list[float] = []
        self._shoulder_mask: list[float] = []
        self._toe_strength: float = 0.0
        self._shoulder_strength: float = 0.0

    # ── coordinate helpers ────────────────────────────────────────────────────

    def _wx(self, dx: float, w: int) -> float:
        return (dx - self._X_MIN) / (self._X_MAX - self._X_MIN) * w

    def _wy(self, dy: float, h: int) -> float:
        return h - (dy - self._Y_MIN) / (self._Y_MAX - self._Y_MIN) * h

    # ── data update ──────────────────────────────────────────────────────────

    def update_curve(self, params) -> None:
        from negpy.features.exposure.logic import LogisticSigmoid
        from negpy.features.exposure.models import EXPOSURE_CONSTANTS
        from negpy.kernel.image.validation import ensure_image

        exposure_shift = 0.1 + (params.density * EXPOSURE_CONSTANTS["density_multiplier"])
        pivot = 1.0 - exposure_shift
        slope = 1.0 + (params.grade * EXPOSURE_CONSTANTS["grade_multiplier"])

        curve = LogisticSigmoid(
            contrast=slope,
            pivot=pivot,
            d_max=3.5,
            toe=params.toe,
            toe_width=params.toe_width,
            shoulder=params.shoulder,
            shoulder_width=params.shoulder_width,
        )

        n = 300
        plt_x = np.linspace(self._X_MIN, self._X_MAX, n)
        x_log_exp = 1.0 - plt_x

        d = curve(ensure_image(x_log_exp))
        t = np.power(10.0, -d)
        y = np.power(t, 1.0 / 2.2)
        self._curve_pts = list(zip(plt_x.tolist(), y.tolist()))

        # Toe/shoulder masks for zone shading (same formula as LogisticSigmoid)
        diff = x_log_exp - pivot
        epsilon = 1e-6
        t_val = params.toe_width * (diff / max(1.0 - pivot, epsilon) - 0.5)
        self._toe_mask = (1.0 / (1.0 + np.exp(-t_val))).tolist()
        s_val = -params.shoulder_width * (diff / max(pivot, epsilon) + 0.5)
        self._shoulder_mask = (1.0 / (1.0 + np.exp(-s_val))).tolist()
        self._toe_strength = params.toe
        self._shoulder_strength = params.shoulder

        # Pivot in widget x-space: x_log_exp = pivot → plt_x = 1 - pivot
        pivot_plt_x = float(np.clip(1.0 - pivot, self._X_MIN, self._X_MAX))
        idx = round((pivot_plt_x - self._X_MIN) / (self._X_MAX - self._X_MIN) * (n - 1))
        idx = max(0, min(len(self._curve_pts) - 1, idx))
        self._pivot_pt = self._curve_pts[idx]

        self.update()

    # ── painting ─────────────────────────────────────────────────────────────

    def paintEvent(self, event) -> None:
        if not self._curve_pts:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w = self.width()
        h = self.height()

        # Background + border
        painter.fillRect(self.rect(), QColor("#050505"))
        painter.setPen(QPen(QColor("#262626"), 1))
        painter.drawRect(self.rect().adjusted(0, 0, -1, -1))

        # Grid at 0.25 intervals
        painter.setPen(QPen(QColor("#1A1A1A"), 1))
        for i in range(1, 4):
            gx = int(self._wx(i * 0.25, w))
            gy = int(self._wy(i * 0.25, h))
            painter.drawLine(gx, 0, gx, h)
            painter.drawLine(0, gy, w, gy)

        # Diagonal reference (dashed)
        painter.setPen(QPen(QColor("#2E2E2E"), 1, Qt.PenStyle.DashLine))
        painter.drawLine(
            int(self._wx(0.0, w)), int(self._wy(0.0, h)),
            int(self._wx(1.0, w)), int(self._wy(1.0, h)),
        )

        # Build the main curve path (reused for fill and line)
        curve_path = QPainterPath()
        curve_path.moveTo(self._wx(self._curve_pts[0][0], w), self._wy(self._curve_pts[0][1], h))
        for px, py in self._curve_pts[1:]:
            curve_path.lineTo(self._wx(px, w), self._wy(py, h))

        # P4: Toe zone shading (warm amber — right side, dense silver = shadows)
        self._draw_zone_shading(painter, w, h, self._toe_mask, self._toe_strength, QColor(255, 140, 50))

        # P4: Shoulder zone shading (cool blue — left side, thin silver = highlights)
        self._draw_zone_shading(painter, w, h, self._shoulder_mask, self._shoulder_strength, QColor(60, 130, 255))

        # P2: Gradient luminance fill under the curve
        fill_path = QPainterPath(curve_path)
        bot = self._wy(self._Y_MIN, h)
        fill_path.lineTo(self._wx(self._curve_pts[-1][0], w), bot)
        fill_path.lineTo(self._wx(self._curve_pts[0][0], w), bot)
        fill_path.closeSubpath()

        gradient = QLinearGradient(0.0, 0.0, float(w), 0.0)
        gradient.setColorAt(0.0, QColor(0, 0, 0, 55))
        gradient.setColorAt(1.0, QColor(255, 255, 255, 55))
        painter.setBrush(QBrush(gradient))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawPath(fill_path)

        # P5: Zone tick marks along the bottom (Adams Zone I–IX)
        painter.setPen(QPen(QColor("#3A3A3A"), 1))
        for i in range(1, 10):
            zx = int(self._wx(i * 0.1, w))
            painter.drawLine(zx, h - 5, zx, h - 1)

        # Curve line (drawn after fills so it sits on top)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(QColor("#FFFFFF"), 1.5))
        painter.drawPath(curve_path)

        # P3: Pivot crosshairs + dot
        if self._pivot_pt:
            wpx = self._wx(self._pivot_pt[0], w)
            wpy = self._wy(self._pivot_pt[1], h)

            painter.setPen(QPen(QColor(200, 200, 200, 45), 1, Qt.PenStyle.DotLine))
            painter.drawLine(int(wpx), 0, int(wpx), h)
            painter.drawLine(0, int(wpy), w, int(wpy))

            painter.setBrush(QBrush(QColor("#FFFFFF")))
            painter.setPen(QPen(QColor("#050505"), 1))
            painter.drawEllipse(QPointF(wpx, wpy), 3.5, 3.5)

    def _draw_zone_shading(
        self,
        painter: QPainter,
        w: int,
        h: int,
        mask: list[float],
        strength: float,
        color: QColor,
    ) -> None:
        if strength < 0.01 or not mask or not self._curve_pts:
            return

        bot = self._wy(self._Y_MIN, h)
        painter.setPen(Qt.PenStyle.NoPen)

        for i in range(len(self._curve_pts) - 1):
            mask_avg = (mask[i] + mask[i + 1]) * 0.5
            alpha = int(mask_avg * strength * 70)
            if alpha < 3:
                continue
            px1, py1 = self._curve_pts[i]
            px2, py2 = self._curve_pts[i + 1]

            strip = QPainterPath()
            strip.moveTo(self._wx(px1, w), self._wy(py1, h))
            strip.lineTo(self._wx(px2, w), self._wy(py2, h))
            strip.lineTo(self._wx(px2, w), bot)
            strip.lineTo(self._wx(px1, w), bot)
            strip.closeSubpath()

            c = QColor(color)
            c.setAlpha(alpha)
            painter.fillPath(strip, QBrush(c))
