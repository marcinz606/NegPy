from typing import Any

import numpy as np
from PyQt6.QtCore import QPointF, QRect, Qt, pyqtSignal
from PyQt6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPen,
)
from PyQt6.QtWidgets import QSizePolicy, QWidget

from negpy.desktop.view.styles.theme import THEME
from negpy.kernel.image.logic import get_luminance, working_oetf_encode

_CLIP_THRESH = 0.005  # fraction of pixels considered "clipping"


class HistogramWidget(QWidget):
    """
    Native high-performance histogram using QPainter.
    Offers additive blending-like visuals and reliable updates.

    Raw per-channel bin counts are retained so the vertical scale can switch
    between linear and logarithmic without re-deriving them from the source.
    """

    # Emitted when the user flips the in-widget LIN/LOG toggle (True == log).
    scale_changed = pyqtSignal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumHeight(40)
        # Raw bin counts per channel (256-length arrays); empty until first frame.
        self._counts: dict[str, np.ndarray] = {}
        self._log_scale: bool = False
        self._clip_low: dict[str, bool] = {}
        self._clip_high: dict[str, bool] = {}
        self._clip_low_frac: float = 0.0
        self._clip_high_frac: float = 0.0
        # Hit rectangles for the LIN/LOG toggle, recomputed each paint.
        self._lin_rect = QRect()
        self._log_rect = QRect()
        self.setMouseTracking(True)

    def clip_fractions(self) -> tuple[float, float]:
        """Worst-channel shadow / highlight clipped fraction (0..1) of the last frame."""
        return self._clip_low_frac, self._clip_high_frac

    def log_scale(self) -> bool:
        return self._log_scale

    def set_log_scale(self, enabled: bool) -> None:
        """Switch the vertical axis between linear and logarithmic scaling."""
        enabled = bool(enabled)
        if enabled == self._log_scale:
            return
        self._log_scale = enabled
        self.update()

    def update_data(self, buffer: Any) -> None:
        """Calculates histograms and triggers repaint."""
        if buffer is None:
            self._counts = {}
            self._clip_low = {}
            self._clip_high = {}
            self._clip_low_frac = 0.0
            self._clip_high_frac = 0.0
            self.update()
            return

        if isinstance(buffer, np.ndarray) and buffer.shape == (4, 256):
            self._counts = {
                "r": buffer[0].astype(float),
                "g": buffer[1].astype(float),
                "b": buffer[2].astype(float),
                "l": buffer[3].astype(float),
            }
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
            self._clip_low_frac = max(float(buffer[c][0]) / totals[c] for c in range(3))
            self._clip_high_frac = max(float(buffer[c][255]) / totals[c] for c in range(3))
            self.update()
            return

        if not isinstance(buffer, np.ndarray):
            return

        if buffer.shape[0] > 500:
            buffer = buffer[::4, ::4]

        lum = get_luminance(buffer)
        self._counts = {
            "r": self._hist_counts(buffer[..., 0]),
            "g": self._hist_counts(buffer[..., 1]),
            "b": self._hist_counts(buffer[..., 2]),
            "l": self._hist_counts(lum),
        }

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
        self._clip_low_frac = max(float(np.sum(buffer[..., c] <= 0.002)) / n for c in range(3))
        self._clip_high_frac = max(float(np.sum(buffer[..., c] >= 0.998)) / n for c in range(3))
        self.update()

    @staticmethod
    def _hist_counts(data: np.ndarray) -> np.ndarray:
        hist, _ = np.histogram(data, bins=256, range=(0, 1))
        return hist.astype(float)

    def _display(self, key: str) -> list:
        """Normalized 0..1 plot values for a channel under the current scale mode."""
        counts = self._counts.get(key)
        if counts is None or counts.size == 0:
            return []
        vals = np.log1p(counts) if self._log_scale else counts
        max_val = float(np.max(vals))
        if max_val <= 0:
            return []
        return (vals / max_val).tolist()

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
        self._draw_channel(painter, self._display("l"), "#D4D4D4", 30, 150, w, h)
        self._draw_channel(painter, self._display("r"), THEME.channel_red, 80, 200, w, h)
        self._draw_channel(painter, self._display("g"), THEME.channel_green, 80, 200, w, h)
        self._draw_channel(painter, self._display("b"), THEME.channel_blue, 80, 200, w, h)

        # H2: Zone tick marks at 0.1 intervals along the bottom
        painter.setPen(QPen(QColor("#3A3A3A"), 1))
        for i in range(1, 10):
            x = int(w * i * 0.1)
            painter.drawLine(x, h - 5, x, h - 1)

        # H1: Per-channel clipping indicators
        self._draw_clip_indicators(painter, w, h)

        # LIN/LOG scale toggle (bottom-right, clear of the top-corner clip marks)
        self._draw_scale_toggle(painter, w, h)

    def _draw_scale_toggle(self, painter: QPainter, w: int, h: int) -> None:
        seg_w, seg_h = 24, 13
        margin = 5
        x0 = w - seg_w * 2 - margin
        y0 = h - seg_h - margin
        self._lin_rect = QRect(x0, y0, seg_w, seg_h)
        self._log_rect = QRect(x0 + seg_w, y0, seg_w, seg_h)
        outer = QRect(x0, y0, seg_w * 2, seg_h)

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor(10, 10, 10, 200)))
        painter.drawRoundedRect(outer, 3, 3)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(QColor("#333333"), 1))
        painter.drawRoundedRect(outer, 3, 3)

        font = QFont()
        font.setPixelSize(8)
        font.setBold(True)
        painter.setFont(font)

        active = QColor("#E5E5E5")
        inactive = QColor("#6B6B6B")
        highlight = QColor(60, 130, 255, 70)

        if not self._log_scale:
            painter.fillRect(self._lin_rect.adjusted(1, 1, -1, -1), QBrush(highlight))
        else:
            painter.fillRect(self._log_rect.adjusted(1, 1, -1, -1), QBrush(highlight))

        painter.setPen(QPen(active if not self._log_scale else inactive))
        painter.drawText(self._lin_rect, Qt.AlignmentFlag.AlignCenter, "LIN")
        painter.setPen(QPen(active if self._log_scale else inactive))
        painter.drawText(self._log_rect, Qt.AlignmentFlag.AlignCenter, "LOG")

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            pos = event.position().toPoint()
            if self._lin_rect.contains(pos) and self._log_scale:
                self.set_log_scale(False)
                self.scale_changed.emit(False)
                return
            if self._log_rect.contains(pos) and not self._log_scale:
                self.set_log_scale(True)
                self.scale_changed.emit(True)
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        pos = event.position().toPoint()
        over = self._lin_rect.contains(pos) or self._log_rect.contains(pos)
        self.setCursor(Qt.CursorShape.PointingHandCursor if over else Qt.CursorShape.ArrowCursor)
        super().mouseMoveEvent(event)

    def _draw_clip_indicators(self, painter: QPainter, w: int, h: int) -> None:
        channels = [("r", THEME.channel_red), ("g", THEME.channel_green), ("b", THEME.channel_blue)]
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
    _X_MIN, _X_MAX = -0.1, 1.1  # plt_x domain
    _Y_MIN, _Y_MAX = -0.05, 1.05  # output domain

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumHeight(40)
        self._curve_pts: list[tuple[float, float]] = []
        # Per-channel (color, points) traces; empty unless Cast Removal diverges the channels.
        self._channel_curves: list[tuple[QColor, list[tuple[float, float]]]] = []
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

    def update_curve(
        self,
        params,
        slope: float | None = None,
        pivot: float | None = None,
        slopes: tuple[float, float, float] | None = None,
        pivots: tuple[float, float, float] | None = None,
        process_mode: str | None = None,
        flat: bool = False,
    ) -> None:
        from negpy.features.exposure.logic import CharacteristicCurve, _expit, compute_pivot, grade_coupled_shape, grade_to_slope
        from negpy.features.exposure.models import EXPOSURE_CONSTANTS
        from negpy.features.exposure.papers import effective_paper_profile
        from negpy.kernel.image.validation import ensure_image

        # process_mode None (e.g. flat-master peek) collapses to the neutral default.
        paper = effective_paper_profile(params.paper_profile, process_mode)
        d_min = paper.d_min if params.paper_dmin else 0.0

        # Slope/pivot come from the render path (session panel); fall back to
        # the same helpers with no metrics when called without them.
        if slope is None:
            slope = grade_to_slope(params.grade, None)
        if pivot is None:
            pivot = compute_pivot(slope, params.density, d_min=d_min, paper=paper)

        flare = EXPOSURE_CONSTANTS["flare_fraction"] if params.flare else 0.0
        surround_gamma = EXPOSURE_CONSTANTS["target_system_gamma"] if params.surround else 1.0

        # Grade-coupled knees — same helper as the render path, so the plotted
        # curve matches the engine at hard grades. Flat has no print knees.
        toe_eff, shoulder_eff = (params.toe, params.shoulder) if flat else grade_coupled_shape(slope, params.toe, params.shoulder)

        n = 300
        plt_x = np.linspace(self._X_MIN, self._X_MAX, n)
        x_log_exp = 1.0 - plt_x

        def _curve_points(s: float, p: float) -> list[tuple[float, float]]:
            if flat:
                # True log master: code value linear in the log signal (1 - val),
                # emitted directly with no 10^-D/sRGB. s=gain, p=lift.
                yv = np.clip(p + s * (1.0 - x_log_exp), 0.0, 1.0)
                return list(zip(plt_x.tolist(), yv.tolist()))
            # d_max/d_min from constants so the chart matches the render exactly.
            curve = CharacteristicCurve(
                contrast=s,
                pivot=p,
                d_min=d_min,
                toe=toe_eff,
                toe_width=params.toe_width,
                shoulder=shoulder_eff,
                shoulder_width=params.shoulder_width,
                flare=flare,
                surround_gamma=surround_gamma,
            )
            d = curve(ensure_image(x_log_exp))
            t = np.power(10.0, -d)
            # Working-space OETF — match the engine output encode.
            yv = np.asarray(working_oetf_encode(t.astype(np.float32))).reshape(-1)
            return list(zip(plt_x.tolist(), yv.tolist()))

        # Base (white) reference curve — also the fill/pivot/zone geometry.
        self._curve_pts = _curve_points(slope, pivot)

        # Per-channel traces only when Cast Removal diverges the channels; else one white curve.
        self._channel_curves = []
        if slopes is not None and pivots is not None:
            diverged = (max(slopes) - min(slopes) > 1e-9) or (max(pivots) - min(pivots) > 1e-9)
            if diverged:
                ch_colors = (QColor(255, 90, 90), QColor(90, 220, 120), QColor(95, 150, 255))
                self._channel_curves = [(ch_colors[ch], _curve_points(slopes[ch], pivots[ch])) for ch in range(3)]

        # Zone shading: toe rolls the shadows (input above the pivot), shoulder
        # rolls the highlights (input below the pivot); smaller width = sharper split.
        epsilon = 1e-6
        self._toe_mask = _expit((x_log_exp - pivot) * (10.0 / max(params.toe_width, epsilon))).tolist()
        self._shoulder_mask = _expit((pivot - x_log_exp) * (10.0 / max(params.shoulder_width, epsilon))).tolist()
        self._toe_strength = toe_eff
        self._shoulder_strength = shoulder_eff

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
            int(self._wx(0.0, w)),
            int(self._wy(0.0, h)),
            int(self._wx(1.0, w)),
            int(self._wy(1.0, h)),
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

        # Curve line on top; per-channel traces replace the white line when present.
        painter.setBrush(Qt.BrushStyle.NoBrush)
        if self._channel_curves:
            for color, pts in self._channel_curves:
                ch_path = QPainterPath()
                ch_path.moveTo(self._wx(pts[0][0], w), self._wy(pts[0][1], h))
                for px, py in pts[1:]:
                    ch_path.lineTo(self._wx(px, w), self._wy(py, h))
                painter.setPen(QPen(color, 1.5))
                painter.drawPath(ch_path)
        else:
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


class MiniHistogramWidget(QWidget):
    """
    20px-tall luminance strip shown behind the Exposure section header.
    Draws only the L channel at ~40% opacity (always linear scale).
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._data_l: list = []
        self._clip_low: bool = False
        self._clip_high: bool = False

    def update_data(self, buffer: Any) -> None:
        if buffer is None or not isinstance(buffer, np.ndarray):
            self._data_l = []
            self._clip_low = False
            self._clip_high = False
            self.update()
            return
        if buffer.shape == (4, 256):
            max_val = float(np.max(buffer[3]))
            self._data_l = (buffer[3].astype(float) / max_val).tolist() if max_val > 0 else []
            total = float(buffer[3].sum())
            if total > 0:
                self._clip_low = float(buffer[3, 0:3].sum()) / total > _CLIP_THRESH
                self._clip_high = float(buffer[3, 253:256].sum()) / total > _CLIP_THRESH
            else:
                self._clip_low = False
                self._clip_high = False
        self.update()

    def paintEvent(self, event) -> None:
        if not self._data_l:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w = self.width()
        h = self.height()

        path = QPainterPath()
        path.moveTo(0, h)
        step = w / (len(self._data_l) - 1)
        for i, val in enumerate(self._data_l):
            path.lineTo(i * step, h - val * h)
        path.lineTo(w, h)
        path.closeSubpath()

        c = QColor(THEME.text_muted)
        c.setAlpha(100)  # ~40% opacity
        painter.setBrush(QBrush(c))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawPath(path)

        # Clipping indicators: 3px vertical strip, full height
        painter.setPen(Qt.PenStyle.NoPen)
        if self._clip_low:
            shadow_color = QColor(80, 140, 220, 180)
            painter.setBrush(QBrush(shadow_color))
            painter.drawRect(0, 0, 3, h)
        if self._clip_high:
            highlight_color = QColor(220, 80, 80, 180)
            painter.setBrush(QBrush(highlight_color))
            painter.drawRect(w - 3, 0, 3, h)


class MiniRGBHistogramWidget(QWidget):
    """
    Per-channel counterpart to MiniHistogramWidget shown behind the Colour section header.
    Overlays the R, G, B channels (~50% opacity) so a colour cast reads as the channels
    pulling apart. Fed the same (4, 256) [R, G, B, L] buffer as the luma mini histogram.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._channels: dict[str, list] = {}

    def update_data(self, buffer: Any) -> None:
        if buffer is None or not isinstance(buffer, np.ndarray) or buffer.shape != (4, 256):
            self._channels = {}
            self.update()
            return
        self._channels = {}
        for idx, key in ((0, "r"), (1, "g"), (2, "b")):
            row = buffer[idx].astype(float)
            max_val = float(row.max())
            self._channels[key] = (row / max_val).tolist() if max_val > 0 else []
        self.update()

    def paintEvent(self, event) -> None:
        if not self._channels:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        w = self.width()
        h = self.height()
        colours = {"r": THEME.channel_red, "g": THEME.channel_green, "b": THEME.channel_blue}
        for key, data in self._channels.items():
            if not data:
                continue
            path = QPainterPath()
            path.moveTo(0, h)
            step = w / (len(data) - 1)
            for i, val in enumerate(data):
                path.lineTo(i * step, h - val * h)
            path.lineTo(w, h)
            path.closeSubpath()
            c = QColor(colours[key])
            c.setAlpha(120)
            painter.setBrush(QBrush(c))
            painter.drawPath(path)
