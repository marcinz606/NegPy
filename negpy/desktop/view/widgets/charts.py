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
from negpy.kernel.image.logic import working_oetf_encode

_CLIP_THRESH = 0.005  # fraction of pixels considered "clipping"


class PhotometricCurveWidget(QWidget):
    """
    The Analysis chart: H&D curve with the negative's density histogram on the
    exposure axis and the print's RGB+L histogram behind the curve, plus pivot,
    zone shading/ticks, clip marks, hover marker + curve dot, LIN/LOG toggle.
    """

    # Data coordinate ranges
    _X_MIN, _X_MAX = -0.1, 1.1  # plt_x domain
    _Y_MIN, _Y_MAX = -0.05, 1.05  # output domain

    # Fraction of the widget height the density histogram may occupy.
    _DENSITY_HIST_FRAC = 0.35

    # Emitted when the user flips the in-widget LIN/LOG toggle (True == log).
    scale_changed = pyqtSignal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumHeight(40)
        self._curve_pts: list[tuple[float, float]] = []
        # Per-channel (color, points) traces; empty unless Cast Removal diverges the channels.
        self._channel_curves: list[tuple[QColor, list[tuple[float, float]]]] = []
        self._pivot_pt: tuple[float, float] | None = None
        # Val-domain bins; DENSITY_HIST_RANGE mirrors _X_MIN/_X_MAX, so bin i
        # maps to plt_x = 1 - val_center(i).
        self._density_bins: np.ndarray | None = None
        self._output_counts: np.ndarray | None = None  # (4, 256) [R, G, B, L]
        self._log_scale: bool = False
        self._clip_low: dict[int, bool] = {}
        self._clip_high: dict[int, bool] = {}
        self._marker: tuple[int, int, int] | None = None
        # Hit rectangles for the LIN/LOG toggle, recomputed each paint.
        self._lin_rect = QRect()
        self._log_rect = QRect()
        self._toe_mask: list[float] = []
        self._shoulder_mask: list[float] = []
        self._toe_strength: float = 0.0
        self._shoulder_strength: float = 0.0
        # Drag feedback: pre-drag curve snapshot + the exposure field being dragged.
        self._active_param: str | None = None
        self._ghost_pts: list[tuple[float, float]] = []
        self._ghost_pivot: tuple[float, float] | None = None
        # Spot-densitometer tracking dot: hovered pixel's val (None = hidden).
        self._tracking_val: float | None = None
        self.setMouseTracking(True)

    def log_scale(self) -> bool:
        return self._log_scale

    def set_log_scale(self, enabled: bool) -> None:
        """Switch both histograms between linear and logarithmic count scaling."""
        enabled = bool(enabled)
        if enabled == self._log_scale:
            return
        self._log_scale = enabled
        self.update()

    def set_marker(self, rgb: tuple | None) -> None:
        """Marks the hovered pixel's R/G/B values (0-255) with vertical lines; None clears."""
        if rgb == self._marker:
            return
        self._marker = rgb
        self.update()

    # ── coordinate helpers ────────────────────────────────────────────────────

    def _wx(self, dx: float, w: int) -> float:
        return (dx - self._X_MIN) / (self._X_MAX - self._X_MIN) * w

    def _wy(self, dy: float, h: int) -> float:
        return h - (dy - self._Y_MIN) / (self._Y_MAX - self._Y_MIN) * h

    # ── data update ──────────────────────────────────────────────────────────

    def set_active_param(self, param: str | None) -> None:
        """Snapshot the current curve as a ghost while `param` drags; empty/None clears."""
        param = param or None
        if param == self._active_param:
            return
        self._active_param = param
        if param is None:
            self._ghost_pts = []
            self._ghost_pivot = None
        else:
            self._ghost_pts = list(self._curve_pts)
            self._ghost_pivot = self._pivot_pt
        self.update()

    def set_tracking_point(self, val: float | None) -> None:
        """Marks the hovered pixel's val on the curve (spot densitometer); None hides."""
        if val == self._tracking_val:
            return
        self._tracking_val = val
        self.update()

    def set_density_histogram(self, bins: Any) -> None:
        """Negative-density occupancy along the curve's exposure axis; None clears."""
        if bins is not None:
            bins = np.asarray(bins, dtype=float)
            if bins.size < 2 or float(bins.max()) <= 0.0:
                bins = None
        if bins is None and self._density_bins is None:
            return
        self._density_bins = bins
        self.update()

    def set_output_histogram(self, bins: Any) -> None:
        """(4, 256) [R, G, B, L] print-tone counts drawn behind the curve; None clears."""
        if bins is not None:
            bins = np.asarray(bins, dtype=float)
            if bins.ndim != 2 or bins.shape[0] != 4:
                bins = None
        if bins is None and self._output_counts is None:
            return
        self._output_counts = bins
        if bins is None:
            self._clip_low = {}
            self._clip_high = {}
        else:
            totals = np.maximum(bins[:3].sum(axis=1), 1.0)
            self._clip_low = {c: bool(bins[c, 0] / totals[c] > _CLIP_THRESH) for c in range(3)}
            self._clip_high = {c: bool(bins[c, -1] / totals[c] > _CLIP_THRESH) for c in range(3)}
        self.update()

    def _hist_display(self, row: int) -> np.ndarray | None:
        """Normalized 0..1 plot values for an output-histogram row under the scale mode."""
        if self._output_counts is None:
            return None
        counts = self._output_counts[row]
        vals = np.log1p(counts) if self._log_scale else counts
        max_val = float(np.max(vals))
        if max_val <= 0:
            return None
        return vals / max_val

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
        from negpy.features.exposure.logic import (
            CharacteristicCurve,
            _expit,
            compute_pivot,
            effective_midtone_gamma,
            grade_coupled_shape,
            grade_to_slope,
            per_channel_midtone_gamma,
            per_channel_toe_shoulder,
            per_channel_widths,
            split_grade_deltas,
        )
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

        # Grade-coupled knees — same helper as the render path, so the plotted
        # curve matches the engine at hard grades. Flat has no print knees.
        toe_eff, shoulder_eff = (params.toe, params.shoulder) if flat else grade_coupled_shape(slope, params.toe, params.shoulder)
        split_sh_trims = (params.shadow_grade_trim_red, params.shadow_grade_trim_green, params.shadow_grade_trim_blue)
        split_hi_trims = (params.highlight_grade_trim_red, params.highlight_grade_trim_green, params.highlight_grade_trim_blue)
        sg3, hg3 = split_grade_deltas(params.grade, params.shadow_grade, params.highlight_grade, split_sh_trims, split_hi_trims)
        # Base (achromatic) trace uses the trim-free global deltas.
        sg_base, hg_base = split_grade_deltas(params.grade, params.shadow_grade, params.highlight_grade)

        n = 300
        plt_x = np.linspace(self._X_MIN, self._X_MAX, n)
        x_log_exp = 1.0 - plt_x

        def _curve_points(
            s: float,
            p: float,
            toe_ch: float | None = None,
            sh_ch: float | None = None,
            mg_ch: float | None = None,
            tw_ch: float | None = None,
            sw_ch: float | None = None,
            sg_ch: float | None = None,
            hg_ch: float | None = None,
        ) -> list[tuple[float, float]]:
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
                toe=toe_eff if toe_ch is None else toe_ch,
                toe_width=params.toe_width if tw_ch is None else tw_ch,
                shoulder=shoulder_eff if sh_ch is None else sh_ch,
                shoulder_width=params.shoulder_width if sw_ch is None else sw_ch,
                midtone_gamma=effective_midtone_gamma(None, params.midtone_gamma) if mg_ch is None else mg_ch,
                bpc=params.true_black,
                shadow_density=params.shadow_density,
                highlight_density=params.highlight_density,
                shadow_grade_delta=sg_base[0] if sg_ch is None else sg_ch,
                highlight_grade_delta=hg_base[0] if hg_ch is None else hg_ch,
            )
            d = curve(ensure_image(x_log_exp))
            t = np.power(10.0, -d)
            # Working-space OETF — match the engine output encode.
            yv = np.asarray(working_oetf_encode(t.astype(np.float32))).reshape(-1)
            return list(zip(plt_x.tolist(), yv.tolist()))

        # Base (white) reference curve — also the fill/pivot/zone geometry.
        self._curve_pts = _curve_points(slope, pivot)

        # Per-channel traces when Cast Removal / grade trims diverge the channels
        # or the knee trims split toe/shoulder; else one white curve.
        self._channel_curves = []
        if slopes is not None and pivots is not None:
            knee_trims = (
                params.toe_trim_red,
                params.toe_trim_green,
                params.toe_trim_blue,
                params.shoulder_trim_red,
                params.shoulder_trim_green,
                params.shoulder_trim_blue,
            )
            snap_trims = (
                params.midtone_gamma_trim_red,
                params.midtone_gamma_trim_green,
                params.midtone_gamma_trim_blue,
            )
            width_trims = (
                params.toe_width_trim_red,
                params.toe_width_trim_green,
                params.toe_width_trim_blue,
                params.shoulder_width_trim_red,
                params.shoulder_width_trim_green,
                params.shoulder_width_trim_blue,
            )
            diverged = (
                (max(slopes) - min(slopes) > 1e-9)
                or (max(pivots) - min(pivots) > 1e-9)
                or any(t != 0.0 for t in knee_trims + snap_trims + width_trims + split_sh_trims + split_hi_trims)
            )
            if diverged:
                toe3, sh3 = per_channel_toe_shoulder(toe_eff, shoulder_eff, knee_trims[:3], knee_trims[3:])
                mg3 = per_channel_midtone_gamma(None, params.midtone_gamma, snap_trims)
                tw3, sw3 = per_channel_widths(params.toe_width, params.shoulder_width, width_trims[:3], width_trims[3:])
                ch_colors = (QColor(255, 90, 90), QColor(90, 220, 120), QColor(95, 150, 255))
                self._channel_curves = [
                    (
                        ch_colors[ch],
                        _curve_points(slopes[ch], pivots[ch], toe3[ch], sh3[ch], mg3[ch], tw3[ch], sw3[ch], sg3[ch], hg3[ch]),
                    )
                    for ch in range(3)
                ]

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

        # Histograms under everything else so the curve/zone tints stay readable.
        self._draw_density_histogram(painter, w, h)
        self._draw_output_histogram(painter, w, h)

        # Build the main curve path (reused for fill and line)
        curve_path = QPainterPath()
        curve_path.moveTo(self._wx(self._curve_pts[0][0], w), self._wy(self._curve_pts[0][1], h))
        for px, py in self._curve_pts[1:]:
            curve_path.lineTo(self._wx(px, w), self._wy(py, h))

        # P4: Toe zone shading (warm amber — right side, dense silver = shadows)
        self._draw_zone_shading(painter, w, h, self._toe_mask, self._toe_strength, QColor(255, 140, 50))

        # P4: Shoulder zone shading (cool blue — left side, thin silver = highlights)
        self._draw_zone_shading(painter, w, h, self._shoulder_mask, self._shoulder_strength, QColor(60, 130, 255))

        # Glow the dragged slider's zone (fixed strength — reads even at zero/negative values)
        if self._active_param in ("toe", "toe_width"):
            self._draw_zone_shading(painter, w, h, self._toe_mask, 0.5, QColor(255, 140, 50))
        elif self._active_param in ("shoulder", "shoulder_width"):
            self._draw_zone_shading(painter, w, h, self._shoulder_mask, 0.5, QColor(60, 130, 255))

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

        # Pre-drag ghost curve
        if self._ghost_pts:
            ghost_path = QPainterPath()
            ghost_path.moveTo(self._wx(self._ghost_pts[0][0], w), self._wy(self._ghost_pts[0][1], h))
            for px, py in self._ghost_pts[1:]:
                ghost_path.lineTo(self._wx(px, w), self._wy(py, h))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(QColor(200, 200, 200, 90), 1, Qt.PenStyle.DashLine))
            painter.drawPath(ghost_path)
            if self._ghost_pivot:
                painter.setBrush(QBrush(QColor(200, 200, 200, 90)))
                painter.setPen(Qt.PenStyle.NoPen)
                painter.drawEllipse(QPointF(self._wx(self._ghost_pivot[0], w), self._wy(self._ghost_pivot[1], h)), 2.5, 2.5)

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

            # Grade/Density act about the pivot — brighten its crosshair while dragged.
            cross_alpha = 110 if self._active_param in ("grade", "density") else 45
            painter.setPen(QPen(QColor(200, 200, 200, cross_alpha), 1, Qt.PenStyle.DotLine))
            painter.drawLine(int(wpx), 0, int(wpx), h)
            painter.drawLine(0, int(wpy), w, int(wpy))

            painter.setBrush(QBrush(QColor("#FFFFFF")))
            painter.setPen(QPen(QColor("#050505"), 1))
            painter.drawEllipse(QPointF(wpx, wpy), 3.5, 3.5)

        # Spot-densitometer tracking dot
        if self._tracking_val is not None:
            plt_x = float(np.clip(1.0 - self._tracking_val, self._X_MIN, self._X_MAX))
            idx = round((plt_x - self._X_MIN) / (self._X_MAX - self._X_MIN) * (len(self._curve_pts) - 1))
            idx = max(0, min(len(self._curve_pts) - 1, idx))
            tx, ty = self._curve_pts[idx]
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(QColor(255, 255, 255, 220), 1.5))
            painter.drawEllipse(QPointF(self._wx(tx, w), self._wy(ty, h)), 4.0, 4.0)

        # Hovered-pixel marker lines
        if self._marker is not None:
            for value, color_hex in zip(self._marker, (THEME.channel_red, THEME.channel_green, THEME.channel_blue)):
                c = QColor(color_hex)
                c.setAlpha(220)
                painter.setPen(QPen(c, 1, Qt.PenStyle.DashLine))
                mx = int(value / 255 * (w - 1))
                painter.drawLine(mx, 0, mx, h)

        self._draw_clip_indicators(painter, w, h)
        self._draw_scale_toggle(painter, w, h)

    def _draw_output_histogram(self, painter: QPainter, w: int, h: int) -> None:
        """Output tones, black left → white right — same direction the curve rises."""
        if self._output_counts is None:
            return
        specs = (
            (3, "#D4D4D4", 26, 120),
            (0, THEME.channel_red, 55, 160),
            (1, THEME.channel_green, 55, 160),
            (2, THEME.channel_blue, 55, 160),
        )
        for row, color_hex, alpha_fill, alpha_line in specs:
            data = self._hist_display(row)
            if data is None:
                continue
            step = w / (data.size - 1)
            values = data.tolist()

            path = QPainterPath()
            path.moveTo(0, h)
            for i, v in enumerate(values):
                path.lineTo(i * step, h - v * h)
            path.lineTo(w, h)
            path.closeSubpath()
            c_fill = QColor(color_hex)
            c_fill.setAlpha(alpha_fill)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(c_fill))
            painter.drawPath(path)

            # Open polyline for the outline so the closing edges don't stroke.
            path_line = QPainterPath()
            path_line.moveTo(0, h - values[0] * h)
            for i, v in enumerate(values):
                path_line.lineTo(i * step, h - v * h)
            c_line = QColor(color_hex)
            c_line.setAlpha(alpha_line)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(c_line, 1.0))
            painter.drawPath(path_line)

    def _draw_clip_indicators(self, painter: QPainter, w: int, h: int) -> None:
        channels = ((0, THEME.channel_red), (1, THEME.channel_green), (2, THEME.channel_blue))
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

    def _draw_density_histogram(self, painter: QPainter, w: int, h: int) -> None:
        bins = self._density_bins
        if bins is None:
            return
        from negpy.features.exposure.analysis import DENSITY_HIST_RANGE

        lo, hi = DENSITY_HIST_RANGE
        n = bins.size
        vals = np.log1p(bins) if self._log_scale else bins
        peak = float(vals.max())
        bot = self._wy(self._Y_MIN, h)
        scale = self._DENSITY_HIST_FRAC * h

        path = QPainterPath()
        xs = [self._wx(1.0 - (lo + (i + 0.5) * (hi - lo) / n), w) for i in range(n)]
        path.moveTo(xs[0], bot)
        for x, count in zip(xs, vals.tolist()):
            path.lineTo(x, bot - count / peak * scale)
        path.lineTo(xs[-1], bot)
        path.closeSubpath()

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor(200, 200, 200, 30)))
        painter.drawPath(path)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(QColor(200, 200, 200, 80), 1))
        painter.drawPath(path)

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


class ZoneStripWidget(QWidget):
    """
    10-cell print-zone occupancy strip (0–IX): cell tone = zone brightness,
    opacity = occupancy (√-scaled so small-but-real mass reads). Extreme cells
    tint red on blocked shadows / blown highlights.
    """

    _LABELS = ("0", "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX")

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(24)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setMouseTracking(True)
        self._occ: np.ndarray | None = None
        self._warn: tuple[bool, bool] = (False, False)

    def update_data(self, occ: np.ndarray | None, warnings: tuple[bool, bool] = (False, False)) -> None:
        self._occ = occ
        self._warn = warnings
        self.update()

    def mouseMoveEvent(self, event) -> None:
        if self._occ is not None and self.width() > 0:
            cell = int(min(max(event.position().x() / self.width() * len(self._LABELS), 0), len(self._LABELS) - 1))
            self.setToolTip(f"Zone {self._LABELS[cell]} — {float(self._occ[cell]) * 100:.1f}%")
        super().mouseMoveEvent(event)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        w = self.width()
        h = self.height()
        rect = self.rect().adjusted(0, 0, -1, -1)
        painter.fillRect(rect, QColor("#050505"))

        if self._occ is not None:
            n = len(self._LABELS)
            font = QFont()
            font.setPixelSize(8)
            painter.setFont(font)
            shadow_warn, highlight_warn = self._warn
            for i in range(n):
                x0 = int(w * i / n)
                x1 = int(w * (i + 1) / n)
                frac = float(self._occ[i])
                tone = 40 + int(215 * i / (n - 1))
                alpha = int(min(1.0, np.sqrt(frac)) * 235)
                painter.fillRect(x0, 0, x1 - x0, h, QColor(tone, tone, tone, alpha))
                if (shadow_warn and i <= 1) or (highlight_warn and i == n - 1):
                    painter.fillRect(x0, 0, x1 - x0, h, QColor(220, 80, 80, 90))
                painter.setPen(QPen(QColor(130, 130, 130, 160)))
                painter.drawText(QRect(x0, 0, x1 - x0, h), Qt.AlignmentFlag.AlignCenter, self._LABELS[i])
                if i:
                    painter.setPen(QPen(QColor("#1A1A1A"), 1))
                    painter.drawLine(x0, 0, x0, h)

        painter.setPen(QPen(QColor("#262626"), 1))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(rect)


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
