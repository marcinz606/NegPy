from negpy.kernel.system.i18n import tr
from typing import Any

import negpy_core
import numpy as np
from PyQt6.QtCore import QPointF, QRect, QRectF, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QImage,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPen,
)
from PyQt6.QtWidgets import QApplication, QSizePolicy, QVBoxLayout, QWidget

from negpy.desktop.view.styles.color_vision import palette_for
from negpy.desktop.view.styles.templates import with_alpha, wrap_tooltip
from negpy.desktop.view.styles.theme import THEME


_CLIP_THRESH = 0.005  # fraction of pixels considered "clipping"


def _draw_plated_text(painter: QPainter, rect: QRect, flags: Qt.AlignmentFlag, text: str) -> None:
    """A chart label on a plate of the chart's own surface, so it reads over a bright trace."""
    box = painter.boundingRect(rect, flags, text).adjusted(-2, 0, 2, 0)
    painter.fillRect(box, with_alpha(THEME.bg_input, 210))
    painter.setPen(with_alpha(THEME.text_hint, 230))
    painter.drawText(rect, flags, text)


def _guide_pen() -> QPen:
    """Every chart's grid and guide lines."""
    return QPen(with_alpha(THEME.border_hover, 150), 1)


def _draw_hover_ring(painter: QPainter, at: QPointF) -> None:
    """The hovered pixel's mark on a scope: the curve's tracking ring over a dark halo, so it holds over a bright trace."""
    painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.setPen(QPen(with_alpha(THEME.canvas_bg_black, 200), 3.5))
    painter.drawEllipse(at, 4.0, 4.0)
    painter.setPen(QPen(with_alpha(THEME.text_on_accent, 220), 1.5))
    painter.drawEllipse(at, 4.0, 4.0)


class PhotometricCurveWidget(QWidget):
    """The Analysis chart: H&D curve over a bottom band that holds the negative's density
    histogram and the print's RGB+L histogram."""

    # Data coordinate ranges
    _X_MIN, _X_MAX = -0.1, 1.1  # plt_x domain
    _Y_MIN, _Y_MAX = -0.05, 1.05  # output domain

    # Fraction of the widget height the bottom histogram band may occupy.
    _DENSITY_HIST_FRAC = 0.35

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumHeight(40)
        self._curve_pts: list[tuple[float, float]] = []
        # Per-channel (color, points) traces; empty unless Cast Removal diverges the channels.
        self._channel_curves: list[tuple[QColor, list[tuple[float, float]]]] = []
        self._pivot_pt: tuple[float, float] | None = None
        # Val-domain bins. DENSITY_HIST_RANGE mirrors _X_MIN/_X_MAX, so bin i maps to
        # plt_x = 1 - val_center(i).
        self._density_bins: np.ndarray | None = None
        self._output_counts: np.ndarray | None = None  # (4, 256) [R, G, B, L]
        # Per-channel density traces only while Peek Negative is up: the print's own output
        # histogram already carries color information the rest of the time.
        self._channel_density: bool = False
        # False during Peek Negative: no curve was applied, so the curve, print histogram
        # and zone traces would describe a print that was never made.
        self._show_print: bool = True
        self._log_scale: bool = False
        self._clip_low: dict[int, bool] = {}
        self._clip_high: dict[int, bool] = {}
        self._marker: tuple[int, int, int] | None = None
        # Toe and shoulder tints: Color vision's warm/cool pair.
        self._zone_colors: tuple[str, str] = (THEME.dodge, THEME.burn)
        self._toe_mask: list[float] = []
        self._shoulder_mask: list[float] = []
        self._toe_strength: float = 0.0
        self._shoulder_strength: float = 0.0
        # Contrast Mask band: the curve on the mask's remapped input. Every pixel prints
        # between this and the base curve, by how much of its own value survives the blur.
        self._mask_pts: list[tuple[float, float]] = []
        # Drag feedback: pre-drag curve snapshot + the exposure field being dragged.
        self._active_param: str | None = None
        self._ghost_pts: list[tuple[float, float]] = []
        self._ghost_pivot: tuple[float, float] | None = None
        # Spot-densitometer tracking dot: hovered pixel's val (None = hidden).
        self._tracking_val: float | None = None

    def log_scale(self) -> bool:
        return self._log_scale

    def set_log_scale(self, enabled: bool) -> None:
        """Switch both histograms between linear and logarithmic count scaling."""
        enabled = bool(enabled)
        if enabled == self._log_scale:
            return
        self._log_scale = enabled
        self.update()

    def set_channel_density(self, enabled: bool) -> None:
        """Draw the density histogram as separate R/G/B/L traces instead of one merged
        luma trace; True only while Peek Negative is up."""
        enabled = bool(enabled)
        if enabled == self._channel_density:
            return
        self._channel_density = enabled
        self.update()

    def set_show_print(self, visible: bool) -> None:
        """Hide the curve, print histogram, zone traces and clip marks; density stays."""
        visible = bool(visible)
        if visible == self._show_print:
            return
        self._show_print = visible
        self.update()

    def set_zone_colors(self, warm: str, cool: str) -> None:
        self._zone_colors = (warm, cool)
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
        """(4, DENSITY_HIST_BINS) [R, G, B, L] negative-density occupancy along the curve's
        exposure axis; None clears."""
        if bins is not None:
            bins = np.asarray(bins, dtype=float)
            if bins.ndim != 2 or bins.shape[0] != 4 or bins.shape[1] < 2 or float(bins.max()) <= 0.0:
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
        curvatures: tuple[float, float, float] | None = None,
        process_mode: str | None = None,
        flat: bool = False,
        mask: tuple[float, float, tuple[float, float, float]] | None = None,
        highlight_density: float | None = None,
        shadow_density: float | None = None,
    ) -> None:
        from negpy.features.exposure.logic import (
            _expit,
            compute_pivot,
            grade_coupled_shape,
            grade_to_slope,
            per_channel_midtone_gamma,
            per_channel_toe_shoulder,
            per_channel_widths,
            print_curve,
            print_curve_output,
            split_grade_deltas,
        )
        from negpy.features.exposure.papers import effective_paper_profile

        # process_mode None (e.g. flat-master peek) collapses to the neutral default.
        paper = effective_paper_profile(params.paper_profile, process_mode)
        d_min = paper.d_min if params.paper_dmin else 0.0

        # Slope and pivot come from the render path (session panel). Fall back to the same
        # helpers with no metrics when called without them.
        if slope is None:
            slope = grade_to_slope(params.grade, None, paper.slope_gain)
        if pivot is None:
            pivot = compute_pivot(slope, params.density, d_min=d_min, paper=paper)

        # Grade-coupled knees, from the same helper as the render path, so the plotted curve
        # matches the engine at hard grades. Flat has no print knees.
        toe_eff, shoulder_eff = (params.toe, params.shoulder) if flat else grade_coupled_shape(slope, params.toe, params.shoulder)
        split_sh_trims = (params.shadow_grade_trim_red, params.shadow_grade_trim_green, params.shadow_grade_trim_blue)
        split_hi_trims = (params.highlight_grade_trim_red, params.highlight_grade_trim_green, params.highlight_grade_trim_blue)
        sg3, hg3 = split_grade_deltas(params.grade, params.shadow_grade, params.highlight_grade, split_sh_trims, split_hi_trims)

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
            curv_ch: float = 0.0,
            x_in: np.ndarray | None = None,
        ) -> list[tuple[float, float]]:
            xv = x_log_exp if x_in is None else x_in
            if flat:
                # True log master: the code value is linear in the log signal (1 - val) and emitted
                # directly, with no 10^-D and no sRGB. s = gain, p = lift.
                yv = np.clip(p + s * (1.0 - xv), 0.0, 1.0)
                return list(zip(plt_x.tolist(), yv.tolist()))
            curve = print_curve(
                params,
                s,
                p,
                process_mode,
                toe=toe_ch,
                shoulder=sh_ch,
                toe_width=tw_ch,
                shoulder_width=sw_ch,
                midtone_gamma=mg_ch,
                highlight_density=highlight_density,
                shadow_density=shadow_density,
                shadow_grade_delta=sg_ch,
                highlight_grade_delta=hg_ch,
                curvature=curv_ch,
            )
            return list(zip(plt_x.tolist(), print_curve_output(curve, xv).tolist()))

        # Base (white) reference curve, and also the fill, pivot and zone geometry.
        self._curve_pts = _curve_points(slope, pivot)

        # Contrast Mask: (centre, density range, filtration offsets) of its plane. The mask has no single
        # curve; a large flat area prints on the mask's curve, so the band spans the base curve and that one.
        self._mask_pts = []
        if mask is not None and not flat:
            from negpy.features.exposure.logic import contrast_mask_coeffs

            centre, density_range, offsets = mask
            coeffs = contrast_mask_coeffs(
                params.contrast_mask_strength,
                centre,
                slopes or (slope, slope, slope),
                pivots or (pivot, pivot, pivot),
                offsets,
                density_range,
            )
            if coeffs is not None:
                stops = negpy_core.contrast_mask_stops(np.ascontiguousarray(x_log_exp - centre, dtype=np.float32), coeffs)
                self._mask_pts = _curve_points(slope, pivot, x_in=x_log_exp + stops * np.log10(2.0) / density_range)

        # Per-channel traces when Cast Removal or grade trims diverge the channels, or the knee
        # trims split toe and shoulder. Otherwise one white curve.
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
            curvs = (0.0, 0.0, 0.0) if curvatures is None else curvatures
            diverged = (
                (max(slopes) - min(slopes) > 1e-9)
                or (max(pivots) - min(pivots) > 1e-9)
                or (max(curvs) - min(curvs) > 1e-9)
                or any(t != 0.0 for t in knee_trims + snap_trims + width_trims + split_sh_trims + split_hi_trims)
            )
            if diverged:
                toe3, sh3 = per_channel_toe_shoulder(toe_eff, shoulder_eff, knee_trims[:3], knee_trims[3:])
                from negpy.features.exposure.papers import effective_paper_profile

                mg3 = per_channel_midtone_gamma(
                    effective_paper_profile(params.paper_profile, process_mode), params.midtone_gamma, snap_trims
                )
                tw3, sw3 = per_channel_widths(params.toe_width, params.shoulder_width, width_trims[:3], width_trims[3:])
                ch_colors = (QColor(THEME.channel_red_text), QColor(THEME.channel_green_text), QColor(THEME.channel_blue_text))
                self._channel_curves = [
                    (
                        ch_colors[ch],
                        _curve_points(slopes[ch], pivots[ch], toe3[ch], sh3[ch], mg3[ch], tw3[ch], sw3[ch], sg3[ch], hg3[ch], curvs[ch]),
                    )
                    for ch in range(3)
                ]

        # Zone shading: the toe rolls the shadows (input above the pivot) and the shoulder rolls
        # the highlights (input below it). A smaller width is a sharper split.
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

    def curves(self) -> tuple[list[tuple[float, float]], list[list[tuple[float, float]]]]:
        """The plotted base curve and the R, G, B traces (empty unless they diverge)."""
        return self._curve_pts, [pts for _color, pts in self._channel_curves]

    # ── painting ─────────────────────────────────────────────────────────────

    def paintEvent(self, event) -> None:
        if not self._curve_pts:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w = self.width()
        h = self.height()

        painter.fillRect(self.rect(), QColor(THEME.bg_input))

        # Grid at 0.25 intervals, including 0 and 1, so the axis padding reads as margin and a
        # curve flat at zero visibly lands on the baseline.
        painter.setPen(_guide_pen())
        for i in range(0, 5):
            gx = int(self._wx(i * 0.25, w))
            gy = int(self._wy(i * 0.25, h))
            painter.drawLine(gx, 0, gx, h)
            painter.drawLine(0, gy, w, gy)

        # Diagonal reference (dashed)
        diagonal_pen = _guide_pen()
        diagonal_pen.setStyle(Qt.PenStyle.DashLine)
        painter.setPen(diagonal_pen)
        painter.drawLine(
            int(self._wx(0.0, w)),
            int(self._wy(0.0, h)),
            int(self._wx(1.0, w)),
            int(self._wy(1.0, h)),
        )

        # Histograms under everything else so the curve/zone tints stay readable.
        self._draw_density_histogram(painter, w, h)

        if self._show_print:
            self._draw_output_histogram(painter, w, h)

            # Build the main curve path (reused for fill and line)
            curve_path = QPainterPath()
            curve_path.moveTo(self._wx(self._curve_pts[0][0], w), self._wy(self._curve_pts[0][1], h))
            for px, py in self._curve_pts[1:]:
                curve_path.lineTo(self._wx(px, w), self._wy(py, h))

            # Toe tint is warm (dense silver = shadows), shoulder tint cool (thin silver = highlights).
            warm, cool = QColor(self._zone_colors[0]), QColor(self._zone_colors[1])
            self._draw_zone_shading(painter, w, h, self._toe_mask, self._toe_strength, warm)
            self._draw_zone_shading(painter, w, h, self._shoulder_mask, self._shoulder_strength, cool)

            # Glow the dragged slider's zone at fixed strength, so it reads at zero and negative values
            if self._active_param in ("toe", "toe_width"):
                self._draw_zone_shading(painter, w, h, self._toe_mask, 0.5, warm)
            elif self._active_param in ("shoulder", "shoulder_width"):
                self._draw_zone_shading(painter, w, h, self._shoulder_mask, 0.5, cool)

            # P2: Gradient luminance fill under the curve
            fill_path = QPainterPath(curve_path)
            bot = self._wy(self._Y_MIN, h)
            fill_path.lineTo(self._wx(self._curve_pts[-1][0], w), bot)
            fill_path.lineTo(self._wx(self._curve_pts[0][0], w), bot)
            fill_path.closeSubpath()

            gradient = QLinearGradient(0.0, 0.0, float(w), 0.0)
            gradient.setColorAt(0.0, with_alpha(THEME.canvas_bg_black, 55))
            gradient.setColorAt(1.0, with_alpha(THEME.text_on_accent, 55))
            painter.setBrush(QBrush(gradient))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawPath(fill_path)

            self._draw_zone_ticks(painter, h)

            self._draw_mask_band(painter, w, h)

            # Pre-drag ghost curve
            if self._ghost_pts:
                ghost_path = QPainterPath()
                ghost_path.moveTo(self._wx(self._ghost_pts[0][0], w), self._wy(self._ghost_pts[0][1], h))
                for px, py in self._ghost_pts[1:]:
                    ghost_path.lineTo(self._wx(px, w), self._wy(py, h))
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.setPen(QPen(with_alpha(THEME.text_secondary, 90), 1, Qt.PenStyle.DashLine))
                painter.drawPath(ghost_path)
                if self._ghost_pivot:
                    painter.setBrush(QBrush(with_alpha(THEME.text_secondary, 90)))
                    painter.setPen(Qt.PenStyle.NoPen)
                    painter.drawEllipse(QPointF(self._wx(self._ghost_pivot[0], w), self._wy(self._ghost_pivot[1], h)), 2.5, 2.5)

            # Curve line on top; per-channel traces replace the white line when present.
            painter.setBrush(Qt.BrushStyle.NoBrush)
            if self._channel_curves:
                # Added, as on the Parade Overlay: where the channels agree the line is white.
                painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Plus)
                for color, pts in self._channel_curves:
                    ch_path = QPainterPath()
                    ch_path.moveTo(self._wx(pts[0][0], w), self._wy(pts[0][1], h))
                    for px, py in pts[1:]:
                        ch_path.lineTo(self._wx(px, w), self._wy(py, h))
                    painter.setPen(QPen(color, 1.5))
                    painter.drawPath(ch_path)
                painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
            else:
                painter.setPen(QPen(QColor(THEME.text_on_accent), 1.5))
                painter.drawPath(curve_path)

            # P3: Pivot crosshairs + dot
            if self._pivot_pt:
                wpx = self._wx(self._pivot_pt[0], w)
                wpy = self._wy(self._pivot_pt[1], h)

                # Grade and Density act about the pivot, so brighten its crosshair while dragged.
                cross_alpha = 110 if self._active_param in ("grade", "density") else 45
                painter.setPen(QPen(with_alpha(THEME.text_secondary, cross_alpha), 1, Qt.PenStyle.DotLine))
                painter.drawLine(int(wpx), 0, int(wpx), h)
                painter.drawLine(0, int(wpy), w, int(wpy))

                painter.setBrush(QBrush(QColor(THEME.text_on_accent)))
                painter.setPen(QPen(QColor(THEME.canvas_bg_black), 1))
                painter.drawEllipse(QPointF(wpx, wpy), 3.5, 3.5)

            # Spot-densitometer tracking dot
            if self._tracking_val is not None:
                plt_x = float(np.clip(1.0 - self._tracking_val, self._X_MIN, self._X_MAX))
                idx = round((plt_x - self._X_MIN) / (self._X_MAX - self._X_MIN) * (len(self._curve_pts) - 1))
                idx = max(0, min(len(self._curve_pts) - 1, idx))
                tx, ty = self._curve_pts[idx]
                _draw_hover_ring(painter, QPointF(self._wx(tx, w), self._wy(ty, h)))

            # Hovered-pixel marker lines, on the print histogram's tone scale
            if self._marker is not None:
                for value, color_hex in zip(self._marker, (THEME.channel_red_text, THEME.channel_green_text, THEME.channel_blue_text)):
                    painter.setPen(QPen(with_alpha(color_hex, 220), 1, Qt.PenStyle.DashLine))
                    mx = int(self._wx(value / 255, w))
                    painter.drawLine(mx, 0, mx, h)

            self._draw_clip_indicators(painter, w, h)

    def _draw_mask_band(self, painter: QPainter, w: int, h: int) -> None:
        """The Contrast Mask's reach: filled between the base curve and the curve a flat
        area prints on, with that far edge dashed because no single tone has to land there.
        """
        if not self._mask_pts or not self._curve_pts:
            return
        band = QPainterPath()
        band.moveTo(self._wx(self._curve_pts[0][0], w), self._wy(self._curve_pts[0][1], h))
        for px, py in self._curve_pts[1:]:
            band.lineTo(self._wx(px, w), self._wy(py, h))
        for px, py in reversed(self._mask_pts):
            band.lineTo(self._wx(px, w), self._wy(py, h))
        band.closeSubpath()

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(with_alpha(THEME.chart_mask, 38)))
        painter.drawPath(band)

        edge = QPainterPath()
        edge.moveTo(self._wx(self._mask_pts[0][0], w), self._wy(self._mask_pts[0][1], h))
        for px, py in self._mask_pts[1:]:
            edge.lineTo(self._wx(px, w), self._wy(py, h))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(with_alpha(THEME.chart_mask, 165), 1, Qt.PenStyle.DashLine))
        painter.drawPath(edge)

    def _draw_output_histogram(self, painter: QPainter, w: int, h: int) -> None:
        """Print tones, black left → white right across the plot's 0..1, in the density
        histogram's bottom band: luminance filled gray, R, G and B as lines."""
        if self._output_counts is None:
            return
        specs = (
            (3, THEME.text_secondary, 34, 130),
            (0, THEME.channel_red, 0, 170),
            (1, THEME.channel_green, 0, 170),
            (2, THEME.channel_blue, 0, 170),
        )
        bot = self._wy(self._Y_MIN, h)
        scale = self._DENSITY_HIST_FRAC * h
        for row, color_hex, alpha_fill, alpha_line in specs:
            data = self._hist_display(row)
            if data is None:
                continue
            n = data.size
            xs = [self._wx((i + 0.5) / n, w) for i in range(n)]
            values = data.tolist()

            if alpha_fill:
                path = QPainterPath()
                path.moveTo(xs[0], bot)
                for x, v in zip(xs, values):
                    path.lineTo(x, bot - v * scale)
                path.lineTo(xs[-1], bot)
                path.closeSubpath()
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QBrush(with_alpha(color_hex, alpha_fill)))
                painter.drawPath(path)

            # Open polyline for the outline so the closing edges don't stroke.
            path_line = QPainterPath()
            path_line.moveTo(xs[0], bot - values[0] * scale)
            for x, v in zip(xs, values):
                path_line.lineTo(x, bot - v * scale)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(with_alpha(color_hex, alpha_line), 1.0))
            painter.drawPath(path_line)

    def _draw_zone_ticks(self, painter: QPainter, h: int) -> None:
        """Zones I–IX on the print axis, on the ruler the zone strip and probe read; V is longer."""
        from negpy.features.exposure.analysis import encoded_of_zone

        painter.setPen(_guide_pen())
        for zone in range(1, 10):
            zy = int(self._wy(encoded_of_zone(zone), h))
            painter.drawLine(0, zy, 8 if zone == 5 else 4, zy)

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

    def _draw_density_histogram(self, painter: QPainter, w: int, h: int) -> None:
        """Luma-only merged trace normally; while Peek Negative is up (`_channel_density`),
        R, G, B and luminance traces instead, each normalized to its own peak like the
        output histogram, so a channel that clips (edge-bin spike) or sits off from the
        others (a dominant mask color) reads at a glance."""
        bins = self._density_bins
        if bins is None:
            return
        from negpy.features.exposure.analysis import DENSITY_HIST_RANGE

        lo, hi = DENSITY_HIST_RANGE
        n = bins.shape[-1]
        bot = self._wy(self._Y_MIN, h)
        scale = self._DENSITY_HIST_FRAC * h
        xs = [self._wx(1.0 - (lo + (i + 0.5) * (hi - lo) / n), w) for i in range(n)]

        if self._channel_density:
            specs = (
                (3, THEME.text_secondary, 26, 120),
                (0, THEME.channel_red, 55, 160),
                (1, THEME.channel_green, 55, 160),
                (2, THEME.channel_blue, 55, 160),
            )
        else:
            specs = ((3, THEME.text_secondary, 26, 120),)
        for row, color_hex, alpha_fill, alpha_line in specs:
            counts = np.log1p(bins[row]) if self._log_scale else bins[row]
            peak = float(counts.max())
            if peak <= 0:
                continue
            vals = (counts / peak).tolist()

            path = QPainterPath()
            path.moveTo(xs[0], bot)
            for x, count in zip(xs, vals):
                path.lineTo(x, bot - count * scale)
            path.lineTo(xs[-1], bot)
            path.closeSubpath()

            c_fill = QColor(color_hex)
            c_fill.setAlpha(alpha_fill)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(c_fill))
            painter.drawPath(path)

            path_line = QPainterPath()
            path_line.moveTo(xs[0], bot - vals[0] * scale)
            for x, count in zip(xs, vals):
                path_line.lineTo(x, bot - count * scale)
            c_line = QColor(color_hex)
            c_line.setAlpha(alpha_line)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(c_line, 1.0))
            painter.drawPath(path_line)

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


class WaveformWidget(QWidget):
    """Print luma across the frame: x is the image column, y the print tone, brightness the pixel
    count, each cell in its pixels' mean color. Columns that reach paper black or paper white are
    marked on the bottom and top edges. With parade=True it draws R, G and B side by side, each
    channel's own tone across the frame; in the scan domain, each channel's scan density, dense at the
    bottom, with solid marks at the density range the stretch maps onto the curve. The parade's overlay
    draws the three on one plot, added: where they coincide the trace is white. The orange mask keeps
    a scan's channels apart, so the overlay and the cross-panel line need the base-relative view there."""

    _GUIDE_ZONES = ((3, "III"), (5, "V"), (7, "VII"))
    _PARADE_GAP = 3
    _SCAN_GUIDES = (1.0, 2.0)

    def __init__(self, parent=None, parade: bool = False):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumHeight(40)
        self._parade = parade
        self._counts: np.ndarray | None = None
        self._color: np.ndarray | None = None
        self._log_scale = False
        self._images: list[QImage] | None = None
        self._marker: tuple[float, ...] | None = None
        self._scan = False
        self._scan_marks: np.ndarray | None = None
        self._scan_caption = ""
        self._full_scale: np.ndarray | None = None
        self._scan_base: np.ndarray | None = None
        self._base_relative = False
        self._overlay = False
        self._clip_colors: tuple[str, str] = palette_for("standard").pair

    def set_clip_colors(self, shadow: str, highlight: str) -> None:
        """The Clipping Overlay's pair, for the bottom and top edge marks."""
        if (shadow, highlight) != self._clip_colors:
            self._clip_colors = (shadow, highlight)
            self.update()

    def set_overlay(self, enabled: bool) -> None:
        """Draw a parade's R, G and B added on one plot instead of side by side."""
        if bool(enabled) != self._overlay:
            self._overlay = bool(enabled)
            self._images = None
            self.update()

    def _aligned(self) -> bool:
        """Whether one height means the same in every channel: always on the print, and on the scan
        only with each channel's base at the top."""
        return not self._scan or self._base_shifts() is not None

    def _split(self) -> bool:
        return self._parade and not (self._overlay and self._aligned())

    def set_scan_domain(
        self, enabled: bool, marks: Any = None, caption: str = "Scan density", full_scale: Any = None, base: Any = None
    ) -> None:
        """Read the counts as scan density (Peek Negative) rather than print tone. ``marks`` is the
        (3, 2) per-channel stretch ends and ``base`` the (3,) clear film base, 0..1 up the panel;
        ``full_scale`` the (3, cols) counts of scan pixels at sensor clip, which set the top ticks.
        A domain change drops the marker."""
        enabled = bool(enabled)
        marks = None if marks is None or not enabled else np.asarray(marks, dtype=float)
        base = None if base is None or not enabled else np.asarray(base, dtype=float).reshape(3)
        if enabled != self._scan:
            self._marker = None
            self._images = None
        old_shifts = self._base_shifts()
        self._scan = enabled
        self._scan_marks = marks
        self._scan_base = base
        new_shifts = self._base_shifts()
        if (old_shifts is None) != (new_shifts is None) or (new_shifts is not None and not np.array_equal(old_shifts, new_shifts)):
            self._images = None
        self._scan_caption = caption if enabled else ""
        self._full_scale = None if full_scale is None or not enabled else np.asarray(full_scale, dtype=float)
        self.update()

    def set_base_relative(self, enabled: bool) -> None:
        """Shift each scan channel so its clear film base, else its thin stretch mark, sits at the top."""
        if bool(enabled) != self._base_relative:
            self._base_relative = bool(enabled)
            self._images = None
            self.update()

    def _base_shifts(self) -> np.ndarray | None:
        """Per-channel shift up the panel, in bins, that puts each film base (else thin mark) at the top."""
        if not (self._scan and self._base_relative):
            return None
        if self._scan_base is not None:
            zero = self._scan_base
        elif self._scan_marks is not None:
            zero = self._scan_marks.max(axis=1)
        else:
            return None
        from negpy.features.exposure.analysis import WAVEFORM_BINS

        # The zero's own bin lands in the top bin.
        return (WAVEFORM_BINS - 1 - np.minimum(np.floor(np.clip(zero, 0.0, 1.0) * WAVEFORM_BINS), WAVEFORM_BINS - 1)).astype(int)

    def _shown_panels(self) -> np.ndarray:
        """The counts as drawn: each scan channel moved up by its base shift, with mass pushed past
        an edge kept in that edge bin."""
        panels = self._panels()
        shifts = self._base_shifts()
        if shifts is None:
            return panels
        out = np.zeros_like(panels)
        bins = panels.shape[1]
        for c, k in enumerate(shifts.tolist()):
            k = max(-bins + 1, min(bins - 1, k))
            if k >= 0:
                out[c, k:] = panels[c, : bins - k]
                out[c, -1] += panels[c, bins - k :].sum(axis=0)
            else:
                out[c, : bins + k] = panels[c, -k:]
                out[c, 0] += panels[c, :-k].sum(axis=0)
        return out

    def set_log_scale(self, enabled: bool) -> None:
        if bool(enabled) != self._log_scale:
            self._log_scale = bool(enabled)
            self._images = None
            self.update()

    def set_marker(self, point: tuple[float, ...] | None) -> None:
        """The hovered pixel as (column 0..1 across the frame, print luma 0..1), or (column, R, G, B)
        on a parade, each 0..1 up the panel; None hides."""
        if point != self._marker:
            self._marker = point
            self.update()

    def set_data(self, counts: Any, color: Any = None) -> None:
        """(bins, cols) counts, bin 0 = black, and optional (3, bins, cols) R, G, B sums of
        each cell; on a parade, (3, bins, cols) channel counts. None counts clears."""
        if counts is not None:
            counts = np.asarray(counts, dtype=float)
            if counts.ndim != (3 if self._parade else 2) or float(counts.max()) <= 0.0:
                counts = None
        if counts is not None and self._parade:
            color = None
        if color is not None:
            color = np.asarray(color, dtype=float)
            if counts is None or color.shape != (3, *counts.shape):
                color = None
        if counts is None and self._counts is None:
            return
        self._counts = counts
        self._color = color
        self._images = None
        self.update()

    def _panels(self) -> np.ndarray:
        """(panels, bins, cols) counts: one panel, or R, G, B on a parade."""
        assert self._counts is not None
        return self._counts if self._parade else self._counts[None]

    def _trace_images(self) -> list[QImage] | None:
        if self._images is None and self._counts is not None:
            vals = np.log1p(self._shown_panels()) if self._log_scale else self._shown_panels()
            # A flat area piles into a few bins; scaling by their peak would leave every other trace black.
            ref = float(np.percentile(vals[vals > 0], 99))
            levels = np.clip(vals / ref, 0.0, 1.0)
            if self._parade:
                hexes = (THEME.channel_red_text, THEME.channel_green_text, THEME.channel_blue_text)
                tints = [np.broadcast_to(np.array(QColor(c).getRgbF()[:3]), (*levels.shape[1:], 3)) for c in hexes]
            elif self._color is None:
                tints = [np.ones((*levels.shape[1:], 3))]
            else:
                # Each cell's mean color at full brightness; the count sets how bright it draws.
                mean = np.moveaxis(self._color, 0, -1)
                peak = mean.max(axis=-1, keepdims=True)
                tints = [np.divide(mean, peak, out=np.ones_like(mean), where=peak > 0)]
            if self._parade and not self._split():
                # Pure primaries, not the channel tokens: only these add up to white where R, G and B coincide.
                levels = levels.max(axis=0, keepdims=True)
                tints = [np.moveaxis(np.clip(vals / ref, 0.0, 1.0), 0, -1) / np.maximum(levels[0][..., None], 1e-9)]
            self._images = []
            for level, tint in zip(levels, tints):
                # Premultiplied, as on the vectorscope: an empty cell shows the chart background.
                rgba = np.concatenate([tint * level[..., None], level[..., None]], axis=-1)
                rgba = np.ascontiguousarray(np.flipud(rgba * 255.0).astype(np.uint8))
                h, w = level.shape
                self._images.append(QImage(rgba.data, w, h, w * 4, QImage.Format.Format_RGBA8888_Premultiplied).copy())
        return self._images

    def _shown_marks(self, tones: np.ndarray) -> np.ndarray:
        """Per-channel axis positions (first axis R, G, B) moved by the base shift, as drawn."""
        shifts = self._base_shifts()
        if shifts is None:
            return tones
        from negpy.features.exposure.analysis import WAVEFORM_BINS

        offset = shifts / WAVEFORM_BINS
        return tones + (offset[:, None] if tones.ndim == 2 else offset)

    def _scan_ranges(self) -> list[float] | None:
        """Each channel's density range between its stretch marks: the film's own range, which a
        scan's exposure moves up or down the axis but does not change."""
        if not self._scan or self._scan_marks is None or self._counts is None:
            return None
        from negpy.features.exposure.analysis import SCAN_DENSITY_RANGE

        lo, hi = SCAN_DENSITY_RANGE
        return [abs(float(top) - float(bottom)) * (hi - lo) for bottom, top in self._scan_marks]

    def _panel_rects(self, n: int) -> list[QRect]:
        w, h = self.width(), self.height()
        gap = self._PARADE_GAP if n > 1 else 0
        pw = (w - gap * (n - 1)) / n
        return [QRect(int(i * (pw + gap)), 0, int((i + 1) * pw + i * gap) - int(i * (pw + gap)), h) for i in range(n)]

    def paintEvent(self, event) -> None:
        from negpy.features.exposure.analysis import encoded_of_zone, scan_density_axis

        painter = QPainter(self)
        w, h = self.width(), self.height()
        painter.fillRect(self.rect(), QColor(THEME.bg_input))

        images = self._trace_images()
        counts = self._counts
        rects = self._panel_rects(3 if self._split() else 1)
        # The rect each channel's marks and rings sit in: its own panel, or the one shared plot.
        channel_rects = rects if self._split() else rects * 3
        ticks: list[tuple[int, int, int]] = []
        if images is not None and counts is not None:
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
            # Ticks read the unshifted counts: past the axis at the bottom, at full scale (or paper white) at the top.
            raw = self._panels()
            bottom, top = raw[:, 0] > 0, raw[:, -1] > 0
            if self._scan and self._full_scale is not None and self._full_scale.shape == top.shape:
                top = self._full_scale > 0
            if not self._split() and self._parade:
                bottom, top = bottom.any(axis=0, keepdims=True), top.any(axis=0, keepdims=True)
            for image, rect, low, high in zip(images, rects, bottom, top):
                painter.drawImage(rect, image)
                cols = low.shape[0]
                for marks, y in ((low, h - 3), (high, 2)):
                    for c in np.flatnonzero(marks).tolist():
                        ticks.append((rect.x() + int(c * rect.width() / cols), rect.x() + int((c + 1) * rect.width() / cols), y))
            if self._split():
                painter.setPen(QPen(QColor(THEME.border_primary), 1))
                for rect in rects[1:]:
                    painter.drawLine(rect.x() - 2, 0, rect.x() - 2, h)

        font = QFont(self.font())
        font.setPixelSize(THEME.font_size_micro)
        painter.setFont(font)
        if self._scan:
            guides = [(float(scan_density_axis(d)), f"{d:.1f}") for d in self._SCAN_GUIDES]
        else:
            guides = [(float(encoded_of_zone(zone)), label) for zone, label in self._GUIDE_ZONES]
        for tone, label in guides:
            y = int(h - tone * h)
            painter.setPen(_guide_pen())
            painter.drawLine(0, y, w, y)
            _draw_plated_text(
                painter, QRect(THEME.space_xs, y - 12, 40, 12), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignBottom, label
            )
        ranges = self._scan_ranges()
        if self._split():
            # Channel letters, so the panel order reads without telling the traces apart by hue.
            for i, (rect, letter) in enumerate(zip(rects, "RGB")):
                text = letter if ranges is None else f"{letter} {ranges[i]:.2f}"
                _draw_plated_text(
                    painter,
                    rect.adjusted(0, THEME.space_xs, -THEME.space_xs, 0),
                    Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop,
                    text,
                )
        if self._scan:
            caption = self._scan_caption
            if ranges is not None and not self._split():
                caption += " · range " + "  ".join(f"{letter} {r:.2f}" for letter, r in zip("RGB", ranges))
            _draw_plated_text(
                painter, QRect(THEME.space_xs, THEME.space_xs, w, 12), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop, caption
            )
            if self._scan_marks is not None and counts is not None:
                hexes = (THEME.channel_red_text, THEME.channel_green_text, THEME.channel_blue_text)
                for rect, ends, hex_ in zip(channel_rects, self._shown_marks(self._scan_marks), hexes):
                    painter.setPen(QPen(with_alpha(THEME.text_primary if self._split() else hex_, 170), 1))
                    for tone in ends:
                        y = int(min(max(h - tone * h, 1.0), h - 2.0))
                        painter.drawLine(rect.x(), y, rect.x() + rect.width() - 1, y)
            if self._scan_base is not None and counts is not None:
                hexes = (THEME.channel_red_text, THEME.channel_green_text, THEME.channel_blue_text)
                for rect, tone, hex_ in zip(channel_rects, self._shown_marks(self._scan_base), hexes):
                    painter.setPen(QPen(with_alpha(THEME.text_primary if self._split() else hex_, 220), 1, Qt.PenStyle.DotLine))
                    y = int(min(max(h - tone * h, 1.0), h - 2.0))
                    painter.drawLine(rect.x(), y, rect.x() + rect.width() - 1, y)

        if self._marker is not None and counts is not None:
            x, *tones = self._marker
            if len(tones) == 3 and self._scan:
                tones = list(self._shown_marks(np.asarray(tones, dtype=float)))
            if len(tones) == 3 and self._split() and self._aligned():
                # Green's level across all three panels: a neutral rings on it in each.
                gy = min(max(h - tones[1] * h, 0.0), h - 1.0)
                painter.setPen(QPen(with_alpha(THEME.text_on_accent, 90), 1, Qt.PenStyle.DashLine))
                painter.drawLine(QPointF(0, gy), QPointF(w, gy))
            for rect, tone in zip(channel_rects if len(tones) == 3 else rects, tones):
                # Kept a ring's width inside the panel, so an edge column's mark is not cut in half.
                mx = min(max(rect.x() + x * rect.width(), rect.x() + 5.0), rect.x() + rect.width() - 5.0)
                my = min(max(h - tone * h, 5.0), h - 5.0)
                painter.setPen(QPen(with_alpha(THEME.text_on_accent, 90), 1, Qt.PenStyle.DashLine))
                painter.drawLine(QPointF(mx, 0), QPointF(mx, h))
                _draw_hover_ring(painter, QPointF(mx, my))

        # Over the label plates, so a clipped column under a caption still shows.
        shadow, highlight = self._clip_colors
        for x1, x2, y in ticks:
            painter.setPen(QPen(QColor(highlight if y < h / 2 else shadow), 2))
            painter.drawLine(x1, y, x2, y)


class VectorscopeWidget(QWidget):
    """The print's CIELAB a*b* plane, +a right and +b up; brightness is the pixel count at a hue
    and chroma. Rim ticks mark the working-space primaries and secondaries. The log chroma scale
    opens up the near-neutral center, where a toner or a cast lives."""

    _SIZE = 160
    _RINGS = (5.0, 20.0, 40.0)
    _LABEL_ROOM = 28  # the hue letters sit outside the rim

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumHeight(40)
        self._counts: np.ndarray | None = None
        self._log_scale = False
        self._image: QImage | None = None
        self._hues: dict[bool, np.ndarray] = {}
        self._ticks: list[tuple[str, float]] = []
        self._marker: tuple[float, float] | None = None

    def set_marker(self, ab: tuple[float, float] | None) -> None:
        """The hovered pixel's (a*, b*); None hides."""
        if ab != self._marker:
            self._marker = ab
            self.update()

    def set_log_scale(self, enabled: bool) -> None:
        if bool(enabled) != self._log_scale:
            self._log_scale = bool(enabled)
            self._image = None
            self.update()

    def set_data(self, counts: Any) -> None:
        """(hue, chroma) polar counts; None clears."""
        if counts is not None:
            counts = np.asarray(counts, dtype=float)
            if counts.ndim != 2 or float(counts.max()) <= 0.0:
                counts = None
        if counts is None and self._counts is None:
            return
        self._counts = counts
        self._image = None
        self.update()

    def _hue_colors(self) -> np.ndarray:
        """(size, size, 3) uint8: each display pixel's own hue at a chroma that reads, so a
        near-neutral toner lobe still shows its color."""
        if self._log_scale not in self._hues:
            from negpy.features.exposure.analysis import scope_chroma
            from negpy.kernel.image.logic import lab_to_rgb_working, working_oetf_encode

            n = self._SIZE
            u = (np.arange(n) + 0.5) / n * 2.0 - 1.0
            x, y = np.meshgrid(u, u[::-1])
            shown = np.maximum(scope_chroma(np.minimum(np.hypot(x, y), 1.0), self._log_scale), 25.0)
            hue = np.arctan2(y, x)
            lab = np.stack([np.full_like(x, 65.0), shown * np.cos(hue), shown * np.sin(hue)], axis=-1).astype(np.float32)
            rgb = np.asarray(working_oetf_encode(np.clip(lab_to_rgb_working(lab), 0.0, 1.0)))
            self._hues[self._log_scale] = (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
        return self._hues[self._log_scale]

    def _trace_image(self) -> QImage | None:
        if self._image is None and self._counts is not None:
            from negpy.features.exposure.analysis import vectorscope_density

            density = np.log1p(vectorscope_density(self._counts, self._SIZE, self._log_scale))
            lit = density[density > 0]
            if lit.size == 0:
                return None
            # The neutral core piles into a few pixels; scaling by its peak would hide every lobe.
            alpha = np.sqrt(np.clip(density / float(np.percentile(lit, 99)), 0.0, 1.0))
            rgba = np.zeros((self._SIZE, self._SIZE, 4), dtype=np.uint8)
            rgba[..., :3] = (self._hue_colors() * alpha[..., None]).astype(np.uint8)
            rgba[..., 3] = (alpha * 255.0).astype(np.uint8)
            rgba = np.ascontiguousarray(rgba)
            self._image = QImage(rgba.data, self._SIZE, self._SIZE, self._SIZE * 4, QImage.Format.Format_RGBA8888_Premultiplied).copy()
        return self._image

    def _hue_ticks(self) -> list[tuple[str, float]]:
        if not self._ticks:
            from negpy.kernel.image.logic import rgb_to_lab_working

            primaries = (("R", (1, 0, 0)), ("Y", (1, 1, 0)), ("G", (0, 1, 0)), ("C", (0, 1, 1)), ("B", (0, 0, 1)), ("M", (1, 0, 1)))
            lab = rgb_to_lab_working(np.array([rgb for _l, rgb in primaries], dtype=np.float32))
            self._ticks = [(label, float(np.arctan2(lab[i, 2], lab[i, 1]))) for i, (label, _rgb) in enumerate(primaries)]
        return self._ticks

    def paintEvent(self, event) -> None:
        from negpy.features.exposure.analysis import scope_radius

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor(THEME.bg_input))
        side = max(8.0, min(self.width(), self.height()) - float(self._LABEL_ROOM))
        rim = side / 2.0
        center = QPointF(self.width() / 2.0, self.height() / 2.0)

        font = QFont(self.font())
        font.setPixelSize(THEME.font_size_micro)
        painter.setFont(font)
        painter.setPen(_guide_pen())
        for chroma in self._RINGS:
            r = float(scope_radius(chroma, self._log_scale)) * rim
            painter.drawEllipse(center, r, r)
        painter.drawEllipse(center, rim, rim)
        painter.drawLine(QPointF(center.x() - 6, center.y()), QPointF(center.x() + 6, center.y()))
        painter.drawLine(QPointF(center.x(), center.y() - 6), QPointF(center.x(), center.y() + 6))

        image = self._trace_image()
        if image is not None:
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
            painter.drawImage(QRectF(center.x() - rim, center.y() - rim, side, side), image)

        # Ring values along the widest gap between hue letters, so no label meets a letter.
        angles = sorted(angle % (2 * np.pi) for _label, angle in self._hue_ticks())
        gaps = [((b if b > a else b + 2 * np.pi) - a, a) for a, b in zip(angles, angles[1:] + angles[:1])]
        width, start = max(gaps)
        cos, sin = float(np.cos(start + width / 2)), float(np.sin(start + width / 2))
        last = 0.0
        for chroma in self._RINGS:
            r = float(scope_radius(chroma, self._log_scale)) * rim
            if r - last < 14.0:  # a label's height: a closer ring would stack labels
                continue
            last = r
            at = QPointF(center.x() + cos * r, center.y() - sin * r)
            _draw_plated_text(painter, QRect(int(at.x()) - 10, int(at.y()) - 6, 20, 12), Qt.AlignmentFlag.AlignCenter, f"{chroma:.0f}")

        hint = QColor(THEME.text_hint)
        for label, angle in self._hue_ticks():
            cos, sin = float(np.cos(angle)), float(np.sin(angle))
            painter.setPen(QPen(hint, 1.5))
            painter.drawLine(
                QPointF(center.x() + cos * rim, center.y() - sin * rim), QPointF(center.x() + cos * (rim - 5), center.y() - sin * (rim - 5))
            )
            lx, ly = center.x() + cos * (rim + 8), center.y() - sin * (rim + 8)
            painter.drawText(QRectF(lx - 6, ly - 6, 12, 12), Qt.AlignmentFlag.AlignCenter, label)

        if self._marker is not None and self._counts is not None:
            a, b = self._marker
            r = min(float(scope_radius(float(np.hypot(a, b)), self._log_scale)), 1.0) * rim
            angle = float(np.arctan2(b, a))
            spot = QPointF(center.x() + np.cos(angle) * r, center.y() - np.sin(angle) * r)
            painter.setPen(QPen(with_alpha(THEME.text_on_accent, 90), 1, Qt.PenStyle.DashLine))
            painter.drawLine(center, spot)
            _draw_hover_ring(painter, spot)


class ZoneStripWidget(QWidget):
    """Print-zone occupancy strip (0 to X), opacity square-root scaled so a small mass reads.
    A cell click asks for that zone and the next photo click places a tone there. A double-click
    toggles the Zone Overlay, so a single click waits out the double-click interval."""

    zone_clicked = pyqtSignal(int)
    zone_double_clicked = pyqtSignal()
    zone_hovered = pyqtSignal(object)

    _LABELS = ("0", "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X")

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(26)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setMouseTracking(True)
        self._occ: np.ndarray | None = None
        self._warn: tuple[bool, bool] = (False, False)
        self._clip_colors: tuple[str, str] = palette_for("standard").pair
        self._armed: int | None = None
        self._placement_enabled = True
        self._overlay_hint = ""
        self._pending: int | None = None
        self._hovered: int | None = None
        self._click_timer = QTimer(self)
        self._click_timer.setSingleShot(True)
        self._click_timer.setInterval(QApplication.doubleClickInterval())
        self._click_timer.timeout.connect(self._fire_click)

    def set_placement_enabled(self, enabled: bool) -> None:
        """Zone placement works on the print curve only; a slide or a Positive frame says so."""
        self._placement_enabled = bool(enabled)
        if not enabled:
            self.unsetCursor()

    def set_overlay_hint(self, markup: str) -> None:
        """The tooltip's second line, the Zone Overlay double-click with its key chip."""
        self._overlay_hint = markup

    def update_data(self, occ: np.ndarray | None, warnings: tuple[bool, bool] = (False, False)) -> None:
        self._occ = occ
        self._warn = warnings
        self.update()

    def set_clip_colors(self, shadow: str, highlight: str) -> None:
        """The Clipping Overlay's pair, for the tint on a clipped end."""
        if (shadow, highlight) != self._clip_colors:
            self._clip_colors = (shadow, highlight)
            self.update()

    def set_armed(self, zone: int | None) -> None:
        """Outline the zone waiting for a canvas click, or clear it."""
        if zone != self._armed:
            self._armed = zone
            self.update()

    def _cell_at(self, x: float) -> int:
        return int(min(max(x / self.width() * len(self._LABELS), 0), len(self._LABELS) - 1))

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self._occ is not None and self.width() > 0:
            self._pending = self._cell_at(event.position().x())
            self._click_timer.start()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._click_timer.stop()
            self._pending = None
            self.zone_double_clicked.emit()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def _fire_click(self) -> None:
        if self._pending is not None:
            cell, self._pending = self._pending, None
            self.zone_clicked.emit(cell)

    def _set_hovered(self, cell: int | None) -> None:
        if cell != self._hovered:
            self._hovered = cell
            self.zone_hovered.emit(cell)

    def leaveEvent(self, event) -> None:
        self._set_hovered(None)
        super().leaveEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._occ is not None and self.width() > 0:
            cell = self._cell_at(event.position().x())
            self._set_hovered(cell)
            text = tr("Zone {value}: {value1:.1f}%", value=self._LABELS[cell], value1=float(self._occ[cell]) * 100)
            if self._placement_enabled:
                text += tr(" · click to place a tone here")
                self.setCursor(Qt.CursorShape.PointingHandCursor)
            else:
                text = tr("<b>Disabled for Slides/Reversal Film</b><br>{text}", text=text)
            if self._overlay_hint:
                text += f"<br>{self._overlay_hint}"
            self.setToolTip(wrap_tooltip(text))
        else:
            self.unsetCursor()
        super().mouseMoveEvent(event)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        w = self.width()
        h = self.height()
        painter.fillRect(self.rect(), QColor(THEME.bg_input))

        if self._occ is not None:
            n = len(self._LABELS)
            font = QFont(self.font())
            font.setPixelSize(THEME.font_size_micro)
            painter.setFont(font)
            shadow_warn, highlight_warn = self._warn
            for i in range(n):
                x0 = int(w * i / n)
                x1 = int(w * (i + 1) / n)
                frac = float(self._occ[i])
                tone = 40 + int(215 * i / (n - 1))
                alpha = int(min(1.0, np.sqrt(frac)) * 235)
                painter.fillRect(x0, 0, x1 - x0, h, QColor(tone, tone, tone, alpha))
                if shadow_warn and i <= 1:
                    painter.fillRect(x0, 0, x1 - x0, h, with_alpha(self._clip_colors[0], 90))
                elif highlight_warn and i >= n - 2:
                    painter.fillRect(x0, 0, x1 - x0, h, with_alpha(self._clip_colors[1], 90))
                armed = i == self._armed
                if armed:
                    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
                    painter.setPen(QPen(QColor(THEME.border_checked_hover), 1))
                    painter.setBrush(with_alpha(THEME.bg_checked_hover, 200))
                    painter.drawRoundedRect(QRectF(x0 + 1.5, 2.5, x1 - x0 - 3, h - 5), THEME.radius_sm, THEME.radius_sm)
                    painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
                painter.setPen(QPen(QColor(THEME.text_primary) if armed else with_alpha(THEME.text_hint, 160)))
                painter.drawText(QRect(x0, 0, x1 - x0, h), Qt.AlignmentFlag.AlignCenter, self._LABELS[i])


class StepWedgeWidget(QWidget):
    """21-step transmission wedge printed through the frame's curve, alternative process and
    toning, paper black left to paper white right. The usable span is bracketed where neighboring
    steps still separate; labels are wedge density in this scan's own D units."""

    # Label every tenth step, so a sidebar-width strip still gets an axis. At sidebar width
    # the 21 patches are far too narrow to label individually.
    _LABEL_EVERY = 10
    _LABEL_MIN_PX = 34.0  # width per label below which they collide

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(26)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setMouseTracking(True)
        self._enc: np.ndarray | None = None
        self._step_d: float = 0.0
        self._colors: list = []

    def update_data(self, enc: Any, step_density: float, color_space: str, monitor_icc_bytes: Any = None, proof: Any = None) -> None:
        """`enc` is the 21 display-encoded patches, gray (21,) or toned (21, 3). They go through the
        canvas's display transform, soft proof included (it rides the display LUT, not the buffer),
        so wedge and frame agree. The usable span and label contrast read the patches' luminance."""
        from negpy.desktop.converters import ImageConverter
        from negpy.features.lab.logic import LUM_B, LUM_G, LUM_R
        from negpy.kernel.image.logic import working_oetf_decode, working_oetf_encode

        self._step_d = step_density
        if enc is None:
            self._enc = None
            self._colors = []
        else:
            patches = np.clip(np.asarray(enc, dtype=np.float32), 0.0, 1.0)
            rgb = patches if patches.ndim == 2 else np.repeat(patches[:, None], 3, axis=1)
            luma = np.asarray(working_oetf_decode(rgb)) @ np.array([LUM_R, LUM_G, LUM_B], dtype=np.float32)
            self._enc = np.asarray(working_oetf_encode(luma.astype(np.float32)))
            img = ImageConverter.to_qimage(rgb.reshape(1, -1, 3), color_space, monitor_icc_bytes, proof)
            self._colors = [img.pixelColor(i, 0) for i in range(img.width())]
        self.update()

    def mouseMoveEvent(self, event) -> None:
        if self._enc is not None and self.width() > 0:
            n = len(self._enc)
            step = int(min(max(event.position().x() / self.width() * n, 0), n - 1))
            self.setToolTip(tr("Step {step}: density {value:.2f}", step=step, value=step * self._step_d))
        super().mouseMoveEvent(event)

    def _patch_bounds(self) -> list:
        """(x0, x1) per patch. Integer-aligned: fractional edges leave antialiased hairlines
        between adjacent fills, which read as extra steps."""
        n = len(self._enc) if self._enc is not None else 0
        xs = [round(self.width() * i / n) for i in range(n + 1)]
        return [(xs[i], xs[i + 1]) for i in range(n)]

    def paintEvent(self, event) -> None:
        from negpy.features.exposure.analysis import wedge_usable_span

        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(THEME.bg_input))
        if self._enc is None or self.width() < len(self._enc):
            return

        h = self.height()
        bounds = self._patch_bounds()
        for (x0, x1), color in zip(bounds, self._colors):
            painter.fillRect(x0, 0, x1 - x0, h, color)

        span = wedge_usable_span(self._enc)
        if span is not None:
            # The hover ring's pens, so a mark holds on black and white steps alike; ticks at the
            # top and bottom edges leave the middle row to the step labels.
            tick = h // 4
            for edge in span:
                x = (bounds[edge][0] + bounds[edge][1]) // 2
                for pen in (QPen(with_alpha(THEME.canvas_bg_black, 200), 3), QPen(with_alpha(THEME.text_on_accent, 220), 1)):
                    painter.setPen(pen)
                    painter.drawLine(x, 0, x, tick)
                    painter.drawLine(x, h - tick, x, h)

        labelled = range(0, len(self._enc), self._LABEL_EVERY)
        if self.width() / len(labelled) >= self._LABEL_MIN_PX:
            font = QFont(self.font())
            font.setPixelSize(THEME.font_size_micro)
            painter.setFont(font)
            last = labelled[-1]
            for i in labelled:
                text = f"{i * self._step_d:.2f}"
                x0, x1 = bounds[i]
                # A label is wider than its own patch and drawText clips to the rect it is given, so
                # widen the box past the patch. Labels sit 5 apart and cannot collide. The two end
                # labels align to the strip edge the way an axis does: sliding their box inward instead
                # would centre them over the wrong step.
                box = QRect(x0 - (x1 - x0), 0, 3 * (x1 - x0), h)
                flags = Qt.AlignmentFlag.AlignVCenter
                if i == 0:
                    box.setLeft(2)
                    flags |= Qt.AlignmentFlag.AlignLeft
                elif i == last:
                    box.setRight(self.width() - 3)
                    flags |= Qt.AlignmentFlag.AlignRight
                else:
                    flags |= Qt.AlignmentFlag.AlignHCenter
                # Flip against the patch: a light label vanishes on paper white, a dark one on black.
                painter.setPen(
                    with_alpha(THEME.canvas_bg_black, 220) if float(self._enc[i]) > 0.5 else with_alpha(THEME.text_on_accent, 235)
                )
                painter.drawText(box, flags, text)


class ChartWell(QWidget):
    """Charts stacked in one track: the first stretches, a hairline parts each shown chart from the
    one above, and an edge drawn over them rounds the square charts' corners."""

    def __init__(self, *charts: QWidget, parent=None):
        super().__init__(parent)
        self._charts = charts
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(1)
        for i, chart in enumerate(charts):
            layout.addWidget(chart, 0 if i else 1)
        self._edge = _WellEdge(self)

    def shown(self) -> list[QWidget]:
        return [chart for chart in self._charts if not chart.isHidden()]

    def resizeEvent(self, event) -> None:
        self._edge.setGeometry(self.rect())
        super().resizeEvent(event)

    def paintEvent(self, event) -> None:
        QPainter(self).fillRect(self.rect(), QColor(THEME.bg_input))


class _WellEdge(QWidget):
    def __init__(self, well: ChartWell):
        super().__init__(well)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        edge = QPainterPath()
        edge.addRoundedRect(rect, THEME.radius_md, THEME.radius_md)
        outside = QPainterPath()
        outside.addRect(QRectF(self.rect()))
        # The well sits on a section card: the corners take its color.
        painter.fillPath(outside.subtracted(edge), QColor(THEME.bg_card))
        painter.setPen(QPen(QColor(THEME.border_primary), 1))
        for chart in self.parentWidget().shown()[1:]:
            y = chart.y() - 0.5
            painter.drawLine(QPointF(rect.left(), y), QPointF(rect.right(), y))
        painter.drawPath(edge)


class MiniHistogramWidget(QWidget):
    """
    20px-tall luminance strip shown behind the Exposure section header.
    Draws only the L channel at ~40% opacity (always linear scale).
    """

    def __init__(self, parent=None, clip_strips: bool = True):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._data_l: list = []
        self._clip_strips = clip_strips
        self._clip_colors: tuple[str, str] = palette_for("standard").pair
        self._clip_low: bool = False
        self._clip_high: bool = False

    def set_clip_colors(self, shadow: str, highlight: str) -> None:
        """The Clipping Overlay's pair, for the strips at the clipped ends."""
        if (shadow, highlight) != self._clip_colors:
            self._clip_colors = (shadow, highlight)
            self.update()

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
        if not self._clip_strips:
            return
        if self._clip_low:
            painter.setBrush(QBrush(with_alpha(self._clip_colors[0], 180)))
            painter.drawRect(0, 0, 3, h)
        if self._clip_high:
            painter.setBrush(QBrush(with_alpha(self._clip_colors[1], 180)))
            painter.drawRect(w - 3, 0, 3, h)


class MiniRGBHistogramWidget(QWidget):
    """
    Per-channel counterpart to MiniHistogramWidget shown behind the Color section header.
    Overlays the R, G, B channels (~50% opacity) so a color cast reads as the channels
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
        colors = {"r": THEME.channel_red, "g": THEME.channel_green, "b": THEME.channel_blue}
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
            c = QColor(colors[key])
            c.setAlpha(120)
            painter.setBrush(QBrush(c))
            painter.drawPath(path)


class MiniCurveWidget(QWidget):
    """The print curve as a faint stroke behind the Paper Response section header, in
    PhotometricCurveWidget's plot domain, with R/G/B traces when the layers diverge."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._base: list[tuple[float, float]] = []
        self._channels: list[list[tuple[float, float]]] = []

    def set_curves(self, base: list[tuple[float, float]], channels: list[list[tuple[float, float]]]) -> None:
        self._base = list(base)
        self._channels = [list(pts) for pts in channels]
        self.update()

    def _path(self, pts: list[tuple[float, float]], w: int, h: int) -> QPainterPath:
        x0, x1 = PhotometricCurveWidget._X_MIN, PhotometricCurveWidget._X_MAX
        y0, y1 = PhotometricCurveWidget._Y_MIN, PhotometricCurveWidget._Y_MAX
        path = QPainterPath()
        for i, (x, y) in enumerate(pts):
            px, py = (x - x0) / (x1 - x0) * w, h - (y - y0) / (y1 - y0) * h
            if i:
                path.lineTo(px, py)
            else:
                path.moveTo(px, py)
        return path

    def paintEvent(self, event) -> None:
        if not self._base:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        w, h = self.width(), self.height()
        for color, pts in zip((THEME.channel_red, THEME.channel_green, THEME.channel_blue), self._channels):
            painter.setPen(QPen(with_alpha(color, 140), 1.5))
            painter.drawPath(self._path(pts, w, h))
        painter.setPen(QPen(with_alpha(THEME.text_muted, 160), 1.5))
        painter.drawPath(self._path(self._base, w, h))
