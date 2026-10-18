"""Measure a Crosstalk matrix from a color chart photographed on the roll, and save it as a profile."""

from negpy.kernel.system.i18n import tr
from typing import Optional

import numpy as np
from PyQt6.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QMouseEvent, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import QDialog, QHBoxLayout, QLabel, QLineEdit, QPushButton, QSizePolicy, QVBoxLayout

from negpy.desktop.converters import ImageConverter
from negpy.desktop.view.styles.templates import field_row, hint_label, pin_dialog_default
from negpy.desktop.view.styles.theme import THEME
from negpy.desktop.view.widgets.choice_button import SegmentedChoice
from negpy.desktop.view.widgets.dialog_geometry import remember_dialog_geometry
from negpy.desktop.workers.chart_fit import ChartFitWorker, ChartSource
from negpy.features.process.chart import CHARTS, ChartFit, grid_corners, patch_centers, sample_radius
from negpy.features.process.models import ProcessMode
from negpy.services.assets.crosstalk import CrosstalkProfiles, CrosstalkType

_MEASURE_LABEL = tr("Measure and Save")
_HANDLE_R = 6
_GRAB_PX = 14
_START_CORNERS = ((0.2, 0.25), (0.8, 0.25), (0.8, 0.75), (0.2, 0.75))
# ΔE00 tiers of a patch after the fit: close, then noticeable side by side, then far.
_DE_CLOSE = 2.0
_DE_NOTICEABLE = 5.0


def _tier_color(de: float) -> QColor:
    return QColor(THEME.status_success if de <= _DE_CLOSE else THEME.warn_amber if de <= _DE_NOTICEABLE else THEME.error)


class _ChartCornersLabel(QLabel):
    """The frame with four draggable dots on the corner patches and the 24 sample boxes they give."""

    changed = pyqtSignal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(320)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self._pixmap: Optional[QPixmap] = None
        self._corners = np.array(_START_CORNERS, dtype=np.float64)
        self._drag: Optional[int] = None
        self._scores: Optional[np.ndarray] = None

    def set_frame(self, pixmap: QPixmap) -> None:
        self._pixmap = pixmap
        self.update()

    def set_scores(self, scores: Optional[np.ndarray]) -> None:
        """ΔE00 per sample box, in the boxes' order; None hides them."""
        self._scores = scores
        self.update()

    def corners(self) -> np.ndarray:
        """TL, TR, BR, BL as fractions of the frame."""
        return self._corners.copy()

    def _draw_rect(self) -> Optional[QRectF]:
        if self._pixmap is None or self._pixmap.isNull():
            return None
        pw, ph = self._pixmap.width(), self._pixmap.height()
        scale = min(self.width() / pw, self.height() / ph)
        dw, dh = pw * scale, ph * scale
        return QRectF((self.width() - dw) / 2, (self.height() - dh) / 2, dw, dh)

    def _to_widget(self, r: QRectF, p: np.ndarray) -> QPointF:
        return QPointF(r.x() + p[0] * r.width(), r.y() + p[1] * r.height())

    def mousePressEvent(self, ev: QMouseEvent) -> None:
        r = self._draw_rect()
        if r is None:
            return
        pos = ev.position()
        dists = [np.hypot(pos.x() - q.x(), pos.y() - q.y()) for q in (self._to_widget(r, c) for c in self._corners)]
        nearest = int(np.argmin(dists))
        self._drag = nearest if dists[nearest] <= _GRAB_PX else None

    def mouseMoveEvent(self, ev: QMouseEvent) -> None:
        r = self._draw_rect()
        if r is None or self._drag is None:
            return
        pos = ev.position()
        self._corners[self._drag] = (
            min(1.0, max(0.0, (pos.x() - r.x()) / r.width())),
            min(1.0, max(0.0, (pos.y() - r.y()) / r.height())),
        )
        self._scores = None
        self.update()

    def mouseReleaseEvent(self, _ev: QMouseEvent) -> None:
        if self._drag is not None:
            self._drag = None
            self.changed.emit()

    def paintEvent(self, _ev) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = self._draw_rect()
        if r is None or self._pixmap is None:
            painter.fillRect(self.rect(), QColor(THEME.bg_dark))
            painter.setPen(QColor(THEME.text_hint))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, tr("Rendering the frame…"))
            painter.end()
            return
        painter.drawPixmap(r, self._pixmap, QRectF(self._pixmap.rect()))
        scale = np.array([r.width(), r.height()])
        px = self._corners * scale
        half = sample_radius(px)
        boxes = [QRectF(r.x() + x - half, r.y() + y - half, 2 * half, 2 * half) for x, y in patch_centers(px)]
        scores = self._scores
        colors = [QColor(THEME.warn_amber)] * len(boxes) if scores is None else [_tier_color(float(v)) for v in scores]
        painter.setBrush(Qt.BrushStyle.NoBrush)
        for box, color in zip(boxes, colors):
            painter.setPen(QPen(color, 1.5))
            painter.drawRect(box)
        painter.setPen(QPen(QColor(THEME.status_success), 1.5))
        pts = [self._to_widget(r, c) for c in grid_corners(self._corners)]
        for a, b in zip(pts, pts[1:] + pts[:1]):
            painter.drawLine(a, b)
        painter.setBrush(QColor(THEME.status_success))
        painter.setPen(Qt.PenStyle.NoPen)
        for p in pts:
            painter.drawEllipse(p, _HANDLE_R, _HANDLE_R)
        if scores is not None:
            self._paint_scores(painter, boxes, colors, scores)
        painter.end()

    def _paint_scores(self, painter: QPainter, boxes: list[QRectF], colors: list[QColor], scores: np.ndarray) -> None:
        """Each box's ΔE00 on a dark chip, over the outline and handles so none hides a number."""
        font = painter.font()
        font.setPixelSize(THEME.font_size_small)
        painter.setFont(font)
        backdrop = QColor(THEME.bg_dark)
        backdrop.setAlpha(200)
        for box, color, score in zip(boxes, colors, scores):
            text = f"{score:.1f}"
            label = QRectF(painter.fontMetrics().boundingRect(text)).adjusted(-THEME.space_sm, 0, THEME.space_sm, 0)
            label.moveCenter(box.center())
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(backdrop)
            painter.drawRoundedRect(label, THEME.radius_sm, THEME.radius_sm)
            painter.setPen(color)
            painter.drawText(label, Qt.AlignmentFlag.AlignCenter, text)


class ChartCalibrationDialog(QDialog):
    """Mark the chart's corner patches, fit the matrix through the real render, save a measured profile."""

    profile_saved = pyqtSignal(str)

    def __init__(self, source: ChartSource, color_space: str, parent=None, *, repo=None) -> None:
        super().__init__(parent)
        self.setWindowTitle(tr("Measure from Chart"))
        self.resize(760, 640)
        self._color_space = color_space
        self._render_shape: Optional[tuple[int, int]] = None
        self._fitting = False
        self._worker = ChartFitWorker(source, self)
        self._worker.previewed.connect(self._on_previewed)
        self._worker.progress.connect(self._on_progress)
        self._worker.fitted.connect(self._on_fitted)
        self._worker.failed.connect(self._on_failed)
        self._init_ui()
        remember_dialog_geometry(self, repo, "chart_calibration")
        self._worker.start_preview()

    def _init_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setSpacing(THEME.space_md)
        intro = hint_label(
            tr(
                "Use a frame of a SpyderCheckr 24 or a ColorChecker 24 shot on this film and scanned on this rig. "
                "Drag the four green dots onto the centers of the chart's corner patches, in any order. The chart "
                "can sit any way up, sideways or mirrored. NegPy reads its orientation from the colors, then "
                "finds the Crosstalk matrix that brings the patches nearest the chart's published colors, through "
                "the frame's own render."
            )
        )
        intro.setWordWrap(True)
        root.addWidget(intro)

        self.view = _ChartCornersLabel()
        root.addWidget(self.view, 1)

        self.chart_choice = SegmentedChoice(
            tuple(("", c.name) for c in CHARTS),
            tr("The chart in the frame. Each has its own published patch colors."),
        )
        root.addLayout(field_row(tr("Chart"), self.chart_choice))

        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText(tr("e.g. Portra 400 + CS-Lite"))
        self.name_edit.setToolTip(tr("The profile's name in the Matrix dropdown. Name the film and the rig."))
        self.name_edit.textChanged.connect(self._refresh)
        root.addLayout(field_row(tr("Name"), self.name_edit))

        self.result_label = hint_label()
        self.result_label.setWordWrap(True)
        self.result_label.setVisible(False)
        root.addWidget(self.result_label)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self.cancel_btn = QPushButton(tr("Cancel"))
        self.cancel_btn.clicked.connect(self.reject)
        self.measure_btn = QPushButton(_MEASURE_LABEL)
        self.measure_btn.setToolTip(tr("Fit the matrix to the marked chart and save it as a Measured profile."))
        self.measure_btn.clicked.connect(self._measure)
        btn_row.addWidget(self.cancel_btn)
        btn_row.addWidget(self.measure_btn)
        pin_dialog_default(self.measure_btn, self.cancel_btn)
        root.addLayout(btn_row)
        self._refresh()

    def _refresh(self) -> None:
        name = self.name_edit.text().strip()
        ready = self._render_shape is not None and not self._fitting
        self.measure_btn.setEnabled(ready and bool(name) and not CrosstalkProfiles.is_bundled(name))
        self.name_edit.setEnabled(not self._fitting)
        self.chart_choice.setEnabled(not self._fitting)

    def _on_previewed(self, encoded: np.ndarray) -> None:
        self._render_shape = encoded.shape[:2]
        self.view.set_frame(QPixmap.fromImage(ImageConverter.to_qimage(np.ascontiguousarray(encoded, dtype=np.float32), self._color_space)))
        self._refresh()

    def _measure(self) -> None:
        if self._render_shape is None:
            return
        h, w = self._render_shape
        self._fitting = True
        self.measure_btn.setText(tr("Measuring…"))
        self._show_result(tr("Rendering the frame with each trial matrix…"))
        self._refresh()
        self._worker.start_fit(self.view.corners() * np.array([w, h]), CHARTS[self.chart_choice.currentIndex()])

    def _on_progress(self, frac: float) -> None:
        self.measure_btn.setText(tr("Measuring… {value:.0f}%", value=frac * 100))

    def _on_fitted(self, fit: ChartFit) -> None:
        self._fitting = False
        self.measure_btn.setText(_MEASURE_LABEL)
        name = self.name_edit.text().strip()
        CrosstalkProfiles.save(name, list(fit.matrix), CrosstalkType.MEASURED, process=ProcessMode.C41)
        chart = CHARTS[self.chart_choice.currentIndex()]
        worst = int(np.argmax(fit.after_color))
        # Tiers and counts use the value as shown, so a "2.0" chip counts within 2.
        shown = np.round(fit.after_color, 1)
        # The boxes are read in grid order; fit.order gives the printed patch each one holds.
        self.view.set_scores(shown[fit.order])
        close = int((shown <= _DE_CLOSE).sum())
        far = int((shown > _DE_NOTICEABLE).sum())
        self._show_result(
            tr(
                "<b>Saved “{value}”</b>. Mean color ΔE00 (hue and chroma) over the 24 patches: {value1:.1f} without the matrix, {value2:.1f} with it; {value3:.1f} and {value4:.1f} with lightness. Per patch, on the frame: {value5} within {value6:g}, {value7} within {value8:g}, {value9} beyond. Farthest patch: {value10}, {value11:.1f}. Re-run Roll Analysis on the roll's other frames. The matrix holds over the frame's current sensor profile: measure again if you change it.",
                value=name,
                value1=fit.before_color.mean(),
                value2=fit.after_color.mean(),
                value3=fit.before.mean(),
                value4=fit.after.mean(),
                value5=close,
                value6=_DE_CLOSE,
                value7=shown.size - close - far,
                value8=_DE_NOTICEABLE,
                value9=far,
                value10=chart.patch_names[worst],
                value11=fit.after_color[worst],
            )
        )
        self._pin_close()
        self._refresh()
        self.profile_saved.emit(name)

    def _on_failed(self, msg: str) -> None:
        self._fitting = False
        self.measure_btn.setText(_MEASURE_LABEL)
        self._show_result(tr("Could not measure the chart: {value}", value=msg))
        self._refresh()

    def _pin_close(self) -> None:
        self.cancel_btn.setText(tr("Close"))
        self.measure_btn.setDefault(False)
        self.measure_btn.setProperty("primary", False)
        pin_dialog_default(self.cancel_btn, self.measure_btn)
        for btn in (self.cancel_btn, self.measure_btn):
            btn.style().unpolish(btn)
            btn.style().polish(btn)

    def _show_result(self, text: str) -> None:
        self.result_label.setText(text)
        self.result_label.setVisible(True)

    def done(self, result: int) -> None:
        self._worker.cancel()
        self._worker.wait()
        super().done(result)
