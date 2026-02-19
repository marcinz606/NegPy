import numpy as np
from typing import Any
from PyQt6.QtWidgets import QWidget, QSizePolicy
from PyQt6.QtCharts import QChart, QChartView, QLineSeries, QValueAxis
from PyQt6.QtGui import QPainter, QColor, QPen, QBrush, QPainterPath
from PyQt6.QtCore import Qt, QPointF, QMargins
from negpy.kernel.image.logic import get_luminance


class HistogramWidget(QWidget):
    """
    Native high-performance histogram using QPainter.
    Offers additive blending-like visuals and reliable updates.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumHeight(40)
        self._data_r = []
        self._data_g = []
        self._data_b = []
        self._data_l = []

    def update_data(self, buffer: Any) -> None:
        """
        Calculates histograms and triggers repaint.
        """
        if buffer is None:
            self._data_r = []
            self._data_g = []
            self._data_b = []
            self._data_l = []
            self.update()
            return

        if isinstance(buffer, np.ndarray) and buffer.shape == (4, 256):
            self._data_r = self._normalize(buffer[0])
            self._data_g = self._normalize(buffer[1])
            self._data_b = self._normalize(buffer[2])
            self._data_l = self._normalize(buffer[3])
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

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w = self.width()
        h = self.height()

        # Background and Border
        rect = self.rect().adjusted(0, 0, -1, -1)
        painter.fillRect(rect, QColor("#050505"))
        painter.setPen(QPen(QColor("#262626"), 1))
        painter.drawRect(rect)

        # Grid lines
        painter.setPen(QPen(QColor("#1A1A1A"), 1))
        for i in range(1, 4):
            x = int(w * i / 4)
            painter.drawLine(x, 0, x, h)
            y = int(h * i / 4)
            painter.drawLine(0, y, w, y)

        self._draw_channel(painter, self._data_l, "#D4D4D4", 30, 150, w, h)
        self._draw_channel(painter, self._data_r, "#D32F2F", 80, 200, w, h)
        self._draw_channel(painter, self._data_g, "#388E3C", 80, 200, w, h)
        self._draw_channel(painter, self._data_b, "#1976D2", 80, 200, w, h)

    def _draw_channel(
        self,
        painter: QPainter,
        data: list,
        color_hex: str,
        alpha_fill: int,
        alpha_line: int,
        w: int,
        h: int,
    ):
        if not data:
            return

        if len(data) < 2:
            return

        path = QPainterPath()
        path.moveTo(0, h)

        step = w / (len(data) - 1)

        for i, val in enumerate(data):
            x = i * step
            y = h - (val * h)
            path.lineTo(x, y)

        path.lineTo(w, h)
        path.closeSubpath()

        c_fill = QColor(color_hex)
        c_fill.setAlpha(alpha_fill)
        painter.setBrush(QBrush(c_fill))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawPath(path)

        path_line = QPainterPath()
        path_line.moveTo(0, h - (data[0] * h))
        for i, val in enumerate(data):
            x = i * step
            y = h - (val * h)
            path_line.lineTo(x, y)

        c_line = QColor(color_hex)
        c_line.setAlpha(alpha_line)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(c_line, 1.5))
        painter.drawPath(path_line)


class PhotometricCurveWidget(QChartView):
    """
    Sigmoid curve visualization using PyQt6-Charts.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setStyleSheet("background-color: #050505; border: 1px solid #262626;")

        self._chart = QChart()
        self._chart.setBackgroundVisible(False)
        self._chart.setMargins(QMargins(0, 0, 0, 0))
        self._chart.legend().hide()

        # diagonal
        self.series_ref = QLineSeries()
        pen_ref = QPen(QColor("#262626"), 1)
        pen_ref.setStyle(Qt.PenStyle.DashLine)
        self.series_ref.setPen(pen_ref)
        self.series_ref.append(0.0, 0.0)
        self.series_ref.append(1.0, 1.0)
        self._chart.addSeries(self.series_ref)

        # curve
        self.series = QLineSeries()
        self.series.setPen(QPen(QColor("#FFFFFF"), 2.0))
        self._chart.addSeries(self.series)

        self.axis_x = QValueAxis()
        self.axis_x.setRange(-0.1, 1.1)
        self.axis_x.setVisible(False)

        self.axis_y = QValueAxis()
        self.axis_y.setRange(-0.05, 1.05)
        self.axis_y.setVisible(False)

        self._chart.addAxis(self.axis_x, Qt.AlignmentFlag.AlignBottom)
        self._chart.addAxis(self.axis_y, Qt.AlignmentFlag.AlignLeft)

        self.series.attachAxis(self.axis_x)
        self.series.attachAxis(self.axis_y)
        self.series_ref.attachAxis(self.axis_x)
        self.series_ref.attachAxis(self.axis_y)

        self.setChart(self._chart)
        self.setMinimumHeight(40)

    def update_curve(self, params) -> None:
        from negpy.features.exposure.logic import LogisticSigmoid
        from negpy.features.exposure.models import EXPOSURE_CONSTANTS
        from negpy.kernel.image.validation import ensure_image

        master_ref = 1.0
        exposure_shift = 0.1 + (params.density * EXPOSURE_CONSTANTS["density_multiplier"])
        pivot = master_ref - exposure_shift
        slope = 1.0 + (params.grade * EXPOSURE_CONSTANTS["grade_multiplier"])

        curve = LogisticSigmoid(
            contrast=slope,
            pivot=pivot,
            d_max=3.5,
            toe=params.toe,
            toe_width=params.toe_width,
            toe_hardness=params.toe_hardness,
            shoulder=params.shoulder,
            shoulder_width=params.shoulder_width,
            shoulder_hardness=params.shoulder_hardness,
            shadows=params.shadows,
            highlights=params.highlights,
        )

        plt_x = np.linspace(-0.1, 1.1, 50)
        x_log_exp = 1.0 - plt_x

        d = curve(ensure_image(x_log_exp))
        t = np.power(10.0, -d)
        y = np.power(t, 1.0 / 2.2)

        points = [QPointF(px, py) for px, py in zip(plt_x, y)]
        self.series.replace(points)
