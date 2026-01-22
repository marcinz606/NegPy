import numpy as np
from typing import Any
from PyQt6.QtCharts import QChart, QChartView, QLineSeries, QValueAxis
from PyQt6.QtGui import QPainter, QColor, QPen
from PyQt6.QtCore import Qt, QPointF, QMargins
from src.kernel.image.logic import get_luminance


class HistogramWidget(QChartView):
    """
    Native high-performance histogram using PyQt6-Charts.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)

        self._chart = QChart()
        self._chart.setBackgroundVisible(False)
        self._chart.setMargins(QMargins(0, 0, 0, 0))
        self._chart.layout().setContentsMargins(0, 0, 0, 0)
        self._chart.legend().hide()

        self.setChart(self._chart)
        self.setMinimumHeight(40)

        # Series setup
        self.series_r = QLineSeries()
        self.series_g = QLineSeries()
        self.series_b = QLineSeries()
        self.series_l = QLineSeries()

        self.series_r.setPen(QPen(QColor("#b10000"), 1.2))  # Primary Red
        self.series_g.setPen(QPen(QColor("#00b1b1"), 1.2))  # Cyan
        self.series_b.setPen(QPen(QColor("#b100b1"), 1.2))  # Magenta
        self.series_l.setPen(QPen(QColor("#c3c3c3"), 1.5))  # Text Grey for Luma

        self._chart.addSeries(self.series_r)
        self._chart.addSeries(self.series_g)
        self._chart.addSeries(self.series_b)
        self._chart.addSeries(self.series_l)

        # Axes
        self.axis_x = QValueAxis()
        self.axis_x.setRange(0, 255)
        self.axis_x.setVisible(False)

        self.axis_y = QValueAxis()
        self.axis_y.setRange(0, 1)  # Normalized
        self.axis_y.setVisible(False)

        self._chart.addAxis(self.axis_x, Qt.AlignmentFlag.AlignBottom)
        self._chart.addAxis(self.axis_y, Qt.AlignmentFlag.AlignLeft)

        for s in [self.series_r, self.series_g, self.series_b, self.series_l]:
            s.attachAxis(self.axis_x)
            s.attachAxis(self.axis_y)

    def update_data(self, buffer: Any) -> None:
        """
        Calculates histograms and updates chart series.
        Supports both NumPy buffers and raw histogram counts.
        """
        if buffer is None:
            return

        # Case 1: Raw Histogram Data (from GPU Engine)
        if isinstance(buffer, np.ndarray) and buffer.shape == (4, 256):

            def get_points_raw(counts: np.ndarray) -> list:
                max_val = float(np.max(counts))
                if max_val <= 0:
                    return []
                return [
                    QPointF(i, float(val) / max_val) for i, val in enumerate(counts)
                ]

            self.series_r.replace(get_points_raw(buffer[0]))
            self.series_g.replace(get_points_raw(buffer[1]))
            self.series_b.replace(get_points_raw(buffer[2]))
            self.series_l.replace(get_points_raw(buffer[3]))
            self.axis_x.setRange(0, 255)
            return

        # Case 2: NumPy Buffer (CPU Fallback)
        if not isinstance(buffer, np.ndarray):
            return

        # Downsample for speed if huge
        if buffer.shape[0] > 500:
            buffer = buffer[::4, ::4]

        lum = get_luminance(buffer)

        def get_points(data):
            hist, _ = np.histogram(data, bins=64, range=(0, 1))
            if hist.max() > 0:
                hist = hist / hist.max()  # Normalize

            points = []
            for i, val in enumerate(hist):
                points.append(QPointF(i * (255 / 63), val))
            return points

        self.series_r.replace(get_points(buffer[..., 0]))
        self.series_g.replace(get_points(buffer[..., 1]))
        self.series_b.replace(get_points(buffer[..., 2]))
        self.series_l.replace(get_points(lum))
        self.axis_x.setRange(0, 255)


class PhotometricCurveWidget(QChartView):
    """
    Sigmoid curve visualization using PyQt6-Charts.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)

        self._chart = QChart()
        self._chart.setBackgroundVisible(False)
        self._chart.setMargins(QMargins(0, 0, 0, 0))
        self._chart.legend().hide()

        self.series = QLineSeries()
        self.series.setPen(QPen(QColor("#e0e0e0"), 2))
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

        self.setChart(self._chart)
        self.setMinimumHeight(40)

    def update_curve(self, params) -> None:
        from src.features.exposure.logic import LogisticSigmoid
        from src.features.exposure.models import EXPOSURE_CONSTANTS
        from src.kernel.image.validation import ensure_image

        master_ref = 1.0
        exposure_shift = 0.1 + (
            params.density * EXPOSURE_CONSTANTS["density_multiplier"]
        )
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
        )

        plt_x = np.linspace(-0.1, 1.1, 50)
        x_log_exp = 1.0 - plt_x

        d = curve(ensure_image(x_log_exp))
        t = np.power(10.0, -d)
        y = np.power(t, 1.0 / 2.2)

        points = [QPointF(px, py) for px, py in zip(plt_x, y)]
        self.series.replace(points)
