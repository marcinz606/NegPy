from PyQt6.QtWidgets import QWidget, QVBoxLayout
from src.desktop.controller import AppController
from src.desktop.view.widgets.charts import HistogramWidget, PhotometricCurveWidget


class AnalysisSidebar(QWidget):
    """
    Panel for real-time histograms and characteristic curves.
    """

    def __init__(self, controller: AppController):
        super().__init__()
        self.controller = controller
        self.state = controller.state

        self._init_ui()
        self._connect_signals()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 0, 5, 5)

        self.hist_widget = HistogramWidget()
        self.curve_widget = PhotometricCurveWidget()

        layout.addWidget(self.hist_widget)
        layout.addWidget(self.curve_widget)

    def _connect_signals(self) -> None:
        self.controller.image_updated.connect(self._refresh_plots)

    def _refresh_plots(self) -> None:
        metrics = self.state.last_metrics
        if "base_positive" in metrics:
            self.hist_widget.update_data(metrics["base_positive"])

        self.curve_widget.update_curve(self.state.config.exposure)
