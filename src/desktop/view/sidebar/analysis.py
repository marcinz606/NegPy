from src.desktop.view.widgets.charts import HistogramWidget, PhotometricCurveWidget
from src.desktop.view.sidebar.base import BaseSidebar


class AnalysisSidebar(BaseSidebar):
    """
    Panel for real-time histograms and characteristic curves.
    """

    def _init_ui(self) -> None:
        self.hist_widget = HistogramWidget()
        self.curve_widget = PhotometricCurveWidget()

        self.layout.addWidget(self.hist_widget)
        self.layout.addWidget(self.curve_widget)

    def _connect_signals(self) -> None:
        self.controller.image_updated.connect(self.sync_ui)

    def sync_ui(self) -> None:
        metrics = self.state.last_metrics
        if "base_positive" in metrics:
            self.hist_widget.update_data(metrics["base_positive"])

        self.curve_widget.update_curve(self.state.config.exposure)
