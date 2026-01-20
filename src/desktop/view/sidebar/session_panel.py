from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QGroupBox,
    QSplitter,
    QLabel,
    QTabWidget,
)
from PyQt6.QtCore import pyqtSignal, Qt, QThread
from src.desktop.controller import AppController
from src.desktop.view.widgets.charts import HistogramWidget, PhotometricCurveWidget
from src.desktop.view.sidebar.files import FileBrowser
from src.desktop.view.sidebar.export import ExportSidebar
from src.kernel.system.version import check_for_updates


class UpdateCheckWorker(QThread):
    """Background worker to check for new releases."""

    finished = pyqtSignal(str)

    def run(self):
        new_ver = check_for_updates()
        if new_ver:
            self.finished.emit(new_ver)


class SessionPanel(QWidget):
    """
    Left sidebar panel containing file browser, update check, and analysis/export tabs.
    """

    def __init__(self, controller: AppController):
        super().__init__()
        self.controller = controller

        self._init_ui()
        self._connect_signals()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Update Notification
        self.update_label = QLabel("")
        self.update_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.update_label.setStyleSheet(
            "font-size: 11px; color: #2e7d32; font-weight: bold; padding: 5px;"
        )
        self.update_label.setVisible(False)
        layout.addWidget(self.update_label)

        self.update_worker = UpdateCheckWorker()
        self.update_worker.finished.connect(self._on_update_found)
        self.update_worker.start()

        # Splitter for Files and Tabs (Analysis/Export)
        self.splitter = QSplitter(Qt.Orientation.Vertical)

        # File Browser
        self.file_browser = FileBrowser(self.controller)
        self.splitter.addWidget(self.file_browser)

        # Bottom Tabs
        self.tabs = QTabWidget()

        # Analysis Tab
        self.analysis_group = QGroupBox()
        analysis_layout = QVBoxLayout(self.analysis_group)
        analysis_layout.setContentsMargins(0, 5, 0, 0)

        self.hist_widget = HistogramWidget()
        self.curve_widget = PhotometricCurveWidget()

        analysis_layout.addWidget(self.hist_widget)
        analysis_layout.addWidget(self.curve_widget)
        self.tabs.addTab(self.analysis_group, "Analysis")

        # Export Tab
        self.export_sidebar = ExportSidebar(self.controller)
        self.tabs.addTab(self.export_sidebar, "Export")

        self.splitter.addWidget(self.tabs)
        self.splitter.setStretchFactor(0, 3)
        self.splitter.setStretchFactor(1, 1)

        layout.addWidget(self.splitter)

    def _connect_signals(self) -> None:
        self.controller.image_updated.connect(self._update_analysis)
        self.controller.config_updated.connect(self.export_sidebar.sync_ui)

    def _update_analysis(self) -> None:
        metrics = self.controller.session.state.last_metrics
        if "base_positive" in metrics:
            self.hist_widget.update_data(metrics["base_positive"])

        self.curve_widget.update_curve(self.controller.session.state.config.exposure)

    def _on_update_found(self, version: str) -> None:
        self.update_label.setText(f"Update Available: v{version}")
        self.update_label.setVisible(True)

    def resizeEvent(self, event) -> None:
        """Enforce bottom tabs max height"""
        super().resizeEvent(event)
        total_h = self.splitter.height()
        if total_h > 0:
            max_tabs_h = int(total_h * 0.3)
            current_sizes = self.splitter.sizes()
            if len(current_sizes) > 1 and current_sizes[1] > max_tabs_h:
                new_list_h = total_h - max_tabs_h
                self.splitter.setSizes([new_list_h, max_tabs_h])
