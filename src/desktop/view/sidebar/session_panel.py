from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QGroupBox,
    QSplitter,
    QLabel,
    QTabWidget,
    QScrollArea,
)
from typing import Dict, Any
from PyQt6.QtCore import pyqtSignal, Qt, QThread
from src.desktop.controller import AppController
from src.desktop.view.widgets.charts import HistogramWidget, PhotometricCurveWidget
from src.desktop.view.sidebar.header import SidebarHeader
from src.desktop.view.sidebar.files import FileBrowser
from src.desktop.view.sidebar.export import ExportSidebar
from src.desktop.view.styles.theme import THEME
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

        self.update_label = QLabel("")
        self.update_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.update_label.setStyleSheet(
            "font-size: 12px; color: #2e7d32; font-weight: bold; padding: 5px;"
        )
        self.update_label.setVisible(False)
        layout.addWidget(self.update_label)

        self.header = SidebarHeader(self.controller)
        layout.addWidget(self.header)

        self.update_worker = UpdateCheckWorker()
        self.update_worker.finished.connect(self._on_update_found)
        self.update_worker.start()

        self.splitter = QSplitter(Qt.Orientation.Vertical)

        self.file_browser = FileBrowser(self.controller)
        self.splitter.addWidget(self.file_browser)

        self.tabs = QTabWidget()
        self.tabs.tabBar().setExpanding(True)
        self.tabs.setStyleSheet(f"""
            QTabWidget::pane {{
                border: 1px solid {THEME.border_color};
                background-color: {THEME.bg_panel};
            }}
            QTabBar::tab {{
                background-color: {THEME.bg_header};
                color: {THEME.text_secondary};
                font-size: {THEME.font_size_header}px;
                padding: 8px 12px;
                border: 1px solid {THEME.border_color};
                min-width: 80px;
            }}
            QTabBar::tab:selected {{
                background-color: #000000;
                color: white;
                font-weight: bold;
                border-bottom-color: {THEME.accent_primary};
            }}
            QTabBar::tab:hover:!selected {{
                background-color: {THEME.accent_secondary};
            }}
        """)

        def wrap_scroll(widget: QWidget) -> QScrollArea:
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setWidget(widget)
            scroll.setStyleSheet("QScrollArea { border: none; }")
            return scroll

        self.analysis_group = QGroupBox()
        analysis_layout = QVBoxLayout(self.analysis_group)
        analysis_layout.setContentsMargins(5, 5, 5, 5)

        self.hist_widget = HistogramWidget()
        self.curve_widget = PhotometricCurveWidget()

        analysis_layout.addWidget(self.hist_widget, 1)
        analysis_layout.addWidget(self.curve_widget, 1)
        self.tabs.addTab(wrap_scroll(self.analysis_group), "Analysis")

        self.export_sidebar = ExportSidebar(self.controller)
        self.tabs.addTab(wrap_scroll(self.export_sidebar), "Export")

        self.splitter.addWidget(self.tabs)
        self.splitter.setStretchFactor(0, 3)
        self.splitter.setStretchFactor(1, 1)

        layout.addWidget(self.splitter)

    def _connect_signals(self) -> None:
        self.controller.image_updated.connect(self._update_analysis)
        self.controller.metrics_available.connect(self._on_metrics_available)
        self.controller.config_updated.connect(self.export_sidebar.sync_ui)

    def _on_metrics_available(self, metrics: Dict[str, Any]) -> None:
        hist_data = metrics.get("histogram_raw")
        if hist_data is not None:
            self.hist_widget.update_data(hist_data)

    def _update_analysis(self) -> None:
        metrics = self.controller.session.state.last_metrics

        hist_data = metrics.get("histogram_raw")
        if hist_data is not None:
            self.hist_widget.update_data(hist_data)
        else:
            buffer = metrics.get("analysis_buffer")
            if buffer is None:
                buffer = metrics.get("base_positive")
            if buffer is not None:
                self.hist_widget.update_data(buffer)

        self.curve_widget.update_curve(self.controller.session.state.config.exposure)

    def _on_update_found(self, version: str) -> None:
        self.update_label.setText(f"Update Available: v{version}")
        self.update_label.setVisible(True)
