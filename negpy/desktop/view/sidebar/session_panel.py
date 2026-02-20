from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGroupBox,
    QSplitter,
    QLabel,
    QStackedWidget,
    QPushButton,
    QScrollArea,
)
from typing import Dict, Any
from PyQt6.QtCore import pyqtSignal, Qt, QThread
import qtawesome as qta
from negpy.desktop.view.styles.theme import THEME
from negpy.desktop.controller import AppController
from negpy.desktop.view.widgets.charts import HistogramWidget, PhotometricCurveWidget
from negpy.desktop.view.sidebar.header import SidebarHeader
from negpy.desktop.view.sidebar.files import FileBrowser
from negpy.desktop.view.sidebar.export import ExportSidebar
from negpy.kernel.system.version import check_for_updates


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
        self.update_label.setStyleSheet("font-size: 12px; color: #2e7d32; font-weight: bold; padding: 5px;")
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

        # Custom Tab Switcher
        self.tab_container = QWidget()
        tab_vbox = QVBoxLayout(self.tab_container)
        tab_vbox.setContentsMargins(0, 0, 0, 0)
        tab_vbox.setSpacing(0)

        switcher_layout = QHBoxLayout()
        switcher_layout.setContentsMargins(0, 0, 0, 0)
        switcher_layout.setSpacing(0)

        self.btn_tab_analysis = QPushButton(" Analysis")
        self.btn_tab_analysis.setIcon(qta.icon("fa5s.chart-bar", color=THEME.text_secondary))
        self.btn_tab_export = QPushButton(" Export")
        self.btn_tab_export.setIcon(qta.icon("fa5s.file-export", color=THEME.text_secondary))

        for btn in [self.btn_tab_analysis, self.btn_tab_export]:
            btn.setCheckable(True)
            btn.setFixedHeight(38)
            btn.setStyleSheet(f"""
                QPushButton {{
                    text-align: center;
                    font-weight: bold;
                    font-size: {THEME.font_size_header}px;
                    background-color: #0D0D0D;
                    border: none;
                    border-bottom: 1px solid #262626;
                    border-right: 1px solid #262626;
                    color: #A0A0A0;
                }}
                QPushButton:hover {{
                    background-color: #262626;
                    color: #FFFFFF;
                }}
                QPushButton:checked {{
                    background-color: #222222;
                    color: #FFFFFF;
                    border-bottom: none;
                }}
            """)
            switcher_layout.addWidget(btn)

        tab_vbox.addLayout(switcher_layout)

        self.stack = QStackedWidget()
        self.stack.setContentsMargins(0, 8, 0, 0)
        tab_vbox.addWidget(self.stack)

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

        self.stack.addWidget(wrap_scroll(self.analysis_group))

        self.export_sidebar = ExportSidebar(self.controller)
        self.stack.addWidget(wrap_scroll(self.export_sidebar))

        # Default state
        self.btn_tab_analysis.setChecked(True)
        self.stack.setCurrentIndex(0)

        self.splitter.addWidget(self.tab_container)
        self.splitter.setStretchFactor(0, 3)
        self.splitter.setStretchFactor(1, 1)

        layout.addWidget(self.splitter)

    def _connect_signals(self) -> None:
        self.controller.image_updated.connect(self._update_analysis)
        self.controller.metrics_available.connect(self._on_metrics_available)
        self.controller.config_updated.connect(self.export_sidebar.sync_ui)

        self.btn_tab_analysis.clicked.connect(lambda: self._switch_tab(0))
        self.btn_tab_export.clicked.connect(lambda: self._switch_tab(1))

    def _switch_tab(self, index: int) -> None:
        self.stack.setCurrentIndex(index)
        self.btn_tab_analysis.setChecked(index == 0)
        self.btn_tab_export.setChecked(index == 1)

        # Sync icon colors
        self.btn_tab_analysis.setIcon(qta.icon("fa5s.chart-bar", color="white" if index == 0 else THEME.text_secondary))
        self.btn_tab_export.setIcon(qta.icon("fa5s.file-export", color="white" if index == 1 else THEME.text_secondary))

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
