import os
from PyQt6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QDockWidget,
    QStatusBar,
    QScrollArea,
    QProgressBar,
)
from PyQt6.QtGui import QShortcut, QKeySequence
from PyQt6.QtCore import Qt, QTimer
from PIL import Image
import numpy as np

import qtawesome as qta
from src.desktop.view.canvas.widget import ImageCanvas
from src.desktop.view.canvas.toolbar import ActionToolbar
from src.desktop.view.sidebar.analysis import AnalysisSidebar
from src.desktop.view.sidebar.presets import PresetsSidebar
from src.desktop.view.sidebar.exposure import ExposureSidebar
from src.desktop.view.sidebar.geometry import GeometrySidebar
from src.desktop.view.sidebar.lab import LabSidebar
from src.desktop.view.sidebar.toning import ToningSidebar
from src.desktop.view.sidebar.retouch import RetouchSidebar
from src.desktop.view.sidebar.metadata import MetadataSidebar
from src.desktop.view.sidebar.icc import ICCSidebar
from src.desktop.view.sidebar.export import ExportSidebar
from src.desktop.view.sidebar.files import FilesSidebar
from src.desktop.view.widgets.collapsible import CollapsibleSection
from src.desktop.view.styles.theme import THEME
from src.desktop.controller import AppController
from src.desktop.session import ToolMode
from src.services.export.print import PrintService
from src.kernel.image.logic import float_to_uint8


class TopStatusBar(QWidget):
    """Integrated status bar at the top of the viewport."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(48)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 0, 10, 0)

        self.msg_label = QLabel("Ready")
        self.msg_label.setStyleSheet("color: #aaa; font-size: 16px;")

        self.progress = QProgressBar()
        self.progress.setMaximumWidth(150)
        self.progress.setFixedHeight(12)
        self.progress.setVisible(False)
        self.progress.setTextVisible(False)
        self.progress.setStyleSheet(
            """
            QProgressBar { background-color: #222; border: 1px solid #333; border-radius: 6px; }
            QProgressBar::chunk { background-color: #2e7d32; border-radius: 5px; }
        """
        )

        layout.addWidget(self.msg_label)
        layout.addStretch()
        layout.addWidget(self.progress)

    def showMessage(self, text: str, timeout: int = 0):
        if text == "Image Updated":
            return
        self.msg_label.setText(text)
        if timeout > 0:
            QTimer.singleShot(timeout, lambda: self.msg_label.setText("Ready"))


class MainWindow(QMainWindow):
    """
    Main application window hosting the canvas, sidebar, and asset browser.
    """

    def __init__(self, controller: AppController):
        super().__init__()
        self.controller = controller
        self.state = controller.state

        self.setWindowTitle("NegPy")
        self.resize(1400, 900)

        self._init_ui()
        self._connect_signals()
        self._setup_shortcuts()

    def _setup_shortcuts(self) -> None:
        """Defines global application hotkeys."""
        # Navigation
        QShortcut(
            QKeySequence(Qt.Key.Key_Left), self, self.controller.session.prev_file
        )
        QShortcut(
            QKeySequence(Qt.Key.Key_Right), self, self.controller.session.next_file
        )

        # Geometry
        QShortcut(QKeySequence("["), self, lambda: self.toolbar.rotate(1))
        QShortcut(QKeySequence("]"), self, lambda: self.toolbar.rotate(-1))
        QShortcut(QKeySequence("H"), self, lambda: self.toolbar.flip("horizontal"))
        QShortcut(QKeySequence("V"), self, lambda: self.toolbar.flip("vertical"))

        # Tools
        QShortcut(
            QKeySequence("W"), self, lambda: self.exposure_sidebar.pick_wb_btn.toggle()
        )
        QShortcut(
            QKeySequence("C"),
            self,
            lambda: self.geometry_sidebar.manual_crop_btn.toggle(),
        )
        QShortcut(
            QKeySequence("D"), self, lambda: self.retouch_sidebar.pick_dust_btn.toggle()
        )

        # Actions
        QShortcut(QKeySequence("Ctrl+E"), self, self.controller.request_export)
        QShortcut(QKeySequence("Ctrl+C"), self, self.controller.session.copy_settings)
        QShortcut(QKeySequence("Ctrl+V"), self, self.controller.session.paste_settings)

    def _init_ui(self) -> None:
        """Setup widgets and layout."""
        # Central Area
        self.central_widget = QWidget()
        self.central_layout = QVBoxLayout(self.central_widget)
        self.central_layout.setContentsMargins(0, 0, 0, 0)
        self.central_layout.setSpacing(0)

        self.top_status = TopStatusBar()
        self.canvas = ImageCanvas(self.state)
        self.toolbar = ActionToolbar(self.controller)

        self.central_layout.addWidget(self.top_status)
        self.central_layout.addWidget(self.canvas, stretch=1)
        self.central_layout.addWidget(self.toolbar)

        self.setCentralWidget(self.central_widget)

        self.drawer = QDockWidget("Controls", self)
        self.drawer.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea
        )

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setStyleSheet("QScrollArea { border: none; }")

        self.sidebar_widget = QWidget()
        self.sidebar_layout = QVBoxLayout(self.sidebar_widget)
        self.sidebar_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.sidebar_layout.setContentsMargins(0, 0, 0, 0)
        self.sidebar_layout.setSpacing(1)

        self._init_sidebar_panels()

        self.scroll.setWidget(self.sidebar_widget)
        self.drawer.setWidget(self.scroll)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.drawer)

        self.session_dock = QDockWidget("Session", self)
        self.files_sidebar = FilesSidebar(self.controller)
        self.session_dock.setWidget(self.files_sidebar)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.session_dock)

        # hide status bar - we use TopStatusBar instead
        self.setStatusBar(QStatusBar())
        self.statusBar().hide()

    def _init_sidebar_panels(self) -> None:
        """Initialize and add all sidebar sections."""
        icon_color = "#aaa"

        self.analysis_sidebar = AnalysisSidebar(self.controller)
        self._add_sidebar_section(
            "Analysis",
            "analysis",
            self.analysis_sidebar,
            icon=qta.icon("fa5s.chart-bar", color=icon_color),
        )

        self.presets_sidebar = PresetsSidebar(self.controller)
        self._add_sidebar_section(
            "Presets",
            "presets",
            self.presets_sidebar,
            icon=qta.icon("fa5s.magic", color=icon_color),
        )

        self.exposure_sidebar = ExposureSidebar(self.controller)
        self._add_sidebar_section(
            "Exposure",
            "exposure",
            self.exposure_sidebar,
            icon=qta.icon("fa5s.sun", color=icon_color),
        )

        self.geometry_sidebar = GeometrySidebar(self.controller)
        self._add_sidebar_section(
            "Geometry",
            "geometry",
            self.geometry_sidebar,
            icon=qta.icon("fa5s.crop", color=icon_color),
        )

        self.lab_sidebar = LabSidebar(self.controller)
        self._add_sidebar_section(
            "Lab",
            "lab",
            self.lab_sidebar,
            icon=qta.icon("fa5s.flask", color=icon_color),
        )

        self.toning_sidebar = ToningSidebar(self.controller)
        self._add_sidebar_section(
            "Toning",
            "toning",
            self.toning_sidebar,
            icon=qta.icon("fa5s.tint", color=icon_color),
        )

        self.retouch_sidebar = RetouchSidebar(self.controller)
        self._add_sidebar_section(
            "Retouch",
            "retouch",
            self.retouch_sidebar,
            icon=qta.icon("fa5s.brush", color=icon_color),
        )

        self.metadata_sidebar = MetadataSidebar(self.controller)
        self._add_sidebar_section(
            "Metadata",
            "metadata",
            self.metadata_sidebar,
            icon=qta.icon("fa5s.info-circle", color=icon_color),
        )

        self.icc_sidebar = ICCSidebar(self.controller)
        self._add_sidebar_section(
            "ICC",
            "icc",
            self.icc_sidebar,
            icon=qta.icon("fa5s.eye", color=icon_color),
        )

        self.export_sidebar = ExportSidebar(self.controller)
        self._add_sidebar_section(
            "Export",
            "export",
            self.export_sidebar,
            icon=qta.icon("fa5s.download", color=icon_color),
        )

    def _add_sidebar_section(
        self, title: str, key: str, widget: QWidget, icon=None
    ) -> None:
        """Helper to create and add a collapsible section."""
        is_expanded = THEME.sidebar_expanded_defaults.get(key, False)
        if key in ["exposure", "geometry", "lab", "retouch", "export", "analysis"]:
            is_expanded = THEME.sidebar_expanded_defaults.get(key, True)

        section = CollapsibleSection(title, expanded=is_expanded, icon=icon)
        section.set_content(widget)
        self.sidebar_layout.addWidget(section)

    def _connect_signals(self) -> None:
        """Wire controller and view."""
        self.controller.image_updated.connect(self._on_image_updated)
        self.controller.image_updated.connect(self._refresh_image_info)
        self.canvas.clicked.connect(self.controller.handle_canvas_clicked)
        self.canvas.crop_completed.connect(self.controller.handle_crop_completed)

        self.controller.export_progress.connect(self._on_export_progress)
        self.controller.export_finished.connect(self._on_export_finished)
        self.controller.tool_sync_requested.connect(self._sync_tool_buttons)
        self.controller.config_updated.connect(self._sync_all_sidebars)

    def _sync_all_sidebars(self) -> None:
        """Force all sidebar panels to update their widgets from current AppState."""
        self.exposure_sidebar.sync_ui()
        self.geometry_sidebar.sync_ui()
        self.lab_sidebar.sync_ui()
        self.toning_sidebar.sync_ui()
        self.retouch_sidebar.sync_ui()
        self.metadata_sidebar.sync_ui()
        self.icc_sidebar.sync_ui()
        self.export_sidebar.sync_ui()
        self.presets_sidebar.sync_ui()

    def _on_image_updated(self) -> None:
        """Refreshes canvas when a new render pass completes."""
        metrics = self.state.last_metrics
        if "base_positive" not in metrics:
            print("DEBUG: 'base_positive' NOT FOUND in metrics!")
            return

        buffer = metrics["base_positive"]
        content_rect = None

        # Apply border preview if enabled
        export_conf = self.state.config.export
        if export_conf.export_border_size > 0:
            pil_img = Image.fromarray(float_to_uint8(buffer))
            try:
                pil_img, content_rect = PrintService.apply_preview_layout_to_pil(
                    pil_img,
                    export_conf.paper_aspect_ratio,
                    export_conf.export_border_size,
                    export_conf.export_print_size,
                    export_conf.export_border_color,
                    1200.0,  # Reference size
                )
                buffer = np.array(pil_img).astype(np.float32) / 255.0
            except Exception as e:
                print(f"DEBUG: Border preview error: {e}")

        self.canvas.update_buffer(
            buffer, self.state.workspace_color_space, content_rect=content_rect
        )

    def _refresh_image_info(self) -> None:
        """Updates the canvas metadata overlay."""
        if not self.state.current_file_path:
            self.canvas.update_overlay("No File", "- x - px", "", "")
            return

        filename = os.path.basename(self.state.current_file_path)
        w, h = self.state.original_res
        res_str = f"{w} x {h} px"
        cs = self.state.workspace_color_space

        # Suggested bottom right: Bit Depth & Process Mode
        mode = f"16-bit | {self.state.config.process_mode}"

        self.canvas.update_overlay(filename, res_str, cs, mode)

    def _on_canvas_clicked(self, nx: float, ny: float) -> None:
        self.top_status.showMessage(f"Clicked at: {nx:.3f}, {ny:.3f}")

    def _on_export_progress(self, current: int, total: int, filename: str) -> None:
        self.top_status.progress.setVisible(True)
        self.top_status.progress.setRange(0, total)
        self.top_status.progress.setValue(current)
        self.top_status.showMessage(f"Exporting {filename} ({current}/{total})...")

    def _on_export_finished(self) -> None:
        self.top_status.progress.setVisible(False)
        self.top_status.showMessage("Export Complete", 5000)

    def _sync_tool_buttons(self) -> None:
        """Updates toggle button states to match active_tool."""
        mode = self.state.active_tool
        self.canvas.set_tool_mode(mode)
        self.geometry_sidebar.sync_ui()
        self.exposure_sidebar.pick_wb_btn.setChecked(mode == ToolMode.WB_PICK)
        self.geometry_sidebar.manual_crop_btn.setChecked(mode == ToolMode.CROP_MANUAL)
        self.retouch_sidebar.pick_dust_btn.setChecked(mode == ToolMode.DUST_PICK)
