import os
from PyQt6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QDockWidget,
    QStatusBar,
    QScrollArea,
)
from PyQt6.QtCore import Qt
from PIL import Image
import numpy as np

from src.desktop.view.canvas.widget import ImageCanvas
from src.desktop.view.canvas.toolbar import ActionToolbar
from src.desktop.view.sidebar.session_panel import SessionPanel
from src.desktop.view.sidebar.controls_panel import ControlsPanel
from src.desktop.view.widgets.status_bar import TopStatusBar
from src.desktop.view.keyboard_shortcuts import setup_keyboard_shortcuts
from src.desktop.controller import AppController
from src.desktop.session import ToolMode
from src.services.export.print import PrintService
from src.kernel.image.logic import float_to_uint8
from src.domain.models import AspectRatio


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
        setup_keyboard_shortcuts(self)

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

        self.controls_panel = ControlsPanel(self.controller)

        self.scroll.setWidget(self.controls_panel)
        self.drawer.setWidget(self.scroll)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.drawer)

        self.session_dock = QDockWidget("Session", self)
        self.session_panel = SessionPanel(self.controller)
        self.session_dock.setWidget(self.session_panel)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.session_dock)

        # hide status bar - we use TopStatusBar instead
        self.setStatusBar(QStatusBar())
        self.statusBar().hide()

    def _connect_signals(self) -> None:
        """Wire controller and view."""
        self.controller.image_updated.connect(self._on_image_updated)
        self.controller.image_updated.connect(self._refresh_image_info)
        self.canvas.clicked.connect(self.controller.handle_canvas_clicked)
        self.canvas.crop_completed.connect(self.controller.handle_crop_completed)

        self.controller.export_progress.connect(self._on_export_progress)
        self.controller.export_finished.connect(self._on_export_finished)
        self.controller.tool_sync_requested.connect(self._sync_tool_buttons)

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
        should_preview = (
            export_conf.export_border_size > 0
            or export_conf.paper_aspect_ratio != AspectRatio.ORIGINAL
        )

        if should_preview:
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

        # We access buttons through the controls panel
        self.controls_panel.exposure_sidebar.pick_wb_btn.setChecked(
            mode == ToolMode.WB_PICK
        )
        self.controls_panel.geometry_sidebar.manual_crop_btn.setChecked(
            mode == ToolMode.CROP_MANUAL
        )
        self.controls_panel.retouch_sidebar.pick_dust_btn.setChecked(
            mode == ToolMode.DUST_PICK
        )
