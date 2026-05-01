import os

import numpy as np
from PIL import Image
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QDockWidget,
    QMainWindow,
    QScrollArea,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from negpy.desktop.controller import AppController
from negpy.desktop.session import ToolMode
from negpy.infrastructure.loaders.constants import SUPPORTED_RAW_EXTENSIONS
from negpy.desktop.view.canvas.toolbar import ActionToolbar
from negpy.desktop.view.canvas.widget import ImageCanvas
from negpy.desktop.view.keyboard_shortcuts import setup_keyboard_shortcuts
from negpy.desktop.view.sidebar.controls_panel import ControlsPanel
from negpy.desktop.view.sidebar.session_panel import SessionPanel
from negpy.desktop.view.styles.theme import THEME
from negpy.desktop.view.widgets.overlays import ImageMetadataPanel
from negpy.desktop.view.widgets.status_bar import TopStatusBar
from negpy.desktop.view.widgets.toast import Toast
from negpy.domain.models import AspectRatio
from negpy.infrastructure.gpu.resources import GPUTexture
from negpy.kernel.image.logic import float_to_uint8
from negpy.kernel.system.config import APP_CONFIG
from negpy.kernel.system.logging import get_logger
from negpy.services.export.print import PrintService

logger = get_logger(__name__)


def _display_buffer_for_canvas(buffer: object) -> object:
    if isinstance(buffer, GPUTexture):
        try:
            readback = buffer.readback()
        except Exception:
            logger.exception("Failed to read back GPU preview for canvas display")
            return buffer

        if isinstance(readback, np.ndarray) and readback.ndim == 3 and readback.shape[2] >= 3:
            return np.ascontiguousarray(readback[:, :, :3])
        return readback

    return buffer


class _EmptyStateOverlay(QWidget):
    """Shown on top of the canvas when no image is loaded."""

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        from PyQt6.QtWidgets import QLabel

        label = QLabel("Load some scans to get started")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setStyleSheet(f"color: {THEME.text_muted}; font-size: 15px;")
        layout.addWidget(label)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self.parent():
            self.setGeometry(self.parent().rect())


class MainWindow(QMainWindow):
    """
    Main application window hosting the canvas, sidebar, and asset browser.
    """

    def __init__(self, controller: AppController):
        super().__init__()
        self.controller = controller
        self.state = controller.state

        self.resize(1400, 900)
        self.setAcceptDrops(True)

        self._init_ui()
        self._connect_signals()
        self.shortcut_manager = setup_keyboard_shortcuts(self)
        self._update_title()

    def _init_ui(self) -> None:
        """Setup widgets and layout."""
        # Main Window Padding
        self.setContentsMargins(8, 8, 8, 8)

        # Central Area
        self.central_widget = QWidget()
        self.central_layout = QVBoxLayout(self.central_widget)
        self.central_layout.setContentsMargins(0, 0, 0, 0)
        self.central_layout.setSpacing(4)

        self.top_status = TopStatusBar()
        self.metadata_top = ImageMetadataPanel()
        self.canvas = ImageCanvas(self.state)
        self.metadata_bottom = ImageMetadataPanel()

        self.controller.register_canvas(self.canvas)
        self.canvas.set_controller(self.controller)
        self.toolbar = ActionToolbar(self.controller)

        self.empty_state = _EmptyStateOverlay(self.canvas)
        self.empty_state.raise_()

        self.central_layout.addWidget(self.top_status)
        self.central_layout.addWidget(self.metadata_top)
        self.central_layout.addWidget(self.canvas, stretch=1)
        self.central_layout.addWidget(self.metadata_bottom)
        self.central_layout.addWidget(self.toolbar)

        self.setCentralWidget(self.central_widget)

        self.drawer = QDockWidget("Controls", self)
        self.drawer.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea)

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

        self.toast = Toast(self)

    def _update_title(self) -> None:
        state = self.controller.session.state
        if state.current_file_path:
            filename = os.path.basename(state.current_file_path)
            prefix = "● " if state.is_dirty else ""
            self.setWindowTitle(f"{prefix}NegPy — {filename}")
        else:
            self.setWindowTitle("NegPy")

    def _connect_signals(self) -> None:
        """Wire controller and view."""
        self.controller.session.state_changed.connect(self._update_title)
        self.controller.image_updated.connect(self._on_image_updated)
        self.controller.preview_loaded.connect(self._refresh_image_info)
        self.controller.loading_started.connect(self.canvas.clear)
        self.controller.loading_started.connect(lambda: self.empty_state.setVisible(False))

        # Metadata updates only on persistent history changes or file selection
        self.controller.session.history_changed.connect(self._refresh_image_info)
        self.controller.session.file_selected.connect(lambda _: self._refresh_image_info())

        self.canvas.clicked.connect(self.controller.handle_canvas_clicked)
        self.canvas.crop_completed.connect(self.controller.handle_crop_completed)

        self.controller.export_progress.connect(self._on_export_progress)
        self.controller.export_finished.connect(self._on_export_finished)
        self.controller.session.settings_copied.connect(lambda: self.toast.show_text("Settings copied"))
        self.controller.session.settings_pasted.connect(lambda: self.toast.show_text("Settings pasted"))
        self.controller.tool_sync_requested.connect(self._sync_tool_buttons)
        self.controller.config_updated.connect(self.canvas.overlay.update)

        self.controller.status_message_requested.connect(self.top_status.showMessage)
        self.controller.status_progress_requested.connect(self.top_status.set_progress)
        self.controller.pixel_readout.connect(self.canvas.pixel_readout_overlay.set_values)

        self.dash_timer = QTimer(self)
        self.dash_timer.timeout.connect(self._refresh_dashboard)
        self.dash_timer.start(2000)
        self._refresh_dashboard()

    def _refresh_dashboard(self) -> None:
        from negpy.desktop.view.styles.theme import THEME

        header = self.session_panel.header
        if self.state.gpu_enabled:
            backend = self.controller.render_worker.processor.backend_name
            header.gpu_badge.setText(backend.upper())
            header.gpu_badge.setStyleSheet(f"color: {THEME.text_secondary}; font-size: 11px; font-weight: bold;")
        else:
            header.gpu_badge.setText("CPU")
            header.gpu_badge.setStyleSheet(f"color: {THEME.text_secondary}; font-size: 11px; font-weight: bold;")

    def _on_image_updated(self) -> None:
        """Refreshes canvas when a new render pass completes."""
        self.empty_state.setVisible(False)
        metrics = self.state.last_metrics
        if "base_positive" not in metrics:
            logger.warning("Render completed but 'base_positive' not found in metrics")
            return

        buffer = _display_buffer_for_canvas(metrics["base_positive"])
        content_rect = metrics.get("content_rect")

        if isinstance(buffer, np.ndarray):
            export_conf = self.state.config.export
            should_preview = export_conf.export_border_size > 0 or export_conf.paper_aspect_ratio != AspectRatio.ORIGINAL

            if should_preview:
                pil_img = Image.fromarray(float_to_uint8(buffer))
                try:
                    pil_img, content_rect = PrintService.apply_preview_layout_to_pil(
                        pil_img,
                        export_conf.paper_aspect_ratio,
                        export_conf.export_border_size,
                        export_conf.export_print_size,
                        export_conf.export_border_color,
                        APP_CONFIG.preview_render_size,
                    )
                    buffer = np.array(pil_img).astype(np.float32) / 255.0
                except Exception as e:
                    logger.error(f"Border preview failure: {e}")

        self.canvas.update_buffer(buffer, self.state.workspace_color_space, content_rect=content_rect)

    def _refresh_image_info(self) -> None:
        """Updates the persistent metadata panels."""
        if not self.state.current_file_path:
            self.metadata_top.update_values("No File", "- x - px")
            self.metadata_bottom.update_values("Edits: 0", "16-bit")
            return

        filename = os.path.basename(self.state.current_file_path)
        w, h = self.state.original_res
        res_str = f"{w} x {h} px"

        mode_str = f"16-bit | {self.state.config.process.process_mode}"
        edits_str = f"Edits: {self.state.undo_index}"

        self.metadata_top.update_values(filename, res_str)
        self.metadata_bottom.update_values(edits_str, mode_str)

    def _on_canvas_clicked(self, nx: float, ny: float) -> None:
        self.top_status.showMessage(f"Clicked at: {nx:.3f}, {ny:.3f}")

    def _on_export_progress(self, current: int, total: int, filename: str) -> None:
        self.top_status.progress.setVisible(True)
        self.top_status.progress.setRange(0, total)
        self.top_status.progress.setValue(current)
        self.top_status.showMessage(f"Exporting {filename} ({current}/{total})...")

    def _on_export_finished(self, elapsed: float) -> None:
        self.top_status.progress.setVisible(False)
        self.toast.show_text(f"Export complete in {elapsed:.2f}s", duration_ms=3000)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if hasattr(self, "toast"):
            self.toast._reposition()
        if hasattr(self, "empty_state"):
            self.empty_state.setGeometry(self.canvas.rect())

    def _sync_tool_buttons(self) -> None:
        """Updates toggle button states to match active_tool."""
        mode = self.state.active_tool
        self.canvas.set_tool_mode(mode)

        # We access buttons through the controls panel
        self.controls_panel.exposure_sidebar.pick_wb_btn.setChecked(mode == ToolMode.WB_PICK)
        self.controls_panel.geometry_sidebar.manual_crop_btn.setChecked(mode == ToolMode.CROP_MANUAL)
        self.controls_panel.retouch_sidebar.pick_dust_btn.setChecked(mode == ToolMode.DUST_PICK)

    def dragEnterEvent(self, event) -> None:
        if event.mimeData().hasUrls():
            paths = [u.toLocalFile() for u in event.mimeData().urls()]
            if any(os.path.splitext(p)[1].lower() in SUPPORTED_RAW_EXTENSIONS or os.path.isdir(p) for p in paths):
                event.acceptProposedAction()
                return
        event.ignore()

    def dropEvent(self, event) -> None:
        paths = [u.toLocalFile() for u in event.mimeData().urls()]
        if paths:
            self.controller.request_asset_discovery(paths)
        event.acceptProposedAction()
