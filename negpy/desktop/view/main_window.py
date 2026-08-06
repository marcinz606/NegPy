import os
from typing import Optional

import numpy as np
from PIL import Image
from PyQt6.QtCore import Qt, QEvent, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication,
    QDialog,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
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
from negpy.desktop.view.shortcut_registry import tooltip_with_shortcut
from negpy.desktop.view.sidebar.right_panel import RightPanel
from negpy.desktop.view.sidebar.session_panel import SessionPanel
from negpy.desktop.view.styles.theme import THEME
from negpy.desktop.view.widgets.loading_overlay import LoadingOverlay
from negpy.desktop.view.widgets.pinnable_dock import PinnableDockWidget
from negpy.desktop.view.widgets.progress_dialog import ProgressDialog
from negpy.domain.models import AspectRatio
from negpy.infrastructure.gpu.resources import GPUTexture
from negpy.kernel.image.logic import float_to_uint8
from negpy.kernel.system.config import APP_CONFIG
from negpy.kernel.system.logging import get_logger
from negpy.services.export.print import PrintService

logger = get_logger(__name__)

_DEFAULT_W, _DEFAULT_H = 1400, 900


def _clamp_geometry(
    saved: Optional[tuple[int, int, int, int]],
    avail: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    """Fit a window geometry inside the available screen rect.

    ``saved`` is (x, y, w, h) or None (use the default size, centered).
    ``avail`` is (x, y, w, h) of the screen work area. The result is sized no
    larger than ``avail`` and positioned fully inside it.
    """
    ax, ay, aw, ah = avail
    if saved is None:
        w, h = min(_DEFAULT_W, aw), min(_DEFAULT_H, ah)
        return ax + (aw - w) // 2, ay + (ah - h) // 2, w, h
    sx, sy, sw, sh = saved
    w, h = min(sw, aw), min(sh, ah)
    x = min(max(sx, ax), ax + aw - w)
    y = min(max(sy, ay), ay + ah - h)
    return x, y, w, h


def _read_screen_icc(screen: object) -> Optional[bytes]:
    """Monitor ICC profile bytes for a QScreen, or None (treat the display as sRGB).

    Detection is per-OS (colord / ColorSync / PIL); see ``detect_monitor_icc``.
    """
    from negpy.infrastructure.display.monitor_profile import detect_monitor_icc

    data = detect_monitor_icc(screen)
    if not data:
        logger.warning("No monitor ICC profile detected; preview will assume sRGB")
    return data


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
    """Shown on top of the canvas when no image is loaded.

    Tracks the canvas rather than the window: hiding a dock resizes the canvas
    without resizing the window, which used to leave this stranded off-centre
    while the floating toolbar (which the canvas lays out) moved with it.
    """

    add_files_requested = pyqtSignal()
    add_folder_requested = pyqtSignal()
    tour_requested = pyqtSignal()

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        parent.installEventFilter(self)

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(12)

        self.load_btn = QPushButton("Load some scans to get started")
        self.load_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.load_btn.setStyleSheet(
            f"QPushButton {{ background: transparent; color: {THEME.text_muted}; "
            f"border: none; font-size: 15px; padding: 4px 8px; }}"
            f"QPushButton:hover {{ color: {THEME.text_primary}; text-decoration: underline; }}"
        )
        self.load_btn.clicked.connect(self._show_load_menu)
        layout.addWidget(self.load_btn, alignment=Qt.AlignmentFlag.AlignHCenter)

        self.tour_btn = QPushButton("Take the tour")
        self.tour_btn.setFixedWidth(140)
        self.tour_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.tour_btn.setStyleSheet(
            f"QPushButton {{ background: transparent; color: {THEME.text_muted}; "
            f"border: 1px solid {THEME.border_primary}; border-radius: 3px; "
            f"padding: 5px 14px; font-size: 12px; }}"
            f"QPushButton:hover {{ color: {THEME.text_primary}; }}"
        )
        self.tour_btn.clicked.connect(self.tour_requested)
        layout.addWidget(self.tour_btn, alignment=Qt.AlignmentFlag.AlignHCenter)

    def _show_load_menu(self) -> None:
        menu = QMenu(self)
        menu.addAction("Add files…").triggered.connect(self.add_files_requested)
        menu.addAction("Add folder…").triggered.connect(self.add_folder_requested)
        menu.exec(self.load_btn.mapToGlobal(self.load_btn.rect().bottomLeft()))

    def eventFilter(self, obj, event) -> bool:
        if obj is self.parent() and event.type() == QEvent.Type.Resize:
            self.setGeometry(self.parent().rect())
        return False

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

        self._restore_window_geometry()
        self.setAcceptDrops(True)

        self._init_ui()
        self._connect_signals()
        self.shortcut_manager = setup_keyboard_shortcuts(self)
        self._update_title()

        from negpy.desktop.view.widgets.tutorial_overlay import TutorialOverlay

        self.tutorial_overlay = TutorialOverlay(self)
        self.tutorial_overlay.finished.connect(self._on_tutorial_finished)

        repo = self.controller.session.repo
        if not repo.get_global_setting("tutorial_seen", False):
            QTimer.singleShot(600, self.show_tutorial)  # the scan-setup wizard follows the tour
        elif repo.get_global_setting("scan_setup") is None:
            QTimer.singleShot(600, self.show_scan_setup)

    def _restore_window_geometry(self) -> None:
        """Open clamped to the screen work area, restoring the saved size/position if any."""
        screen = QApplication.primaryScreen()
        if screen is None:
            self.resize(_DEFAULT_W, _DEFAULT_H)
            return
        raw = self.controller.session.repo.get_global_setting("window_geometry")
        saved = None
        if isinstance(raw, list) and len(raw) == 4 and raw[2] > 0 and raw[3] > 0:
            saved = tuple(int(v) for v in raw)
        rect = screen.availableGeometry()
        x, y, w, h = _clamp_geometry(saved, (rect.x(), rect.y(), rect.width(), rect.height()))
        self.resize(w, h)
        self.move(x, y)

    def closeEvent(self, event) -> None:
        try:
            self.controller.session.repo.save_global_setting("window_geometry", [self.x(), self.y(), self.width(), self.height()])
        except Exception:
            logger.exception("Failed to persist window geometry")
        super().closeEvent(event)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        # The window handle (and thus its screen) only exists once shown. Wire
        # monitor-profile detection once, then track screen changes.
        if not getattr(self, "_monitor_wired", False):
            self._monitor_wired = True
            handle = self.windowHandle()
            if handle is not None:
                handle.screenChanged.connect(lambda _screen: self._refresh_monitor_profile())
            # force=True so a persisted override is resolved even when detection is None.
            self._refresh_monitor_profile(force=True)

        if not getattr(self, "_session_restore_checked", False):
            self._session_restore_checked = True
            QTimer.singleShot(0, self._maybe_restore_session)

    def _maybe_restore_session(self) -> None:
        """Offers to reopen the previous session's files on first show, and otherwise
        opens the library — with nothing loaded, a list of rolls beats a blank panel."""
        paths = self.controller.saved_session_paths()
        if paths:
            reply = QMessageBox.question(
                self,
                "Restore Session",
                f"Reopen your last session ({len(paths)} file(s))?",
            )
            if reply == QMessageBox.StandardButton.Yes:
                self.controller.restore_session()
                return
        self.session_panel.show_library(ask_if_unset=False)

    def _refresh_monitor_profile(self, force: bool = False) -> None:
        """Detect the active screen's ICC profile and hand it to the controller, which
        resolves it against any manual override and pushes it to the display paths."""
        handle = self.windowHandle()
        screen = handle.screen() if handle is not None else self.screen()
        data = _read_screen_icc(screen) if screen is not None else None
        if not force and data == self.state.monitor_icc_detected_bytes:
            return
        self.controller.set_monitor_detected(data)

    def _init_ui(self) -> None:
        """Setup widgets and layout."""
        # Main Window Padding
        self.setContentsMargins(8, 8, 8, 8)

        # Central Area
        self.central_widget = QWidget()
        self.central_layout = QVBoxLayout(self.central_widget)
        self.central_layout.setContentsMargins(0, 0, 0, 0)
        self.central_layout.setSpacing(4)

        self.canvas = ImageCanvas(self.state)

        self.controller.register_canvas(self.canvas)
        self.canvas.set_controller(self.controller)
        self.toolbar = ActionToolbar(self.controller)
        self.canvas.set_floating_toolbar(self.toolbar)

        self.empty_state = _EmptyStateOverlay(self.canvas)
        self.empty_state.tour_requested.connect(self.show_tutorial)
        # session_panel is built further down; resolve the browser lazily.
        self.empty_state.add_files_requested.connect(lambda: self.session_panel.file_browser.prompt_add_files())
        self.empty_state.add_folder_requested.connect(lambda: self.session_panel.file_browser.prompt_add_folder())
        self.empty_state.raise_()

        self.loading_overlay = LoadingOverlay(self.canvas)
        self.loading_overlay.raise_()

        self.central_layout.addWidget(self.canvas, stretch=1)

        self.setCentralWidget(self.central_widget)

        self.drawer = PinnableDockWidget(
            "Controls",
            self,
            pin_tooltip=tooltip_with_shortcut("Dock controls panel to right", "toggle_right_panel"),
            on_pin=self.dock_controls_panel,
        )
        self.drawer.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea)

        self.right_panel = RightPanel(self.controller)
        # Back-compat alias: tutorial, keyboard shortcuts, and _sync_tool_buttons reach feature sidebars here.
        self.controls_panel = self.right_panel.controls_panel

        # Object names are required for saveState/restoreState to identify the docks.
        self.drawer.setObjectName("controls_dock")
        self.drawer.setWidget(self.right_panel)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.drawer)

        self.session_dock = PinnableDockWidget(
            "Session",
            self,
            pin_tooltip=tooltip_with_shortcut("Dock session panel to left", "toggle_left_panel"),
            on_pin=self.dock_session_panel,
        )
        self.session_dock.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea)
        self.session_dock.setObjectName("session_dock")
        self.session_panel = SessionPanel(self.controller)
        self.session_dock.setWidget(self.session_panel)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.session_dock)

        # Snapshot the pristine layout (both docked at their home edges, default
        # widths) so the pin and Reset Panel Layout can restore the true original
        # position and size, not merely re-add the dock where it currently sits.
        self._default_dock_state = self.saveState()

        # Restore saved panel visibility
        repo = self.controller.session.repo
        self.session_dock.setVisible(repo.get_global_setting("panel_left_visible", True))
        self.drawer.setVisible(repo.get_global_setting("panel_right_visible", True))

        # hide native status bar - status lives in the canvas HUD
        self.setStatusBar(QStatusBar())
        self.statusBar().hide()

    TOOL_LABELS: dict[ToolMode, str] = {
        ToolMode.WB_PICK: "WB Picker",
        ToolMode.CROP_MANUAL: "Crop",
        ToolMode.DUST_PICK: "Heal Tool",
    }

    def _update_title(self) -> None:
        state = self.controller.session.state
        if state.current_file_path:
            filename = os.path.basename(state.current_file_path)
            prefix = "● " if state.is_dirty else ""
            tool = self.TOOL_LABELS.get(state.active_tool)
            tool_prefix = f"[{tool}] " if tool else ""
            self.setWindowTitle(f"{prefix}NegPy — {tool_prefix}{filename}")
        else:
            self.setWindowTitle("NegPy")

    def _on_immersive_changed(self) -> None:
        if getattr(self, "_last_immersive", None) != self.controller.session.state.immersive_canvas:
            self._last_immersive = self.controller.session.state.immersive_canvas
            self.canvas.fit_to_window()

    def show_tutorial(self) -> None:
        from negpy.desktop.view.widgets.tutorial_steps import build

        self.tutorial_overlay.start(build(self))

    def _on_tutorial_finished(self, _completed: bool) -> None:
        repo = self.controller.session.repo
        repo.save_global_setting("tutorial_seen", True)
        # Unset only until the wizard is answered once, so replaying the tour later
        # from the ⋯ menu doesn't ask again.
        if repo.get_global_setting("scan_setup") is None:
            QTimer.singleShot(0, self.show_scan_setup)

    def show_scan_setup(self) -> None:
        from negpy.desktop.view.widgets.scan_setup_dialog import ScanSetupDialog

        repo = self.controller.session.repo
        dlg = ScanSetupDialog(self, repo.get_global_setting("scan_setup"))
        # Cancel leaves scan_setup unset: an unanswered wizard is not an answer, so it asks again.
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self.controller.apply_scan_setup(**dlg.choice())

    def _restore_default_dock_state(self) -> None:
        """Restore both docks to the snapshot taken at startup (home edge + width)."""
        if self._default_dock_state is not None:
            self.restoreState(self._default_dock_state)

    def dock_session_panel(self) -> None:
        # Restore the pristine layout (returns this panel to its original edge and
        # width) but preserve the other panel's current visibility so pinning one
        # never forces the other to reappear.
        right_visible = self.drawer.isVisible()
        self._restore_default_dock_state()
        self.drawer.setVisible(right_visible)
        self.session_dock.setVisible(True)
        self.toolbar.btn_toggle_left.setChecked(True)
        self.controller.session.repo.save_global_setting("panel_left_visible", True)

    def dock_controls_panel(self) -> None:
        left_visible = self.session_dock.isVisible()
        self._restore_default_dock_state()
        self.session_dock.setVisible(left_visible)
        self.drawer.setVisible(True)
        self.toolbar.btn_toggle_right.setChecked(True)
        self.controller.session.repo.save_global_setting("panel_right_visible", True)

    def toggle_session_dock(self) -> None:
        if self.session_dock.isFloating():
            self.dock_session_panel()
            return
        visible = not self.session_dock.isVisible()
        self.session_dock.setVisible(visible)
        self.controller.session.repo.save_global_setting("panel_left_visible", visible)

    def toggle_controls_dock(self) -> None:
        if self.drawer.isFloating():
            self.dock_controls_panel()
            return
        visible = not self.drawer.isVisible()
        self.drawer.setVisible(visible)
        self.controller.session.repo.save_global_setting("panel_right_visible", visible)

    def reset_panel_layout(self) -> None:
        """Restore both side panels to their original edges, widths and visibility."""
        self._restore_default_dock_state()
        self.session_dock.setVisible(True)
        self.drawer.setVisible(True)
        self.toolbar.btn_toggle_left.setChecked(True)
        self.toolbar.btn_toggle_right.setChecked(True)
        self.controller.session.repo.save_global_setting("panel_left_visible", True)
        self.controller.session.repo.save_global_setting("panel_right_visible", True)
        self.canvas.hud.showMessage("panel layout reset", timeout=1500)

    def _connect_signals(self) -> None:
        """Wire controller and view."""
        self.controller.session.state_changed.connect(self._update_title)
        self.controller.session.state_changed.connect(self._on_immersive_changed)

        # visibilityChanged only mirrors the button — it also fires on close/minimize,
        # so we persist in the toggle methods to avoid clobbering the saved state on exit.
        self.toolbar.btn_toggle_left.clicked.connect(self.toggle_session_dock)
        self.toolbar.btn_toggle_right.clicked.connect(self.toggle_controls_dock)
        self.session_dock.visibilityChanged.connect(self.toolbar.btn_toggle_left.setChecked)
        self.drawer.visibilityChanged.connect(self.toolbar.btn_toggle_right.setChecked)
        self.toolbar.btn_toggle_left.setChecked(self.session_dock.isVisible())
        self.toolbar.btn_toggle_right.setChecked(self.drawer.isVisible())
        self.controller.image_updated.connect(self._on_image_updated)
        self.controller.preview_loaded.connect(self._refresh_image_info)
        self.controller.loading_started.connect(self._on_loading_started)
        self.controller.image_updated.connect(self.loading_overlay.stop)
        self.controller.load_failed.connect(self._on_load_failed)
        self.controller.zoom_changed.connect(self._on_zoom_info_changed)

        # Metadata updates only on persistent history changes or file selection
        self.controller.session.history_changed.connect(self._refresh_image_info)
        self.controller.session.file_selected.connect(lambda _: self._refresh_image_info())
        self.controller.session.session_emptied.connect(self._on_session_emptied)

        self.canvas.clicked.connect(self.controller.handle_canvas_clicked)
        self.canvas.crop_rect_changed.connect(self.controller.handle_crop_rect_changed)
        self.canvas.crop_rotation_changed.connect(self.controller.handle_crop_rotation_changed)
        self.canvas.crop_confirmed.connect(self.controller.confirm_manual_crop)
        self.canvas.analysis_rect_changed.connect(self.controller.handle_analysis_rect_changed)
        self.canvas.analysis_confirmed.connect(self.controller.confirm_analysis_region)
        self.canvas.lasso_completed.connect(self.controller.handle_lasso_completed)
        self.canvas.scratch_completed.connect(self.controller.handle_heal_stroke_completed)
        self.canvas.straighten_completed.connect(self.controller.handle_straighten_completed)
        self.canvas.zone_pin_moved.connect(self.controller.move_zone_pin)
        self.canvas.zone_placement_confirmed.connect(self.controller.apply_zone_placement)
        self.canvas.local_mask_selected.connect(self.controller.select_local_mask)
        self.canvas.local_mask_edited.connect(self.controller.handle_local_mask_edited)
        self.canvas.local_vertex_deleted.connect(self.controller.delete_local_vertex)

        self.controller.export_progress.connect(self._on_export_progress)
        self.controller.export_finished.connect(self._on_export_finished)
        self.controller.session.settings_copied.connect(lambda: self.canvas.hud.showMessage("settings copied", timeout=1500))
        self.controller.session.settings_pasted.connect(lambda: self.canvas.hud.showMessage("settings pasted", timeout=1500))
        self.controller.session.settings_synced.connect(lambda msg: self.canvas.hud.showMessage(msg, timeout=2500))
        self.controller.tool_sync_requested.connect(self._sync_tool_buttons)
        self.controller.config_updated.connect(self.canvas.overlay.update)
        self.controller.analysis_buffer_preview_requested.connect(self.canvas.overlay.show_analysis_buffer)
        self.controller.rotation_guide_requested.connect(self.canvas.overlay.show_rotation_grid)
        self.controller.crop_guide_changed.connect(self.canvas.overlay.update)
        self.controller.dust_overlay_changed.connect(self.canvas.overlay.update)
        self.controller.zones_overlay_changed.connect(lambda _on: self.canvas.overlay.update())
        self.controller.grain_focuser_changed.connect(lambda _on: self.canvas.overlay.update())
        self.controller.test_strip_changed.connect(lambda _up: self.canvas.overlay.on_test_strip_changed())
        self.canvas.test_strip_picked.connect(self.controller.apply_test_strip_pick)
        self.controller.zone_pins_changed.connect(self.canvas.overlay.update)
        self.controller.zone_arm_changed.connect(self._on_zone_armed)

        self.controller.status_message_requested.connect(self.canvas.hud.showMessage)
        self.controller.status_progress_requested.connect(self.canvas.hud.set_progress)

        self.progress_dialog = ProgressDialog(self)
        self.controller.batch_started.connect(self.progress_dialog.start)
        self.controller.batch_progress.connect(self.progress_dialog.set_progress)
        self.controller.batch_finished.connect(self.progress_dialog.finish)
        self.progress_dialog.abort_requested.connect(self.controller.abort_active_batch)

        self.dash_timer = QTimer(self)
        self.dash_timer.timeout.connect(self._refresh_dashboard)
        self.dash_timer.start(2000)
        self._refresh_dashboard()

    def _refresh_dashboard(self) -> None:
        self.toolbar.refresh_gpu_status()

    def _display_buffer_for_canvas(self, buffer):
        if isinstance(buffer, GPUTexture):
            buffer = buffer.readback()

        if isinstance(buffer, np.ndarray) and buffer.ndim == 3 and buffer.shape[2] == 4:
            return buffer[:, :, :3]

        return buffer

    def _on_loading_started(self) -> None:
        """Keep the previous frame visible (dimmed) under a spinner instead of blanking."""
        self.empty_state.setVisible(False)
        self.loading_overlay.start()

    def _on_load_failed(self) -> None:
        self.loading_overlay.stop()
        self.canvas.clear()

    def _on_session_emptied(self) -> None:
        """Last file was unloaded/cleared: blank the viewer (the removed image must
        not linger with no way to dismiss it) and bring the empty-state hint back."""
        self.loading_overlay.stop()
        self.canvas.clear()
        self.empty_state.setVisible(True)
        self._refresh_image_info()

    def _on_image_updated(self) -> None:
        """Refreshes canvas when a new render pass completes."""
        if not self.state.uploaded_files:
            # A render that was in flight when the session emptied — the frame
            # belongs to a removed file; keep the viewer blank.
            return
        self.empty_state.setVisible(False)
        metrics = self.state.last_metrics
        if "base_positive" not in metrics:
            logger.warning("Render completed but 'base_positive' not found in metrics")
            return

        buffer = _display_buffer_for_canvas(metrics["base_positive"])
        content_rect = metrics.get("content_rect")

        if isinstance(buffer, np.ndarray) and not self.state.gpu_enabled:
            finish_conf = self.state.config.finish
            export_conf = self.state.config.export
            # Crop/analysis render the uncropped border-less frame — padding it would
            # misalign the tool rect (GPU skips the layout pass there too).
            should_preview = (
                finish_conf.border_size > 0 or export_conf.paper_aspect_ratio != AspectRatio.ORIGINAL
            ) and self.state.active_tool not in (ToolMode.CROP_MANUAL, ToolMode.ANALYSIS_DRAW)

            if should_preview:
                pil_img = Image.fromarray(float_to_uint8(buffer))
                try:
                    pil_img, content_rect = PrintService.apply_preview_layout_to_pil(
                        pil_img,
                        export_conf.paper_aspect_ratio,
                        finish_conf.border_size,
                        export_conf.export_print_size,
                        PrintService.effective_border_color(finish_conf, self.state.config.toning),
                        APP_CONFIG.preview_render_size,
                        finish=finish_conf,
                    )
                    buffer = np.array(pil_img).astype(np.float32) / 255.0
                except Exception as e:
                    logger.error(f"Border preview failure: {e}")

        # Shared with the filmstrip thumbnail so the same frame can't render two
        # different colours in the two places (see display_transform_params).
        display_cs, monitor_bytes = self.controller.display_transform_params(splash=bool(metrics.get("splash")))
        self.canvas.update_buffer(buffer, display_cs, content_rect=content_rect, monitor_icc_bytes=monitor_bytes)

    def _refresh_image_info(self) -> None:
        """Updates the canvas HUD corner pills."""
        if not self.state.current_file_path:
            self.canvas.hud.update_info("", "", "", "", "", "")
            return

        filename = os.path.basename(self.state.current_file_path)
        w, h = self.state.original_res
        res_str = f"{w} x {h} px"

        cs = self.state.config.export.export_color_space
        if cs == "Same as Source":
            cs = self.state.source_cs or "Same as Source"
        mode_str = f"{self.state.config.process.process_mode} | {cs}"
        edits_str = f"Edits: {self.state.undo_index}"

        tool_label = self.TOOL_LABELS.get(self.state.active_tool, "")
        total = len(self.state.uploaded_files)
        idx = self.state.selected_file_idx
        file_pos = f"{idx + 1} / {total}" if total > 1 and idx >= 0 else ""

        self.canvas.hud.update_info(filename, res_str, mode_str, edits_str, tool_label, file_pos)

    def _on_zoom_info_changed(self, zoom: float) -> None:
        pass

    def _on_export_progress(self, current: int, total: int, filename: str) -> None:
        self.canvas.hud.set_progress(current, total)
        self.canvas.hud.showMessage(f"Exporting {filename} ({current}/{total})...")

    def _on_export_finished(self, elapsed: float, failed: int) -> None:
        self.canvas.hud.hide_progress()
        msg = f"export complete in {elapsed:.2f}s"
        if failed:
            msg += f" — {failed} failed"
        self.canvas.hud.showMessage(msg, timeout=6000 if failed else 3000)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if hasattr(self, "empty_state"):
            self.empty_state.setGeometry(self.canvas.rect())
        if hasattr(self, "loading_overlay"):
            self.loading_overlay.setGeometry(self.canvas.rect())

    def _on_zone_armed(self, zone) -> None:
        """Name the armed zone on the canvas: the strip highlight is in the sidebar,
        away from the click it is waiting for."""
        if zone is None:
            return
        from negpy.features.exposure.densitometer import zone_roman

        self.canvas.hud.showMessage(f"click the photo to place zone {zone_roman(float(zone))}", timeout=3000)

    def _sync_tool_buttons(self) -> None:
        """Updates toggle button states to match active_tool."""
        mode = self.state.active_tool
        self.canvas.set_tool_mode(mode)

        # We access buttons through the controls panel
        self.controls_panel.colour_sidebar.pick_wb_btn.setChecked(mode == ToolMode.WB_PICK)
        self.controls_panel.geometry_sidebar.manual_crop_btn.setChecked(mode == ToolMode.CROP_MANUAL)
        self.controls_panel.retouch_sidebar.pick_dust_btn.setChecked(mode == ToolMode.DUST_PICK)

        self._update_title()
        self._refresh_image_info()

    def dragEnterEvent(self, event) -> None:
        if event.mimeData().hasUrls():
            paths = [u.toLocalFile() for u in event.mimeData().urls()]
            if any(os.path.splitext(p)[1].lower() in SUPPORTED_RAW_EXTENSIONS or os.path.isdir(p) for p in paths):
                event.acceptProposedAction()
                return
        event.ignore()

    def dropEvent(self, event) -> None:
        paths = [u.toLocalFile() for u in event.mimeData().urls()]
        if len(paths) == 1 and os.path.isdir(paths[0]):
            # Same courtesy as Add folder: a dropped folder that only holds subfolders
            # shows them rather than reporting that it found nothing.
            self.session_panel.file_browser.open_or_browse(paths[0])
        elif paths:
            self.controller.request_asset_discovery(paths, auto_open=True)
        event.acceptProposedAction()
