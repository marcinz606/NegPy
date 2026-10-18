from negpy.kernel.system.i18n import tr
import os
from typing import Optional

import numpy as np
from PIL import Image
from PyQt6.QtCore import QByteArray, Qt, QEvent, QTimer, pyqtSignal
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QDialog,
    QLabel,
    QMainWindow,
    QMenu,
    QMessageBox,
    QSplitter,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from negpy.kernel.system.text import count_of
from negpy.desktop.controller import AppController
from negpy.desktop.session import ToolMode
from negpy.infrastructure.loaders.constants import SUPPORTED_RAW_EXTENSIONS
from negpy.desktop.converters import ImageConverter
from negpy.desktop.view.canvas.hud import INFO_FULL, frame_info
from negpy.desktop.view.canvas.reference_pane import ReferencePane
from negpy.desktop.view.canvas.toolbar import ActionToolbar, frame_position
from negpy.desktop.view.canvas.widget import ImageCanvas
from negpy.desktop.view.keyboard_shortcuts import SpacePanKeyFilter, setup_keyboard_shortcuts
from negpy.desktop.view.mac_menu_bar import install_mac_menus
from negpy.desktop.view.shortcut_registry import tooltip_with_shortcut
from negpy.desktop.view.sidebar.custom_routing_dialog import CustomRoutingDialog
from negpy.desktop.view.sidebar.pick_better_window import PickBetterWindow
from negpy.desktop.view.sidebar.right_panel import RightPanel
from negpy.desktop.view.sidebar.session_panel import SessionPanel
from negpy.desktop.view.styles.templates import hint_label, labeled_action
from negpy.desktop.view.widgets.command_palette import CommandPalette, FindField
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


class _EmptyStateOverlay(QWidget):
    """Shown on top of the canvas when no image is loaded.

    Tracks the canvas rather than the window: hiding a dock resizes the canvas
    without resizing the window, which used to leave this stranded off-centre
    while the floating toolbar (which the canvas lays out) moved with it.
    """

    add_files_requested = pyqtSignal()
    import_roll_requested = pyqtSignal()
    scan_requested = pyqtSignal()
    tour_requested = pyqtSignal()

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        parent.installEventFilter(self)

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(12)

        self.load_btn = labeled_action(
            "fa5s.folder-open", tr("Load Scans…"), tr("Import a folder as a roll, add single files, or scan film"), primary=True
        )
        self.load_btn.clicked.connect(self._show_load_menu)
        layout.addWidget(self.load_btn, alignment=Qt.AlignmentFlag.AlignHCenter)
        drop_hint = hint_label(tr("Or drop pictures or a folder onto the window"))
        drop_hint.setWordWrap(False)
        layout.addWidget(drop_hint, alignment=Qt.AlignmentFlag.AlignHCenter)

        self.tour_btn = labeled_action("", tr("Take the Tour"), tr("A short guided walk through the panels"))
        self.tour_btn.clicked.connect(self.tour_requested)
        layout.addWidget(self.tour_btn, alignment=Qt.AlignmentFlag.AlignHCenter)

    def _show_load_menu(self) -> None:
        menu = QMenu(self)
        menu.addAction(tr("Import Folder as a Roll…")).triggered.connect(self.import_roll_requested)
        menu.addAction(tr("Add Files…")).triggered.connect(self.add_files_requested)
        menu.addAction(tr("Scan…")).triggered.connect(self.scan_requested)
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
        self._pick_better_window = PickBetterWindow(self)
        self._connect_signals()
        self.shortcut_manager = setup_keyboard_shortcuts(self)
        self.space_pan_filter = SpacePanKeyFilter(self)
        self.space_pan_filter.space_held_changed.connect(self.canvas.set_space_pan_held)
        # macOS only: the global menu bar costs no window space, and elsewhere this is a
        # no-op. After the shortcut manager, whose actions the menu items dispatch through.
        self.mac_menus = install_mac_menus(self)
        self._update_title()

        from negpy.desktop.view.widgets.tutorial_overlay import TutorialOverlay

        self.tutorial_overlay = TutorialOverlay(self)
        self.tutorial_overlay.finished.connect(self._on_tutorial_finished)

    def _restore_window_geometry(self) -> None:
        """Restore the window state and geometry, with a legacy size/position fallback."""
        encoded = self.controller.session.repo.get_global_setting("window_geometry_qt")
        if isinstance(encoded, str) and self.restoreGeometry(QByteArray.fromBase64(encoded.encode("utf-8"))):
            return
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
        if self.controller.session.repo.get_global_setting("window_maximized", False):
            self.setWindowState(Qt.WindowState.WindowMaximized)

    def closeEvent(self, event) -> None:
        space_pan_filter = getattr(self, "space_pan_filter", None)
        if space_pan_filter is not None:
            space_pan_filter.uninstall()
        try:
            geo = self.normalGeometry() if self.isMaximized() or self.isFullScreen() else self.geometry()
            self.controller.session.repo.save_global_settings(
                {
                    "window_geometry_qt": bytes(self.saveGeometry().toBase64()).decode("ascii"),
                    "window_geometry": [geo.x(), geo.y(), geo.width(), geo.height()],
                    "window_maximized": self.isMaximized(),
                    "dock_state": bytes(self.saveState().toBase64()).decode("ascii"),
                }
            )
        except Exception:
            logger.exception("Failed to persist window layout")
        super().closeEvent(event)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        # The window handle, and so its screen, only exists once shown. Wire monitor-profile
        # detection once, then track screen changes.
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
                tr("Restore Session"),
                tr("Reopen your last session ({value})?", value=count_of(len(paths), "file")),
            )
            if reply == QMessageBox.StandardButton.Yes:
                self.controller.restore_session()
                self._schedule_onboarding()
                return
        self.session_panel.show_library(ask_if_unset=False)
        self._schedule_onboarding()

    def _schedule_onboarding(self) -> None:
        """Runs after the restore question, so the first-run tour never opens under a message box."""
        repo = self.controller.session.repo
        if not repo.get_global_setting("tutorial_seen", False):
            QTimer.singleShot(600, self.show_tutorial)  # the scan-setup wizard follows the tour
        elif repo.get_global_setting("scan_setup") is None:
            QTimer.singleShot(600, self.show_scan_setup)

    def _refresh_monitor_profile(self, force: bool = False) -> None:
        """Detect the active screen's ICC profile and hand it to the controller, which
        resolves it against any manual override and pushes it to the display paths."""
        handle = self.windowHandle()
        screen = handle.screen() if handle is not None else self.screen()
        data = _read_screen_icc(screen) if screen is not None else None
        if not force and data == self.state.monitor_icc_detected_bytes:
            return
        if data is None and not self.state.monitor_profile_override and not getattr(self, "_icc_miss_notified", False):
            self._icc_miss_notified = True
            self.canvas.hud.showMessage(tr("No monitor ICC profile detected — preview assumes sRGB"), 6000, kind="warning")
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
        self.empty_state.import_roll_requested.connect(lambda: self.session_panel.library_tree.prompt_import_folder())
        self.empty_state.scan_requested.connect(self._show_scan_tab)
        self.empty_state.raise_()

        self.loading_overlay = LoadingOverlay(self.canvas)
        self.loading_overlay.raise_()

        if self.state.gpu_viewport_failed:
            QTimer.singleShot(
                0,
                lambda: self.canvas.hud.showMessage(
                    tr(
                        "GPU viewport failed to start — display runs on the CPU ({gpu_viewport_failed})",
                        gpu_viewport_failed=self.state.gpu_viewport_failed,
                    ),
                    8000,
                    kind="warning",
                ),
            )

        self.central_stack = QStackedWidget()
        self.central_stack.addWidget(self.canvas)
        self.reference_pane = ReferencePane(self.canvas.background_color)
        self.reference_pane.closed.connect(self.close_reference)
        self.reference_pane.hide()
        self.central_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.central_splitter.addWidget(self.reference_pane)
        self.central_splitter.addWidget(self.central_stack)
        self.central_splitter.setCollapsible(1, False)
        self.central_layout.addWidget(self.central_splitter, stretch=1)
        self._panels_before_light_table: list = []

        self.setCentralWidget(self.central_widget)

        self.drawer = PinnableDockWidget(
            tr("Controls"),
            self,
            pin_tooltip=tooltip_with_shortcut(tr("Dock controls panel to right"), "toggle_right_panel"),
            on_pin=self.dock_controls_panel,
            docked_title=FindField(self.show_command_palette),
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
            tr("Session"),
            self,
            pin_tooltip=tooltip_with_shortcut(tr("Dock session panel to left"), "toggle_left_panel"),
            on_pin=self.dock_session_panel,
        )
        self.session_dock.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea)
        self.session_dock.setObjectName("session_dock")
        self.session_panel = SessionPanel(self.controller)
        self.session_dock.setWidget(self.session_panel)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.session_dock)
        self.central_stack.addWidget(self.session_panel.file_browser.light_table_view)

        # Snapshot the pristine layout, with both docked at their home edges at default widths,
        # so the pin and Reset Panel Layout restore the true original position and size rather
        # than re-adding the dock where it currently sits.
        self._default_dock_state = self.saveState()

        repo = self.controller.session.repo
        saved_docks = repo.get_global_setting("dock_state")
        if isinstance(saved_docks, str) and saved_docks:
            self.restoreState(QByteArray.fromBase64(saved_docks.encode("ascii")))
        # The visibility keys are what the toggles write, so they win over the dock snapshot.
        self.session_dock.setVisible(repo.get_global_setting("panel_left_visible", True))
        self.drawer.setVisible(repo.get_global_setting("panel_right_visible", True))

    TOOL_LABELS: dict[ToolMode, str] = {
        ToolMode.WB_PICK: tr("WB Picker"),
        ToolMode.CROP_MANUAL: "Crop",
        ToolMode.DUST_PICK: tr("Heal Tool"),
        ToolMode.CLONE: tr("Clone Tool"),
    }

    def _update_title(self) -> None:
        state = self.controller.session.state
        if state.current_file_path:
            filename = os.path.basename(state.current_file_path)
            prefix = "● " if state.is_dirty else ""
            tool = self.TOOL_LABELS.get(state.active_tool)
            tool_prefix = f"[{tool}] " if tool else ""
            self.setWindowTitle(tr("{prefix}NegPy — {tool_prefix}{filename}", prefix=prefix, tool_prefix=tool_prefix, filename=filename))
        else:
            self.setWindowTitle(tr("NegPy"))

    def _on_immersive_changed(self) -> None:
        if getattr(self, "_last_immersive", None) != self.controller.session.state.immersive_canvas:
            self._last_immersive = self.controller.session.state.immersive_canvas
            self.canvas.fit_to_window()

    def _show_scan_tab(self) -> None:
        if not self.drawer.isVisible():
            self.toggle_controls_dock()
        self.right_panel.show_tab_by_key("scan")

    def show_tutorial(self) -> None:
        from negpy.desktop.view.widgets.tutorial_steps import build

        self.tutorial_overlay.start(build(self))

    def _on_tutorial_finished(self, completed: bool) -> None:
        repo = self.controller.session.repo
        repo.save_global_setting("tutorial_seen", True)
        repo.save_global_setting("tutorial_resume", 0 if completed else self.tutorial_overlay.index)
        # Unset only until the wizard is answered once, so replaying the tour later from the
        # menu does not ask again.
        if repo.get_global_setting("scan_setup") is None:
            QTimer.singleShot(0, self.show_scan_setup)

    def show_about(self) -> None:
        from negpy.kernel.system.version import GITHUB_REPO, get_app_version

        box = QMessageBox(self)
        box.setWindowTitle(tr("About NegPy"))
        box.setIconPixmap(self.windowIcon().pixmap(64, 64))
        box.setTextFormat(Qt.TextFormat.RichText)
        box.setText(
            tr(
                '<b>NegPy</b> {value}<br>Film negatives printed through a virtual darkroom.<br><a href="https://github.com/{GITHUB_REPO}">github.com/{value1}</a>',
                value=get_app_version(),
                GITHUB_REPO=GITHUB_REPO,
                value1=GITHUB_REPO,
            )
        )
        box.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)
        label = box.findChild(QLabel, "qt_msgbox_label")
        if label is not None:
            label.setOpenExternalLinks(True)
        box.setStandardButtons(QMessageBox.StandardButton.Close)
        box.exec()

    def show_command_palette(self) -> None:
        CommandPalette(self).exec()

    def show_scan_setup(self) -> None:
        from negpy.desktop.view.widgets.scan_setup_dialog import ScanSetupDialog

        repo = self.controller.session.repo
        dlg = ScanSetupDialog(self, repo.get_global_setting("scan_setup"), repo=repo)
        # Cancel leaves scan_setup unset: an unanswered wizard is not an answer, so it asks again.
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self.controller.apply_scan_setup(**dlg.choice())

    def _restore_default_dock_state(self) -> None:
        """Restore both docks to the snapshot taken at startup (home edge + width)."""
        if self._default_dock_state is not None:
            self.restoreState(self._default_dock_state)

    def dock_session_panel(self) -> None:
        # Restore the pristine layout, which returns this panel to its original edge and width,
        # but preserve the other panel's current visibility so pinning one never forces the
        # other to reappear.
        right_visible = self.drawer.isVisible()
        self._restore_default_dock_state()
        self.drawer.setVisible(right_visible)
        self.session_dock.setVisible(True)
        self.controller.session.repo.save_global_setting("panel_left_visible", True)

    def dock_controls_panel(self) -> None:
        left_visible = self.session_dock.isVisible()
        self._restore_default_dock_state()
        self.session_dock.setVisible(left_visible)
        self.drawer.setVisible(True)
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

    def toggle_side_panels(self) -> None:
        show = not (self.session_dock.isVisible() or self.drawer.isVisible())
        for dock, key in ((self.session_dock, "panel_left_visible"), (self.drawer, "panel_right_visible")):
            if dock.isFloating():
                continue
            dock.setVisible(show)
            self.controller.session.repo.save_global_setting(key, show)

    def toggle_reference(self) -> None:
        if self.reference_pane.isVisible():
            self.close_reference()
            return
        image = self._reference_snapshot()
        if image is None:
            self.canvas.hud.showMessage(tr("Open a frame to pin it as the reference"), 3000)
            return
        self.reference_pane.set_reference(image, os.path.basename(self.state.current_file_path))
        self.reference_pane.show()
        half = self.central_splitter.width() // 2
        self.central_splitter.setSizes([half, self.central_splitter.width() - half])

    def close_reference(self) -> None:
        self.reference_pane.hide()
        self.reference_pane.set_reference(None)

    def _reference_snapshot(self):
        """The frame on the canvas as displayed now; the pane keeps it until pinned again."""
        shown = self.state.canvas_value
        buffer = shown("base_positive")
        if not self.state.current_file_path or buffer is None or shown("splash"):
            return None
        if isinstance(buffer, GPUTexture):
            buffer = buffer.readback()[:, :, :3]
        display_cs, monitor, proof = self.controller.display_transform_params(proofed=bool(shown("proof", True)))
        return ImageConverter.to_qimage(np.ascontiguousarray(buffer, dtype=np.float32), display_cs, monitor, proof)

    def light_table_active(self) -> bool:
        return self.central_stack.currentIndex() == 1

    def _leave_empty_light_table(self, *_args) -> None:
        if self.light_table_active() and self.controller.session.asset_model.rowCount() == 0:
            self.set_light_table(False)

    def set_light_table(self, on: bool) -> None:
        """Both side panels hide while the grid shows, so it has the whole window; their saved
        visibility is untouched. Esc, Shift+G or opening a frame leaves it."""
        browser = self.session_panel.file_browser
        # Both panels hide with it, so an empty grid would have no way back but Esc.
        if on and self.controller.session.asset_model.rowCount() == 0:
            on = False
            self.controller.set_status(tr("No frames match the filter") if self.state.uploaded_files else tr("No frames loaded"), 3000)
        browser.light_table_btn.blockSignals(True)
        browser.light_table_btn.setChecked(on)
        browser.light_table_btn.blockSignals(False)
        if on == self.light_table_active():
            return
        view = browser.light_table_view
        if on:
            self._panels_before_light_table = [d for d in (self.session_dock, self.drawer) if d.isVisible() and not d.isFloating()]
            for dock in self._panels_before_light_table:
                dock.setVisible(False)
            self.central_stack.setCurrentIndex(1)
            row = self.controller.session.asset_model.actual_to_display(self.state.selected_file_idx)
            if row >= 0:
                view.scrollTo(self.controller.session.asset_model.index(row, 0))
            view.setFocus()
        else:
            self.central_stack.setCurrentIndex(0)
            for dock in self._panels_before_light_table:
                dock.setVisible(True)

    def reset_panel_layout(self) -> None:
        """Restore both side panels to their original edges, widths and visibility."""
        self._restore_default_dock_state()
        self.session_dock.setVisible(True)
        self.drawer.setVisible(True)
        self.controller.session.repo.save_global_setting("panel_left_visible", True)
        self.controller.session.repo.save_global_setting("panel_right_visible", True)
        self.canvas.hud.showMessage(tr("Panel layout reset"), timeout=1500)

    def _connect_signals(self) -> None:
        """Wire controller and view."""
        browser = self.session_panel.file_browser
        browser.light_table_btn.toggled.connect(self.set_light_table)
        browser.light_table_opened.connect(lambda: self.set_light_table(False))
        self.session_panel.update_found.connect(self.toolbar.set_update_available)
        self.controller.session.state_changed.connect(self._update_title)
        self.controller.session.state_changed.connect(self._on_immersive_changed)

        # visibilityChanged only mirrors the button, and it also fires on close and minimize, so
        # persist in the toggle methods instead and the saved state survives exit.
        self.toolbar.btn_toggle_left.clicked.connect(self.toggle_session_dock)
        self.toolbar.btn_toggle_right.clicked.connect(self.toggle_controls_dock)
        self.controller.image_updated.connect(self._on_image_updated)
        self.controller.preview_loaded.connect(self._refresh_image_info)
        self.controller.loading_started.connect(self._on_loading_started)
        self.controller.gpu_textures_released.connect(self.canvas.release_gpu_texture)
        self.controller.image_updated.connect(self.loading_overlay.stop)
        self.controller.load_failed.connect(self._on_load_failed)
        self.controller.zoom_changed.connect(self._on_zoom_info_changed)

        # Metadata updates only on persistent history changes or file selection
        self.controller.session.history_changed.connect(self._refresh_image_info)
        self.controller.session.file_selected.connect(lambda _: self._refresh_image_info())
        self.controller.session.asset_model.layoutChanged.connect(lambda *_: self._refresh_image_info())
        self.controller.session.session_emptied.connect(self._on_session_emptied)

        self.canvas.clicked.connect(self.controller.handle_canvas_clicked)
        self.canvas.crop_rect_changed.connect(self.controller.handle_crop_rect_changed)
        self.canvas.crop_rotation_changed.connect(self.controller.handle_crop_rotation_changed)
        self.canvas.crop_confirmed.connect(self.controller.confirm_manual_crop)
        self.canvas.analysis_rect_changed.connect(self.controller.handle_analysis_rect_changed)
        self.canvas.analysis_confirmed.connect(self.controller.confirm_analysis_region)
        self.canvas.local_mask_created.connect(self.controller.handle_local_mask_created)
        self.canvas.leak_shape_created.connect(self.controller.handle_leak_shape_created)
        self.canvas.leak_shape_edited.connect(self.controller.handle_leak_shape_edited)
        self.canvas.leak_paint_completed.connect(self.controller.handle_leak_paint_completed)
        self.canvas.scratch_completed.connect(self.controller.handle_heal_stroke_completed)
        self.canvas.clone_stroke_completed.connect(self.controller.handle_clone_stroke_completed)
        self.canvas.clone_source_picked.connect(self.controller.set_clone_source)
        self.canvas.dust_exclusion_painted.connect(self.controller.handle_dust_exclusion_painted)
        self.canvas.straighten_completed.connect(self.controller.handle_straighten_completed)
        self.canvas.keystone_line_marked.connect(self.controller.handle_keystone_line_marked)
        self.controller.keystone_lines_cleared.connect(self.canvas.clear_keystone_lines)
        self.canvas.zone_pin_moved.connect(self.controller.move_zone_pin)
        self.canvas.zone_placement_confirmed.connect(self.controller.apply_zone_placement)
        self.canvas.local_mask_selected.connect(self.controller.select_local_mask)
        self.canvas.local_mask_edited.connect(self.controller.handle_local_mask_edited)
        self.canvas.local_vertex_deleted.connect(self.controller.delete_local_vertex)

        self.controller.export_progress.connect(self._on_export_progress)
        self.controller.export_finished.connect(self._on_export_finished)
        self.controller.session.settings_copied.connect(lambda: self.canvas.hud.showMessage(tr("Settings copied"), timeout=1500))
        self.controller.session.settings_pasted.connect(lambda: self.canvas.hud.showMessage(tr("Settings pasted"), timeout=1500))
        self.controller.session.color_vision_changed.connect(self.canvas.overlay.update)
        model = self.controller.session.asset_model
        for signal in (model.layoutChanged, model.modelReset, model.rowsRemoved):
            signal.connect(self._leave_empty_light_table)
        self.controller.session.settings_synced.connect(lambda msg: self.canvas.hud.showMessage(msg, timeout=2500))
        self.controller.tool_sync_requested.connect(self._sync_tool_buttons)
        self.controller.config_updated.connect(self.canvas.overlay.update)
        self.controller.analysis_buffer_preview_requested.connect(self.canvas.overlay.show_analysis_buffer)
        self.controller.analysis_buffer_drag_changed.connect(self.canvas.overlay.set_analysis_buffer_dragging)
        self.controller.rotation_guide_requested.connect(self.canvas.overlay.show_rotation_grid)
        self.controller.rotation_guide_requested.connect(self.canvas.overlay.show_crop_preview)
        self.controller.crop_guide_changed.connect(self.canvas.overlay.update)
        self.controller.dust_overlay_changed.connect(self.canvas.overlay.update)
        self.controller.zones_overlay_changed.connect(lambda _on: self.canvas.overlay.update())
        self.controller.zones_style_changed.connect(lambda _style: self.canvas.overlay.update())
        self.canvas.overlay.zone_shares_changed.connect(self.toolbar.set_zone_shares)
        self.canvas.overlay.zone_pointer_changed.connect(self.toolbar.set_zone_pointer)
        self.controller.clipping_overlay_changed.connect(lambda _on: self.canvas.overlay.update())
        self.controller.zone_highlight_changed.connect(self.canvas.overlay.update)
        self.controller.grain_focuser_changed.connect(lambda _on: self.canvas.overlay.update())
        self.controller.printing_notes_changed.connect(lambda _on: self.canvas.overlay.update())
        self.controller.cursor_probe_changed.connect(lambda _on: self.canvas.overlay.update())
        self.controller.densitometer_readout.connect(self.canvas.overlay.set_probe_reading)
        self.controller.canvas_info_changed.connect(lambda _level: self._refresh_image_info())
        self.controller.printing_notes_requested.connect(self._save_printing_notes)
        self.controller.compare_frame_ready.connect(self.canvas.refresh_compare)
        self.controller.compare_changed.connect(lambda _on: self.canvas.refresh_compare())
        self.controller.compare_hold_changed.connect(lambda _on: self.canvas.overlay.update())
        self.controller.test_strip_changed.connect(lambda _up: self.canvas.overlay.on_test_strip_changed())
        self.canvas.test_strip_picked.connect(self.controller.apply_test_strip_pick)
        self.canvas.paper_strip_kept.connect(self.controller.keep_paper_strip)
        self.controller.pick_better_changed.connect(self._on_pick_better_changed)
        self.controller.pick_better_ready.connect(self._on_pick_better_ready)
        self._pick_better_window.picked.connect(self.controller.pick_better_select)
        self._pick_better_window.rejected.connect(self.controller.pick_better_reject)
        self._pick_better_window.undoRequested.connect(self.controller.pick_better_undo)
        self._pick_better_window.closed.connect(self.controller.pick_better_finish)
        self.controller.pick_better_param.connect(self._pick_better_window.set_param_name)
        self.controller.pick_better_moved.connect(self._pick_better_window.show_move)
        self._pick_better_window.diffChanged.connect(self.controller.pick_better_set_diff)
        self._pick_better_window.customRoutingClicked.connect(self._on_custom_routing_clicked)
        self.controller.local_drag_changed.connect(self.canvas.overlay.set_local_slider_drag)
        self.controller.zone_pins_changed.connect(self.canvas.overlay.update)
        self.controller.zone_arm_changed.connect(self._on_zone_armed)

        self.controller.status_message_requested.connect(self.canvas.hud.showMessage)
        self.controller.status_progress_requested.connect(self.canvas.hud.set_progress)

        self.progress_dialog = ProgressDialog(self)
        self.controller.batch_started.connect(self._on_batch_started)
        self.controller.batch_progress.connect(self.progress_dialog.set_progress)
        self.controller.batch_finished.connect(self.progress_dialog.finish)
        self.progress_dialog.abort_requested.connect(self.controller.abort_active_batch)

    def _on_pick_better_changed(self, up: bool) -> None:
        """Show the separate Pick Better window while the mode is active; hide it when it ends."""
        if not up:
            self._pick_better_window.hide()
            return
        self._pick_better_window.show()
        self._pick_better_window.raise_()

    def _on_pick_better_ready(self, current, left, right) -> None:
        """Render the three committed frames to QPixmaps and push them onto the window's labels."""
        display_cs, monitor, proof = self.controller.display_transform_params(proofed=True)
        self._pick_better_window.set_images(
            self._to_pick_better_pixmap(current, display_cs, monitor, proof),
            self._to_pick_better_pixmap(left, display_cs, monitor, proof),
            self._to_pick_better_pixmap(right, display_cs, monitor, proof),
        )

    def _on_custom_routing_clicked(self) -> None:
        """Open the Custom Routing window seeded with the current route; on Apply, commit it."""
        order, randomize = self.controller.pick_better_route()
        dialog = CustomRoutingDialog(self, order, randomize, repo=self.controller.session.repo)
        if dialog.exec():
            self.controller.pick_better_set_route(dialog.route(), dialog.randomize_on())

    @staticmethod
    def _to_pick_better_pixmap(arr, display_cs, monitor, proof) -> QPixmap:
        """One rendered frame (read back from the worker) through the display transform, as a QPixmap."""
        qimage = ImageConverter.to_qimage(np.ascontiguousarray(arr, dtype=np.float32), display_cs, monitor, proof)
        return QPixmap.fromImage(qimage)

    def _on_batch_started(self, title: str, abortable: bool) -> None:
        """Hot Folder polls every 2 s, so its per-frame import batches (the only
        non-abortable ones) would pop the dialog on every capture. The HUD status
        line still reports those; abortable batches always show the popup.

        Suppression keys off which sequence actually owns this batch
        (`hot_folder_sequence_active`), not whether the toggle happens to be
        checked — a manual Add Files/Add Folder/drop while Hot Folder is on must
        still show progress."""
        if not abortable and self.controller.hot_folder_sequence_active:
            return
        self.progress_dialog.start(title, abortable)

    def _save_printing_notes(self) -> None:
        """Write the marked-up work print. The annotated pixels are the canvas's own
        render, so the composing happens here rather than in an export worker."""
        sheet = self.canvas.overlay.printing_notes_sheet()
        if sheet is None:
            self.controller.set_status(tr("Printing notes need a rendered frame"), 4000)
            return
        path = self.controller.printing_notes_target_path()
        if not path:
            return
        if sheet.save(path, "JPEG", self.controller.state.config.export.jpeg_quality):
            self.controller.set_status(tr("Printing notes saved: {value}", value=os.path.basename(path)), 4000)
        else:
            self.controller.set_status(tr("Could not write {path}", path=path), 4000, kind="error")

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
        # With nothing else open, a black canvas after the toast expires is a dead end.
        self.empty_state.setVisible(not self.state.uploaded_files)

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
            # A render that was in flight when the session emptied. The frame belongs to a removed
            # file, so keep the viewer blank.
            return
        self.empty_state.setVisible(False)
        shown = self.state.canvas_value
        # Passed on as it is: the GPU display path samples the texture and applies the
        # working-to-display LUT in its shader.
        buffer = shown("base_positive")
        if buffer is None:
            logger.warning("Render completed but 'base_positive' not found in metrics")
            return
        content_rect = shown("content_rect")

        if isinstance(buffer, np.ndarray) and not self.state.gpu_enabled:
            finish_conf = self.state.config.finish
            export_conf = self.state.config.export
            # No padding for a crop_preview_full buffer (the uncropped frame), as on the GPU: it would misalign
            # the tool rect. The buffer's flag decides, not the live tool: a render can land after the tool changes.
            should_preview = (finish_conf.border_size > 0 or export_conf.paper_aspect_ratio != AspectRatio.ORIGINAL) and not shown(
                "crop_preview_full"
            )

            if should_preview:
                pil_img = Image.fromarray(float_to_uint8(buffer))
                try:
                    pil_img, content_rect = PrintService.apply_preview_layout_to_pil(
                        pil_img,
                        export_conf.paper_aspect_ratio,
                        finish_conf.border_size,
                        export_conf.export_print_size,
                        PrintService.effective_border_color(finish_conf, self.state.config.toning),
                        # Capped at what this frame holds: the CPU layout resamples the
                        # content to the preview long edge, and a smaller buffer must not
                        # be upscaled into it (see GPUEngine._calculate_layout_dims).
                        min(APP_CONFIG.preview_render_size, max(buffer.shape[:2])),
                        finish=finish_conf,
                    )
                    buffer = np.array(pil_img).astype(np.float32) / 255.0
                except Exception as e:
                    logger.error(f"Border preview failure: {e}")

        # Shared with the filmstrip thumbnail, so the same frame cannot render two different
        # colors in the two places (see display_transform_params).
        display_cs, monitor_bytes, proof = self.controller.display_transform_params(
            splash=bool(shown("splash")), proofed=bool(shown("proof", True))
        )
        self.canvas.update_buffer(buffer, display_cs, content_rect=content_rect, monitor_icc_bytes=monitor_bytes, proof=proof)

    def _refresh_image_info(self) -> None:
        """Updates the canvas HUD corner pills."""
        st = self.state
        if not st.current_file_path:
            self.canvas.hud.set_zoom_note("")
            self.canvas.hud.update_info("", "", "")
            return
        w, h = st.original_res
        proof = os.path.splitext(os.path.basename(st.proof_icc_path))[0] if st.soft_proof_enabled and st.proof_icc_path else ""
        self.canvas.hud.update_info(
            *frame_info(
                st.canvas_info_level,
                st.config,
                os.path.basename(st.current_file_path),
                f"{w} x {h} px",
                str(st.config.process.process_mode),
                frame_position(self.controller.session.asset_model, st.selected_file_idx),
                proof,
            )
        )
        self._on_zoom_info_changed(self.canvas.zoom_level)

    def _on_zoom_info_changed(self, zoom: float) -> None:
        """The zoom readout: under the navigator while it is up, else the bottom-right pill."""
        if not self.state.current_file_path:
            return
        note = self.canvas.zoom_note()
        percent = f"{self.canvas.current_zoom_percent()}%"
        if not self.canvas.navigator.isHidden():
            self.canvas.navigator.set_caption(" · ".join(s for s in (percent, note) if s))
            self.canvas.hud.set_zoom_note("")
            return
        full = self.state.canvas_info_level == INFO_FULL
        self.canvas.hud.set_zoom_note(" · ".join(s for s in (percent if full else "", note) if s))

    def _on_export_progress(self, current: int, total: int, filename: str) -> None:
        self.canvas.hud.set_progress(current, total)
        self.canvas.hud.showMessage(tr("Exporting {filename} ({current}/{total})…", filename=filename, current=current, total=total))

    def _on_export_finished(self, elapsed: float, errors: list) -> None:
        self.canvas.hud.hide_progress()
        msg = f"Export complete in {elapsed:.2f}s"
        if errors:
            msg += f" — {len(errors)} failed"
        self.canvas.hud.showMessage(msg, timeout=6000 if errors else 3000, kind="warning" if errors else "info")
        if errors:
            # A toast holds one line and the next one replaces it; the box keeps every file.
            box = QMessageBox(
                QMessageBox.Icon.Warning,
                tr("Export"),
                tr("{value} could not be exported.", value=count_of(len(errors), "file")),
                parent=self,
            )
            box.setDetailedText("\n".join(errors))
            box.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
            box.open()

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

        self.canvas.hud.showMessage(tr("Click the photo to place zone {value}", value=zone_roman(float(zone))), timeout=3000)

    def _sync_tool_buttons(self) -> None:
        """Updates toggle button states to match active_tool."""
        mode = self.state.active_tool
        self.canvas.set_tool_mode(mode)

        # We access buttons through the controls panel
        self.controls_panel.color_sidebar.pick_wb_btn.setChecked(mode == ToolMode.WB_PICK)
        self.controls_panel.geometry_sidebar.manual_crop_btn.setChecked(mode == ToolMode.CROP_MANUAL)
        self.controls_panel.retouch_sidebar.pick_dust_btn.setChecked(mode == ToolMode.DUST_PICK)
        self.controls_panel.retouch_sidebar.clone_btn.setChecked(mode == ToolMode.CLONE)

        self._update_title()

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
            # The same courtesy as Add folder: a dropped folder that only holds subfolders shows
            # them rather than reporting that it found nothing.
            self.session_panel.file_browser.open_or_browse(paths[0])
        elif paths:
            self.controller.request_asset_discovery(paths, auto_open=True)
        event.acceptProposedAction()
