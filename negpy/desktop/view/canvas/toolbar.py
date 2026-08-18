import os
from dataclasses import dataclass

import qtawesome as qta
from PyQt6.QtCore import QSize, Qt
from PyQt6.QtGui import QActionGroup
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QMenu,
    QMessageBox,
    QPushButton,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from negpy.desktop.controller import AppController
from negpy.desktop.view.keyboard_shortcuts import _context_undo
from negpy.desktop.view.widgets.granular_settings_dialog import open_paste_dialog
from negpy.desktop.view.widgets.sliders import apply_slider_value_visibility
from negpy.kernel.system.parallel import parallel_enabled, set_parallel_enabled
from negpy.desktop.view.shortcut_registry import key_for, tooltip_with_shortcut
from negpy.desktop.view.styles.theme import THEME
from negpy.infrastructure.gpu.device import GPUDevice

CANVAS_COLORS = [
    ("#050505", (0.02, 0.02, 0.02), "Black"),
    ("#1C1C1C", (0.11, 0.11, 0.11), "Dark Grey"),
    ("#404040", (0.25, 0.25, 0.25), "Mid Grey"),
    ("#FFFFFF", (1.0, 1.0, 1.0), "White"),
]


@dataclass(frozen=True)
class ToolbarItem:
    id: str
    category: str
    label: str


# The row is customizable; the overflow menu is not. Category drives both the dialog's grouping
# and the separators, so a reordered row still gets dividers where the kinds change.
TOOLBAR_ITEMS: tuple[ToolbarItem, ...] = (
    ToolbarItem("prev", "Navigation", "Previous"),
    ToolbarItem("next", "Navigation", "Next"),
    ToolbarItem("zoom_label", "Zoom", "Zoom Readout"),
    ToolbarItem("zoom_fit", "Zoom", "Fit to Window"),
    ToolbarItem("zoom_original", "Zoom", "Original Size (1:1)"),
    ToolbarItem("hq", "Zoom", "HQ Preview"),
    ToolbarItem("rot_l", "Geometry", "Rotate CCW"),
    ToolbarItem("rot_r", "Geometry", "Rotate CW"),
    ToolbarItem("flip_h", "Geometry", "Flip Horizontal"),
    ToolbarItem("flip_v", "Geometry", "Flip Vertical"),
    ToolbarItem("undo", "View", "Undo"),
    ToolbarItem("redo", "View", "Redo"),
    ToolbarItem("compare", "View", "Before / After"),
    ToolbarItem("flat_peek", "View", "Peek Flat Scan"),
    ToolbarItem("negative_peek", "View", "Peek Negative"),
    ToolbarItem("zones", "View", "Zone Overlay"),
    ToolbarItem("loupe", "View", "Grain Focuser"),
)

TOOLBAR_ITEM_BY_ID = {item.id: item for item in TOOLBAR_ITEMS}

DEFAULT_TOOLBAR_IDS: tuple[str, ...] = (
    "prev",
    "next",
    "zoom_label",
    "zoom_fit",
    "zoom_original",
    "hq",
    "rot_l",
    "rot_r",
    "flip_h",
    "flip_v",
    "undo",
    "redo",
    "compare",
    "negative_peek",
    "zones",
    "loupe",
)

_TOOLBAR_SETTING_KEY = "toolbar_items"


def load_toolbar_items(repo) -> list[str]:
    """No stored row means the stock one; unknown ids drop so a retired button degrades quietly."""
    stored = repo.get_global_setting(_TOOLBAR_SETTING_KEY)
    if not isinstance(stored, list):
        return list(DEFAULT_TOOLBAR_IDS)
    return [item_id for item_id in stored if item_id in TOOLBAR_ITEM_BY_ID]


class ActionToolbar(QWidget):
    """
    Unified toolbar for file navigation, geometry actions, and session management.
    """

    def __init__(self, controller: AppController):
        super().__init__()
        self.controller = controller
        self.session = controller.session

        self._init_ui()
        self._connect_signals()

    def _create_separator(self) -> QFrame:
        line = QFrame()
        line.setFrameShape(QFrame.Shape.VLine)
        line.setFrameShadow(QFrame.Shadow.Plain)
        line.setObjectName("toolbar_separator")
        line.setFixedWidth(1)
        return line

    def _init_ui(self) -> None:
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 10, 0, 10)

        container = QFrame()
        container.setObjectName("toolbar_container")
        self._toolbar_container = container
        v_layout = QVBoxLayout(container)
        v_layout.setContentsMargins(6, 4, 6, 4)
        v_layout.setSpacing(0)

        row_layout = QHBoxLayout()
        row_layout.setSpacing(6)

        icon_color = THEME.text_primary
        icon_size = QSize(16, 16)
        btn_height = 32

        # 0. Panel toggles (live at the toolbar's outer edges)
        self.btn_toggle_left = QToolButton()
        self.btn_toggle_left.setCheckable(True)
        self.btn_toggle_left.setChecked(True)
        self.btn_toggle_left.setIcon(qta.icon("fa5s.columns", color=icon_color))
        self.btn_toggle_left.setToolTip(tooltip_with_shortcut("Toggle Session Panel", "toggle_left_panel"))
        self.btn_toggle_right = QToolButton()
        self.btn_toggle_right.setCheckable(True)
        self.btn_toggle_right.setChecked(True)
        self.btn_toggle_right.setIcon(qta.icon("fa5s.sliders-h", color=icon_color))
        self.btn_toggle_right.setToolTip(tooltip_with_shortcut("Toggle Controls Panel", "toggle_right_panel"))

        # 1. Navigation
        self.btn_prev = QToolButton()
        self.btn_prev.setIcon(qta.icon("fa5s.chevron-left", color=icon_color))
        self.btn_prev.setToolTip("Previous")
        self.btn_next = QToolButton()
        self.btn_next.setIcon(qta.icon("fa5s.chevron-right", color=icon_color))
        self.btn_next.setToolTip("Next")

        # Undo and Redo live in the main toolbar (mdi arrows, distinct from the rotate icons'
        # file-with-arrow glyphs below).
        self.btn_undo = QToolButton()
        self.btn_undo.setIcon(qta.icon("mdi.undo", color=icon_color))
        self.btn_undo.setToolTip(tooltip_with_shortcut("Undo", "undo"))
        self.btn_redo = QToolButton()
        self.btn_redo.setIcon(qta.icon("mdi.redo", color=icon_color))
        self.btn_redo.setToolTip(tooltip_with_shortcut("Redo", "redo"))

        # kept as internal state holders, not added to layout
        self.btn_copy = QPushButton()
        self.btn_paste = QPushButton()
        self.btn_reset = QPushButton()
        self.btn_unload = QPushButton()

        # 2. Geometry
        self.btn_rot_l = QToolButton()
        self.btn_rot_l.setIcon(qta.icon("mdi6.file-rotate-left", color=icon_color))
        self.btn_rot_l.setToolTip(tooltip_with_shortcut("Rotate CCW", "rotate_ccw"))
        self.btn_rot_r = QToolButton()
        self.btn_rot_r.setIcon(qta.icon("mdi6.file-rotate-right", color=icon_color))
        self.btn_rot_r.setToolTip(tooltip_with_shortcut("Rotate CW", "rotate_cw"))
        self.btn_flip_h = QToolButton()
        self.btn_flip_h.setCheckable(True)
        self.btn_flip_h.setIcon(qta.icon("fa5s.arrows-alt-h", color=icon_color))
        self.btn_flip_h.setToolTip(tooltip_with_shortcut("Flip Horizontal", "flip_h"))
        self.btn_flip_v = QToolButton()
        self.btn_flip_v.setCheckable(True)
        self.btn_flip_v.setIcon(qta.icon("fa5s.arrows-alt-v", color=icon_color))
        self.btn_flip_v.setToolTip(tooltip_with_shortcut("Flip Vertical", "flip_v"))

        # 3. Zoom: a read-only percent readout, since users zoom directly on the canvas. Match the
        # button height and centre both axes so it sits on the same line as the icons rather than
        # floating, a touch larger and bolder than a caption.
        self.zoom_label = QLabel("100%")
        self.zoom_label.setFixedSize(48, btn_height)
        self.zoom_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.zoom_label.setStyleSheet(f"color: {THEME.text_secondary}; font-size: {THEME.font_size_header}px; font-weight: 600;")

        self.btn_zoom_fit = QToolButton()
        self.btn_zoom_fit.setIcon(qta.icon("fa5s.expand", color=icon_color))
        self.btn_zoom_fit.setToolTip(tooltip_with_shortcut("Fit to Window", "fit_view"))
        self.btn_zoom_original = QToolButton()
        self.btn_zoom_original.setText("1:1")
        self.btn_zoom_original.setToolTip(
            tooltip_with_shortcut(
                "Original size (100%). Displays a lower-resolution preview unless HQ is enabled.",
                "zoom_100",
            )
        )

        self.btn_hq = QToolButton()
        self.btn_hq.setText("HQ")
        self.btn_hq.setCheckable(True)
        self.btn_hq.setToolTip("Toggle High Quality Preview")

        self.btn_compare = QToolButton()
        self.btn_compare.setCheckable(True)
        self.btn_compare.setIcon(qta.icon("fa5s.adjust", color=icon_color))
        self.btn_compare.setToolTip(
            tooltip_with_shortcut("Before / After — split against the auto baseline, drag the divider", "toggle_compare")
        )

        # Overflow-only (kept as a state holder so the checked-state mirror still works).
        self.btn_flat_peek = QToolButton()
        self.btn_flat_peek.setCheckable(True)
        self.btn_flat_peek.setIcon(qta.icon("fa5s.eye", color=icon_color))
        self.btn_flat_peek.setToolTip(
            tooltip_with_shortcut("Peek flat scan — temporarily show the flat master (does not change your edit)", "toggle_flat_peek")
        )

        self.btn_negative_peek = QToolButton()
        self.btn_negative_peek.setCheckable(True)
        self.btn_negative_peek.setIcon(qta.icon("fa5s.film", color=icon_color))
        self.btn_negative_peek.setToolTip(
            tooltip_with_shortcut(
                "Peek negative — show the source as it was loaded, un-inverted and unedited, at your crop and rotation (no colour management)",
                "toggle_negative_peek",
            )
        )

        self.btn_zones = QToolButton()
        self.btn_zones.setCheckable(True)
        self.btn_zones.setIcon(qta.icon("mdi.grid", color=icon_color))
        self.btn_zones.setToolTip(
            tooltip_with_shortcut("Zone overlay — label each region of the print with its Adams zone", "toggle_zones")
        )

        self.btn_loupe = QToolButton()
        self.btn_loupe.setCheckable(True)
        self.btn_loupe.setIcon(qta.icon("fa5s.search-plus", color=icon_color))
        self.btn_loupe.setToolTip(
            tooltip_with_shortcut(
                "Grain focuser — a loupe at the cursor showing the frame's own pixels, with an "
                "acutance figure for comparing sharpness across the frame (reads true on HQ)",
                "toggle_grain_focuser",
            )
        )

        # GPU availability drives the overflow toggle built below (btn moved off the row).
        self._gpu_available = GPUDevice.get().is_available

        # 4. Overflow menu & responsive groups
        self.btn_overflow = QToolButton()
        self.btn_overflow.setIcon(qta.icon("fa5s.ellipsis-h", color=icon_color))
        self.btn_overflow.setToolTip("More actions")
        self.btn_overflow.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)

        overflow_menu = QMenu(self.btn_overflow)
        overflow_menu.setToolTipsVisible(True)  # so the GPU action can surface its backend/status

        # Overflow always mirrors the full action set, whatever the toolbar row shows at the
        # current canvas width. "More actions" is a stable, complete menu the user can always
        # find everything in, not a residue of the row's responsive collapse. It used to lose
        # entries whenever a side panel toggle gave the row enough width to show them directly.
        self._ov_hq_action = overflow_menu.addAction("Toggle HQ Preview")
        self._ov_hq_action.setCheckable(True)
        self._ov_hq_action.setToolTip("Toggle High Quality Preview")

        # GPU acceleration lives here, not on the editing row. Details are in the tooltip,
        # refreshed by the dashboard through refresh_gpu_status().
        self._ov_gpu_action = overflow_menu.addAction(qta.icon("fa5s.bolt", color=icon_color), "GPU Acceleration")
        self._ov_gpu_action.setCheckable(True)
        self._ov_gpu_action.setToolTip("GPU Acceleration")
        if self._gpu_available:
            self._ov_gpu_action.setChecked(self.session.state.gpu_enabled)
        else:
            self._ov_gpu_action.setEnabled(False)
            self._ov_gpu_action.setChecked(False)

        # Beside GPU Acceleration because it answers the same question, which processor does the
        # work, and the two trade against each other. With the GPU carrying the pipeline, the CPU
        # kernels are left with the source assembly, where an HDR solve takes seconds.
        self._parallel_action = overflow_menu.addAction("Multi-core CPU Rendering")
        self._parallel_action.setCheckable(True)
        self._parallel_action.triggered.connect(self._on_cpu_parallel_toggled)
        self.refresh_cpu_parallel_status()
        overflow_menu.addSeparator()

        # Canvas background, overflow-only with no toolbar swatches. An exclusive checkable group,
        # so the menu itself shows which color is active.
        self._ov_color_group = QActionGroup(self)
        self._ov_color_group.setExclusive(True)
        self._ov_color_actions: list = []
        for i, (_, _, label) in enumerate(CANVAS_COLORS):
            action = overflow_menu.addAction(f"Canvas: {label}")
            action.setCheckable(True)
            action.setChecked(i == self.session.state.canvas_bg_index)
            action.setToolTip(f"Set the canvas background to {label.lower()}")
            self._ov_color_group.addAction(action)
            self._ov_color_actions.append(action)
        overflow_menu.addSeparator()

        overflow_menu.addSeparator()
        self._ov_fit_action = overflow_menu.addAction(qta.icon("fa5s.expand", color=icon_color), "Fit to Window")
        self._ov_fit_action.setToolTip(tooltip_with_shortcut("Fit to Window", "fit_view"))
        self._ov_original_action = overflow_menu.addAction("Original Size (1:1)")
        self._ov_original_action.setToolTip(
            tooltip_with_shortcut(
                "Original size (100%). Displays a lower-resolution preview unless HQ is enabled.",
                "zoom_100",
            )
        )
        self._ov_compare_action = overflow_menu.addAction(qta.icon("fa5s.adjust", color=icon_color), "Before / After")
        self._ov_compare_action.setCheckable(True)
        self._ov_compare_action.setToolTip(
            tooltip_with_shortcut("Before / After — split against the auto baseline, drag the divider", "toggle_compare")
        )
        self._ov_flat_peek_action = overflow_menu.addAction(qta.icon("fa5s.eye", color=icon_color), "Peek Flat Scan")
        self._ov_flat_peek_action.setCheckable(True)
        self._ov_flat_peek_action.setToolTip(
            tooltip_with_shortcut("Peek flat scan — temporarily show the flat master (does not change your edit)", "toggle_flat_peek")
        )
        self._ov_negative_peek_action = overflow_menu.addAction(qta.icon("fa5s.film", color=icon_color), "Peek Negative")
        self._ov_negative_peek_action.setCheckable(True)
        self._ov_negative_peek_action.setToolTip(
            tooltip_with_shortcut(
                "Peek negative — show the source as it was loaded, un-inverted and unedited, at your crop and rotation (no colour management)",
                "toggle_negative_peek",
            )
        )
        self._ov_zones_action = overflow_menu.addAction(qta.icon("mdi.grid", color=icon_color), "Zone Overlay")
        self._ov_zones_action.setCheckable(True)
        self._ov_zones_action.setToolTip(
            tooltip_with_shortcut("Zone overlay — label each region of the print with its Adams zone", "toggle_zones")
        )
        self._ov_loupe_action = overflow_menu.addAction(qta.icon("fa5s.search-plus", color=icon_color), "Grain Focuser")
        self._ov_loupe_action.setCheckable(True)
        self._ov_loupe_action.setToolTip(
            tooltip_with_shortcut(
                "Grain focuser — a loupe at the cursor showing the frame's own pixels, with an "
                "acutance figure for comparing sharpness across the frame (reads true on HQ)",
                "toggle_grain_focuser",
            )
        )
        self._ov_undo_action = overflow_menu.addAction(qta.icon("mdi.undo", color=icon_color), "Undo")
        self._ov_undo_action.setToolTip(tooltip_with_shortcut("Undo", "undo"))
        self._ov_redo_action = overflow_menu.addAction(qta.icon("mdi.redo", color=icon_color), "Redo")
        self._ov_redo_action.setToolTip(tooltip_with_shortcut("Redo", "redo"))

        overflow_menu.addSeparator()
        self._ov_rot_l_action = overflow_menu.addAction(qta.icon("mdi6.file-rotate-left", color=icon_color), "Rotate CCW")
        self._ov_rot_l_action.setToolTip(tooltip_with_shortcut("Rotate CCW", "rotate_ccw"))
        self._ov_rot_r_action = overflow_menu.addAction(qta.icon("mdi6.file-rotate-right", color=icon_color), "Rotate CW")
        self._ov_rot_r_action.setToolTip(tooltip_with_shortcut("Rotate CW", "rotate_cw"))
        self._ov_flip_h_action = overflow_menu.addAction(qta.icon("fa5s.arrows-alt-h", color=icon_color), "Flip Horizontal")
        self._ov_flip_h_action.setCheckable(True)
        self._ov_flip_h_action.setToolTip(tooltip_with_shortcut("Flip Horizontal", "flip_h"))
        self._ov_flip_v_action = overflow_menu.addAction(qta.icon("fa5s.arrows-alt-v", color=icon_color), "Flip Vertical")
        self._ov_flip_v_action.setCheckable(True)
        self._ov_flip_v_action.setToolTip(tooltip_with_shortcut("Flip Vertical", "flip_v"))
        overflow_menu.addSeparator()

        # Edits auto-save to the DB and surface in History, so an explicit Save lives here in the
        # overflow rather than the main toolbar.
        save_action = overflow_menu.addAction(qta.icon("fa5s.save", color=icon_color), "Save Edits", self.controller.save_current_edits)
        save_action.setToolTip("Write the current edit to the database now (edits also auto-save)")
        overflow_menu.addSeparator()
        self._action_copy = overflow_menu.addAction(
            qta.icon("fa5s.copy", color=icon_color), "Copy Settings  Ctrl+C", self.session.copy_settings
        )
        self._action_copy.setToolTip("Copy this image's settings to the clipboard")
        self._action_copy_bounds = overflow_menu.addAction(
            qta.icon("fa5s.copy", color=icon_color), "Copy Settings + Bounds  Ctrl+Shift+C", self.session.copy_settings_with_bounds
        )
        self._action_copy_bounds.setToolTip("Copy settings plus the metering/normalization bounds")
        self._action_paste = overflow_menu.addAction(
            qta.icon("fa5s.paste", color=icon_color), "Paste Settings  Ctrl+V", lambda: open_paste_dialog(self, self.controller)
        )
        self._action_paste.setToolTip("Paste the copied settings onto this image")
        overflow_menu.addSeparator()
        reset_settings_action = overflow_menu.addAction(
            qta.icon("fa5s.history", color=icon_color), "Reset Settings", self.session.reset_settings
        )
        reset_settings_action.setToolTip("Discard all edits and return this image to its default look")
        overflow_menu.addSeparator()
        unload_action = overflow_menu.addAction(qta.icon("fa5s.times-circle", color=icon_color), "Unload", self._on_overflow_unload)
        unload_action.setToolTip("Remove this image from the session (its saved edit is kept)")
        overflow_menu.addSeparator()
        scale_menu = overflow_menu.addMenu(qta.icon("fa5s.search-plus", color=icon_color), "UI Scale")
        scale_menu.setToolTipsVisible(True)
        scale_menu.menuAction().setToolTip("Scale the whole interface (applies after a restart)")
        self._ui_scale_group = QActionGroup(self)
        self._ui_scale_group.setExclusive(True)
        current_scale = float(self.session.repo.get_global_setting("ui_scale", 1.0) or 1.0)
        for pct in (80, 90, 100, 110, 120):
            val = pct / 100.0
            act = scale_menu.addAction(f"{pct}%")
            act.setCheckable(True)
            act.setChecked(abs(val - current_scale) < 0.001)
            self._ui_scale_group.addAction(act)
            act.triggered.connect(lambda _checked=False, v=val, p=pct: self._on_ui_scale_selected(v, p))

        overflow_menu.addSeparator()

        reset_key = key_for("reset_panel_layout")
        reset_label = "Reset Panel Layout" + (f"  {reset_key}" if reset_key else "")
        reset_layout_action = overflow_menu.addAction(
            qta.icon("fa5s.thumbtack", color=icon_color),
            reset_label,
            self._reset_panel_layout,
        )
        reset_layout_action.setToolTip("Restore the default panel sizes and positions")
        edit_toolbar_action = overflow_menu.addAction(
            qta.icon("fa5s.wrench", color=icon_color),
            "Edit Toolbar…",
            self.open_toolbar_editor,
        )
        edit_toolbar_action.setToolTip(tooltip_with_shortcut("Choose which controls sit on the toolbar, and in what order", "edit_toolbar"))
        self._ov_immersive_action = overflow_menu.addAction("Immersive Canvas")
        self._ov_immersive_action.setCheckable(True)
        self._ov_immersive_action.setChecked(self.session.state.immersive_canvas)
        self._ov_immersive_action.setToolTip(
            tooltip_with_shortcut("Toolbar overlaps image — turn off to fit the image above the toolbar", "toggle_immersive_canvas")
        )
        self._ov_sticky_zoom_action = overflow_menu.addAction("Sticky Zoom")
        self._ov_sticky_zoom_action.setCheckable(True)
        self._ov_sticky_zoom_action.setChecked(self.session.state.sticky_zoom)
        self._ov_sticky_zoom_action.setToolTip(
            tooltip_with_shortcut("Keep the current zoom level when switching images, instead of resetting to fit", "toggle_sticky_zoom")
        )
        self._ov_invert_zoom_action = overflow_menu.addAction("Reverse Scroll Zoom")
        self._ov_invert_zoom_action.setCheckable(True)
        self._ov_invert_zoom_action.setChecked(self.session.state.invert_zoom_scroll)
        self._ov_invert_zoom_action.setToolTip(tooltip_with_shortcut("Scroll up zooms out instead of in", "toggle_invert_zoom_scroll"))
        self._ov_slider_values_action = overflow_menu.addAction("Show Slider Values")
        self._ov_slider_values_action.setCheckable(True)
        self._ov_slider_values_action.setChecked(bool(self.session.repo.get_global_setting("show_slider_values", default=False)))
        self._ov_slider_values_action.setToolTip(
            tooltip_with_shortcut("Keep every slider's value box open, instead of revealing it on hover", "toggle_slider_values")
        )
        overflow_menu.addSeparator()

        db_action = overflow_menu.addAction(qta.icon("fa5s.database", color=icon_color), "Manage Database…", self._show_database_dialog)
        db_action.setToolTip("View stored data and clear saved edits")
        overflow_menu.addSeparator()

        tour_action = overflow_menu.addAction(qta.icon("fa5s.map-signs", color=icon_color), "Take the tour", self._show_tour)
        tour_action.setToolTip("Replay the guided feature tour")
        shortcuts_action = overflow_menu.addAction(
            qta.icon("fa5s.keyboard", color=icon_color), "Keyboard Shortcuts  ?", self._show_shortcuts
        )
        shortcuts_action.setToolTip("Show the full keyboard shortcuts reference")
        self.btn_overflow.setMenu(overflow_menu)

        standard_buttons = [
            self.btn_toggle_left,
            self.btn_toggle_right,
            self.btn_prev,
            self.btn_next,
            self.btn_flip_h,
            self.btn_flip_v,
            self.btn_undo,
            self.btn_redo,
            self.btn_zoom_fit,
            self.btn_zoom_original,
            self.btn_hq,
            self.btn_compare,
            self.btn_flat_peek,
            self.btn_negative_peek,
            self.btn_zones,
            self.btn_loupe,
            self.btn_overflow,
        ]
        for btn in standard_buttons:
            btn.setIconSize(icon_size)
            btn.setFixedHeight(btn_height)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)

        # The file-rotate glyphs (page plus arrow) read as a blob at the standard 16px icon size.
        # A touch larger keeps the page and arrow legible without changing the button's own
        # footprint, since btn_height still applies.
        rotate_icon_size = QSize(20, 20)
        for btn in (self.btn_rot_l, self.btn_rot_r):
            btn.setIconSize(rotate_icon_size)
            btn.setFixedHeight(btn_height)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)

        self._row_layout = row_layout
        self._row_widgets: dict[str, QWidget] = {
            "prev": self.btn_prev,
            "next": self.btn_next,
            "zoom_label": self.zoom_label,
            "zoom_fit": self.btn_zoom_fit,
            "zoom_original": self.btn_zoom_original,
            "hq": self.btn_hq,
            "rot_l": self.btn_rot_l,
            "rot_r": self.btn_rot_r,
            "flip_h": self.btn_flip_h,
            "flip_v": self.btn_flip_v,
            "undo": self.btn_undo,
            "redo": self.btn_redo,
            "compare": self.btn_compare,
            "flat_peek": self.btn_flat_peek,
            "negative_peek": self.btn_negative_peek,
            "zones": self.btn_zones,
            "loupe": self.btn_loupe,
        }
        self._row_sequence: list[tuple[QWidget, bool]] = []
        self._available_width = 0
        self._item_ids = load_toolbar_items(self.session.repo)
        self._rebuild_row()

        v_layout.addLayout(row_layout)
        # Size the pill to its controls; don't stretch it across the canvas.
        main_layout.addWidget(container, 0, Qt.AlignmentFlag.AlignCenter)

    def _on_zones_changed(self, active: bool) -> None:
        for widget in (self.btn_zones, self._ov_zones_action):
            widget.blockSignals(True)
            widget.setChecked(active)
            widget.blockSignals(False)

    def _on_grain_focuser_changed(self, active: bool) -> None:
        for widget in (self.btn_loupe, self._ov_loupe_action):
            widget.blockSignals(True)
            widget.setChecked(active)
            widget.blockSignals(False)

    def _on_flat_peek_changed(self, active: bool) -> None:
        self.btn_flat_peek.blockSignals(True)
        self.btn_flat_peek.setChecked(active)
        self.btn_flat_peek.blockSignals(False)
        self._ov_flat_peek_action.blockSignals(True)
        self._ov_flat_peek_action.setChecked(active)
        self._ov_flat_peek_action.blockSignals(False)

    def _on_negative_peek_changed(self, active: bool) -> None:
        for widget in (self.btn_negative_peek, self._ov_negative_peek_action):
            widget.blockSignals(True)
            widget.setChecked(active)
            widget.blockSignals(False)

    def _connect_signals(self) -> None:
        self.btn_prev.clicked.connect(self.session.prev_file)
        self.btn_next.clicked.connect(self.session.next_file)

        self.btn_rot_l.clicked.connect(lambda: self.rotate(1))
        self.btn_rot_r.clicked.connect(lambda: self.rotate(-1))
        self.btn_flip_h.clicked.connect(lambda: self.flip("horizontal"))
        self.btn_flip_v.clicked.connect(lambda: self.flip("vertical"))

        # Same context routing as Ctrl+Z: heal-undo while a heal tool is in hand.
        self.btn_undo.clicked.connect(lambda: _context_undo(self.controller))
        self.btn_redo.clicked.connect(self.session.redo)

        self.btn_zoom_fit.clicked.connect(self._on_fit_clicked)
        self.btn_zoom_original.clicked.connect(self._on_original_clicked)
        self.btn_hq.clicked.connect(self.controller.toggle_hq_preview)
        self.btn_compare.clicked.connect(self.controller.toggle_compare)
        self.controller.compare_changed.connect(self.btn_compare.setChecked)
        self.controller.compare_changed.connect(self._ov_compare_action.setChecked)
        self.btn_flat_peek.toggled.connect(lambda checked: self.controller.toggle_flat_peek(force=checked))
        self.controller.flat_peek_changed.connect(self._on_flat_peek_changed)
        self.btn_negative_peek.toggled.connect(lambda checked: self.controller.toggle_negative_peek(force=checked))
        self._ov_negative_peek_action.triggered.connect(lambda checked: self.controller.toggle_negative_peek(force=checked))
        self.controller.negative_peek_changed.connect(self._on_negative_peek_changed)
        self.btn_zones.toggled.connect(lambda checked: self.controller.toggle_zones_overlay(force=checked))
        self._ov_zones_action.triggered.connect(lambda checked: self.controller.toggle_zones_overlay(force=checked))
        self.controller.zones_overlay_changed.connect(self._on_zones_changed)
        self.btn_loupe.toggled.connect(lambda checked: self.controller.toggle_grain_focuser(force=checked))
        self._ov_loupe_action.triggered.connect(lambda checked: self.controller.toggle_grain_focuser(force=checked))
        self.controller.grain_focuser_changed.connect(self._on_grain_focuser_changed)
        self._ov_gpu_action.toggled.connect(self._on_gpu_toggled)
        self.controller.zoom_changed.connect(self._on_zoom_changed)

        self.session.state_changed.connect(self._update_ui_state)
        self.session.asset_model.layoutChanged.connect(self._update_ui_state)

        # Overflow menu action connections
        self._ov_hq_action.triggered.connect(self.controller.toggle_hq_preview)
        for i, action in enumerate(self._ov_color_actions):
            action.triggered.connect(lambda checked, idx=i: self._on_canvas_color_changed(idx, True))
        self._ov_rot_l_action.triggered.connect(lambda: self.rotate(1))
        self._ov_rot_r_action.triggered.connect(lambda: self.rotate(-1))
        self._ov_flip_h_action.triggered.connect(lambda: self.flip("horizontal"))
        self._ov_flip_v_action.triggered.connect(lambda: self.flip("vertical"))
        self._ov_fit_action.triggered.connect(self._on_fit_clicked)
        self._ov_original_action.triggered.connect(self._on_original_clicked)
        self._ov_compare_action.triggered.connect(self.controller.toggle_compare)
        self._ov_flat_peek_action.triggered.connect(lambda checked: self.controller.toggle_flat_peek(force=checked))
        self._ov_undo_action.triggered.connect(lambda: _context_undo(self.controller))
        self._ov_redo_action.triggered.connect(self.session.redo)
        self._ov_immersive_action.triggered.connect(self._on_immersive_toggled)
        self._ov_sticky_zoom_action.triggered.connect(self._on_sticky_zoom_toggled)
        self._ov_invert_zoom_action.triggered.connect(self._on_invert_zoom_toggled)
        self._ov_slider_values_action.triggered.connect(self._on_slider_values_toggled)

    def _on_overflow_unload(self) -> None:
        from negpy.desktop.view.confirm import confirm_unload

        if self.session.state.selected_file_idx < 0:
            return
        if confirm_unload(self):
            self.session.remove_current_file()

    def _on_immersive_toggled(self, checked: bool) -> None:
        self.session.set_immersive_canvas(checked)

    def _on_slider_values_toggled(self, checked: bool) -> None:
        self.session.repo.save_global_setting("show_slider_values", bool(checked))
        apply_slider_value_visibility(self.window(), bool(checked))

    def _on_sticky_zoom_toggled(self, checked: bool) -> None:
        self.session.set_sticky_zoom(checked)

    def _on_invert_zoom_toggled(self, checked: bool) -> None:
        self.session.set_invert_zoom_scroll(checked)

    def _on_gpu_toggled(self, checked: bool) -> None:
        if checked != self.session.state.gpu_enabled:
            self.session.set_gpu_enabled(checked)

    def refresh_gpu_status(self) -> None:
        """Reflect current GPU on/off state and active backend in the overflow toggle."""
        enabled = self.session.state.gpu_enabled
        action = self._ov_gpu_action

        action.blockSignals(True)
        action.setChecked(enabled and self._gpu_available)
        action.blockSignals(False)

        icon_color = THEME.accent_primary if (enabled and self._gpu_available) else THEME.text_primary
        action.setIcon(qta.icon("fa5s.bolt", color=icon_color))

        if not self._gpu_available:
            action.setToolTip("GPU not available on this hardware")
        elif enabled:
            try:
                backend = self.controller.render_worker.processor.backend_name
            except Exception:
                backend = "GPU"
            action.setToolTip(f"GPU Acceleration: ON — {backend}\nClick to force the CPU pipeline.")
        else:
            action.setToolTip("GPU Acceleration: OFF — CPU pipeline\nClick to enable WebGPU for near-instant previews.")

    def refresh_cpu_parallel_status(self) -> None:
        """Reflect on/off in the icon, the way the GPU toggle does.

        A checkable QAction that also carries an icon shows the icon in the indicator slot
        instead of a tick, so the state has to live in the icon itself or the entry looks
        identical either way.
        """
        enabled = parallel_enabled()
        action = self._parallel_action

        action.blockSignals(True)
        action.setChecked(enabled)
        action.blockSignals(False)

        action.setIcon(qta.icon("fa5s.microchip", color=THEME.accent_primary if enabled else THEME.text_primary))
        if enabled:
            action.setToolTip(
                f"Multi-core CPU Rendering: ON — {os.cpu_count() or '?'} cores\n"
                "The rendering kernels run 5-8x faster, which is about 10% off a whole HDR\n"
                "merge — the rest of that time is decoding the RAW files, which this does\n"
                "not touch. Experimental: if the app closes without warning, turn it off."
            )
        else:
            action.setToolTip(
                "Multi-core CPU Rendering: OFF — one core\n"
                "Click to spread the CPU rendering kernels across cores. They run 5-8x\n"
                "faster, worth roughly 10% off a whole HDR merge — most of that time is\n"
                "decoding the RAW files. Experimental on macOS."
            )

    def _on_cpu_parallel_toggled(self, checked: bool) -> None:
        """Takes effect at once: every kernel is compiled both ways and dispatched per
        call, so there is nothing to recompile and no restart to wait for."""
        self.session.repo.save_global_setting("cpu_parallel", bool(checked))
        set_parallel_enabled(bool(checked))
        self.session.repo.save_global_setting("cpu_parallel_active", bool(checked))
        self.refresh_cpu_parallel_status()
        self.controller.set_status(
            "Multi-core CPU rendering on — turn it off if the app closes unexpectedly" if checked else "Multi-core CPU rendering off",
            5000,
        )

    def _on_ui_scale_selected(self, value: float, pct: int) -> None:
        self.session.repo.save_global_setting("ui_scale", value)
        QMessageBox.information(
            self,
            "UI Scale",
            f"UI scale set to {pct}%.\n\nRestart NegPy to apply the change.",
        )

    def _on_canvas_color_changed(self, idx: int, checked: bool) -> None:
        if checked:
            self.session.set_canvas_bg(idx)
            if self.controller.canvas:
                _, (r, g, b), _ = CANVAS_COLORS[idx]
                self.controller.canvas.set_background_color(r, g, b)

    def _on_zoom_changed(self, zoom: float) -> None:
        # The label shows the true pixel zoom (zoom_level x fit_scale), which is what the user
        # cares about. zoom_level itself is fit-relative.
        canvas = getattr(self.controller, "canvas", None)
        pct = canvas.current_zoom_percent() if canvas is not None else int(round(max(0.0, zoom) * 100.0))
        self.zoom_label.setText(f"{pct}%")

    def _on_fit_clicked(self) -> None:
        canvas = getattr(self.controller, "canvas", None)
        if canvas is not None:
            canvas.fit_to_window()

    def _on_original_clicked(self) -> None:
        canvas = getattr(self.controller, "canvas", None)
        if canvas is not None:
            canvas.zoom_to_original()

    def rotate(self, direction: int) -> None:
        from dataclasses import replace

        from negpy.features.geometry.logic import rotate_normalized_rect

        # A proof on the canvas takes the rotation instead of the image. Must precede the
        # handedness fix below, which is geometry-only.
        if self.controller.rotate_test_strip(direction):
            return

        config = self.session.state.config
        geo = config.geometry
        # The button's labelled direction is the visual rotation the user sees, and the handedness
        # fix below only keeps that promise under a flip. Crop and analysis rects live in display
        # space, so they rotate by that visual quarter-turn.
        visual_turns_ccw = direction
        # Pipeline applies rotate-then-flip; a single mirror inverts rotation handedness.
        if geo.flip_horizontal != geo.flip_vertical:
            direction = -direction
        new_rot = (geo.rotation + direction) % 4
        new_geo = replace(geo, rotation=new_rot)
        # Rotate the manual crop rect with the content so it keeps framing the same area. Without
        # this it stayed put and misaligned after a quarter or half turn.
        if geo.crop_rect is not None:
            new_geo = replace(new_geo, crop_rect=rotate_normalized_rect(geo.crop_rect, visual_turns_ccw))
        new_config = replace(config, geometry=new_geo)
        # The freehand analysis region is display-space too; rotate it alongside.
        if config.process.analysis_rect is not None:
            new_rect = rotate_normalized_rect(config.process.analysis_rect, visual_turns_ccw)
            new_config = replace(new_config, process=replace(config.process, analysis_rect=new_rect))
        self.session.update_config(new_config, persist=True)
        # Rotating must not drop an active before/after or flat-peek, so re-render in place within
        # whichever view is on.
        self.controller.rerender_active_view()

    def flip(self, axis: str) -> None:
        from dataclasses import replace

        from negpy.features.geometry.logic import mirror_normalized_rect, toggle_flip

        horizontal = axis == "horizontal"
        config = self.session.state.config
        # toggle_flip negates fine rotation and mirrors the crop rect, so the result is a true
        # mirror of the current render (see its docstring).
        new_config = replace(config, geometry=toggle_flip(config.geometry, horizontal))
        # The freehand analysis region is transformed-space like the crop rect, so mirroring it
        # keeps the meters reading the same picture content.
        if config.process.analysis_rect is not None:
            new_rect = mirror_normalized_rect(config.process.analysis_rect, horizontal)
            new_config = replace(new_config, process=replace(config.process, analysis_rect=new_rect))
        self.session.update_config(new_config, persist=True)
        # Flipping shouldn't drop an active before/after or flat-peek (see rotate()).
        self.controller.rerender_active_view()

    def _reset_panel_layout(self) -> None:
        from negpy.desktop.view.main_window import MainWindow

        win = self.window()
        if isinstance(win, MainWindow):
            win.reset_panel_layout()

    def _show_tour(self) -> None:
        from negpy.desktop.view.main_window import MainWindow

        win = self.window()
        if isinstance(win, MainWindow):
            win.show_tutorial()

    def _show_shortcuts(self) -> None:
        from negpy.desktop.view.widgets.shortcuts_overlay import ShortcutsOverlay

        dlg = ShortcutsOverlay(self.window().shortcut_manager, self.window())
        dlg.exec()

    def _show_database_dialog(self) -> None:
        from negpy.desktop.view.widgets.database_dialog import DatabaseDialog

        DatabaseDialog(self.session.repo, self.controller, self.window()).exec()

    def _update_ui_state(self) -> None:
        state = self.session.state
        model = self.session.asset_model
        display_idx = model.actual_to_display(state.selected_file_idx)
        self.btn_prev.setEnabled(display_idx > 0)
        self.btn_next.setEnabled(0 <= display_idx < model.rowCount() - 1)
        self.btn_hq.setChecked(state.hq_preview)
        self._ov_hq_action.setChecked(state.hq_preview)
        self.btn_compare.setChecked(state.compare_mode)
        self._ov_compare_action.setChecked(state.compare_mode)
        self.btn_flat_peek.setChecked(state.flat_peek)
        self._ov_flat_peek_action.setChecked(state.flat_peek)
        self._on_negative_peek_changed(state.negative_peek)
        self._on_zones_changed(state.zones_overlay)
        self._on_grain_focuser_changed(state.grain_focuser)

        geo = state.config.geometry
        self.btn_flip_h.setChecked(geo.flip_horizontal)
        self.btn_flip_v.setChecked(geo.flip_vertical)
        self._ov_flip_h_action.setChecked(geo.flip_horizontal)
        self._ov_flip_v_action.setChecked(geo.flip_vertical)
        self._ov_immersive_action.setChecked(state.immersive_canvas)
        self._ov_sticky_zoom_action.setChecked(state.sticky_zoom)
        self._ov_invert_zoom_action.setChecked(state.invert_zoom_scroll)

        self.btn_undo.setEnabled(state.undo_index > 0)
        self.btn_redo.setEnabled(state.undo_index < state.max_history_index)
        self._ov_undo_action.setEnabled(state.undo_index > 0)
        self._ov_redo_action.setEnabled(state.undo_index < state.max_history_index)
        self._action_paste.setEnabled(state.clipboard is not None)

    @staticmethod
    def _toolbar_width_budget(canvas_width: int) -> int:
        """Horizontal space the pill may occupy inside the canvas."""
        return max(240, canvas_width - 2 * THEME.space_xl)

    def _activate_layout(self) -> None:
        layout = self.layout()
        if layout is not None:
            layout.activate()
        container = self._toolbar_container
        if container is not None:
            inner = container.layout()
            if inner is not None:
                inner.activate()

    def _pill_width(self) -> int:
        """Measured pill width after the current visibility set (sizeHint can stay stale)."""
        self._activate_layout()
        self.adjustSize()
        return self.minimumSizeHint().width()

    def pill_size_hint(self) -> QSize:
        """Preferred floating size for the canvas layout pass."""
        self._activate_layout()
        self.adjustSize()
        base = self.sizeHint()
        return QSize(self._pill_width(), base.height())

    def _rebuild_row(self) -> None:
        """Lay the row out as left anchor, the chosen items, then overflow and right anchor.

        Widgets left out stay parented but hidden: they are still state holders the overflow
        menu and the controller sync against."""
        layout = self._row_layout
        while layout.count():
            layout.takeAt(0)
        for widget, is_separator in self._row_sequence:
            if is_separator:
                widget.setParent(None)
                widget.deleteLater()
        self._row_sequence = []

        for widget in self._row_widgets.values():
            widget.setVisible(False)

        layout.addWidget(self.btn_toggle_left)
        previous_category = None
        for item_id in self._item_ids:
            item = TOOLBAR_ITEM_BY_ID[item_id]
            if previous_category is not None and item.category != previous_category:
                separator = self._create_separator()
                layout.addWidget(separator)
                self._row_sequence.append((separator, True))
            previous_category = item.category
            widget = self._row_widgets[item_id]
            layout.addWidget(widget)
            widget.setVisible(True)
            self._row_sequence.append((widget, False))
        layout.addWidget(self.btn_overflow)
        layout.addWidget(self.btn_toggle_right)
        self._sync_separators()

    def _sync_separators(self) -> None:
        """A divider earns its place only between two visible controls."""
        seen_visible = False
        pending: list[QWidget] = []
        for widget, is_separator in self._row_sequence:
            if is_separator:
                pending.append(widget)
            elif not widget.isHidden():
                for i, separator in enumerate(pending):
                    separator.setVisible(seen_visible and i == 0)
                pending.clear()
                seen_visible = True
        for separator in pending:
            separator.setVisible(False)

    def open_toolbar_editor(self) -> None:
        from PyQt6.QtWidgets import QDialog

        from negpy.desktop.view.widgets.favourites_dialog import FavouritesDialog

        dialog = FavouritesDialog(
            self,
            [(item.id, item.category, item.label) for item in TOOLBAR_ITEMS],
            self._item_ids,
            title="Edit Toolbar",
            chosen_header="TOOLBAR",
            hint="Whatever does not fit the canvas width collapses from the right. The ⋯ menu always holds every action, whichever ones the row shows.",
            defaults=list(DEFAULT_TOOLBAR_IDS),
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        self._item_ids = dialog.selected_ids()
        self.session.repo.save_global_setting(_TOOLBAR_SETTING_KEY, self._item_ids)
        self._rebuild_row()
        self._update_ui_state()
        canvas = self.parentWidget()
        if canvas is not None and hasattr(canvas, "relayout_floating_widgets"):
            canvas.relayout_floating_widgets()
        else:
            self.set_available_width(self._available_width or self.width())

    def set_available_width(self, w: int) -> None:
        """Show as many of the chosen controls as fit the canvas width.

        Hiding runs from the end of the user's order inward, so the items they put first are
        the ones that survive a narrow canvas. The overflow menu is not touched here — it
        always carries the full action set (see _init_ui), so a control leaving the row never
        changes what the menu contains."""
        self._available_width = w
        budget = self._toolbar_width_budget(w)

        widgets = [self._row_widgets[item_id] for item_id in self._item_ids]
        for widget in widgets:
            widget.setVisible(True)
        self._sync_separators()

        for widget in reversed(widgets):
            if self._pill_width() <= budget:
                break
            widget.setVisible(False)
            self._sync_separators()

        self._activate_layout()
        self.adjustSize()
