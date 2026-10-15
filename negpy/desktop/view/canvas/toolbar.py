from dataclasses import dataclass

import qtawesome as qta
from PyQt6.QtCore import QSize, Qt
from PyQt6.QtGui import QActionGroup, QCursor, QIcon
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QMenu,
    QToolButton,
    QVBoxLayout,
    QWidget,
    QWidgetAction,
)

from negpy.desktop.controller import AppController
from negpy.desktop.session import ZONE_STYLES, ToolMode
from negpy.desktop.view.canvas.hud import INFO_LEVEL_LABELS
from negpy.desktop.view.canvas.overlay import zone_colors
from negpy.desktop.view.canvas.zone_legend import ZoneLegend
from negpy.desktop.view.keyboard_shortcuts import _context_undo
from negpy.desktop.view.widgets.granular_settings_dialog import open_paste_dialog, open_sync_bounds_dialog
from negpy.desktop.view.shortcut_registry import label_with_shortcut, tooltip_with_shortcut
from negpy.desktop.view.widgets.choice_button import ChoiceButton, MenuButton, SegmentedChoice, ToggleMenuButton
from negpy.desktop.view.widgets.collapsible import hidden_by_gating, roll_revert_icon
from negpy.desktop.view.styles.templates import (
    EditedDot,
    default_button_height,
    field_label,
    hint_label,
    icon_button,
    labeled_action,
    section_subheader,
    wrap_tooltip,
)
from negpy.desktop.view.styles.theme import THEME

CANVAS_COLORS = [
    (THEME.canvas_bg_black, (0.02, 0.02, 0.02), "Black"),
    (THEME.canvas_bg_dark_grey, (0.11, 0.11, 0.11), "Dark Gray"),
    (THEME.canvas_bg_mid_grey, (0.25, 0.25, 0.25), "Mid Gray"),
    (THEME.canvas_bg_white, (1.0, 1.0, 1.0), "White"),
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
    ToolbarItem("position", "Navigation", "Frame Counter"),
    ToolbarItem("next", "Navigation", "Next"),
    ToolbarItem("zoom", "Zoom", "Zoom"),
    ToolbarItem("rot_l", "Geometry", "Rotate CCW"),
    ToolbarItem("rot_r", "Geometry", "Rotate CW"),
    ToolbarItem("flip_h", "Geometry", "Flip Horizontal"),
    ToolbarItem("flip_v", "Geometry", "Flip Vertical"),
    ToolbarItem("tools", "Tools", "Tools"),
    ToolbarItem("undo", "Edit", "Undo"),
    ToolbarItem("redo", "Edit", "Redo"),
    ToolbarItem("source", "View", "Print or Peek"),
    ToolbarItem("overlays", "View", "Overlays"),
    ToolbarItem("reference", "View", "Reference View"),
    ToolbarItem("light_table", "View", "Light Table"),
    ToolbarItem("copy", "Settings", "Copy Settings"),
    ToolbarItem("copy_bounds", "Settings", "Copy Settings + Bounds"),
    ToolbarItem("paste", "Settings", "Paste Settings"),
    ToolbarItem("sync_bounds", "Settings", "Sync Bounds…"),
    ToolbarItem("reset", "Settings", "Reset Settings"),
    ToolbarItem("reset_to_roll", "Settings", "Reset to Roll Settings"),
    ToolbarItem("unload", "Settings", "Unload…"),
    ToolbarItem("palette", "App", "Find Control or Action…"),
    ToolbarItem("preferences", "App", "Preferences…"),
    ToolbarItem("shortcuts", "App", "Keyboard Shortcuts…"),
)

TOOLBAR_ITEM_BY_ID = {item.id: item for item in TOOLBAR_ITEMS}

DEFAULT_TOOLBAR_IDS: tuple[str, ...] = (
    "prev",
    "position",
    "next",
    "zoom",
    "rot_l",
    "rot_r",
    "flip_h",
    "flip_v",
    "tools",
    "undo",
    "redo",
    "source",
    "overlays",
)

# Buttons folded into one menu control, so a stored row keeps what it showed.
_RETIRED_IDS = {
    "zoom_label": "zoom",
    "zoom_fit": "zoom",
    "zoom_original": "zoom",
    "hq": "zoom",
    "compare": "source",
    "flat_peek": "source",
    "negative_peek": "source",
    "embedded_peek": "source",
    "zones": "overlays",
    "loupe": "overlays",
}

_TOOLBAR_SETTING_KEY = "toolbar_items"

# A new item reaches a stored row once, before its anchor (else last); the record keeps a removed one out.
_ADDED_IDS = {"tools": "undo"}
_ADDED_SETTING_KEY = "toolbar_added_items"

# A narrow canvas hides these first to last; items not listed (menu actions put on the row) go
# before them, from the end of the user's order.
_COLLAPSE_ORDER = (
    "flip_v",
    "flip_h",
    "rot_r",
    "rot_l",
    "light_table",
    "reference",
    "overlays",
    "redo",
    "undo",
    "tools",
    "source",
    "zoom",
    "position",
    "next",
    "prev",
)

_ZOOM_PRESETS = (25, 50, 100, 200, 400)

# (state flag, icon, label, shortcut id, tooltip); index 0 is the edit itself.
_VIEW_SOURCES: tuple[tuple[str, str, str, str, str], ...] = (
    ("", "fa5s.image", "Print", "", "The edit, as it prints"),
    ("compare_mode", "fa5s.adjust", "Before / After", "toggle_compare", "Split against the auto baseline; drag the divider"),
    (
        "negative_peek",
        "fa5s.film",
        "Negative",
        "toggle_negative_peek",
        "The source as it was loaded, un-inverted and unedited, at your crop and rotation (no color management)",
    ),
    (
        "flat_peek",
        "fa5s.eye",
        "Flat Scan",
        "toggle_flat_peek",
        "The flat master: the scan as a log-encoded digital intermediate, with no print curve",
    ),
    (
        "embedded_peek",
        "fa5s.camera",
        "Embedded",
        "toggle_embedded_peek",
        "Embedded preview: the camera's own JPEG of this capture, as a reference for what the scan looks like",
    ),
    (
        "flatfield_peek",
        "fa5s.lightbulb",
        "Flat Field",
        "toggle_flatfield_peek",
        "Flat Field check: how well the Flat Field profile corrects its own reference. Choose a profile in the Flat Field card to use it.",
    ),
)

_SOURCE_INDEX = {flag: i for i, (flag, *_rest) in enumerate(_VIEW_SOURCES)}

_BRUSH_SIZE = "controls_panel.retouch_sidebar.manual_size_slider"
_FINE_ROTATION = "controls_panel.geometry_sidebar.fine_rot_slider"

# The row an armed canvas tool shows in place of the view row: (name, hint, slider path from MainWindow).
_TOOL_ROWS: dict[ToolMode, tuple[str, str, str]] = {
    ToolMode.WB_PICK: ("WB Picker", "Click a neutral gray; the selected region's CMY is solved so it prints neutral", ""),
    ToolMode.CROP_MANUAL: ("Crop", "Drag a corner or an edge; drag outside the box to redraw it", _FINE_ROTATION),
    ToolMode.DUST_PICK: ("Heal", "Paint over dust; only the marks inside the brush are repaired", _BRUSH_SIZE),
    ToolMode.SCRATCH_PICK: ("Scratch", "Click points along the scratch; Backspace deletes the last point", _BRUSH_SIZE),
    ToolMode.SCRATCH_LINE: (
        "Transport Line",
        "Click once on the line; the whole line is traced and repaired",
        "controls_panel.retouch_sidebar.line_threshold_slider",
    ),
    ToolMode.CLONE: ("Clone", "Paint over the defect; the dashed circle marks the source, Alt-click picks a new one", _BRUSH_SIZE),
    ToolMode.LOCAL_DRAW: ("Draw Mask", "Click to place vertices; a click near the start closes the mask", ""),
    ToolMode.LOCAL_OVAL: ("Oval Mask", "Drag out an oval", ""),
    ToolMode.LOCAL_GRADIENT: ("Card Edge", "Drag from the full-exposure edge to where it fades out", ""),
    ToolMode.ANALYSIS_DRAW: ("Draw Region", "Drag the area the meters read", ""),
    ToolMode.STRAIGHTEN: ("Straighten", "Draw a line along a horizon or an edge to level the frame", _FINE_ROTATION),
    ToolMode.KEYSTONE_LINES: (
        "Keystone Lines",
        "Drag a line along each converging edge; four lines solve Tilt and Swing",
        "controls_panel.geometry_sidebar.converge_v_slider",
    ),
    ToolMode.ZONE_PLACE: ("Zone Placement", "Click the photo to pin a spot; drag a pin to move it", ""),
}
_CLONE_SOURCE_HINT = "Click the photo to pick the area to copy from"

# The Tools menu, in panel order: a panel section's title, then its tools as (tool, action id, sidebar button
# path from MainWindow). Zone Placement has no entry: it arms only with a zone picked on the zone strip.
_CANVAS_TOOLS: tuple[tuple[ToolMode, str, str] | str, ...] = (
    "Geometry",
    (ToolMode.CROP_MANUAL, "manual_crop", "controls_panel.geometry_sidebar.manual_crop_btn"),
    (ToolMode.STRAIGHTEN, "straighten", "controls_panel.geometry_sidebar.straighten_btn"),
    (ToolMode.KEYSTONE_LINES, "keystone_lines", "controls_panel.geometry_sidebar.keystone_lines_btn"),
    "Metering",
    (ToolMode.ANALYSIS_DRAW, "analysis_draw", "controls_panel.process_sidebar.analysis_region_btn"),
    "Filtration",
    (ToolMode.WB_PICK, "pick_wb", "controls_panel.color_sidebar.pick_wb_btn"),
    "Dodge & Burn",
    (ToolMode.LOCAL_DRAW, "local_draw", "controls_panel.local_sidebar.draw_btn"),
    (ToolMode.LOCAL_OVAL, "local_oval", "controls_panel.local_sidebar.oval_btn"),
    (ToolMode.LOCAL_GRADIENT, "local_gradient", "controls_panel.local_sidebar.gradient_btn"),
    "Retouch",
    (ToolMode.DUST_PICK, "pick_dust", "controls_panel.retouch_sidebar.pick_dust_btn"),
    (ToolMode.SCRATCH_PICK, "pick_scratch", "controls_panel.retouch_sidebar.pick_scratch_btn"),
    (ToolMode.SCRATCH_LINE, "pick_scratch_line", "controls_panel.retouch_sidebar.pick_line_btn"),
    (ToolMode.CLONE, "clone_tool", "controls_panel.retouch_sidebar.clone_btn"),
)
_TOOL_BUTTON_PATHS = {spec[0]: spec[2] for spec in _CANVAS_TOOLS if not isinstance(spec, str)}


def _checkable_menu_icon(name: str):
    """The icon takes the checkmark's place in the menu, so the accent marks the item on."""
    return qta.icon(name, color=THEME.text_primary, color_on=THEME.accent_primary, color_disabled=THEME.text_muted)


def frame_position(model, actual_idx: int) -> str:
    """n / total in the Film Strip's order and filter, the one Next and Previous follow."""
    total = model.rowCount()
    row = model.actual_to_display(actual_idx)
    return f"{row + 1} / {total}" if total > 1 and row >= 0 else ""


def _migrate_retired(stored: list[str]) -> list[str]:
    out: list[str] = []
    for item_id in stored:
        new_ids = [_RETIRED_IDS.get(item_id, item_id)] + (["position"] if item_id == "prev" else [])
        out.extend(new_id for new_id in new_ids if new_id not in out)
    return out


def _add_new_items(stored: list[str], added: list[str]) -> list[str]:
    out = list(stored)
    for item_id, anchor in _ADDED_IDS.items():
        if item_id in added or item_id in out:
            continue
        out.insert(out.index(anchor) if anchor in out else len(out), item_id)
    return out


def load_toolbar_items(repo) -> list[str]:
    """No stored row means the stock one; unknown ids drop so a retired button degrades quietly."""
    stored = repo.get_global_setting(_TOOLBAR_SETTING_KEY)
    if not isinstance(stored, list):
        return list(DEFAULT_TOOLBAR_IDS)
    # Saved back at once, so the frame counter a migration adds is not re-added after a removal.
    if any(item_id in _RETIRED_IDS for item_id in stored):
        stored = _migrate_retired(stored)
        repo.save_global_setting(_TOOLBAR_SETTING_KEY, stored)
    added = repo.get_global_setting(_ADDED_SETTING_KEY, [])
    added = added if isinstance(added, list) else []
    if any(item_id not in added for item_id in _ADDED_IDS):
        stored = _add_new_items(stored, added)
        repo.save_global_settings({_TOOLBAR_SETTING_KEY: stored, _ADDED_SETTING_KEY: sorted(set(added) | set(_ADDED_IDS))})
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
        self._update_ui_state()

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

        # Controls need a parent before _rebuild_row() applies visibility.
        self._view_row = QWidget()
        row_layout = QHBoxLayout(self._view_row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(6)
        v_layout.addWidget(self._view_row)

        icon_color = THEME.text_primary
        icon_size = QSize(16, 16)
        btn_height = default_button_height()
        self._shortcut_tips: list = []
        self._shortcut_labels: list = []

        # 0. Panel toggles (live at the toolbar's outer edges)
        # Not checkable: an open panel is the resting state, and the accent means armed.
        self.btn_toggle_left = QToolButton()
        self.btn_toggle_left.setIcon(qta.icon("fa5s.columns", color=icon_color))
        self._tip(self.btn_toggle_left, "Toggle Session Panel", "toggle_left_panel")
        self.btn_toggle_right = QToolButton()
        self.btn_toggle_right.setIcon(qta.icon("fa5s.sliders-h", color=icon_color))
        self._tip(self.btn_toggle_right, "Toggle Controls Panel", "toggle_right_panel")

        # 1. Navigation
        self.btn_prev = QToolButton()
        self.btn_prev.setIcon(qta.icon("fa5s.chevron-left", color=icon_color))
        self._tip(self.btn_prev, "Previous", "prev_file")
        self.btn_next = QToolButton()
        self.btn_next.setIcon(qta.icon("fa5s.chevron-right", color=icon_color))
        self._tip(self.btn_next, "Next", "next_file")

        # Undo and Redo live in the main toolbar (mdi arrows, distinct from the rotate icons'
        # file-with-arrow glyphs below).
        self.btn_undo = QToolButton()
        self.btn_undo.setIcon(qta.icon("mdi.undo", color=icon_color))
        self._tip(self.btn_undo, "Undo", "undo")
        self.btn_redo = QToolButton()
        self.btn_redo.setIcon(qta.icon("mdi.redo", color=icon_color))
        self._tip(self.btn_redo, "Redo", "redo")

        # 2. Geometry
        self.btn_rot_l = QToolButton()
        self.btn_rot_l.setIcon(qta.icon("mdi6.file-rotate-left", color=icon_color))
        self._tip(self.btn_rot_l, "Rotate CCW", "rotate_ccw")
        self.btn_rot_r = QToolButton()
        self.btn_rot_r.setIcon(qta.icon("mdi6.file-rotate-right", color=icon_color))
        self._tip(self.btn_rot_r, "Rotate CW", "rotate_cw")
        self.btn_flip_h = QToolButton()
        self.btn_flip_h.setCheckable(True)
        self.btn_flip_h.setIcon(qta.icon("fa5s.arrows-alt-h", color=icon_color))
        self._tip(self.btn_flip_h, "Flip Horizontal", "flip_h")
        self.btn_flip_v = QToolButton()
        self.btn_flip_v.setCheckable(True)
        self.btn_flip_v.setIcon(qta.icon("fa5s.arrows-alt-v", color=icon_color))
        self._tip(self.btn_flip_v, "Flip Vertical", "flip_v")

        self.position_label = QLabel()
        self.position_label.setFixedHeight(btn_height)
        self.position_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.position_label.setStyleSheet(f"color: {THEME.text_secondary}; font-size: {THEME.font_size_base}px; padding: 0 2px;")
        self.position_label.setToolTip("This frame's place in the Film Strip's order and filter")

        # 3. Zoom: the readout opens the presets, since users zoom directly on the canvas.
        self.btn_zoom = MenuButton("Zoom — the true pixel zoom, 100% being one scan pixel per screen pixel. Pick a preset or HQ Preview.")
        zoom_menu = self.btn_zoom.choice_menu
        self._zoom_fit_action = zoom_menu.addAction("Fit to Window")
        self._zoom_fit_action.triggered.connect(self._on_fit_clicked)
        self._label(self._zoom_fit_action, "Fit to Window", "fit_view")
        zoom_menu.addSeparator()
        for percent in _ZOOM_PRESETS:
            action = zoom_menu.addAction(f"{percent}%")
            action.triggered.connect(lambda _checked=False, p=percent: self._on_zoom_preset(p))
            if percent == 100:
                self._label(action, "100%", "zoom_100")
                action.setToolTip("Below HQ the preview is scaled up to it, so the framing is right and the detail is not.")
        zoom_menu.addSeparator()
        self._zoom_hq_action = zoom_menu.addAction("HQ Preview")
        self._zoom_hq_action.setCheckable(True)
        self._zoom_hq_action.setToolTip("Render the full-resolution scan instead of a downscaled preview")
        self._zoom_hq_action.triggered.connect(self.controller.toggle_hq_preview)
        self._label(self._zoom_hq_action, "HQ Preview", "toggle_hq")
        # Sized for the common readouts, so zooming up to 1:1 does not move the row.
        self.btn_zoom.setText("100%")
        self._zoom_min_width = self.btn_zoom.sizeHint().width()
        self.btn_zoom.setMinimumWidth(self._zoom_min_width)

        # Every source here takes the whole canvas, and the controller keeps them exclusive.
        self.source_choice = ChoiceButton(
            tuple((icon, label) for _flag, icon, label, _key, _tip in _VIEW_SOURCES),
            "What the canvas shows: the print, a before/after split, or a peek that leaves your edit alone",
        )
        for i, (_flag, icon, label, key, tip) in enumerate(_VIEW_SOURCES):
            self.source_choice.choice_action(i).setIcon(_checkable_menu_icon(icon))
            if key:
                self._label(self.source_choice.choice_action(i), label, key)
            self.source_choice.set_choice_tooltip(i, tip)

        self.overlays_menu = ToggleMenuButton(
            "fa5s.layer-group", "", "Overlays — marks drawn over the frame; none of them changes the edit"
        )
        self._overlay_actions = {}
        for flag, icon, label, key, tip, toggle in (
            (
                "zones_overlay",
                "mdi6.numeric-5-box-outline",
                "Zone Overlay",
                "toggle_zones",
                "Show which Adams zone each part of the print falls in",
                self.controller.toggle_zones_overlay,
            ),
            (
                "clipping_overlay",
                "fa5s.adjust",
                "Clipping Overlay",
                "toggle_clipping_overlay",
                "Mark paper black and paper white on the frame",
                self.controller.toggle_clipping_overlay,
            ),
            (
                "printing_notes",
                "mdi.playlist-edit",
                "Printing Notes",
                "toggle_printing_notes",
                "The dodge and burn map and the print recipe",
                self.controller.toggle_printing_notes,
            ),
            (
                "grain_focuser",
                "fa5s.search-plus",
                "Grain Focuser",
                "toggle_grain_focuser",
                "A loupe at the cursor showing the frame's own pixels, with an acutance figure for comparing sharpness across the frame (reads true on HQ)",
                self.controller.toggle_grain_focuser,
            ),
            (
                "cursor_probe",
                "fa5s.crosshairs",
                "Probe at Cursor",
                "toggle_cursor_probe",
                "Zone, print density and ΔD of the pixel under the pointer, in a chip beside it",
                self.controller.toggle_cursor_probe,
            ),
        ):
            action = self.overlays_menu.add_toggle(label, tip)
            action.setIcon(_checkable_menu_icon(icon))
            self._label(action, label, key)
            action.triggered.connect(lambda checked, toggle=toggle: toggle(force=checked))
            self._overlay_actions[flag] = action
        # A submenu, not a toggle: the pills are on by default and the button lights only for marks.
        self.overlays_menu.choice_menu.addSeparator()
        info_menu = self.overlays_menu.choice_menu.addMenu("Frame Info")
        info_menu.setToolTipsVisible(True)
        info_group = QActionGroup(self)
        self._info_actions = []
        for level, label in enumerate(INFO_LEVEL_LABELS):
            action = info_menu.addAction(label)
            action.setCheckable(True)
            info_group.addAction(action)
            action.triggered.connect(lambda _checked=False, level=level: self.controller.set_canvas_info_level(level))
            self._info_actions.append(action)
        self._info_actions[0].setToolTip("No corner pills")
        self._info_actions[1].setToolTip("File name, scan size and film mode")
        self._info_actions[2].setToolTip("Adds the frame's place in the strip, film, camera and lens, the proof and the export size")

        self.btn_tools = MenuButton("Tools — crop, straighten, the pickers, masks and retouching; each opens its own row here")
        self.btn_tools.setIcon(qta.icon("fa5s.tools", color=icon_color, color_disabled=THEME.text_muted))
        self.btn_tools.setMinimumWidth(self.btn_tools.sizeHint().width())
        self._fill_tools_menu(self.btn_tools.choice_menu)

        # One-shot actions that otherwise live only in the ⋯ menu.
        self.btn_reference = self._action_button(
            qta.icon("fa5s.thumbtack", color=icon_color),
            "Reference view — pin this frame beside the canvas to match others to it",
            "toggle_reference",
            self._toggle_reference,
        )
        self.btn_light_table = self._action_button(
            qta.icon("fa5s.th", color=icon_color),
            "Light Table — the roll as a grid in place of the canvas",
            "toggle_light_table",
            self._show_light_table,
        )
        self.btn_copy = self._action_button(
            qta.icon("fa5s.copy", color=icon_color), "Copy this image's settings to the clipboard", "copy", self.session.copy_settings
        )
        self.btn_copy_bounds = self._action_button(
            qta.icon("fa5s.clone", color=icon_color),
            "Copy settings plus the metering/normalization bounds",
            "copy_with_bounds",
            self.session.copy_settings_with_bounds,
        )
        self.btn_paste = self._action_button(
            qta.icon("fa5s.paste", color=icon_color),
            "Paste the copied settings onto this frame, or every selected frame",
            "paste",
            lambda: open_paste_dialog(self, self.controller),
        )
        self.btn_sync_bounds = self._action_button(
            qta.icon("fa5s.crosshairs", color=icon_color),
            "Sync Bounds — give other frames this image's metering bounds, and nothing else",
            "sync_bounds",
            lambda: open_sync_bounds_dialog(self, self.session),
        )
        self.btn_reset = self._action_button(
            qta.icon("fa5s.history", color=icon_color),
            "Reset Settings — discard all edits and return this image to its default look",
            None,
            self.session.reset_settings,
        )
        self.btn_reset_to_roll = self._action_button(
            roll_revert_icon(icon_color),
            "Reset to Roll Settings — return every card that differs from the roll to the roll's settings",
            "reset_to_roll",
            self.controller.revert_frame_to_roll,
        )
        self.btn_unload = self._action_button(
            qta.icon("fa5s.times-circle", color=icon_color),
            "Unload — remove this image from the session (its saved edit is kept)",
            None,
            self._on_overflow_unload,
        )
        self.btn_palette = self._action_button(
            qta.icon("fa5s.search", color=icon_color),
            "Find any slider, card or action by name, and open it",
            "command_palette",
            self._show_palette,
        )
        self.btn_preferences = self._action_button(
            qta.icon("fa5s.cog", color=icon_color),
            "Preferences — interface, performance and storage settings for the whole app",
            "open_preferences",
            self._show_preferences,
        )
        self.btn_shortcuts = self._action_button(
            qta.icon("fa5s.keyboard", color=icon_color),
            "Show the full keyboard shortcuts reference",
            "show_shortcuts",
            self._show_shortcuts,
        )

        # 4. Overflow menu & responsive groups
        self.btn_overflow = QToolButton()
        self.btn_overflow.setObjectName("toolbar_overflow")
        self.btn_overflow.setIcon(qta.icon("fa5s.ellipsis-h", color=icon_color))
        self.btn_overflow.setToolTip("More actions")
        self.btn_overflow.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)

        overflow_menu = QMenu(self.btn_overflow)
        overflow_menu.setToolTipsVisible(True)  # so the GPU action can surface its backend/status

        # Overflow always mirrors the full action set, whatever the toolbar row shows at the
        # current canvas width. "More actions" is a stable, complete menu the user can always
        # find everything in, not a residue of the row's responsive collapse. It used to lose
        # entries whenever a side panel toggle gave the row enough width to show them directly.
        # A checkable item carries no icon: the stylesheet puts a menu icon in the check
        # column, which hides the checkmark that says the view is on.
        find_action = overflow_menu.addAction(qta.icon("fa5s.search", color=icon_color), "Find Control or Action…", self._show_palette)
        self._label(find_action, "Find Control or Action…", "command_palette")
        find_action.setToolTip("Find any slider, card or action by name, and open it")
        self._fill_tools_menu(overflow_menu.addMenu(qta.icon("fa5s.tools", color=icon_color), "Tools"))
        overflow_menu.addSeparator()

        self._ov_hq_action = overflow_menu.addAction("HQ Preview")
        self._ov_hq_action.setCheckable(True)
        self._menu_tip(self._ov_hq_action, "Toggle high-quality (full-resolution) preview", "toggle_hq")
        overflow_menu.addSeparator()

        self._ov_fit_action = overflow_menu.addAction(qta.icon("fa5s.expand", color=icon_color), "Fit to Window")
        self._menu_tip(self._ov_fit_action, "Fit to Window", "fit_view")
        self._ov_original_action = overflow_menu.addAction("Original Size (1:1)")
        self._ov_original_action.setCheckable(True)
        self._menu_tip(
            self._ov_original_action,
            "Original size (100%) — one scan pixel per screen pixel. Below HQ the preview is scaled up to it, "
            "so the framing is right and the detail is not.",
            "zoom_100",
        )
        reference_action = overflow_menu.addAction("Reference View", self._toggle_reference)
        self._menu_tip(reference_action, "Reference view — pin this frame beside the canvas to match others to it", "toggle_reference")
        light_table_action = overflow_menu.addAction("Light Table", self._show_light_table)
        self._menu_tip(light_table_action, "Light Table — the roll as a grid in place of the canvas", "toggle_light_table")
        self._ov_compare_action = overflow_menu.addAction("Before / After")
        self._ov_compare_action.setCheckable(True)
        self._menu_tip(self._ov_compare_action, "Before / After — split against the auto baseline, drag the divider", "toggle_compare")
        self._ov_flat_peek_action = overflow_menu.addAction("Peek Flat Scan")
        self._ov_flat_peek_action.setCheckable(True)
        self._menu_tip(
            self._ov_flat_peek_action, "Peek flat scan — temporarily show the flat master (does not change your edit)", "toggle_flat_peek"
        )
        self._ov_negative_peek_action = overflow_menu.addAction("Peek Negative")
        self._ov_negative_peek_action.setCheckable(True)
        self._menu_tip(
            self._ov_negative_peek_action,
            "Peek negative — show the source as it was loaded, un-inverted and unedited, at your crop and rotation (no color management)",
            "toggle_negative_peek",
        )
        self._ov_embedded_peek_action = overflow_menu.addAction("Peek Embedded Preview")
        self._ov_embedded_peek_action.setCheckable(True)
        self._menu_tip(
            self._ov_embedded_peek_action,
            "Peek embedded preview — the camera's own JPEG of this capture, as a reference for what the scan looks like",
            "toggle_embedded_peek",
        )
        self._ov_zones_action = overflow_menu.addAction("Zone Overlay")
        self._ov_zones_action.setCheckable(True)
        self._menu_tip(self._ov_zones_action, "Zone overlay — show which Adams zone each part of the print falls in", "toggle_zones")
        self._ov_loupe_action = overflow_menu.addAction("Grain Focuser")
        self._ov_loupe_action.setCheckable(True)
        self._menu_tip(
            self._ov_loupe_action,
            "Grain focuser — a loupe at the cursor showing the frame's own pixels, with an "
            "acutance figure for comparing sharpness across the frame (reads true on HQ)",
            "toggle_grain_focuser",
        )
        self._ov_clipping_action = overflow_menu.addAction("Clipping Overlay")
        self._ov_clipping_action.setCheckable(True)
        self._menu_tip(self._ov_clipping_action, "Mark paper black and paper white on the frame", "toggle_clipping_overlay")
        self._ov_notes_action = overflow_menu.addAction("Printing Notes")
        self._ov_notes_action.setCheckable(True)
        self._menu_tip(self._ov_notes_action, "The dodge and burn map and the print recipe", "toggle_printing_notes")
        self._ov_probe_action = overflow_menu.addAction("Probe at Cursor")
        self._ov_probe_action.setCheckable(True)
        self._menu_tip(self._ov_probe_action, "Zone, print density and ΔD of the pixel under the pointer", "toggle_cursor_probe")
        self._ov_undo_action = overflow_menu.addAction(qta.icon("mdi.undo", color=icon_color), "Undo")
        self._menu_tip(self._ov_undo_action, "Undo", "undo")
        self._ov_redo_action = overflow_menu.addAction(qta.icon("mdi.redo", color=icon_color), "Redo")
        self._menu_tip(self._ov_redo_action, "Redo", "redo")

        overflow_menu.addSeparator()
        self._ov_rot_l_action = overflow_menu.addAction(qta.icon("mdi6.file-rotate-left", color=icon_color), "Rotate CCW")
        self._menu_tip(self._ov_rot_l_action, "Rotate CCW", "rotate_ccw")
        self._ov_rot_r_action = overflow_menu.addAction(qta.icon("mdi6.file-rotate-right", color=icon_color), "Rotate CW")
        self._menu_tip(self._ov_rot_r_action, "Rotate CW", "rotate_cw")
        self._ov_flip_h_action = overflow_menu.addAction("Flip Horizontal")
        self._ov_flip_h_action.setCheckable(True)
        self._menu_tip(self._ov_flip_h_action, "Flip Horizontal", "flip_h")
        self._ov_flip_v_action = overflow_menu.addAction("Flip Vertical")
        self._ov_flip_v_action.setCheckable(True)
        self._menu_tip(self._ov_flip_v_action, "Flip Vertical", "flip_v")
        overflow_menu.addSeparator()

        self._action_copy = overflow_menu.addAction(qta.icon("fa5s.copy", color=icon_color), "Copy Settings", self.session.copy_settings)
        self._label(self._action_copy, "Copy Settings", "copy")
        self._action_copy.setToolTip("Copy this image's settings to the clipboard")
        self._action_copy_bounds = overflow_menu.addAction(
            qta.icon("fa5s.copy", color=icon_color),
            "Copy Settings + Bounds",
            self.session.copy_settings_with_bounds,
        )
        self._label(self._action_copy_bounds, "Copy Settings + Bounds", "copy_with_bounds")
        self._action_copy_bounds.setToolTip("Copy settings plus the metering/normalization bounds")
        self._action_paste = overflow_menu.addAction(
            qta.icon("fa5s.paste", color=icon_color),
            "Paste Settings",
            lambda: open_paste_dialog(self, self.controller),
        )
        self._label(self._action_paste, "Paste Settings", "paste")
        self._action_paste.setToolTip("Paste the copied settings onto this frame, or every selected frame")
        self._action_sync_bounds = overflow_menu.addAction(
            qta.icon("fa5s.crosshairs", color=icon_color),
            "Sync Bounds…",
            lambda: open_sync_bounds_dialog(self, self.session),
        )
        self._label(self._action_sync_bounds, "Sync Bounds…", "sync_bounds")
        self._action_sync_bounds.setToolTip("Give other frames this image's metering bounds, and nothing else")
        overflow_menu.addSeparator()
        reset_settings_action = overflow_menu.addAction(
            qta.icon("fa5s.history", color=icon_color), "Reset Settings", self.session.reset_settings
        )
        reset_settings_action.setToolTip("Discard all edits and return this image to its default look")
        self._action_reset_to_roll = overflow_menu.addAction(
            roll_revert_icon(icon_color), "Reset to Roll Settings", self.controller.revert_frame_to_roll
        )
        self._label(self._action_reset_to_roll, "Reset to Roll Settings", "reset_to_roll")
        self._action_reset_to_roll.setToolTip("Return every card that differs from the roll to the roll's settings")
        overflow_menu.aboutToShow.connect(lambda: self._action_reset_to_roll.setEnabled(self.controller.can_revert_frame_to_roll()))
        overflow_menu.addSeparator()
        unload_action = overflow_menu.addAction(qta.icon("fa5s.times-circle", color=icon_color), "Unload…", self._on_overflow_unload)
        unload_action.setToolTip("Remove this image from the session (its saved edit is kept)")
        overflow_menu.addSeparator()

        prefs_action = overflow_menu.addAction(qta.icon("fa5s.sliders-h", color=icon_color), "Preferences…", self._show_preferences)
        self._label(prefs_action, "Preferences…", "open_preferences")
        prefs_action.setToolTip("Interface, performance and storage settings for the whole app")
        overflow_menu.addSeparator()

        self._update_action = overflow_menu.addAction(
            qta.icon("fa5s.sync-alt", color=icon_color), "Check for Updates…", self._check_for_updates
        )
        about_action = overflow_menu.addAction(qta.icon("fa5s.info-circle", color=icon_color), "About NegPy…", self._show_about)
        about_action.setToolTip("Version and project page")
        tour_action = overflow_menu.addAction(qta.icon("fa5s.map-signs", color=icon_color), "Take the Tour", self._show_tour)
        self._label(tour_action, "Take the Tour", "show_tour")
        tour_action.setToolTip("Replay the guided feature tour")
        shortcuts_action = overflow_menu.addAction(qta.icon("fa5s.keyboard", color=icon_color), "Keyboard Shortcuts…", self._show_shortcuts)
        self._label(shortcuts_action, "Keyboard Shortcuts…", "show_shortcuts")
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
            self.btn_reference,
            self.btn_light_table,
            self.btn_copy,
            self.btn_copy_bounds,
            self.btn_paste,
            self.btn_sync_bounds,
            self.btn_reset,
            self.btn_reset_to_roll,
            self.btn_unload,
            self.btn_palette,
            self.btn_preferences,
            self.btn_shortcuts,
            self.btn_overflow,
        ]
        for btn in standard_buttons:
            btn.setIconSize(icon_size)
            btn.setFixedHeight(btn_height)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setToolTip(wrap_tooltip(btn.toolTip()))

        # The file-rotate glyphs (page plus arrow) read as a blob at the standard 16px icon size.
        # A touch larger keeps the page and arrow legible without changing the button's own
        # footprint, since btn_height still applies.
        rotate_icon_size = QSize(20, 20)
        for btn in (self.btn_rot_l, self.btn_rot_r):
            btn.setIconSize(rotate_icon_size)
            btn.setFixedHeight(btn_height)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setToolTip(wrap_tooltip(btn.toolTip()))
        for btn in (self.btn_zoom, self.btn_tools, self.source_choice, self.overlays_menu):
            btn.setCursor(Qt.CursorShape.PointingHandCursor)

        self._row_layout = row_layout
        self._row_widgets: dict[str, QWidget] = {
            "prev": self.btn_prev,
            "position": self.position_label,
            "next": self.btn_next,
            "zoom": self.btn_zoom,
            "rot_l": self.btn_rot_l,
            "rot_r": self.btn_rot_r,
            "flip_h": self.btn_flip_h,
            "flip_v": self.btn_flip_v,
            "tools": self.btn_tools,
            "undo": self.btn_undo,
            "redo": self.btn_redo,
            "source": self.source_choice,
            "overlays": self.overlays_menu,
            "reference": self.btn_reference,
            "light_table": self.btn_light_table,
            "copy": self.btn_copy,
            "copy_bounds": self.btn_copy_bounds,
            "paste": self.btn_paste,
            "sync_bounds": self.btn_sync_bounds,
            "reset": self.btn_reset,
            "reset_to_roll": self.btn_reset_to_roll,
            "unload": self.btn_unload,
            "palette": self.btn_palette,
            "preferences": self.btn_preferences,
            "shortcuts": self.btn_shortcuts,
        }
        self._row_sequence: list[tuple[QWidget, bool]] = []
        self._available_width = 0
        self._zoom_percent = 100
        self._item_ids = load_toolbar_items(self.session.repo)
        self._rebuild_row()

        self._build_tool_row(v_layout, icon_color, icon_size, btn_height)
        self._build_zone_row(v_layout)

        # Size the pill to its controls; don't stretch it across the canvas.
        main_layout.addWidget(container, 0, Qt.AlignmentFlag.AlignCenter)

    def _build_tool_row(self, v_layout: QVBoxLayout, icon_color: str, icon_size: QSize, btn_height: int) -> None:
        """While a canvas tool is armed the pill shows its own row: what the tool is, how to use it,
        the slider it is sized or tuned by, undo and Done."""
        self._tool_row = QWidget()
        self._tool_row.hide()
        layout = QHBoxLayout(self._tool_row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        self._tool_layout = layout
        self._shown_tool = ToolMode.NONE
        self._shown_rows = (False, False)  # (tool row, zone row); neither means the view row
        self._tool_slider = None
        self._tool_slider_source = None

        # Names the armed tool and switches to another one, with no Done in between.
        self.tool_name = MenuButton("Switch to another canvas tool")
        self._fill_tools_menu(self.tool_name.choice_menu)
        self.tool_hint = hint_label()
        self.tool_hint.setWordWrap(False)
        self.btn_tool_undo = QToolButton()
        self.btn_tool_undo.setIcon(qta.icon("mdi.undo", color=icon_color, color_disabled=THEME.text_muted))
        self._tip(self.btn_tool_undo, "Undo", "undo")
        self.btn_tool_undo.clicked.connect(lambda: _context_undo(self.controller))
        self.btn_tool_redo = QToolButton()
        self.btn_tool_redo.setIcon(qta.icon("mdi.redo", color=icon_color, color_disabled=THEME.text_muted))
        self._tip(self.btn_tool_redo, "Redo", "redo")
        self.btn_tool_redo.clicked.connect(self.session.redo)
        for btn in (self.btn_tool_undo, self.btn_tool_redo):
            btn.setIconSize(icon_size)
            btn.setFixedHeight(btn_height)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_tool_done = labeled_action("fa5s.check", "Done", "", primary=True)
        self._tip(
            self.btn_tool_done, "Finish the line or shape in progress and leave the tool. Esc leaves without finishing it.", "cancel_tool"
        )
        self.btn_tool_done.setFixedHeight(btn_height)
        # The pill lays out at minimum widths, which would cut the label down to the icon.
        self.btn_tool_done.setMinimumWidth(self.btn_tool_done.sizeHint().width())
        self.btn_tool_done.clicked.connect(self._on_tool_done)

        layout.addWidget(self.tool_name)
        layout.addWidget(self.tool_hint)
        self._tool_slider_index = layout.count()
        layout.addWidget(self._create_separator())
        layout.addWidget(self.btn_tool_undo)
        layout.addWidget(self.btn_tool_redo)
        layout.addWidget(self.btn_tool_done)
        v_layout.addWidget(self._tool_row)

    def _build_zone_row(self, v_layout: QVBoxLayout) -> None:
        """While the Zone Overlay is up the pill shows its look, its legend and a close button."""
        self._zone_row = QWidget()
        self._zone_row.hide()
        layout = QHBoxLayout(self._zone_row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        self.zone_style = SegmentedChoice(
            tuple(zip(("fa5s.palette", "mdi.gradient-horizontal", "mdi.grid"), ZONE_STYLES.values())),
            "Zone Overlay look: a color per zone over the photo, the print in eleven gray steps, or a grid of zone numerals",
            data=tuple(ZONE_STYLES),
        )
        self.zone_style.currentChanged.connect(lambda _i: self.controller.set_zones_style(self.zone_style.currentData()))
        self.zone_legend = ZoneLegend()
        self.zone_legend.zone_hovered.connect(self.controller.highlight_zone)
        self.btn_zone_close = icon_button("fa5s.times", "Close the Zone Overlay")
        self.btn_zone_close.clicked.connect(lambda: self.controller.toggle_zones_overlay(force=False))
        layout.addWidget(field_label("Zone overlay"))
        layout.addWidget(self.zone_style)
        layout.addWidget(self._create_separator())
        layout.addWidget(self.zone_legend)
        layout.addWidget(self.btn_zone_close)
        v_layout.addWidget(self._zone_row)

    def _sync_zone_row(self) -> None:
        state = self.session.state
        self.zone_style.blockSignals(True)
        self.zone_style.setCurrentIndex(max(0, self.zone_style.findData(state.zones_style)))
        self.zone_style.blockSignals(False)
        self.zone_legend.set_colors(zone_colors(state.zones_style, state.color_vision))

    def set_zone_shares(self, shares) -> None:
        self.zone_legend.set_shares(shares)

    def set_zone_pointer(self, zone) -> None:
        self.zone_legend.set_ringed(zone)

    def _sync_tool_row(self) -> None:
        """Re-entrant: a sidebar sync can cancel the tool from inside tool_sync_requested."""
        state = self.session.state
        mode = state.active_tool
        spec = _TOOL_ROWS.get(mode)
        if spec is not None:
            name, hint, _path = spec
            button = self._find_widget(_TOOL_BUTTON_PATHS.get(mode, ""))
            self.tool_name.setIcon(button.icon() if button is not None else QIcon())
            self.tool_name.setText(f" {name}" if button is not None else name)
            self.tool_name.setMinimumWidth(self.tool_name.sizeHint().width())
            self.tool_hint.setText(_CLONE_SOURCE_HINT if mode == ToolMode.CLONE and state.clone_picking else hint)
            self.tool_name.setToolTip(self.tool_hint.text())
            self.btn_tool_undo.setEnabled(state.undo_index > 0)
            self.btn_tool_redo.setEnabled(state.undo_index < state.max_history_index)
        # After the texts: the canvas sizes the pill from them.
        if mode != self._shown_tool:
            self._shown_tool = mode
            self._swap_tool_slider(spec[2] if spec else "")
        # An armed tool outranks the Zone Overlay, which outranks the view row.
        rows = (spec is not None, spec is None and bool(state.zones_overlay))
        if rows != self._shown_rows:
            self._shown_rows = rows
            if rows[1]:
                self._sync_zone_row()
            self._tool_row.setVisible(rows[0])
            self._zone_row.setVisible(rows[1])
            self._view_row.setVisible(not any(rows))
            canvas = self.parentWidget()
            if hasattr(canvas, "relayout_floating_widgets"):
                canvas.relayout_floating_widgets()
        if self._tool_slider is not None:
            self._tool_slider.setEnabled(self._tool_slider_source.isEnabled())

    def _find_widget(self, path: str):
        from negpy.desktop.view.main_window import MainWindow

        win = self.window()
        if not path or not isinstance(win, MainWindow):
            return None
        target = win
        for attr in path.split("."):
            target = getattr(target, attr)
        return target

    def _swap_tool_slider(self, path: str) -> None:
        """A mirror of the sidebar's slider, so a drag runs the sidebar's own binding (see clone_slider)."""
        from negpy.desktop.view.widgets.sliders import clone_slider

        if self._tool_slider is not None:
            self._tool_slider_source.valueChanged.disconnect(self._tool_slider.setValue)
            self._tool_slider.setParent(None)
            self._tool_slider.deleteLater()
            self._tool_slider = self._tool_slider_source = None
        source = self._find_widget(path)
        if source is None:
            return
        mirror = clone_slider(source)
        mirror.setValue(source.value())
        mirror.setFixedHeight(default_button_height())
        mirror.setFixedWidth(mirror.sizeHint().width())
        mirror.valueChanged.connect(lambda v, s=source: s.mirror_value(v, commit=False))
        mirror.valueCommitted.connect(lambda v, s=source: s.mirror_value(v, commit=True))
        # setValue does not emit, so the two never echo.
        source.valueChanged.connect(mirror.setValue)
        self._tool_layout.insertWidget(self._tool_slider_index, mirror)
        self._tool_slider, self._tool_slider_source = mirror, source

    def _fill_tools_menu(self, menu: QMenu) -> None:
        """Each item runs the tool's own shortcut action, which shows its tab and refuses a gated tool."""
        menu.setToolTipsVisible(True)
        entries = []
        for spec in _CANVAS_TOOLS:
            if isinstance(spec, str):
                header = section_subheader(spec)
                header.setContentsMargins(THEME.space_lg, 0, THEME.space_lg, 0)
                section = QWidgetAction(menu)
                section.setDefaultWidget(header)
                section.setEnabled(False)
                menu.addAction(section)
                continue
            mode, action_id, path = spec
            name, hint, _slider = _TOOL_ROWS[mode]
            action = menu.addAction(name)
            self._label(action, name, action_id)
            action.setToolTip(hint)
            action.triggered.connect(lambda _checked=False, a=action_id: self._arm_tool(a))
            entries.append((mode, path, action))

        def refresh() -> None:
            current = self.session.state.active_tool
            for mode, path, action in entries:
                button = self._find_widget(path)
                usable = button is None or (button.isEnabled() and not hidden_by_gating(button))
                if button is not None:
                    action.setIcon(button.icon())
                # The armed tool's own item would toggle it off.
                action.setEnabled(mode != current and usable)
                font = action.font()
                font.setBold(mode == current)
                action.setFont(font)

        menu.aboutToShow.connect(refresh)

    def _arm_tool(self, action_id: str) -> None:
        from negpy.desktop.view.main_window import MainWindow

        win = self.window()
        if isinstance(win, MainWindow):
            action = win.shortcut_manager.action_for(action_id)
            if action is not None:
                action()

    def open_tools_menu(self) -> None:
        """At the cursor: the row may have hidden the Tools button."""
        self.btn_tools.choice_menu.exec(QCursor.pos())

    def _on_tool_done(self) -> None:
        overlay = getattr(self.parentWidget(), "overlay", None)
        if overlay is not None:
            overlay.finish_in_progress()
        self.controller.cancel_active_tool()

    def _sync_view_state(self, *_args) -> None:
        """Mirror the canvas source and the overlays from state, which the controller sets before it signals."""
        state = self.session.state
        active = next((i for i, (flag, *_rest) in enumerate(_VIEW_SOURCES) if flag and getattr(state, flag)), 0)
        self.source_choice.blockSignals(True)
        self.source_choice.setCurrentIndex(active)
        self.source_choice.blockSignals(False)
        self.source_choice.set_choice_enabled(_SOURCE_INDEX["flatfield_peek"], bool(state.config.flatfield.profile_id))
        for flag, action in self._overlay_actions.items():
            action.setChecked(bool(getattr(state, flag)))
        for flag, action in self._ov_view_actions.items():
            action.setChecked(bool(getattr(state, flag)))
        self._info_actions[state.canvas_info_level].setChecked(True)

    def _on_source_chosen(self, index: int) -> None:
        state = self.session.state
        flag = _VIEW_SOURCES[index][0]
        if flag == "compare_mode":
            if not state.compare_mode:
                self.controller.toggle_compare()
        elif flag:
            self._peek_toggles[flag](force=True)
        else:
            if state.compare_mode:
                self.controller.exit_compare()
            for peek, toggle in self._peek_toggles.items():
                if getattr(state, peek):
                    toggle(force=False)
        # The controller can refuse (no frame, no embedded preview) and says why in a toast.
        self._sync_view_state()

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

        controller = self.controller
        self._peek_toggles = {
            "negative_peek": controller.toggle_negative_peek,
            "flat_peek": controller.toggle_flat_peek,
            "embedded_peek": controller.toggle_embedded_peek,
            "flatfield_peek": controller.toggle_flatfield_peek,
        }
        self._ov_view_actions = {
            "compare_mode": self._ov_compare_action,
            "flat_peek": self._ov_flat_peek_action,
            "negative_peek": self._ov_negative_peek_action,
            "embedded_peek": self._ov_embedded_peek_action,
            "zones_overlay": self._ov_zones_action,
            "grain_focuser": self._ov_loupe_action,
            "clipping_overlay": self._ov_clipping_action,
            "printing_notes": self._ov_notes_action,
            "cursor_probe": self._ov_probe_action,
        }
        self.source_choice.currentChanged.connect(self._on_source_chosen)
        for signal in (
            controller.compare_changed,
            controller.flat_peek_changed,
            controller.negative_peek_changed,
            controller.embedded_peek_changed,
            controller.flatfield_peek_changed,
            controller.zones_overlay_changed,
            controller.clipping_overlay_changed,
            controller.printing_notes_changed,
            controller.grain_focuser_changed,
            controller.cursor_probe_changed,
            controller.canvas_info_changed,
        ):
            signal.connect(self._sync_view_state)
        controller.zoom_changed.connect(self._on_zoom_changed)

        self.session.state_changed.connect(self._update_ui_state)
        self.session.asset_model.layoutChanged.connect(self._update_ui_state)
        controller.tool_sync_requested.connect(self._sync_tool_row)
        controller.zones_overlay_changed.connect(lambda _on: self._sync_tool_row())
        controller.zones_style_changed.connect(lambda _style: self._sync_zone_row())
        self.session.color_vision_changed.connect(self._sync_zone_row)

        # Overflow menu action connections
        self._ov_hq_action.triggered.connect(controller.toggle_hq_preview)
        self._ov_rot_l_action.triggered.connect(lambda: self.rotate(1))
        self._ov_rot_r_action.triggered.connect(lambda: self.rotate(-1))
        self._ov_flip_h_action.triggered.connect(lambda: self.flip("horizontal"))
        self._ov_flip_v_action.triggered.connect(lambda: self.flip("vertical"))
        self._ov_fit_action.triggered.connect(self._on_fit_clicked)
        self._ov_original_action.triggered.connect(self._on_original_clicked)
        self._ov_compare_action.triggered.connect(controller.toggle_compare)
        for flag, toggle in (
            ("flat_peek", controller.toggle_flat_peek),
            ("negative_peek", controller.toggle_negative_peek),
            ("embedded_peek", controller.toggle_embedded_peek),
            ("zones_overlay", self.controller.toggle_zones_overlay),
            ("grain_focuser", controller.toggle_grain_focuser),
            ("clipping_overlay", controller.toggle_clipping_overlay),
            ("printing_notes", self.controller.toggle_printing_notes),
            ("cursor_probe", controller.toggle_cursor_probe),
        ):
            self._ov_view_actions[flag].triggered.connect(lambda checked, toggle=toggle: toggle(force=checked))
        self._ov_undo_action.triggered.connect(lambda: _context_undo(controller))
        self._ov_redo_action.triggered.connect(self.session.redo)

    def _on_overflow_unload(self) -> None:
        from negpy.desktop.view.confirm import confirm_unload

        if self.session.state.selected_file_idx < 0:
            return
        if confirm_unload(self):
            self.session.remove_current_file()

    def _on_zoom_changed(self, zoom: float) -> None:
        # The readout is the true pixel zoom (zoom_level x fit_scale); zoom_level itself is fit-relative.
        canvas = getattr(self.controller, "canvas", None)
        self._zoom_percent = canvas.current_zoom_percent() if canvas is not None else int(round(max(0.0, zoom) * 100.0))
        self._ov_original_action.setChecked(self._zoom_percent == 100)
        self._refresh_zoom_readout()

    def _refresh_zoom_readout(self) -> None:
        hq = " HQ" if self.session.state.hq_preview else ""
        self.btn_zoom.setText(f"{self._zoom_percent}%{hq}")
        # The pill is laid out from minimum widths and the canvas fixes its size, so a longer readout
        # raises the minimum and asks for a new layout, or it clips under the chevron.
        width = max(self._zoom_min_width, self.btn_zoom.sizeHint().width())
        if width != self.btn_zoom.minimumWidth():
            self.btn_zoom.setMinimumWidth(width)
            canvas = self.parentWidget()
            if hasattr(canvas, "relayout_floating_widgets"):
                canvas.relayout_floating_widgets()

    def _on_zoom_preset(self, percent: int) -> None:
        canvas = getattr(self.controller, "canvas", None)
        if canvas is not None:
            canvas.zoom_to_percent(percent)

    def _on_fit_clicked(self) -> None:
        canvas = getattr(self.controller, "canvas", None)
        if canvas is not None:
            canvas.fit_to_window()

    def _on_original_clicked(self) -> None:
        canvas = getattr(self.controller, "canvas", None)
        if canvas is None:
            # Nothing will emit zoom_changed to correct the click's own check.
            self._ov_original_action.setChecked(False)
            return
        canvas.zoom_to_original()

    def rotate(self, direction: int) -> None:
        from dataclasses import replace

        from negpy.features.geometry.logic import rotate_geometry_and_analysis

        # A proof on the canvas takes the rotation instead of the image. Must precede the
        # handedness fix below, which is geometry-only.
        if self.controller.rotate_test_strip(direction):
            return

        state = self.session.state
        # A multi-selection that has explicitly excluded the active frame (ctrl-click
        # toggled it off) must not rotate it anyway: selection wins over "what's on
        # screen" once there is one, the same rule toggle_mark already uses.
        include_active = len(state.selected_indices) <= 1 or state.selected_file_idx in state.selected_indices
        if include_active:
            config = state.config
            new_geo, new_rect = rotate_geometry_and_analysis(config.geometry, config.process.analysis_rect, direction)
            new_config = replace(config, geometry=new_geo)
            if config.process.analysis_rect is not None:
                new_config = replace(new_config, process=replace(config.process, analysis_rect=new_rect))
            self.session.update_config(new_config, persist=True)
        # A multi-selection rotates every other selected frame too, each by its own
        # current geometry rather than a copy of the active frame's new one.
        before = self.controller.thumbnail_turn_snapshot()
        touched = self.session.rotate_selected_frames(direction, active_included=include_active)
        if touched:
            self.controller.rotate_thumbnails(touched, direction, before)
        if include_active:
            # Rotating must not drop an active before/after or flat-peek, so re-render in
            # place within whichever view is on.
            self.controller.rerender_active_view()

    def flip(self, axis: str) -> None:
        from dataclasses import replace

        from negpy.features.geometry.logic import flip_geometry_and_analysis

        horizontal = axis == "horizontal"
        state = self.session.state
        # See rotate(): a multi-selection that excludes the active frame skips it too.
        include_active = len(state.selected_indices) <= 1 or state.selected_file_idx in state.selected_indices
        if include_active:
            config = state.config
            new_geo, new_rect = flip_geometry_and_analysis(config.geometry, config.process.analysis_rect, horizontal)
            new_config = replace(config, geometry=new_geo)
            if config.process.analysis_rect is not None:
                new_config = replace(new_config, process=replace(config.process, analysis_rect=new_rect))
            self.session.update_config(new_config, persist=True)
        # A multi-selection flips every other selected frame too, each by its own
        # current geometry rather than a copy of the active frame's new one.
        before = self.controller.thumbnail_turn_snapshot()
        touched = self.session.flip_selected_frames(horizontal, active_included=include_active)
        if touched:
            self.controller.flip_thumbnails(touched, horizontal, before)
        if include_active:
            # Flipping shouldn't drop an active before/after or flat-peek (see rotate()).
            self.controller.rerender_active_view()

    def contextMenuEvent(self, event) -> None:  # noqa: N802
        menu = QMenu(self)
        menu.addAction(label_with_shortcut("Edit Toolbar…", "edit_toolbar"), self.open_toolbar_editor)
        menu.exec(event.globalPos())

    def _show_tour(self) -> None:
        from negpy.desktop.view.main_window import MainWindow

        win = self.window()
        if isinstance(win, MainWindow):
            win.show_tutorial()

    def set_update_available(self, version: str) -> None:
        if not hasattr(self, "_update_dot"):
            self._update_dot = EditedDot(self.btn_overflow, color=THEME.status_success)
        self._update_dot.set_active(True)
        self._update_action.setIcon(qta.icon("fa5s.download", color=THEME.status_success))
        self._update_action.setText(f"Update to v{version}…")

    def _check_for_updates(self) -> None:
        from negpy.desktop.view.main_window import MainWindow

        win = self.window()
        if isinstance(win, MainWindow):
            win.session_panel.check_for_updates()

    def _show_about(self) -> None:
        from negpy.desktop.view.main_window import MainWindow

        win = self.window()
        if isinstance(win, MainWindow):
            win.show_about()

    def _toggle_reference(self) -> None:
        from negpy.desktop.view.main_window import MainWindow

        win = self.window()
        if isinstance(win, MainWindow):
            win.toggle_reference()

    def _show_light_table(self) -> None:
        from negpy.desktop.view.main_window import MainWindow

        win = self.window()
        if isinstance(win, MainWindow):
            win.set_light_table(True)

    def _show_palette(self) -> None:
        from negpy.desktop.view.main_window import MainWindow

        win = self.window()
        if isinstance(win, MainWindow):
            win.show_command_palette()

    def _show_shortcuts(self) -> None:
        from negpy.desktop.view.widgets.shortcuts_overlay import ShortcutsOverlay

        dlg = ShortcutsOverlay(self.window().shortcut_manager, self.window(), repo=self.session.repo)
        dlg.exec()

    def _show_preferences(self) -> None:
        from negpy.desktop.view.widgets.preferences_dialog import open_preferences

        open_preferences(self.window(), self.controller)

    def _update_ui_state(self) -> None:
        state = self.session.state
        model = self.session.asset_model
        display_idx = model.actual_to_display(state.selected_file_idx)
        self.btn_prev.setEnabled(display_idx > 0)
        self.btn_next.setEnabled(0 <= display_idx < model.rowCount() - 1)
        self.position_label.setText(frame_position(model, state.selected_file_idx))
        self._ov_hq_action.setChecked(state.hq_preview)
        self._zoom_hq_action.setChecked(state.hq_preview)
        self._refresh_zoom_readout()
        self._sync_view_state()
        self._sync_tool_row()

        geo = state.config.geometry
        self.btn_flip_h.setChecked(geo.flip_horizontal)
        self.btn_flip_v.setChecked(geo.flip_vertical)
        self._ov_flip_h_action.setChecked(geo.flip_horizontal)
        self._ov_flip_v_action.setChecked(geo.flip_vertical)

        self.btn_undo.setEnabled(state.undo_index > 0)
        self.btn_redo.setEnabled(state.undo_index < state.max_history_index)
        self._ov_undo_action.setEnabled(state.undo_index > 0)
        self._ov_redo_action.setEnabled(state.undo_index < state.max_history_index)
        self._action_paste.setEnabled(state.clipboard is not None)
        self.btn_paste.setEnabled(state.clipboard is not None)
        # A roll lookup on every state change, so only while the button is on the row.
        if "reset_to_roll" in self._item_ids:
            self.btn_reset_to_roll.setEnabled(bool(self.controller.can_revert_frame_to_roll()))

    @staticmethod
    def _toolbar_width_budget(canvas_width: int) -> int:
        """Horizontal space the pill may occupy inside the canvas."""
        return max(240, canvas_width - 2 * THEME.space_xl)

    def _activate_layout(self) -> None:
        # Innermost first: an outer layout reads its rows' size hints as of their last activation.
        for widget in (self._view_row, self._tool_row, self._toolbar_container, self):
            layout = widget.layout()
            if layout is not None:
                layout.activate()

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

    def _action_button(self, icon, text: str, action_id: str | None, slot) -> QToolButton:
        btn = QToolButton()
        btn.setIcon(icon)
        self._tip(btn, text, action_id)
        btn.clicked.connect(slot)
        return btn

    def _tip(self, target, text: str, action_ids) -> None:
        """A tooltip that carries the action's key chip; recorded so a rebind re-renders it."""
        self._shortcut_tips.append((target, text, action_ids))
        target.setToolTip(wrap_tooltip(tooltip_with_shortcut(text, action_ids)))

    def _label(self, action, text: str, action_id: str) -> None:
        """A menu label with the action's key appended; recorded like _tip."""
        self._shortcut_labels.append((action, text, action_id))
        action.setText(label_with_shortcut(text, action_id))

    def _menu_tip(self, action, text: str, action_id: str) -> None:
        """A menu item carries its key in the label, like every other menu, and the explanation in its tooltip."""
        self._label(action, action.text(), action_id)
        action.setToolTip(text)

    def apply_shortcut_tooltips(self) -> None:
        for target, text, ids in self._shortcut_tips:
            target.setToolTip(wrap_tooltip(tooltip_with_shortcut(text, ids)))
        for action, text, action_id in self._shortcut_labels:
            action.setText(label_with_shortcut(text, action_id))

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
            hint="Drag to reorder. What does not fit the canvas width hides, geometry first and navigation last; the ⋯ menu holds every action.",
            defaults=list(DEFAULT_TOOLBAR_IDS),
            repo=self.session.repo,
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
        """Show as many of the chosen controls as fit the canvas width. The overflow menu always
        carries the full action set (see _init_ui), so the menu does not follow the row."""
        self._available_width = w
        budget = self._toolbar_width_budget(w)

        if not self._tool_row.isHidden():
            # The hint goes first, then the slider: the name, undo and Done always stay.
            optional = [widget for widget in (self.tool_hint, self._tool_slider) if widget is not None]
            for widget in optional:
                widget.setVisible(True)
            for widget in optional:
                if self._pill_width() <= budget:
                    break
                widget.setVisible(False)
            self._activate_layout()
            self.adjustSize()
            return

        widgets = [self._row_widgets[item_id] for item_id in self._item_ids]
        for widget in widgets:
            widget.setVisible(True)
        self._sync_separators()

        extras = [item_id for item_id in reversed(self._item_ids) if item_id not in _COLLAPSE_ORDER]
        for item_id in extras + [item_id for item_id in _COLLAPSE_ORDER if item_id in self._item_ids]:
            if self._pill_width() <= budget:
                break
            self._row_widgets[item_id].setVisible(False)
            self._sync_separators()

        self._activate_layout()
        self.adjustSize()
