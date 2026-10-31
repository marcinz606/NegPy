from negpy.kernel.system.i18n import tr
from dataclasses import dataclass

import qtawesome as qta
from PyQt6.QtCore import QEvent, QObject, QPoint, QRect, QRectF, QSize, Qt
from PyQt6.QtGui import QActionGroup, QColor, QCursor, QIcon, QPainter, QPen
from PyQt6.QtWidgets import (
    QAbstractButton,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLayout,
    QMenu,
    QPushButton,
    QSizePolicy,
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
from negpy.desktop.view.widgets.segmented_choice import MenuButton, RightClickToggleAll, SegmentedChoice, ToggleMenuButton
from negpy.desktop.view.widgets.collapsible import hidden_by_gating, roll_revert_icon
from negpy.desktop.view.styles.templates import (
    BAR_BUTTON_HEIGHT,
    BAR_TRACK_BUTTON_HEIGHT,
    BAR_TRACK_PAD,
    BAR_ICON_SIZE,
    EditedDot,
    default_button_height,
    field_label,
    hint_label,
    icon_button,
    icon_text,
    labeled_action,
    section_subheader,
    set_hint_kind,
    tool_toggle,
    tool_toggle_qss,
    wrap_tooltip,
)
from negpy.desktop.view.styles.theme import THEME
from negpy.desktop.view.widgets.split_button import make_split_button
from negpy.desktop.view.widgets.step_combo import StepCombo
from negpy.features.exposure.papers import resolve_paper
from negpy.features.process.models import ProcessMode
from negpy.features.process.path import RenderPath, render_path
from negpy.services.assets import rolls
from negpy.kernel.system.text import count_of

# The index is what is saved, so a new color goes at the end; Preferences shows them darkest first.
CANVAS_COLORS = [
    (THEME.canvas_bg_black, (0.027, 0.027, 0.027), tr("Black")),
    (THEME.canvas_bg_dark_grey, (0.227, 0.227, 0.227), tr("Dark Gray")),
    (THEME.canvas_bg_mid_grey, (0.435, 0.435, 0.435), tr("Mid Gray")),
    (THEME.canvas_bg_white, (0.91, 0.91, 0.91), tr("White")),
    (THEME.canvas_bg_light_grey, (0.667, 0.667, 0.667), tr("Light Gray")),
]


@dataclass(frozen=True)
class ToolbarItem:
    id: str
    category: str
    label: str


# The row is customizable; the overflow menu is not. Category drives both the dialog's grouping
# and the separators, so a reordered row still gets dividers where the kinds change.
TOOLBAR_ITEMS: tuple[ToolbarItem, ...] = (
    ToolbarItem("prev", tr("Navigation"), tr("Previous")),
    ToolbarItem("position", tr("Navigation"), tr("Frame Counter")),
    ToolbarItem("next", tr("Navigation"), tr("Next")),
    ToolbarItem("zoom", tr("Zoom"), tr("Zoom")),
    ToolbarItem("rot_l", tr("Geometry"), tr("Rotate CCW")),
    ToolbarItem("rot_r", tr("Geometry"), tr("Rotate CW")),
    ToolbarItem("flip_h", tr("Geometry"), tr("Flip Horizontal")),
    ToolbarItem("flip_v", tr("Geometry"), tr("Flip Vertical")),
    ToolbarItem("tools", tr("Tools"), tr("Tools")),
    ToolbarItem("undo", tr("Edit"), tr("Undo")),
    ToolbarItem("redo", tr("Edit"), tr("Redo")),
    ToolbarItem("source", tr("View"), tr("Print or Peek")),
    ToolbarItem("average", tr("View"), tr("Roll Average")),
    ToolbarItem("overlays", tr("View"), tr("Overlays")),
    ToolbarItem("reference", tr("View"), tr("Reference View")),
    ToolbarItem("copy", tr("Settings"), tr("Copy Settings")),
    ToolbarItem("copy_bounds", tr("Settings"), tr("Copy Settings + Bounds")),
    ToolbarItem("paste", tr("Settings"), tr("Paste Settings…")),
    ToolbarItem("sync_bounds", tr("Settings"), tr("Sync Bounds…")),
    ToolbarItem("reset", tr("Settings"), tr("Reset Settings")),
    ToolbarItem("reset_to_roll", tr("Settings"), tr("Reset to Roll Settings")),
    ToolbarItem("unload", tr("Settings"), tr("Unload…")),
    ToolbarItem("palette", tr("App"), tr("Find Control or Action…")),
    ToolbarItem("preferences", tr("App"), tr("Preferences…")),
    ToolbarItem("shortcuts", tr("App"), tr("Keyboard Shortcuts…")),
    ToolbarItem("export", tr("Export"), tr("Export")),
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
    "average",
    "export",
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
_ADDED_IDS = {"tools": "undo", "export": "overlays", "average": "export"}
_ADDED_SETTING_KEY = "toolbar_added_items"

# A narrow bar hides these first to last; items not listed (menu actions put on the row) go
# before them, from the end of the user's order.
_COLLAPSE_ORDER = (
    "flip_v",
    "flip_h",
    "rot_r",
    "rot_l",
    "reference",
    "overlays",
    "average",
    "redo",
    "undo",
    "tools",
    "export",
    "source",
    "zoom",
    "position",
    "next",
    "prev",
)

_ZOOM_PRESETS = (25, 50, 100, 200, 400)

# What the bar shows when no canvas row has it: the photo's tools or the Film Strip's.
_MODES = ("photo", "roll")
_MODE_SETTING_KEY = "toolbar_mode"

# (state flag, icon, label, shortcut id, tooltip); index 0 is the edit itself.
_VIEW_SOURCES: tuple[tuple[str, str, str, str, str], ...] = (
    ("", "fa5s.image", tr("Print"), "", tr("The edit, as it prints")),
    (
        "export_size_view",
        "fa5s.ruler-combined",
        tr("Export Size"),
        "toggle_export_size_view",
        tr("The frame as it exports, at the size the Export panel sets, to judge sharpening and grain"),
    ),
    ("compare_mode", "fa5s.adjust", tr("Before / After"), "toggle_compare", tr("Split against the default conversion; drag the divider")),
    (
        "negative_peek",
        "fa5s.film",
        tr("Negative"),
        "toggle_negative_peek",
        tr("The scan as loaded, before inversion and edits."),
    ),
    (
        "average_peek",
        "mdi6.filmstrip",
        tr("Roll Average"),
        "toggle_average_peek",
        tr("This frame printed on the roll's average, to try it before you apply it"),
    ),
    (
        "flat_peek",
        "fa5s.eye",
        tr("Flat Scan"),
        "toggle_flat_peek",
        tr("The flat master: the scan as a log-encoded digital intermediate, with no print curve"),
    ),
    (
        "embedded_peek",
        "fa5s.camera",
        tr("Embedded"),
        "toggle_embedded_peek",
        tr("The camera's own JPEG of this shot, for comparison."),
    ),
    (
        "flatfield_peek",
        "fa5s.lightbulb",
        tr("Flat Field"),
        "toggle_flatfield_peek",
        tr("Flat Field check: how evenly the Flat Field profile from Optics corrects the light"),
    ),
)

_SOURCE_INDEX = {flag: i for i, (flag, *_rest) in enumerate(_VIEW_SOURCES)}

_BRUSH_SIZE = "controls_panel.retouch_sidebar.manual_size_slider"
_FINE_ROTATION = "controls_panel.geometry_sidebar.fine_rot_slider"

# The row an armed canvas tool shows in place of the view row: (name, hint, slider path from MainWindow).
_TOOL_ROWS: dict[ToolMode, tuple[str, str, str]] = {
    ToolMode.WB_PICK: (tr("WB Picker"), tr("Click a neutral gray; the selected region's CMY is solved so it prints neutral"), ""),
    ToolMode.CROP_MANUAL: (tr("Crop"), tr("Drag a corner or an edge; drag outside the box to redraw it"), _FINE_ROTATION),
    ToolMode.DUST_PICK: (tr("Heal"), tr("Paint over dust; only the marks inside the brush are repaired"), _BRUSH_SIZE),
    ToolMode.SCRATCH_PICK: (tr("Scratch"), tr("Click points along the scratch; Backspace deletes the last point"), _BRUSH_SIZE),
    ToolMode.SCRATCH_LINE: (
        tr("Transport Line"),
        tr("Click once on the line; the whole line is traced and repaired"),
        "controls_panel.retouch_sidebar.line_threshold_slider",
    ),
    ToolMode.CLONE: (tr("Clone"), tr("Paint over the defect; the dashed circle marks the source, Alt-click picks a new one"), _BRUSH_SIZE),
    ToolMode.LEAK_BAND: (tr("Leak Band"), tr("Drag along the leak's strongest line, then drag a dashed edge to where it fades out"), ""),
    ToolMode.LEAK_EDGE: (tr("Leak Edge"), tr("Drag along the side the leak comes in from; the line snaps to it"), ""),
    ToolMode.LEAK_SPOT: (tr("Leak Spot"), tr("Drag out an oval over a round leak"), ""),
    ToolMode.LEAK_PAINT: (tr("Leak Paint"), tr("Brush over a leak with no line or center, wider than the leak"), _BRUSH_SIZE),
    ToolMode.LOCAL_DRAW: (tr("Draw Mask"), tr("Click to place vertices; a click near the start closes the mask"), ""),
    ToolMode.LOCAL_OVAL: (tr("Oval Mask"), tr("Drag out an oval"), ""),
    ToolMode.LOCAL_GRADIENT: (tr("Card Edge"), tr("Drag from the full-exposure edge to where it fades out"), ""),
    ToolMode.ANALYSIS_DRAW: (tr("Draw Region"), tr("Drag the area the meters read"), ""),
    ToolMode.FILM_BASE: (tr("Film Base"), tr("Click the clear rebate to set the roll's film base"), ""),
    ToolMode.STRAIGHTEN: (tr("Level"), tr("Draw a line along a horizon or an edge to level the frame"), _FINE_ROTATION),
    ToolMode.KEYSTONE_LINES: (
        tr("Keystone Lines"),
        tr("Drag a line along each converging edge; four lines solve Tilt and Swing"),
        "controls_panel.geometry_sidebar.converge_v_slider",
    ),
    ToolMode.ZONE_PLACE: (tr("Zone Placement"), tr("Click the photo to pin a spot; drag a pin to move it"), ""),
}
_CLONE_SOURCE_HINT = tr("Click the photo to pick the area to copy from")

# The Tools menu, in panel order: a panel section's title, then its tools as (tool, action id, sidebar button
# path from MainWindow). Zone Placement has no entry: it arms only with a zone picked on the zone strip.
_CANVAS_TOOLS: tuple[tuple[ToolMode, str, str] | str, ...] = (
    tr("Crop"),
    (ToolMode.CROP_MANUAL, "manual_crop", "controls_panel.autocrop_sidebar.manual_crop_btn"),
    tr("Straighten"),
    (ToolMode.STRAIGHTEN, "straighten", "controls_panel.geometry_sidebar.straighten_btn"),
    (ToolMode.KEYSTONE_LINES, "keystone_lines", "controls_panel.geometry_sidebar.keystone_lines_btn"),
    tr("Metering"),
    (ToolMode.ANALYSIS_DRAW, "analysis_draw", "controls_panel.process_sidebar.analysis_region_btn"),
    (ToolMode.FILM_BASE, "film_base", "controls_panel.process_sidebar.film_base_btn"),
    tr("Filtration"),
    (ToolMode.WB_PICK, "pick_wb", "controls_panel.color_sidebar.pick_wb_btn"),
    tr("Dodge & Burn"),
    (ToolMode.LOCAL_DRAW, "local_draw", "controls_panel.local_sidebar.draw_btn"),
    (ToolMode.LOCAL_OVAL, "local_oval", "controls_panel.local_sidebar.oval_btn"),
    (ToolMode.LOCAL_GRADIENT, "local_gradient", "controls_panel.local_sidebar.gradient_btn"),
    tr("Light Leak"),
    (ToolMode.LEAK_BAND, "tool_leak_band", "controls_panel.leak_sidebar.band_btn"),
    (ToolMode.LEAK_EDGE, "tool_leak_edge", "controls_panel.leak_sidebar.edge_btn"),
    (ToolMode.LEAK_SPOT, "tool_leak_spot", "controls_panel.leak_sidebar.spot_btn"),
    (ToolMode.LEAK_PAINT, "tool_leak_paint", "controls_panel.leak_sidebar.paint_btn"),
    tr("Retouch"),
    (ToolMode.DUST_PICK, "pick_dust", "controls_panel.retouch_sidebar.pick_dust_btn"),
    (ToolMode.SCRATCH_PICK, "pick_scratch", "controls_panel.retouch_sidebar.pick_scratch_btn"),
    (ToolMode.SCRATCH_LINE, "pick_scratch_line", "controls_panel.retouch_sidebar.pick_line_btn"),
    (ToolMode.CLONE, "clone_tool", "controls_panel.retouch_sidebar.clone_btn"),
)
_TOOL_BUTTON_PATHS = {spec[0]: spec[2] for spec in _CANVAS_TOOLS if not isinstance(spec, str)}


def _icon(name: str, checkable: bool = False) -> QIcon:
    """A bar or menu icon, gray while disabled; a checkable one is dim while off. In a menu the
    icon takes the checkmark's place, and the checked item's chip marks it on."""
    if checkable:
        return qta.icon(name, color=THEME.text_hint, color_on=THEME.text_primary, color_disabled=THEME.text_muted)
    return qta.icon(name, color=THEME.text_primary, color_disabled=THEME.text_muted)


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
    """The bar under the Film Strip: the photo's tools or, in Roll mode, the strip's own row."""

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
        self.setObjectName("action_bar")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        # The window sets the width, never the buttons: set_available_width hides what does not fit.
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        main_layout = QHBoxLayout(self)
        main_layout.setSizeConstraint(QLayout.SizeConstraint.SetNoConstraint)
        main_layout.setContentsMargins(THEME.space_lg, THEME.space_lg, THEME.space_lg, THEME.space_lg)
        main_layout.setSpacing(THEME.space_md)

        self._rows = QWidget()
        v_layout = QVBoxLayout(self._rows)
        v_layout.setContentsMargins(0, 0, 0, 0)
        v_layout.setSpacing(0)
        self._rows_layout = v_layout
        self._roll_row: QWidget | None = None

        # Controls need a parent before _rebuild_row() applies visibility.
        self._view_row = QWidget()
        row_layout = QHBoxLayout(self._view_row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(THEME.space_md)
        v_layout.addWidget(self._view_row, 0, Qt.AlignmentFlag.AlignHCenter)

        icon_size = QSize(BAR_ICON_SIZE, BAR_ICON_SIZE)
        btn_height = BAR_TRACK_BUTTON_HEIGHT
        self._shortcut_tips: list = []
        self._shortcut_labels: list = []

        saved_mode = self.session.repo.get_global_setting(_MODE_SETTING_KEY, _MODES[0])
        self._mode = saved_mode if saved_mode in _MODES else _MODES[0]
        mode_tip = tr("What the toolbar shows: the photo's tools, or the roll's filter, sort and actions")
        self.mode_switch = SegmentedChoice((("fa5s.image", tr("Photo")), ("fa5s.film", tr("Roll"))), mode_tip, data=_MODES)
        self._tip(self.mode_switch, mode_tip, "toggle_toolbar_mode")
        self.mode_switch.setCurrentIndex(_MODES.index(self._mode))
        self.mode_switch.setFixedHeight(BAR_BUTTON_HEIGHT)
        self.mode_switch.currentChanged.connect(lambda _i: self.set_mode(self.mode_switch.currentData()))

        # Not checkable: an open panel is the resting state, and the checked look means armed.
        self.btn_toggle_strip = QToolButton()
        self._tip(self.btn_toggle_strip, tr("Collapse or expand the Film Strip"), "toggle_film_strip")
        self.set_strip_expanded(True)
        self.btn_toggle_right = QToolButton()
        self.btn_toggle_right.setIcon(_icon("fa5s.sliders-h"))
        self._tip(self.btn_toggle_right, tr("Toggle Controls Panel"), "toggle_right_panel")

        self.btn_prev = QToolButton()
        self.btn_prev.setIcon(_icon("fa5s.chevron-left"))
        self._tip(self.btn_prev, tr("Previous"), "prev_file")
        self.btn_next = QToolButton()
        self.btn_next.setIcon(_icon("fa5s.chevron-right"))
        self._tip(self.btn_next, tr("Next"), "next_file")

        # Undo and Redo live in the main toolbar (mdi arrows, distinct from the rotate icons'
        # file-with-arrow glyphs below).
        self.btn_undo = QToolButton()
        self.btn_undo.setIcon(_icon("mdi.undo"))
        self._tip(self.btn_undo, tr("Undo"), "undo")
        self.btn_redo = QToolButton()
        self.btn_redo.setIcon(_icon("mdi.redo"))
        self._tip(self.btn_redo, tr("Redo"), "redo")

        self.btn_rot_l = QToolButton()
        self.btn_rot_l.setIcon(_icon("mdi6.file-rotate-left"))
        self._tip(self.btn_rot_l, tr("Rotate CCW"), "rotate_ccw")
        self.btn_rot_r = QToolButton()
        self.btn_rot_r.setIcon(_icon("mdi6.file-rotate-right"))
        self._tip(self.btn_rot_r, tr("Rotate CW"), "rotate_cw")
        self.btn_flip_h = QToolButton()
        self.btn_flip_h.setCheckable(True)
        self.btn_flip_h.setIcon(_icon("fa5s.arrows-alt-h", checkable=True))
        self._tip(self.btn_flip_h, tr("Flip Horizontal"), "flip_h")
        self.btn_flip_v = QToolButton()
        self.btn_flip_v.setCheckable(True)
        self.btn_flip_v.setIcon(_icon("fa5s.arrows-alt-v", checkable=True))
        self._tip(self.btn_flip_v, tr("Flip Vertical"), "flip_v")

        self.position_label = QLabel()
        self.position_label.setFixedHeight(btn_height)
        self.position_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.position_label.setStyleSheet(
            f"color: {THEME.text_secondary}; font-size: {THEME.font_size_header}px; padding: 0 {THEME.space_xs}px;"
        )
        self.position_label.setToolTip(tr("This frame's place in the Film Strip's order and filter"))

        # The zoom readout opens the presets, since users zoom directly on the canvas.
        self.btn_zoom = MenuButton(tr("Zoom: at 100%, one scan pixel fills one screen pixel"))
        zoom_menu = self.btn_zoom.choice_menu
        self._zoom_fit_action = zoom_menu.addAction(tr("Fit to Window"))
        self._zoom_fit_action.triggered.connect(self._on_fit_clicked)
        self._label(self._zoom_fit_action, tr("Fit to Window"), "fit_view")
        zoom_menu.addSeparator()
        for percent in _ZOOM_PRESETS:
            action = zoom_menu.addAction(tr("{value}%", value=percent))
            action.triggered.connect(lambda _checked=False, p=percent: self._on_zoom_preset(p))
            if percent == 100:
                self._label(action, "100%", "zoom_100")
                action.setToolTip(tr("Below HQ the preview is enlarged: the framing is right, but fine detail is not."))
        zoom_menu.addSeparator()
        self._zoom_hq_action = zoom_menu.addAction(tr("HQ Preview"))
        self._zoom_hq_action.setCheckable(True)
        self._zoom_hq_action.setToolTip(tr("Render the full-resolution scan instead of a downscaled preview"))
        self._zoom_hq_action.triggered.connect(self.controller.toggle_hq_preview)
        self._label(self._zoom_hq_action, tr("HQ Preview"), "toggle_hq")
        # Sized for the common readouts, so zooming up to 1:1 does not move the row.
        self.btn_zoom.setText(tr("100%"))
        self._zoom_min_width = self.btn_zoom.sizeHint().width()
        self.btn_zoom.setMinimumWidth(self._zoom_min_width)

        # Every source here takes the whole canvas, and the controller keeps them exclusive.
        self.source_choice = MenuButton(tr("What the canvas shows: the print, a before/after split, or a peek."))
        source_group = QActionGroup(self)
        self._source_actions = []
        for i, (_flag, icon, label, key, tip) in enumerate(_VIEW_SOURCES):
            action = self.source_choice.choice_menu.addAction(_icon(icon), label)
            action.setCheckable(True)
            source_group.addAction(action)
            if key:
                self._menu_tip(action, tip, key)
            else:
                action.setToolTip(tip)
            action.triggered.connect(lambda _checked=False, i=i: self._on_source_chosen(i))
            self._source_actions.append(action)
        # Sized for the longest view, so changing it does not move the row.
        source_width = 0
        for _flag, _icon_name, label, _key, _tip in _VIEW_SOURCES:
            self.source_choice.setText(icon_text(label))
            source_width = max(source_width, self.source_choice.sizeHint().width())
        self.source_choice.setMinimumWidth(source_width)
        self._show_source(0)

        # Mirrors the Export panel's button once MainWindow calls mirror_export.
        self._export_menu = QMenu(self)
        self._export_menu.setToolTipsVisible(True)
        self.export_split, self.btn_export, self.btn_export_menu = make_split_button(
            tr("Export"), "mdi6.export-variant", self._export_menu, primary=True
        )
        self.export_split.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        self.btn_export.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)

        self.overlays_menu = ToggleMenuButton("fa5s.layer-group", "", tr("Overlays: marks drawn over the frame that never change the edit"))
        self._overlay_actions = {}
        for flag, icon, label, key, tip, toggle in (
            (
                "zones_overlay",
                "mdi6.numeric-5-box-outline",
                tr("Zone Overlay"),
                "toggle_zones",
                tr("Show which Adams zone each part of the print falls in"),
                self.controller.toggle_zones_overlay,
            ),
            (
                "clipping_overlay",
                "fa5s.adjust",
                tr("Clipping Overlay"),
                "toggle_clipping_overlay",
                tr("Mark paper black and paper white on the frame"),
                self.controller.toggle_clipping_overlay,
            ),
            (
                "printing_notes",
                "mdi.playlist-edit",
                tr("Printing Notes"),
                "toggle_printing_notes",
                tr("The dodge and burn map and the print recipe"),
                self.controller.toggle_printing_notes,
            ),
            (
                "grain_focuser",
                "fa5s.search-plus",
                tr("Grain Focuser"),
                "toggle_grain_focuser",
                tr("Grain focuser: shows the exported pixels at the pointer at their true size, with a sharpness number to compare areas."),
                self.controller.toggle_grain_focuser,
            ),
            (
                "cursor_probe",
                "fa5s.crosshairs",
                tr("Probe at Cursor"),
                "toggle_cursor_probe",
                tr("Zone, print density and ΔD of the pixel under the pointer"),
                self.controller.toggle_cursor_probe,
            ),
        ):
            action = self.overlays_menu.add_toggle(label, tip)
            action.setIcon(_icon(icon))
            self._label(action, label, key)
            action.triggered.connect(lambda checked, toggle=toggle: toggle(force=checked))
            self._overlay_actions[flag] = action
        # A submenu, not a toggle: the pills are on by default and the button lights only for marks.
        self.overlays_menu.choice_menu.addSeparator()
        info_menu = self.overlays_menu.choice_menu.addMenu(tr("Frame Info"))
        info_menu.setToolTipsVisible(True)
        info_group = QActionGroup(self)
        self._info_actions = []
        for level, label in enumerate(INFO_LEVEL_LABELS):
            action = info_menu.addAction(label)
            action.setCheckable(True)
            info_group.addAction(action)
            action.triggered.connect(lambda _checked=False, level=level: self.controller.set_canvas_info_level(level))
            self._info_actions.append(action)
        self._info_actions[0].setToolTip(tr("No corner pills"))
        self._info_actions[1].setToolTip(tr("File name, scan size and film mode"))
        self._info_actions[2].setToolTip(tr("Adds the frame's place in the strip, film, camera and lens, the proof and the export size"))

        # Opens the Roll Average row; checked while the frame uses an average.
        self._average_icon = _icon("mdi6.filmstrip", checkable=True)
        self.btn_average = self._action_button(
            self._average_icon,
            tr("Roll Average: try the average on this frame; right-click turns every average on or off."),
            "toggle_average_peek",
            self._on_average_clicked,
        )
        self.btn_average.setCheckable(True)
        self._average_right_click = RightClickToggleAll([self.btn_average], self._frame_average_states)
        self._average_right_click.toggledAll.connect(self._on_use_all_averages)

        self.btn_tools = MenuButton(tr("Tools: crop, straighten, the pickers, masks and retouching, each with its own row here"))
        self.btn_tools.setIcon(_icon("fa5s.tools"))
        self._show_tools_name(True)
        self._fill_tools_menu(self.btn_tools.choice_menu)

        # One-shot actions that otherwise live only in the ⋯ menu.
        self.btn_reference = self._action_button(
            _icon("fa5s.thumbtack"),
            tr("Reference view: pin this frame beside the canvas, to match others to it"),
            "toggle_reference",
            self._toggle_reference,
        )
        self.btn_light_table = self._action_button(
            _icon("fa5s.th", checkable=True),
            tr("Light Table: the whole roll in rows, in place of the canvas"),
            "toggle_light_table",
            self._toggle_light_table,
        )
        self.btn_light_table.setCheckable(True)
        self.btn_copy = self._action_button(
            _icon("fa5s.copy"), tr("Copy this frame's settings to the clipboard"), "copy", self.session.copy_settings
        )
        self.btn_copy_bounds = self._action_button(
            _icon("fa5s.clone"),
            tr("Copy settings plus the metering/normalization bounds"),
            "copy_with_bounds",
            self.session.copy_settings_with_bounds,
        )
        self.btn_paste = self._action_button(
            _icon("fa5s.paste"),
            tr("Paste the copied settings onto this frame, or every selected frame"),
            "paste",
            lambda: open_paste_dialog(self, self.controller),
        )
        self.btn_sync_bounds = self._action_button(
            _icon("fa5s.crosshairs"),
            tr("Sync Bounds: give other frames this frame's measured black and white points, and nothing else"),
            "sync_bounds",
            lambda: open_sync_bounds_dialog(self, self.session),
        )
        self.btn_reset = self._action_button(
            _icon("fa5s.history"),
            tr("Reset Settings: discard all edits and return this frame to its default look"),
            None,
            self.session.reset_settings,
        )
        self.btn_reset_to_roll = self._action_button(
            roll_revert_icon(THEME.text_primary),
            tr("Reset to Roll Settings: put the roll's settings back on every card that differs."),
            "reset_to_roll",
            self.controller.revert_frame_to_roll,
        )
        self.btn_unload = self._action_button(
            _icon("fa5s.times-circle"),
            tr("Unload: remove this frame from the session and keep its saved edit"),
            None,
            self._on_overflow_unload,
        )
        self.btn_palette = self._action_button(
            _icon("fa5s.search"),
            tr("Find any slider, card or action by name, and open it"),
            "command_palette",
            self._show_palette,
        )
        self.btn_preferences = self._action_button(
            _icon("fa5s.cog"),
            tr("Preferences: interface, performance and storage settings for the whole app"),
            "open_preferences",
            self._show_preferences,
        )
        self.btn_shortcuts = self._action_button(
            _icon("fa5s.keyboard"),
            tr("Show the full keyboard shortcuts reference"),
            "show_shortcuts",
            self._show_shortcuts,
        )

        self.btn_overflow = QToolButton()
        self.btn_overflow.setObjectName("toolbar_overflow")
        self.btn_overflow.setIcon(_icon("fa5s.ellipsis-h"))
        self.btn_overflow.setToolTip(tr("More actions"))
        self.btn_overflow.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)

        overflow_menu = QMenu(self.btn_overflow)
        overflow_menu.setToolTipsVisible(True)  # so the GPU action can surface its backend/status

        # The menu holds every action, whatever the row shows at the current width.
        find_action = overflow_menu.addAction(_icon("fa5s.search"), tr("Find Control or Action…"), self._show_palette)
        self._label(find_action, tr("Find Control or Action…"), "command_palette")
        find_action.setToolTip(tr("Find any slider, card or action by name, and open it"))
        self._fill_tools_menu(overflow_menu.addMenu(_icon("fa5s.tools"), tr("Tools")))

        # What the canvas shows, in the groups the row's own Zoom, Print and Overlays menus use.
        view_menu = overflow_menu.addMenu(_icon("fa5s.eye"), tr("View"))
        view_menu.setToolTipsVisible(True)
        self._menu_section(view_menu, tr("Zoom"))
        self._ov_hq_action = view_menu.addAction(tr("HQ Preview"))
        self._ov_hq_action.setCheckable(True)
        self._menu_tip(self._ov_hq_action, tr("Toggle high-quality (full-resolution) preview"), "toggle_hq")
        self._ov_fit_action = view_menu.addAction(_icon("fa5s.expand"), tr("Fit to Window"))
        self._menu_tip(self._ov_fit_action, tr("Fit to Window"), "fit_view")
        self._ov_original_action = view_menu.addAction(tr("Original Size (1:1)"))
        self._ov_original_action.setCheckable(True)
        self._menu_tip(
            self._ov_original_action,
            tr("Original size (100%): one scan pixel per screen pixel, with exact fine detail only in HQ"),
            "zoom_100",
        )
        self._menu_section(view_menu, tr("Compare"))
        self._ov_compare_action = view_menu.addAction(tr("Before / After"))
        self._ov_compare_action.setCheckable(True)
        self._menu_tip(
            self._ov_compare_action, tr("Before / After: split against the default conversion; drag the divider"), "toggle_compare"
        )
        reference_action = view_menu.addAction(tr("Reference View"), self._toggle_reference)
        self._menu_tip(reference_action, tr("Reference view: pin this frame beside the canvas, to match others to it"), "toggle_reference")
        light_table_action = view_menu.addAction(tr("Light Table"), self._toggle_light_table)
        self._menu_tip(light_table_action, tr("Light Table: the whole roll in rows, in place of the canvas"), "toggle_light_table")
        self._menu_section(view_menu, tr("Peek"))
        self._ov_flat_peek_action = view_menu.addAction(tr("Peek Flat Scan"))
        self._ov_flat_peek_action.setCheckable(True)
        self._menu_tip(
            self._ov_flat_peek_action,
            tr("Peek flat scan: show the flat master for a moment, with no change to your edit"),
            "toggle_flat_peek",
        )
        self._ov_negative_peek_action = view_menu.addAction(tr("Peek Negative"))
        self._ov_negative_peek_action.setCheckable(True)
        self._menu_tip(
            self._ov_negative_peek_action,
            tr("Peek Negative: the scan as loaded, before inversion and edits."),
            "toggle_negative_peek",
        )
        self._ov_embedded_peek_action = view_menu.addAction(tr("Peek Embedded Preview"))
        self._ov_embedded_peek_action.setCheckable(True)
        self._menu_tip(
            self._ov_embedded_peek_action,
            tr("Peek Embedded Preview: the camera's own JPEG of this shot, for comparison."),
            "toggle_embedded_peek",
        )
        self._menu_section(view_menu, tr("Overlays"))
        overlay_icon = {flag: action.icon() for flag, action in self._overlay_actions.items()}
        self._ov_zones_action = view_menu.addAction(overlay_icon["zones_overlay"], tr("Zone Overlay"))
        self._ov_zones_action.setCheckable(True)
        self._menu_tip(self._ov_zones_action, tr("Show which Adams zone each part of the print falls in"), "toggle_zones")
        self._ov_clipping_action = view_menu.addAction(overlay_icon["clipping_overlay"], tr("Clipping Overlay"))
        self._ov_clipping_action.setCheckable(True)
        self._menu_tip(self._ov_clipping_action, tr("Mark paper black and paper white on the frame"), "toggle_clipping_overlay")
        self._ov_probe_action = view_menu.addAction(overlay_icon["cursor_probe"], tr("Probe at Cursor"))
        self._ov_probe_action.setCheckable(True)
        self._menu_tip(self._ov_probe_action, tr("Zone, print density and ΔD of the pixel under the pointer"), "toggle_cursor_probe")
        self._ov_loupe_action = view_menu.addAction(overlay_icon["grain_focuser"], tr("Grain Focuser"))
        self._ov_loupe_action.setCheckable(True)
        self._menu_tip(
            self._ov_loupe_action,
            tr("Grain focuser: shows the exported pixels at the pointer at their true size, with a sharpness number to compare areas."),
            "toggle_grain_focuser",
        )
        self._ov_notes_action = view_menu.addAction(overlay_icon["printing_notes"], tr("Printing Notes"))
        self._ov_notes_action.setCheckable(True)
        self._menu_tip(self._ov_notes_action, tr("The dodge and burn map and the print recipe"), "toggle_printing_notes")
        overflow_menu.addSeparator()

        self._ov_undo_action = overflow_menu.addAction(_icon("mdi.undo"), tr("Undo"))
        self._menu_tip(self._ov_undo_action, tr("Undo"), "undo")
        self._ov_redo_action = overflow_menu.addAction(_icon("mdi.redo"), tr("Redo"))
        self._menu_tip(self._ov_redo_action, tr("Redo"), "redo")
        overflow_menu.addSeparator()

        self._ov_rot_l_action = overflow_menu.addAction(_icon("mdi6.file-rotate-left"), tr("Rotate CCW"))
        self._menu_tip(self._ov_rot_l_action, tr("Rotate CCW"), "rotate_ccw")
        self._ov_rot_r_action = overflow_menu.addAction(_icon("mdi6.file-rotate-right"), tr("Rotate CW"))
        self._menu_tip(self._ov_rot_r_action, tr("Rotate CW"), "rotate_cw")
        # Actions, like Rotate, so they carry the row buttons' icons; the row's buttons show the flip state.
        self._ov_flip_h_action = overflow_menu.addAction(_icon("fa5s.arrows-alt-h"), tr("Flip Horizontal"))
        self._menu_tip(self._ov_flip_h_action, tr("Flip Horizontal"), "flip_h")
        self._ov_flip_v_action = overflow_menu.addAction(_icon("fa5s.arrows-alt-v"), tr("Flip Vertical"))
        self._menu_tip(self._ov_flip_v_action, tr("Flip Vertical"), "flip_v")
        overflow_menu.addSeparator()

        self._action_copy = overflow_menu.addAction(_icon("fa5s.copy"), tr("Copy Settings"), self.session.copy_settings)
        self._label(self._action_copy, tr("Copy Settings"), "copy")
        self._action_copy.setToolTip(tr("Copy this frame's settings to the clipboard"))
        self._action_copy_bounds = overflow_menu.addAction(
            _icon("fa5s.copy"),
            tr("Copy Settings + Bounds"),
            self.session.copy_settings_with_bounds,
        )
        self._label(self._action_copy_bounds, tr("Copy Settings + Bounds"), "copy_with_bounds")
        self._action_copy_bounds.setToolTip(tr("Copy settings plus the metering/normalization bounds"))
        self._action_paste = overflow_menu.addAction(
            _icon("fa5s.paste"),
            tr("Paste Settings…"),
            lambda: open_paste_dialog(self, self.controller),
        )
        self._label(self._action_paste, tr("Paste Settings…"), "paste")
        self._action_paste.setToolTip(tr("Paste the copied settings onto this frame, or every selected frame"))
        self._action_sync_bounds = overflow_menu.addAction(
            _icon("fa5s.crosshairs"),
            tr("Sync Bounds…"),
            lambda: open_sync_bounds_dialog(self, self.session),
        )
        self._label(self._action_sync_bounds, tr("Sync Bounds…"), "sync_bounds")
        self._action_sync_bounds.setToolTip(tr("Give other frames this frame's metering bounds, and nothing else"))
        overflow_menu.addSeparator()
        reset_settings_action = overflow_menu.addAction(_icon("fa5s.history"), tr("Reset Settings"), self.session.reset_settings)
        reset_settings_action.setToolTip(tr("Discard all edits and return this frame to its default look"))
        self._action_reset_to_roll = overflow_menu.addAction(
            roll_revert_icon(THEME.text_primary), tr("Reset to Roll Settings"), self.controller.revert_frame_to_roll
        )
        self._label(self._action_reset_to_roll, tr("Reset to Roll Settings"), "reset_to_roll")
        self._action_reset_to_roll.setToolTip(tr("Return every card that differs from the roll to the roll's settings"))
        overflow_menu.aboutToShow.connect(lambda: self._action_reset_to_roll.setEnabled(self.controller.can_revert_frame_to_roll()))
        unload_action = overflow_menu.addAction(_icon("fa5s.times-circle"), tr("Unload…"), self._on_overflow_unload)
        unload_action.setToolTip(tr("Remove this frame from the session and keep its saved edit"))
        overflow_menu.addSeparator()

        prefs_action = overflow_menu.addAction(_icon("fa5s.sliders-h"), tr("Preferences…"), self._show_preferences)
        self._label(prefs_action, tr("Preferences…"), "open_preferences")
        prefs_action.setToolTip(tr("Interface, performance and storage settings for the whole app"))
        self._help_menu = overflow_menu.addMenu(_icon("fa5s.question-circle"), tr("Help"))
        self._help_menu.setToolTipsVisible(True)
        tour_action = self._help_menu.addAction(_icon("fa5s.map-signs"), tr("Take the Tour"), self._show_tour)
        self._label(tour_action, tr("Take the Tour"), "show_tour")
        tour_action.setToolTip(tr("Replay the guided feature tour"))
        shortcuts_action = self._help_menu.addAction(_icon("fa5s.keyboard"), tr("Keyboard Shortcuts…"), self._show_shortcuts)
        self._label(shortcuts_action, tr("Keyboard Shortcuts…"), "show_shortcuts")
        shortcuts_action.setToolTip(tr("Show the full keyboard shortcuts reference"))
        self._help_menu.addSeparator()
        self._update_action = self._help_menu.addAction(_icon("fa5s.sync-alt"), tr("Check for Updates…"), self._check_for_updates)
        about_action = self._help_menu.addAction(_icon("fa5s.info-circle"), tr("About NegPy…"), self._show_about)
        about_action.setToolTip(tr("Version and project page"))
        self.btn_overflow.setMenu(overflow_menu)

        standard_buttons = [
            self.btn_toggle_strip,
            self.btn_toggle_right,
            self.btn_prev,
            self.btn_next,
            self.btn_flip_h,
            self.btn_flip_v,
            self.btn_undo,
            self.btn_redo,
            self.btn_reference,
            self.btn_light_table,
            self.btn_average,
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

        # The file-rotate glyphs (page plus arrow) read as a blob at the standard icon size.
        # A touch larger keeps the page and arrow legible without changing the button's own
        # footprint, since btn_height still applies.
        rotate_icon_size = QSize(BAR_ICON_SIZE + 4, BAR_ICON_SIZE + 4)
        for btn in (self.btn_rot_l, self.btn_rot_r):
            btn.setIconSize(rotate_icon_size)
            btn.setFixedHeight(btn_height)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setToolTip(wrap_tooltip(btn.toolTip()))
        for btn in (
            self.btn_zoom,
            self.btn_tools,
            self.source_choice,
            self.overlays_menu,
            self.btn_export,
            self.btn_export_menu,
        ):
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setFixedHeight(btn_height)
            btn.setIconSize(icon_size)

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
            "average": self.btn_average,
            "overlays": self.overlays_menu,
            "reference": self.btn_reference,
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
            "export": self.export_split,
        }
        self._row_sequence: list[tuple[QWidget, bool]] = []
        self._available_width = 0
        self._zoom_percent = 100
        self._item_ids = load_toolbar_items(self.session.repo)
        self._rebuild_row()

        self._build_tool_row(v_layout, icon_size, btn_height)
        self._build_paper_row(v_layout, btn_height)
        self._build_zone_row(v_layout)
        self._build_average_row(v_layout)
        self._build_light_table_row(v_layout, icon_size, rotate_icon_size, btn_height)

        main_layout.addWidget(self.mode_switch)
        main_layout.addWidget(self._create_separator())
        main_layout.addWidget(self._rows, 1)
        main_layout.addWidget(self._create_separator())
        for btn in (self.btn_light_table, self.btn_overflow, self.btn_toggle_right, self.btn_toggle_strip):
            main_layout.addWidget(btn)
        self._adopt_row(self)
        self._sync_rows()

    def _build_tool_row(self, v_layout: QVBoxLayout, icon_size: QSize, btn_height: int) -> None:
        """While a canvas tool is armed the bar shows its own row: what the tool is, how to use it,
        the slider it is sized or tuned by, undo and Done."""
        self._tool_row = QWidget()
        self._tool_row.hide()
        layout = QHBoxLayout(self._tool_row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(THEME.space_md)
        self._tool_layout = layout
        self._shown_tool = ToolMode.NONE
        self._shown_rows = (False, False, False, False)  # (tool, paper, zone, average); none means the view row
        self._tool_slider = None
        self._tool_slider_source = None

        # Names the armed tool and switches to another one, with no Done in between.
        self.tool_name = MenuButton(tr("Switch to another canvas tool"))
        self.tool_name.setFixedHeight(btn_height)
        self._fill_tools_menu(self.tool_name.choice_menu)
        self.tool_hint = hint_label()
        self.tool_hint.setWordWrap(False)
        self.btn_tool_undo = QToolButton()
        self.btn_tool_undo.setIcon(_icon("mdi.undo"))
        self._tip(self.btn_tool_undo, tr("Undo"), "undo")
        self.btn_tool_undo.clicked.connect(lambda: _context_undo(self.controller))
        self.btn_tool_redo = QToolButton()
        self.btn_tool_redo.setIcon(_icon("mdi.redo"))
        self._tip(self.btn_tool_redo, tr("Redo"), "redo")
        self.btn_tool_redo.clicked.connect(self.session.redo)
        for btn in (self.btn_tool_undo, self.btn_tool_redo):
            btn.setIconSize(icon_size)
            btn.setFixedHeight(btn_height)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_tool_done = labeled_action("fa5s.check", tr("Done"), "", primary=True)
        self.btn_tool_done.setToolTip(wrap_tooltip(tr("Finish the line or shape in progress and leave the tool, keeping its changes")))
        self.btn_tool_done.setFixedHeight(BAR_BUTTON_HEIGHT)
        # A narrow bar lays out at minimum widths, which would cut the label down to the icon.
        self.btn_tool_done.setMinimumWidth(self.btn_tool_done.sizeHint().width())
        self.btn_tool_done.clicked.connect(self._on_tool_done)

        layout.addWidget(self.tool_name)
        layout.addWidget(self.tool_hint)
        self._tool_slider_index = layout.count()
        layout.addWidget(self._create_separator())
        layout.addWidget(self.btn_tool_undo)
        layout.addWidget(self.btn_tool_redo)
        layout.addWidget(self.btn_tool_done)
        v_layout.addWidget(self._tool_row, 0, Qt.AlignmentFlag.AlignHCenter)

    def _build_paper_row(self, v_layout: QVBoxLayout, btn_height: int) -> None:
        """While the paper test strip is up the bar names it; once a band is shown full frame it
        steps between papers and keeps one."""
        self._paper_row = QWidget()
        self._paper_row.hide()
        layout = QHBoxLayout(self._paper_row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(THEME.space_md)
        self.paper_layout = SegmentedChoice(
            (("mdi.view-column-outline", tr("Strip")), ("mdi.view-grid-outline", tr("Grid"))),
            tr("Strip shows a band of the frame per paper; Grid shows every paper on the whole frame, small."),
            data=("bands", "grid"),
        )
        self.paper_layout.currentChanged.connect(lambda _i: self.controller.set_paper_strip_layout(self.paper_layout.currentData()))
        self.paper_strip_hint = hint_label(tr("Click a paper to see it on the whole frame"))
        self.paper_strip_hint.setWordWrap(False)
        self.paper_strip_combo = StepCombo()
        self._tip(
            self.paper_strip_combo,
            tr("The paper shown full frame: the arrows step to the neighbor paper, the name opens the list"),
            ("prev_file", "next_file"),
        )
        self.paper_strip_combo.currentIndexChanged.connect(self.controller.show_paper_strip)
        self._paper_separator = self._create_separator()
        self.btn_paper_back = labeled_action("fa5s.arrow-left", tr("Strip"), "")
        self._tip(self.btn_paper_back, tr("Back to every paper"), "cancel_tool")
        self.btn_paper_back.clicked.connect(lambda: self.controller.show_paper_strip(None))
        self.btn_paper_keep = labeled_action("fa5s.check", tr("Keep"), tr("Print on this paper: make it the Print stock"), primary=True)
        self.btn_paper_keep.clicked.connect(lambda: self.controller.keep_paper_strip())
        self.btn_paper_back.setFixedHeight(btn_height)
        self.btn_paper_keep.setFixedHeight(BAR_BUTTON_HEIGHT)
        for btn in (self.btn_paper_back, self.btn_paper_keep):
            # A narrow bar lays out at minimum widths, which would cut the label down to the icon.
            btn.setMinimumWidth(btn.sizeHint().width())
        self.btn_paper_close = icon_button("fa5s.times", tr("Close the Paper Test Strip"))
        self.btn_paper_close.clicked.connect(lambda: self.controller.toggle_paper_strip(force=False))
        for widget in (self.paper_layout, self.paper_strip_combo):
            widget.setFixedHeight(BAR_BUTTON_HEIGHT)
        self.btn_paper_close.setFixedHeight(btn_height)
        layout.addWidget(field_label(tr("Paper test strip")))
        for widget in (
            self.paper_layout,
            self.paper_strip_hint,
            self.paper_strip_combo,
            self._paper_separator,
            self.btn_paper_back,
            self.btn_paper_keep,
            self.btn_paper_close,
        ):
            layout.addWidget(widget)
        v_layout.addWidget(self._paper_row, 0, Qt.AlignmentFlag.AlignHCenter)

    def _sync_paper_row(self) -> None:
        combo = self.paper_strip_combo
        labels = [resolve_paper(key).label for key in self.controller.paper_strip_keys()]
        state = self.session.state
        shown = state.test_strip_expanded
        self.paper_layout.blockSignals(True)
        self.paper_layout.setCurrentIndex(max(0, self.paper_layout.findData(state.test_strip_layout)))
        self.paper_layout.blockSignals(False)
        self.btn_paper_back.setText(icon_text(tr("Grid") if state.test_strip_layout == "grid" else tr("Strip")))
        self.btn_paper_back.setMinimumWidth(self.btn_paper_back.sizeHint().width())
        combo.blockSignals(True)
        if [combo.itemText(i) for i in range(combo.count())] != labels:
            combo.clear()
            combo.addItems(labels)
            combo.setMinimumWidth(combo.sizeHint().width())
        if shown is not None:
            combo.setCurrentIndex(shown)
        combo.blockSignals(False)
        self.paper_layout.setVisible(shown is None)
        self.paper_strip_hint.setVisible(shown is None)
        for widget in (combo, self._paper_separator, self.btn_paper_back, self.btn_paper_keep):
            widget.setVisible(shown is not None)
        self._relayout()

    def _build_zone_row(self, v_layout: QVBoxLayout) -> None:
        """While the Zone Overlay is up the bar shows its look, its legend and a close button."""
        self._zone_row = QWidget()
        self._zone_row.hide()
        layout = QHBoxLayout(self._zone_row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(THEME.space_md)
        self.zone_style = SegmentedChoice(
            tuple(zip(("fa5s.palette", "mdi6.gradient-horizontal", "mdi.grid"), ZONE_STYLES.values())),
            tr("Zone Overlay look: a color per zone, eleven gray steps, or zone numbers on a grid."),
            data=tuple(ZONE_STYLES),
        )
        self.zone_style.currentChanged.connect(lambda _i: self.controller.set_zones_style(self.zone_style.currentData()))
        self.zone_legend = ZoneLegend()
        self.zone_legend.zone_hovered.connect(self.controller.highlight_zone)
        self.btn_zone_close = icon_button("fa5s.times", tr("Close the Zone Overlay"))
        self.btn_zone_close.clicked.connect(lambda: self.controller.toggle_zones_overlay(force=False))
        self.zone_style.setFixedHeight(BAR_BUTTON_HEIGHT)
        self.btn_zone_close.setFixedHeight(BAR_TRACK_BUTTON_HEIGHT)
        self._zone_label = field_label(tr("Zone overlay"))
        layout.addWidget(self._zone_label)
        layout.addWidget(self.zone_style)
        layout.addWidget(self._create_separator())
        layout.addWidget(self.zone_legend)
        layout.addWidget(self.btn_zone_close)
        v_layout.addWidget(self._zone_row, 0, Qt.AlignmentFlag.AlignHCenter)

    def _build_average_row(self, v_layout: QVBoxLayout) -> None:
        """While the Roll Average peek is up the bar names the baseline, holds the averages it
        prints with, and keeps them or leaves the frame as it was."""
        self._average_row = QWidget()
        self._average_row.hide()
        layout = QHBoxLayout(self._average_row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(THEME.space_md)
        self._average_label = field_label(tr("Roll average"))
        self.average_scope = SegmentedChoice(
            (("mdi6.select-group", tr("Scene")), ("fa5s.film", tr("Roll"))),
            tr("Which average the peek prints with: the scene's or the roll's"),
            data=("scene", "roll"),
        )
        self.average_scope.setFixedHeight(BAR_BUTTON_HEIGHT)
        self.average_scope.currentChanged.connect(
            lambda _i: self.controller.set_average_peek_scope("roll" if self.average_scope.currentData() == "roll" else None)
        )
        self.average_hint = hint_label()
        self.average_hint.setWordWrap(False)
        self._average_toggles = {
            "use_luma_average": tool_toggle(
                "mdi6.brightness-6", tr("Luma"), tr("The roll's average black and white points; right-click turns every average on or off.")
            ),
            "use_color_average": tool_toggle("mdi6.palette-outline", tr("Color"), tr("The roll's average color balance.")),
            "use_cast_average": tool_toggle("mdi6.scale-balance", tr("Cast"), tr("The roll's gray balance for Cast Removal.")),
        }
        for field_name, toggle in self._average_toggles.items():
            toggle.setFixedHeight(BAR_BUTTON_HEIGHT)
            toggle.toggled.connect(lambda on, f=field_name: self.controller.set_average_peek_axes(**{f: on}))
        self._average_row_right_click = RightClickToggleAll(list(self._average_toggles.values()))
        self._average_row_right_click.toggledAll.connect(self._on_average_all)
        self.btn_average_keep = labeled_action("fa5s.check", tr("Apply"), "", primary=True)
        self.btn_average_keep.setToolTip(wrap_tooltip(tr("Apply these averages to this frame, as one edit")))
        self.btn_average_keep.setMinimumWidth(self.btn_average_keep.sizeHint().width())
        self.btn_average_keep.clicked.connect(self.controller.keep_average_peek)
        self.btn_average_close = icon_button("fa5s.times", tr("Leave the frame as it was"))
        self.btn_average_close.clicked.connect(lambda: self.controller.toggle_average_peek(force=False))
        self.btn_average_keep.setFixedHeight(BAR_BUTTON_HEIGHT)
        self.btn_average_close.setFixedHeight(BAR_TRACK_BUTTON_HEIGHT)
        layout.addWidget(self._average_label)
        layout.addWidget(self.average_scope)
        layout.addWidget(self.average_hint)
        for toggle in self._average_toggles.values():
            layout.addWidget(toggle)
        layout.addWidget(self._create_separator())
        layout.addWidget(self.btn_average_keep)
        layout.addWidget(self.btn_average_close)
        v_layout.addWidget(self._average_row, 0, Qt.AlignmentFlag.AlignHCenter)

    def _on_average_all(self, on: bool) -> None:
        shown = {f: on for f, t in self._average_toggles.items() if not t.isHidden() and t.isEnabled()}
        self.controller.set_average_peek_axes(**shown)

    def _sync_average_row(self) -> None:
        state = self.session.state
        process = state.config.process
        cast = process.process_mode == ProcessMode.C41
        self._average_toggles["use_cast_average"].setVisible(cast)
        for field_name, toggle in self._average_toggles.items():
            toggle.blockSignals(True)
            toggle.setChecked(bool(state.average_peek_axes.get(field_name)))
            toggle.blockSignals(False)
        scopes = self.controller.average_scopes()
        self.average_scope.setVisible(bool(scopes))
        self.average_scope.blockSignals(True)
        self.average_scope.setCurrentIndex(1 if state.average_peek_scope == "roll" else 0)
        self.average_scope.blockSignals(False)
        found = self._saved_baseline(state, state.average_peek_scope or "scene")
        self._average_label.setText(rolls.source_label(self.session.repo, found[1]) if found else tr("Roll average"))
        outlier = self._average_outlier(state, found)
        scene = found is not None and found[1].startswith("scene:")
        self.average_hint.setText((tr("Far from the scene color") if scene else tr("Far from the roll color")) if outlier else "")
        set_hint_kind(self.average_hint, "warning" if outlier else "muted")
        self.average_hint.setVisible(outlier)

    def _saved_baseline(self, state, scope=None):
        """The baseline the frame's averages read, or the one *scope* names."""
        scope = scope or rolls.average_scope(state.config.process)
        return rolls.frame_saved_baseline(self.session.repo, state.active_roll_id, state.current_file_hash or "", scope)

    def _average_outlier(self, state, found) -> bool:
        """Whether the baseline *found* for this frame measured it far from the roll color."""
        if found is None:
            return False
        h = state.current_file_hash or ""
        return bool({h, rolls.unforked_hash(h)} & set(found[0].get("outliers", ())))

    def _build_light_table_row(self, v_layout: QVBoxLayout, icon_size: QSize, rotate_icon_size: QSize, btn_height: int) -> None:
        """While the Light Table is open, Photo mode acts on the selected frames: Keep and Reject,
        turn, Auto Crop and Export, the rest of the frame menu behind ⋯, then the frame size."""
        from negpy.desktop.view.widgets.sliders import CompactSlider

        self._light_table = False
        self._light_table_row = QWidget()
        self._light_table_row.hide()
        layout = QHBoxLayout(self._light_table_row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(THEME.space_md)
        self.selection_count = hint_label()
        self.selection_count.setWordWrap(False)
        self.selection_count.setToolTip(tr("A click selects a frame, and a double-click or Enter opens it"))

        self.btn_lt_keep = tool_toggle("mdi6.check-circle-outline", tr("Keep"), "")
        self._tip(self.btn_lt_keep, tr("Keep the selected frames"), "toggle_keep")
        self.btn_lt_keep.clicked.connect(lambda: self._mark_selection("keeper"))
        self.btn_lt_reject = tool_toggle("mdi6.close-circle-outline", tr("Reject"), "")
        self._tip(self.btn_lt_reject, tr("Reject the selected frames: batch export skips them"), "toggle_reject")
        self.btn_lt_reject.clicked.connect(lambda: self._mark_selection("excluded"))
        turn = [
            self._action_button(_icon("mdi6.file-rotate-left"), tr("Rotate CCW"), "rotate_ccw", lambda: self.rotate(1)),
            self._action_button(_icon("mdi6.file-rotate-right"), tr("Rotate CW"), "rotate_cw", lambda: self.rotate(-1)),
            self._action_button(_icon("fa5s.arrows-alt-h"), tr("Flip Horizontal"), "flip_h", lambda: self.flip("horizontal")),
            self._action_button(_icon("fa5s.arrows-alt-v"), tr("Flip Vertical"), "flip_v", lambda: self.flip("vertical")),
        ]
        for btn, size in zip(turn, (rotate_icon_size, rotate_icon_size, icon_size, icon_size)):
            btn.setIconSize(size)
        self.btn_lt_auto_crop = labeled_action("fa5s.magic", tr("Auto Crop"), "")
        self._tip(self.btn_lt_auto_crop, tr("Auto Crop the selected frames; crops you drew stay"), None)
        self.btn_lt_auto_crop.clicked.connect(lambda: self.controller.request_batch_auto_crop(assets=self._selected_frames()))
        self.btn_lt_export = labeled_action("mdi6.export-variant", tr("Export"), "")
        self._tip(self.btn_lt_export, tr("Export the selected frames with the Export tab's settings, skipping rejected frames"), None)
        self.btn_lt_export.clicked.connect(self.controller.request_export_selected)
        self.btn_lt_more = self._action_button(
            _icon("fa5s.ellipsis-h"), tr("More actions for the selected frames"), None, self._show_frame_menu
        )
        self._lt_labeled = [
            (btn, btn.text(), btn.styleSheet()) for btn in (self.btn_lt_keep, self.btn_lt_reject, self.btn_lt_auto_crop, self.btn_lt_export)
        ]
        for btn in (self.btn_lt_keep, self.btn_lt_reject, self.btn_lt_auto_crop, self.btn_lt_export, *turn, self.btn_lt_more):
            btn.setFixedHeight(btn_height)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)

        # Inverted, so the frames grow to the right like the + key's.
        self.frames_per_row = CompactSlider(tr("Frames per Row"), 1, 2, 1, step=1, precision=1, inverted=True)
        self._tip(
            self.frames_per_row,
            tr("Light Table: the frames to a row; fewer frames are larger"),
            ("light_table_zoom_in", "light_table_zoom_out"),
        )
        # An icon in place of the name, which keeps the name's drag and double-click reset.
        self.frames_per_row.label.setPixmap(qta.icon("mdi6.resize", color=THEME.text_secondary).pixmap(icon_size))
        self.frames_per_row.setFixedHeight(default_button_height())
        self.frames_per_row.slider.setMinimumWidth(self.frames_per_row.slider.sizeHint().width())

        # A narrow bar folds these away in order, each with the separator that leads it: the
        # count, then the turn buttons (their keys stay), then Auto Crop and Export (Export stays in ⋯).
        self._lt_folds: list[list[QWidget]] = [[self.selection_count]]
        for group in (
            (self.btn_lt_keep, self.btn_lt_reject),
            turn,
            (self.btn_lt_auto_crop, self.btn_lt_export),
            (self.btn_lt_more,),
            (self.frames_per_row,),
        ):
            separator = self._create_separator()
            if group is turn or self.btn_lt_export in group:
                self._lt_folds.append([separator, *group])
            elif group[0] is self.btn_lt_keep:
                self._lt_folds[0].append(separator)
            layout.addWidget(separator)
            for widget in group:
                layout.addWidget(widget)
        layout.insertWidget(0, self.selection_count)
        v_layout.addWidget(self._light_table_row, 0, Qt.AlignmentFlag.AlignHCenter)

    def _selected_frames(self) -> list[dict]:
        state = self.session.state
        files = state.uploaded_files
        return [files[i] for i in (state.selected_indices or [state.selected_file_idx]) if 0 <= i < len(files)]

    def _mark_selection(self, mark: str) -> None:
        self.session.toggle_mark(mark)
        self._sync_light_table_row()

    def _sync_light_table_row(self) -> None:
        """The count and the marks every selected frame shares; the toggles never set state themselves."""
        frames = self._selected_frames()
        self.selection_count.setText(count_of(len(frames), "frame"))
        for btn, mark in ((self.btn_lt_keep, "keeper"), (self.btn_lt_reject, "excluded")):
            btn.blockSignals(True)
            btn.setChecked(bool(frames) and all(f.get(mark) for f in frames))
            btn.blockSignals(False)

    def _show_frame_menu(self) -> None:
        from negpy.desktop.view.main_window import MainWindow

        win = self.window()
        if isinstance(win, MainWindow):
            menu = win.file_browser.build_frame_menu()
            menu.exec(self.btn_lt_more.mapToGlobal(QPoint(0, -menu.sizeHint().height())))

    def _name_light_table_buttons(self, named: bool) -> None:
        for btn, text, sheet in self._lt_labeled:
            btn.setText(text if named else "")
            btn.setStyleSheet(sheet if named else tool_toggle_qss(icon_only=True))
            # A narrow bar lays out at minimum widths, which would cut a name down to the icon.
            btn.setMinimumWidth(btn.sizeHint().width())

    def set_light_table(self, on: bool) -> None:
        self._light_table = on
        self._sync_light_table_row()
        self._sync_rows()

    def set_frames_per_row(self, columns: int, fewest: int, default: int, most: int) -> None:
        self.frames_per_row.set_range(fewest, most)
        self.frames_per_row.set_default(default)
        self.frames_per_row.setValue(columns)

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
            self.tool_name.setText(icon_text(name) if button is not None else name)
            self.tool_name.setMinimumWidth(self.tool_name.sizeHint().width())
            self.tool_hint.setText(_CLONE_SOURCE_HINT if mode == ToolMode.CLONE and state.clone_picking else hint)
            self.tool_name.setToolTip(self.tool_hint.text())
            self.btn_tool_undo.setEnabled(state.undo_index > 0)
            self.btn_tool_redo.setEnabled(state.undo_index < state.max_history_index)
        # After the texts: the bar measures them.
        if mode != self._shown_tool:
            self._shown_tool = mode
            self._swap_tool_slider(spec[2] if spec else "")
        # An armed tool outranks the paper strip, which outranks the Zone Overlay, which outranks
        # the view row.
        paper = spec is None and state.test_strip_kind == "paper" and bool(state.test_strip)
        average = spec is None and not paper and state.average_peek
        rows = (spec is not None, paper, spec is None and not paper and not average and bool(state.zones_overlay), average)
        if paper:
            self._sync_paper_row()
        if average:
            self._sync_average_row()
        if rows != self._shown_rows:
            self._shown_rows = rows
            if rows[2]:
                self._sync_zone_row()
            self._tool_row.setVisible(rows[0])
            self._paper_row.setVisible(rows[1])
            self._zone_row.setVisible(rows[2])
            self._average_row.setVisible(rows[3])
            self._sync_rows()
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

    @staticmethod
    def _menu_section(menu: QMenu, text: str) -> None:
        """A sub-header row that opens a group of a menu's items."""
        header = section_subheader(text)
        header.setContentsMargins(THEME.space_lg, 0, THEME.space_lg, 0)
        section = QWidgetAction(menu)
        section.setDefaultWidget(header)
        section.setEnabled(False)
        menu.addAction(section)

    def _fill_tools_menu(self, menu: QMenu) -> None:
        """Each item runs the tool's own shortcut action, which shows its tab and refuses a gated tool."""
        menu.setToolTipsVisible(True)
        entries = []
        for spec in _CANVAS_TOOLS:
            if isinstance(spec, str):
                self._menu_section(menu, spec)
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
                action.setCheckable(True)
                action.setChecked(mode == current)

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
        canvas = getattr(self.controller, "canvas", None)
        if canvas is not None:
            canvas.overlay.finish_in_progress()
        self.controller.cancel_active_tool()

    def set_roll_row(self, row: QWidget) -> None:
        """The Film Strip's own row, which Roll mode shows."""
        self._roll_row = row
        self._rows_layout.addWidget(row)
        self._adopt_row(row)
        self._sync_rows()

    def _sync_rows(self) -> None:
        """A canvas row (an armed tool, the paper strip, the Zone Overlay) outranks the mode's row."""
        free = not any(self._shown_rows)
        self._view_row.setVisible(free and self._mode == "photo" and not self._light_table)
        self._light_table_row.setVisible(free and self._mode == "photo" and self._light_table)
        if self._roll_row is not None:
            self._roll_row.setVisible(free and self._mode == "roll")
        self._relayout()

    def mode(self) -> str:
        return self._mode

    def set_mode(self, mode: str) -> None:
        if mode == self._mode:
            return
        self._mode = mode
        self.session.repo.save_global_setting(_MODE_SETTING_KEY, mode)
        self.mode_switch.blockSignals(True)
        self.mode_switch.setCurrentIndex(_MODES.index(mode))
        self.mode_switch.blockSignals(False)
        self._sync_rows()

    def toggle_mode(self) -> None:
        self.set_mode("roll" if self._mode == "photo" else "photo")

    def reveal(self, target: QWidget) -> None:
        """Switch to the mode whose row holds target."""
        if self._view_row.isAncestorOf(target):
            self.set_mode("photo")
        elif self._roll_row is not None and self._roll_row.isAncestorOf(target):
            self.set_mode("roll")

    def set_strip_expanded(self, expanded: bool) -> None:
        self.btn_toggle_strip.setIcon(_icon("fa5s.chevron-down" if expanded else "fa5s.chevron-up"))

    def _sync_view_state(self, *_args) -> None:
        """Mirror the canvas source and the overlays from state, which the controller sets before it signals."""
        state = self.session.state
        active = next((i for i, (flag, *_rest) in enumerate(_VIEW_SOURCES) if flag and getattr(state, flag)), 0)
        self._show_source(active)
        self._source_actions[_SOURCE_INDEX["flatfield_peek"]].setEnabled(bool(state.config.flatfield.profile_id))
        for flag, action in self._overlay_actions.items():
            action.setChecked(bool(getattr(state, flag)))
        for flag, action in self._ov_view_actions.items():
            action.setChecked(bool(getattr(state, flag)))
        self._info_actions[state.canvas_info_level].setChecked(True)

    def _show_source(self, index: int) -> None:
        _flag, icon, label, _key, _tip = _VIEW_SOURCES[index]
        self._source_actions[index].setChecked(True)
        self.source_choice.setIcon(_icon(icon))
        self.source_choice.setText(icon_text(label))

    def mirror_export(self, source: QPushButton, scope_menu: QMenu) -> None:
        """The bar's Export clicks the Export panel's button and opens its scope menu, so both share one scope."""
        self.btn_export.clicked.connect(source.click)
        self._export_menu.addActions(scope_menu.actions())
        # The menu lists the current, selected and all-visible scopes in that order.
        labels = (tr("Frame"), tr("Selected"), tr("Roll"))
        width = 0
        for label in labels:
            self._show_export_scope(label)
            width = max(width, self.btn_export.sizeHint().width())
        self.btn_export.setMinimumWidth(width)
        for action, label in zip(scope_menu.actions(), labels, strict=True):
            action.toggled.connect(lambda checked, label=label: checked and self._show_export_scope(label))
            if action.isChecked():
                self._show_export_scope(label)
        _ExportMirror(source, self.btn_export, self.btn_export_menu)

    def _show_export_scope(self, label: str) -> None:
        self.btn_export.setText(icon_text(label))

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
            "average_peek": controller.toggle_average_peek,
            "flat_peek": controller.toggle_flat_peek,
            "embedded_peek": controller.toggle_embedded_peek,
            "flatfield_peek": controller.toggle_flatfield_peek,
            "export_size_view": controller.toggle_export_size_view,
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
        for signal in (
            controller.compare_changed,
            controller.flat_peek_changed,
            controller.negative_peek_changed,
            controller.average_peek_changed,
            controller.embedded_peek_changed,
            controller.flatfield_peek_changed,
            controller.export_size_view_changed,
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
        controller.config_updated.connect(lambda: "average" in self._item_ids and self._sync_average_button())
        self.session.asset_model.layoutChanged.connect(self._update_ui_state)
        self.session.files_changed.connect(self._sync_light_table_row)
        controller.tool_sync_requested.connect(self._sync_tool_row)
        controller.zones_overlay_changed.connect(lambda _on: self._sync_tool_row())
        controller.average_peek_changed.connect(lambda _on: self._sync_tool_row())
        controller.test_strip_changed.connect(lambda _up: self._sync_tool_row())
        controller.zones_style_changed.connect(lambda _style: self._sync_zone_row())
        self.session.color_vision_changed.connect(self._sync_zone_row)

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
        percent = self._zoom_percent
        self.btn_zoom.setText(tr("{value}% HQ", value=percent) if self.session.state.hq_preview else tr("{value}%", value=percent))
        # A longer readout raises the minimum and asks for a new layout, or it clips under the chevron.
        width = max(self._zoom_min_width, self.btn_zoom.sizeHint().width())
        if width != self.btn_zoom.minimumWidth():
            self.btn_zoom.setMinimumWidth(width)
            self._relayout()

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
        # A selection that leaves out the open frame (a ctrl-click toggled it off, or a Light
        # Table click picked another frame) must not rotate it: selection wins over "what's on
        # screen" once there is one, the same rule toggle_mark already uses.
        include_active = not state.selected_indices or state.selected_file_idx in state.selected_indices
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
        include_active = not state.selected_indices or state.selected_file_idx in state.selected_indices
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
        menu.addAction(label_with_shortcut(tr("Edit Toolbar…"), "edit_toolbar"), self.open_toolbar_editor)
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
        # The dot on ⋯ leads to Help, where the update waits.
        self._help_menu.menuAction().setIcon(qta.icon("fa5s.download", color=THEME.status_success))
        self._update_action.setIcon(qta.icon("fa5s.download", color=THEME.status_success))
        self._update_action.setText(tr("Update to v{version}…", version=version))

    def _check_for_updates(self) -> None:
        from negpy.desktop.view.main_window import MainWindow

        win = self.window()
        if isinstance(win, MainWindow):
            win.update_notice.check_for_updates()

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

    def _toggle_light_table(self) -> None:
        from negpy.desktop.view.main_window import MainWindow

        win = self.window()
        if isinstance(win, MainWindow):
            win.set_light_table(not win.light_table_active())

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
        self._sync_light_table_row()

        geo = state.config.geometry
        self.btn_flip_h.setChecked(geo.flip_horizontal)
        self.btn_flip_v.setChecked(geo.flip_vertical)

        self.btn_undo.setEnabled(state.undo_index > 0)
        self.btn_redo.setEnabled(state.undo_index < state.max_history_index)
        self._ov_undo_action.setEnabled(state.undo_index > 0)
        self._ov_redo_action.setEnabled(state.undo_index < state.max_history_index)
        self._action_paste.setEnabled(state.clipboard is not None)
        self.btn_paste.setEnabled(state.clipboard is not None)
        # A roll lookup on every state change, so only while the button is on the row.
        if "reset_to_roll" in self._item_ids:
            self.btn_reset_to_roll.setEnabled(bool(self.controller.can_revert_frame_to_roll()))
        if "average" in self._item_ids:
            self._sync_average_button()

    def _on_average_clicked(self) -> None:
        self.controller.toggle_average_peek(force=True)
        self._sync_average_button()

    def _average_fields(self) -> list[str]:
        """The averages that apply to this frame: Cast only on a color negative with a neutral axis to use."""
        state = self.session.state
        process = state.config.process
        fields = ["use_luma_average", "use_color_average"]
        if process.process_mode == ProcessMode.C41:
            found = self._saved_baseline(state) if state.active_roll_id else None
            if (found is not None and found[0]["axis"] is not None) or process.use_cast_average:
                fields.append("use_cast_average")
        return fields

    def _frame_average_states(self) -> list[bool]:
        process = self.session.state.config.process
        return [bool(getattr(process, f)) for f in self._average_fields()]

    def _on_use_all_averages(self, on: bool) -> None:
        self.controller.set_roll_average(**{f: on for f in self._average_fields()})

    def _sync_average_button(self) -> None:
        """Checked while the frame uses an average; usable once the roll has a baseline for the frame, or to turn one off."""
        state = self.session.state
        process = state.config.process
        found = self._saved_baseline(state) if state.active_roll_id else None
        on = process.use_luma_average or process.use_color_average or process.use_cast_average
        self.btn_average.setChecked(on)
        # The transfer path prints from a fixed window, with no measurement to average.
        printed = render_path(process) is RenderPath.PRINT
        self.btn_average.setEnabled(printed and (found is not None or on))
        outlier = self._average_outlier(state, found)
        self.btn_average.setIcon(qta.icon("mdi6.alert", color=THEME.warn_amber) if outlier else self._average_icon)
        if self.btn_average.isEnabled() or not printed:
            tip = tooltip_with_shortcut(
                tr("Roll Average: try the average on this frame; right-click turns every average on or off."), "toggle_average_peek"
            )
        elif state.active_roll_id and self.controller.measuring_baseline() is True:
            tip = tr("Measuring the average…")
        else:
            tip = tr("No average for this frame yet")
        self.btn_average.setToolTip(wrap_tooltip(tip))

    def _activate_layout(self) -> None:
        # Innermost first: an outer layout reads its rows' size hints as of their last activation.
        for widget in (
            self._view_row,
            self._tool_row,
            self._paper_row,
            self._zone_row,
            self._average_row,
            self._light_table_row,
            self._rows,
            self,
        ):
            layout = widget.layout()
            if layout is not None:
                layout.activate()

    def _show_tools_name(self, shown: bool) -> None:
        self.btn_tools.setText(icon_text(tr("Tools")) if shown else "")
        self.btn_tools.setMinimumWidth(self.btn_tools.sizeHint().width())

    def _bar_width(self) -> int:
        """Measured width of what the bar shows now (sizeHint can stay stale)."""
        self._activate_layout()
        return self.minimumSizeHint().width()

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
        """Lay the chosen items out in order.

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
            title=tr("Edit Toolbar"),
            chosen_header=tr("TOOLBAR"),
            hint=tr(
                "Drag to reorder. What does not fit the toolbar hides, geometry first and navigation last; the ⋯ menu holds every action."
            ),
            defaults=list(DEFAULT_TOOLBAR_IDS),
            repo=self.session.repo,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        self._item_ids = dialog.selected_ids()
        self.session.repo.save_global_setting(_TOOLBAR_SETTING_KEY, self._item_ids)
        self._rebuild_row()
        self._update_ui_state()
        self._relayout()

    def _adopt_row(self, root: QWidget) -> None:
        """Flatten the buttons under root into their group's track, clear its row containers so
        the tracks show through, and repaint the tracks when any of its layouts changes. A popup
        menu's widgets keep their own look."""
        window = root.window()
        for btn in root.findChildren(QPushButton):
            if btn.window() is window and not btn.property("primary"):
                btn.setFlat(True)
        for widget in (root, *root.findChildren(QWidget)):
            if widget.window() is window and widget.layout() is not None:
                widget.installEventFilter(self)
                if type(widget) is QWidget and not widget.objectName():
                    widget.setObjectName("toolbar_row")
                    widget.style().unpolish(widget)
                    widget.style().polish(widget)

    def eventFilter(self, obj, event) -> bool:  # noqa: N802
        if event.type() == QEvent.Type.LayoutRequest:
            self.update()
        return super().eventFilter(obj, event)

    def _track_rects(self) -> list[QRect]:
        """Each run of buttons in bar coordinates, with a label between two of them (the frame
        count between Previous and Next). A separator, a field, a slider or a primary button
        ends a run; a segmented control carries its own track, and Export takes a track of its own."""
        rects: list[QRect] = []

        def walk(layout) -> None:
            run: list[QWidget] = []

            def flush() -> None:
                while run and not isinstance(run[0], QAbstractButton):
                    run.pop(0)
                while run and not isinstance(run[-1], QAbstractButton):
                    run.pop()
                if run:
                    rect = QRect()
                    for w in run:
                        rect = rect.united(QRect(w.mapTo(self, QPoint(0, 0)), w.size()))
                    rects.append(rect)
                run.clear()

            for i in range(layout.count()):
                item = layout.itemAt(i)
                widget = item.widget()
                if widget is self._rows:
                    flush()
                    for j in range(self._rows_layout.count()):
                        row = self._rows_layout.itemAt(j).widget()
                        if row is not None and row.isVisible() and row.layout() is not None:
                            walk(row.layout())
                elif item.isEmpty():
                    continue
                elif isinstance(widget, (QAbstractButton, QLabel)) and not widget.property("primary"):
                    run.append(widget)
                elif widget is self.export_split:
                    flush()
                    rects.append(QRect(widget.mapTo(self, QPoint(0, 0)), widget.size()))
                else:
                    flush()
            flush()

        walk(self.layout())
        return rects

    def paintEvent(self, event) -> None:  # noqa: N802
        super().paintEvent(event)
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(QPen(QColor(THEME.border_primary), 1))
        p.setBrush(QColor(THEME.bg_input))
        pad = BAR_TRACK_PAD
        for rect in self._track_rects():
            p.drawRoundedRect(QRectF(rect).adjusted(-pad, -pad, pad, pad), THEME.radius_lg, THEME.radius_lg)

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        # A bar on its own (a test) is sized by set_available_width alone.
        if not self.isWindow() and event.size().width() != event.oldSize().width():
            self.set_available_width(self.width())

    def _relayout(self) -> None:
        self.set_available_width(self._available_width or self.width())

    def set_available_width(self, w: int) -> None:
        """Show as many of the chosen controls as fit the bar width. The overflow menu always
        carries the full action set (see _init_ui), so the menu does not follow the row."""
        self._available_width = w

        if not self._tool_row.isHidden():
            # The hint goes first, then the slider: the name, undo and Done always stay.
            optional = [widget for widget in (self.tool_hint, self._tool_slider) if widget is not None]
            for widget in optional:
                widget.setVisible(True)
            for widget in optional:
                if self._bar_width() <= w:
                    break
                widget.setVisible(False)
            self._activate_layout()
            return
        if not self._average_row.isHidden():
            # The name goes first, then the warning; the averages, Apply and close always stay.
            optional = (self._average_label, self.average_hint)
            for widget in optional:
                widget.setVisible(widget is self._average_label or bool(self.average_hint.text()))
            for widget in optional:
                if self._bar_width() <= w:
                    break
                widget.setVisible(False)
            self._activate_layout()
            return
        if not self._zone_row.isHidden():
            # The name goes first, then the legend's readout.
            steps = (self._zone_label.setVisible, self.zone_legend.show_readout)
            for step in steps:
                step(True)
            for step in steps:
                if self._bar_width() <= w:
                    break
                step(False)
            self._activate_layout()
            return
        if not self._light_table_row.isHidden():
            # The count goes first, then the buttons' names (their tooltips still name them), then
            # the groups _lt_folds lists after the count.
            def fold(widgets: list[QWidget]):
                return lambda shown: [widget.setVisible(shown) for widget in widgets]

            steps = [fold(self._lt_folds[0]), self._name_light_table_buttons, *(fold(group) for group in self._lt_folds[1:])]
            for step in steps:
                step(True)
            for step in steps:
                if self._bar_width() <= w:
                    break
                step(False)
            self._activate_layout()
            return
        if self._view_row.isHidden():
            return

        widgets = [self._row_widgets[item_id] for item_id in self._item_ids]
        for widget in widgets:
            widget.setVisible(True)
        self._sync_separators()
        # A name goes before a button does.
        self._show_tools_name(True)
        if self._bar_width() > w:
            self._show_tools_name(False)

        extras = [item_id for item_id in reversed(self._item_ids) if item_id not in _COLLAPSE_ORDER]
        for item_id in extras + [item_id for item_id in _COLLAPSE_ORDER if item_id in self._item_ids]:
            if self._bar_width() <= w:
                break
            self._row_widgets[item_id].setVisible(False)
            self._sync_separators()

        self._activate_layout()


class _ExportMirror(QObject):
    """Copies the Export panel button's tooltip and enabled state to the bar's split button."""

    def __init__(self, source: QPushButton, main: QPushButton, menu_btn: QToolButton):
        super().__init__(main)
        self._source, self._main, self._menu_btn = source, main, menu_btn
        source.installEventFilter(self)
        self._sync()

    def eventFilter(self, obj, event) -> bool:  # noqa: N802
        if event.type() in (QEvent.Type.EnabledChange, QEvent.Type.ToolTipChange):
            self._sync()
        return False

    def _sync(self) -> None:
        self._main.setEnabled(self._source.isEnabled())
        self._menu_btn.setEnabled(self._source.isEnabled())
        self._main.setToolTip(self._source.toolTip())
