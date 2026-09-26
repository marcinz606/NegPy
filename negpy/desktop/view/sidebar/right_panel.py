from typing import Any, Dict

import numpy as np
from PyQt6.QtCore import QPoint, Qt, QTimer
from PyQt6.QtGui import QFont, QFontMetrics
from PyQt6.QtWidgets import (
    QApplication,
    QPushButton,
    QScrollArea,
    QSplitter,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from negpy.desktop.controller import AppController
from negpy.desktop.view.shortcut_registry import tooltip_with_shortcut
from negpy.desktop.view.sidebar.controls_panel import ControlsPanel
from negpy.desktop.view.sidebar.export import ExportSidebar
from negpy.desktop.view.sidebar.favourites import FavouritesSidebar
from negpy.desktop.view.sidebar.history import HistoryPanel
from negpy.desktop.view.sidebar.metadata import MetadataSidebar
from negpy.desktop.view.styles.fonts import ui_font_family
from negpy.desktop.view.styles.templates import EditedDot
from negpy.desktop.view.styles.theme import THEME
from negpy.desktop.view.widgets.charts import PhotometricCurveWidget, StepWedgeWidget, ZoneStripWidget
from negpy.desktop.view.widgets.collapsible import CollapsibleSection, make_section
from negpy.desktop.view.widgets.gear_library_panel import GearLibraryPanel
from negpy.desktop.view.widgets.granular_settings_dialog import open_apply_dialog
from negpy.desktop.view.widgets.tab_header import TabHeader
from negpy.desktop.settings_catalog import rows_for_fields
from negpy.services.assets.rolls import card_fields
from negpy.desktop.view.widgets.stats import DensitometerRow, NegativeStatsWidget, ZonePlacementRows
from negpy.desktop.view.widgets.overflow_bar import OverflowBar

# ControlsPanel sections built into the Roll tab (_build_roll_page), not a Frame sub-tab --
# reveal_section routes these to the Roll group instead of Frame's inner tab switcher.
# The Roll tab's cards that own settings, for its header's count, reset and apply.
_ROLL_TAB_CARDS = ("film", "sensor", "cast_removal", "autocrop", "baseline", "process", "demosaic", "lens", "flatfield")

_ROLL_SECTION_ATTRS = frozenset(
    {
        "assembly_section",
        "sensor_section",
        "demosaic_section",
        "baseline_section",
        "process_section",
        "autocrop_section",
        "optics_section",
    }
)


def default_analysis_split(screen) -> list[int]:
    height = screen.availableGeometry().height() if screen is not None else 1080
    top = min(320, height * 3 // 10)
    return [top, max(600, height - top)]


def _tab_width(labels: list[str], pixel_size: int) -> int:
    """Widest label at the checked weight, so a tab spills into » before its text clips."""
    font = QFont(ui_font_family())
    font.setPixelSize(pixel_size)
    font.setWeight(QFont.Weight.DemiBold)
    return max(QFontMetrics(font).horizontalAdvance(label) for label in labels) + 2 * THEME.space_md


class RightPanel(QWidget):
    """Right sidebar panel: a flat tab switcher across Roll / Frame / Metadata / Gear /
    Export / Scan, in the order an edit moves through them. Frame, Metadata and Gear pin
    a section above their own scroll area (Analysis, Preview, the Items/Presets
    switcher); Export and Scan are plain pages."""

    def __init__(self, controller: AppController):
        super().__init__()
        self.controller = controller
        # Heal/scratch tool suspended by leaving the Retouch tab; restored on return.
        self._suspended_retouch_tool = None

        self._init_ui()
        self._connect_signals()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        def wrap_scroll(widget: QWidget) -> QScrollArea:
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setWidget(widget)
            return scroll

        frame_page = self._build_frame_page(wrap_scroll)
        roll_page = self._build_roll_page()

        self.export_sidebar = ExportSidebar(self.controller)
        self.metadata_sidebar = MetadataSidebar(self.controller)
        self.gear_panel = GearLibraryPanel(current_config_fn=lambda: self.controller.state.config, repo=self.controller.session.repo)
        self.gear_panel.library_changed.connect(self.metadata_sidebar._on_library_changed)
        self.gear_panel.presets_changed.connect(self.metadata_sidebar._refresh_metadata_presets)

        from negpy.desktop.view.sidebar.scan import ScanSidebar
        from negpy.desktop.view.sidebar.scanlight import ScanlightSidebar

        self.scan_sidebar = ScanSidebar(self.controller)
        self.scanlight_sidebar = ScanlightSidebar(self.controller)
        self.scan_page = self._build_scan_page()

        # (key, label, content_widget)
        group_specs = [
            ("roll", "Roll", roll_page),
            ("frame", "Frame", frame_page),
            ("metadata", "Metadata", self.metadata_sidebar),
            ("gear", "Gear", self.gear_panel),
            ("export", "Export", self.export_sidebar),
            ("scan", "Scan", self.scan_page),
        ]

        # Spills into a » menu when the panel is narrowed
        self.group_switcher = OverflowBar(
            tile=True, height=38, min_item=_tab_width([spec[1] for spec in group_specs], THEME.font_size_base)
        )
        self.group_stack = QStackedWidget()
        self.group_stack.setContentsMargins(0, 0, 0, 0)

        self._group_buttons: list[QPushButton] = []
        self._group_keys: list[str] = []
        self._group_tooltips: list[str] = []
        self._active_group = 0
        self._scan_group_index = -1

        for i, (key, tooltip, content) in enumerate(group_specs):
            btn = QPushButton(tooltip)
            btn.setObjectName("right_tab_btn")
            btn.setToolTip(tooltip)
            btn.setCheckable(True)
            btn.setFixedHeight(38)
            btn.clicked.connect(lambda _checked=False, idx=i: self._switch_group(idx))
            self.group_switcher.add_button(btn, tooltip)

            # Frame, Metadata and Gear manage their own scrolling (a pinned section or subtab
            # switcher above a scroll area); the other pages are one control column each, so
            # the page itself needs it.
            page = content if key in ("frame", "metadata", "gear") else wrap_scroll(content)
            self.group_stack.addWidget(page)
            self._group_buttons.append(btn)
            self._group_keys.append(key)
            self._group_tooltips.append(tooltip)
            if key == "scan":
                self._scan_group_index = i

        layout.addWidget(self.group_switcher)
        layout.addWidget(self.group_stack, 1)

        self.apply_shortcut_tooltips()

        repo = self.controller.session.repo
        saved_group = repo.get_global_setting("right_panel_group", 0)
        self._switch_group(saved_group if isinstance(saved_group, int) and 0 <= saved_group < len(self._group_buttons) else 0)

    def _build_frame_page(self, wrap_scroll) -> QWidget:
        """Sticky Analysis section pinned above a second, inner tab switcher for the
        per-image workflow control groups, Favorites and History -- everything that
        changes what the canvas shows for the one loaded frame."""
        page = QWidget()
        page_layout = QVBoxLayout(page)
        page_layout.setContentsMargins(0, 0, 0, 0)
        page_layout.setSpacing(0)

        analysis_content = QWidget()
        analysis_layout = QVBoxLayout(analysis_content)
        analysis_layout.setContentsMargins(5, 5, 5, 5)

        self.curve_widget = PhotometricCurveWidget()
        self.step_wedge = StepWedgeWidget()
        self.zone_strip = ZoneStripWidget()
        self.probe_row = DensitometerRow()
        self.zone_placement = ZonePlacementRows()
        self.stats_widget = NegativeStatsWidget()
        self._clip_fracs: tuple = (None, None)

        repo = self.controller.session.repo
        self.curve_widget.set_log_scale(bool(repo.get_global_setting("histogram_log_scale")))
        self.curve_widget.scale_changed.connect(lambda enabled: repo.save_global_setting("histogram_log_scale", bool(enabled)))

        analysis_layout.addWidget(self.curve_widget, 1)
        analysis_layout.addWidget(self.step_wedge, 0)
        analysis_layout.addWidget(self.zone_strip, 0)
        analysis_layout.addWidget(self.probe_row, 0)
        analysis_layout.addWidget(self.zone_placement, 0)
        analysis_layout.addWidget(self.stats_widget, 0)

        repo = self.controller.session.repo
        self.analysis_section = make_section(
            repo, "Analysis", "analysis", analysis_content, "fa5s.chart-bar", THEME.sidebar_expanded_defaults["analysis"]
        )
        analysis_expanded = self.analysis_section.toggle_button.isChecked()

        # Tab content widgets
        self.controls_panel = ControlsPanel(self.controller)
        self.favourites_sidebar = FavouritesSidebar(self.controller, self.controls_panel)
        self.history_panel = HistoryPanel(self.controller)

        favourites_page = QWidget()
        favourites_layout = QVBoxLayout(favourites_page)
        favourites_layout.setContentsMargins(0, 0, 0, 0)
        favourites_layout.setSpacing(THEME.space_lg)
        favourites_layout.addWidget(self.favourites_sidebar)
        favourites_layout.addWidget(self.controls_panel.presets_section)
        favourites_layout.addStretch(1)

        # Tab descriptors: the workflow control-group pages, then Favorites and History.
        # (key, label, tooltip, content_widget, [section_attrs])
        tab_specs = [(page["key"], page["label"], page["tooltip"], page["widget"], page["sections"]) for page in self.controls_panel.pages]
        self._frame_tab_headers = {page["key"]: page["header"] for page in self.controls_panel.pages if page["header"]}
        tab_specs += [
            ("favourites", "Favorites", "Favorites", favourites_page, ["presets_section"]),
            ("history", "History", "History", self.history_panel, []),
        ]

        # Spills into a » menu when the panel is narrowed
        self.switcher = OverflowBar(tile=True, height=30, min_item=_tab_width([spec[1] for spec in tab_specs], THEME.font_size_small))

        self.stack = QStackedWidget()
        self.stack.setContentsMargins(0, 8, 0, 0)

        self._tab_buttons: list[QPushButton] = []
        self._tab_keys: list[str] = []
        self._tab_tooltips: list[str] = []
        self._section_tab_index: dict[str, int] = {}
        self._tab_sections: dict[int, list[str]] = {}
        self._tab_edited: list[bool] = []
        self._active_index = 0

        for i, (key, label, tooltip, content, section_attrs) in enumerate(tab_specs):
            btn = QPushButton(label)
            btn.setObjectName("sub_tab_btn")
            btn.setToolTip(tooltip)
            btn.setCheckable(True)
            btn.setFixedHeight(30)
            btn.edited_dot = EditedDot(btn)
            btn.clicked.connect(lambda _checked=False, idx=i: self._switch_tab(idx))
            self.switcher.add_button(btn, tooltip)

            self.stack.addWidget(wrap_scroll(content))
            self._tab_buttons.append(btn)
            self._tab_keys.append(key)
            self._tab_tooltips.append(tooltip)
            self._tab_edited.append(False)
            if section_attrs:
                self._tab_sections[i] = section_attrs
            for attr in section_attrs:
                self._section_tab_index[attr] = i

        # Tabs (switcher + stack) live in the bottom splitter pane
        tabs_container = QWidget()
        tabs_vbox = QVBoxLayout(tabs_container)
        tabs_vbox.setContentsMargins(0, 0, 0, 0)
        tabs_vbox.setSpacing(0)
        tabs_vbox.addWidget(self.switcher)
        tabs_vbox.addWidget(self.stack, 1)

        # Vertical splitter lets the user resize Analysis vs. the tabs below
        self.splitter = QSplitter(Qt.Orientation.Vertical)
        self.splitter.addWidget(self.analysis_section)
        self.splitter.addWidget(tabs_container)
        self.splitter.setCollapsible(0, False)
        self.splitter.setCollapsible(1, False)
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)

        saved_sizes = repo.get_global_setting("analysis_splitter_sizes")
        if isinstance(saved_sizes, list) and len(saved_sizes) == 2:
            self.splitter.setSizes([int(s) for s in saved_sizes])
        else:
            self.splitter.setSizes(default_analysis_split(QApplication.primaryScreen()))
        self.splitter.splitterMoved.connect(lambda *_: repo.save_global_setting("analysis_splitter_sizes", self.splitter.sizes()))

        # Collapsing the Analysis section hands its splitter space back to the tabs below and pins
        # the header at the top, instead of leaving a large empty pane.
        self._analysis_expanded_size = self.splitter.sizes()[0]
        self.analysis_section.expanded_changed.connect(self._resize_splitter_for_analysis)
        if not analysis_expanded:
            self._resize_splitter_for_analysis(False)

        page_layout.addWidget(self.splitter, 1)

        saved_tab = repo.get_global_setting("right_panel_tab", 0)
        self._switch_tab(saved_tab if isinstance(saved_tab, int) and 0 <= saved_tab < len(self._tab_buttons) else 0)

        return page

    def _build_roll_page(self) -> QWidget:
        """Facts the whole roll shares, not one frame's own edit: what film it is (Film
        Mode), how its files become frames (Frame Assembly), what the rig does to it
        (Calibration) and the frame's shape (Crop), its shared exposure baseline (Roll
        Analysis) and each frame's own (Metering), and how it decodes and the scanning
        optics (Raw Decode, Optics). Film Mode leads, since it decides which of the others
        even apply."""
        cp = self.controls_panel
        page = QWidget()
        page_layout = QVBoxLayout(page)
        page_layout.setContentsMargins(0, 0, 0, 0)
        page_layout.setSpacing(THEME.space_lg)

        self.roll_tab_header = TabHeader("Roll")
        self.roll_tab_header.bind(
            (
                cp.film_section,
                cp.sensor_section,
                cp.autocrop_section,
                cp.baseline_section,
                cp.process_section,
                cp.demosaic_section,
                cp.optics_section,
            )
        )
        self.roll_tab_header.apply_requested.connect(self._apply_roll_tab)
        self.roll_tab_header.roll_revert_requested.connect(lambda: cp.revert_cards_to_roll(tuple(key for key, _ in cp._roll_sections())))
        self.controls_panel.modified_synced.connect(self.roll_tab_header.refresh)
        page_layout.addWidget(self.roll_tab_header)

        page_layout.addWidget(cp.roll_override_summary)
        for section in (
            cp.film_section,
            cp.assembly_section,
            cp.sensor_section,
            cp.autocrop_section,
            cp.baseline_section,
            cp.process_section,
            cp.demosaic_section,
            cp.optics_section,
        ):
            page_layout.addWidget(section)
        page_layout.addStretch(1)
        return page

    def active_tab_header(self):
        """The header bar of whatever tab is in front, or None on a tab that owns no
        settings (Gear, Export, Scan, Favorites, History)."""
        group = self._group_keys[self._active_group]
        if group == "roll":
            return self.roll_tab_header
        if group == "metadata":
            return self.metadata_sidebar.tab_header
        if group == "frame":
            return self._frame_tab_headers.get(self._tab_keys[self._active_index])
        return None

    def _apply_roll_tab(self) -> None:
        """Every Roll card's settings in one picker. The cards themselves stay roll
        defaults; this carries their current values onto other frames, the same as the
        Film Strip's own apply with a narrower list."""
        fields = tuple(f for card in _ROLL_TAB_CARDS for f in card_fields(card))
        open_apply_dialog(self, self.controller.session, rows=rows_for_fields(fields))

    def _build_scan_page(self) -> QWidget:
        """The 'Scan' tab hosts two collapsible sections (like Frame's Look tab): the
        SANE flatbed/film scanner on top, the RGB-Scan trichromatic capture below."""
        repo = self.controller.session.repo
        self.scan_sane_section = make_section(repo, "Film Scanner", "scan_sane", self.scan_sidebar, "fa5s.camera-retro", False)
        self.scan_rgb_section = make_section(repo, "Camera Scanning", "scan_rgb", self.scanlight_sidebar, "fa5s.camera", True)

        page = QWidget()
        page_layout = QVBoxLayout(page)
        page_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        page_layout.setContentsMargins(0, 0, 0, 0)
        page_layout.setSpacing(THEME.space_lg)
        page_layout.addWidget(self.scan_sane_section)
        page_layout.addWidget(self.scan_rgb_section)
        return page

    def show_analysis_help(self) -> None:
        from negpy.desktop.view.widgets.section_help_dialog import SectionHelpDialog

        SectionHelpDialog("analysis", "Analysis", self).exec()

    def _resize_splitter_for_analysis(self, expanded: bool) -> None:
        """Pin the collapsed Analysis header at the top: shrink pane 0 to the header and
        give the rest to the tabs; restore the prior size when re-expanded."""
        sizes = self.splitter.sizes()
        total = sum(sizes)
        if total <= 0:
            return
        if expanded:
            top = min(max(self._analysis_expanded_size, 120), max(120, total - 120))
        else:
            self._analysis_expanded_size = sizes[0]
            top = max(1, self.analysis_section.sizeHint().height())
        self.splitter.setSizes([top, max(0, total - top)])

    def apply_shortcut_tooltips(self) -> None:
        """Append the current keyboard shortcut (action id `tab_<key>`) to each tab tooltip,
        and pass the call on to the panel that owns bound controls of its own."""
        for btn, key, base in zip(self._tab_buttons, self._tab_keys, self._tab_tooltips):
            btn.setToolTip(tooltip_with_shortcut(base, f"tab_{key}"))
        for btn, key, base in zip(self._group_buttons, self._group_keys, self._group_tooltips):
            btn.setToolTip(tooltip_with_shortcut(base, f"tab_{key}"))
        self.metadata_sidebar.apply_shortcut_tooltips()
        self.gear_panel.apply_shortcut_tooltips()

    def _connect_signals(self) -> None:
        self.controller.image_updated.connect(self._update_analysis)
        self.controller.pixel_readout_rgb.connect(self.curve_widget.set_marker)
        self.controller.densitometer_readout.connect(self._on_densitometer)
        self.controller.zone_pins_changed.connect(self._refresh_zone_placement)
        self.zone_strip.zone_clicked.connect(lambda zone: self.controller.arm_zone_target(float(zone)))
        self.controller.zone_arm_changed.connect(lambda zone: self.zone_strip.set_armed(None if zone is None else int(zone)))
        self.zone_placement.target_changed.connect(self.controller.set_zone_pin_target)
        self.zone_placement.apply_clicked.connect(self.controller.apply_zone_placement)
        self.zone_placement.remove_clicked.connect(self.controller.remove_zone_pin)
        self.controller.tone_drag_changed.connect(self.curve_widget.set_active_param)
        self.controls_panel.modified_synced.connect(self._sync_tab_edited)

        # The sync_ui calls scan gear/template files; never per drag tick.
        self._sync_debounce = QTimer()
        self._sync_debounce.setSingleShot(True)
        self._sync_debounce.setInterval(150)
        self._sync_debounce.timeout.connect(self.export_sidebar.sync_ui)
        self._sync_debounce.timeout.connect(self.metadata_sidebar.sync_ui)
        self.controller.config_updated.connect(self._sync_debounce.start)

    def _sync_tab_edited(self) -> None:
        """Mark control-group tabs whose sections have edits (corner dot, like edited sliders)."""
        for i, attrs in self._tab_sections.items():
            self._tab_edited[i] = any(getattr(getattr(self.controls_panel, a), "modified_count", 0) for a in attrs)
        self._refresh_tab_dots()

    def _refresh_tab_dots(self) -> None:
        for i, btn in enumerate(self._tab_buttons):
            btn.edited_dot.set_active(self._tab_edited[i])

    def _switch_tab(self, index: int) -> None:
        self._active_index = index
        self.controller.session.repo.save_global_setting("right_panel_tab", index)
        self.stack.setCurrentIndex(index)
        for i, btn in enumerate(self._tab_buttons):
            btn.setChecked(i == index)
        self.switcher.set_pinned(index)

        # The heal and scratch tools live on the tab hosting the Retouch section. Navigating to
        # another tab suspends the active one, so clicks on the canvas do not keep placing heals
        # with their controls out of sight. Returning to the tab restores the suspended tool and
        # its overlay, unless nothing was active when the user left or another tool has been
        # picked up meanwhile.
        from negpy.desktop.session import ToolMode

        state = self.controller.session.state
        retouch_tab = self._section_tab_index.get("retouch_section")
        if index != retouch_tab:
            if state.active_tool in (ToolMode.DUST_PICK, ToolMode.SCRATCH_PICK):
                self._suspended_retouch_tool = state.active_tool
                self.controller.cancel_active_tool()
        else:
            if self._suspended_retouch_tool is not None and state.active_tool == ToolMode.NONE:
                self.controller.set_active_tool(self._suspended_retouch_tool)
            self._suspended_retouch_tool = None

    def _switch_group(self, index: int) -> None:
        self._active_group = index
        self.controller.session.repo.save_global_setting("right_panel_group", index)
        self.group_stack.setCurrentIndex(index)
        for i, btn in enumerate(self._group_buttons):
            btn.setChecked(i == index)
        self.group_switcher.set_pinned(index)

        # Trigger device detection and a gating refresh when the Scan tab is selected. It hosts
        # both the SANE scanner and the RGB-Scan capture as collapsible sections.
        if index == self._scan_group_index:
            if hasattr(self.scan_sidebar, "on_activated"):
                self.scan_sidebar.on_activated()
            if hasattr(self.scanlight_sidebar, "on_activated"):
                self.scanlight_sidebar.on_activated()

    def reveal_section(self, section_attr: str) -> None:
        """Switch to the tab containing the given ControlsPanel section."""
        if section_attr in _ROLL_SECTION_ATTRS:
            self._switch_group(self._group_keys.index("roll"))
            return
        idx = self._section_tab_index.get(section_attr)
        if idx is not None:
            self._switch_group(self._group_keys.index("frame"))
            self._switch_tab(idx)

    def show_tab_by_key(self, key: str) -> None:
        if key in self._group_keys:
            self._switch_group(self._group_keys.index(key))
            return
        if key in self._tab_keys:
            self._switch_group(self._group_keys.index("frame"))
            self._switch_tab(self._tab_keys.index(key))

    def show_gear_section_by_key(self, key: str) -> None:
        self._switch_group(self._group_keys.index("gear"))
        self.gear_panel.show_section_by_key(key)

    def _pages_holding(self, widget: QWidget) -> list[tuple[QStackedWidget, int]]:
        found = []
        for stack in (self.group_stack, self.stack):
            for i in range(stack.count()):
                if stack.widget(i).isAncestorOf(widget):
                    found.append((stack, i))
        return found

    def tab_path(self, widget: QWidget) -> list[str]:
        return [
            self._group_tooltips[i] if stack is self.group_stack else self._tab_buttons[i].text()
            for stack, i in self._pages_holding(widget)
        ]

    def reveal_widget(self, widget: QWidget) -> None:
        for stack, i in self._pages_holding(widget):
            (self._switch_group if stack is self.group_stack else self._switch_tab)(i)
        parent = widget
        while parent is not None:
            if isinstance(parent, CollapsibleSection):
                parent.expand()
            parent = parent.parentWidget()
        # A card just opened has no geometry until the next layout pass.
        QTimer.singleShot(0, lambda: self.scroll_to(widget, centered=True))

    def scroll_to(self, widget: QWidget, centered: bool = False) -> None:
        """Ensure *widget* is visible within its enclosing scroll area; centered puts it a third
        of the way down, which ensureWidgetVisible never does for a widget already in view."""
        parent = widget.parent()
        while parent is not None:
            if isinstance(parent, QScrollArea):
                if centered and parent.widget() is not None:
                    bar = parent.verticalScrollBar()
                    y = widget.mapTo(parent.widget(), QPoint(0, 0)).y()
                    bar.setValue(max(0, min(bar.maximum(), y - parent.viewport().height() // 3)))
                else:
                    parent.ensureWidgetVisible(widget)
                return
            parent = parent.parent()

    def _on_densitometer(self, reading: Any) -> None:
        self.probe_row.set_reading(reading)
        self.curve_widget.set_tracking_point(None if reading is None else reading.val_luma)

    def _update_histograms(self, metrics: Dict[str, Any]) -> None:
        """Feed the merged chart's two distributions, the zone strip and the clip stats."""
        from negpy.features.exposure.analysis import (
            output_clip_fractions,
            output_histogram,
            zone_occupancy,
            zone_warnings,
        )

        self.curve_widget.set_density_histogram(metrics.get("histogram_density"))

        # Peek Negative applied no curve, so the print histogram, curve and zone strip
        # would describe a print that was never made. Density is unaffected: it reads
        # the scan itself, before the curve, and splits into channels since there is no
        # print histogram here to carry color information.
        if self.controller.state.negative_peek:
            self.curve_widget.set_output_histogram(None)
            self.curve_widget.set_show_print(False)
            self.curve_widget.set_channel_density(True)
            self._clip_fracs = (None, None)
            self.zone_strip.setVisible(False)
            return
        self.curve_widget.set_show_print(True)
        self.curve_widget.set_channel_density(False)

        source = metrics.get("histogram_raw")
        if source is None:
            source = metrics.get("analysis_buffer")
        if source is None:
            # A GPU texture cannot be binned here without a full readback on the UI thread, so keep
            # the last histogram rather than blanking the chart.
            candidate = metrics.get("base_positive")
            source = candidate if isinstance(candidate, np.ndarray) else None
        if source is None:
            return
        bins = output_histogram(source)

        self.curve_widget.set_output_histogram(bins)
        self._clip_fracs = output_clip_fractions(bins) if bins is not None else (None, None)

        # A flat log master has no print zones, so hide them rather than mislead.
        if bins is None or self.controller.state.flat_peek:
            self.zone_strip.setVisible(False)
        else:
            occ = zone_occupancy(bins[3])
            self.zone_strip.setVisible(True)
            self.zone_strip.update_data(occ, zone_warnings(occ))

    def _refresh_zone_placement(self) -> None:
        # readouts() first: it is what refreshes the cached solution the caption reads.
        self.zone_placement.refresh(self.controller.zone_pin_readouts(), self.controller.zone_solve_caption())

    def _update_analysis(self) -> None:
        metrics = self.controller.session.state.last_metrics
        # Mid-gesture frames carry no metrics; the settle frame refreshes all of this.
        if metrics.get("interactive"):
            return

        # The only histogram refresh: a peek paint emits image_updated without
        # metrics_available, so this path must cover it.
        self._update_histograms(metrics)
        # Measured zones read through the current config, so they track every render.
        self._refresh_zone_placement()

        from negpy.features.exposure.logic import auto_highlight_from_metrics, curve_params_from_metrics

        config = self.controller.session.state.config.exposure
        process_mode = self.controller.session.state.config.process.process_mode

        # While peeking the flat master, plot the flat curve so the chart matches what the canvas
        # is showing.
        if self.controller.state.flat_peek:
            from negpy.domain.models import flat_master_config
            from negpy.features.exposure.logic import flat_curve_params

            flat_cfg = flat_master_config(self.controller.session.state.config).exposure
            gain, lift = flat_curve_params()
            self.curve_widget.update_curve(flat_cfg, slope=gain, pivot=lift, flat=True)
            # A flat master has no print curve, so there is no wedge to print through it.
            self.step_wedge.setVisible(False)
        elif self.controller.state.negative_peek:
            # No curve ran, so there is nothing to plot or to print the wedge through;
            # the chart itself already suppressed the curve in _update_histograms.
            self.step_wedge.setVisible(False)
        else:
            # Mirror PhotometricProcessor, so the plotted curve matches the render under the Auto
            # Grade, Auto Density and Cast Removal toggles.
            slopes, pivots, curvatures = curve_params_from_metrics(config, process_mode, metrics)
            # Green channel is the base curve (white reference + stats slope).
            slope, pivot = slopes[1], pivots[1]
            highlight_density = config.highlight_density + auto_highlight_from_metrics(config, process_mode, metrics)
            self.curve_widget.update_curve(
                config,
                slope=slope,
                pivot=pivot,
                slopes=slopes,
                pivots=pivots,
                curvatures=curvatures,
                highlight_density=highlight_density,
                process_mode=process_mode,
                mask_centre=metrics.get("contrast_mask_centre"),
            )
            self._update_step_wedge(config, process_mode, slope, pivot, metrics)

        from negpy.features.exposure.stats import negative_statistics

        clip_low, clip_high = self._clip_fracs
        self.stats_widget.update_stats(
            negative_statistics(
                metrics.get("norm_density_range"),
                metrics.get("metered_anchor"),
                clip_low,
                clip_high,
                scan_clip=metrics.get("scan_clip_fractions"),
                repair=metrics.get("repair_fractions"),
                gamut=self._gamut_fraction(metrics),
            )
        )

    def _gamut_fraction(self, metrics: Dict[str, Any]) -> Any:
        """Share of the frame the proofed output profile cannot print, or None when
        nothing is being proofed to. The engine measures the frame's colors and knows
        nothing about profiles; the gamut mask is built here, where the ICC lives."""
        from negpy.features.exposure.analysis import gamut_fraction
        from negpy.infrastructure.display.color_mgmt import get_gamut_lut

        _display_cs, _monitor, proof = self.controller.display_transform_params()
        lut = get_gamut_lut(self.controller.state.workspace_color_space, proof)
        return gamut_fraction(metrics.get("histogram_color"), lut)

    def _update_step_wedge(self, config: Any, process_mode: Any, slope: float, pivot: float, metrics: Dict[str, Any]) -> None:
        """21 known log exposures through the same curve the chart just plotted, so the wedge
        and the chart can never describe different prints."""
        from negpy.features.exposure.analysis import wedge_step_density, wedge_vals
        from negpy.features.exposure.logic import print_curve, print_curve_output

        enc = print_curve_output(print_curve(config, slope, pivot, process_mode), wedge_vals())
        display_cs, monitor_bytes, proof = self.controller.display_transform_params()
        self.step_wedge.setVisible(True)
        self.step_wedge.update_data(enc, wedge_step_density(metrics.get("norm_density_range")), display_cs, monitor_bytes, proof)
