from typing import Any, Dict

import numpy as np
from PyQt6.QtCore import QPoint, Qt, QTimer
from PyQt6.QtGui import QFont, QFontMetrics
from PyQt6.QtWidgets import (
    QApplication,
    QHBoxLayout,
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
from negpy.desktop.view.styles.color_vision import palette_for
from negpy.desktop.view.styles.fonts import ui_font_family
from negpy.desktop.view.styles.templates import EditedDot
from negpy.desktop.view.styles.theme import THEME
from negpy.desktop.view.widgets.choice_button import SegmentedChoice
from negpy.desktop.view.widgets.charts import PhotometricCurveWidget, StepWedgeWidget, VectorscopeWidget, WaveformWidget, ZoneStripWidget
from negpy.desktop.view.widgets.collapsible import CollapsibleSection, make_section
from negpy.desktop.view.widgets.gear_library_panel import GearLibraryPanel
from negpy.desktop.view.widgets.granular_settings_dialog import open_apply_dialog
from negpy.desktop.view.widgets.tab_header import TabHeader
from negpy.desktop.settings_catalog import rows_for_fields
from negpy.features.process.models import ProcessMode
from negpy.services.assets.rolls import card_fields
from negpy.desktop.view.widgets.stats import DensitometerRow, NegativeStatsWidget, ZonePlacementRows
from negpy.desktop.view.widgets.overflow_bar import OverflowBar

# The Roll tab's cards that own settings, for its header's Apply picker.
_ROLL_TAB_CARDS = (
    "film",
    "sensor",
    "cast_removal",
    "autocrop",
    "auto_crop",
    "baseline",
    "process",
    "demosaic",
    "lens",
    "flatfield",
)

# ControlsPanel sections built into the Roll tab (_build_roll_page): reveal_section routes them to the Roll group.
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
    Export / Scan, in the order an edit moves through them. Analysis is pinned above
    Roll and Frame. Metadata and Gear pin a section above their own scroll area
    (Preview, the Items/Presets switcher); Export and Scan are plain pages."""

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
        from negpy.desktop.view.sidebar.scan_output import ScanOutputPanel
        from negpy.desktop.view.sidebar.scanlight import ScanlightSidebar

        self.scan_output = ScanOutputPanel(self.controller.session.repo)
        self.scan_sidebar = ScanSidebar(self.controller, self.scan_output)
        self.scanlight_sidebar = ScanlightSidebar(self.controller, self.scan_output)
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

            # These pages scroll inside, beside a pinned part; every other page scrolls whole.
            page = content if key in ("frame", "metadata", "gear", "scan") else wrap_scroll(content)
            self.group_stack.addWidget(page)
            self._group_buttons.append(btn)
            self._group_keys.append(key)
            self._group_tooltips.append(tooltip)
            if key == "scan":
                self._scan_group_index = i

        # Vertical splitter lets the user resize Analysis vs. the tabs below
        self.splitter = QSplitter(Qt.Orientation.Vertical)
        self.splitter.addWidget(self.analysis_section)
        self.splitter.addWidget(self.group_stack)
        self.splitter.setCollapsible(0, False)
        self.splitter.setCollapsible(1, False)
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)

        repo = self.controller.session.repo
        saved_sizes = repo.get_global_setting("analysis_splitter_sizes")
        if isinstance(saved_sizes, list) and len(saved_sizes) == 2:
            sizes = [int(s) for s in saved_sizes]
        else:
            sizes = default_analysis_split(QApplication.primaryScreen())
        self.splitter.setSizes(sizes)
        self.splitter.splitterMoved.connect(self._on_analysis_splitter_moved)

        # Collapsing the Analysis section hands its splitter space back to the tabs below and pins
        # the header at the top, instead of leaving a large empty pane.
        self._analysis_expanded_size = sizes[0]
        self._analysis_split_shown = False
        self.analysis_section.expanded_changed.connect(self._resize_splitter_for_analysis)

        layout.addWidget(self.group_switcher)
        layout.addWidget(self.splitter, 1)

        self.apply_shortcut_tooltips()

        saved_group = repo.get_global_setting("right_panel_group", 0)
        self._switch_group(saved_group if isinstance(saved_group, int) and 0 <= saved_group < len(self._group_buttons) else 0)

    def _build_frame_page(self, wrap_scroll) -> QWidget:
        """The Analysis section, and a second, inner tab switcher for the per-image
        workflow control groups, Favorites and History -- everything that changes what
        the canvas shows for the one loaded frame."""
        page = QWidget()
        page_layout = QVBoxLayout(page)
        page_layout.setContentsMargins(0, 0, 0, 0)
        page_layout.setSpacing(0)

        analysis_content = QWidget()
        analysis_layout = QVBoxLayout(analysis_content)
        analysis_layout.setContentsMargins(0, THEME.space_sm, 0, THEME.space_sm)
        analysis_layout.setSpacing(THEME.space_sm)

        self.curve_widget = PhotometricCurveWidget()
        self.waveform_widget = WaveformWidget()
        self.vectorscope_widget = VectorscopeWidget()
        self.step_wedge = StepWedgeWidget()
        self.zone_strip = ZoneStripWidget()
        self.probe_row = DensitometerRow()
        self.zone_placement = ZonePlacementRows()
        self.stats_widget = NegativeStatsWidget()
        self._clip_fracs: tuple = (None, None)

        repo = self.controller.session.repo
        self.scale_btn = SegmentedChoice(
            (("", "Lin"), ("", "Log")),
            "Histogram and waveform count scale: Log lifts small counts so thin tails and clipped ends "
            "read. On the Vectorscope it is the chroma scale, kept apart for B&W and color frames: Log "
            "opens up the near-neutral center, where a toner lives.",
        )
        self.scale_btn.setCurrentIndex(1 if repo.get_global_setting("histogram_log_scale") else 0)
        self.curve_widget.set_log_scale(self.scale_btn.currentIndex() == 1)
        self.waveform_widget.set_log_scale(self.scale_btn.currentIndex() == 1)
        self.scale_btn.currentChanged.connect(self._on_histogram_scale)
        self._sync_chart_colors()
        self.controller.session.color_vision_changed.connect(self._sync_chart_colors)

        self.chart_stack = QStackedWidget()
        self.chart_stack.addWidget(self.curve_widget)
        self.chart_stack.addWidget(self.waveform_widget)
        self.chart_stack.addWidget(self.vectorscope_widget)
        self.view_btn = SegmentedChoice(
            (("", "Curve"), ("", "Waveform"), ("", "Vectorscope")),
            "Analysis chart",
            data=("curve", "waveform", "vectorscope"),
        )
        self.view_btn.set_choice_tooltip(0, "The paper curve over the scan's and the print's histograms")
        self.view_btn.set_choice_tooltip(1, "The print's tones across the frame, column by column, on the zone scale")
        self.view_btn.set_choice_tooltip(2, "The print's colors on the a*b* plane: a cast, a toner, a split tone")
        self.view_btn.currentChanged.connect(self._on_analysis_view)
        saved_view = self.view_btn.findData(repo.get_global_setting("analysis_view", "curve"))
        self.view_btn.setCurrentIndex(max(0, saved_view))
        self.chart_stack.setCurrentIndex(self.view_btn.currentIndex())
        self._sync_scale_btn()

        chart_row = QHBoxLayout()
        chart_row.setContentsMargins(0, 0, 0, 0)
        chart_row.addWidget(self.view_btn)
        chart_row.addStretch(1)
        chart_row.addWidget(self.scale_btn)

        analysis_layout.addWidget(self.chart_stack, 1)
        analysis_layout.addLayout(chart_row)
        analysis_layout.addWidget(self.step_wedge, 0)
        analysis_layout.addWidget(self.zone_strip, 0)
        analysis_layout.addWidget(self.probe_row, 0)
        analysis_layout.addWidget(self.zone_placement, 0)
        analysis_layout.addWidget(self.stats_widget, 0)

        repo = self.controller.session.repo
        # Roll and Frame each keep their own expanded state; _switch_group restores it.
        self.analysis_section = make_section(repo, "Analysis", "analysis", analysis_content, "fa5s.chart-bar", persist=False)
        self.analysis_section.expanded_changed.connect(lambda checked: repo.save_global_setting(self._analysis_expanded_setting(), checked))
        self.clipping_btn = self.analysis_section.add_header_toggle("fa5s.adjust", "Clipping Overlay")
        self.clipping_btn.setChecked(self.controller.state.clipping_overlay)
        self.clipping_btn.toggled.connect(self.controller.toggle_clipping_overlay)
        self.zone_highlight_btn = self.analysis_section.add_header_toggle("fa5s.highlighter", "Zone Highlight")
        self.zone_highlight_btn.setChecked(self.controller.state.zone_hover_highlight)
        self.zone_highlight_btn.toggled.connect(self.controller.toggle_zone_hover_highlight)
        self._refresh_zone_strip_hint()

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

        page_layout.addWidget(self.switcher)
        page_layout.addWidget(self.stack, 1)

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
        """Scanner choice and cards in a scroll area; the active scanner's footer stays pinned below."""
        repo = self.controller.session.repo
        scan, cam = self.scan_sidebar, self.scanlight_sidebar

        self.scan_source_btn = SegmentedChoice(
            (("fa5s.camera-retro", "Film Scanner"), ("fa5s.camera", "Camera")),
            "Scan with a film scanner or a camera on a copy stand",
            data=("film", "camera"),
        )
        self.scan_source_btn.setCurrentIndex(max(self.scan_source_btn.findData(repo.get_global_setting("scan_source", "camera")), 0))
        self.scan_source_section = make_section(
            repo, "Scanner", "scan_source", self.scan_source_btn, "fa5s.exchange-alt", collapsible=False
        )
        self.scan_device_section = make_section(repo, "Device", "scan_device", scan.device_body, "fa5s.plug", True)
        self.scan_quality_section = make_section(repo, "Film & Quality", "scan_quality", scan.quality_body, "fa5s.sliders-h", True)
        self.scan_framing_section = make_section(repo, "Framing", "scan_framing", scan.framing_body, "fa5s.crop-alt", True)
        self.scan_camera_section = make_section(repo, "Camera", "scan_camera", cam.camera_body, "fa5s.camera", True)
        self.scan_light_section = make_section(repo, "Preset & Light", "scan_light", cam.light_body, "fa5s.lightbulb", True)
        output_body = QWidget()
        output_layout = QVBoxLayout(output_body)
        output_layout.setContentsMargins(0, 0, 0, 0)
        output_layout.setSpacing(THEME.space_md)
        output_layout.addWidget(self.scan_output)
        output_layout.addWidget(scan.output_body)
        self.scan_output_section = make_section(repo, "Output", "scan_output", output_body, "fa5s.folder-open", True)

        cards = QWidget()
        cards_layout = QVBoxLayout(cards)
        cards_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        cards_layout.setContentsMargins(0, 0, 0, 0)
        cards_layout.setSpacing(THEME.space_lg)
        for section in (
            self.scan_source_section,
            self.scan_device_section,
            self.scan_quality_section,
            self.scan_framing_section,
            self.scan_camera_section,
            self.scan_light_section,
            self.scan_output_section,
        ):
            cards_layout.addWidget(section)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(cards)

        page = QWidget()
        page_layout = QVBoxLayout(page)
        page_layout.setContentsMargins(0, 0, 0, 0)
        page_layout.setSpacing(0)
        page_layout.addWidget(scroll, 1)
        for footer in (scan.footer, cam.footer):
            # Outside any card, so it takes the inset a card would give it.
            footer.setContentsMargins(THEME.space_xl, THEME.space_md, THEME.space_xl, THEME.space_md)
            page_layout.addWidget(footer)

        self.scan_source_btn.currentChanged.connect(self._on_scan_source_changed)
        scan.cards_changed.connect(self._sync_scan_cards)
        cam.cards_changed.connect(self._sync_scan_cards)
        self._sync_scan_cards()
        return page

    def _scan_film(self) -> bool:
        return self.scan_source_btn.currentData() == "film"

    def _active_scan_sidebar(self):
        return self.scan_sidebar if self._scan_film() else self.scanlight_sidebar

    def show_scan_source(self, source: str) -> None:
        """Switch the Scan tab to "film" or "camera"."""
        self.scan_source_btn.setCurrentIndex(self.scan_source_btn.findData(source))

    def _on_scan_source_changed(self, _index: int) -> None:
        self.controller.session.repo.save_global_setting("scan_source", self.scan_source_btn.currentData())
        self._sync_scan_cards()
        if self._active_group == self._scan_group_index:
            self._active_scan_sidebar().on_activated()

    def _sync_scan_cards(self) -> None:
        film = self._scan_film()
        scan, cam = self.scan_sidebar, self.scanlight_sidebar
        for section, body, owned in (
            (self.scan_device_section, scan.device_body, film),
            (self.scan_quality_section, scan.quality_body, film),
            (self.scan_framing_section, scan.framing_body, film),
            (self.scan_camera_section, cam.camera_body, not film),
            (self.scan_light_section, cam.light_body, not film),
        ):
            section.setVisible(owned and not body.isHidden())
        scan.output_body.setVisible(film)
        scan.footer.setVisible(film)
        cam.footer.setVisible(not film)

    def show_analysis_help(self) -> None:
        from negpy.desktop.view.widgets.section_help_dialog import SectionHelpDialog

        SectionHelpDialog("analysis", "Analysis", self, repo=self.controller.session.repo).exec()

    def _analysis_expanded_setting(self) -> str:
        return "section_expanded_analysis_roll" if self._group_keys[self._active_group] == "roll" else "section_expanded_analysis"

    def showEvent(self, event) -> None:
        super().showEvent(event)
        # Before the first show the splitter has no real height, so a split set then is rescaled.
        if not self._analysis_split_shown:
            self._analysis_split_shown = True
            QTimer.singleShot(0, lambda: self._resize_splitter_for_analysis(self.analysis_section.toggle_button.isChecked()))

    def _on_analysis_splitter_moved(self, *_) -> None:
        if not self.analysis_section.toggle_button.isChecked():
            return
        sizes = self.splitter.sizes()
        self._analysis_expanded_size = sizes[0]
        self.controller.session.repo.save_global_setting("analysis_splitter_sizes", sizes)

    def _resize_splitter_for_analysis(self, expanded: bool) -> None:
        """Pin the collapsed Analysis header at the top: shrink pane 0 to the header and
        give the rest to the tabs; restore the dragged size when re-expanded."""
        total = sum(self.splitter.sizes())
        if total <= 0 or not self._analysis_split_shown:
            return
        if expanded:
            top = min(max(self._analysis_expanded_size, 120), max(120, total - 120))
        else:
            top = max(1, self.analysis_section.sizeHint().height())
        self.splitter.setSizes([top, max(0, total - top)])

    def apply_shortcut_tooltips(self) -> None:
        """Append the current keyboard shortcut (action id `tab_<key>`) to each tab tooltip,
        and pass the call on to the panel that owns bound controls of its own."""
        for btn, key, base in zip(self._tab_buttons, self._tab_keys, self._tab_tooltips):
            btn.setToolTip(tooltip_with_shortcut(base, f"tab_{key}"))
        for btn, key, base in zip(self._group_buttons, self._group_keys, self._group_tooltips):
            btn.setToolTip(tooltip_with_shortcut(base, f"tab_{key}"))
        self._refresh_zone_strip_hint()
        self.metadata_sidebar.apply_shortcut_tooltips()
        self.export_sidebar.apply_shortcut_tooltips()
        self.gear_panel.apply_shortcut_tooltips()
        self.scanlight_sidebar.lv_window.apply_shortcut_tooltips()

    def _sync_clipping_btn(self, on: bool) -> None:
        self.clipping_btn.blockSignals(True)
        self.clipping_btn.setChecked(on)
        self.clipping_btn.blockSignals(False)

    def _sync_zone_highlight_btn(self, on: bool) -> None:
        self.zone_highlight_btn.blockSignals(True)
        self.zone_highlight_btn.setChecked(on)
        self.zone_highlight_btn.blockSignals(False)

    def _on_pixel_readout(self, readout) -> None:
        """Marks the hovered pixel on the waveform (its column and tone) and the vectorscope (its a*b*)."""
        if readout is None:
            self.waveform_widget.set_marker(None)
            self.vectorscope_widget.set_marker(None)
            return
        from negpy.features.exposure.analysis import scope_point

        nx, _ny, rgb = readout
        luma, a, b = scope_point(rgb)
        self.waveform_widget.set_marker((min(max(nx, 0.0), 1.0), luma))
        self.vectorscope_widget.set_marker((a, b))

    def _on_analysis_view(self, index: int) -> None:
        self.chart_stack.setCurrentIndex(index)
        self.controller.session.repo.save_global_setting("analysis_view", self.view_btn.currentData())
        self._sync_scale_btn()

    def _scope_scale_key(self) -> str:
        bw = self.controller.state.config.process.process_mode == ProcessMode.BW
        return "vectorscope_log_bw" if bw else "vectorscope_log_color"

    def _scope_log(self) -> bool:
        # Toned B&W sits within a few units of neutral, color frames spread far past it.
        key = self._scope_scale_key()
        return bool(self.controller.session.repo.get_global_setting(key, key == "vectorscope_log_bw"))

    def _sync_scale_btn(self) -> None:
        """Lin/Log shows the visible chart's own scale: the histograms' one, or the vectorscope's for this film mode."""
        scope_log = self._scope_log()
        self.vectorscope_widget.set_log_scale(scope_log)
        if self.view_btn.currentData() == "vectorscope":
            log = scope_log
        else:
            log = bool(self.controller.session.repo.get_global_setting("histogram_log_scale"))
        self.scale_btn.blockSignals(True)
        self.scale_btn.setCurrentIndex(1 if log else 0)
        self.scale_btn.blockSignals(False)

    def set_analysis_view(self, key: str) -> None:
        self.view_btn.setCurrentIndex(max(0, self.view_btn.findData(key)))

    def _on_histogram_scale(self, index: int) -> None:
        log = index == 1
        if self.view_btn.currentData() == "vectorscope":
            self.vectorscope_widget.set_log_scale(log)
            self.controller.session.repo.save_global_setting(self._scope_scale_key(), log)
            return
        self.curve_widget.set_log_scale(log)
        self.waveform_widget.set_log_scale(log)
        self.controller.session.repo.save_global_setting("histogram_log_scale", log)

    def _sync_chart_colors(self) -> None:
        self.curve_widget.set_zone_colors(*palette_for(self.controller.state.color_vision).dodge_burn)

    def _refresh_zone_strip_hint(self) -> None:
        self.zone_strip.set_overlay_hint(tooltip_with_shortcut("Double-click: Zone Overlay on or off", "toggle_zones"))
        self.clipping_btn.setToolTip(
            tooltip_with_shortcut(
                "Clipping Overlay: mark the pixels that print as paper black or paper white, in the Color vision pair (Preferences)",
                "toggle_clipping_overlay",
            )
        )
        self.zone_highlight_btn.setToolTip(
            tooltip_with_shortcut(
                "Zone Highlight: hover a zone strip cell to show that zone alone on the frame", "toggle_zone_hover_highlight"
            )
        )

    def _connect_signals(self) -> None:
        self.controller.image_updated.connect(self._update_analysis)
        self.controller.pixel_readout_rgb.connect(self.curve_widget.set_marker)
        self.controller.pixel_readout.connect(self._on_pixel_readout)
        self.controller.densitometer_readout.connect(self._on_densitometer)
        self.controller.zone_pins_changed.connect(self._refresh_zone_placement)
        self.zone_strip.zone_clicked.connect(lambda zone: self.controller.arm_zone_target(float(zone)))
        self.controller.zone_arm_changed.connect(lambda zone: self.zone_strip.set_armed(None if zone is None else int(zone)))
        self.zone_strip.zone_double_clicked.connect(self.controller.toggle_zones_overlay)
        self.zone_strip.zone_hovered.connect(self.controller.set_zone_highlight)
        self.controller.clipping_overlay_changed.connect(self._sync_clipping_btn)
        self.controller.zone_hover_highlight_changed.connect(self._sync_zone_highlight_btn)
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
        self._sync_local_masks()

        # The heal and scratch tools live on the tab hosting the Retouch section. Navigating to
        # another tab suspends the active one, so clicks on the canvas do not keep placing heals
        # with their controls out of sight. Returning to the tab restores the suspended tool and
        # its overlay, unless nothing was active when the user left or another tool has been
        # picked up meanwhile.
        from negpy.desktop.session import ToolMode

        state = self.controller.session.state
        retouch_tab = self._section_tab_index.get("retouch_section")
        if index != retouch_tab:
            if state.active_tool in (ToolMode.DUST_PICK, ToolMode.SCRATCH_PICK, ToolMode.CLONE):
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
        analysis_shown = self._group_keys[index] in ("roll", "frame")
        self.analysis_section.setVisible(analysis_shown)
        if analysis_shown:
            saved = self.controller.session.repo.get_global_setting(self._analysis_expanded_setting())
            self.analysis_section.set_expanded(THEME.sidebar_expanded_defaults["analysis"] if saved is None else bool(saved))
        for i, btn in enumerate(self._group_buttons):
            btn.setChecked(i == index)
        self.group_switcher.set_pinned(index)
        self._sync_local_masks()

        if index == self._scan_group_index:
            self._active_scan_sidebar().on_activated()

    def _sync_local_masks(self) -> None:
        """Mask outlines show only while the Dodge & Burn tab shows."""
        # The frame page switches its first tab while it is built, before the groups exist.
        if not getattr(self, "_group_keys", None):
            return
        shown = self._group_keys[self._active_group] == "frame" and self._active_index == self._section_tab_index.get("local_section")
        state = self.controller.session.state
        if state.local_masks_shown != shown:
            state.local_masks_shown = shown
            self.controller.config_updated.emit()

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
        of the way down, or higher so a tall one fits, which ensureWidgetVisible never does for a
        widget already in view."""
        parent = widget.parent()
        while parent is not None:
            if isinstance(parent, QScrollArea):
                if centered and parent.widget() is not None:
                    bar = parent.verticalScrollBar()
                    y = widget.mapTo(parent.widget(), QPoint(0, 0)).y()
                    vh = parent.viewport().height()
                    bar.setValue(max(0, min(bar.maximum(), y - max(0, min(vh // 3, vh - widget.height())))))
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
        # A peek renders without metrics, so the last print's waveform would describe the wrong image.
        peeking = self.controller.state.negative_peek or self.controller.state.flat_peek
        self.waveform_widget.set_data(None if peeking else metrics.get("waveform"), metrics.get("waveform_color"))
        self.vectorscope_widget.set_data(None if peeking else metrics.get("vectorscope"))

        # Peek Negative applied no curve, so the print histogram, curve and zone strip
        # would describe a print that was never made. Density is unaffected: it reads
        # the scan itself, before the curve, and splits into channels since there is no
        # print histogram here to carry color information.
        if self.controller.state.negative_peek:
            self.curve_widget.set_output_histogram(None)
            self.curve_widget.set_show_print(False)
            self.curve_widget.set_channel_density(True)
            self._clip_fracs = (None, None)
            self.controls_panel.process_sidebar.set_clipping(None, None)
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
        self.controls_panel.process_sidebar.set_clipping(*self._clip_fracs)  # the Metering card's line, one computation

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
        self.zone_strip.set_placement_enabled(self.controller.zone_placement_available())
        # The film mode picks the vectorscope's scale, and it can change with the frame.
        self._sync_scale_btn()

        # The only histogram refresh: a peek paint emits image_updated without
        # metrics_available, so this path must cover it.
        self._update_histograms(metrics)
        # Measured zones read through the current config, so they track every render.
        self._refresh_zone_placement()

        from negpy.features.exposure.logic import auto_highlight_from_metrics, auto_shadow_from_metrics, curve_params_from_metrics

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
            shadow_density = config.shadow_density + auto_shadow_from_metrics(config, process_mode, metrics)
            self.curve_widget.update_curve(
                config,
                slope=slope,
                pivot=pivot,
                slopes=slopes,
                pivots=pivots,
                curvatures=curvatures,
                highlight_density=highlight_density,
                shadow_density=shadow_density,
                process_mode=process_mode,
                mask_centre=metrics.get("contrast_mask_centre"),
            )
            self.controls_panel.paper_curve.set_curves(*self.curve_widget.curves())
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
        """21 known log exposures through the same curve the chart just plotted, then the
        render's own alternative process and toning, so the wedge is the print as made."""
        from negpy.features.exposure.analysis import wedge_step_density, wedge_vals
        from negpy.features.exposure.logic import print_curve, print_curve_output
        from negpy.services.rendering.engine import print_wedge

        curve = print_curve_output(print_curve(config, slope, pivot, process_mode), wedge_vals())
        enc = print_wedge(self.controller.session.state.config, curve)
        display_cs, monitor_bytes, proof = self.controller.display_transform_params()
        self.step_wedge.setVisible(True)
        self.step_wedge.update_data(enc, wedge_step_density(metrics.get("norm_density_range")), display_cs, monitor_bytes, proof)
