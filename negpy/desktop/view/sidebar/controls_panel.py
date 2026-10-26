from negpy.kernel.system.i18n import tr
import qtawesome as qta
from PyQt6.QtWidgets import (
    QMenu,
    QWidget,
    QVBoxLayout,
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal

from negpy.desktop.controller import AppController
from negpy.desktop.view.shortcut_registry import label_with_shortcut, tooltip_with_shortcut
from negpy.desktop.view.styles.color_vision import palette_for
from negpy.desktop.view.styles.templates import gradient_icon, header_row, hint_label, section_subheader, set_hint_kind, wrap_tooltip
from negpy.desktop.view.widgets.collapsible import NO_ROLL_SCOPE_HINT, CollapsibleSection, make_section
from negpy.desktop.view.widgets.charts import MiniCurveWidget, MiniHistogramWidget, MiniRGBHistogramWidget
from negpy.desktop.view.styles.theme import THEME
from negpy.features.lab.models import LabConfig
from negpy.features.altprocess.models import AltProcessConfig
from negpy.features.leak.models import LeakConfig
from negpy.features.toning.models import ToningConfig
from negpy.features.geometry.logic import has_manual_crop
from negpy.features.process.models import auto_meter_for_mode, cast_removal_for_mode
from negpy.features.finish.models import FinishConfig
from negpy.features.flatfield.models import FlatFieldConfig
from negpy.kernel.system.config import DEFAULT_WORKSPACE_CONFIG
from negpy.services.assets.rolls import ROLL_DEFAULT_FIELDS
from negpy.desktop.settings_catalog import (
    COLOR_FIELDS,
    CROP_FIELDS,
    FRAME_CARD_FIELDS,
    GEOMETRY_FIELDS,
    PAPER_FIELDS,
    TONE_FIELDS,
    frame_card_rows,
)
from negpy.desktop.view.widgets.granular_settings_dialog import open_apply_dialog
from negpy.desktop.view.widgets.tab_header import TabHeader

# Sidebar Components
from negpy.desktop.view.sidebar.presets import PresetsSidebar
from negpy.desktop.view.sidebar.flatfield import FlatFieldSidebar
from negpy.desktop.view.sidebar.process import ProcessSidebar
from negpy.desktop.view.sidebar.roll import RollAnalysisSidebar
from negpy.desktop.view.sidebar.demosaic import DemosaicSidebar
from negpy.desktop.view.sidebar.scan_setup import ScanSetupSidebar, resolved_rig
from negpy.desktop.view.sidebar.sensor import SensorSidebar
from negpy.desktop.view.sidebar.color import ColorSidebar
from negpy.desktop.view.sidebar.tone import ToneSidebar
from negpy.desktop.view.sidebar.paper import PaperSidebar
from negpy.desktop.view.sidebar.geometry import GeometrySidebar
from negpy.desktop.view.sidebar.autocrop import AutocropSidebar
from negpy.desktop.view.sidebar.trichrome import TrichromeSidebar
from negpy.desktop.view.sidebar.half_frame import HalfFrameSidebar
from negpy.desktop.view.sidebar.lens import LensSidebar
from negpy.desktop.view.sidebar.lab import LabSidebar
from negpy.desktop.view.sidebar.altprocess import AltProcessSidebar
from negpy.desktop.view.sidebar.toning import ToningSidebar
from negpy.desktop.view.sidebar.leak import LeakSidebar
from negpy.desktop.view.sidebar.retouch import RetouchSidebar
from negpy.desktop.view.sidebar.local import LocalSidebar
from negpy.desktop.view.sidebar.finish import FinishSidebar

_DEMOSAIC_FIELDS = ("highlight_reconstruction",)
# GeometryConfig is split across three cards: Crop and Geometry are this frame's own,
# and how the scanning lens bends is the roll's (Optics).
_LENS_FIELDS = (
    "distortion_k1",
    "lens_distortion_from_metadata",
    "lens_ca_from_metadata",
)
_SENSOR_FIELDS = (
    "sensor_profile",
    "sensor_matrix",
    "sensor_unmix",
    "crosstalk_profile",
    "crosstalk_strength",
    "crosstalk_matrix",
    "crosstalk_process",
    "hue_trim",
)
# ProcessConfig is split across five cards. Each tuple is both the card's reset scope and
# its modified count, so a field is resettable from the one card that counts it.
# Linear RAW and the two demosaic choices are in none: they come from the scanning
# setup, which no card or frame reset changes (_with_scan_setup).
# locked_floors/locked_ceils/locked_neutral_axis are in none: they are Roll Analysis's measured result.
_FILM_FIELDS = (
    "process_mode",
    "positive_source",
)
_METERING_FIELDS = (
    "analysis_buffer",
    "analysis_rect",
    "lock_bounds",
    "luma_range_clip",
    "color_range_clip",
    "white_point_offset",
    "black_point_offset",
    "white_point_trim_red",
    "white_point_trim_green",
    "white_point_trim_blue",
    "black_point_trim_red",
    "black_point_trim_green",
    "black_point_trim_blue",
)
_BASELINE_FIELDS = (
    "use_luma_average",
    "use_color_average",
    "use_cast_average",
)

# Constant frozen-dataclass defaults, built once rather than per resync. Exposure/process/
# geometry/config come from DEFAULT_WORKSPACE_CONFIG, not their own bare dataclass default:
# the autocrop fields hold the shipped values only there, as an untouched or reset file does.
_DEFAULT_EXPOSURE = DEFAULT_WORKSPACE_CONFIG.exposure
_DEFAULT_LAB = LabConfig()
_DEFAULT_TONING = ToningConfig()
_DEFAULT_ALTPROC = AltProcessConfig()
_DEFAULT_LEAK = LeakConfig()
_DEFAULT_GEOMETRY = DEFAULT_WORKSPACE_CONFIG.geometry
_DEFAULT_PROCESS = DEFAULT_WORKSPACE_CONFIG.process
_DEFAULT_FINISH = FinishConfig()
_DEFAULT_FLATFIELD = FlatFieldConfig()
_DEFAULT_CONFIG = DEFAULT_WORKSPACE_CONFIG

_AUTO_METER_FIELDS = ("auto_exposure", "auto_normalize_contrast")

# Roll-tab sections that drive more than one roll card, keyed by section key.
_SECTION_CARDS: dict[str, tuple[str, ...]] = {
    "optics": ("lens", "flatfield"),
}


def _default_exposure_field(field: str, process_mode: str):
    """The value *field* defaults to on this frame. Auto Density/Auto Grade
    (auto_meter_for_mode) and Cast Removal (cast_removal_for_mode) default per mode;
    every other ExposureConfig field has one flat default."""
    default = getattr(_DEFAULT_EXPOSURE, field)
    if field in _AUTO_METER_FIELDS:
        return auto_meter_for_mode(process_mode, default)
    if field == "cast_removal_strength":
        return cast_removal_for_mode(process_mode, default)
    return default


class ControlsPanel(QWidget):
    """
    Right sidebar panel aggregating all tool controls (Exposure, Geometry, etc.).
    """

    modified_synced = pyqtSignal()

    def __init__(self, controller: AppController):
        super().__init__()
        self.controller = controller
        self._last_histogram_buf = None
        self._read_only = False

        self._init_ui()
        self._connect_signals()

    def _init_ui(self) -> None:
        self.presets_sidebar = PresetsSidebar(self.controller)
        self.presets_section = self._make_section(
            tr("Presets"),
            "presets",
            self.presets_sidebar,
            icon_name="fa5s.magic",
        )

        self.geometry_sidebar = GeometrySidebar(self.controller)
        self.geometry_section = self._make_section(
            tr("Straighten"),
            "geometry",
            self.geometry_sidebar,
            icon_name="fa5s.ruler",
        )

        self.autocrop_sidebar = AutocropSidebar(self.controller)
        # The persisted "autocrop" section key stays.
        self.autocrop_section = self._make_section(
            tr("Crop"),
            "autocrop",
            self.autocrop_sidebar,
            icon_name="fa5s.crop-alt",
        )

        # The scanning optics: lens warp and light falloff. One card, still two roll cards
        # (_SECTION_CARDS), since each lives on its own config section.
        self.lens_sidebar = LensSidebar(self.controller)
        self.flatfield_sidebar = FlatFieldSidebar(self.controller)
        optics_body = QWidget()
        optics_layout = QVBoxLayout(optics_body)
        optics_layout.setContentsMargins(0, 0, 0, 0)
        optics_layout.setSpacing(THEME.space_sm)
        optics_layout.addWidget(section_subheader(tr("LENS CORRECTION")))
        optics_layout.addWidget(self.lens_sidebar)
        optics_layout.addLayout(
            header_row(section_subheader(tr("FLAT FIELD CORRECTION")), self.flatfield_sidebar.add_btn, self.flatfield_sidebar.delete_btn)
        )
        optics_layout.addWidget(self.flatfield_sidebar)
        self.optics_section = self._make_section(
            tr("Optics"),
            "optics",
            optics_body,
            icon_name="fa5s.circle-notch",
        )

        self.scan_setup_sidebar = ScanSetupSidebar(self.controller)
        self.process_sidebar = ProcessSidebar(self.controller)
        # Always expanded (no chevron): the first choices of every edit, and the ones
        # every other Roll-tab card's fields assume are already settled. The roll shares
        # the film mode; the scanning setup is one setting for the app.
        setup_body = QWidget()
        setup_layout = QVBoxLayout(setup_body)
        setup_layout.setContentsMargins(0, 0, 0, 0)
        setup_layout.setSpacing(THEME.space_lg)
        setup_layout.addWidget(self.process_sidebar.mode_bar)
        setup_layout.addWidget(self.scan_setup_sidebar)
        self.film_section = self._make_section(
            tr("Setup"),
            "film",
            setup_body,
            icon_name="mdi6.film",
            collapsible=False,
        )
        # Each roll keeps its own Trichrome and Half Frame state, so neither has a scope pair.
        self.trichrome_sidebar = TrichromeSidebar(self.controller)
        self.half_frame_sidebar = HalfFrameSidebar(self.controller)
        assembly_body = QWidget()
        assembly_layout = QVBoxLayout(assembly_body)
        assembly_layout.setContentsMargins(0, 0, 0, 0)
        assembly_layout.setSpacing(THEME.space_lg)
        assembly_layout.addWidget(self.trichrome_sidebar)
        assembly_layout.addWidget(self.half_frame_sidebar)
        self.assembly_section = self._make_section(
            tr("Frame Assembly"),
            "assembly",
            assembly_body,
            icon_name="mdi.view-split-vertical",
        )

        # Where this frame's bounds come from: the roll and scene baselines, then the Use
        # Luma/Color Average switches that read them.
        self.roll_sidebar = RollAnalysisSidebar(self.controller)
        baseline_body = QWidget()
        baseline_layout = QVBoxLayout(baseline_body)
        baseline_layout.setContentsMargins(0, 0, 0, 0)
        baseline_layout.setSpacing(THEME.space_sm)
        baseline_layout.addWidget(self.roll_sidebar)
        baseline_layout.addWidget(self.process_sidebar.baseline_bar)
        self.baseline_section = self._make_section(
            tr("Roll Analysis"),
            "baseline",
            baseline_body,
            icon_name="mdi6.filmstrip",
        )

        # How this frame measures its own bounds, and what nudges the result. The persisted
        # "process" section key stays.
        metering_body = QWidget()
        metering_layout = QVBoxLayout(metering_body)
        metering_layout.setContentsMargins(0, 0, 0, 0)
        metering_layout.setSpacing(THEME.space_sm)
        metering_layout.addWidget(self.process_sidebar)
        metering_layout.addWidget(self.process_sidebar.analysis_bar)
        self.metering_histogram = MiniHistogramWidget(clip_strips=False)  # the Clipping line below is the card's one clip readout
        self.process_section = self._make_section(
            tr("Metering"),
            "process",
            metering_body,
            icon_name="fa5s.tachometer-alt",
            background_widget=self.metering_histogram,
        )

        self.sensor_sidebar = SensorSidebar(self.controller)
        self.sensor_section = self._make_section(
            # Bare name: it holds the crosstalk matrix and Hue Trim as well as the sensor unmix. The
            # persisted "sensor" section key stays.
            tr("Calibration"),
            "sensor",
            self.sensor_sidebar,
            icon_name="fa5s.vials",
        )

        self.demosaic_sidebar = DemosaicSidebar(self.controller)
        # The persisted "demosaic" section key stays.
        self.demosaic_section = self._make_section(
            tr("Raw Decode"),
            "demosaic",
            self.demosaic_sidebar,
            icon_name="mdi6.grid",
        )

        # One-line answer to "roll-wide or this frame's own": which Roll-tab cards
        # (if any) this frame overrides. RightPanel places
        # it above every Roll-tab card; _sync_roll_locks keeps it current.
        self.roll_override_summary = hint_label("", "muted")
        self.roll_override_summary.setVisible(False)

        self.color_sidebar = ColorSidebar(self.controller)
        self.color_histogram = MiniRGBHistogramWidget()
        # Titled "Filtration"; the persisted "color" section key stays.
        self.color_section = self._make_section(
            tr("Filtration"),
            "color",
            self.color_sidebar,
            icon_name="fa5s.palette",
            background_widget=self.color_histogram,
        )

        self.tone_sidebar = ToneSidebar(self.controller)
        self.tone_histogram = MiniHistogramWidget()
        self.tone_section = self._make_section(
            tr("Tone"),
            "tone",
            self.tone_sidebar,
            icon_name="fa5s.sun",
            background_widget=self.tone_histogram,
        )

        self.paper_sidebar = PaperSidebar(self.controller)
        self.paper_curve = MiniCurveWidget()
        self.paper_section = self._make_section(
            tr("Paper Response"),
            "paper",
            self.paper_sidebar,
            icon_name="fa5s.scroll",
            background_widget=self.paper_curve,
        )

        self.lab_sidebar = LabSidebar(self.controller)
        self.lab_section = self._make_section(
            tr("Lab"),
            "lab",
            self.lab_sidebar,
            icon_name="fa5s.flask",
        )

        self.altproc_sidebar = AltProcessSidebar(self.controller)
        self.altproc_section = self._make_section(
            tr("Alternative Processes"),
            "altproc",
            self.altproc_sidebar,
            icon_name="fa5s.fire",
        )

        self.toning_sidebar = ToningSidebar(self.controller)
        self.toning_section = self._make_section(
            tr("Toning"),
            "toning",
            self.toning_sidebar,
            icon_name="fa5s.tint",
        )

        self.leak_sidebar = LeakSidebar(self.controller)
        self.leak_section = self._make_section(
            tr("Light Leak"),
            "leak",
            self.leak_sidebar,
            icon_name="fa5s.sun",
        )

        self.retouch_sidebar = RetouchSidebar(self.controller)
        self.retouch_section = self._make_section(
            tr("Retouch"),
            "retouch",
            self.retouch_sidebar,
            icon_name="fa5s.brush",
        )

        self.local_sidebar = LocalSidebar(self.controller)
        self.local_section = self._make_section(
            tr("Dodge & Burn"),
            "local",
            self.local_sidebar,
            icon_name="fa5s.adjust",
        )

        self.finish_sidebar = FinishSidebar(self.controller)
        self.finish_section = self._make_section(
            tr("Finishing"),
            "finish",
            self.finish_sidebar,
            icon_name="fa5s.paint-brush",
        )

        # Group the sections into workflow pages (each becomes a tab in RightPanel). Calibration,
        # Demosaic, Roll Analysis and Normalization are roll-wide facts, not per-frame edits --
        # RightPanel builds them into its own top-level Roll tab instead of a page here, and
        # places Presets on its Favorites tab.
        groups = [
            (
                "geometry",
                tr("Geometry"),
                tr("Geometry"),
                [self.autocrop_section, self.geometry_section],
                ["autocrop_section", "geometry_section"],
            ),
            (
                "tone",
                tr("Exposure: Filtration, Tone, Paper Response, Dodge & Burn"),
                tr("Exposure"),
                [self.color_section, self.tone_section, self.paper_section, self.local_section],
                ["color_section", "tone_section", "paper_section", "local_section"],
            ),
            (
                "color",
                tr("Look: Lab, Alternative Processes, Toning"),
                tr("Look"),
                [self.lab_section, self.altproc_section, self.toning_section],
                ["lab_section", "altproc_section", "toning_section"],
            ),
            (
                "finish",
                tr("Finish: Light Leak, Retouch, Finishing"),
                tr("Finish"),
                [self.leak_section, self.retouch_section, self.finish_section],
                ["leak_section", "retouch_section", "finish_section"],
            ),
        ]

        self.pages = []
        self.tab_headers: list[TabHeader] = []
        for key, tooltip, title, sections, section_attrs in groups:
            page = QWidget()
            page_layout = QVBoxLayout(page)
            page_layout.setContentsMargins(0, 0, 0, 0)
            page_layout.setSpacing(THEME.space_lg)
            # A one-card tab has no header: that card's own is already the whole tab's.
            header = (
                self._make_tab_header(title, sections, [a.removesuffix("_section") for a in section_attrs]) if len(sections) > 1 else None
            )
            if header is not None:
                page_layout.addWidget(header)
                if key == "tone":
                    self._add_pick_better(header)
                    self._add_auto_mirror(header)
                elif key == "geometry":
                    self.auto_crop_mirror_btn = header.add_mirror(self.autocrop_sidebar.auto_frame_btn)
                    self.square_to_film_mirror_btn = header.add_mirror(
                        self.geometry_sidebar.auto_skew_btn,
                        qta.icon("fa5s.ruler-combined", color=THEME.text_hint, color_disabled=THEME.text_muted),
                    )
                elif key == "color":
                    self.auto_masking_mirror_btn = header.add_mirror(self.lab_sidebar.auto_masking_btn)
                elif key == "finish":
                    self.optical_removal_mirror_btn = header.add_mirror(self.retouch_sidebar.auto_dust_btn)
                    self.ir_removal_mirror_btn = header.add_mirror(self.retouch_sidebar.ir_dust_btn)
            for section in sections:
                page_layout.addWidget(section)
            page_layout.addStretch(1)
            self.pages.append(
                {
                    "key": key,
                    "label": title,
                    "tooltip": tooltip,
                    "widget": page,
                    "sections": section_attrs,
                    "header": header,
                }
            )

    def _make_tab_header(self, title: str, sections: list, card_keys: list[str]) -> TabHeader:
        """One tab's header bar: how many of the cards below it are edited, and the reset
        and apply that reach all of them."""
        header = TabHeader(title)
        header.bind(sections)
        header.apply_requested.connect(lambda keys=tuple(card_keys): self._apply_tab(keys))
        header.roll_revert_requested.connect(lambda keys=tuple(card_keys): self.revert_cards_to_roll(keys))
        self.tab_headers.append(header)
        return header

    def _add_pick_better(self, header: TabHeader) -> None:
        """Pick Better tunes Filtration, Tone and Paper Response, so it sits on the Exposure tab's header."""
        self.pick_better_btn = header.add_lead_toggle(gradient_icon("fa5s.balance-scale"), self._pick_better_tooltip())
        self.pick_better_btn.clicked.connect(lambda checked: self.controller.toggle_pick_better(force=checked))
        # Session state, not config, so the button follows the controller rather than sync_ui.
        self.controller.pick_better_changed.connect(self._sync_pick_better_btn)

    def _add_auto_mirror(self, header: TabHeader) -> None:
        """The Tone card's Auto button again on the tab's header: the one menu, so both stay in step."""
        auto = self.tone_sidebar.auto_btn
        icon = qta.icon("fa5s.magic", color=THEME.text_hint, color_on=THEME.text_primary, color_disabled=THEME.text_muted)
        self.auto_mirror_btn = header.add_lead_toggle(icon, auto.plain_tooltip)
        self.auto_mirror_btn.setMenu(auto.choice_menu)
        self.auto_mirror_btn.setChecked(auto.isChecked())
        auto.toggled.connect(self.auto_mirror_btn.setChecked)

    @staticmethod
    def _pick_better_tooltip(printing: bool = False) -> str:
        if printing:
            return tr("Printing Pick Better…")
        return tooltip_with_shortcut(
            tr("Pick Better: shows two versions beside your edit, and each click on the better one tunes one control."),
            "toggle_pick_better",
        )

    def _sync_pick_better_btn(self, _up: bool) -> None:
        """ponytail: Pick Better is one job with no per-pair progress — the button is
        icon-only, so it goes dead with a 'printing' tooltip rather than changing its
        label. Per-pair progress if the endless refinement ever feels slow."""
        state = self.controller.state
        self.pick_better_btn.setChecked(state.pick_better or state.pick_better_pending)
        self.pick_better_btn.setEnabled(not state.pick_better_pending)
        self.pick_better_btn.setToolTip(wrap_tooltip(self._pick_better_tooltip(printing=state.pick_better_pending)))

    def _apply_tab(self, card_keys: tuple) -> None:
        """Every card on the tab in one picker. A whole-roll apply is recorded per card,
        the same record a card's own Roll button writes, so each header still reads back
        what went out."""
        live = [k for k in card_keys if not getattr(self, f"{k}_section").isHidden()]
        rows = list(dict.fromkeys(r for k in live for r in frame_card_rows(k)))
        if not rows:
            return
        applied = open_apply_dialog(self, self.controller.session, rows=rows)
        if applied and applied[1] == "roll":
            self.controller.record_roll_apply(applied[0])

    def _make_section(
        self,
        title: str,
        key: str,
        widget: QWidget,
        icon_name: str,
        background_widget=None,
        collapsible: bool = True,
    ) -> CollapsibleSection:
        return make_section(
            self.controller.session.repo,
            title,
            key,
            widget,
            icon_name,
            default_expanded=THEME.sidebar_expanded_defaults.get(key, False),
            background_widget=background_widget,
            collapsible=collapsible,
        )

    def _connect_signals(self) -> None:
        self._sync_debounce = QTimer()
        self._sync_debounce.setSingleShot(True)
        self._sync_debounce.setInterval(150)
        self._sync_debounce.timeout.connect(self._sync_all_sidebars)
        self.controller.config_updated.connect(self._sync_debounce.start)
        self.controller.tool_sync_requested.connect(self._sync_tool_buttons)
        # The histogram only changes on render completion, so refresh there, not on every resync.
        self.controller.image_updated.connect(self._update_histogram)
        self._sync_clip_colors()
        self.controller.session.color_vision_changed.connect(self._sync_clip_colors)

        self.color_section.reset_requested.connect(lambda: self._reset_exposure_fields(COLOR_FIELDS))
        self.tone_section.reset_requested.connect(self._reset_tone_fields)
        self.paper_section.reset_requested.connect(lambda: self._reset_exposure_fields(PAPER_FIELDS))
        self.lab_section.reset_requested.connect(lambda: self.controller.session.reset_section("lab"))
        self.altproc_section.reset_requested.connect(lambda: self.controller.session.reset_section("altproc"))
        self.toning_section.reset_requested.connect(lambda: self.controller.session.reset_section("toning"))
        self.geometry_section.reset_requested.connect(self._reset_geometry_fields)
        self.autocrop_section.reset_requested.connect(self._reset_crop_card)
        self.optics_section.reset_requested.connect(self._reset_optics)
        self.process_section.reset_requested.connect(lambda: self._reset_process_fields(_METERING_FIELDS))
        self.baseline_section.reset_requested.connect(lambda: self._reset_process_fields(_BASELINE_FIELDS))
        self.retouch_section.reset_requested.connect(lambda: self.controller.session.reset_section("retouch"))
        self.leak_section.reset_requested.connect(lambda: self.controller.session.reset_section("leak"))
        self.local_section.reset_requested.connect(lambda: self.controller.session.reset_section("local"))
        self.finish_section.reset_requested.connect(lambda: self.controller.session.reset_section("finish"))
        self.film_section.reset_requested.connect(self._reset_film_fields)
        self.sensor_section.reset_requested.connect(self._reset_sensor_fields)
        self.demosaic_section.reset_requested.connect(lambda: self._reset_process_fields(_DEMOSAIC_FIELDS))

        self.controller.batch_started.connect(lambda *_: self._sync_crop_block())
        self.controller.batch_finished.connect(self._sync_crop_block)

        for key, section in self._roll_sections() + self._frame_sections():
            section.scope_selected.connect(lambda scope, k=key: self._on_scope_selected(k, scope))
            section.roll_revert_requested.connect(lambda k=key: self.controller.revert_to_roll(_SECTION_CARDS.get(k, (k,))))
        for key, section in self._roll_sections() + self._frame_sections():
            section.toggle_button.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
            section.toggle_button.customContextMenuRequested.connect(lambda pos, k=key, s=section: self._show_card_menu(k, s, pos))

    def _show_card_menu(self, key: str, section: CollapsibleSection, pos) -> None:
        menu = QMenu(self)
        menu.setToolTipsVisible(True)
        if key in dict(self._roll_sections()):
            undo = menu.addAction(label_with_shortcut(tr("Undo Apply to Roll"), "undo_roll_push"))
            undo.setToolTip(tr("Give the roll back the values it had before the last Roll push from this frame"))
            undo.setEnabled(self.controller.can_undo_roll_push(_SECTION_CARDS.get(key, (key,))))
            undo.triggered.connect(self.controller.undo_roll_push)
        else:
            copy = menu.addAction(tr("Copy Card Settings"))
            copy.setToolTip(tr("Copy this card alone; Paste then offers only its settings"))
            copy.triggered.connect(lambda: self.controller.session.copy_card_settings(frame_card_rows(key)))
        menu.exec(section.toggle_button.mapToGlobal(pos))

    def apply_shortcut_tooltips(self) -> None:
        """Single source for every shortcut-bearing widget tooltip — re-run on each
        rebind to re-render the key chips. Don't set these locally in the sidebars:
        this pass overwrites them."""
        col = self.color_sidebar
        for btn, action_id in (
            (self.retouch_sidebar.auto_dust_btn, "toggle_optical_removal"),
            (self.retouch_sidebar.right_click_btn, "toggle_right_click_excludes"),
            (self.retouch_sidebar.ir_dust_btn, "toggle_ir_removal"),
            (self.leak_sidebar.leak_btn, "toggle_light_leak"),
            (self.leak_sidebar.band_btn, "tool_leak_band"),
            (self.leak_sidebar.edge_btn, "tool_leak_edge"),
            (self.leak_sidebar.spot_btn, "tool_leak_spot"),
            (self.leak_sidebar.paint_btn, "tool_leak_paint"),
            (self.leak_sidebar.auto_btn, "leak_auto"),
            (self.leak_sidebar.roll_btn, "leak_from_roll"),
            (self.leak_sidebar.burned_btn, "toggle_leak_burned"),
            (self.optical_removal_mirror_btn, "toggle_optical_removal"),
            (self.ir_removal_mirror_btn, "toggle_ir_removal"),
            (self.auto_crop_mirror_btn, "auto_crop"),
            (self.lab_sidebar.auto_masking_btn, "toggle_auto_masking"),
            (self.auto_masking_mirror_btn, "toggle_auto_masking"),
            (self.flatfield_sidebar.enable_btn, "toggle_flat_field"),
            (self.flatfield_sidebar.check_btn, "toggle_flatfield_peek"),
            (self.autocrop_sidebar.auto_frame_btn, "auto_crop"),
            (self.autocrop_sidebar.square_btn, "auto_crop_square"),
            (self.autocrop_sidebar.auto_crop_all_btn, "batch_autocrop"),
            (self.tone_sidebar.auto_density_action, "toggle_auto_density"),
            (self.tone_sidebar.auto_grade_action, "toggle_auto_grade"),
            (self.tone_sidebar.auto_both_action, "toggle_auto_both"),
            (self.tone_sidebar.hold_black_action, "toggle_hold_black"),
            (self.presets_sidebar.apply_btn, "preset_apply"),
            (self.presets_sidebar.save_btn, "preset_save"),
        ):
            btn.setToolTip(wrap_tooltip(tooltip_with_shortcut(btn.plain_tooltip, action_id)))
        exp = self.tone_sidebar
        paper = self.paper_sidebar
        geo = self.geometry_sidebar
        crop = self.autocrop_sidebar
        lens = self.lens_sidebar
        lab = self.lab_sidebar
        proc = self.process_sidebar
        sen = self.sensor_sidebar
        ret = self.retouch_sidebar
        ton = self.toning_sidebar
        fin = self.finish_sidebar
        lens.metadata_distortion_btn.setToolTip(
            tooltip_with_shortcut(
                tr("Use the file's own lens distortion correction, in place of the slider."),
                "lens_distortion_from_metadata",
            )
        )
        lens.metadata_ca_btn.setToolTip(
            tooltip_with_shortcut(
                tr("Use the file's own correction for color fringes."),
                "lens_ca_from_metadata",
            )
        )

        col.pick_wb_btn.setToolTip(
            tooltip_with_shortcut(
                tr("Click something in the picture that should be gray, and the color is corrected to make it gray."),
                "pick_wb",
            )
        )
        col.temp_slider.setToolTip(
            tooltip_with_shortcut(
                tr("Makes the picture warmer or cooler."),
                ["temp_warm", "temp_cool"],
            )
        )
        col.cyan_slider.setToolTip(
            tooltip_with_shortcut(
                tr("Shifts the color between cyan and red."),
                ["cyan_inc", "cyan_dec"],
            )
        )
        col.magenta_slider.setToolTip(
            tooltip_with_shortcut(
                tr("Shifts the color between magenta and green."),
                ["magenta_up", "magenta_down"],
            )
        )
        col.yellow_slider.setToolTip(
            tooltip_with_shortcut(
                tr("Shifts the color between yellow and blue."),
                ["yellow_up", "yellow_down"],
            )
        )
        exp.density_slider.setToolTip(
            tooltip_with_shortcut(
                tr("How light or dark the print is, where lower is lighter."),
                ["density_up", "density_down"],
            )
        )
        exp.grade_slider.setToolTip(
            tooltip_with_shortcut(
                tr("Contrast: a lower number gives more contrast, and about 110 is a normal print."),
                ["grade_up", "grade_down"],
            )
        )
        paper.toe_slider.setToolTip(
            tooltip_with_shortcut(
                tr("Shapes the shadows: positive lifts them, and negative makes them deeper."),
                ["toe_inc", "toe_dec"],
            )
        )
        paper.toe_w_slider.setToolTip(
            tooltip_with_shortcut(
                tr("How far the Toe change reaches into the midtones."),
                ["toe_width_inc", "toe_width_dec"],
            )
        )
        paper.sh_slider.setToolTip(
            tooltip_with_shortcut(
                tr("Shapes the highlights: positive softens them, and negative brightens them and can lose detail."),
                ["shoulder_inc", "shoulder_dec"],
            )
        )
        paper.sh_w_slider.setToolTip(
            tooltip_with_shortcut(
                tr("How far the Shoulder change reaches into the midtones."),
                ["shoulder_width_inc", "shoulder_width_dec"],
            )
        )
        paper.midtone_gamma_slider.setToolTip(
            tooltip_with_shortcut(
                tr("Midtone contrast, with no change to the deepest black and the paper white."),
                ["snap_inc", "snap_dec"],
            )
        )
        exp.shadow_density_slider.setToolTip(
            tooltip_with_shortcut(
                tr("Darkens (positive) or lightens (negative) only the shadows, and Auto Grade can set it too."),
                ["shadow_density_inc", "shadow_density_dec"],
            )
        )
        exp.highlight_density_slider.setToolTip(
            tooltip_with_shortcut(
                tr("Darkens (positive) or lightens (negative) only the highlights, and Auto Grade can set it too."),
                ["highlight_density_inc", "highlight_density_dec"],
            )
        )
        exp.shadow_grade_slider.setToolTip(
            tooltip_with_shortcut(
                tr("Contrast in the shadows only."),
                ["shadow_grade_inc", "shadow_grade_dec"],
            )
        )
        exp.highlight_grade_slider.setToolTip(
            tooltip_with_shortcut(
                tr("Contrast in the highlights only."),
                ["highlight_grade_inc", "highlight_grade_dec"],
            )
        )

        crop.manual_crop_btn.setToolTip(
            tooltip_with_shortcut(
                tr("Draw: drag a crop on the picture."),
                "manual_crop",
            )
        )
        geo.auto_skew_btn.setToolTip(
            tooltip_with_shortcut(
                tr("Square to Film: line the picture up with the edges of the film."),
                "auto_skew",
            )
        )
        geo.straighten_btn.setToolTip(
            tooltip_with_shortcut(
                tr("Level: draw a line along the horizon or a vertical edge, and the picture turns to level it."),
                "straighten",
            )
        )
        for btn, text, action_id in (
            (geo.rot_l_btn, tr("Rotate 90° counter-clockwise."), "rotate_ccw"),
            (geo.rot_r_btn, tr("Rotate 90° clockwise."), "rotate_cw"),
            (geo.flip_h_btn, tr("Flip the picture left to right."), "flip_h"),
            (geo.flip_v_btn, tr("Flip the picture top to bottom."), "flip_v"),
        ):
            btn.setToolTip(tooltip_with_shortcut(text, action_id))
        geo.keystone_lines_btn.setToolTip(
            tooltip_with_shortcut(
                tr("Draw a line along each edge of the film, and Tilt and Swing make them square."),
                "keystone_lines",
            )
        )
        crop.offset_slider.setToolTip(
            tooltip_with_shortcut(
                tr("Moves the crop inside (positive) or outside (negative) the found edge."),
                ["offset_inc", "offset_dec"],
            )
        )
        geo.fine_rot_slider.setToolTip(
            tooltip_with_shortcut(
                tr("Turns the picture by small amounts, clockwise for a positive value."),
                ["fine_rot_inc", "fine_rot_dec"],
            )
        )

        proc.lock_bounds_btn.setToolTip(
            tooltip_with_shortcut(
                tr("Lock Bounds: keep this frame's measured black and white points, even when the crop or Roll Analysis changes."),
                "lock_bounds_toggle",
            )
        )
        proc.use_cast_avg_btn.setToolTip(
            tooltip_with_shortcut(
                tr("Color negatives only: use the roll's gray balance for Cast Removal, so frames under one light match."),
                "cast_average_toggle",
            )
        )
        proc.analysis_buffer_slider.setToolTip(
            tooltip_with_shortcut(
                tr("Ignores a border around the frame when measuring, so film edges and holders do not count."),
                ["analysis_buffer_inc", "analysis_buffer_dec"],
            )
        )
        proc.luma_range_clip_slider.setToolTip(
            tooltip_with_shortcut(
                tr(
                    "How tightly black and white are set: positive gives more contrast, and negative keeps more shadow and highlight detail."
                ),
                ["luma_range_clip_inc", "luma_range_clip_dec"],
            )
        )
        proc.color_range_clip_slider.setToolTip(
            tooltip_with_shortcut(
                tr("How tightly the color balance is set: positive balances more strongly, and negative is gentler."),
                ["color_range_clip_inc", "color_range_clip_dec"],
            )
        )
        proc.white_point_slider.setToolTip(
            tooltip_with_shortcut(
                tr("Moves the white point: positive brightens the print, and negative brings back highlights."),
                ["white_point_inc", "white_point_dec"],
            )
        )
        proc.black_point_slider.setToolTip(
            tooltip_with_shortcut(
                tr("Moves the black point: positive lifts the blacks, and negative makes them deeper."),
                ["black_point_inc", "black_point_dec"],
            )
        )

        sen.crosstalk_strength_slider.setToolTip(
            tooltip_with_shortcut(
                tr("How much crosstalk correction to apply; run Roll Analysis again after a change."),
                ["separation_inc", "separation_dec"],
            )
        )
        lab.chroma_denoise_slider.setToolTip(
            tooltip_with_shortcut(
                tr("Removes colored speckles and keeps the film grain."),
                ["chroma_denoise_inc", "chroma_denoise_dec"],
            )
        )
        lab.saturation_slider.setToolTip(
            tooltip_with_shortcut(
                tr("Color strength. 1 is unchanged, 0 is black and white."),
                ["saturation_inc", "saturation_dec"],
            )
        )
        lab.skin_protection_slider.setToolTip(tr("Keeps skin tones from getting too strong in color. 0 is off."))
        paper.dye_separation_slider.setToolTip(
            tooltip_with_shortcut(
                tr("Color strength in the print, through the paper's dyes. 1 is unchanged."),
                ["dye_separation_inc", "dye_separation_dec"],
            )
        )
        paper.separation_damping_slider.setToolTip(
            tooltip_with_shortcut(
                tr("Puts more of the Dye Separation change on dull colors and less on strong ones."),
                ["separation_damping_inc", "separation_damping_dec"],
            )
        )
        exp.preflash_slider.setToolTip(
            tooltip_with_shortcut(
                tr("Preflash: brings back detail in the brightest areas, and also lowers contrast a little."),
                ["preflash_inc", "preflash_dec"],
            )
        )
        lab.clahe_slider.setToolTip(
            tooltip_with_shortcut(
                tr("Adds local contrast to bring out detail, but strong values can look unnatural."),
                ["clahe_inc", "clahe_dec"],
            )
        )
        lab.sharpen_slider.setToolTip(
            tooltip_with_shortcut(
                tr("Sharpens detail without colored edges."),
                ["sharpen_inc", "sharpen_dec"],
            )
        )
        lab.sharpen_method_combo.setToolTip(
            tr("Unsharp Mask makes edges crisper, and Deconvolution undoes the scanner's blur when Radius is the blur size.")
        )
        lab.sharpen_radius_slider.setToolTip(tr("Size of the detail to sharpen, in pixels: small for fine grain, larger for soft scans."))
        lab.sharpen_masking_slider.setToolTip(tr("Sharpens only edges, so sky, skin and grain stay smooth."))
        lab.glow_slider.setToolTip(
            tooltip_with_shortcut(
                tr("A soft glow around bright areas, like an old lens."),
                ["glow_inc", "glow_dec"],
            )
        )
        lab.halation_slider.setToolTip(
            tooltip_with_shortcut(
                tr("A red glow around bright areas, as some films give."),
                ["halation_inc", "halation_dec"],
            )
        )

        ret.pick_dust_btn.setToolTip(
            tooltip_with_shortcut(
                tr("Heal: click or paint over dust to remove it, and clean film under the brush stays as it is."),
                "pick_dust",
            )
        )
        ret.clone_btn.setToolTip(
            tooltip_with_shortcut(
                tr("Clone: Alt-click to pick an area, then copy its film over a defect."),
                "clone_tool",
            )
        )
        ret.threshold_slider.setToolTip(
            tooltip_with_shortcut(
                tr("How easily dust specks are found: lower finds more but can remove real detail, and 1 is off."),
                ["threshold_inc", "threshold_dec"],
            )
        )
        ret.hair_threshold_slider.setToolTip(
            tooltip_with_shortcut(
                tr("How easily hairs are found: lower finds more, and 1 is off."),
                ["hair_threshold_inc", "hair_threshold_dec"],
            )
        )
        ret.auto_size_slider.setToolTip(
            tooltip_with_shortcut(
                tr("Largest dust size to remove, where larger values can remove real detail."),
                ["auto_size_inc", "auto_size_dec"],
            )
        )
        ret.manual_size_slider.setToolTip(
            tooltip_with_shortcut(
                tr("Radius of the manual heal brush"),
                ["manual_size_inc", "manual_size_dec"],
            )
        )

        ton.selenium_slider.setToolTip(
            tooltip_with_shortcut(
                tr("Selenium toner, B&W negatives only: deeper, cooler blacks."),
                ["selenium_inc", "selenium_dec"],
            )
        )
        ton.sepia_slider.setToolTip(
            tooltip_with_shortcut(
                tr("Sepia toner, B&W negatives only: warm brown tones, from the highlights down."),
                ["sepia_inc", "sepia_dec"],
            )
        )
        ton.shadow_hue_slider.setToolTip(
            tooltip_with_shortcut(
                tr("Color added to the shadows."),
                ["shadow_hue_inc", "shadow_hue_dec"],
            )
        )
        ton.shadow_str_slider.setToolTip(
            tooltip_with_shortcut(
                tr("How much color is added to the shadows."),
                ["shadow_strength_inc", "shadow_strength_dec"],
            )
        )
        ton.highlight_hue_slider.setToolTip(
            tooltip_with_shortcut(
                tr("Color added to the highlights."),
                ["highlight_hue_inc", "highlight_hue_dec"],
            )
        )
        ton.highlight_str_slider.setToolTip(
            tooltip_with_shortcut(
                tr("How much color is added to the highlights."),
                ["highlight_strength_inc", "highlight_strength_dec"],
            )
        )

        fin.vignette_burn_slider.setToolTip(
            tooltip_with_shortcut(
                tr("Darkens (positive) or lightens (negative) the edges of the print. 0 is off."),
                ["vignette_str_inc", "vignette_str_dec"],
            )
        )
        fin.vignette_size_slider.setToolTip(
            tooltip_with_shortcut(
                tr("How far the effect reaches into the picture."),
                ["vignette_size_inc", "vignette_size_dec"],
            )
        )
        fin.vignette_roundness_slider.setToolTip(tr("Shape: 0 is round, like a lens; 1 follows the edges of the print."))
        fin.border_slider.setToolTip(
            tooltip_with_shortcut(
                tr("Border width. 0 is no border."),
                ["border_size_inc", "border_size_dec"],
            )
        )

    _DIPTYCH_HINT = tr("Diptych: the edits belong to the halves. Turn Half Frame Mode on to edit either one.")

    def _set_read_only(self, read_only: bool) -> None:
        """A diptych renders from the two halves' own configs, so this panel drives nothing.

        Announced on the transition rather than as a tooltip: Qt gives no tooltip to a
        disabled widget.
        """
        if read_only == self._read_only:
            return
        self._read_only = read_only
        for page in self.pages:
            page["widget"].setEnabled(not read_only)
        if read_only:
            self.controller.set_status(self._DIPTYCH_HINT, 6000)

    def _sync_all_sidebars(self) -> None:
        """Force all sidebar panels to update their widgets from current AppState."""
        from negpy.features.process.models import ProcessMode

        self._set_read_only(self.controller.active_diptych() is not None)
        process = self.controller.state.config.process
        self.color_section.setVisible(process.process_mode != ProcessMode.BW or bool(process.source_color))
        self.process_sidebar.sync_ui()
        self.roll_sidebar.sync_ui()
        self.color_sidebar.sync_ui()
        self.tone_sidebar.sync_ui()
        self.paper_sidebar.sync_ui()
        self.geometry_sidebar.sync_ui()
        self.lab_sidebar.sync_ui()
        self.altproc_sidebar.sync_ui()
        self.toning_sidebar.sync_ui()
        self.retouch_sidebar.sync_ui()
        self.leak_sidebar.sync_ui()
        self.local_sidebar.sync_ui()
        self.finish_sidebar.sync_ui()
        self.presets_sidebar.sync_ui()
        self.flatfield_sidebar.sync_ui()
        self.autocrop_sidebar.sync_ui()
        self.lens_sidebar.sync_ui()
        self.trichrome_sidebar.sync_ui()
        self.half_frame_sidebar.sync_ui()
        self.sensor_sidebar.sync_ui()
        self.demosaic_sidebar.sync_ui()
        self.scan_setup_sidebar.sync_ui()
        self._sync_scan_setup()
        # Scope first: the tab headers refresh at the end of the modified pass and read
        # each card's Reset to Roll state.
        self._sync_scope_buttons()
        self._sync_modified_dots()

    def _sync_scan_setup(self) -> None:
        """The scanning setup decides which rig cards show: Raw Decode and Optics are a camera's,
        Trichrome a narrowband camera's. A card or row that holds a setting stays in view."""
        state = self.controller.state
        capture, light = resolved_rig(self.controller.session.repo, state)
        camera = capture == "camera"
        cfg = state.config
        self.trichrome_sidebar.setVisible(
            (camera and light == "narrowband") or bool(self.controller.rgb_scan_mode_for_roll(state.active_roll_id))
        )
        self.sensor_section.setVisible(self.sensor_sidebar.has_controls())
        self.demosaic_section.setVisible(camera or cfg.process.highlight_reconstruction != _DEFAULT_PROCESS.highlight_reconstruction)
        optics_edited = any(getattr(cfg.geometry, f) != getattr(_DEFAULT_GEOMETRY, f) for f in _LENS_FIELDS) or cfg.flatfield.apply
        self.optics_section.setVisible(camera or optics_edited)

    _ROLL_CARD_LABELS = AppController._ROLL_CARD_LABELS

    def _roll_sections(self) -> tuple:
        return (
            ("film", self.film_section),
            ("demosaic", self.demosaic_section),
            ("optics", self.optics_section),
            ("sensor", self.sensor_section),
            ("process", self.process_section),
            ("baseline", self.baseline_section),
        )

    def _sync_scope_buttons(self) -> None:
        """Each card's Frame/Roll pair and Reset to Roll, and roll_override_summary's one-line
        answer to "roll-wide or this frame's own" alongside them. A Roll-tab card reads its
        own lock; a frame card reads the whole-roll apply recorded for it.

        Frames that are not one roll (a library search's results, several folders at once)
        read Frame with Roll disabled: every value there is the frame's own, since no roll
        spans them to hold a shared one. Save as Roll gives them one."""
        has_roll = self.controller.state.active_roll_id is not None
        overridden = []
        locked_cards = self.controller.locked_roll_cards()
        roll_sections = self._roll_sections()
        frame_sections = self._frame_sections()
        revertible = self.controller.roll_revert_cards(
            [card for key, _ in roll_sections + frame_sections for card in _SECTION_CARDS.get(key, (key,))]
        )
        for section_key, section in roll_sections:
            cards = _SECTION_CARDS.get(section_key, (section_key,))
            locked = any(card in locked_cards for card in cards)
            label = self._ROLL_CARD_LABELS[cards[0]]
            section.set_scope_buttons(
                True,
                "frame" if locked or not has_roll else "roll",
                roll_tooltip=(
                    tr("{value} follows the roll: click to give the roll this frame's value", value=label)
                    if has_roll
                    else NO_ROLL_SCOPE_HINT
                ),
                frame_tooltip=tr("{value} follows this frame alone", value=label)
                if has_roll
                else tr("{value} is this frame's own", value=label),
                roll_enabled=has_roll,
            )
            section.set_roll_revert(any(card in revertible for card in cards))
            if locked:
                overridden.append(label)

        scopes = self.controller.frame_section_scopes(tuple(key for key, _ in frame_sections))
        for key, section in frame_sections:
            section.set_scope_buttons(
                True,
                scopes[key],
                roll_tooltip="" if has_roll else NO_ROLL_SCOPE_HINT,
                roll_enabled=has_roll,
            )
            section.set_roll_revert(key in revertible)

        if overridden:
            set_hint_kind(self.roll_override_summary, "warning")
            self.roll_override_summary.setText(tr("This frame overrides: {value}", value=", ".join(overridden)))
        else:
            self.roll_override_summary.setText("")
        self.roll_override_summary.setVisible(bool(overridden))

    def revert_cards_to_roll(self, section_keys) -> None:
        """Every live card among *section_keys* back to the roll, as one undo step. A card
        the film mode has retired stays as it is."""
        live = [k for k in section_keys if not getattr(self, f"{k}_section").isHidden()]
        self.controller.revert_to_roll([card for k in live for card in _SECTION_CARDS.get(k, (k,))])

    def _frame_sections(self) -> tuple:
        return tuple((key, getattr(self, f"{key}_section")) for key in FRAME_CARD_FIELDS)

    def _on_scope_selected(self, key: str, scope: str) -> None:
        """Roll on a Roll-tab card pushes that card out; on a frame card it opens the
        picker over that card's own settings. Frame on a Roll-tab card locks it here;
        a frame card is already there, so the pair's own click guard swallows it."""
        if key in dict(self._roll_sections()):
            self.controller.set_card_scope(_SECTION_CARDS.get(key, key), scope)
            self._sync_scope_buttons()
            return
        applied = open_apply_dialog(self, self.controller.session, rows=frame_card_rows(key))
        if applied and applied[1] == "roll":
            self.controller.record_roll_apply(applied[0])

    def _sync_clip_colors(self) -> None:
        self.tone_histogram.set_clip_colors(*palette_for(self.controller.state.color_vision).pair)

    def _update_histogram(self) -> None:
        """Repaint only when the render produced a new buffer."""
        buf = self.controller.state.last_metrics.get("histogram_raw")
        if buf is self._last_histogram_buf:
            return
        self._last_histogram_buf = buf
        self.tone_histogram.update_data(buf)
        self.color_histogram.update_data(buf)
        self.metering_histogram.update_data(buf)

    def _reset_sensor_fields(self) -> None:
        self._reset_process_fields(_SENSOR_FIELDS)

    def _reset_film_fields(self) -> None:
        """Film Mode and Positive both carry side effects their plain fields do not
        describe (a decode, the auto-meter defaults, the card's roll lock), so the reset
        goes through the same controller calls the two controls use."""
        proc = self.controller.state.config.process
        if proc.positive_source != _DEFAULT_PROCESS.positive_source:
            self.controller.set_positive_source(_DEFAULT_PROCESS.positive_source)
        if proc.process_mode != _DEFAULT_PROCESS.process_mode:
            self.controller.set_process_mode(_DEFAULT_PROCESS.process_mode)

    def _reset_tone_fields(self) -> None:
        self._reset_exposure_fields(TONE_FIELDS)

    def _reset_process_fields(self, fields) -> None:
        """Calibration, Demosaic and Normalization all live on ProcessConfig, so each
        reset is scoped to its own fields -- a plain session.reset_section("process")
        would reset all three cards (and Film Mode, and Positive) at once. apply_config,
        not update_config: some of these fields are a decode or a source bake."""
        from dataclasses import replace

        cfg = self.controller.state.config
        new_proc = replace(cfg.process, **{f: getattr(_DEFAULT_PROCESS, f) for f in fields})
        self.controller.apply_config(replace(cfg, process=new_proc), persist=True)

    def _reset_optics(self) -> None:
        self._reset_card_fields("lens")
        self._reset_card_fields("flatfield")

    def _sync_crop_block(self) -> None:
        """Crop controls are disabled while Auto-crop all frames runs (controller.crop_edits_blocked)."""
        enabled = not self.controller.crop_edits_blocked()
        for widget in (self.autocrop_sidebar, self.autocrop_section.reset_btn, self.autocrop_section.roll_revert_btn):
            widget.setEnabled(enabled)

    def _reset_geometry_fields(self) -> None:
        """Geometry's own fields alone: a plain session.reset_section("geometry") would
        take Crop and Lens Correction with it."""
        from dataclasses import replace

        cfg = self.controller.state.config
        new_geo = replace(cfg.geometry, **{f: getattr(_DEFAULT_GEOMETRY, f) for f in GEOMETRY_FIELDS})
        self.controller.apply_config(replace(cfg, geometry=new_geo), persist=True)

    def _reset_crop_card(self) -> None:
        self.controller.reset_crop(**{f: getattr(_DEFAULT_GEOMETRY, f) for f in CROP_FIELDS if f.startswith("autocrop_")})

    def _reset_card_fields(self, card_key: str) -> None:
        """Reset one roll card through set_roll_default, so the reset follows the roll
        or locks away from it exactly as an edit by hand would."""
        section, fields = ROLL_DEFAULT_FIELDS[card_key]
        default = getattr(_DEFAULT_CONFIG, section)
        self.controller.set_roll_default(card_key, **{f: getattr(default, f) for f in fields})

    def _reset_exposure_fields(self, fields) -> None:
        """Reset only the given ExposureConfig fields to defaults (scoped section reset).
        A reset means the value the frame's own defaulting rules would carry, not the
        flat ExposureConfig default (_default_exposure_field)."""
        from dataclasses import replace

        cfg = self.controller.state.config
        exp = cfg.exposure
        defaults = {f: _default_exposure_field(f, cfg.process.process_mode) for f in fields}
        new_exp = replace(exp, **defaults)
        new_config = replace(cfg, exposure=new_exp)
        self.controller.session.update_config(new_config, persist=True)

    def _sync_modified_dots(self) -> None:
        """Update modified-indicator dots on collapsible section headers."""
        from negpy.features.process.models import ProcessMode

        cfg = self.controller.state.config
        _exp = _DEFAULT_EXPOSURE
        _lab = _DEFAULT_LAB
        _alt = _DEFAULT_ALTPROC
        _ton = _DEFAULT_TONING
        _geo = _DEFAULT_GEOMETRY
        _proc = _DEFAULT_PROCESS

        exp = cfg.exposure
        mode = cfg.process.process_mode
        # The card shows only the Filter in B&W, and the Filter only there.
        shown_color = ("bw_filter",) if mode == ProcessMode.BW else tuple(f for f in COLOR_FIELDS if f != "bw_filter")
        color_count = sum(getattr(exp, f) != _default_exposure_field(f, mode) for f in shown_color)
        tone_count = sum(getattr(exp, f) != _default_exposure_field(f, mode) for f in TONE_FIELDS)
        paper_count = sum(getattr(exp, f) != _default_exposure_field(f, mode) for f in PAPER_FIELDS)

        lab = cfg.lab
        lab_count = sum(
            [
                lab.saturation != _lab.saturation,
                lab.clahe_strength != _lab.clahe_strength,
                lab.sharpen != _lab.sharpen,
                lab.chroma_denoise != _lab.chroma_denoise,
                lab.glow_amount != _lab.glow_amount,
                lab.halation_strength != _lab.halation_strength,
            ]
        )

        alt = cfg.altproc
        altproc_count = sum(
            [
                alt.alt_process != _alt.alt_process,
                alt.lith_exposure != _alt.lith_exposure,
                alt.lith_snatch != _alt.lith_snatch,
                alt.lith_abruptness != _alt.lith_abruptness,
                alt.cyano_sensitizer != _alt.cyano_sensitizer,
                alt.cyano_exposure != _alt.cyano_exposure,
                alt.cyano_scale != _alt.cyano_scale,
                alt.cyano_bleach != _alt.cyano_bleach,
                alt.cyano_tannin != _alt.cyano_tannin,
                alt.sabattier_strength != _alt.sabattier_strength,
                alt.sabattier_reexposure != _alt.sabattier_reexposure,
                alt.sabattier_agitation != _alt.sabattier_agitation,
            ]
        )

        ton = cfg.toning
        toning_count = sum(
            [
                ton.selenium_strength != _ton.selenium_strength,
                ton.sepia_strength != _ton.sepia_strength,
                ton.gold_strength != _ton.gold_strength,
                ton.blue_strength != _ton.blue_strength,
                ton.copper_strength != _ton.copper_strength,
                ton.vanadium_strength != _ton.vanadium_strength,
                ton.shadow_tint_hue != _ton.shadow_tint_hue,
                ton.shadow_tint_strength != _ton.shadow_tint_strength,
                ton.highlight_tint_hue != _ton.highlight_tint_hue,
                ton.highlight_tint_strength != _ton.highlight_tint_strength,
            ]
        )

        geo = cfg.geometry
        geometry_count = sum(getattr(geo, f) != getattr(_geo, f) for f in GEOMETRY_FIELDS)
        # A rect counts once, as a hand-drawn crop; an auto rect is Auto Crop's result.
        autocrop_count = sum(getattr(geo, f) != getattr(_geo, f) for f in CROP_FIELDS if f not in ("crop_rect", "crop_detect_key"))
        autocrop_count += has_manual_crop(geo)
        lens_count = sum(getattr(geo, f) != getattr(_geo, f) for f in _LENS_FIELDS)

        proc = cfg.process
        film_count = sum(getattr(proc, f) != getattr(_proc, f) for f in _FILM_FIELDS)
        process_count = sum(getattr(proc, f) != getattr(_proc, f) for f in _METERING_FIELDS)
        demosaic_count = sum(getattr(proc, f) != getattr(_proc, f) for f in _DEMOSAIC_FIELDS)
        sensor_count = sum(getattr(proc, f) != getattr(_proc, f) for f in _SENSOR_FIELDS)

        ff = cfg.flatfield
        _ff = _DEFAULT_FLATFIELD
        flatfield_count = sum(
            [
                ff.apply != _ff.apply,
                ff.profile_id != _ff.profile_id,
            ]
        )

        ret = cfg.retouch
        # Heal-tool clicks and scratch polylines both commit into manual_heal_strokes, where
        # manual_dust_spots is the legacy list, so count them or the Finish tab's edited dot
        # never lights for healed images.
        retouch_count = int(ret.dust_remove) + len(ret.manual_dust_spots) + len(ret.manual_heal_strokes)
        leak = cfg.leak
        _leak = _DEFAULT_LEAK
        leak_count = len(leak.leak_shapes) + sum(
            getattr(leak, f) != getattr(_leak, f) for f in ("leak_remove", "leak_dmax", "leak_recover", "leak_grain", "leak_contrast")
        )

        _fin = _DEFAULT_FINISH
        fin = cfg.finish
        finish_count = sum(
            [
                fin.vignette_stops != _fin.vignette_stops,
                fin.vignette_size != _fin.vignette_size,
                fin.vignette_roundness != _fin.vignette_roundness,
                fin.carrier_width != _fin.carrier_width,
                fin.carrier_rough != _fin.carrier_rough,
                fin.carrier_flare != _fin.carrier_flare,
                fin.carrier_corner != _fin.carrier_corner,
                fin.border_size != _fin.border_size,
                fin.border_color != _fin.border_color,
                fin.border_bottom_weight != _fin.border_bottom_weight,
                fin.border_match_paper != _fin.border_match_paper,
            ]
        )

        self.film_section.set_modified(film_count)
        self.color_section.set_modified(color_count)
        self.tone_section.set_modified(tone_count)
        self.paper_section.set_modified(paper_count)
        self.lab_section.set_modified(lab_count)
        self.altproc_section.set_modified(altproc_count)
        self.toning_section.set_modified(toning_count)
        self.geometry_section.set_modified(geometry_count)
        self.autocrop_section.set_modified(autocrop_count)
        self.optics_section.set_modified(lens_count + flatfield_count)
        self.process_section.set_modified(process_count)
        # The picked roll counts against Roll Analysis, the card it sits on.
        self.baseline_section.set_modified(
            sum(getattr(proc, f) != getattr(_proc, f) for f in _BASELINE_FIELDS) + (proc.roll_name is not None)
        )
        self.retouch_section.set_modified(retouch_count)
        self.leak_section.set_modified(leak_count)
        # Presets and the two Scan sections stay out: they own no WorkspaceConfig fields.
        self.sensor_section.set_modified(sensor_count)
        self.demosaic_section.set_modified(demosaic_count)
        self.local_section.set_modified(len(cfg.local.masks))
        self.finish_section.set_modified(finish_count)
        for header in self.tab_headers:
            header.refresh()
        self.modified_synced.emit()

    def _sync_tool_buttons(self) -> None:
        """Updates toggle button states to match active_tool."""
        self.geometry_sidebar.sync_ui()
        self.local_sidebar.sync_ui()
        self.process_sidebar.sync_ui()
        # Retouch's heal and scratch toggles and the color sidebar's WB picker follow active_tool too.
        self.retouch_sidebar.sync_ui()
        self.leak_sidebar.sync_ui()
        self.color_sidebar.sync_ui()
