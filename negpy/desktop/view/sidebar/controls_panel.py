from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
)
from PyQt6.QtCore import QTimer, pyqtSignal

from negpy.desktop.controller import AppController
from negpy.desktop.view.shortcut_registry import tooltip_with_shortcut
from negpy.desktop.view.styles.templates import hint_label, set_hint_kind, wrap_tooltip
from negpy.desktop.view.widgets.collapsible import NO_ROLL_SCOPE_HINT, CollapsibleSection, make_section
from negpy.desktop.view.widgets.charts import MiniHistogramWidget, MiniRGBHistogramWidget
from negpy.desktop.view.styles.theme import THEME
from negpy.features.lab.models import LabConfig
from negpy.features.altprocess.models import AltProcessConfig
from negpy.features.toning.models import ToningConfig
from negpy.features.process.models import auto_meter_for_positive_source, cast_removal_for_mode
from negpy.features.finish.models import FinishConfig
from negpy.features.flatfield.models import FlatFieldConfig
from negpy.kernel.system.config import DEFAULT_WORKSPACE_CONFIG
from negpy.services.assets.rolls import ROLL_DEFAULT_FIELDS
from negpy.desktop.settings_catalog import rows_for_fields, rows_for_section, selected_flat_dict
from negpy.desktop.view.widgets.granular_settings_dialog import open_apply_dialog
from negpy.desktop.view.widgets.tab_header import TabHeader

# Sidebar Components
from negpy.desktop.view.sidebar.presets import PresetsSidebar
from negpy.desktop.view.sidebar.flatfield import FlatFieldSidebar
from negpy.desktop.view.sidebar.process import ProcessSidebar
from negpy.desktop.view.sidebar.roll import RollAnalysisSidebar
from negpy.desktop.view.sidebar.demosaic import DemosaicSidebar
from negpy.desktop.view.sidebar.sensor import SensorSidebar
from negpy.desktop.view.sidebar.color import ColorSidebar
from negpy.desktop.view.sidebar.tone import ToneSidebar
from negpy.desktop.view.sidebar.geometry import GeometrySidebar
from negpy.desktop.view.sidebar.autocrop import AutocropSidebar
from negpy.desktop.view.sidebar.trichrome import TrichromeSidebar
from negpy.desktop.view.sidebar.half_frame import HalfFrameSidebar
from negpy.desktop.view.sidebar.lens import LensSidebar
from negpy.desktop.view.sidebar.lab import LabSidebar
from negpy.desktop.view.sidebar.altprocess import AltProcessSidebar
from negpy.desktop.view.sidebar.toning import ToningSidebar
from negpy.desktop.view.sidebar.retouch import RetouchSidebar
from negpy.desktop.view.sidebar.local import LocalSidebar
from negpy.desktop.view.sidebar.finish import FinishSidebar

# Exposure field partitions: the Filtration and Tone sections split ExposureConfig, for
# both per-section modified counts and scoped resets. render_intent is in neither, since
# it is flat-master output.
_COLOR_FIELDS = (
    "wb_cyan",
    "wb_magenta",
    "wb_yellow",
    "shadow_cyan",
    "shadow_magenta",
    "shadow_yellow",
    "highlight_cyan",
    "highlight_magenta",
    "highlight_yellow",
    "cast_removal_strength",
)
_DEMOSAIC_FIELDS = (
    "demosaic_preview",
    "demosaic_export",
)
# GeometryConfig is split across three cards. The rect auto crop resolves, the rotation
# and the easel movements are this frame's own placement and stay on Geometry; what the
# detector looks for and how the scanning lens bends are the roll's.
_AUTOCROP_FIELDS = (
    "autocrop_mode",
    "autocrop_offset",
    "autocrop_rebate_trim",
    "autocrop_ratio",
)
_LENS_FIELDS = (
    "distortion_k1",
    "lens_distortion_from_metadata",
    "lens_ca_from_metadata",
)
_GEOMETRY_FIELDS = (
    "rotation",
    "fine_rotation",
    "flip_horizontal",
    "flip_vertical",
    "converge_v",
    "converge_h",
    "crop_to_valid",
    "crop_rect",
    "crop_from_auto",
    "crop_detect_key",
)
_SENSOR_FIELDS = (
    "linear_raw",
    "sensor_profile",
    "crosstalk_profile",
    "crosstalk_strength",
    "hue_trim",
)
# ProcessConfig is split across four cards. Each tuple is both the card's reset scope and
# its modified count, so a field is resettable from the one card that counts it.
# locked_floors/locked_ceils are in none: they are Batch Analysis's measured result.
_FILM_FIELDS = (
    "process_mode",
    "positive_source",
)
_NORMALIZATION_FIELDS = (
    "analysis_buffer",
    "analysis_rect",
    "lock_bounds",
    "luma_range_clip",
    "color_range_clip",
    "e6_normalize",
    "use_luma_average",
    "use_color_average",
)
# Normalization's White/Black Point block: counted and reset with that card, never a roll default.
_WHITE_BLACK_POINT_FIELDS = (
    "white_point_offset",
    "black_point_offset",
    "white_point_trim_red",
    "white_point_trim_green",
    "white_point_trim_blue",
    "black_point_trim_red",
    "black_point_trim_green",
    "black_point_trim_blue",
)
_TONE_FIELDS = (
    "density",
    "grade",
    "grade_trim_red",
    "grade_trim_green",
    "grade_trim_blue",
    "paper_black",
    "shadow_density",
    "highlight_density",
    "shadow_grade",
    "highlight_grade",
    "shadow_grade_trim_red",
    "shadow_grade_trim_green",
    "shadow_grade_trim_blue",
    "highlight_grade_trim_red",
    "highlight_grade_trim_green",
    "highlight_grade_trim_blue",
    "paper_dmin",
    "auto_exposure",
    "auto_normalize_contrast",
    "paper_profile",
    "midtone_gamma",
    "midtone_gamma_trim_red",
    "midtone_gamma_trim_green",
    "midtone_gamma_trim_blue",
    "toe",
    "toe_width",
    "toe_trim_red",
    "toe_trim_green",
    "toe_trim_blue",
    "toe_width_trim_red",
    "toe_width_trim_green",
    "toe_width_trim_blue",
    "shoulder",
    "shoulder_width",
    "shoulder_trim_red",
    "shoulder_trim_green",
    "shoulder_trim_blue",
    "shoulder_width_trim_red",
    "shoulder_width_trim_green",
    "shoulder_width_trim_blue",
    "dye_separation",
    "dye_separation_trim_red",
    "dye_separation_trim_green",
    "dye_separation_trim_blue",
    "separation_damping",
    "contrast_mask",
    "mask_spacer",
)

# Constant frozen-dataclass defaults, built once rather than per resync. Exposure/process/
# geometry/config come from DEFAULT_WORKSPACE_CONFIG, not their own bare dataclass default:
# grade, crosstalk_strength and the autocrop fields are calibrated there (transfer_grade_ref
# and friends), which is also what an untouched or reset file actually carries.
_DEFAULT_EXPOSURE = DEFAULT_WORKSPACE_CONFIG.exposure
_DEFAULT_LAB = LabConfig()
_DEFAULT_TONING = ToningConfig()
_DEFAULT_ALTPROC = AltProcessConfig()
_DEFAULT_GEOMETRY = DEFAULT_WORKSPACE_CONFIG.geometry
_DEFAULT_PROCESS = DEFAULT_WORKSPACE_CONFIG.process
_DEFAULT_FINISH = FinishConfig()
_DEFAULT_FLATFIELD = FlatFieldConfig()
_DEFAULT_CONFIG = DEFAULT_WORKSPACE_CONFIG

# Frame cards whose settings can be pushed to other frames, and the fields each owns. A
# card keyed by its own config section needs no tuple. Roll-tab cards drive roll defaults
# instead, and Dodge & Burn has no catalog row: a mask means nothing on the next frame.
_APPLY_FIELDS: dict[str, tuple | None] = {
    "geometry": _GEOMETRY_FIELDS,
    "color": _COLOR_FIELDS,
    "tone": _TONE_FIELDS,
    "lab": None,
    "altproc": None,
    "toning": None,
    "retouch": None,
    "finish": None,
}

_AUTO_METER_FIELDS = ("auto_exposure", "auto_normalize_contrast")


def _default_exposure_field(field: str, positive_source: bool, process_mode: str):
    """The value *field* defaults to on this frame. Auto Density/Auto Grade default
    differently on a Positive frame (auto_meter_for_positive_source) and Cast Removal
    differently per mode (cast_removal_for_mode); every other ExposureConfig field has
    one flat default."""
    default = getattr(_DEFAULT_EXPOSURE, field)
    if field in _AUTO_METER_FIELDS:
        return auto_meter_for_positive_source(positive_source, default)
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
            "Presets",
            "presets",
            self.presets_sidebar,
            icon_name="fa5s.magic",
        )

        self.flatfield_sidebar = FlatFieldSidebar(self.controller)
        self.flatfield_section = self._make_section(
            "Flat Field",
            "flatfield",
            self.flatfield_sidebar,
            icon_name="fa5s.adjust",
        )

        self.geometry_sidebar = GeometrySidebar(self.controller)
        self.geometry_section = self._make_section(
            "Geometry",
            "geometry",
            self.geometry_sidebar,
            icon_name="fa5s.crop",
        )

        self.autocrop_sidebar = AutocropSidebar(self.controller)
        self.autocrop_section = self._make_section(
            "Auto Crop",
            "autocrop",
            self.autocrop_sidebar,
            icon_name="fa5s.magic",
        )

        self.lens_sidebar = LensSidebar(self.controller)
        self.lens_section = self._make_section(
            "Lens Correction",
            "lens",
            self.lens_sidebar,
            icon_name="fa5s.circle-notch",
        )

        self.process_sidebar = ProcessSidebar(self.controller)
        # Always expanded (no chevron): the first choice of every edit, and the one
        # every other Roll-tab card's fields assume is already settled.
        self.film_section = self._make_section(
            "Film Mode",
            "film",
            self.process_sidebar.mode_bar,
            icon_name="mdi6.film",
            collapsible=False,
        )
        # How the files become frames, which is upstream of every rig card below.
        self.trichrome_sidebar = TrichromeSidebar(self.controller)
        self.trichrome_section = self._make_section(
            "Trichrome",
            "trichrome",
            self.trichrome_sidebar,
            icon_name="mdi.google-circles-communities",
        )

        self.half_frame_sidebar = HalfFrameSidebar(self.controller)
        self.half_frame_section = self._make_section(
            "Half Frame",
            "half_frame",
            self.half_frame_sidebar,
            icon_name="mdi.view-split-vertical",
        )

        self.roll_sidebar = RollAnalysisSidebar(self.controller)
        # Roll Analysis and Normalization are one job, getting a negative to a correctly
        # normalized positive, so they share a card: this frame's own analysis first, then
        # the roll picker it feeds, with Lock Bounds in the picker's button row because it
        # is about this frame's relationship to Batch Analysis.
        self.roll_sidebar.insert_lock_button(self.process_sidebar.lock_bounds_btn)
        normalization_body = QWidget()
        normalization_layout = QVBoxLayout(normalization_body)
        normalization_layout.setContentsMargins(0, 0, 0, 0)
        normalization_layout.setSpacing(4)
        normalization_layout.addWidget(self.process_sidebar.analysis_bar)
        normalization_layout.addWidget(self.roll_sidebar)
        normalization_layout.addWidget(self.process_sidebar)
        self.process_section = self._make_section(
            "Normalization",
            "process",
            normalization_body,
            icon_name="fa5s.cogs",
        )

        self.sensor_sidebar = SensorSidebar(self.controller)
        self.sensor_section = self._make_section(
            # Bare name: it holds the crosstalk matrix and Hue Trim as well as the sensor unmix. The
            # persisted "sensor" section key stays.
            "Calibration",
            "sensor",
            self.sensor_sidebar,
            icon_name="fa5s.vials",
        )

        self.demosaic_sidebar = DemosaicSidebar(self.controller)
        self.demosaic_section = self._make_section(
            "Demosaic",
            "demosaic",
            self.demosaic_sidebar,
            icon_name="mdi6.grid",
        )

        # One-line answer to "roll-wide or this frame's own": which Roll-tab cards
        # (if any) this frame overrides. RightPanel places
        # it above every Roll-tab card; _sync_roll_locks keeps it current.
        self.roll_override_summary = hint_label("", "muted")

        self.color_sidebar = ColorSidebar(self.controller)
        self.color_histogram = MiniRGBHistogramWidget()
        # "Filtration", not "Color", which names the Lab & Toning tab. The persisted "color"
        # section key stays.
        self.color_section = self._make_section(
            "Filtration",
            "color",
            self.color_sidebar,
            icon_name="fa5s.palette",
            background_widget=self.color_histogram,
        )

        self.tone_sidebar = ToneSidebar(self.controller)
        self.tone_histogram = MiniHistogramWidget()
        self.tone_section = self._make_section(
            "Tone",
            "tone",
            self.tone_sidebar,
            icon_name="fa5s.sun",
            background_widget=self.tone_histogram,
        )

        self.lab_sidebar = LabSidebar(self.controller)
        self.lab_section = self._make_section(
            "Lab",
            "lab",
            self.lab_sidebar,
            icon_name="fa5s.flask",
        )

        self.altproc_sidebar = AltProcessSidebar(self.controller)
        self.altproc_section = self._make_section(
            "Alternative Processes",
            "altproc",
            self.altproc_sidebar,
            icon_name="fa5s.fire",
        )

        self.toning_sidebar = ToningSidebar(self.controller)
        self.toning_section = self._make_section(
            "Toning",
            "toning",
            self.toning_sidebar,
            icon_name="fa5s.tint",
        )

        self.retouch_sidebar = RetouchSidebar(self.controller)
        self.retouch_section = self._make_section(
            "Retouch",
            "retouch",
            self.retouch_sidebar,
            icon_name="fa5s.brush",
        )

        self.local_sidebar = LocalSidebar(self.controller)
        self.local_section = self._make_section(
            "Dodge & Burn",
            "local",
            self.local_sidebar,
            icon_name="fa5s.adjust",
        )

        self.finish_sidebar = FinishSidebar(self.controller)
        self.finish_section = self._make_section(
            "Finishing",
            "finish",
            self.finish_sidebar,
            icon_name="fa5s.paint-brush",
        )

        # Group the sections into workflow pages (each becomes an icon tab in RightPanel). Calibration,
        # Demosaic, Roll Analysis, Normalization and Presets are roll-wide facts, not per-frame edits --
        # RightPanel builds them into its own top-level Roll tab instead of a page here.
        groups = [
            (
                "geometry",
                "fa5s.crop",
                "Geometry",
                "Geometry",
                [self.geometry_section],
                ["geometry_section"],
            ),
            (
                "tone",
                "fa5s.sun",
                "Exposure — Filtration, Tone, Dodge & Burn",
                "Exposure",
                [self.color_section, self.tone_section, self.local_section],
                ["color_section", "tone_section", "local_section"],
            ),
            (
                "color",
                "fa5s.flask",
                "Lab & Toning",
                "Lab & Toning",
                [self.lab_section, self.altproc_section, self.toning_section],
                ["lab_section", "altproc_section", "toning_section"],
            ),
            (
                "finish",
                "fa5s.brush",
                "Finish — Retouch, Finishing",
                "Finish",
                [self.retouch_section, self.finish_section],
                ["retouch_section", "finish_section"],
            ),
        ]

        self.pages = []
        self.tab_headers: list[TabHeader] = []
        for key, icon_name, tooltip, title, sections, section_attrs in groups:
            page = QWidget()
            page_layout = QVBoxLayout(page)
            page_layout.setContentsMargins(0, 0, 0, 0)
            page_layout.setSpacing(8)
            # A one-card tab has no header: that card's own is already the whole tab's.
            header = (
                self._make_tab_header(title, sections, [a.removesuffix("_section") for a in section_attrs]) if len(sections) > 1 else None
            )
            if header is not None:
                page_layout.addWidget(header)
            for section in sections:
                page_layout.addWidget(section)
            page_layout.addStretch(1)
            self.pages.append(
                {
                    "key": key,
                    "icon_name": icon_name,
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
        self.tab_headers.append(header)
        return header

    def _card_rows(self, card_key: str) -> list:
        """A frame card's catalog rows: its own field tuple, or its whole config section."""
        fields = _APPLY_FIELDS.get(card_key, ())
        if fields is None:
            return rows_for_section(card_key)
        return rows_for_fields(fields) if fields else []

    def _apply_tab(self, card_keys: tuple) -> None:
        """Every card on the tab in one picker. A whole-roll apply is recorded per card,
        the same record a card's own Roll button writes, so each header still reads back
        what went out."""
        live = [k for k in card_keys if not getattr(self, f"{k}_section").isHidden()]
        rows_by_card = {k: self._card_rows(k) for k in live}
        rows = list(dict.fromkeys(r for card_rows in rows_by_card.values() for r in card_rows))
        if not rows:
            return
        applied = open_apply_dialog(self, self.controller.session, rows=rows)
        if not applied or applied[1] != "roll":
            return
        for key, card_rows in rows_by_card.items():
            own = [r for r in applied[0] if r in card_rows]
            if own:
                self.controller.record_section_push(key, selected_flat_dict(self.controller.state.config, own))

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

        self.color_section.reset_requested.connect(lambda: self._reset_exposure_fields(_COLOR_FIELDS))
        self.tone_section.reset_requested.connect(self._reset_tone_fields)
        self.lab_section.reset_requested.connect(lambda: self.controller.session.reset_section("lab"))
        self.altproc_section.reset_requested.connect(lambda: self.controller.session.reset_section("altproc"))
        self.toning_section.reset_requested.connect(lambda: self.controller.session.reset_section("toning"))
        self.geometry_section.reset_requested.connect(self._reset_geometry_fields)
        self.autocrop_section.reset_requested.connect(lambda: self._reset_card_fields("autocrop"))
        self.lens_section.reset_requested.connect(lambda: self._reset_card_fields("lens"))
        self.process_section.reset_requested.connect(lambda: self._reset_process_fields(_NORMALIZATION_FIELDS + _WHITE_BLACK_POINT_FIELDS))
        self.retouch_section.reset_requested.connect(lambda: self.controller.session.reset_section("retouch"))
        self.local_section.reset_requested.connect(lambda: self.controller.session.reset_section("local"))
        self.finish_section.reset_requested.connect(lambda: self.controller.session.reset_section("finish"))
        self.film_section.reset_requested.connect(self._reset_film_fields)
        self.sensor_section.reset_requested.connect(self._reset_sensor_fields)
        self.demosaic_section.reset_requested.connect(lambda: self._reset_process_fields(_DEMOSAIC_FIELDS))
        self.flatfield_section.reset_requested.connect(self._reset_flatfield)

        for key, section in self._roll_sections() + self._frame_sections():
            section.scope_selected.connect(lambda scope, k=key: self._on_scope_selected(k, scope))

    def apply_shortcut_tooltips(self) -> None:
        """Single source for every shortcut-bearing widget tooltip — re-run on each
        rebind to re-render the key chips. Don't set these locally in the sidebars:
        this pass overwrites them."""
        col = self.color_sidebar
        for btn, action_id in (
            (self.retouch_sidebar.auto_dust_btn, "toggle_optical_removal"),
            (self.retouch_sidebar.right_click_btn, "toggle_right_click_excludes"),
            (self.retouch_sidebar.ir_dust_btn, "toggle_ir_removal"),
            (self.flatfield_sidebar.enable_btn, "toggle_flat_field"),
            (self.autocrop_sidebar.auto_crop_all_btn, "batch_autocrop"),
            (self.tone_sidebar.auto_density_btn, "toggle_auto_density"),
            (self.tone_sidebar.auto_grade_btn, "toggle_auto_grade"),
            (self.presets_sidebar.apply_btn, "preset_apply"),
            (self.presets_sidebar.save_btn, "preset_save"),
        ):
            btn.setToolTip(wrap_tooltip(tooltip_with_shortcut(btn.plain_tooltip, action_id)))
        exp = self.tone_sidebar
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
                "Apply embedded scanning-lens distortion correction. Replaces manual distortion.",
                "lens_distortion_from_metadata",
            )
        )
        lens.metadata_ca_btn.setToolTip(
            tooltip_with_shortcut(
                "Apply embedded lateral chromatic aberration correction. Can be used with manual distortion.",
                "lens_ca_from_metadata",
            )
        )

        col.pick_wb_btn.setToolTip(
            tooltip_with_shortcut(
                "Activate eyedropper — click a neutral gray pixel to auto-compute white balance offsets",
                "pick_wb",
            )
        )
        col.temp_slider.setToolTip(
            tooltip_with_shortcut(
                "Color temperature lever over the Global Magenta/Yellow white balance — moving it "
                "steers M/Y along the warm-cool axis (tint preserved); moving M/Y updates the readout. "
                "Mired-linear travel, warm right; Kelvin is nominal",
                ["temp_warm", "temp_cool"],
            )
        )
        col.cyan_slider.setToolTip(
            tooltip_with_shortcut(
                "Cyan↔Red white balance shift; negative = cyan, positive = red. Applies to selected region (Global/Shadows/Highlights)",
                ["cyan_inc", "cyan_dec"],
            )
        )
        col.magenta_slider.setToolTip(
            tooltip_with_shortcut(
                "Magenta↔Green white balance shift. Applies to selected region (Global/Shadows/Highlights)",
                ["magenta_up", "magenta_down"],
            )
        )
        col.yellow_slider.setToolTip(
            tooltip_with_shortcut(
                "Yellow↔Blue white balance shift. Applies to selected region (Global/Shadows/Highlights)",
                ["yellow_up", "yellow_down"],
            )
        )
        exp.density_slider.setToolTip(
            tooltip_with_shortcut(
                "Overall print density — simulates enlarger exposure time. Lower = brighter, higher = darker",
                ["density_up", "density_down"],
            )
        )
        exp.grade_slider.setToolTip(
            tooltip_with_shortcut(
                "Contrast (ISO R paper exposure range): R180 = very soft, R50 = very hard; R110 ≈ grade 2 paper",
                ["grade_up", "grade_down"],
            )
        )
        exp.toe_slider.setToolTip(
            tooltip_with_shortcut(
                "Shadow toe: positive lifts shadows for a gentle film toe; negative deepens blacks",
                ["toe_inc", "toe_dec"],
            )
        )
        exp.toe_w_slider.setToolTip(
            tooltip_with_shortcut(
                "How broadly the shadow toe transition spreads into the midtones",
                ["toe_width_inc", "toe_width_dec"],
            )
        )
        exp.sh_slider.setToolTip(
            tooltip_with_shortcut(
                "Highlight shoulder: positive compresses highlights (film roll-off); negative extends them and risks clipping",
                ["shoulder_inc", "shoulder_dec"],
            )
        )
        exp.sh_w_slider.setToolTip(
            tooltip_with_shortcut(
                "How broadly the highlight shoulder transition spreads into the midtones",
                ["shoulder_width_inc", "shoulder_width_dec"],
            )
        )
        exp.midtone_gamma_slider.setToolTip(
            tooltip_with_shortcut(
                "Snap — paper midtone gamma trim: steepens or flattens the S-curve around the reference "
                "tone; paper white/black stay put. In R/G/B mode: this layer's Snap trim",
                ["snap_inc", "snap_dec"],
            )
        )
        exp.shadow_density_slider.setToolTip(
            tooltip_with_shortcut(
                "Shadow zone density (ΔD): weighted to the deep shadows, bounded by paper black. "
                "Positive darkens shadows; negative lifts them",
                ["shadow_density_inc", "shadow_density_dec"],
            )
        )
        exp.highlight_density_slider.setToolTip(
            tooltip_with_shortcut(
                "Highlight zone density (ΔD): weighted to the highlights, bounded by paper white. "
                "Positive burns highlights in; negative bleaches them",
                ["highlight_density_inc", "highlight_density_dec"],
            )
        )
        exp.shadow_grade_slider.setToolTip(
            tooltip_with_shortcut(
                "Split grade — shadow zone contrast trim (ISO-R): rotates the curve locally in the deep "
                "shadows. In R/G/B mode: this layer's shadow-grade trim",
                ["shadow_grade_inc", "shadow_grade_dec"],
            )
        )
        exp.highlight_grade_slider.setToolTip(
            tooltip_with_shortcut(
                "Split grade — highlight zone contrast trim (ISO-R): rotates the curve locally in the "
                "highlights. In R/G/B mode: this layer's highlight-grade trim",
                ["highlight_grade_inc", "highlight_grade_dec"],
            )
        )

        geo.manual_crop_btn.setToolTip(
            tooltip_with_shortcut(
                "Draw a crop rectangle on the canvas — drag to set, constrained by the current aspect ratio",
                "manual_crop",
            )
        )
        geo.straighten_btn.setToolTip(
            tooltip_with_shortcut(
                "Straighten with a reference line — draw along the horizon or a vertical edge "
                "(a building, a door frame) and the image rotates to make it level or plumb. "
                "Applies once per line; Esc cancels an in-progress line",
                "straighten",
            )
        )
        crop.offset_slider.setToolTip(
            tooltip_with_shortcut(
                "Insets the auto-crop border from the detected film edge. Positive = trim more; negative = bleed outside",
                ["offset_inc", "offset_dec"],
            )
        )
        geo.fine_rot_slider.setToolTip(
            tooltip_with_shortcut(
                "Sub-degree rotation correction for tilted scans: positive turns clockwise, negative counter-clockwise. "
                "For quick rotation, drag the round handles outside the crop box in the Crop tool",
                ["fine_rot_inc", "fine_rot_dec"],
            )
        )

        proc.lock_bounds_btn.setToolTip(
            tooltip_with_shortcut(
                "Lock Bounds — freeze normalization bounds so crop and analysis sliders no longer re-analyze the frame",
                "lock_bounds_toggle",
            )
        )
        proc.analysis_buffer_slider.setToolTip(
            tooltip_with_shortcut(
                "Insets the analysis window from the frame edge so rebate, sprocket holes, and scanner borders don't skew black/white-point detection",
                ["analysis_buffer_inc", "analysis_buffer_dec"],
            )
        )
        proc.luma_range_clip_slider.setToolTip(
            tooltip_with_shortcut(
                "Tonal-range normalization (black/white-point span). Neutral already applies a small robust clip. "
                "Positive: clips the top/bottom for more aggressive highlight/shadow recovery. "
                "Negative: outward headroom — lifted blacks / unclipped highlights for a gentler stretch",
                ["luma_range_clip_inc", "luma_range_clip_dec"],
            )
        )
        proc.color_range_clip_slider.setToolTip(
            tooltip_with_shortcut(
                "Per-channel color-balance clip percentile (orange-mask cast removal), independent of tonal range. "
                "Neutral: P1 clip. Negative: gentler, samples nearer the extremes. Positive: tighter channel balance",
                ["color_range_clip_inc", "color_range_clip_dec"],
            )
        )
        proc.white_point_slider.setToolTip(
            tooltip_with_shortcut(
                "Shifts the normalization floor (scan white point). Positive = brighter; negative = pull highlights "
                "back. In R/G/B mode: this layer's trim — per-layer film-base correction",
                ["white_point_inc", "white_point_dec"],
            )
        )
        proc.black_point_slider.setToolTip(
            tooltip_with_shortcut(
                "Shifts the normalization ceiling (scan black point). Positive = lifted blacks; negative = deeper "
                "blacks. In R/G/B mode: this layer's trim — per-layer Dmax correction",
                ["black_point_inc", "black_point_dec"],
            )
        )

        sen.crosstalk_strength_slider.setToolTip(
            tooltip_with_shortcut(
                "Channel unmix on the raw negative densities — how much of the matrix to apply. 1.0 = each "
                "channel's leak fully subtracted from the others; 0 = scanned densities untouched. The leak "
                "comes from the film's dyes, your light's spectrum and your sensor's filters together, so "
                "tune this per scanning setup rather than per stock. Re-run Batch Analysis after changing it",
                ["separation_inc", "separation_dec"],
            )
        )
        lab.chroma_denoise_slider.setToolTip(
            tooltip_with_shortcut(
                "Chroma denoise in Lab space — smooths color noise while preserving luminance grain",
                ["chroma_denoise_inc", "chroma_denoise_dec"],
            )
        )
        lab.saturation_slider.setToolTip(
            tooltip_with_shortcut(
                "Linear chroma scale (CIELAB a*/b*) after the print is decoded — a retouching move, "
                "applied evenly to every tone. Dye Separation in Tone is the density-space equivalent: "
                "it works on the print's dye densities, so it stays in step with the paper and the curve. "
                "1.0 = unchanged, 0 = grayscale, 2.0 = double",
                ["saturation_inc", "saturation_dec"],
            )
        )
        lab.skin_protection_slider.setToolTip(
            "Holds skin-hued color under a chroma ceiling so faces don't go sunburnt — hue and lightness "
            "untouched, and chroma is only ever pulled down. Independent of Chroma: it also reins in skin "
            "that arrived over-saturated from the print curve. 0 = off, 1.0 = matte"
        )
        exp.dye_separation_slider.setToolTip(
            tooltip_with_shortcut(
                "Pushes density apart before decode. On a print, in the same matrix slot as the "
                "paper's own dye crosstalk — so it responds to the paper profile and eases off where the "
                "curve is already compressed at toe and shoulder, and takes per-layer R/G/B trims. On a "
                "slide with Normalize off, applied directly with no paper matrix or trims. Chroma in "
                "Color is the flat version: an even a*/b* scale after decode. 1.0 = off/identity",
                ["dye_separation_inc", "dye_separation_dec"],
            )
        )
        exp.separation_damping_slider.setToolTip(
            tooltip_with_shortcut(
                "Decides where Dye Separation's push lands instead of adding one of its own — at 0 every "
                "color gets the same push, at 1 muted color takes it all while color that is already "
                "saturated gets the opposite, so a hard push adds color where there was none instead of "
                "flattening the strongest colors. Dead at Dye Separation 1.0. 0 = flat",
                ["separation_damping_inc", "separation_damping_dec"],
            )
        )
        lab.clahe_slider.setToolTip(
            tooltip_with_shortcut(
                "Local contrast (CLAHE) without blowing global highlights or crushing shadows. Use sparingly — near 1.0 can look cartoonish",
                ["clahe_inc", "clahe_dec"],
            )
        )
        lab.sharpen_slider.setToolTip(
            tooltip_with_shortcut(
                "L-channel unsharp mask with halo suppression — crisps detail without bright edge outlines or color fringing",
                ["sharpen_inc", "sharpen_dec"],
            )
        )
        lab.sharpen_method_combo.setToolTip(
            "Unsharp Mask boosts edge contrast; Deconvolution (Richardson–Lucy) reverses the scanner's optical blur — set Radius to the blur width of the scan"
        )
        lab.sharpen_radius_slider.setToolTip(
            "Blur radius in pixels — small for fine grain and detail, larger for smoother films and soft scans"
        )
        lab.sharpen_masking_slider.setToolTip(
            "Restricts sharpening to edges — higher values protect flat areas (sky, skin, grain) from being crisped"
        )
        lab.glow_slider.setToolTip(
            tooltip_with_shortcut(
                "Lens bloom — bright highlights scatter equally across all channels, softening edges and adding a dreamy quality",
                ["glow_inc", "glow_dec"],
            )
        )
        lab.halation_slider.setToolTip(
            tooltip_with_shortcut(
                "Simulates the red glow from light scattering back through the film base. Affects highlights only, strongly red-dominant",
                ["halation_inc", "halation_dec"],
            )
        )

        ret.pick_dust_btn.setToolTip(
            tooltip_with_shortcut(
                "Toggle manual heal brush — click dust spots in the preview to paint them out one at a time. "
                "Right-click an existing heal overlay to delete it",
                "pick_dust",
            )
        )
        ret.threshold_slider.setToolTip(
            tooltip_with_shortcut(
                "Brightness delta above which a pixel is classified as dust. Lower = catch more (risk false positives on real detail)",
                ["threshold_inc", "threshold_dec"],
            )
        )
        ret.auto_size_slider.setToolTip(
            tooltip_with_shortcut(
                "Maximum radius of auto-detected dust spots. Larger catches bigger blobs but risks eating fine detail",
                ["auto_size_inc", "auto_size_dec"],
            )
        )
        ret.manual_size_slider.setToolTip(
            tooltip_with_shortcut(
                "Radius of the manual heal brush",
                ["manual_size_inc", "manual_size_dec"],
            )
        )

        ton.selenium_slider.setToolTip(
            tooltip_with_shortcut(
                "Simulates selenium toning — converts the densest silver first: deeper blacks, cool eggplant shadows. B&W Negative mode only",
                ["selenium_inc", "selenium_dec"],
            )
        )
        ton.sepia_slider.setToolTip(
            tooltip_with_shortcut(
                "Simulates sepia bleach-redevelop toning — warms the highlights first while shadows hold; "
                "partial strength gives the classic split-sepia look. B&W Negative mode only",
                ["sepia_inc", "sepia_dec"],
            )
        )
        ton.shadow_hue_slider.setToolTip(
            tooltip_with_shortcut(
                "Hue of the shadow split-tone color injection",
                ["shadow_hue_inc", "shadow_hue_dec"],
            )
        )
        ton.shadow_str_slider.setToolTip(
            tooltip_with_shortcut(
                "How strongly the shadow hue is mixed in",
                ["shadow_strength_inc", "shadow_strength_dec"],
            )
        )
        ton.highlight_hue_slider.setToolTip(
            tooltip_with_shortcut(
                "Hue of the highlight split-tone color injection",
                ["highlight_hue_inc", "highlight_hue_dec"],
            )
        )
        ton.highlight_str_slider.setToolTip(
            tooltip_with_shortcut(
                "How strongly the highlight hue is mixed in",
                ["highlight_strength_inc", "highlight_strength_dec"],
            )
        )

        fin.vignette_burn_slider.setToolTip(
            tooltip_with_shortcut(
                "Edge exposure in stops: positive = burn in the edges (darken); negative = hold back (lighten). 0 = off",
                ["vignette_str_inc", "vignette_str_dec"],
            )
        )
        fin.vignette_size_slider.setToolTip(
            tooltip_with_shortcut(
                "Falloff radius: smaller = tight corner effect; larger = burn spreads well into the frame",
                ["vignette_size_inc", "vignette_size_dec"],
            )
        )
        fin.vignette_roundness_slider.setToolTip(
            "Falloff shape: 0 = radial (lens-like), 1 = rectangular card burn following the print edges"
        )
        fin.border_slider.setToolTip(
            tooltip_with_shortcut(
                "Border thickness as a fraction of the image dimensions. Zero = no border",
                ["border_size_inc", "border_size_dec"],
            )
        )

    _DIPTYCH_HINT = "Diptych — the edits live on the halves. Turn Half Frame Mode on to edit either one."

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
        self.color_section.setVisible(self.controller.state.config.process.process_mode != ProcessMode.BW)
        self.process_sidebar.sync_ui()
        self.roll_sidebar.sync_ui()
        self.color_sidebar.sync_ui()
        self.tone_sidebar.sync_ui()
        self.geometry_sidebar.sync_ui()
        self.lab_sidebar.sync_ui()
        self.altproc_sidebar.sync_ui()
        self.toning_sidebar.sync_ui()
        self.retouch_sidebar.sync_ui()
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
        self._sync_modified_dots()
        self._sync_scope_buttons()

    _ROLL_CARD_LABELS = AppController._ROLL_CARD_LABELS

    def _roll_sections(self) -> tuple:
        return (
            ("film", self.film_section),
            ("sensor", self.sensor_section),
            ("demosaic", self.demosaic_section),
            ("process", self.process_section),
            ("autocrop", self.autocrop_section),
            ("lens", self.lens_section),
            ("flatfield", self.flatfield_section),
        )

    def _sync_scope_buttons(self) -> None:
        """Each card's Frame/Roll pair, and roll_override_summary's one-line answer to
        "roll-wide or this frame's own" alongside it. A Roll-tab card reads its own lock;
        a frame card is always Frame, since a sync is a copy rather than a binding.

        Frames that are not one roll (a library search's results, several folders at once)
        read Frame with Roll disabled: every value there is the frame's own, since no roll
        spans them to hold a shared one. Save as Roll gives them one."""
        has_roll = self.controller.state.active_roll_id is not None
        overridden = []
        for card_key, section in self._roll_sections():
            locked = self.controller.roll_card_locked(card_key)
            label = self._ROLL_CARD_LABELS[card_key]
            section.set_scope_buttons(
                True,
                "frame" if locked or not has_roll else "roll",
                roll_tooltip=(
                    f"{label} follows the roll — click to give the roll this frame's value" if has_roll else NO_ROLL_SCOPE_HINT
                ),
                frame_tooltip=(
                    f"{label} follows this frame alone — click to rejoin the roll"
                    if has_roll
                    else f"{label} is this frame's own"
                ),
                roll_enabled=has_roll,
            )
            if locked:
                overridden.append(label)

        for key, section in self._frame_sections():
            section.set_scope_buttons(
                True,
                self.controller.frame_section_scope(key),
                roll_tooltip="" if has_roll else NO_ROLL_SCOPE_HINT,
                roll_enabled=has_roll,
            )

        if overridden:
            set_hint_kind(self.roll_override_summary, "warning")
            self.roll_override_summary.setText(f"This frame overrides: {', '.join(overridden)}")
        else:
            self.roll_override_summary.setText("")

    def _frame_sections(self) -> tuple:
        return tuple((key, getattr(self, f"{key}_section")) for key in _APPLY_FIELDS)

    def _on_scope_selected(self, key: str, scope: str) -> None:
        """Roll on a Roll-tab card pushes that card out; on a frame card it opens the
        picker over that card's own settings. Frame on a Roll-tab card locks it here;
        a frame card is already there, so the pair's own click guard swallows it."""
        if key in dict(self._roll_sections()):
            self.controller.set_card_scope(key, scope)
            self._sync_scope_buttons()
            return
        fields = _APPLY_FIELDS[key]
        rows = rows_for_fields(fields) if fields else rows_for_section(key)
        applied = open_apply_dialog(self, self.controller.session, rows=rows)
        if applied and applied[1] == "roll":
            self.controller.record_section_push(key, selected_flat_dict(self.controller.state.config, applied[0]))

    def _update_histogram(self) -> None:
        """Repaint only when the render produced a new buffer."""
        buf = self.controller.state.last_metrics.get("histogram_raw")
        if buf is self._last_histogram_buf:
            return
        self._last_histogram_buf = buf
        self.tone_histogram.update_data(buf)
        self.color_histogram.update_data(buf)

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
        self._reset_exposure_fields(_TONE_FIELDS)

    def _reset_process_fields(self, fields) -> None:
        """Calibration, Demosaic and Normalization all live on ProcessConfig, so each
        reset is scoped to its own fields -- a plain session.reset_section("process")
        would reset all three cards (and Film Mode, and Positive) at once. apply_config,
        not update_config: some of these fields are a decode or a source bake."""
        from dataclasses import replace

        cfg = self.controller.state.config
        new_proc = replace(cfg.process, **{f: getattr(_DEFAULT_PROCESS, f) for f in fields})
        self.controller.apply_config(replace(cfg, process=new_proc), persist=True)

    def _reset_flatfield(self) -> None:
        self._reset_card_fields("flatfield")

    def _reset_geometry_fields(self) -> None:
        """Geometry's own fields alone: a plain session.reset_section("geometry") would
        take Auto Crop and Lens Correction with it."""
        from dataclasses import replace

        cfg = self.controller.state.config
        new_geo = replace(cfg.geometry, **{f: getattr(_DEFAULT_GEOMETRY, f) for f in _GEOMETRY_FIELDS})
        self.controller.apply_config(replace(cfg, geometry=new_geo), persist=True)

    def _reset_card_fields(self, card_key: str) -> None:
        """Reset one roll card through set_roll_default, so the reset follows the roll
        or locks away from it exactly as an edit by hand would. Ratio keeps its own
        entry point, which reshapes a drawn crop rather than leaving it stale."""
        section, fields = ROLL_DEFAULT_FIELDS[card_key]
        default = getattr(_DEFAULT_CONFIG, section)
        if "autocrop_ratio" in fields:
            self.controller.set_crop_ratio(default.autocrop_ratio)
        self.controller.set_roll_default(card_key, **{f: getattr(default, f) for f in fields if f != "autocrop_ratio"})

    def _reset_exposure_fields(self, fields) -> None:
        """Reset only the given ExposureConfig fields to defaults (scoped section reset).
        A reset means the value the frame's own defaulting rules would carry, not the
        flat ExposureConfig default (_default_exposure_field)."""
        from dataclasses import replace

        cfg = self.controller.state.config
        exp = cfg.exposure
        defaults = {f: _default_exposure_field(f, cfg.process.positive_source, cfg.process.process_mode) for f in fields}
        new_exp = replace(exp, **defaults)
        new_config = replace(cfg, exposure=new_exp)
        self.controller.session.update_config(new_config, persist=True)

    def _sync_modified_dots(self) -> None:
        """Update modified-indicator dots on collapsible section headers."""
        cfg = self.controller.state.config
        _exp = _DEFAULT_EXPOSURE
        _lab = _DEFAULT_LAB
        _alt = _DEFAULT_ALTPROC
        _ton = _DEFAULT_TONING
        _geo = _DEFAULT_GEOMETRY
        _proc = _DEFAULT_PROCESS

        exp = cfg.exposure
        positive_source = cfg.process.positive_source
        mode = cfg.process.process_mode
        color_count = sum(getattr(exp, f) != _default_exposure_field(f, positive_source, mode) for f in _COLOR_FIELDS)
        tone_count = sum(getattr(exp, f) != _default_exposure_field(f, positive_source, mode) for f in _TONE_FIELDS)

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
        # crop_rect counts as set rather than as different: its default is None, and a
        # resolved auto rect is not an edit the way a hand-drawn one is.
        geometry_count = sum(getattr(geo, f) != getattr(_geo, f) for f in _GEOMETRY_FIELDS if f != "crop_rect")
        geometry_count += geo.crop_rect is not None
        autocrop_count = sum(getattr(geo, f) != getattr(_geo, f) for f in _AUTOCROP_FIELDS)
        lens_count = sum(getattr(geo, f) != getattr(_geo, f) for f in _LENS_FIELDS)

        proc = cfg.process
        film_count = sum(getattr(proc, f) != getattr(_proc, f) for f in _FILM_FIELDS)
        process_count = sum(getattr(proc, f) != getattr(_proc, f) for f in _NORMALIZATION_FIELDS + _WHITE_BLACK_POINT_FIELDS)
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
        self.lab_section.set_modified(lab_count)
        self.altproc_section.set_modified(altproc_count)
        self.toning_section.set_modified(toning_count)
        self.geometry_section.set_modified(geometry_count)
        self.autocrop_section.set_modified(autocrop_count)
        self.lens_section.set_modified(lens_count)
        # The picked Roll Baseline counts against Normalization, the card it sits on.
        self.process_section.set_modified(process_count + (proc.roll_name is not None))
        self.retouch_section.set_modified(retouch_count)
        # Presets and the two Scan sections stay out: they own no WorkspaceConfig fields.
        self.sensor_section.set_modified(sensor_count)
        self.demosaic_section.set_modified(demosaic_count)
        self.flatfield_section.set_modified(flatfield_count)
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
        # Retouch hosts two tool toggles, heal and scratch. Without this sync, activating one
        # left the other highlighted as if both were live. The color sidebar's WB picker had the
        # same latent stale-check bug.
        self.retouch_sidebar.sync_ui()
        self.color_sidebar.sync_ui()
