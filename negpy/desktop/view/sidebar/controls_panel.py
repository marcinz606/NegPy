from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
)
from PyQt6.QtCore import Qt, QTimer
import qtawesome as qta

from negpy.desktop.controller import AppController
from negpy.desktop.view.shortcut_registry import tooltip_with_shortcut
from negpy.desktop.view.widgets.collapsible import CollapsibleSection
from negpy.desktop.view.widgets.charts import MiniHistogramWidget
from negpy.desktop.view.styles.theme import THEME
from negpy.features.exposure.models import ExposureConfig
from negpy.features.lab.models import LabConfig
from negpy.features.toning.models import ToningConfig
from negpy.features.geometry.models import GeometryConfig
from negpy.features.process.models import ProcessConfig

# Sidebar Components
from negpy.desktop.view.sidebar.presets import PresetsSidebar
from negpy.desktop.view.sidebar.process import ProcessSidebar
from negpy.desktop.view.sidebar.exposure import ExposureSidebar
from negpy.desktop.view.sidebar.geometry import GeometrySidebar
from negpy.desktop.view.sidebar.lab import LabSidebar
from negpy.desktop.view.sidebar.toning import ToningSidebar
from negpy.desktop.view.sidebar.retouch import RetouchSidebar
from negpy.desktop.view.sidebar.icc import ICCSidebar
from negpy.desktop.view.sidebar.finish import FinishSidebar


class ControlsPanel(QWidget):
    """
    Right sidebar panel aggregating all tool controls (Exposure, Geometry, etc.).
    """

    def __init__(self, controller: AppController):
        super().__init__()
        self.controller = controller

        self._init_ui()
        self._connect_signals()

    def _init_ui(self) -> None:
        self.layout = QVBoxLayout(self)
        self.layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(8)

        icon_color = "#aaa"

        self.presets_sidebar = PresetsSidebar(self.controller)
        self._add_sidebar_section(
            "Presets",
            "presets",
            self.presets_sidebar,
            icon=qta.icon("fa5s.magic", color=icon_color),
        )

        self.geometry_sidebar = GeometrySidebar(self.controller)
        self.geometry_section = self._add_sidebar_section(
            "Geometry",
            "geometry",
            self.geometry_sidebar,
            icon=qta.icon("fa5s.crop", color=icon_color),
        )

        self.process_sidebar = ProcessSidebar(self.controller)
        self.process_section = self._add_sidebar_section(
            "Process",
            "process",
            self.process_sidebar,
            icon=qta.icon("fa5s.cogs", color=icon_color),
        )

        self.exposure_sidebar = ExposureSidebar(self.controller)
        self.exposure_histogram = MiniHistogramWidget()
        self.exposure_section = self._add_sidebar_section(
            "Exposure",
            "exposure",
            self.exposure_sidebar,
            icon=qta.icon("fa5s.sun", color=icon_color),
            background_widget=self.exposure_histogram,
        )

        self.lab_sidebar = LabSidebar(self.controller)
        self.lab_section = self._add_sidebar_section(
            "Lab",
            "lab",
            self.lab_sidebar,
            icon=qta.icon("fa5s.flask", color=icon_color),
        )

        self.toning_sidebar = ToningSidebar(self.controller)
        self.toning_section = self._add_sidebar_section(
            "Toning",
            "toning",
            self.toning_sidebar,
            icon=qta.icon("fa5s.tint", color=icon_color),
        )

        self.retouch_sidebar = RetouchSidebar(self.controller)
        self.retouch_section = self._add_sidebar_section(
            "Retouch",
            "retouch",
            self.retouch_sidebar,
            icon=qta.icon("fa5s.brush", color=icon_color),
        )

        self.finish_sidebar = FinishSidebar(self.controller)
        self.finish_section = self._add_sidebar_section(
            "Finishing",
            "finish",
            self.finish_sidebar,
            icon=qta.icon("fa5s.paint-brush", color=icon_color),
        )

        self.icc_sidebar = ICCSidebar(self.controller)
        self._add_sidebar_section(
            "ICC",
            "icc",
            self.icc_sidebar,
            icon=qta.icon("fa5s.eye", color=icon_color),
        )

    def _add_sidebar_section(
        self,
        title: str,
        key: str,
        widget: QWidget,
        icon=None,
        background_widget=None,
    ) -> CollapsibleSection:
        """Helper to create and add a collapsible section. Returns the section widget."""
        repo = self.controller.session.repo
        persisted = repo.get_global_setting(f"section_expanded_{key}")
        if persisted is not None:
            is_expanded = bool(persisted)
        else:
            is_expanded = THEME.sidebar_expanded_defaults.get(key, False)
            if key in ["process", "exposure", "geometry", "lab", "retouch", "export", "analysis"]:
                is_expanded = THEME.sidebar_expanded_defaults.get(key, True)

        section = CollapsibleSection(title, expanded=is_expanded, icon=icon, background_widget=background_widget)
        section.set_content(widget)
        self.layout.addWidget(section)

        section.expanded_changed.connect(lambda checked, k=key: repo.save_global_setting(f"section_expanded_{k}", checked))
        return section

    def _connect_signals(self) -> None:
        self._sync_debounce = QTimer()
        self._sync_debounce.setSingleShot(True)
        self._sync_debounce.setInterval(150)
        self._sync_debounce.timeout.connect(self._sync_all_sidebars)
        self.controller.config_updated.connect(self._sync_debounce.start)
        self.controller.tool_sync_requested.connect(self._sync_tool_buttons)

        self.exposure_section.reset_requested.connect(lambda: self.controller.session.reset_section("exposure"))
        self.lab_section.reset_requested.connect(lambda: self.controller.session.reset_section("lab"))
        self.toning_section.reset_requested.connect(lambda: self.controller.session.reset_section("toning"))
        self.geometry_section.reset_requested.connect(lambda: self.controller.session.reset_section("geometry"))
        self.process_section.reset_requested.connect(lambda: self.controller.session.reset_section("process"))
        self.retouch_section.reset_requested.connect(lambda: self.controller.session.reset_section("retouch"))
        self.finish_section.reset_requested.connect(lambda: self.controller.session.reset_section("finish"))

    def apply_shortcut_tooltips(self) -> None:
        exp = self.exposure_sidebar
        geo = self.geometry_sidebar
        lab = self.lab_sidebar
        proc = self.process_sidebar
        ret = self.retouch_sidebar
        ton = self.toning_sidebar
        fin = self.finish_sidebar

        exp.pick_wb_btn.setToolTip(
            tooltip_with_shortcut(
                "Activate eyedropper — click a neutral grey pixel to auto-compute white balance offsets",
                "pick_wb",
            )
        )
        exp.cyan_slider.setToolTip(
            tooltip_with_shortcut(
                "Cyan↔Red white balance shift; negative = cyan, positive = red. Applies to selected region (Global/Shadows/Highlights)",
                ["cyan_inc", "cyan_dec"],
            )
        )
        exp.magenta_slider.setToolTip(
            tooltip_with_shortcut(
                "Magenta↔Green white balance shift. Applies to selected region (Global/Shadows/Highlights)",
                ["magenta_up", "magenta_down"],
            )
        )
        exp.yellow_slider.setToolTip(
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
                "Contrast grade, like paper grades in the darkroom. 0 = very soft (flat), 5 = very hard (contrasty)",
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

        geo.manual_crop_btn.setToolTip(
            tooltip_with_shortcut(
                "Draw a crop rectangle on the canvas — drag to set, constrained by the current aspect ratio",
                "manual_crop",
            )
        )
        geo.offset_slider.setToolTip(
            tooltip_with_shortcut(
                "Insets the auto-crop border from the detected film edge. Positive = trim more; negative = bleed outside",
                ["offset_inc", "offset_dec"],
            )
        )
        geo.fine_rot_slider.setToolTip(
            tooltip_with_shortcut(
                "Sub-degree rotation correction for tilted scans, applied after auto-crop",
                ["fine_rot_inc", "fine_rot_dec"],
            )
        )

        proc.lock_bounds_btn.setToolTip(
            tooltip_with_shortcut(
                "Freeze normalization bounds — crop and analysis sliders no longer re-analyze the frame",
                "lock_bounds_toggle",
            )
        )
        proc.analysis_buffer_slider.setToolTip(
            tooltip_with_shortcut(
                "Insets the analysis window from the frame edge so rebate, sprocket holes, and scanner borders don't skew black/white-point detection",
                ["analysis_buffer_inc", "analysis_buffer_dec"],
            )
        )
        proc.drange_clip_slider.setToolTip(
            tooltip_with_shortcut(
                "Statistical percentile clip before locating black/white points. Higher = discard more outlier pixels",
                ["drange_clip_inc", "drange_clip_dec"],
            )
        )
        proc.white_point_slider.setToolTip(
            tooltip_with_shortcut(
                "Manual offset on top of the auto-detected white point. Positive = brighter; negative = pull highlights back",
                ["white_point_inc", "white_point_dec"],
            )
        )
        proc.black_point_slider.setToolTip(
            tooltip_with_shortcut(
                "Manual offset for the black point. Positive = lifted blacks; negative = deeper blacks",
                ["black_point_inc", "black_point_dec"],
            )
        )

        lab.separation_slider.setToolTip(
            tooltip_with_shortcut(
                "Amplifies differences between R, G, B channels. Higher = richer colour separation; 1.0 = identity",
                ["separation_inc", "separation_dec"],
            )
        )
        lab.chroma_denoise_slider.setToolTip(
            tooltip_with_shortcut(
                "Chroma denoise in Lab space — smooths colour noise while preserving luminance grain",
                ["chroma_denoise_inc", "chroma_denoise_dec"],
            )
        )
        lab.saturation_slider.setToolTip(
            tooltip_with_shortcut(
                "Linear saturation. 1.0 = unchanged, 0 = greyscale, 2.0 = double saturation",
                ["saturation_inc", "saturation_dec"],
            )
        )
        lab.vibrance_slider.setToolTip(
            tooltip_with_shortcut(
                "Smart saturation that boosts muted colours more than already-saturated ones — gentler on skin tones than raw Saturation",
                ["vibrance_inc", "vibrance_dec"],
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
                "L-channel unsharp mask — crisps detail without introducing colour halos around edges",
                ["sharpen_inc", "sharpen_dec"],
            )
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
                "Toggle manual heal brush — click dust spots in the preview to paint them out one at a time",
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
                "Simulates selenium toning — adds a cool blue-purple cast to shadows. B&W mode only",
                ["selenium_inc", "selenium_dec"],
            )
        )
        ton.sepia_slider.setToolTip(
            tooltip_with_shortcut(
                "Simulates sepia toning — adds a warm brown cast across the full tonal range. B&W mode only",
                ["sepia_inc", "sepia_dec"],
            )
        )
        ton.shadow_hue_slider.setToolTip(
            tooltip_with_shortcut(
                "Hue of the shadow split-tone colour injection",
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
                "Hue of the highlight split-tone colour injection",
                ["highlight_hue_inc", "highlight_hue_dec"],
            )
        )
        ton.highlight_str_slider.setToolTip(
            tooltip_with_shortcut(
                "How strongly the highlight hue is mixed in",
                ["highlight_strength_inc", "highlight_strength_dec"],
            )
        )

        fin.vignette_strength_slider.setToolTip(
            tooltip_with_shortcut(
                "Negative = darken corners (classic vignette); positive = lighten corners. 0 = off",
                ["vignette_str_inc", "vignette_str_dec"],
            )
        )
        fin.vignette_size_slider.setToolTip(
            tooltip_with_shortcut(
                "Falloff radius: smaller = tight corner effect; larger = vignette spreads well into the frame",
                ["vignette_size_inc", "vignette_size_dec"],
            )
        )
        fin.border_slider.setToolTip(
            tooltip_with_shortcut(
                "Border thickness as a fraction of the image dimensions. Zero = no border",
                ["border_size_inc", "border_size_dec"],
            )
        )

    def _sync_all_sidebars(self) -> None:
        """Force all sidebar panels to update their widgets from current AppState."""
        self.process_sidebar.sync_ui()
        self.exposure_sidebar.sync_ui()
        self.geometry_sidebar.sync_ui()
        self.lab_sidebar.sync_ui()
        self.toning_sidebar.sync_ui()
        self.retouch_sidebar.sync_ui()
        self.icc_sidebar.sync_ui()
        self.finish_sidebar.sync_ui()
        self.presets_sidebar.sync_ui()
        self._sync_modified_dots()
        buf = self.controller.state.last_metrics.get("histogram_raw")
        self.exposure_histogram.update_data(buf)

    def _sync_modified_dots(self) -> None:
        """Update modified-indicator dots on collapsible section headers."""
        cfg = self.controller.state.config
        _exp = ExposureConfig()
        _lab = LabConfig()
        _ton = ToningConfig()
        _geo = GeometryConfig()
        _proc = ProcessConfig()

        exp = cfg.exposure
        exposure_count = sum(
            [
                exp.density != _exp.density,
                exp.grade != _exp.grade,
                exp.linear_raw != _exp.linear_raw,
                exp.wb_cyan != _exp.wb_cyan,
                exp.wb_magenta != _exp.wb_magenta,
                exp.wb_yellow != _exp.wb_yellow,
                exp.shadow_cyan != _exp.shadow_cyan,
                exp.shadow_magenta != _exp.shadow_magenta,
                exp.shadow_yellow != _exp.shadow_yellow,
                exp.highlight_cyan != _exp.highlight_cyan,
                exp.highlight_magenta != _exp.highlight_magenta,
                exp.highlight_yellow != _exp.highlight_yellow,
                exp.toe != _exp.toe,
                exp.toe_width != _exp.toe_width,
                exp.shoulder != _exp.shoulder,
                exp.shoulder_width != _exp.shoulder_width,
            ]
        )

        lab = cfg.lab
        lab_count = sum(
            [
                lab.color_separation != _lab.color_separation,
                lab.saturation != _lab.saturation,
                lab.vibrance != _lab.vibrance,
                lab.clahe_strength != _lab.clahe_strength,
                lab.sharpen != _lab.sharpen,
                lab.chroma_denoise != _lab.chroma_denoise,
                lab.glow_amount != _lab.glow_amount,
                lab.halation_strength != _lab.halation_strength,
            ]
        )

        ton = cfg.toning
        toning_count = sum(
            [
                ton.paper_profile != _ton.paper_profile,
                ton.selenium_strength != _ton.selenium_strength,
                ton.sepia_strength != _ton.sepia_strength,
                ton.shadow_tint_hue != _ton.shadow_tint_hue,
                ton.shadow_tint_strength != _ton.shadow_tint_strength,
                ton.highlight_tint_hue != _ton.highlight_tint_hue,
                ton.highlight_tint_strength != _ton.highlight_tint_strength,
            ]
        )

        geo = cfg.geometry
        geometry_count = sum(
            [
                geo.fine_rotation != _geo.fine_rotation,
                geo.flip_horizontal != _geo.flip_horizontal,
                geo.flip_vertical != _geo.flip_vertical,
                geo.auto_crop_enabled != _geo.auto_crop_enabled,
                geo.manual_crop_rect is not None,
                geo.autocrop_ratio != _geo.autocrop_ratio,
                geo.autocrop_offset != _geo.autocrop_offset,
            ]
        )

        proc = cfg.process
        process_count = sum(
            [
                proc.process_mode != _proc.process_mode,
                proc.analysis_buffer != _proc.analysis_buffer,
                proc.drange_clip != _proc.drange_clip,
                proc.white_point_offset != _proc.white_point_offset,
                proc.black_point_offset != _proc.black_point_offset,
            ]
        )

        ret = cfg.retouch
        retouch_count = int(ret.dust_remove) + len(ret.manual_dust_spots)

        from negpy.features.finish.models import FinishConfig

        _fin = FinishConfig()
        fin = cfg.finish
        finish_count = sum(
            [
                fin.vignette_strength != _fin.vignette_strength,
                fin.vignette_size != _fin.vignette_size,
                fin.border_size != _fin.border_size,
                fin.border_color != _fin.border_color,
            ]
        )

        self.exposure_section.set_modified(exposure_count)
        self.lab_section.set_modified(lab_count)
        self.toning_section.set_modified(toning_count)
        self.geometry_section.set_modified(geometry_count)
        self.process_section.set_modified(process_count)
        self.retouch_section.set_modified(retouch_count)
        self.finish_section.set_modified(finish_count)

    def _sync_tool_buttons(self) -> None:
        """Updates toggle button states to match active_tool."""
        self.geometry_sidebar.sync_ui()
