from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
)
from PyQt6.QtCore import Qt
import qtawesome as qta

from negpy.desktop.controller import AppController
from negpy.desktop.view.widgets.collapsible import CollapsibleSection
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
        self.exposure_section = self._add_sidebar_section(
            "Exposure",
            "exposure",
            self.exposure_sidebar,
            icon=qta.icon("fa5s.sun", color=icon_color),
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

        self.icc_sidebar = ICCSidebar(self.controller)
        self._add_sidebar_section(
            "ICC",
            "icc",
            self.icc_sidebar,
            icon=qta.icon("fa5s.eye", color=icon_color),
        )

    def _add_sidebar_section(self, title: str, key: str, widget: QWidget, icon=None) -> CollapsibleSection:
        """Helper to create and add a collapsible section. Returns the section widget."""
        is_expanded = THEME.sidebar_expanded_defaults.get(key, False)
        if key in [
            "process",
            "exposure",
            "geometry",
            "lab",
            "retouch",
            "export",
            "analysis",
        ]:
            is_expanded = THEME.sidebar_expanded_defaults.get(key, True)

        section = CollapsibleSection(title, expanded=is_expanded, icon=icon)
        section.set_content(widget)
        self.layout.addWidget(section)
        return section

    def _connect_signals(self) -> None:
        self.controller.config_updated.connect(self._sync_all_sidebars)
        self.controller.tool_sync_requested.connect(self._sync_tool_buttons)

        self.exposure_section.reset_requested.connect(lambda: self.controller.session.reset_section("exposure"))
        self.lab_section.reset_requested.connect(lambda: self.controller.session.reset_section("lab"))
        self.toning_section.reset_requested.connect(lambda: self.controller.session.reset_section("toning"))
        self.geometry_section.reset_requested.connect(lambda: self.controller.session.reset_section("geometry"))
        self.process_section.reset_requested.connect(lambda: self.controller.session.reset_section("process"))
        self.retouch_section.reset_requested.connect(lambda: self.controller.session.reset_section("retouch"))

    def _sync_all_sidebars(self) -> None:
        """Force all sidebar panels to update their widgets from current AppState."""
        self.process_sidebar.sync_ui()
        self.exposure_sidebar.sync_ui()
        self.geometry_sidebar.sync_ui()
        self.lab_sidebar.sync_ui()
        self.toning_sidebar.sync_ui()
        self.retouch_sidebar.sync_ui()
        self.icc_sidebar.sync_ui()
        self.presets_sidebar.sync_ui()
        self._sync_modified_dots()

    def _sync_modified_dots(self) -> None:
        """Update modified-indicator dots on collapsible section headers."""
        cfg = self.controller.state.config
        _exp = ExposureConfig()
        _lab = LabConfig()
        _ton = ToningConfig()
        _geo = GeometryConfig()
        _proc = ProcessConfig()

        exp = cfg.exposure
        exposure_modified = (
            exp.density != _exp.density
            or exp.grade != _exp.grade
            or exp.use_camera_wb != _exp.use_camera_wb
            or exp.wb_cyan != _exp.wb_cyan
            or exp.wb_magenta != _exp.wb_magenta
            or exp.wb_yellow != _exp.wb_yellow
            or exp.shadow_cyan != _exp.shadow_cyan
            or exp.shadow_magenta != _exp.shadow_magenta
            or exp.shadow_yellow != _exp.shadow_yellow
            or exp.highlight_cyan != _exp.highlight_cyan
            or exp.highlight_magenta != _exp.highlight_magenta
            or exp.highlight_yellow != _exp.highlight_yellow
            or exp.toe != _exp.toe
            or exp.toe_width != _exp.toe_width
            or exp.shoulder != _exp.shoulder
            or exp.shoulder_width != _exp.shoulder_width
        )

        lab = cfg.lab
        lab_modified = (
            lab.color_separation != _lab.color_separation
            or lab.saturation != _lab.saturation
            or lab.vibrance != _lab.vibrance
            or lab.clahe_strength != _lab.clahe_strength
            or lab.sharpen != _lab.sharpen
            or lab.chroma_denoise != _lab.chroma_denoise
            or lab.glow_amount != _lab.glow_amount
            or lab.halation_strength != _lab.halation_strength
        )

        ton = cfg.toning
        toning_modified = (
            ton.paper_profile != _ton.paper_profile
            or ton.selenium_strength != _ton.selenium_strength
            or ton.sepia_strength != _ton.sepia_strength
            or ton.shadow_tint_hue != _ton.shadow_tint_hue
            or ton.shadow_tint_strength != _ton.shadow_tint_strength
            or ton.highlight_tint_hue != _ton.highlight_tint_hue
            or ton.highlight_tint_strength != _ton.highlight_tint_strength
        )

        geo = cfg.geometry
        geometry_modified = (
            geo.fine_rotation != _geo.fine_rotation
            or geo.flip_horizontal != _geo.flip_horizontal
            or geo.flip_vertical != _geo.flip_vertical
            or geo.manual_crop_rect is not None
            or geo.autocrop_ratio != _geo.autocrop_ratio
            or geo.autocrop_offset != _geo.autocrop_offset
        )

        proc = cfg.process
        process_modified = (
            proc.process_mode != _proc.process_mode
            or proc.analysis_buffer != _proc.analysis_buffer
            or proc.drange_clip != _proc.drange_clip
            or proc.white_point_offset != _proc.white_point_offset
            or proc.black_point_offset != _proc.black_point_offset
        )

        ret = cfg.retouch
        retouch_modified = ret.dust_remove or len(ret.manual_dust_spots) > 0

        self.exposure_section.set_modified(exposure_modified)
        self.lab_section.set_modified(lab_modified)
        self.toning_section.set_modified(toning_modified)
        self.geometry_section.set_modified(geometry_modified)
        self.process_section.set_modified(process_modified)
        self.retouch_section.set_modified(retouch_modified)

    def _sync_tool_buttons(self) -> None:
        """Updates toggle button states to match active_tool."""
        self.geometry_sidebar.sync_ui()
