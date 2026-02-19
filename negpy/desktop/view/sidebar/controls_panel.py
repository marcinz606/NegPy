from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
)
from PyQt6.QtCore import Qt
import qtawesome as qta

from negpy.desktop.controller import AppController
from negpy.desktop.view.widgets.collapsible import CollapsibleSection
from negpy.desktop.view.styles.theme import THEME

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
        self._add_sidebar_section(
            "Geometry",
            "geometry",
            self.geometry_sidebar,
            icon=qta.icon("fa5s.crop", color=icon_color),
        )

        self.process_sidebar = ProcessSidebar(self.controller)
        self._add_sidebar_section(
            "Process",
            "process",
            self.process_sidebar,
            icon=qta.icon("fa5s.cogs", color=icon_color),
        )

        self.exposure_sidebar = ExposureSidebar(self.controller)
        self._add_sidebar_section(
            "Exposure",
            "exposure",
            self.exposure_sidebar,
            icon=qta.icon("fa5s.sun", color=icon_color),
        )

        self.lab_sidebar = LabSidebar(self.controller)
        self._add_sidebar_section(
            "Lab",
            "lab",
            self.lab_sidebar,
            icon=qta.icon("fa5s.flask", color=icon_color),
        )

        self.retouch_sidebar = RetouchSidebar(self.controller)
        self._add_sidebar_section(
            "Retouch",
            "retouch",
            self.retouch_sidebar,
            icon=qta.icon("fa5s.brush", color=icon_color),
        )

        self.toning_sidebar = ToningSidebar(self.controller)
        self._add_sidebar_section(
            "Toning",
            "toning",
            self.toning_sidebar,
            icon=qta.icon("fa5s.tint", color=icon_color),
        )

        self.icc_sidebar = ICCSidebar(self.controller)
        self._add_sidebar_section(
            "ICC",
            "icc",
            self.icc_sidebar,
            icon=qta.icon("fa5s.eye", color=icon_color),
        )

    def _add_sidebar_section(self, title: str, key: str, widget: QWidget, icon=None) -> None:
        """Helper to create and add a collapsible section."""
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

    def _connect_signals(self) -> None:
        self.controller.config_updated.connect(self._sync_all_sidebars)
        self.controller.tool_sync_requested.connect(self._sync_tool_buttons)

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

    def _sync_tool_buttons(self) -> None:
        """Updates toggle button states to match active_tool."""
        self.geometry_sidebar.sync_ui()
