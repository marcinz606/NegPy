from typing import Any
from dataclasses import replace
from PyQt6.QtWidgets import QWidget, QVBoxLayout
from negpy.desktop.controller import AppController


class BaseSidebar(QWidget):
    """
    Base class for all sidebar panels.
    Handles common setup and configuration updates.
    """

    def __init__(self, controller: AppController):
        super().__init__()
        self.controller = controller
        self.state = controller.state

        self._init_layout()
        self._init_ui()
        self._connect_signals()

    def _init_layout(self) -> None:
        """Sets up the default QVBoxLayout."""
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(5, 0, 5, 5)
        self.layout.setSpacing(10)

    def _init_ui(self) -> None:
        """Override to add widgets to self.layout."""
        pass

    def _connect_signals(self) -> None:
        """Override to connect widget signals."""
        pass

    def sync_ui(self) -> None:
        """Override to update widgets from current AppState."""
        pass

    def update_config_section(
        self,
        section_name: str,
        render: bool = True,
        persist: bool = False,
        readback_metrics: bool = True,
        **changes: Any,
    ) -> None:
        """
        Updates a specific section (e.g., 'exposure') of the configuration.

        Args:
            section_name: Name of the config field (e.g. 'exposure', 'geometry').
            render: Whether to request a new render after update.
            persist: Whether to save this change to disk (sidecar).
            readback_metrics: Whether to read back metrics (histogram, etc.) after render.
            changes: Key-value pairs to update in that section.
        """
        current_section = getattr(self.state.config, section_name)
        new_section = replace(current_section, **changes)

        # Replace the section in the main config object
        new_config = replace(self.state.config, **{section_name: new_section})

        self.controller.session.update_config(new_config, persist=persist, render=render)

        if render:
            self.controller.request_render(readback_metrics=readback_metrics)

    def update_config_root(
        self,
        render: bool = True,
        persist: bool = False,
        readback_metrics: bool = True,
        **changes: Any,
    ) -> None:
        """
        Updates fields on the root config object directly.
        """
        new_config = replace(self.state.config, **changes)
        self.controller.session.update_config(new_config, persist=persist, render=render)

        if render:
            self.controller.request_render(readback_metrics=readback_metrics)
