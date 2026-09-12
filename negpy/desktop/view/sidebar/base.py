import types
from typing import Any
from dataclasses import replace
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QComboBox, QPushButton, QWidget, QVBoxLayout
from negpy.desktop.controller import AppController
from negpy.desktop.view.styles.templates import ICON_BUTTON_WIDTH, icon_button, labeled_action, labeled_toggle, tool_toggle
from negpy.desktop.view.styles.theme import THEME


def install_wheel_guards(widget: QWidget) -> None:
    """Scroll must not change a combo's value unless it has focus — otherwise scrolling
    a panel silently edits every combo the pointer crosses. Module-level so panels that
    are not BaseSidebar subclasses (ScanSidebar) can share it."""
    for combo in widget.findChildren(QComboBox):
        combo.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        def _wheel(c, event) -> None:
            if c.hasFocus():
                QComboBox.wheelEvent(c, event)
            else:
                event.ignore()

        combo.wheelEvent = types.MethodType(_wheel, combo)


class BaseSidebar(QWidget):
    """
    Base class for all sidebar panels.
    Handles common setup and configuration updates.
    """

    # 0 = the card owns horizontal inset; standalone panels override.
    SIDE_MARGIN = 0

    def __init__(self, controller: AppController):
        super().__init__()
        self.controller = controller
        self.state = controller.state

        self._init_layout()
        self._init_ui()
        self._connect_signals()
        self._install_wheel_guards()

    def _install_wheel_guards(self) -> None:
        install_wheel_guards(self)

    def _init_layout(self) -> None:
        """Sets up the default QVBoxLayout."""
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(self.SIDE_MARGIN, 0, self.SIDE_MARGIN, 5)
        self.layout.setSpacing(THEME.space_lg)

    def _init_ui(self) -> None:
        """Override to add widgets to self.layout."""
        pass

    def _connect_signals(self) -> None:
        """Override to connect widget signals."""
        pass

    def sync_ui(self) -> None:
        """Override to update widgets from current AppState."""
        pass

    def _tool_toggle(self, icon_name: str, label: str, tooltip: str) -> QPushButton:
        return tool_toggle(icon_name, label, tooltip)

    def _small_toggle(self, icon_name: str, label: str, checked: bool, tooltip: str) -> QPushButton:
        """_tool_toggle with an initial checked state; the name marks the role."""
        btn = tool_toggle(icon_name, label, tooltip)
        btn.setChecked(checked)
        return btn

    def _labeled_action(self, icon_name: str, label: str, tooltip: str) -> QPushButton:
        return labeled_action(icon_name, label, tooltip)

    def _icon_action(self, icon_name: str, tooltip: str, width: int | None = ICON_BUTTON_WIDTH) -> QPushButton:
        return icon_button(icon_name, tooltip, width)

    def _labeled_toggle(self, icon_name: str, label: str, checked: bool, tooltip: str) -> QPushButton:
        return labeled_toggle(icon_name, label, checked, tooltip)

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

        if render:
            # apply_config, not request_render: a change to a source input needs the source decoded
            # again, and only it knows which changes those are.
            self.controller.apply_config(new_config, persist=persist, readback_metrics=readback_metrics)
        else:
            self.controller.session.update_config(new_config, persist=persist, render=False)

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
        if render:
            self.controller.apply_config(new_config, persist=persist, readback_metrics=readback_metrics)
        else:
            self.controller.session.update_config(new_config, persist=persist, render=False)
