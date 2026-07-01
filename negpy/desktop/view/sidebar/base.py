import types
from typing import Any
from dataclasses import replace
import qtawesome as qta
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QComboBox, QPushButton, QWidget, QVBoxLayout
from negpy.desktop.controller import AppController
from negpy.desktop.view.styles.theme import THEME


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
        self._install_wheel_guards()

    def _install_wheel_guards(self) -> None:
        for combo in self.findChildren(QComboBox):
            combo.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

            def _wheel(c, event) -> None:
                if c.hasFocus():
                    QComboBox.wheelEvent(c, event)
                else:
                    event.ignore()

            combo.wheelEvent = types.MethodType(_wheel, combo)

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

    def _icon_toggle(self, icon_name: str, checked: bool, tooltip: str) -> QPushButton:
        """Compact icon-only checkable button placed beside a slider."""
        btn = QPushButton()
        btn.setCheckable(True)
        btn.setChecked(checked)
        btn.setIcon(qta.icon(icon_name, color=THEME.text_primary))
        btn.setStyleSheet(f"font-size: {THEME.font_size_base}px; padding: 6px;")
        btn.setFixedWidth(36)
        btn.setToolTip(tooltip)
        return btn

    def _labeled_toggle(self, icon_name: str, label: str, checked: bool, tooltip: str) -> QPushButton:
        """Labeled checkable button (icon + text), styled like Pick WB / Linear RAW."""
        btn = QPushButton(label)
        btn.setCheckable(True)
        btn.setChecked(checked)
        btn.setIcon(qta.icon(icon_name, color=THEME.text_primary))
        btn.setStyleSheet(f"font-size: {THEME.font_size_base}px; padding: 8px;")
        btn.setToolTip(tooltip)
        return btn

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
