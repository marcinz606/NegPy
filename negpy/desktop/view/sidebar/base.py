import types
from typing import Any
from dataclasses import replace
import qtawesome as qta
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QComboBox, QPushButton, QWidget, QVBoxLayout
from negpy.desktop.controller import AppController
from negpy.desktop.view.styles.templates import default_button_height, labeled_toggle_qss, small_toggle_qss, tool_toggle_qss, wrap_tooltip
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

    def _tool_toggle(self, icon_name: str, label: str, tooltip: str) -> QPushButton:
        """Checkable tool-mode button — arms a transient canvas tool or preview.
        The armed state reads loud (accent red), unlike _small_toggle's quiet
        settings amber. Empty label keeps it icon-only."""
        btn = QPushButton((" " + label) if label else "")
        btn.setCheckable(True)
        btn.setIcon(qta.icon(icon_name, color=THEME.text_primary, color_on="#FFFFFF", color_disabled=THEME.text_muted))
        btn.setStyleSheet(tool_toggle_qss(icon_only=not label))
        btn.setFixedHeight(default_button_height())
        btn.setToolTip(wrap_tooltip(tooltip))
        return btn

    def _small_toggle(self, icon_name: str, label: str, checked: bool, tooltip: str) -> QPushButton:
        """Compact labeled toggle for persistent settings — visually lighter than
        action/tool/selector buttons: subtle checked state (the app-wide 2px
        border reads heavy at this size) and an amber icon while enabled.
        Empty label keeps it icon-only."""
        btn = QPushButton((" " + label) if label else "")
        btn.setCheckable(True)
        btn.setChecked(checked)
        btn.setIcon(qta.icon(icon_name, color=THEME.text_secondary, color_on=THEME.accent_edited, color_disabled=THEME.text_muted))
        btn.setStyleSheet(small_toggle_qss())
        # Match the default/tool button height so mixed rows (e.g. Auto Dust
        # beside Heal Tool) don't stair-step.
        btn.setFixedHeight(default_button_height())
        btn.setToolTip(wrap_tooltip(tooltip))
        return btn

    def _icon_action(self, icon_name: str, tooltip: str, width: int = 36) -> QPushButton:
        """Icon-only one-shot action button, sized to sit flush beside toggles."""
        btn = QPushButton()
        btn.setIcon(qta.icon(icon_name, color=THEME.text_primary, color_disabled=THEME.text_muted))
        btn.setStyleSheet("QPushButton {padding: 6px;}")
        btn.setFixedWidth(width)
        btn.setFixedHeight(default_button_height())
        btn.setToolTip(wrap_tooltip(tooltip))
        return btn

    def _labeled_toggle(self, icon_name: str, label: str, checked: bool, tooltip: str) -> QPushButton:
        """Labeled checkable button (icon + text), styled like Pick WB / Linear RAW."""
        btn = QPushButton(label)
        btn.setCheckable(True)
        btn.setChecked(checked)
        btn.setIcon(qta.icon(icon_name, color=THEME.text_primary, color_disabled=THEME.text_muted))
        btn.setStyleSheet(labeled_toggle_qss())
        btn.setToolTip(wrap_tooltip(tooltip))
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
