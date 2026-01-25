from typing import Optional
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QPushButton, QFrame
from PyQt6.QtGui import QIcon
from PyQt6.QtCore import QSize
from src.desktop.view.styles.theme import THEME


class CollapsibleSection(QWidget):
    """
    A simple collapsible container with a header button and configurable initial state.
    """

    def __init__(
        self,
        title: str,
        expanded: bool = True,
        icon: Optional[QIcon] = None,
        parent=None,
    ):
        super().__init__(parent)
        self._title_text = title

        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)

        self.toggle_button = QPushButton(title)
        self.toggle_button.setCheckable(True)
        self.toggle_button.setChecked(expanded)

        if icon:
            self.toggle_button.setIcon(icon)
            self.toggle_button.setIconSize(QSize(16, 16))

        self.toggle_button.setStyleSheet(
            f"""
            QPushButton {{
                text-align: left;
                font-weight: bold;
                font-size: {THEME.font_size_header}px;
                padding: 10px;
                background-color: {THEME.bg_header};
                border: none;
                border-bottom: 1px solid {THEME.border_color};
                color: {THEME.text_primary};
            }}
            QPushButton:hover {{
                background-color: #333;
            }}
            QPushButton:checked {{
                background-color: #000000;
                color: {THEME.text_primary};
                border-bottom: 1px solid {THEME.accent_primary};
            }}
        """
        )

        self.content_area = QFrame()
        self.content_layout = QVBoxLayout(self.content_area)
        self.content_layout.setContentsMargins(0, 5, 0, 10)
        self.content_layout.setSpacing(5)
        self.content_area.setVisible(expanded)

        self.main_layout.addWidget(self.toggle_button)
        self.main_layout.addWidget(self.content_area)

        self.toggle_button.toggled.connect(self._on_toggle)

    def set_content(self, widget: QWidget) -> None:
        """
        Adds the main widget to the collapsible area.
        """
        self.content_layout.addWidget(widget)

    def _on_toggle(self, checked: bool) -> None:
        self.content_area.setVisible(checked)
        self.toggle_button.setText(self._title_text)
