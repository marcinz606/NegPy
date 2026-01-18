from PyQt6.QtWidgets import QWidget, QVBoxLayout, QPushButton, QFrame
from src.desktop.view.styles.theme import THEME


class CollapsibleSection(QWidget):
    """
    A simple collapsible container with a header button and configurable initial state.
    """

    def __init__(self, title: str, expanded: bool = True, parent=None):
        super().__init__(parent)
        self._title_text = title

        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)

        prefix = "▼" if expanded else "▶"
        self.toggle_button = QPushButton(f"{prefix} {title}")
        self.toggle_button.setCheckable(True)
        self.toggle_button.setChecked(expanded)
        self.toggle_button.setStyleSheet(
            f"""
            QPushButton {{
                text-align: left;
                font-weight: bold;
                font-size: {THEME.font_size_header}px;
                padding: 10px;
                background-color: #2a2a2a;
                border: none;
                border-bottom: 1px solid #333;
                color: #ddd;
            }}
            QPushButton:hover {{
                background-color: #333;
            }}
            QPushButton:checked {{
                border-bottom: 1px solid #444;
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
        prefix = "▼" if checked else "▶"
        self.toggle_button.setText(f"{prefix} {self._title_text}")
