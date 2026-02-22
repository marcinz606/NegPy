from typing import Optional
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QPushButton, QFrame, QHBoxLayout, QLabel
from PyQt6.QtGui import QIcon
from PyQt6.QtCore import Qt
from negpy.desktop.view.styles.theme import THEME
import qtawesome as qta


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

        self.toggle_button = QPushButton()
        self.toggle_button.setCheckable(True)
        self.toggle_button.setChecked(expanded)
        self.toggle_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.toggle_button.setFixedHeight(38)

        self.toggle_button.setStyleSheet(
            f"""
            QPushButton {{
                text-align: left;
                background-color: #1A1A1A;
                border: none;
                border-bottom: 1px solid #262626;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                color: #FFFFFF;
                padding: 0;
            }}
            QPushButton:hover {{
                background-color: #222222;
            }}
            QPushButton:checked {{
                background-color: #1A1A1A;
                border-bottom: 1px solid {THEME.accent_primary};
            }}
        """
        )

        # Create a layout inside the toggle button for custom icon/text/chevron placement
        btn_layout = QHBoxLayout(self.toggle_button)
        btn_layout.setContentsMargins(16, 12, 16, 12)
        btn_layout.setSpacing(10)

        if icon:
            icon_label = QLabel()
            icon_label.setPixmap(icon.pixmap(14, 14))
            btn_layout.addWidget(icon_label)

        title_label = QLabel(self._title_text)
        title_label.setStyleSheet(f"font-weight: bold; font-size: {THEME.font_size_header}px; background: transparent;")
        btn_layout.addWidget(title_label)

        btn_layout.addStretch()

        self.chevron_label = QLabel()
        self.chevron_label.setStyleSheet("background: transparent;")
        # Set initial chevron
        self._update_chevron(expanded)
        btn_layout.addWidget(self.chevron_label)

        self.content_area = QFrame()
        self.content_area.setStyleSheet("""
            QFrame {
                background-color: #121212;
                border-bottom-left-radius: 4px;
                border-bottom-right-radius: 4px;
                border: 1px solid #1A1A1A;
                border-top: none;
            }
        """)
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

    def _update_chevron(self, expanded: bool) -> None:
        if expanded:
            self.chevron_label.setPixmap(qta.icon("fa5s.chevron-down", color="#A0A0A0").pixmap(12, 12))
        else:
            self.chevron_label.setPixmap(qta.icon("fa5s.chevron-right", color="#A0A0A0").pixmap(12, 12))

    def _on_toggle(self, checked: bool) -> None:
        self.content_area.setVisible(checked)
        self._update_chevron(checked)
