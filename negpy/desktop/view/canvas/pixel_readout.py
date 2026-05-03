from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt6.QtCore import Qt
from negpy.desktop.view.styles.theme import THEME


class PixelReadoutOverlay(QWidget):
    """
    Translucent bottom-right overlay showing RGB and Lab pixel values on canvas hover.
    Must be a child of ImageCanvas; call raise_() after adding to ensure it renders on top.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setVisible(False)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        self.setStyleSheet(f"""
            QWidget {{
                background-color: {THEME.surface_overlay};
                border: 1px solid {THEME.border_primary};
                border-radius: 4px;
            }}
            QLabel {{
                color: {THEME.text_secondary};
                font-family: monospace;
                font-size: {THEME.font_size_xs}px;
                background: transparent;
                border: none;
            }}
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(2)

        self.rgb_label = QLabel("RGB —")
        self.lab_label = QLabel("Lab —")

        layout.addWidget(self.rgb_label)
        layout.addWidget(self.lab_label)

        self.adjustSize()

    def set_values(self, rgb_text: str, lab_text: str) -> None:
        self.rgb_label.setText(rgb_text)
        self.lab_label.setText(lab_text)
        self.adjustSize()
        self._reposition()

    def _reposition(self) -> None:
        p = self.parent()
        if p is None:
            return
        margin = 12
        x = p.width() - self.width() - margin
        y = p.height() - self.height() - margin
        self.move(max(0, x), max(0, y))
