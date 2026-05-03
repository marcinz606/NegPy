from PyQt6.QtWidgets import QWidget, QLabel, QHBoxLayout
from negpy.desktop.view.styles.theme import THEME


class InfoLabel(QLabel):
    """Subtle label for image metadata."""

    def __init__(self, text: str = ""):
        super().__init__(text)
        self.setStyleSheet(f"color: {THEME.text_secondary}; font-size: {THEME.font_size_xs}px; font-weight: bold;")


class ImageMetadataPanel(QWidget):
    """
    Persistent panel for image metadata, placed above/below the preview.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(28)
        self._init_ui()

    def _init_ui(self) -> None:
        self.layout = QHBoxLayout(self)
        self.layout.setContentsMargins(THEME.space_md, THEME.space_sm, THEME.space_md, THEME.space_sm)
        self.layout.setSpacing(20)

        self.lbl_left = InfoLabel("-")
        self.lbl_right = InfoLabel("-")

        self.layout.addWidget(self.lbl_left)
        self.layout.addStretch()
        self.layout.addWidget(self.lbl_right)

    def update_values(self, left: str, right: str) -> None:
        self.lbl_left.setText(left)
        self.lbl_right.setText(right)
