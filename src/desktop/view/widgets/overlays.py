from PyQt6.QtWidgets import QWidget, QLabel, QGridLayout
from PyQt6.QtCore import Qt
from src.desktop.view.styles.theme import THEME


class InfoLabel(QLabel):
    """Subtle corner label for image metadata."""

    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        self.setStyleSheet(f"""
            QLabel {{
                color: {THEME.text_secondary};
                background-color: rgba(15, 15, 15, 180);
                padding: 4px 8px;
                border-radius: 4px;
                font-size: 12px;
                font-family: '{THEME.font_family}';
            }}
        """)


class ImageInfoOverlay(QWidget):
    """Overlay containing metadata labels for the 4 corners of the preview."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self._init_ui()

    def _init_ui(self) -> None:
        layout = QGridLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)

        self.lbl_filename = InfoLabel("No File")
        self.lbl_resolution = InfoLabel("- x - px")
        self.lbl_colorspace = InfoLabel("Working Space")
        self.lbl_extra = InfoLabel("Mode")

        layout.addWidget(
            self.lbl_filename,
            0,
            0,
            Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft,
        )
        layout.addWidget(
            self.lbl_resolution,
            0,
            1,
            Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignRight,
        )
        layout.addWidget(
            self.lbl_colorspace,
            1,
            0,
            Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignLeft,
        )
        layout.addWidget(
            self.lbl_extra,
            1,
            1,
            Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignRight,
        )

    def update_info(self, filename: str, res: str, colorspace: str, mode: str) -> None:
        self.lbl_filename.setText(filename)
        self.lbl_resolution.setText(res)
        self.lbl_colorspace.setText(colorspace)
        self.lbl_extra.setText(mode)
