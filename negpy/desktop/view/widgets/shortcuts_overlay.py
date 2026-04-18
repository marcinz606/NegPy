from PyQt6.QtWidgets import QDialog, QVBoxLayout, QGridLayout, QLabel, QPushButton, QFrame
from PyQt6.QtCore import Qt
from negpy.desktop.view.styles.theme import THEME


# (category, key_display, description)
SHORTCUT_TABLE = [
    ("Navigation", "← / →", "Previous / Next file"),
    ("Geometry", "[ / ]", "Rotate 90° CW / CCW"),
    ("Geometry", "H", "Flip horizontal"),
    ("Geometry", "V", "Flip vertical"),
    ("Tools", "Shift+W", "Toggle WB picker"),
    ("Tools", "Shift+C", "Toggle manual crop"),
    ("Tools", "Shift+D", "Toggle heal tool"),
    ("Density", "Q / A", "Density +0.01 / −0.01"),
    ("Grade", "W / S", "Grade +0.01 / −0.01"),
    ("Magenta", "E / D", "Magenta +0.01 / −0.01"),
    ("Yellow", "R / F", "Yellow +0.01 / −0.01"),
    ("Crop", "X / Z", "Crop offset +1 / −1"),
    ("Actions", "Ctrl+E", "Export"),
    ("Actions", "Ctrl+C / Ctrl+V", "Copy / Paste settings"),
    ("Actions", "Ctrl+Z / Ctrl+Y", "Undo / Redo"),
    ("Help", "?", "Show this overlay"),
]


class ShortcutsOverlay(QDialog):
    """Modal keyboard shortcut reference, opened with '?'."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Keyboard Shortcuts")
        self.setWindowFlags(Qt.WindowType.Dialog | Qt.WindowType.WindowCloseButtonHint)
        self.setModal(True)
        self._init_ui()

    def _init_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 20)
        root.setSpacing(0)

        self.setStyleSheet(f"""
            QDialog {{
                background-color: {THEME.bg_panel};
                border: 1px solid {THEME.border_primary};
            }}
            QLabel {{
                color: {THEME.text_primary};
                font-size: 12px;
            }}
        """)

        grid = QGridLayout()
        grid.setSpacing(0)
        grid.setColumnMinimumWidth(0, 90)
        grid.setColumnMinimumWidth(1, 160)
        grid.setColumnMinimumWidth(2, 260)

        prev_category = None
        row = 0
        for category, key, desc in SHORTCUT_TABLE:
            if category != prev_category:
                if prev_category is not None:
                    sep = QFrame()
                    sep.setFrameShape(QFrame.Shape.HLine)
                    sep.setStyleSheet(f"background-color: {THEME.border_primary}; border: none; margin: 4px 0;")
                    sep.setFixedHeight(1)
                    grid.addWidget(sep, row, 0, 1, 3)
                    row += 1
                cat_lbl = QLabel(category)
                cat_lbl.setStyleSheet(f"color: {THEME.text_secondary}; font-size: 10px; font-weight: bold; padding: 6px 0 2px 0;")
                grid.addWidget(cat_lbl, row, 0, 1, 3)
                row += 1
                prev_category = category

            key_lbl = QLabel(key)
            key_lbl.setStyleSheet(f"""
                color: {THEME.text_primary};
                background-color: {THEME.bg_header};
                border: 1px solid {THEME.border_primary};
                border-radius: 3px;
                font-family: monospace;
                font-size: 11px;
                padding: 1px 5px;
            """)
            key_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            desc_lbl = QLabel(desc)
            desc_lbl.setStyleSheet(f"color: {THEME.text_secondary}; font-size: 12px; padding-left: 8px;")
            grid.addWidget(key_lbl, row, 1)
            grid.addWidget(desc_lbl, row, 2)
            row += 1

        root.addLayout(grid)
        root.addSpacing(16)

        close_btn = QPushButton("Close")
        close_btn.setStyleSheet(
            f"font-size: 12px; padding: 6px 20px; background: {THEME.accent_primary}; color: white; border: none; border-radius: 3px;"
        )
        close_btn.clicked.connect(self.accept)
        root.addWidget(close_btn, alignment=Qt.AlignmentFlag.AlignRight)
