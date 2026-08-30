from typing import Optional

from PyQt6.QtWidgets import (
    QButtonGroup,
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QRadioButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from negpy.desktop.view.styles.templates import hint_label, pane_header_qss
from negpy.desktop.view.styles.theme import THEME
from negpy.features.process.models import scan_setup_values

_CAPTURE = (
    ("camera", "Digital camera", "A camera on a copy stand shooting the negative over a light source"),
    ("scanner", "Film scanner", "A dedicated film scanner"),
)

_LIGHT = {
    "camera": (
        ("white", "White light", "A lightbox or high-CRI LED panel — one broadband exposure"),
        ("narrowband", "Narrowband RGB", "A Scanlight or RGB-LED source — pure red, green and blue bands"),
    ),
    "scanner": (
        ("white", "White light", "A cold-cathode or white-LED lamp — Plustek, Epson, most flatbeds"),
        ("narrowband", "Narrowband RGB", "RGB-LED illumination — Nikon Coolscan, Kodak Pakon"),
    ),
}


class ScanSetupDialog(QDialog):
    """Two-page wizard asking what rig the negatives were scanned on, so Linear RAW and
    Narrowband can be set from the answer instead of by trial and error."""

    def __init__(self, parent, saved: Optional[dict] = None):
        super().__init__(parent)
        self._capture = str((saved or {}).get("capture") or "camera")
        self._light = str((saved or {}).get("light") or "white")

        self.setWindowTitle("Scanning Setup")
        self.setMinimumWidth(460)

        root = QVBoxLayout(self)
        root.setContentsMargins(THEME.space_2xl, THEME.space_2xl, THEME.space_2xl, THEME.space_2xl)
        root.setSpacing(THEME.space_xl)

        self.pages = QStackedWidget()
        self.capture_group = QButtonGroup(self)
        self.light_group = QButtonGroup(self)
        self.pages.addWidget(self._build_capture_page())
        self.pages.addWidget(self._build_light_page())
        root.addWidget(self.pages)

        self.summary = hint_label()
        root.addWidget(self.summary)
        root.addLayout(self._build_footer())

        self._sync()

    def _build_capture_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(THEME.space_md)

        title = QLabel("HOW DO YOU SCAN?")
        title.setStyleSheet(pane_header_qss())
        layout.addWidget(title)

        for i, (key, label, blurb) in enumerate(_CAPTURE):
            btn = QRadioButton(label)
            btn.setChecked(key == self._capture)
            self.capture_group.addButton(btn, i)
            layout.addWidget(btn)
            layout.addWidget(hint_label(blurb))
        self.capture_group.idToggled.connect(lambda i, on: self._set_capture(_CAPTURE[i][0]) if on else None)
        layout.addStretch()
        return page

    def _build_light_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(THEME.space_md)

        title = QLabel("WHAT LIGHT SOURCE?")
        title.setStyleSheet(pane_header_qss())
        layout.addWidget(title)

        self.light_buttons: list[QRadioButton] = []
        self.light_hints: list[QLabel] = []
        for i in range(2):
            btn = QRadioButton()
            self.light_group.addButton(btn, i)
            hint = hint_label()
            self.light_buttons.append(btn)
            self.light_hints.append(hint)
            layout.addWidget(btn)
            layout.addWidget(hint)
        self.light_group.idToggled.connect(lambda i, on: self._set_light(_LIGHT[self._capture][i][0]) if on else None)
        layout.addStretch()
        return page

    def _build_footer(self) -> QHBoxLayout:
        row = QHBoxLayout()
        self.back_btn = QPushButton("← Back")
        self.back_btn.clicked.connect(self._back)
        row.addWidget(self.back_btn)
        row.addStretch()
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        self.next_btn = QPushButton()
        self.next_btn.setProperty("primary", True)
        self.next_btn.clicked.connect(self._advance)
        row.addWidget(cancel)
        row.addWidget(self.next_btn)
        return row

    def _set_capture(self, capture: str) -> None:
        self._capture = capture
        self._sync()

    def _set_light(self, light: str) -> None:
        self._light = light
        self._sync()

    def _advance(self) -> None:
        if self.pages.currentIndex() == 0:
            self.pages.setCurrentIndex(1)
            self._sync()
        else:
            self.accept()

    def _back(self) -> None:
        self.pages.setCurrentIndex(0)
        self._sync()

    def _sync(self) -> None:
        """Re-label the light page for the chosen capture and refresh the footer/summary."""
        options = _LIGHT[self._capture]
        for i, (key, label, blurb) in enumerate(options):
            self.light_buttons[i].setText(label)
            self.light_hints[i].setText(blurb)
            self.light_buttons[i].blockSignals(True)
            self.light_buttons[i].setChecked(key == self._light)
            self.light_buttons[i].blockSignals(False)

        on_light_page = self.pages.currentIndex() == 1
        self.back_btn.setVisible(on_light_page)
        self.next_btn.setText("Apply" if on_light_page else "Next →")
        linear_raw, narrowband = scan_setup_values(self._capture, self._light)
        self.summary.setText(
            f"Sets Linear RAW {'on' if linear_raw else 'off'} · Narrowband {'on' if narrowband else 'off'}"
            if on_light_page
            else "Two questions — they set Linear RAW and Narrowband for your rig."
        )

    def choice(self) -> dict:
        return {"capture": self._capture, "light": self._light}
