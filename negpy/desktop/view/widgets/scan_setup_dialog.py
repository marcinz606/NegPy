from negpy.kernel.system.i18n import tr
from typing import Optional

from PyQt6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from negpy.desktop.view.styles.templates import pin_dialog_default
from negpy.desktop.view.styles.theme import THEME
from negpy.desktop.view.widgets.choice_button import SegmentedChoice
from negpy.desktop.view.widgets.dialog_geometry import remember_dialog_geometry
from negpy.features.exposure.papers import PAPER_PROFILES
from negpy.features.process.models import scan_setup_values

_CAPTURE = (
    ("camera", "fa5s.camera", tr("Camera"), tr("A camera on a copy stand shooting the negative over a light source")),
    ("scanner", "fa5s.camera-retro", tr("Film Scanner"), tr("A dedicated film scanner")),
)

_LIGHT = (
    ("white", "mdi6.lightbulb-outline", tr("White Light")),
    ("narrowband", "mdi6.led-on", tr("Narrowband RGB")),
)

_LIGHT_HINT = {
    "camera": {
        "white": tr("A lightbox or high-CRI LED panel — one broadband exposure"),
        "narrowband": tr("A Scanlight or RGB-LED source — pure red, green and blue bands"),
    },
    "scanner": {
        "white": tr("A cold-cathode or white-LED lamp — Plustek, Epson, most flatbeds"),
        "narrowband": tr("RGB-LED illumination — Nikon Coolscan, Kodak Pakon"),
    },
}

_TRY_STOCKS = tr("Try them under Print stock in Paper Response and keep the one you like.")
_STOCK_HINT = {
    "white": tr(
        "{neutral} is a neutral print. Each paper and print film has its own colors and contrast. ",
        neutral=PAPER_PROFILES["neutral"].label,
    )
    + _TRY_STOCKS,
    "narrowband": tr(
        "{pakon} gives a minilab's colors. {kodak} and {fuji} are near it, and each paper and print film has its own colors. ",
        pakon=PAPER_PROFILES["pakon"].label,
        kodak=PAPER_PROFILES["kodak_supra_endura"].label,
        fuji=PAPER_PROFILES["fuji_crystal"].label,
    )
    + _TRY_STOCKS,
}


def _text(color: str) -> QLabel:
    label = QLabel()
    label.setWordWrap(True)
    label.setStyleSheet(f"color: {color}; font-size: {THEME.font_size_base}px;")
    return label


class ScanSetupDialog(QDialog):
    """Two-page wizard asking what rig the negatives were scanned on, so the decode and the
    print stock are set from the answer instead of by trial and error."""

    def __init__(self, parent, saved: Optional[dict] = None, *, repo=None):
        super().__init__(parent)
        self._capture = str((saved or {}).get("capture") or "camera")
        self._light = str((saved or {}).get("light") or "white")

        self.setWindowTitle(tr("Scanning Setup"))
        self.setMinimumWidth(460)

        root = QVBoxLayout(self)
        root.setContentsMargins(THEME.space_2xl, THEME.space_2xl, THEME.space_2xl, THEME.space_2xl)
        root.setSpacing(THEME.space_xl)

        self.pages = QStackedWidget()
        self.capture_btn = SegmentedChoice(
            tuple((icon, label) for _key, icon, label, _hint in _CAPTURE),
            tr("What the negatives were scanned with"),
            data=tuple(key for key, *_ in _CAPTURE),
        )
        self.capture_hint = _text(THEME.text_secondary)
        self.light_btn = SegmentedChoice(
            tuple((icon, label) for _key, icon, label in _LIGHT),
            tr("The light behind the negative"),
            data=tuple(key for key, *_ in _LIGHT),
        )
        self.light_hint = _text(THEME.text_secondary)
        self.stock_hint = _text(THEME.text_secondary)
        self.pages.addWidget(self._page(tr("How Do You Scan?"), self.capture_btn, self.capture_hint))
        self.pages.addWidget(self._page(tr("What Light Source?"), self.light_btn, self.light_hint, self.stock_hint))
        root.addWidget(self.pages)

        self.capture_btn.setCurrentIndex(max(self.capture_btn.findData(self._capture), 0))
        self.light_btn.setCurrentIndex(max(self.light_btn.findData(self._light), 0))
        self.capture_btn.currentChanged.connect(lambda i: self._set_capture(_CAPTURE[i][0]))
        self.light_btn.currentChanged.connect(lambda i: self._set_light(_LIGHT[i][0]))

        self.summary = _text(THEME.text_primary)
        root.addWidget(self.summary)
        root.addLayout(self._build_footer())

        self._sync()
        remember_dialog_geometry(self, repo, "scan_setup")

    def _page(self, title: str, choice: SegmentedChoice, *hints: QLabel) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(THEME.space_md)

        header = QLabel(title)
        header.setStyleSheet(f"color: {THEME.text_primary}; font-size: {THEME.font_size_title}px; font-weight: bold;")
        layout.addWidget(header)
        layout.addWidget(choice)
        for hint in hints:
            layout.addWidget(hint)
        layout.addStretch()
        return page

    def _build_footer(self) -> QHBoxLayout:
        row = QHBoxLayout()
        self.back_btn = QPushButton(tr("← Back"))
        self.back_btn.clicked.connect(self._back)
        row.addWidget(self.back_btn)
        row.addStretch()
        cancel = QPushButton(tr("Cancel"))
        cancel.clicked.connect(self.reject)
        self.next_btn = QPushButton()
        self.next_btn.clicked.connect(self._advance)
        row.addWidget(cancel)
        row.addWidget(self.next_btn)
        pin_dialog_default(self.next_btn, cancel, self.back_btn)
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
        """Refresh the hints, footer and summary for the current page and answers."""
        self.capture_hint.setText(_CAPTURE[self.capture_btn.currentIndex()][3])
        self.light_hint.setText(_LIGHT_HINT[self._capture][self._light])

        on_light_page = self.pages.currentIndex() == 1
        self.back_btn.setVisible(on_light_page)
        self.next_btn.setText(tr("Apply") if on_light_page else tr("Next →"))
        linear_raw, paper = scan_setup_values(self._capture, self._light)
        self.summary.setText(
            tr(
                "Sets {decode} · Print stock {paper}",
                decode=tr("Linear RAW") if linear_raw else tr("Camera WB"),
                paper=PAPER_PROFILES[paper].label,
            )
            if on_light_page
            else tr("Two questions — they set the decode and the print stock for your rig.")
        )
        self.stock_hint.setText(_STOCK_HINT[self._light])

    def choice(self) -> dict:
        return {"capture": self._capture, "light": self._light}
