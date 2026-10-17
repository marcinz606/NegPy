from negpy.kernel.system.i18n import tr
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

from negpy.desktop.view.styles.templates import hint_label, pane_header_qss, pin_dialog_default
from negpy.desktop.view.styles.theme import THEME
from negpy.desktop.view.widgets.dialog_geometry import remember_dialog_geometry
from negpy.features.process.models import scan_setup_values

_CAPTURE = (
    ("camera", tr("Digital camera"), tr("A camera on a copy stand shooting the negative over a light source")),
    ("scanner", tr("Film scanner"), tr("A dedicated film scanner")),
)

# Only a camera is asked: a scanner decodes Linear RAW under any light.
_LIGHT = (
    ("white", tr("White light"), tr("A lightbox or high-CRI LED panel — one broadband exposure")),
    ("narrowband", tr("Narrowband RGB"), tr("A Scanlight or RGB-LED source — pure red, green and blue bands")),
)


class ScanSetupDialog(QDialog):
    """Wizard asking what rig the negatives were scanned on, so Camera WB or Linear RAW can be
    set from the answer instead of by trial and error."""

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
        self.capture_group = QButtonGroup(self)
        self.light_group = QButtonGroup(self)
        self.pages.addWidget(self._build_capture_page())
        self.pages.addWidget(self._build_light_page())
        root.addWidget(self.pages)

        self.summary = hint_label()
        root.addWidget(self.summary)
        root.addLayout(self._build_footer())

        self._sync()
        remember_dialog_geometry(self, repo, "scan_setup")

    def _build_capture_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(THEME.space_md)

        title = QLabel(tr("HOW DO YOU SCAN?"))
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

        title = QLabel(tr("WHAT LIGHT SOURCE?"))
        title.setStyleSheet(pane_header_qss())
        layout.addWidget(title)

        self.light_buttons: list[QRadioButton] = []
        for i, (key, label, blurb) in enumerate(_LIGHT):
            btn = QRadioButton(label)
            btn.setChecked(key == self._light)
            self.light_group.addButton(btn, i)
            self.light_buttons.append(btn)
            layout.addWidget(btn)
            layout.addWidget(hint_label(blurb))
        self.light_group.idToggled.connect(lambda i, on: self._set_light(_LIGHT[i][0]) if on else None)
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
        if self.pages.currentIndex() == 0 and self._capture == "camera":
            self.pages.setCurrentIndex(1)
            self._sync()
        else:
            self.accept()

    def _back(self) -> None:
        self.pages.setCurrentIndex(0)
        self._sync()

    def _sync(self) -> None:
        """Refresh the footer and the summary for the current page and answers."""
        on_light_page = self.pages.currentIndex() == 1
        last_page = on_light_page or self._capture != "camera"
        self.back_btn.setVisible(on_light_page)
        self.next_btn.setText(tr("Apply") if last_page else tr("Next →"))
        linear_raw = scan_setup_values(self._capture, self._light)
        self.summary.setText(
            (tr("Sets Linear RAW") if linear_raw else tr("Sets Camera WB"))
            if last_page
            else tr("The light source decides Camera WB or Linear RAW.")
        )

    def choice(self) -> dict:
        return {"capture": self._capture, "light": self._light}
