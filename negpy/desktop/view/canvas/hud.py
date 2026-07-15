from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import QGridLayout, QLabel, QProgressBar, QWidget

from negpy.desktop.view.styles.theme import THEME

_DEFAULT_TOAST_MS = 2500

_PILL_QSS = (
    f"color: {THEME.text_secondary}; font-size: {THEME.font_size_xs}px; font-weight: 500; "
    "background-color: rgba(0, 0, 0, 140); border-radius: 4px; padding: 2px 8px;"
)

# Status toast ("rendering...", "galleries updated"): unlike the passive corner
# pills, it announces app activity — bigger type, near-white on a solid dark
# plate with an outline so it reads against any canvas brightness.
_TOAST_QSS = (
    f"color: {THEME.text_primary}; font-size: {THEME.font_size_lg}px; font-weight: 600; "
    "background-color: rgba(10, 10, 10, 225); border: 1px solid rgba(255, 255, 255, 55); "
    "border-radius: 6px; padding: 7px 18px;"
)


class CanvasHud(QWidget):
    """
    Translucent info layer floating over the image canvas: corner metadata pills,
    a transient top-center toast, and a thin progress bar along the top edge.
    Fully mouse-transparent so canvas pan/zoom/tools work underneath.
    """

    # Clears the floating toolbar pill anchored to the canvas bottom.
    _BOTTOM_MARGIN = 72

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)

        layout = QGridLayout(self)
        layout.setContentsMargins(THEME.space_xl, THEME.space_xl, THEME.space_xl, self._BOTTOM_MARGIN)

        self.lbl_top_left = QLabel()
        self.lbl_top_right = QLabel()
        self.lbl_bottom_left = QLabel()
        self.lbl_bottom_right = QLabel()
        self.toast = QLabel()
        for lbl in (self.lbl_top_left, self.lbl_top_right, self.lbl_bottom_left, self.lbl_bottom_right):
            lbl.setStyleSheet(_PILL_QSS)
            lbl.hide()
        self.toast.setStyleSheet(_TOAST_QSS)
        self.toast.hide()

        top = Qt.AlignmentFlag.AlignTop
        bottom = Qt.AlignmentFlag.AlignBottom
        layout.addWidget(self.lbl_top_left, 0, 0, top | Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(self.lbl_top_right, 0, 2, top | Qt.AlignmentFlag.AlignRight)
        # Own row so a long toast never squeezes the corner pills.
        layout.addWidget(self.toast, 1, 0, 1, 3, top | Qt.AlignmentFlag.AlignHCenter)
        layout.addWidget(self.lbl_bottom_left, 2, 0, bottom | Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(self.lbl_bottom_right, 2, 2, bottom | Qt.AlignmentFlag.AlignRight)
        layout.setRowStretch(1, 1)
        layout.setColumnStretch(1, 1)

        self.progress = QProgressBar(self)
        self.progress.setTextVisible(False)
        self.progress.setStyleSheet(
            f"QProgressBar {{ background-color: {THEME.border_primary}; border: none; border-radius: 0; }}"
            f"QProgressBar::chunk {{ background-color: {THEME.accent_primary}; border-radius: 0; }}"
        )
        self.progress.hide()

        self._toast_timer = QTimer(self)
        self._toast_timer.setSingleShot(True)
        self._toast_timer.timeout.connect(self.toast.hide)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self.progress.setGeometry(0, 0, self.width(), 3)

    @staticmethod
    def _set_pill(lbl: QLabel, text: str) -> None:
        lbl.setText(text)
        lbl.setVisible(bool(text))

    def update_info(self, filename: str, res: str, mode: str, edits: str, tool: str, file_pos: str) -> None:
        self._set_pill(self.lbl_top_left, " · ".join(s for s in (filename, res) if s))
        self._set_pill(self.lbl_top_right, mode)
        self._set_pill(self.lbl_bottom_left, " · ".join(s for s in (edits, tool) if s))
        self._set_pill(self.lbl_bottom_right, file_pos)

    def showMessage(self, text: str, timeout: int = 0) -> None:
        if text == "Image Updated":
            return
        self.toast.setText(text.lower())
        self.toast.show()
        self._toast_timer.start(timeout if timeout > 0 else _DEFAULT_TOAST_MS)

    def set_progress(self, current: int, total: int) -> None:
        if total <= 0:
            self.progress.hide()
            return
        self.progress.show()
        if self.progress.maximum() != total:
            self.progress.setRange(0, total)
        self.progress.setValue(current)
        if current >= total:
            QTimer.singleShot(1000, self.progress.hide)

    def hide_progress(self) -> None:
        self.progress.hide()
