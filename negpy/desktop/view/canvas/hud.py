from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import QGridLayout, QLabel, QProgressBar, QWidget

from negpy.desktop.view.styles.theme import THEME

_DEFAULT_TOAST_MS = 2500
#: Toast wrapping bounds. The ratio keeps a long message clear of the corner pills, and
#: the floor stops it collapsing to a sliver on a narrow canvas.
_TOAST_WIDTH_RATIO = 0.55
_TOAST_MIN_WIDTH = 320
#: Horizontal padding in _TOAST_QSS, so a one-line measurement matches the rendered pill.
_TOAST_PADDING = 40

_PILL_QSS = (
    f"color: {THEME.text_secondary}; font-size: {THEME.font_size_xs}px; font-weight: 500; "
    "background-color: rgba(0, 0, 0, 140); border-radius: 4px; padding: 2px 8px;"
)

# Status toast ("rendering...", "galleries updated"). Unlike the passive corner pills it
# announces app activity, so it uses bigger type, near-white on a solid dark plate with
# an outline, and reads against any canvas brightness.
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
        # Wraps rather than running off the canvas: a message that has to explain itself, why a
        # file will not open and what to do about it, does not fit one line.
        self.toast.setWordWrap(True)
        self.toast.setAlignment(Qt.AlignmentFlag.AlignCenter)
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

        self._file_pos = ""
        self._zoom_note = ""

        self._toast_timer = QTimer(self)
        self._toast_timer.setSingleShot(True)
        self._toast_timer.timeout.connect(self.toast.hide)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self.progress.setGeometry(0, 0, self.width(), 3)
        # A cap, not a width: wordWrap only wraps at the widget's width, and without a bound the
        # label grows to the canvas. Short toasts still size to content.
        self.toast.setMaximumWidth(max(_TOAST_MIN_WIDTH, int(self.width() * _TOAST_WIDTH_RATIO)))

    @staticmethod
    def _set_pill(lbl: QLabel, text: str) -> None:
        lbl.setText(text)
        lbl.setVisible(bool(text))

    def update_info(self, filename: str, res: str, mode: str, edits: str, tool: str, file_pos: str) -> None:
        self._set_pill(self.lbl_top_left, " · ".join(s for s in (filename, res) if s))
        self._set_pill(self.lbl_top_right, mode)
        self._set_pill(self.lbl_bottom_left, " · ".join(s for s in (edits, tool) if s))
        self._file_pos = file_pos
        self._refresh_bottom_right()

    def set_zoom_note(self, note: str) -> None:
        """Qualifier for what the canvas shows at this zoom. It shares the bottom-right
        pill with the file position, which changes on a different schedule."""
        if note == self._zoom_note:
            return
        self._zoom_note = note
        self._refresh_bottom_right()

    def _refresh_bottom_right(self) -> None:
        self._set_pill(self.lbl_bottom_right, " · ".join(s for s in (self._zoom_note, self._file_pos) if s))

    def showMessage(self, text: str, timeout: int = 0) -> None:
        if text == "Image Updated":
            return
        if not text:  # a step that posted a long-lived toast has finished
            self._toast_timer.stop()
            self.toast.hide()
            return
        # Cap here as well as on resize, because a toast can be posted before the HUD has ever
        # been resized. Set the minimum too: a wrapping QLabel's sizeHint aims for a squarish
        # block, so a long message folds into narrow lines rather than using the width it is
        # allowed. Short toasts keep their natural width.
        cap = max(_TOAST_MIN_WIDTH, int(self.width() * _TOAST_WIDTH_RATIO))
        one_line = self.toast.fontMetrics().horizontalAdvance(text) + _TOAST_PADDING
        self.toast.setMaximumWidth(cap)
        self.toast.setMinimumWidth(min(cap, one_line))
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
