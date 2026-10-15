from typing import Any, Tuple

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import QGridLayout, QLabel, QProgressBar, QWidget

from negpy.desktop.view.styles.templates import toast_qss
from negpy.desktop.view.styles.theme import THEME
from negpy.domain.models import ExportResolutionMode
from negpy.features.geometry.models import AspectRatio

_DEFAULT_TOAST_MS = 2500
#: Toast wrapping bounds. The ratio keeps a long message clear of the corner pills, and
#: the floor stops it collapsing to a sliver on a narrow canvas.
_TOAST_WIDTH_RATIO = 0.55
_TOAST_MIN_WIDTH = 320
#: Horizontal padding in toast_qss(), so a one-line measurement matches the rendered pill.
_TOAST_PADDING = 40

INFO_OFF, INFO_BASIC, INFO_FULL = range(3)
INFO_LEVEL_LABELS = ("Off", "Basic", "Full")


def _film_line(meta: Any) -> str:
    film = meta.film.strip()
    if film and meta.push_pull and meta.film_iso:
        return f"{film} @ EI {int(meta.film_iso * 2**meta.push_pull)}"
    return film or (f"ISO {meta.film_iso}" if meta.film_iso else "")


def _camera_line(meta: Any) -> str:
    make, model = meta.camera_make.strip(), meta.camera_model.strip()
    camera = model if make and model.lower().startswith(make.lower()) else " ".join(p for p in (make, model) if p)
    lens = meta.lens_model.strip()
    if not lens and meta.focal_length_mm:
        lens = f"{meta.focal_length_mm:g}mm" + (f" f/{meta.max_aperture:g}" if meta.max_aperture else "")
    return " · ".join(p for p in (camera, lens) if p)


def _export_line(export: Any) -> str:
    mode = export.export_resolution_mode
    if mode == ExportResolutionMode.PRINT:
        size = f"Print {export.export_print_size:g} cm @ {export.export_dpi} dpi"
    elif mode == ExportResolutionMode.TARGET_PX:
        size = f"Export {export.export_target_long_edge_px} px"
    else:
        size = ""
    paper = export.paper_aspect_ratio
    paper = f"{paper} paper" if paper and paper not in (AspectRatio.ORIGINAL, AspectRatio.FREE) else ""
    return " · ".join(p for p in (size, paper) if p)


def frame_info(level: int, config: Any, filename: str, res: str, mode: str, position: str = "", proof: str = "") -> Tuple[str, str, str]:
    """Top-left, top-right and bottom-left pill text for an info level."""
    if level == INFO_OFF or not filename:
        return "", "", ""
    if level == INFO_BASIC:
        return " · ".join(s for s in (filename, res) if s), mode, ""
    head = " · ".join(s for s in (position, filename, res) if s)
    top_left = "\n".join(s for s in (head, _film_line(config.metadata), _camera_line(config.metadata)) if s)
    top_right = "\n".join(s for s in (mode, f"Proof: {proof}" if proof else "") if s)
    return top_left, top_right, _export_line(config.export)


_PILL_QSS = (
    f"color: {THEME.text_secondary}; font-size: {THEME.font_size_small}px; font-weight: 500; "
    f"background-color: {THEME.surface_pill}; border-radius: {THEME.radius_md}px; padding: 2px 8px;"
)


class CanvasHud(QWidget):
    """
    Translucent info layer floating over the image canvas: corner metadata pills,
    a transient top-center toast, and a thin progress bar along the top edge.
    Fully mouse-transparent so canvas pan/zoom/tools work underneath.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)

        layout = QGridLayout(self)
        layout.setContentsMargins(THEME.space_xl, THEME.space_xl, THEME.space_xl, THEME.space_xl)

        self.lbl_top_left = QLabel()
        self.lbl_top_right = QLabel()
        self.lbl_bottom_left = QLabel()
        self.lbl_bottom_right = QLabel()
        self.toast = QLabel()
        for lbl in (self.lbl_top_left, self.lbl_top_right, self.lbl_bottom_left, self.lbl_bottom_right):
            lbl.setStyleSheet(_PILL_QSS)
            lbl.hide()
        self.toast.setStyleSheet(toast_qss())
        # Wraps rather than running off the canvas: a message that has to explain itself, why a
        # file will not open and what to do about it, does not fit one line.
        self.toast.setWordWrap(True)
        self.toast.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.toast.hide()

        top = Qt.AlignmentFlag.AlignTop
        bottom = Qt.AlignmentFlag.AlignBottom
        layout.addWidget(self.lbl_top_left, 0, 0, top | Qt.AlignmentFlag.AlignLeft)
        self.lbl_top_right.setAlignment(Qt.AlignmentFlag.AlignRight)
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
        # A cap, not a width: wordWrap only wraps at the widget's width, and without a bound the
        # label grows to the canvas. Short toasts still size to content.
        self.toast.setMaximumWidth(max(_TOAST_MIN_WIDTH, int(self.width() * _TOAST_WIDTH_RATIO)))

    def set_bottom_clearance(self, px: int) -> None:
        """Keeps the bottom pills above the floating toolbar, whatever height it lays out at."""
        m = self.layout().contentsMargins()
        if m.bottom() != px:
            self.layout().setContentsMargins(m.left(), m.top(), m.right(), px)

    @staticmethod
    def _set_pill(lbl: QLabel, text: str) -> None:
        lbl.setText(text)
        lbl.setVisible(bool(text))

    def update_info(self, top_left: str, top_right: str, bottom_left: str) -> None:
        self._set_pill(self.lbl_top_left, top_left)
        self._set_pill(self.lbl_top_right, top_right)
        self._set_pill(self.lbl_bottom_left, bottom_left)

    def set_zoom_note(self, note: str) -> None:
        """Qualifier for what the canvas shows at this zoom."""
        self._set_pill(self.lbl_bottom_right, note)

    def showMessage(self, text: str, timeout: int = 0, kind: str = "info") -> None:
        """kind: "info" | "warning" | "error" — a failure must not read like a progress note."""
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
        self.toast.setStyleSheet(toast_qss(kind))
        self.toast.setText(text)
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
