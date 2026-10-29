from negpy.kernel.system.i18n import tr
from typing import Any, List, Tuple

from PyQt6.QtCore import QRectF, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QPainter
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QWidget

from negpy.desktop.view.styles.templates import TOP_ROW_HEIGHT, labeled_action
from negpy.desktop.view.styles.theme import THEME
from negpy.desktop.view.widgets.elided_label import ElidedLabel
from negpy.domain.models import ExportResolutionMode
from negpy.features.geometry.models import AspectRatio

_DEFAULT_MESSAGE_MS = 2500
_BAR_HEIGHT = 4
#: Up to this many items, the bar splits into one segment per item.
_MAX_SEGMENTS = 40
_SEGMENT_GAP = 2

INFO_OFF, INFO_BASIC, INFO_FULL = range(3)
INFO_LEVEL_LABELS = (tr("Off"), tr("Basic"), tr("Full"))
_ABORTING = tr("Aborting…")
_MESSAGE_COLORS = {"warning": THEME.warn_amber, "error": THEME.error_text}


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
        size = tr("Print {size:g} cm @ {dpi} dpi", size=export.export_print_size, dpi=export.export_dpi)
    elif mode == ExportResolutionMode.TARGET_PX:
        size = tr("Export {value} px", value=export.export_target_long_edge_px)
    else:
        size = ""
    paper = export.paper_aspect_ratio
    paper = tr("{ratio} paper", ratio=paper) if paper and paper not in (AspectRatio.ORIGINAL, AspectRatio.FREE) else ""
    return " · ".join(p for p in (size, paper) if p)


def frame_info(level: int, config: Any, filename: str, res: str, recipe: List[str], position: str = "", proof: str = "") -> Tuple[str, str]:
    """Left and right strip text for an info level: the frame on the left, its print recipe on the right.
    The caller passes the brief recipe at Basic and the whole one at Full."""
    if level == INFO_OFF or not filename:
        return "", ""
    if level == INFO_BASIC:
        return " · ".join(s for s in (filename, res) if s), " · ".join(recipe)
    meta = config.metadata
    left = " · ".join(s for s in (position, filename, res, _film_line(meta), _camera_line(meta)) if s)
    right = " · ".join(s for s in (*recipe, _export_line(config.export), tr("Proof: {value}", value=proof) if proof else "") if s)
    return left, right


class CanvasHud(QWidget):
    """
    Opaque strip docked above the canvas: frame info at the sides, the latest message or job
    in the middle, and job progress as a bar along its bottom edge. It hides when it has
    nothing to show, so Frame Info Off gives the canvas its full height at rest.
    """

    abort_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        # A minimum, not a fixed height: a long message wraps and the strip grows for it.
        self.setMinimumHeight(TOP_ROW_HEIGHT)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(THEME.space_xl, THEME.space_sm, THEME.space_xl, THEME.space_sm + _BAR_HEIGHT)
        layout.setSpacing(THEME.space_xl)

        side_qss = f"color: {THEME.text_secondary}; font-size: {THEME.font_size_small}px; background: transparent;"
        self.lbl_left = ElidedLabel()
        self.lbl_right = ElidedLabel()
        self.lbl_right.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.lbl_left.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        for lbl in (self.lbl_left, self.lbl_right):
            lbl.setStyleSheet(side_qss)

        # Wraps rather than eliding: a message that explains why a file will not open and what to
        # do about it does not fit one line, so the strip grows for it.
        self.message = QLabel()
        self.message.setWordWrap(True)
        self.message.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.message.hide()

        self.abort_button = labeled_action("", tr("Abort"), tr("Stop the running job."))
        self.abort_button.clicked.connect(self._on_abort)
        self.abort_button.hide()

        layout.addWidget(self.lbl_left, 1)
        layout.addWidget(self.message, 2)
        layout.addWidget(self.abort_button)
        layout.addWidget(self.lbl_right, 1)

        self._info = ("", "")
        self._zoom_note = ""
        self._done = 0
        self._total = 0

        self._message_timer = QTimer(self)
        self._message_timer.setSingleShot(True)
        self._message_timer.timeout.connect(lambda: self.showMessage(""))
        self._progress_timer = QTimer(self)
        self._progress_timer.setSingleShot(True)
        self._progress_timer.timeout.connect(self.hide_progress)
        self._sync()

    def progress_visible(self) -> bool:
        return self._total > 0

    def update_info(self, left: str, right: str) -> None:
        self._info = (left, right)
        self._sync()

    def set_zoom_note(self, note: str) -> None:
        """Qualifier for what the canvas shows at this zoom."""
        self._zoom_note = note
        self._sync()

    def showMessage(self, text: str, timeout: int = 0, kind: str = "info") -> None:
        """kind: "info" | "warning" | "error". A negative timeout keeps the message until
        clear_message; an empty text clears any message."""
        if not text:
            self._message_timer.stop()
            self.message.hide()
            self._sync()
            return
        color = _MESSAGE_COLORS.get(kind, THEME.text_primary)
        self.message.setStyleSheet(
            f"color: {color}; font-size: {THEME.font_size_base}px; font-weight: {THEME.weight_semibold}; background: transparent;"
        )
        self.message.setText(text)
        self.message.show()
        self._message_timer.stop()
        if timeout >= 0:
            self._message_timer.start(timeout or _DEFAULT_MESSAGE_MS)
        self._sync()

    def clear_message(self, text: str) -> None:
        """Clears the message only if it is still `text`, so a later message survives."""
        if text and self.message.text() == text and not self.message.isHidden():
            self.showMessage("")

    def set_abortable(self, on: bool) -> None:
        self.abort_button.setEnabled(True)
        self.abort_button.setVisible(on)
        if not on:
            self.clear_message(_ABORTING)
        self._sync()

    def _on_abort(self) -> None:
        # A job stops at its next item, so the strip says so until the batch ends.
        self.abort_button.setEnabled(False)
        self.showMessage(_ABORTING, -1)
        self.abort_requested.emit()

    def set_progress(self, current: int, total: int) -> None:
        if total <= 0:
            self.hide_progress()
            return
        self._progress_timer.stop()
        self._done, self._total = max(0, min(current, total)), total
        if current >= total:
            self._progress_timer.start(1000)
        self._sync()

    def hide_progress(self) -> None:
        self._progress_timer.stop()
        self._done = self._total = 0
        self._sync()

    def _sync(self) -> None:
        left, right = self._info
        self.lbl_left.setText(left)
        self.lbl_left.setToolTip(left)
        self.lbl_right.setText(" · ".join(s for s in (right, self._zoom_note) if s))
        self.setVisible(
            bool(left or right or self._zoom_note or self.message.isVisibleTo(self) or self._total or self.abort_button.isVisibleTo(self))
        )
        self.update()

    def paintEvent(self, a0) -> None:
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(THEME.bg_panel))
        p.fillRect(0, self.height() - 1, self.width(), 1, QColor(THEME.border_primary))
        if not self._total:
            return
        top = self.height() - _BAR_HEIGHT
        track, fill = QColor(THEME.border_primary), QColor(THEME.progress_fill)
        if self._total > _MAX_SEGMENTS:
            p.fillRect(0, top, self.width(), _BAR_HEIGHT, track)
            p.fillRect(QRectF(0, top, self.width() * self._done / self._total, _BAR_HEIGHT), fill)
            return
        step = (self.width() + _SEGMENT_GAP) / self._total
        for i in range(self._total):
            p.fillRect(QRectF(i * step, top, step - _SEGMENT_GAP, _BAR_HEIGHT), fill if i < self._done else track)
