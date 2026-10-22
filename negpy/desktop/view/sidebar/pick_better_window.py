"""Separate, modeless window for the Pick Better mode.

Shows three equal frames in a row: the current committed edit in the centre, bracketed by
the two proposed candidates (the stronger on the left, the weaker on the right). Each frame is
captioned above: "A" over the stronger left, "Skip" over the centre and "B" over the weaker
right. The in-flight parameter name floats above the centre "Skip" caption and is always shown.
A click on a candidate commits it; a click on the centre current photo rejects both candidates
and advances to the next proposal; Undo re-shows the pair the last pick came from; OK ends the
mode and closes the window. A hover draws a gray ring around the frame under the cursor.
A Difference slider widens or narrows the A/B spread live, and a Custom Routing button opens a
window that reorders which parameter each round proposes. The window floats over the editor panels
and each label letterboxes its frame, so a preview keeps its aspect ratio — it is never stretched
and never width-capped.
"""

from negpy.kernel.system.i18n import tr
from PyQt6.QtCore import Qt, QPoint, QTimer, pyqtSignal
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QDialog,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from negpy.desktop.view.sidebar.roi_image import RoiImageLabel, _RING_MARGIN
from negpy.desktop.view.styles.templates import (
    caption_label,
    labeled_action,
    pin_dialog_default,
)
from negpy.desktop.view.widgets.floating_panel import float_over_app
from negpy.desktop.view.widgets.sliders import CompactSlider

#: Gap between a caption's bottom edge and its photo's top edge.
_CAPTION_GAP = 4
#: The move summary's clear delay and its margin from the panel's top edge.
_MOVE_CLEAR_MS = 3000
_MOVE_TOP_MARGIN = 12


class PickBetterWindow(QDialog):
    """Modeless window for Pick Better: the current edit centred, the two candidates either side."""

    picked = pyqtSignal(int)  # side: 0 = left (stronger), 1 = right (weaker)
    rejected = pyqtSignal()
    undoRequested = pyqtSignal()
    closed = pyqtSignal()
    diffChanged = pyqtSignal(float)  # the Difference slider's A/B spread multiplier
    customRoutingClicked = pyqtSignal()  # open the Custom Routing window

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle(tr("Pick Better"))
        self.setModal(False)
        float_over_app(self)  # must precede the first show(): the panel flags hide a visible window
        self._user_resized = False  # a manual drag is remembered for the app session, then the aspect follow stops
        self._sizing = False  # a programmatic resize must not read as a user drag

        self._last_pw = 0  # the last frame's size; a show before set_images re-sizes to it
        self._last_ph = 0

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        # Three equal frames in a row: the current committed edit in the centre, bracketed by the
        # two candidates (the stronger on the left, the weaker on the right).
        self._image_panel = QWidget(self)
        image_row = QHBoxLayout(self._image_panel)
        image_row.setSpacing(10)
        self.left_label = RoiImageLabel()
        self.current_label = RoiImageLabel()
        self.right_label = RoiImageLabel()
        for label in (self.left_label, self.current_label, self.right_label):
            label.roi_mode = False  # a tap emits `clicked`, not a calibration crosshair
            label.hover_ring = True
        image_row.addWidget(self.left_label, 1)
        image_row.addWidget(self.current_label, 1)
        image_row.addWidget(self.right_label, 1)
        layout.addWidget(self._image_panel, 1)

        # The A/Skip/B captions are children of the panel, repositioned to each photo's top edge on
        # resize and on a new frame, so they hug the photo (a layout row would sit at the panel's top).
        self._cap_left = caption_label(tr("A"))
        self._cap_center = caption_label(tr("Skip"))
        self._cap_right = caption_label(tr("B"))
        for cap in (self._cap_left, self._cap_center, self._cap_right):
            cap.setParent(self._image_panel)
            cap.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)

        # The in-flight parameter name floats above the centre "Skip" caption, always shown. It
        # passes mouse events through, so a click in its band still reaches the photo beneath.
        self._name_label = caption_label("")
        self._name_label.setParent(self._image_panel)
        self._name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._name_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)

        # The move summary floats at the panel's top: the "Name +delta" line shown for a few
        # seconds after a left or right pick. A single-shot timer clears it.
        self._move_label = caption_label("")
        self._move_label.setParent(self._image_panel)
        self._move_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._move_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self._move_timer = QTimer(self)
        self._move_timer.setSingleShot(True)
        self._move_timer.timeout.connect(lambda: self._move_label.setText(""))

        # The photos inset from the panel's top by the reserved band — the move summary, the in-flight
        # name and the A/Skip/B captions — so a caption that hugs a photo's top edge sits clear of
        # both the move summary and the photo, even when a short window's letterbox lifts the photos
        # up toward the top.
        image_row.setContentsMargins(0, self._reserved_top(), 0, 0)

        # A control row under the photos: the Difference slider widens or narrows the A/B spread
        # live, and the Custom Routing button opens a window to reorder the round order.
        control_row = QHBoxLayout()
        self._diff_slider = CompactSlider(tr("Difference"), 0.25, 4.0, 1.0, step=0.05, unit="×")
        self._diff_slider.valueChanged.connect(self.diffChanged)
        self._custom_routing_btn = labeled_action(
            "fa5s.route", tr("Custom Routing"), tr("Open the window that reorders which parameter each round proposes")
        )
        self._custom_routing_btn.clicked.connect(self.customRoutingClicked)
        control_row.addWidget(self._diff_slider, 1)
        control_row.addWidget(self._custom_routing_btn)
        layout.addLayout(control_row)

        # Bottom-right: Undo (re-show the previous pair) and OK (close = end the mode).
        button_row = QHBoxLayout()
        button_row.addStretch(1)
        self.undo_btn = labeled_action("fa5s.undo", tr("Undo"), tr("Revert the last pick and re-show that pair"))
        self.ok_btn = labeled_action("fa5s.check", tr("OK"), tr("Finish Pick Better and close the window"), primary=True)
        button_row.addWidget(self.undo_btn)
        button_row.addWidget(self.ok_btn)
        layout.addLayout(button_row)

        self.left_label.clicked.connect(lambda _fx, _fy: self.picked.emit(0))
        self.right_label.clicked.connect(lambda _fx, _fy: self.picked.emit(1))
        self.current_label.clicked.connect(lambda _fx, _fy: self.rejected.emit())
        self.ok_btn.clicked.connect(lambda: self.close())
        self.undo_btn.clicked.connect(lambda: self.undoRequested.emit())

        # Pin OK as the dialog's one Enter target; Undo opts out.
        pin_dialog_default(self.ok_btn, self.undo_btn)

        self._schedule_reposition()

    def set_images(self, current: QPixmap, left: QPixmap, right: QPixmap) -> None:
        """Push the three rendered frames onto their labels (each letterboxes to fit). The window
        follows each new image's aspect ratio unless the user has resized it, which is then kept for
        the app session."""
        self.current_label.set_frame(current)
        self.left_label.set_frame(left)
        self.right_label.set_frame(right)
        if not current.isNull():
            pw, ph = current.width(), current.height()
            # Follow each new image's aspect unless the user has resized; a manual resize is kept for the app session.
            if not self._user_resized and (pw, ph) != (self._last_pw, self._last_ph):
                self._size_to_image(pw, ph)
            self._last_pw = pw
            self._last_ph = ph
        self._schedule_reposition()

    def set_param_name(self, text: str) -> None:
        """The always-on in-flight parameter name, shown above the centre Skip caption."""
        self._name_label.setText(text)
        self._schedule_reposition()

    def show_move(self, text: str) -> None:
        """Show the "Name +delta" move summary for a few seconds, then clear it."""
        self._move_label.setText(text)
        self._reposition_move_label()
        self._move_timer.start(_MOVE_CLEAR_MS)

    def _reposition_move_label(self) -> None:
        """Centre the move summary at the panel's top; it hides itself when empty."""
        if not self._move_label.text():
            self._move_label.hide()
            return
        self._move_label.adjustSize()
        self._move_label.show()
        x = (self._image_panel.width() - self._move_label.width()) // 2
        self._move_label.move(max(0, x), _MOVE_TOP_MARGIN)
        self._move_label.raise_()

    def _reposition_captions(self) -> None:
        """Sit each caption just above its photo's top edge, tracking a resize or a new frame; the
        photo's top is a paint-time letterbox position, not a fixed layout slot.

        The photos inset from the panel's top by the reserved band (the move summary, the in-flight
        name and the A/Skip/B captions), so a caption that hugs a photo's top edge sits clear of
        both the move summary and the photo, even when a short window's letterbox lifts the photos
        toward the top."""
        # Each caption's natural position sits just above its photo's top edge.
        cap_pos: dict[QLabel, tuple[int, int]] = {}
        for label, cap in (
            (self.left_label, self._cap_left),
            (self.current_label, self._cap_center),
            (self.right_label, self._cap_right),
        ):
            rect = label.frame_rect()
            if rect is None:
                cap.hide()
                continue
            cap.adjustSize()
            origin = label.mapTo(self._image_panel, QPoint(0, 0))
            x = origin.x() + rect.x() + (rect.width() - cap.width()) // 2
            y = origin.y() + rect.y() - cap.height() - _CAPTION_GAP
            cap_pos[cap] = (x, y)
        # The in-flight parameter name sits just above the centre Skip caption.
        name_visible = False
        name_x = name_y = 0
        if self._cap_center in cap_pos:
            cx, cy = cap_pos[self._cap_center]
            self._name_label.adjustSize()
            name_x = cx + (self._cap_center.width() - self._name_label.width()) // 2
            name_y = cy - self._name_label.height() - _CAPTION_GAP
            name_visible = True
        for cap, (x, y) in cap_pos.items():
            cap.show()
            cap.move(x, y)
        if name_visible:
            self._name_label.show()
            self._name_label.move(name_x, name_y)
            self._name_label.raise_()
        else:
            self._name_label.hide()
        self._reposition_move_label()

    def _line_height(self) -> int:
        """One caption's line height, the reserved height of the move summary's band at the panel top."""
        return self._cap_center.sizeHint().height()

    def _reserved_top(self) -> int:
        """The band at the panel's top the move summary, the in-flight name and the A/Skip/B captions
        occupy; the photos inset by this much so a caption that hugs a photo's top edge sits clear
        of both the move summary and the photo."""
        return _MOVE_TOP_MARGIN + 3 * _CAPTION_GAP + 3 * self._line_height()

    def _schedule_reposition(self) -> None:
        # A label gets its final size only after the layout runs, which is after this event
        # returns; defer so frame_rect() reads settled geometry, not a stale (too-small) size.
        QTimer.singleShot(0, self._reposition_captions)

    def resizeEvent(self, ev) -> None:
        super().resizeEvent(ev)
        # A spontaneous resize is a user drag, remembered for the app session; a programmatic resize from
        # _size_to_image is not, so the aspect follow keeps working until the user drags.
        if ev.spontaneous() and not self._sizing:
            self._user_resized = True
        self._schedule_reposition()

    def showEvent(self, ev) -> None:
        super().showEvent(ev)
        # Follow the image's aspect on each open unless the user has resized; a manual resize is kept
        # for the app session, and an app exit drops it (no disk persistence).
        if not self._user_resized and self._last_pw > 0:
            self._size_to_image(self._last_pw, self._last_ph)
        self._schedule_reposition()

    def _size_to_image(self, pw: int, ph: int) -> None:
        """Size the window so the three photos fill their frames at the largest scale that
        fits the monitor, with no black gap between them. A smaller monitor down-scales the
        photos rather than leaving black beside them. The photo row is measured at zero height
        so the window minimum size stays small and the final resize is not clamped by it."""
        ring = 2 * _RING_MARGIN
        gap = 10  # the image row's spacing between the three frames
        margin = 10  # the panel's left and right content margin
        labels = (self.left_label, self.current_label, self.right_label)
        for label in labels:
            label.setMinimumSize(0, 0)
        self.layout().activate()
        # The band the photos do not occupy: the reserved top, the control row, the button
        # row, and the panel's margins and inter-row spacing.
        fixed_h = self.layout().minimumSize().height()
        # Largest per-photo width that keeps the window inside the monitor in both axes; pw
        # (full resolution) when the full-res size already fits.
        screen = QApplication.primaryScreen()
        if screen is None:
            screens = QApplication.screens()
            screen = screens[0] if screens else None
        avail = screen.availableGeometry() if screen is not None else None
        s = float(pw)
        if avail is not None:
            s = min(s, (avail.width() - 2 * margin - 2 * gap - 3 * ring) / 3.0)
            s = min(s, (avail.height() - fixed_h - ring) * pw / ph)
        s = max(s, 1.0)
        w = int(3 * (s + ring) + 2 * gap + 2 * margin)
        h = int(s * ph / pw + ring + fixed_h)
        if avail is not None:
            w = min(w, avail.width())
            h = min(h, avail.height())
        parent = self.parent()
        if parent is not None:
            geo = parent.geometry()
            x = geo.x() + (geo.width() - w) // 2
            y = geo.y() + (geo.height() - h) // 2
            if avail is not None:
                x = max(avail.x(), min(x, avail.x() + avail.width() - w))
                y = max(avail.y(), min(y, avail.y() + avail.height() - h))
            self.move(x, y)
        for label in labels:
            label.setMinimumSize(0, 160)
        self._sizing = True
        self.resize(w, h)
        self._sizing = False

    def done(self, result: int) -> None:
        # Esc reaches here without a closeEvent, so the session cleanup hangs off done().
        self.closed.emit()
        super().done(result)
