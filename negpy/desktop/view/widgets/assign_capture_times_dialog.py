"""Assign Capture Times: a day, a time in film order, a time zone and a place for each frame of a
roll, from phone photos placed in the film strip between the frames they were taken between."""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

from PyQt6.QtCore import QDate, QRect, QRectF, QSize, Qt
from PyQt6.QtGui import QColor, QFont, QFontMetrics, QIcon, QImage, QKeySequence, QPainter, QPen, QPixmap, QShortcut
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QCalendarWidget,
    QDialog,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLineEdit,
    QListView,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QPushButton,
    QScrollArea,
    QStackedWidget,
    QStyle,
    QStyledItemDelegate,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
    QWidgetAction,
)

from negpy.desktop.view.styles.templates import (
    field_row,
    hint_label,
    icon_button,
    labeled_action,
    labeled_toggle,
    pin_dialog_default,
    section_subheader,
    set_hint_kind,
    wrap_tooltip,
)
from negpy.desktop.view.styles.theme import THEME
from negpy.desktop.view.widgets.capture_compare import CompareItem, CompareRow, PictureBox, QuickLook
from negpy.desktop.view.widgets.capture_strip import CaptureStrip, StripFrame, StripPhoto
from negpy.desktop.view.widgets.dialog_geometry import remember_dialog_geometry
from negpy.desktop.view.widgets.sliders import CompactSlider
from negpy.desktop.view.widgets.fold import group_fold
from negpy.desktop.view.widgets.granular_settings_dialog import build_scope_row
from negpy.desktop.view.widgets.location_picker_dialog import LocationPickerDialog
from negpy.desktop.view.widgets.searchable_gear_combo import SearchableGearCombo
from negpy.desktop.view.widgets.section_help_dialog import SectionHelpDialog, has_guide
from negpy.desktop.view.widgets.segmented_choice import SegmentedChoice
from negpy.desktop.view.widgets.step_combo import StepCombo
from negpy.desktop.workers.frame_pictures import FramePictures
from negpy.desktop.workers.reference_thumbnails import ReferenceThumbnails
from negpy.features.metadata.capture import place_summary
from negpy.features.metadata.capture_plan import CapturePlan, Place, PlanFrame, photo_moment
from negpy.features.process.models import ProcessMode
from negpy.features.metadata.capture_times import (
    AFTER_SCAN,
    DST_CHANGE,
    LONG_GAP_BETWEEN,
    NEXT_DAY,
    NO_DAY,
    NO_ZONE,
    OUT_OF_ORDER,
    SEGMENTS_OUT_OF_ORDER,
    SOURCE_AFTER,
    SOURCE_BEFORE,
    SOURCE_BETWEEN,
    SOURCE_START,
    SOURCE_TYPED,
    FrameTime,
    Issue,
)
from negpy.features.metadata.reference import read_reference
from negpy.features.metadata.timezone import offset_text, zone_names
from negpy.kernel.system.i18n import mark, tr
from negpy.kernel.system.text import count_of
from negpy.services.assets.thumbnails import asset_thumbnail_key

_GUIDE_KEY = "assign_capture_times"
_FRAME_PIXMAP = QSize(512, 512)  # the frame's thumbnail at its full size; the strip scales it
_REFERENCE_THUMB = 256
_PHOTO_LARGE = 1200
_FRAME_SIZES = (54, 80, 110, 150, 200)  # Frame Size 1 to 5: the picture height in the strip
_SIZE_KEY = "assign_capture_times_frame_size"
_LARGE_KEPT = 32  # large frame and photo pictures kept in memory
_QUEUE_ICON = QSize(64, 48)
_PHOTO_SUFFIXES = (".heic", ".heif", ".jpg", ".jpeg")
_PHOTO_FILTER = mark("Photos (*.heic *.HEIC *.heif *.jpg *.jpeg *.JPG *.JPEG)")
_STEP_CHOICES = (
    (mark("Auto"), 0),
    (mark("30 s"), 30),
    (mark("1 min"), 60),
    (mark("2 min"), 120),
    (mark("5 min"), 300),
    (mark("10 min"), 600),
    (mark("30 min"), 1800),
)
_COLUMNS = (mark("Frame"), mark("Capture time"), mark("Time zone"), mark("Place"), mark("Set by"), mark("Warnings"))
_CARD_ROLE = Qt.ItemDataRole.UserRole + 1
_FLAG_NAMES = {
    NEXT_DAY: mark("Next day"),
    AFTER_SCAN: mark("After the scan"),
    DST_CHANGE: mark("Daylight saving"),
    SEGMENTS_OUT_OF_ORDER: mark("Out of order"),
    LONG_GAP_BETWEEN: mark("Long gap"),
}


class _ReferenceCard(QStyledItemDelegate):
    """A reference photo: thumbnail, time, date and offset, place, file name and its place in the
    film strip. The lines come in the item's card role."""

    SIZE = QSize(178, 206)
    _THUMB_H = 96

    def sizeHint(self, _option, _index) -> QSize:  # noqa: N802
        return self.SIZE

    def paint(self, painter: QPainter, option, index) -> None:
        lines = index.data(_CARD_ROLE) or {}
        enabled = bool(option.state & QStyle.StateFlag.State_Enabled)
        selected = bool(option.state & QStyle.StateFlag.State_Selected)
        placed = lines.get("placed", "")
        rect = option.rect.adjusted(3, 3, -3, -3)
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        edge = THEME.border_checked if selected else (THEME.warn_amber if placed else THEME.border_input)
        painter.setPen(QPen(QColor(edge), 1))
        painter.setBrush(QColor(THEME.bg_checked if selected else THEME.bg_input))
        painter.drawRoundedRect(QRectF(rect).adjusted(0.5, 0.5, -0.5, -0.5), THEME.radius_md, THEME.radius_md)
        thumb = QRect(rect.x() + THEME.space_md, rect.y() + THEME.space_md, rect.width() - 2 * THEME.space_md, self._THUMB_H)
        icon = index.data(Qt.ItemDataRole.DecorationRole)
        if isinstance(icon, QIcon) and not icon.isNull():
            icon.paint(painter, thumb, Qt.AlignmentFlag.AlignCenter)
        else:
            painter.fillRect(thumb, QColor(THEME.bg_card))
        big = QFont(option.font)
        big.setPixelSize(THEME.font_size_header)
        big.setWeight(QFont.Weight(THEME.weight_semibold))
        small = QFont(option.font)
        small.setPixelSize(THEME.font_size_small)
        y = thumb.bottom() + THEME.space_md
        rows = [
            (lines.get("time", ""), big, THEME.text_primary if enabled else THEME.text_muted, 20),
            (lines.get("date", ""), small, THEME.text_hint, 16),
            (lines.get("place", ""), small, THEME.text_hint, 16),
            (lines.get("name", ""), small, THEME.text_secondary, 16),
            (placed, small, THEME.warn_amber, 16),
        ]
        for text, font, color, height in rows:
            if not text:
                continue
            painter.setFont(font)
            painter.setPen(QColor(color))
            line = QRect(rect.x() + THEME.space_lg, y, rect.width() - 2 * THEME.space_lg, height)
            painter.drawText(
                line,
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                QFontMetrics(font).elidedText(text, Qt.TextElideMode.ElideRight, line.width()),
            )
            y += height
        painter.restore()


_QUEUE_ROLE = Qt.ItemDataRole.UserRole + 2


class _QueueCard(QStyledItemDelegate):
    """A phone photo in the Compare queue: its picture, its time, and its place in the strip in the
    color of its state. The lines come in the item's queue role as (time, state, color)."""

    SIZE = QSize(_QUEUE_ICON.width() + 2 * THEME.space_2xl, _QUEUE_ICON.height() + 44)

    def sizeHint(self, _option, _index) -> QSize:  # noqa: N802
        return self.SIZE

    def paint(self, painter: QPainter, option, index) -> None:
        when, state, color = index.data(_QUEUE_ROLE) or ("", "", THEME.text_secondary)
        selected = bool(option.state & QStyle.StateFlag.State_Selected)
        rect = option.rect.adjusted(2, 2, -2, -2)
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(QPen(QColor(THEME.border_checked if selected else THEME.border_input), 1))
        painter.setBrush(QColor(THEME.bg_checked if selected else THEME.bg_input))
        painter.drawRoundedRect(QRectF(rect).adjusted(0.5, 0.5, -0.5, -0.5), THEME.radius_md, THEME.radius_md)
        thumb = QRect(rect.x() + THEME.space_sm, rect.y() + THEME.space_sm, rect.width() - 2 * THEME.space_sm, _QUEUE_ICON.height())
        icon = index.data(Qt.ItemDataRole.DecorationRole)
        if isinstance(icon, QIcon) and not icon.isNull():
            icon.paint(painter, thumb, Qt.AlignmentFlag.AlignCenter)
        font = QFont(option.font)
        font.setPixelSize(THEME.font_size_small)
        painter.setFont(font)
        metrics = QFontMetrics(font)
        y = thumb.bottom() + THEME.space_xs
        for text, tone in ((when, THEME.text_primary), (state, color)):
            line = QRect(rect.x() + THEME.space_sm, y, rect.width() - 2 * THEME.space_sm, metrics.height())
            painter.setPen(QColor(tone))
            painter.drawText(line, Qt.AlignmentFlag.AlignCenter, metrics.elidedText(text, Qt.TextElideMode.ElideRight, line.width()))
            y += metrics.height()
        painter.restore()


def _short_name(name: str) -> str:
    """The frame number of a scan name such as "26-11-14 R01 F037"."""
    parts = name.split()
    return parts[-1] if len(parts) > 1 else name


class AssignCaptureTimesDialog(QDialog):
    """Place phone photos between the frames of the strip, cut the roll where shooting stopped,
    then apply. `changes()` holds the MetadataConfig fields per frame after Apply."""

    def __init__(
        self,
        parent,
        frames_by_scope: dict[str, list[PlanFrame]],
        thumbnails: dict[str, QIcon],
        *,
        scope: str = "roll",
        scan_times: Optional[dict[str, datetime]] = None,
        repo=None,
        sources: Optional[dict[str, tuple[dict, Any]]] = None,
        bw_roll: bool = False,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(tr("Assign Capture Times"))
        self.setAcceptDrops(True)
        self._frames_by_scope = frames_by_scope
        self._thumbnails = thumbnails
        self._repo = repo
        self._scan_times = scan_times or {}
        self._results: dict[str, FrameTime] = {}
        self._issues: list[Issue] = []
        self._changes: dict[str, dict] = {}
        self._sel_frame = 0
        self._sel_photo = ""
        self._armed = ""
        self._notice = ""
        # Compare: the photo under comparison, the gap it is shown in, and the photos left out.
        self._view = "panels"
        self._cmp_path = ""
        self._cmp_gap = 0
        self._cmp_msg = ""
        self._skipped: set[str] = set()
        # Phone photos as read (small for cards and the strip, large for Compare and Quick Look),
        # and the pixmaps shown, which follow Photos in B&W.
        self._photo_small: dict[str, QImage] = {}
        self._photo_large: dict[str, QImage] = {}
        self._photo_shown: dict[tuple[str, int, bool], QPixmap] = {}
        self._asked_small: set[str] = set()
        self._asked_large: set[str] = set()
        # Frames rendered large with their edits, by frame key; failed ones keep their thumbnail.
        self._frame_large: dict[str, QPixmap] = {}
        self._frame_failed: set[str] = set()
        self._ql_item: Optional[tuple[str, str]] = None  # ("frame", key) or ("photo", path)
        self._loader = ReferenceThumbnails(self)
        self._loader.loaded.connect(self._on_reference_thumbnail)
        self._large_loader = ReferenceThumbnails(self)
        self._large_loader.loaded.connect(self._on_large_photo)
        self._pictures = FramePictures(sources, parent=self) if sources else None
        if self._pictures is not None:
            self._pictures.ready.connect(self._on_frame_picture)
        # The plan always holds the whole roll: values and photos on frames outside a selection
        # still shape the times. The scope only chooses the frames that Apply writes.
        self.plan = CapturePlan(frames_by_scope.get("roll", []))
        self._selection = {f.key for f in frames_by_scope.get("selection", [])}
        self._only: Optional[set[str]] = set(self._selection) if scope == "selection" else None

        root = QVBoxLayout(self)
        root.setContentsMargins(THEME.space_2xl, THEME.space_2xl, THEME.space_2xl, THEME.space_2xl)
        root.setSpacing(THEME.space_lg)
        root.addLayout(self._build_top_row(scope, bw_roll))
        strip_row, self.strip_fold = self._build_strip_fold()
        root.addLayout(strip_row)
        root.addWidget(self.strip_fold)
        self.pages = QStackedWidget()
        self.pages.addWidget(self._build_panels_page())
        self.pages.addWidget(self._build_compare_page())
        root.addWidget(self.pages, 1)
        self.status = hint_label()
        root.addWidget(self.status)
        root.addLayout(self._build_footer())
        self.quick_look = QuickLook(self)
        self.quick_look.closed.connect(self._on_quick_look_closed)
        self._build_keys()

        self._apply_frame_size(self._saved_frame_size())
        self._load_frame_pixmaps()
        self._refresh()
        self.resize(1100, 760)
        remember_dialog_geometry(self, repo, "assign_capture_times")

    # ── layout ────────────────────────────────────────────────────────────
    def _build_top_row(self, scope: str, bw_roll: bool) -> QHBoxLayout:
        sel = len(self._frames_by_scope.get("selection", []))
        roll = len(self._frames_by_scope.get("roll", []))
        row, self.scope_choice = build_scope_row(sel, roll)
        self.scope_choice.setCurrentIndex(self.scope_choice.findData(scope))
        self.scope_choice.currentChanged.connect(self._on_scope_changed)
        self.view_choice = SegmentedChoice(
            (("", tr("Frames and Photos")), ("", tr("Compare"))),
            tr("Edit the frame values, or compare each phone photo with the frames around it, large"),
            data=("panels", "compare"),
        )
        self.view_choice.currentChanged.connect(self._on_view_changed)
        self.direction_choice = SegmentedChoice(
            (("", tr("Forward")), ("", tr("Reverse"))),
            tr("Reverse when the scanner fed the film last frame first: the strip then shows the frames in the order you shot them"),
            data=("forward", "reverse"),
        )
        self.direction_choice.setCurrentIndex(1 if self.plan.reverse else 0)
        self.direction_choice.currentChanged.connect(self._on_direction_changed)
        # A button reads a single & as a mnemonic mark.
        label = tr("Photos in B&W").replace("&", "&&")
        self.bw_toggle = labeled_toggle("fa5s.adjust", label, bw_roll, tr("Show the phone photos in black and white, like the film"))
        self.bw_toggle.toggled.connect(self._on_bw_toggled)
        # Checked, the label turns semibold and outgrows the size hint of the regular weight.
        self.bw_toggle.setMinimumWidth(self.bw_toggle.sizeHint().width() + THEME.space_xl)
        # In line after the scope choice, before the row's stretch.
        for index, widget in enumerate((self.view_choice, self.direction_choice, self.bw_toggle)):
            row.insertSpacing(1 + 2 * index, THEME.space_2xl)
            row.insertWidget(2 + 2 * index, widget)
        return row

    def _build_strip_fold(self):
        self.strip_note = hint_label()
        self.strip_note.setWordWrap(False)
        self.size_slider = CompactSlider(tr("Frame Size"), 1, len(_FRAME_SIZES), 3, step=1, precision=1)
        self.size_slider.setToolTip(wrap_tooltip(tr("The height of the frames in the film strip; Ctrl and the mouse wheel also change it")))
        self.size_slider.setFixedWidth(240)
        self.size_slider.valueChanged.connect(lambda value: self._apply_frame_size(round(value)))
        row, fold = group_fold(
            "assign_capture_times_strip",
            tr("Film Strip"),
            self._build_strip(),
            self._build_action_bar(),
            buttons=(self.strip_note, self.size_slider),
            default_open=True,
        )
        fold.button.toggled.connect(self._on_strip_folded)
        return row, fold

    def _build_strip(self) -> QScrollArea:
        self.strip = CaptureStrip()
        self.strip.frameClicked.connect(self._on_frame_clicked)
        self.strip.cutClicked.connect(self._on_cut_clicked)
        self.strip.photoClicked.connect(self._on_strip_photo_clicked)
        self.strip.slotClicked.connect(self._on_slot_clicked)
        self.strip.segmentClicked.connect(self._on_segment_clicked)
        self.strip.photoDropped.connect(self._on_photo_dropped)
        self.strip.quickLookRequested.connect(self._on_strip_quick_look)
        self.strip.sizeStepped.connect(lambda step: self._step_frame_size(step))
        self.strip.set_gap_rule(self.plan.allowed_gaps)
        self.strip_scroll = QScrollArea()
        self.strip_scroll.setWidget(self.strip)
        self.strip_scroll.setWidgetResizable(False)
        self.strip_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.strip_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        # The segment labels follow the visible part of the strip.
        self.strip_scroll.horizontalScrollBar().valueChanged.connect(lambda _v: self.strip.update())
        return self.strip_scroll

    def _build_panels_page(self) -> QWidget:
        page = QWidget()
        col = QVBoxLayout(page)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(THEME.space_lg)
        body = QHBoxLayout()
        body.setSpacing(THEME.space_2xl)
        body.addWidget(self._build_frame_panel(), 2)
        body.addWidget(self._build_reference_panel(), 3)
        col.addLayout(body, 1)
        preview_row, self.preview_fold = self._build_preview()
        col.addLayout(preview_row)
        col.addWidget(self.preview_fold, 1)
        return page

    def _build_compare_page(self) -> QWidget:
        page = QWidget()
        col = QVBoxLayout(page)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(THEME.space_md)
        self.cmp_title = section_subheader("")
        self.cmp_detail = hint_label()
        col.addWidget(self.cmp_title)
        col.addWidget(self.cmp_detail)
        self.cmp_row = CompareRow()
        self.cmp_row.itemClicked.connect(self._on_compare_item)
        self.cmp_row.quickLookRequested.connect(self._quick_look)
        col.addWidget(self.cmp_row, 1)
        nav = QHBoxLayout()
        self.earlier_btn = labeled_action("fa5s.arrow-left", tr("Earlier Gap"), tr("Show the photo one gap earlier in the film strip"))
        self.later_btn = labeled_action("fa5s.arrow-right", tr("Later Gap"), tr("Show the photo one gap later in the film strip"))
        self.earlier_btn.clicked.connect(lambda: self._compare_move(-1))
        self.later_btn.clicked.connect(lambda: self._compare_move(1))
        middle = QVBoxLayout()
        middle.setSpacing(THEME.space_xs)
        self.cmp_gap_label = section_subheader("")
        self.cmp_segment = hint_label()
        for label in (self.cmp_gap_label, self.cmp_segment):
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            middle.addWidget(label)
        nav.addWidget(self.earlier_btn)
        nav.addLayout(middle, 1)
        nav.addWidget(self.later_btn)
        col.addLayout(nav)
        actions = QHBoxLayout()
        actions.setSpacing(THEME.space_md)
        self.place_btn = labeled_action(
            "fa5s.check", tr("Place Here"), tr("Put the photo in this gap, or take it out when it is already here"), primary=True
        )
        self.skip_btn = labeled_action("fa5s.forward", tr("Skip"), tr("Leave this photo out of the strip and show the next one"))
        self.place_btn.clicked.connect(self._compare_toggle)
        self.skip_btn.clicked.connect(self._compare_skip)
        for button in (self.earlier_btn, self.later_btn, self.place_btn, self.skip_btn):
            button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        actions.addWidget(self.place_btn)
        actions.addWidget(self.skip_btn)
        actions.addWidget(
            hint_label(
                tr(
                    "Keys: left and right arrows change the gap, Enter places or removes, S skips, up and down arrows change the photo, Space opens Quick Look."
                )
            ),
            1,
        )
        col.addLayout(actions)
        self.cmp_msg = hint_label()
        col.addWidget(self.cmp_msg)
        self.queue = QListWidget()
        self.queue.setViewMode(QListView.ViewMode.IconMode)
        self.queue.setFlow(QListView.Flow.LeftToRight)
        self.queue.setWrapping(False)
        self.queue.setMovement(QListView.Movement.Static)
        self.queue.setIconSize(_QUEUE_ICON)
        self.queue.setItemDelegate(_QueueCard(self.queue))
        self.queue.setUniformItemSizes(True)
        grid = _QueueCard.SIZE + QSize(THEME.space_md, THEME.space_md)
        self.queue.setGridSize(grid)
        self.queue.setFixedHeight(grid.height() + self.queue.horizontalScrollBar().sizeHint().height() + THEME.space_md)
        self.queue.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.queue.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.queue.itemClicked.connect(lambda item: self._compare_show(item.data(Qt.ItemDataRole.UserRole)))
        col.addWidget(self.queue)
        return page

    def _build_keys(self) -> None:
        """Keys of the dialog. The Compare and Quick Look keys only act while those are shown, so
        the arrows and Enter keep their usual work in the fields and lists."""

        def key(sequence, slot) -> QShortcut:
            shortcut = QShortcut(QKeySequence(sequence), self)
            shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
            shortcut.activated.connect(slot)
            return shortcut

        self._compare_keys = [
            key(Qt.Key.Key_Left, lambda: self._on_arrow(-1)),
            key(Qt.Key.Key_Right, lambda: self._on_arrow(1)),
            key(Qt.Key.Key_Up, lambda: self._compare_step(-1)),
            key(Qt.Key.Key_Down, lambda: self._compare_step(1)),
            key(Qt.Key.Key_Return, self._compare_toggle),
            key(Qt.Key.Key_Enter, self._compare_toggle),
            key(Qt.Key.Key_S, self._compare_skip),
        ]
        key(Qt.Key.Key_Space, self._on_space)
        for sequence, step in (("+", 1), ("=", 1), ("-", -1)):
            key(sequence, lambda step=step: self._step_frame_size(step))
        self._sync_keys()

    def _build_action_bar(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(THEME.space_md)
        self.action_text = hint_label()
        self.move_btn = labeled_action("fa5s.arrows-alt-h", tr("Move"), tr("Choose a new place in the film strip for this photo"))
        self.remove_btn = labeled_action("fa5s.times", tr("Remove from Strip"), tr("Take this photo out of the film strip"))
        self.stop_btn = labeled_action("fa5s.ban", tr("Stop Placing"), tr("Stop placing the chosen photo"))
        row.addWidget(self.action_text, 1)
        for widget in (self.move_btn, self.remove_btn, self.stop_btn):
            row.addWidget(widget)
        self.move_btn.clicked.connect(self._on_move)
        self.remove_btn.clicked.connect(self._on_remove)
        self.stop_btn.clicked.connect(self._disarm)
        return row

    def _build_frame_panel(self) -> QWidget:
        panel = QWidget()
        col = QVBoxLayout(panel)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(THEME.space_md)
        self.frame_header = section_subheader("")
        col.addWidget(self.frame_header)
        self.frame_reach = hint_label()
        col.addWidget(self.frame_reach)
        self.frame_picture = PictureBox(150)
        self.frame_picture.clicked.connect(
            lambda: self._quick_look("frame", self.plan.frames[self._position()].key) if self.plan.frames else None
        )
        col.addWidget(self.frame_picture)

        self.time_edit = QLineEdit()
        self.time_edit.setToolTip(wrap_tooltip(tr("A known time for this frame, in its time zone")))
        self.time_edit.editingFinished.connect(self._on_time_edited)
        self.time_clear_btn = icon_button("fa5s.times", tr("Remove the typed time"))
        self.time_clear_btn.clicked.connect(self._on_time_cleared)
        col.addLayout(field_row(tr("Time"), self.time_edit, self.time_clear_btn))

        self.day_edit = QLineEdit()
        self.day_edit.setToolTip(wrap_tooltip(tr("The day of this frame and the frames after it in the segment")))
        self.day_edit.editingFinished.connect(self._on_day_edited)
        self.day_calendar_btn = icon_button("fa5s.calendar-alt", tr("Pick the day on a calendar"))
        self.day_calendar_btn.clicked.connect(self._on_pick_day)
        self.day_clear_btn = icon_button("fa5s.times", tr("Follow the frame before"))
        self.day_clear_btn.clicked.connect(lambda: self._clear_own("day"))
        col.addLayout(field_row(tr("Day"), self.day_edit, self.day_calendar_btn, self.day_clear_btn))

        self.place_edit = QLineEdit()
        self.place_edit.setReadOnly(True)
        self.place_edit.setToolTip(wrap_tooltip(tr("A place for this frame and the frames after it in the segment")))
        self.place_map_btn = icon_button("fa5s.map-marked-alt", tr("Pick the capture place on a map (contacts OpenStreetMap)"))
        self.place_map_btn.clicked.connect(self._on_pick_place)
        self.place_clear_btn = icon_button("fa5s.times", tr("Follow the frame before"))
        self.place_clear_btn.clicked.connect(lambda: self._clear_own("place"))
        col.addLayout(field_row(tr("Place"), self.place_edit, self.place_map_btn, self.place_clear_btn))

        self.zone_combo = SearchableGearCombo(placeholder=tr("The zone of each frame's place"))
        self.zone_combo.set_labeled_items([(name, name) for name in zone_names()], "")
        self.zone_combo.setToolTip(
            tr("A time zone for this frame and the frames after it, or empty to take the zone of each frame's place")
        )
        self.zone_combo.selection_changed.connect(self._on_zone_chosen)
        self.zone_clear_btn = icon_button("fa5s.times", tr("Follow the frame before"))
        self.zone_clear_btn.clicked.connect(lambda: self._clear_own("zone"))
        col.addLayout(field_row(tr("Time zone"), self.zone_combo, self.zone_clear_btn))

        self.step_combo = StepCombo.of(
            [tr(label) for label, _ in _STEP_CHOICES],
            tr("The time from the frame before, outside the reference photos, for this frame and the frames after it"),
            [seconds for _, seconds in _STEP_CHOICES],
        )
        self.step_combo.currentIndexChanged.connect(self._on_step_changed)
        col.addLayout(field_row(tr("Time step"), self.step_combo))
        col.addWidget(
            hint_label(
                tr(
                    "Day and Time step time the frames with no photo nearby. Without a place here, a frame keeps the place it has, or takes the place and time zone of the photo nearest in time."
                )
            )
        )
        col.addStretch()
        return panel

    def _build_reference_panel(self) -> QWidget:
        panel = QWidget()
        col = QVBoxLayout(panel)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(THEME.space_md)
        head = QHBoxLayout()
        head.addWidget(section_subheader(tr("Reference Photos")))
        head.addStretch()
        self.days_choice = SegmentedChoice(
            (("", tr("Segment Day")), ("", tr("All Days"))), tr("Show the photos of the segment's day, or all photos"), data=("day", "all")
        )
        self.days_choice.currentChanged.connect(lambda _i: self._fill_references())
        head.addWidget(self.days_choice)
        self.add_photos_btn = labeled_action(
            "mdi6.file-image-outline", tr("Add Photos…"), tr("Add phone photos (HEIC or JPEG) of the same days")
        )
        self.add_photos_btn.clicked.connect(self._on_add_photos)
        head.addWidget(self.add_photos_btn)
        self.remove_photo_btn = labeled_action(
            "fa5s.trash-alt", tr("Remove Photo"), tr("Take the selected photo out of the list and the film strip")
        )
        self.remove_photo_btn.clicked.connect(self._on_remove_photo)
        head.addWidget(self.remove_photo_btn)
        col.addLayout(head)
        self.references = QListWidget()
        self.references.setViewMode(QListWidget.ViewMode.IconMode)
        self.references.setIconSize(QSize(_REFERENCE_THUMB, _REFERENCE_THUMB))
        self.references.setItemDelegate(_ReferenceCard(self.references))
        self.references.setGridSize(_ReferenceCard.SIZE + QSize(THEME.space_md, THEME.space_md))
        self.references.setUniformItemSizes(True)
        self.references.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.references.setMovement(QListWidget.Movement.Static)
        self.references.setWordWrap(True)
        self.references.setDragEnabled(True)
        self.references.setDragDropMode(QAbstractItemView.DragDropMode.DragOnly)
        self.references.itemClicked.connect(self._on_reference_clicked)
        col.addWidget(self.references, 1)
        col.addWidget(
            hint_label(
                tr("Click a photo, then a + in the film strip between the two frames it was taken between. You can also drag it there.")
            )
        )
        return panel

    def _build_preview(self) -> tuple[QHBoxLayout, QWidget]:
        self.preview_choice = SegmentedChoice(
            (("", tr("Frames Only")), ("", tr("With Phone Photos"))),
            tr("Show your phone photos between the frames, in the order photo apps sort them"),
            data=("frames", "with"),
        )
        self.preview_choice.setCurrentIndex(1)
        self.preview_choice.currentChanged.connect(lambda _i: self._fill_preview())
        self.table = QTableWidget(0, len(_COLUMNS))
        self.table.setHorizontalHeaderLabels([tr(c) for c in _COLUMNS])
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.cellClicked.connect(self._on_preview_clicked)
        row, fold = group_fold("assign_capture_times_preview", tr("Preview"), self.table, buttons=(self.preview_choice,))
        self.preview_summary = hint_label()
        row.insertWidget(1, self.preview_summary)
        if self._repo is not None:
            fold.bind(self._repo)
        return row, fold

    def _build_footer(self) -> QHBoxLayout:
        footer = QHBoxLayout()
        self.info_btn = icon_button("fa5s.info-circle", tr("How Assign Capture Times works"))
        self.info_btn.setVisible(has_guide(_GUIDE_KEY))
        self.info_btn.clicked.connect(lambda: SectionHelpDialog(_GUIDE_KEY, tr("Assign Capture Times"), self, repo=self._repo).exec())
        footer.addWidget(self.info_btn)
        footer.addStretch()
        cancel_btn = QPushButton(tr("Cancel"))
        self.apply_btn = QPushButton(tr("Apply"))
        cancel_btn.clicked.connect(self.reject)
        self.apply_btn.clicked.connect(self._on_apply)
        footer.addWidget(cancel_btn)
        footer.addWidget(self.apply_btn)
        pin_dialog_default(self.apply_btn, cancel_btn)
        return footer

    # ── names ─────────────────────────────────────────────────────────────
    def _frame_name(self, key: str) -> str:
        return next((_short_name(f.name) for f in self.plan.frames if f.key == key), key)

    def _full_name(self, key: str) -> str:
        return next((f.name for f in self.plan.frames if f.key == key), key)

    def _display(self, ident: str) -> str:
        """A frame key as its frame name; a reference name as it is."""
        return self._frame_name(ident)

    def _photo_name(self, path: str) -> str:
        return Path(path).name

    def _gap_frame(self, gap: int) -> int:
        return max(0, min(len(self.plan.frames) - 1, gap))

    def _where_text(self, path: str) -> str:
        gap, names = self.plan.gap_of(path), [_short_name(f.name) for f in self.plan.frames]
        photo = self._photo_name(path)
        if gap is None or not names:
            return photo
        if gap < 0:
            return tr("{photo} sits before {frame}.", photo=photo, frame=names[0])
        if gap >= len(names) - 1:
            return tr("{photo} sits after {frame}.", photo=photo, frame=names[-1])
        return tr("{photo} sits between {first} and {second}.", photo=photo, first=names[gap], second=names[gap + 1])

    def _source_text(self, result: Optional[FrameTime]) -> str:
        if result is None or not result.source:
            return "—"
        if result.kept:
            return tr("From the last Apply")
        names = [self._display(b) for b in result.basis]
        if result.source == SOURCE_BETWEEN and len(names) == 2:
            return tr("Between {first} and {second}", first=names[0], second=names[1])
        if result.source == SOURCE_BEFORE and names:
            return tr("Before {name}", name=names[0])
        if result.source == SOURCE_AFTER and names:
            return tr("After {name}", name=names[0])
        if result.source == SOURCE_TYPED:
            return tr("Typed time")
        if result.source == SOURCE_START:
            return tr("Day and Time step")
        return result.source

    def _zone_text(self, result: Optional[FrameTime]) -> str:
        if result is None or not result.zone:
            return "—"
        if result.offset is None:
            return result.zone
        return f"{result.zone} · UTC{offset_text(result.offset)}"

    def _gap_text(self, gap: int) -> str:
        names = [_short_name(f.name) for f in self.plan.frames]
        if not names:
            return ""
        if gap < 0:
            return tr("Before {name}", name=names[0])
        if gap >= len(names) - 1:
            return tr("After {name}", name=names[-1])
        return tr("Between {first} and {second}", first=names[gap], second=names[gap + 1])

    # ── state to view ─────────────────────────────────────────────────────
    def _refresh(self) -> None:
        self._results, self._issues = self.plan.compute(self._scan_times)
        self._flags = self._flags_by_id()
        self._fill_strip()
        self._fill_action_bar()
        if self._view == "compare":
            self._fill_compare()
        else:
            self._fill_frame_panel()
            self._fill_references()
            self._fill_preview()
        self._show_issues()

    def _flags_by_id(self) -> dict[str, list[tuple[str, str]]]:
        """(kind, text) marks per frame key or reference name."""
        flags: dict[str, list[tuple[str, str]]] = {}
        for issue in self._issues:
            if issue.code == OUT_OF_ORDER:
                for ident in issue.frames:
                    flags.setdefault(ident, []).append(("error", tr("Order conflict")))
            elif issue.code in _FLAG_NAMES:
                idents = issue.frames[2:] if issue.code == LONG_GAP_BETWEEN else issue.frames
                for ident in idents:
                    flags.setdefault(ident, []).append(("warning", tr(_FLAG_NAMES[issue.code])))
        return flags

    def _selected_segment(self) -> int:
        if not self.plan.frames:
            return -1
        if self._sel_photo:
            segment = self.plan.segment_of_photo(self._sel_photo)
            if segment is not None:
                return segment
        return self.plan.segment_at(min(self._sel_frame, len(self.plan.frames) - 1))

    def _fill_strip(self) -> None:
        compared = self.plan.photo(self._cmp_path) if self._view == "compare" else None
        frames = []
        for i, frame in enumerate(self.plan.frames):
            result = self._results.get(frame.key)
            marks = self._flags.get(frame.key, [])
            state = "error" if any(k == "error" for k, _ in marks) else ("warning" if marks else "")
            when = result.local.strftime("%Y-%m-%d %H:%M:%S") if result and result.local else tr("No time")
            frames.append(
                StripFrame(
                    name=_short_name(frame.name),
                    time=result.local.strftime("%H:%M") if result and result.local else "",
                    tooltip=f"{frame.name} · {when} · {self._source_text(result)}",
                    segment=self.plan.segment_at(i),
                    rejected=frame.rejected,
                    typed=frame.key in self.plan.typed,
                    state=state,
                    own=bool(self.plan.own.get(frame.key)),
                    muted=self._only is not None and frame.key not in self._only,
                )
            )
        gaps: dict[int, list[StripPhoto]] = {}
        for gap in sorted(set(self.plan.placed.values())):
            for photo in self.plan.refs_at(gap):
                name = self._photo_name(photo.path)
                when = photo.local.strftime("%Y-%m-%d %H:%M:%S") if photo.local else ""
                segment = self.plan.segment_of_photo(photo.path)
                gaps.setdefault(gap, []).append(
                    StripPhoto(
                        photo.path,
                        photo.local.strftime("%H:%M") if photo.local else "",
                        Path(photo.path).stem,
                        f"{when} · {self._where_text(photo.path)}",
                        segment=segment if segment is not None else self.plan.segment_at(self._gap_frame(gap)),
                        error=name in self._flags,
                    )
                )
                self._push_strip_photo(photo.path)
        if compared is not None and compared.local is not None and self.plan.gap_of(compared.path) != self._cmp_gap:
            # Compare shows the photo dashed where it would go, among the photos of that gap in time order.
            row = gaps.setdefault(self._cmp_gap, [])
            at = sum(1 for p in self.plan.refs_at(self._cmp_gap) if p.path != compared.path and photo_moment(p) <= photo_moment(compared))
            row.insert(
                at,
                StripPhoto(
                    compared.path, compared.local.strftime("%H:%M"), Path(compared.path).stem, self._gap_text(self._cmp_gap), ghost=True
                ),
            )
            self._push_strip_photo(compared.path)
        moving = self._armed if self._armed in self.plan.placed else ""
        self.strip.set_content(
            frames,
            gaps,
            {gap: self.plan.split_of(gap) or 0 for gap in self.plan.cuts},
            [self._segment_label(n) for n in range(len(self.plan.bounds()))],
            selected_frame=self._sel_frame,
            selected_photo=self._sel_photo,
            selected_segment=self._selected_segment(),
            armed=bool(self._armed),
            moving=moving,
            slot_range=self.plan.allowed_gaps(self._armed) if self._armed else None,
        )
        placed = len(self.plan.placed)
        self.strip_note.setText(
            tr("Phone photos in the strip: {count}", count=placed) if placed else tr("Click a reference photo, then a + in the strip.")
        )

    def _segment_label(self, segment: int) -> str:
        v = self.plan.values_at(self.plan.bounds()[segment][0])
        parts = [tr("Segment {n}", n=segment + 1)]
        if v.day:
            parts.append(v.day.isoformat())
        if v.place is not None:
            parts.append(v.place.city or v.place.country or place_summary("", "", "", v.place.latitude, v.place.longitude))
        return " · ".join(parts)

    def _fill_action_bar(self) -> None:
        armed, photo_selected = bool(self._armed), bool(self._sel_photo) and not self._armed
        frame_selected = not armed and not photo_selected and bool(self.plan.frames)
        if armed:
            if self._notice:
                text = self._notice
            elif self._armed in self.plan.placed:
                text = tr("Click a + in the film strip to move {photo}.", photo=self._photo_name(self._armed))
            else:
                text = tr(
                    "Click a + in the film strip to place {photo}. The frame before it was shot earlier, the frame after it later.",
                    photo=self._photo_name(self._armed),
                )
        elif photo_selected:
            text = self._where_text(self._sel_photo)
        elif frame_selected:
            frame = self.plan.frames[self._sel_frame]
            text = f"{_short_name(frame.name)}: {self._source_text(self._results.get(frame.key))}"
        else:
            text = ""
        self.action_text.setText(text)
        set_hint_kind(self.action_text, "warning" if armed else "muted")
        self.move_btn.setVisible(photo_selected)
        self.remove_btn.setVisible(photo_selected)
        self.stop_btn.setVisible(armed)

    def _fill_frame_panel(self) -> None:
        if not self.plan.frames:
            return
        pos = min(self._sel_frame, len(self.plan.frames) - 1)
        frame = self.plan.frames[pos]
        own, values = self.plan.own_of(pos), self.plan.values_at(pos)
        _, last = self.plan.bounds()[self.plan.segment_at(pos)]
        end = next((i - 1 for i in range(pos + 1, last + 1) if self.plan.own_of(i)), last)
        self.frame_header.setText(tr("Frame {frame}", frame=_short_name(frame.name)))
        self.frame_picture.set_pixmap(self._frame_pixmap(frame.key))
        self._want_frames([frame.key])
        if end > pos:
            self.frame_reach.setText(
                tr(
                    "A value set here applies to {first} to {last}.",
                    first=_short_name(frame.name),
                    last=_short_name(self.plan.frames[end].name),
                )
            )
        else:
            self.frame_reach.setText(tr("A value set here applies to this frame only."))
        for widget in (self.time_edit, self.day_edit, self.zone_combo, self.step_combo):
            widget.blockSignals(True)
        result = self._results.get(frame.key)
        typed = self.plan.typed.get(frame.key)
        self.time_edit.setText(typed.strftime("%H:%M:%S") if typed else "")
        self.time_edit.setPlaceholderText(result.local.strftime("%H:%M:%S") if result and result.local else tr("HH:MM or HH:MM:SS"))
        self.time_clear_btn.setEnabled(typed is not None)

        self._show(self.day_edit, own.get("day"), values.day, values.origins.get("day"), lambda d: d.isoformat(), tr("YYYY-MM-DD"))
        self.day_clear_btn.setEnabled("day" in own)

        def place_text(p: Place) -> str:
            return place_summary(p.city, p.state, p.country, p.latitude, p.longitude)

        self._show(
            self.place_edit, own.get("place"), values.place, values.origins.get("place"), place_text, tr("Own place, or the nearest photo")
        )
        self.place_clear_btn.setEnabled("place" in own)

        self.zone_combo.set_selected_id(own.get("zone", ""))
        if values.zone and "zone" not in own:
            zone_hint = tr("From {frame}: {value}", frame=self._frame_name(values.origins["zone"]), value=values.zone)
        elif not values.zone and self.plan.zone_at(pos):
            zone_hint = tr("From the place: {zone}", zone=self.plan.zone_at(pos))
        else:
            zone_hint = tr("The zone of each frame's place")
        self.zone_combo.line_edit().setPlaceholderText(zone_hint)
        self.zone_clear_btn.setEnabled("zone" in own)

        step_label = tr(_STEP_CHOICES[0][0])
        if values.step and "step" not in own:
            index = self.step_combo.findData(int(values.step.total_seconds()))
            step_label = tr("From {frame}: {value}", frame=self._frame_name(values.origins["step"]), value=self.step_combo.itemText(index))
        self.step_combo.setItemText(0, step_label)
        index = self.step_combo.findData(int(own["step"].total_seconds())) if "step" in own else 0
        self.step_combo.setCurrentIndex(index if index >= 0 else 0)
        for widget in (self.time_edit, self.day_edit, self.zone_combo, self.step_combo):
            widget.blockSignals(False)
        for edit in (self.day_edit, self.time_edit):
            edit.setStyleSheet("")

    def _show(self, edit: QLineEdit, own, value, origin: Optional[str], text, placeholder: str) -> None:
        """A field shows the frame's own value, else the value it follows as a grey hint."""
        if own is not None:
            edit.setText(text(own))
            edit.setPlaceholderText(placeholder)
        else:
            edit.setText("")
            follow = value is not None and origin is not None
            edit.setPlaceholderText(
                tr("From {frame}: {value}", frame=self._frame_name(origin), value=text(value)) if follow else placeholder
            )

    def _fill_references(self) -> None:
        day = None
        if self.plan.frames and self.days_choice.currentData() == "day":
            day = self.plan.values_at(min(self._sel_frame, len(self.plan.frames) - 1)).day
        photos = self.plan.references_on(day)
        self.references.blockSignals(True)
        self.references.clear()
        for photo in photos:
            card = self._card_lines(photo)
            item = QListWidgetItem(self._photo_icon(photo.path), f"{card['time']}\n{card['name']}")
            item.setData(Qt.ItemDataRole.UserRole, photo.path)
            item.setData(_CARD_ROLE, card)
            item.setToolTip(self._where_text(photo.path) if photo.path in self.plan.placed else self._photo_name(photo.path))
            if photo.local is None:
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEnabled)
            self.references.addItem(item)
            if photo.path in (self._armed, self._sel_photo):
                item.setSelected(True)
            self._request_thumbnail(photo.path)
        self.references.blockSignals(False)
        self.remove_photo_btn.setEnabled(bool(self._chosen_photo()))

    def _card_lines(self, photo) -> dict:
        date_text = ""
        if photo.local is not None:
            date_text = photo.local.strftime("%Y-%m-%d")
            if photo.offset is not None:
                date_text += f" · UTC{offset_text(photo.offset)}"
        place = place_summary("", "", "", photo.latitude, photo.longitude) if photo.latitude is not None else tr("No place")
        return {
            "time": photo.local.strftime("%H:%M:%S") if photo.local else tr("No capture time"),
            "date": date_text,
            "place": place,
            "name": self._photo_name(photo.path),
            "placed": self._placed_text(photo.path),
        }

    def _placed_text(self, path: str) -> str:
        gap, names = self.plan.gap_of(path), [_short_name(f.name) for f in self.plan.frames]
        if gap is None or not names:
            return ""
        if gap < 0:
            return tr("Before {name}", name=names[0])
        if gap >= len(names) - 1:
            return tr("After {name}", name=names[-1])
        return tr("Between {first} and {second}", first=names[gap], second=names[gap + 1])

    def _fill_preview(self) -> None:
        rows: list[tuple[datetime, int, str, str]] = []  # (moment, kind order, kind, ident)
        untimed = []
        for frame in self.plan.frames:
            result = self._results.get(frame.key)
            if result is None or result.local is None:
                untimed.append(frame.key)
            else:
                moment = result.local - result.offset if result.offset is not None else result.local
                rows.append((moment, 0, "frame", frame.key))
        if self.preview_choice.currentData() == "with":
            days = {self.plan.values_at(i).day for i in range(len(self.plan.frames))}
            for photo in self.plan.references:
                if photo.local is not None and (photo.local.date() in days or photo.path in self.plan.placed):
                    rows.append((photo_moment(photo), 1, "photo", photo.path))
            rows.sort(key=lambda r: (r[0], r[1]))
        else:
            order = {f.key: i for i, f in enumerate(self.plan.frames)}
            rows.sort(key=lambda r: order[r[3]])
        entries = [(kind, ident) for _, _, kind, ident in rows] + [("frame", key) for key in untimed]
        self.table.setRowCount(len(entries))
        hint = QColor(THEME.text_hint)
        for row, (kind, ident) in enumerate(entries):
            if kind == "frame":
                result = self._results.get(ident)
                marks = self._flags.get(ident, [])
                cells = [
                    self._full_name(ident),
                    result.capture_date_text() if result and result.local else "—",
                    self._zone_text(result),
                    place_summary("", "", "", result.latitude, result.longitude) if result and result.latitude is not None else "—",
                    self._source_text(result),
                    "; ".join(text for _, text in marks),
                ]
            else:
                photo = self.plan.photo(ident)
                cells = [
                    self._photo_name(ident),
                    photo.local.strftime("%Y-%m-%d %H:%M:%S") if photo and photo.local else "—",
                    f"UTC{offset_text(photo.offset)}" if photo and photo.offset is not None else "—",
                    place_summary("", "", "", photo.latitude, photo.longitude) if photo and photo.latitude is not None else "—",
                    tr("In the film strip") if ident in self.plan.placed else tr("Phone photo"),
                    "; ".join(text for _, text in self._flags.get(self._photo_name(ident), [])),
                ]
            for col, text in enumerate(cells):
                item = QTableWidgetItem(text)
                if col == 0:
                    item.setData(Qt.ItemDataRole.UserRole, (kind, ident))
                if kind == "photo":
                    item.setForeground(QColor(THEME.warn_amber) if ident in self.plan.placed and col == 4 else hint)
                elif self._only is not None and ident not in self._only:
                    item.setForeground(hint)
                self.table.setItem(row, col, item)
        errors = sum(1 for i in self._issues if i.is_error)
        warned = sum(1 for marks in self._flags.values() if any(k == "warning" for k, _ in marks))
        parts = [count_of(len(self._only) if self._only is not None else len(self.plan.frames), "frame")]
        if errors:
            parts.append(count_of(errors, "error"))
        if warned:
            parts.append(count_of(warned, "warning"))
        self.preview_summary.setText(" · ".join(parts))

    def _show_issues(self) -> None:
        errors = [i for i in self._issues if i.is_error]
        shown = errors or self._issues
        lines = list(dict.fromkeys(self._issue_text(i) for i in shown))
        self.status.setText("\n".join(lines[:3]))
        set_hint_kind(self.status, "error" if errors else ("warning" if shown else "muted"))
        self.apply_btn.setEnabled(not errors and bool(self.plan.frames))

    def _issue_text(self, issue: Issue) -> str:
        names = [self._display(k) for k in issue.frames]
        if issue.code == OUT_OF_ORDER:
            return tr(
                "{later} has an earlier time than {earlier}, which comes before it on the film. Check the photos or the direction.",
                later=names[0],
                earlier=names[1],
            )
        if issue.code == NO_DAY:
            return tr("Set a day for {frame} or a frame before it.", frame=names[0])
        if issue.code == NEXT_DAY:
            return tr("Frames that move to the next day: {count}.", count=len(names))
        if issue.code == DST_CHANGE:
            return tr("A typed time is in a daylight saving change. NegPy uses the first valid time.")
        if issue.code == NO_ZONE:
            return tr("Some frames have no place or time zone. Photo apps then use the time zone of the computer.")
        if issue.code == AFTER_SCAN:
            return tr("Frames with a time after their scan: {count}. Check the day.", count=len(names))
        if issue.code == SEGMENTS_OUT_OF_ORDER:
            return tr(
                "{later} has an earlier time than {earlier}, in the segment before. Check the days and photos.",
                later=names[0],
                earlier=names[1],
            )
        if issue.code == LONG_GAP_BETWEEN:
            hours = self._hours_between(issue.frames[0], issue.frames[1])
            frames = names[2:]
            if len(frames) == 1:
                return tr(
                    "{frame} lies between {earlier} and {later}, {hours} h apart. Split the roll where you stopped shooting.",
                    frame=frames[0],
                    earlier=names[0],
                    later=names[1],
                    hours=hours,
                )
            return tr(
                "{first} to {last} lie between {earlier} and {later}, {hours} h apart. Split the roll where you stopped shooting.",
                first=frames[0],
                last=frames[-1],
                earlier=names[0],
                later=names[1],
                hours=hours,
            )
        return issue.code

    def _hours_between(self, a: str, b: str) -> int:
        def moment(ident: str) -> Optional[datetime]:
            photo = next((p for p in self.plan.references if self._photo_name(p.path) == ident), None)
            if photo is not None:
                return photo_moment(photo)
            result = self._results.get(ident)
            if result is not None and result.local is not None:
                return result.local - result.offset if result.offset is not None else result.local
            return None

        first, second = moment(a), moment(b)
        return round(abs((second - first).total_seconds()) / 3600) if first and second else 0

    # ── pictures ──────────────────────────────────────────────────────────
    def _request_thumbnail(self, path: str) -> None:
        if path not in self._asked_small:
            self._asked_small.add(path)
            if Path(path).parent != Path("."):
                self._loader.request(path, _REFERENCE_THUMB)

    def _request_large_photo(self, path: str) -> None:
        if path not in self._asked_large:
            self._asked_large.add(path)
            if Path(path).parent != Path("."):
                self._large_loader.request(path, _PHOTO_LARGE)

    def _photo_pixmap(self, path: str, large: bool = False) -> Optional[QPixmap]:
        """The photo as shown: large when asked and loaded, else small; in black and white when
        Photos in B&W is on. None before it is read."""
        image = self._photo_large.get(path) if large else None
        image = image if image is not None else self._photo_small.get(path)
        if image is None or image.isNull():
            return None
        bw = self.bw_toggle.isChecked()
        key = (path, image.cacheKey(), bw)
        pixmap = self._photo_shown.get(key)
        if pixmap is None:
            pixmap = QPixmap.fromImage(image.convertToFormat(QImage.Format.Format_Grayscale8) if bw else image)
            self._photo_shown[key] = pixmap
        return pixmap

    def _photo_icon(self, path: str) -> QIcon:
        pixmap = self._photo_pixmap(path)
        return QIcon(pixmap) if pixmap is not None else QIcon()

    def _push_strip_photo(self, path: str) -> None:
        pixmap = self._photo_pixmap(path, large=True)
        if pixmap is not None:
            self.strip.set_photo_pixmap(path, pixmap)
        else:
            self._request_thumbnail(path)

    def _show_photo_everywhere(self, path: str) -> None:
        """A photo was read, or Photos in B&W changed: update every place that shows it."""
        icon = self._photo_icon(path)
        for i in range(self.references.count()):
            item = self.references.item(i)
            if item.data(Qt.ItemDataRole.UserRole) == path:
                item.setIcon(icon)
        for i in range(self.queue.count()):
            item = self.queue.item(i)
            if item.data(Qt.ItemDataRole.UserRole) == path:
                item.setIcon(icon)
        large = self._photo_pixmap(path, large=True)
        if large is not None:
            self.strip.set_photo_pixmap(path, large)
            self.cmp_row.set_pixmap("photo", path, large)
            if self._ql_item == ("photo", path):
                self.quick_look.set_pixmap(large)

    def _frame_pixmap(self, key: str) -> Optional[QPixmap]:
        """The frame rendered large when ready, else its thumbnail."""
        if key in self._frame_large:
            return self._frame_large[key]
        icon = self._thumbnails.get(key)
        return icon.pixmap(_FRAME_PIXMAP) if icon is not None and not icon.isNull() else None

    def _want_frames(self, keys: list[str]) -> None:
        """Render these frames large, in this order, when they are not yet."""
        if self._pictures is not None:
            self._pictures.request([k for k in keys if k not in self._frame_large and k not in self._frame_failed])

    def _keep_few(self, kept: dict) -> None:
        """Forget the oldest large pictures over the limit; a later look renders them again."""
        while len(kept) > _LARGE_KEPT:
            kept.pop(next(iter(kept)))

    def _on_frame_picture(self, key: str, image: QImage) -> None:
        if image.isNull():
            self._frame_failed.add(key)
            return
        pixmap = QPixmap.fromImage(image)
        self._frame_large[key] = pixmap
        self._keep_few(self._frame_large)
        position = self.plan.position_of(key)
        if position >= 0:
            self.strip.set_frame_pixmap(position, pixmap)
        self.cmp_row.set_pixmap("frame", key, pixmap)
        if self._ql_item == ("frame", key):
            self.quick_look.set_pixmap(pixmap)
        if self._view == "panels" and self.plan.frames and self.plan.frames[self._position()].key == key:
            self.frame_picture.set_pixmap(pixmap)

    def _on_large_photo(self, path: str, image: QImage) -> None:
        if image.isNull():
            return
        self._photo_large[path] = image
        self._keep_few(self._photo_large)
        self._show_photo_everywhere(path)

    def _load_frame_pixmaps(self) -> None:
        for i, frame in enumerate(self.plan.frames):
            pixmap = self._frame_pixmap(frame.key)
            if pixmap is not None:
                self.strip.set_frame_pixmap(i, pixmap)

    # ── frame size ────────────────────────────────────────────────────────
    def _saved_frame_size(self) -> int:
        saved = self._repo.get_global_setting(_SIZE_KEY) if self._repo is not None else None
        return saved if isinstance(saved, int) and 1 <= saved <= len(_FRAME_SIZES) else 3

    def _apply_frame_size(self, size: int) -> None:
        size = max(1, min(len(_FRAME_SIZES), size))
        if round(self.size_slider.value()) != size:
            self.size_slider.blockSignals(True)
            self.size_slider.setValue(size)
            self.size_slider.blockSignals(False)
        self.strip.set_thumb_height(_FRAME_SIZES[size - 1])
        self.strip_scroll.setFixedHeight(self.strip.height() + self.strip_scroll.horizontalScrollBar().sizeHint().height() + THEME.space_sm)
        if self._repo is not None:
            self._repo.save_global_setting(_SIZE_KEY, size)

    def _step_frame_size(self, step: int) -> None:
        self._apply_frame_size(round(self.size_slider.value()) + step)

    # ── Compare ───────────────────────────────────────────────────────────
    def _photos_in_order(self) -> list:
        """The photos that can be placed, in time order."""
        return [p for p in self.plan.references if p.local is not None]

    def _next_open(self, after: str) -> str:
        """The next photo in time order that is neither placed nor skipped, from the start again."""
        order = self._photos_in_order()
        start = next((i for i, p in enumerate(order) if p.path == after), -1)
        for k in range(1, len(order) + 1):
            photo = order[(start + k) % len(order)]
            if photo.path not in self.plan.placed and photo.path not in self._skipped:
                return photo.path
        return ""

    def _compare_bounds(self) -> tuple[int, int]:
        first, last = self.plan.allowed_gaps(self._cmp_path)
        return (-1, len(self.plan.frames) - 1) if first > last else (first, last)

    def _open_compare(self) -> None:
        self._view = "compare"
        order = self._photos_in_order()
        if order:
            chosen = self._chosen_photo()
            self._compare_show(chosen if self.plan.photo(chosen) in order else (self._next_open("") or order[0].path), refresh=False)
        else:
            self._cmp_path = ""
        self._armed, self._notice = "", ""
        self.pages.setCurrentIndex(1)
        self.strip_fold.set_open(False)
        self._sync_keys()
        self._refresh()

    def _leave_compare(self) -> None:
        if self._view != "compare":
            return
        self._view = "panels"
        if self.plan.frames:
            self._sel_frame = max(0, min(len(self.plan.frames) - 1, self._cmp_gap))
        self._sel_photo, self._cmp_msg = "", ""
        self.view_choice.blockSignals(True)
        self.view_choice.setCurrentIndex(self.view_choice.findData("panels"))
        self.view_choice.blockSignals(False)
        self.pages.setCurrentIndex(0)
        self.strip_fold.set_open(True)
        self._sync_keys()
        self._refresh()

    def _compare_show(self, path: str, refresh: bool = True) -> None:
        """Compare this photo, in its gap or where its time puts it."""
        if self.plan.photo(path) is None:
            return
        self._cmp_path, self._cmp_msg = path, ""
        self._cmp_gap = self.plan.suggested_gap(path, self._results)
        if refresh:
            self._refresh()
            self._scroll_to_compared()

    def _compare_move(self, step: int) -> None:
        if self._view != "compare" or not self._cmp_path:
            return
        first, last = self._compare_bounds()
        self._cmp_gap = max(first, min(last, self._cmp_gap + step))
        self._cmp_msg = ""
        self._refresh()
        self._scroll_to_compared()

    def _compare_toggle(self) -> None:
        """Enter: take the photo out when it is shown in its own gap, else put it here and go on to
        the next photo in time order."""
        path = self._cmp_path
        if self._view != "compare" or not path:
            return
        if self.plan.gap_of(path) == self._cmp_gap:
            self.plan.remove(path)
            self._refresh()
            return
        done = self.plan.move(path, self._cmp_gap) if path in self.plan.placed else self.plan.place(path, self._cmp_gap)
        if done:
            self._skipped.discard(path)
            self._results, self._issues = self.plan.compute(self._scan_times)
            following = self._next_open(path)
            if following:
                self._compare_show(following, refresh=False)
            else:
                self._cmp_msg = tr("All photos are placed or skipped. Check the frame values, then click Apply.")
        self._refresh()
        self._scroll_to_compared()

    def _compare_skip(self) -> None:
        path = self._cmp_path
        if self._view != "compare" or not path or path in self.plan.placed:
            return
        self._skipped.add(path)
        following = self._next_open(path)
        if following:
            self._compare_show(following, refresh=False)
        else:
            self._cmp_msg = tr("All photos are placed or skipped. Check the frame values, then click Apply.")
        self._refresh()

    def _compare_step(self, step: int) -> None:
        if self._view != "compare" or self.quick_look.isVisible():
            return
        order = self._photos_in_order()
        index = next((i for i, p in enumerate(order) if p.path == self._cmp_path), 0)
        target = max(0, min(len(order) - 1, index + step)) if order else -1
        if target >= 0 and order[target].path != self._cmp_path:
            self._compare_show(order[target].path)

    def _scroll_to_compared(self) -> None:
        rect = self.strip.rect_of("ghost", path=self._cmp_path) or self.strip.rect_of("photo", path=self._cmp_path)
        if rect is not None:
            self.strip_scroll.ensureVisible(rect.center().x(), rect.center().y(), self.strip_scroll.width() // 2, 0)

    def _compare_items(self) -> list[CompareItem]:
        """The frames around the gap, the photos in it and next to it, as in the strip."""
        g, n, frames = self._cmp_gap, len(self.plan.frames), self.plan.frames
        compared = self.plan.photo(self._cmp_path)

        def frame(i: int, role: str) -> CompareItem:
            if not 0 <= i < n:
                empty = "" if role == "context" else (tr("Start of the roll") if i < 0 else tr("End of the roll"))
                return CompareItem("frame", f"#{i}", role, empty=empty or " ")
            result = self._results.get(frames[i].key)
            when = result.local.strftime("%H:%M") if result and result.local else ""
            return CompareItem("frame", frames[i].key, role, _short_name(frames[i].name), "" if role == "context" else when)

        def photo(p, role: str) -> CompareItem:
            when = p.local.strftime("%H:%M") if p.local else ""
            if role == "current":
                return CompareItem("photo", p.path, role, self._photo_name(p.path), when)
            return CompareItem("photo", p.path, role, when, Path(p.path).stem if role == "placed" else "")

        in_gap = [p for p in self.plan.refs_at(g) if p.path != compared.path] + [compared]
        in_gap.sort(key=photo_moment)
        return [
            frame(g - 1, "context"),
            *(photo(p, "near") for p in self.plan.refs_at(g - 1) if p.path != compared.path),
            frame(g, "frame"),
            *(photo(p, "current" if p.path == compared.path else "placed") for p in in_gap),
            frame(g + 1, "frame"),
            *(photo(p, "near") for p in self.plan.refs_at(g + 1) if p.path != compared.path),
            frame(g + 2, "context"),
        ]

    def _fill_compare(self) -> None:
        compared = self.plan.photo(self._cmp_path)
        has_photo = compared is not None and compared.local is not None
        for widget in (self.earlier_btn, self.later_btn, self.place_btn):
            widget.setEnabled(has_photo)
        if not has_photo:
            self.cmp_title.setText(tr("No phone photos"))
            self.cmp_detail.setText(tr("Add photos under Frames and Photos, then compare them here."))
            self.cmp_row.set_items([])
            self.cmp_gap_label.setText("")
            self.cmp_segment.setText("")
            self.skip_btn.setVisible(False)
            self.queue.clear()
            return
        g, at = self._cmp_gap, self.plan.gap_of(compared.path)
        order = self._photos_in_order()
        self.cmp_title.setText(f"{self._photo_name(compared.path)} · {compared.local.strftime('%Y-%m-%d %H:%M:%S')}")
        place = place_summary("", "", "", compared.latitude, compared.longitude) if compared.latitude is not None else tr("No place")
        state = (
            self._placed_text(compared.path) if at is not None else (tr("Skipped") if compared.path in self._skipped else tr("Not placed"))
        )
        parts = [place]
        if compared.offset is not None:
            parts.append(f"UTC{offset_text(compared.offset)}")
        parts += [state, tr("Photo {n} of {count}", n=order.index(compared) + 1, count=len(order))]
        self.cmp_detail.setText(" · ".join(parts))

        items = self._compare_items()
        self.cmp_row.set_items(items)
        frame_keys = [i.ident for i in items if i.kind == "frame" and not i.empty]
        for item in items:
            if item.empty:
                continue
            if item.kind == "frame":
                pixmap = self._frame_pixmap(item.ident)
            else:
                self._request_large_photo(item.ident)
                pixmap = self._photo_pixmap(item.ident, large=True)
                if pixmap is None:
                    self._request_thumbnail(item.ident)
            if pixmap is not None:
                self.cmp_row.set_pixmap(item.kind, item.ident, pixmap)
        ahead = [f.key for f in self.plan.frames[max(0, g + 3) : g + 6]] + [f.key for f in self.plan.frames[max(0, g - 2) : max(0, g - 1)]]
        self._want_frames(frame_keys[1:-1] + frame_keys[:1] + frame_keys[-1:] + ahead)

        first, last = self._compare_bounds()
        self.earlier_btn.setEnabled(g > first)
        self.later_btn.setEnabled(g < last)
        self.cmp_gap_label.setText(self._gap_text(g))
        self.cmp_segment.setText(self._segment_label(self.plan.segment_at(self._gap_frame(g + 1 if g < 0 else g))))
        self.place_btn.setText(tr("Remove from Strip") if at == g else tr("Place Here"))
        self.skip_btn.setVisible(at is None)
        self.cmp_msg.setText(self._cmp_msg)
        self._fill_queue(order)

    def _fill_queue(self, order: list) -> None:
        self.queue.clear()
        for photo in order:
            gap = self.plan.gap_of(photo.path)
            names = [_short_name(f.name) for f in self.plan.frames]
            if gap is not None and 0 <= gap < len(names) - 1:
                state, color = f"{names[gap]}–{names[gap + 1]}", THEME.warn_amber
            elif gap is not None:
                state, color = self._gap_text(gap), THEME.warn_amber
            elif photo.path in self._skipped:
                state, color = tr("Skipped"), THEME.text_hint
            else:
                state, color = tr("Not placed"), THEME.text_secondary
            when = photo.local.strftime("%H:%M")
            item = QListWidgetItem(self._photo_icon(photo.path), f"{when} {state}")
            item.setData(Qt.ItemDataRole.UserRole, photo.path)
            item.setData(_QUEUE_ROLE, (when, state, color))
            item.setToolTip(f"{self._photo_name(photo.path)} · {photo.local.strftime('%Y-%m-%d %H:%M:%S')}")
            self.queue.addItem(item)
            if photo.path == self._cmp_path:
                item.setSelected(True)
                self.queue.scrollToItem(item)
            self._request_thumbnail(photo.path)

    # ── Quick Look ────────────────────────────────────────────────────────
    def _quick_look(self, kind: str, ident: str) -> None:
        """Show a frame (by key) or a photo (by path) as large as the dialog allows."""
        if kind == "frame":
            position = self.plan.position_of(ident)
            if position < 0:
                return
            frame = self.plan.frames[position]
            result = self._results.get(frame.key)
            when = result.local.strftime("%Y-%m-%d %H:%M:%S") if result and result.local else tr("No time")
            caption, detail = f"{frame.name} · {when}", self._source_text(result)
            pixmap = self._frame_pixmap(frame.key)
            neighbours = [f.key for f in self.plan.frames[max(0, position - 1) : position + 2] if f.key != frame.key]
            self._want_frames([frame.key, *neighbours])
        else:
            photo = self.plan.photo(ident)
            if photo is None:
                return
            when = photo.local.strftime("%Y-%m-%d %H:%M:%S") if photo.local else tr("No capture time")
            caption = f"{self._photo_name(ident)} · {when}"
            detail = self._placed_text(ident) or tr("Not placed")
            self._request_large_photo(ident)
            pixmap = self._photo_pixmap(ident, large=True)
        self._ql_item = (kind, ident)
        self.quick_look.show_picture(pixmap, caption, detail)
        self._sync_keys()

    def _quick_look_sequence(self) -> list[tuple[str, str]]:
        """Frames and placed photos in strip order."""
        sequence = [("photo", p.path) for p in self.plan.refs_at(-1)]
        for i, frame in enumerate(self.plan.frames):
            sequence.append(("frame", frame.key))
            sequence += [("photo", p.path) for p in self.plan.refs_at(i)]
        return sequence

    def _quick_look_step(self, step: int) -> None:
        sequence = self._quick_look_sequence()
        if self._ql_item in sequence:
            index = sequence.index(self._ql_item) + step
            if 0 <= index < len(sequence):
                self._quick_look(*sequence[index])

    def _on_quick_look_closed(self) -> None:
        self._ql_item = None
        self._sync_keys()

    def _on_strip_quick_look(self, kind: str, ident: str) -> None:
        if kind == "frame":
            self._quick_look("frame", self.plan.frames[int(ident)].key)
        else:
            self._quick_look("photo", ident)

    def _on_space(self) -> None:
        if self.quick_look.isVisible():
            self.quick_look.close_view()
            return
        item = self.strip.hovered_item() if self.strip_fold.is_open() else None
        if item is not None and item[0] == "frame":
            item = ("frame", self.plan.frames[int(item[1])].key)
        if item is None and self._view == "compare":
            item = self.cmp_row.hovered_item() or (("photo", self._cmp_path) if self._cmp_path else None)
        if item is None and self.plan.frames and self._view == "panels":
            item = ("frame", self.plan.frames[self._position()].key)
        if item is not None:
            self._quick_look(*item)

    def _on_arrow(self, step: int) -> None:
        if self.quick_look.isVisible():
            self._quick_look_step(step)
        else:
            self._compare_move(step)

    def _sync_keys(self) -> None:
        active = self._view == "compare" or self.quick_look.isVisible()
        for shortcut in self._compare_keys:
            shortcut.setEnabled(active)

    # ── view, size and B&W events ─────────────────────────────────────────
    def _on_view_changed(self, _index: int) -> None:
        if self.view_choice.currentData() == "compare":
            self._open_compare()
        else:
            self._leave_compare()

    def _on_strip_folded(self, open_: bool) -> None:
        if open_ and self._view == "compare":
            self._fill_strip()
            self._scroll_to_compared()

    def _on_bw_toggled(self, _on: bool) -> None:
        for path in set(self._photo_small) | set(self._photo_large):
            self._show_photo_everywhere(path)
        self._refresh()

    def _on_compare_item(self, kind: str, ident: str, role: str) -> None:
        """A side frame steps the photo toward it; a photo of the row is compared next; the large
        frames and the compared photo open in Quick Look."""
        if role == "context":
            self._compare_move(-1 if self.plan.position_of(ident) < self._cmp_gap else 1)
        elif kind == "photo" and role in ("placed", "near"):
            self._compare_show(ident)
        else:
            self._quick_look(kind, ident)

    def _scroll_to_photo(self, path: str) -> None:
        rect = self.strip.rect_of("photo", path=path)
        if rect is not None:
            self.strip_scroll.ensureVisible(rect.center().x(), rect.center().y(), rect.width(), 0)

    def _flag(self, edit: QLineEdit, valid: bool) -> None:
        edit.setStyleSheet("" if valid else f"border: 1px solid {THEME.error};")

    # ── placing photos ────────────────────────────────────────────────────
    def _arm(self, path: str) -> None:
        photo = self.plan.photo(path)
        if photo is None or photo.local is None:
            return
        self._armed = "" if self._armed == path else path
        self._notice = ""
        self._refresh()

    def _disarm(self) -> None:
        if self._armed in self.plan.placed:
            self._sel_photo = self._armed
        self._armed = ""
        self._notice = ""
        self._refresh()

    def _put(self, path: str, gap: int) -> None:
        done = self.plan.move(path, gap) if path in self.plan.placed else self.plan.place(path, gap)
        if done:
            self._armed, self._notice, self._sel_photo = "", "", path
            self._sel_frame = self._gap_frame(gap)
        elif self.plan.photo(path) is not None and not self.plan.keeps_order(path, gap):
            # Show the gaps that keep the time order instead.
            self._armed, self._notice = path, tr("A photo goes after the older photos and before the newer ones.")
        self._refresh()
        if done:
            self._scroll_to_photo(path)

    def _select_photo(self, path: str) -> None:
        self._sel_photo, self._armed, self._notice = path, "", ""
        gap = self.plan.gap_of(path)
        if gap is not None:
            self._sel_frame = self._gap_frame(gap)
        self._refresh()
        self._scroll_to_photo(path)

    # ── events ────────────────────────────────────────────────────────────
    def _on_scope_changed(self, _index: int) -> None:
        self._only = set(self._selection) if self.scope_choice.currentData() == "selection" else None
        self._refresh()

    def _reverse(self) -> bool:
        return self.direction_choice.currentData() == "reverse"

    def _on_direction_changed(self, _index: int) -> None:
        self.plan.set_reverse(self._reverse())
        self._sel_frame = len(self.plan.frames) - 1 - self._sel_frame if self.plan.frames else 0
        self._load_frame_pixmaps()
        self._refresh()

    def _on_frame_clicked(self, index: int) -> None:
        if self._view == "compare":
            # Compare next to the clicked frame, on the side that faces the photo.
            first, last = self._compare_bounds()
            gap = index - 1 if index > self._cmp_gap else index
            self._cmp_gap = max(first, min(last, gap))
            self._refresh()
            return
        if self._armed:
            self._notice = tr("Photos go between frames. Click a + next to the frame.")
            self._fill_action_bar()
            return
        self._sel_frame, self._sel_photo = index, ""
        self._refresh()

    def _on_cut_clicked(self, gap: int, photos_before: int) -> None:
        self.plan.toggle_cut(gap, photos_before)
        self._refresh()

    def _on_segment_clicked(self, segment: int) -> None:
        if self._armed:
            return
        self._sel_frame, self._sel_photo = self.plan.bounds()[segment][0], ""
        self._refresh()

    def _on_strip_photo_clicked(self, path: str) -> None:
        if self._view == "compare":
            self._compare_show(path)
            return
        if self._armed:
            return
        self._select_photo(path)

    def _on_slot_clicked(self, gap: int) -> None:
        if self._armed:
            self._put(self._armed, gap)

    def _on_photo_dropped(self, path: str, gap: int) -> None:
        self._put(path, gap)

    def _on_reference_clicked(self, item: QListWidgetItem) -> None:
        path = item.data(Qt.ItemDataRole.UserRole)
        if path in self.plan.placed:
            self._select_photo(path)
        else:
            self._arm(path)

    def _on_preview_clicked(self, row: int, _col: int) -> None:
        item = self.table.item(row, 0)
        if item is None:
            return
        kind, ident = item.data(Qt.ItemDataRole.UserRole)
        if kind == "photo":
            if ident in self.plan.placed:
                self._select_photo(ident)
            else:
                self._arm(ident)
            return
        self._sel_frame = next(i for i, f in enumerate(self.plan.frames) if f.key == ident)
        self._sel_photo, self._armed = "", ""
        self._refresh()

    def _on_move(self) -> None:
        if self._sel_photo:
            self._armed, self._notice = self._sel_photo, ""
            self._refresh()

    def _on_remove(self) -> None:
        if self._sel_photo:
            self.plan.remove(self._sel_photo)
            self._sel_photo = ""
            self._refresh()

    def _position(self) -> int:
        return min(self._sel_frame, len(self.plan.frames) - 1)

    def _clear_own(self, name: str) -> None:
        if self.plan.frames:
            self.plan.clear_own(self._position(), name)
            self._refresh()

    def _on_time_edited(self) -> None:
        if not self.plan.frames:
            return
        frame = self.plan.frames[self._position()]
        text = self.time_edit.text().strip()
        if not text:
            if frame.key in self.plan.typed:
                self._on_time_cleared()
            return
        clock = _parse_clock(text)
        result = self._results.get(frame.key)
        day = result.local.date() if result and result.local else self.plan.values_at(self._position()).day
        if clock is None or day is None:
            self._flag(self.time_edit, False)
            return
        self.plan.set_typed(frame.key, datetime.combine(day, clock))
        self._refresh()

    def _on_time_cleared(self) -> None:
        if self.plan.frames:
            self.plan.remove_typed(self.plan.frames[self._position()].key)
            self._refresh()

    def _on_day_edited(self) -> None:
        text = self.day_edit.text().strip()
        if not text:
            self._clear_own("day")
            return
        try:
            day = date.fromisoformat(text)
        except ValueError:
            self._flag(self.day_edit, False)
            return
        self._set_day(day)

    def _on_pick_day(self) -> None:
        current = self.plan.values_at(self._position()).day if self.plan.frames else None
        if current is None:
            current = next((p.local.date() for p in self.plan.references if p.local is not None), date.today())
        menu = QMenu(self)
        calendar = QCalendarWidget(menu)
        calendar.setGridVisible(False)
        calendar.setSelectedDate(QDate(current.year, current.month, current.day))
        action = QWidgetAction(menu)
        action.setDefaultWidget(calendar)
        menu.addAction(action)

        def chosen(picked: QDate) -> None:
            menu.close()
            self._set_day(date(picked.year(), picked.month(), picked.day()))

        calendar.clicked.connect(chosen)
        menu.exec(self.day_calendar_btn.mapToGlobal(self.day_calendar_btn.rect().bottomLeft()))

    def _set_day(self, day: date) -> None:
        if self.plan.frames:
            self.plan.set_own(self._position(), day=day)
            self._refresh()

    def _on_step_changed(self, _index: int) -> None:
        seconds = self.step_combo.currentData() or 0
        if seconds:
            self.plan.set_own(self._position(), step=timedelta(seconds=seconds))
            self._refresh()
        else:
            self._clear_own("step")

    def _on_zone_chosen(self, zone: str) -> None:
        if zone:
            self.plan.set_own(self._position(), zone=zone)
            self._refresh()
        else:
            self._clear_own("zone")

    def _on_pick_place(self) -> None:
        place = self.plan.values_at(self._position()).place
        if place is not None:
            dlg = LocationPickerDialog(
                place.latitude, place.longitude, place.city, place.state, place.country, parent=self, repo=self._repo
            )
        else:
            dlg = LocationPickerDialog(parent=self, repo=self._repo)
        if dlg.exec() != dlg.DialogCode.Accepted:
            return
        lat, lon, city, state, country = dlg.location()
        if lat is None or lon is None:
            self._clear_own("place")
            return
        self.plan.set_own(self._position(), place=Place(lat, lon, city, state, country))
        self._refresh()

    def _on_add_photos(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(self, tr("Add Reference Photos"), "", tr(_PHOTO_FILTER))
        self._add_photo_files(paths)

    def _add_photo_files(self, paths: list[str]) -> None:
        if not paths:
            return
        self.plan.add_references([read_reference(p) for p in paths])
        self._refresh()

    def _on_reference_thumbnail(self, path: str, image: QImage) -> None:
        if image.isNull():
            return
        self._photo_small[path] = image
        self._show_photo_everywhere(path)
        small = self._photo_pixmap(path)
        if small is not None and path not in self._photo_large:
            self.strip.set_photo_pixmap(path, small)
            self.cmp_row.set_pixmap("photo", path, small)

    def dragEnterEvent(self, event) -> None:  # noqa: N802
        if any(url.toLocalFile().lower().endswith(_PHOTO_SUFFIXES) for url in event.mimeData().urls()):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event) -> None:  # noqa: N802
        self._add_photo_files([url.toLocalFile() for url in event.mimeData().urls() if url.toLocalFile().lower().endswith(_PHOTO_SUFFIXES)])
        event.acceptProposedAction()

    def keyPressEvent(self, event) -> None:  # noqa: N802
        if event.key() == Qt.Key.Key_Escape:
            # Esc steps back one level: Quick Look, then Compare, then placing; only then it closes.
            if self.quick_look.isVisible():
                self.quick_look.close_view()
                return
            if self._view == "compare":
                self._leave_compare()
                return
            if self._armed:
                self._disarm()
                return
        if event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace) and self.references.hasFocus() and self._chosen_photo():
            self._on_remove_photo()
            return
        super().keyPressEvent(event)

    def _chosen_photo(self) -> str:
        """The photo the user has in hand: the one being placed, else the one selected."""
        return self._armed or self._sel_photo

    def _on_remove_photo(self) -> None:
        path = self._chosen_photo()
        if not path:
            return
        self.plan.remove_reference(path)
        self._armed, self._sel_photo, self._notice = "", "", ""
        self._refresh()

    def _on_apply(self) -> None:
        self._changes = self.plan.changes(self._results, only=self._only)
        self.accept()

    def done(self, result: int) -> None:
        self._loader.stop()
        self._large_loader.stop()
        if self._pictures is not None:
            self._pictures.stop()
        super().done(result)

    def changes(self) -> dict[str, dict]:
        return self._changes


def _parse_clock(text: str) -> Optional[time]:
    for fmt in ("%H:%M:%S", "%H:%M"):
        try:
            return datetime.strptime(text, fmt).time()
        except ValueError:
            continue
    return None


def open_assign_capture_times_dialog(parent, session) -> int:
    """Open the dialog for the selected frames (when there are several) or the whole roll, and
    write the result. Returns the frames changed."""
    state = session.state
    if state.selected_file_idx == -1:
        return 0
    files = state.uploaded_files
    roll = [i for i in session.asset_model.visible_actual_indices_ordered() if 0 <= i < len(files)]
    chosen = set(state.selected_indices) | {state.selected_file_idx}

    configs = {idx: state.config if idx == state.selected_file_idx else session.config_for_asset(files[idx]) for idx in roll}

    def plan_frame(idx: int) -> PlanFrame:
        asset = files[idx]
        return PlanFrame(
            key=asset["hash"],
            name=Path(asset.get("name") or asset["path"]).stem,
            metadata=configs[idx].metadata,
            rejected=bool(asset.get("excluded")),
        )

    frames_by_scope = {
        "roll": [plan_frame(i) for i in roll],
        "selection": [plan_frame(i) for i in roll if i in chosen],
    }
    # A frame is shot before its file is written: the modification time is an upper limit for the capture time.
    scan_times = {files[i]["hash"]: datetime.fromtimestamp(files[i]["mtime"], timezone.utc) for i in roll if files[i].get("mtime")}
    thumbnails = {}
    for idx in roll:
        icon = state.thumbnails.get(asset_thumbnail_key(files[idx]))
        if icon is not None:
            thumbnails[files[idx]["hash"]] = icon if isinstance(icon, QIcon) else QIcon(icon)
    # Each frame's file and edits, for the large pictures of Compare and Quick Look.
    sources = {files[idx]["hash"]: (files[idx], configs[idx]) for idx in roll}
    bw_roll = 2 * sum(1 for idx in roll if configs[idx].process.process_mode == ProcessMode.BW) > len(roll)
    scope = "selection" if len(frames_by_scope["selection"]) > 1 else "roll"
    dialog = AssignCaptureTimesDialog(
        parent, frames_by_scope, thumbnails, scope=scope, scan_times=scan_times, repo=session.repo, sources=sources, bw_roll=bw_roll
    )
    if dialog.exec() != QDialog.DialogCode.Accepted:
        return 0
    return session.assign_capture_times(dialog.changes())
