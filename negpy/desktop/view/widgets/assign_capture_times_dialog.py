"""Assign Capture Times: a day, a time in film order, a time zone and a place for each frame of a
roll, from phone photos placed in the film strip between the frames they were taken between."""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import QDate, QRect, QRectF, QSize, Qt
from PyQt6.QtGui import QColor, QFont, QFontMetrics, QIcon, QImage, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QCalendarWidget,
    QDialog,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QPushButton,
    QScrollArea,
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
    pin_dialog_default,
    section_subheader,
    set_hint_kind,
    wrap_tooltip,
)
from negpy.desktop.view.styles.theme import THEME
from negpy.desktop.view.widgets.capture_strip import HEIGHT as STRIP_HEIGHT
from negpy.desktop.view.widgets.capture_strip import CaptureStrip, StripFrame, StripPhoto
from negpy.desktop.view.widgets.dialog_geometry import remember_dialog_geometry
from negpy.desktop.view.widgets.fold import group_fold
from negpy.desktop.view.widgets.granular_settings_dialog import build_scope_row
from negpy.desktop.view.widgets.location_picker_dialog import LocationPickerDialog
from negpy.desktop.view.widgets.searchable_gear_combo import SearchableGearCombo
from negpy.desktop.view.widgets.section_help_dialog import SectionHelpDialog, has_guide
from negpy.desktop.view.widgets.segmented_choice import SegmentedChoice
from negpy.desktop.view.widgets.step_combo import StepCombo
from negpy.desktop.workers.reference_thumbnails import ReferenceThumbnails
from negpy.features.metadata.capture import place_summary
from negpy.features.metadata.capture_plan import CapturePlan, Place, PlanFrame, photo_moment
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
_FRAME_PIXMAP = QSize(160, 108)
_REFERENCE_THUMB = 128
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
        self._reference_icons: dict[str, QIcon] = {}
        self._sel_frame = 0
        self._sel_photo = ""
        self._armed = ""
        self._notice = ""
        self._loader = ReferenceThumbnails(self)
        self._loader.loaded.connect(self._on_reference_thumbnail)
        # The plan always holds the whole roll: values and photos on frames outside a selection
        # still shape the times. The scope only chooses the frames that Apply writes.
        self.plan = CapturePlan(frames_by_scope.get("roll", []))
        self._selection = {f.key for f in frames_by_scope.get("selection", [])}
        self._only: Optional[set[str]] = set(self._selection) if scope == "selection" else None

        root = QVBoxLayout(self)
        root.setContentsMargins(THEME.space_2xl, THEME.space_2xl, THEME.space_2xl, THEME.space_2xl)
        root.setSpacing(THEME.space_lg)
        root.addLayout(self._build_top_row(scope))
        root.addLayout(self._build_strip_header())
        root.addWidget(self._build_strip())
        root.addLayout(self._build_action_bar())
        body = QHBoxLayout()
        body.setSpacing(THEME.space_2xl)
        body.addWidget(self._build_frame_panel(), 2)
        body.addWidget(self._build_reference_panel(), 3)
        root.addLayout(body, 1)
        preview_row, self.preview_fold = self._build_preview()
        root.addLayout(preview_row)
        root.addWidget(self.preview_fold, 1)
        self.status = hint_label()
        root.addWidget(self.status)
        root.addLayout(self._build_footer())

        self._load_frame_pixmaps()
        self._refresh()
        self.resize(1100, 760)
        remember_dialog_geometry(self, repo, "assign_capture_times")

    # ── layout ────────────────────────────────────────────────────────────
    def _build_top_row(self, scope: str) -> QHBoxLayout:
        sel = len(self._frames_by_scope.get("selection", []))
        roll = len(self._frames_by_scope.get("roll", []))
        row, self.scope_choice = build_scope_row(sel, roll)
        self.scope_choice.setCurrentIndex(self.scope_choice.findData(scope))
        self.scope_choice.currentChanged.connect(self._on_scope_changed)
        self.direction_choice = SegmentedChoice(
            (("", tr("Forward")), ("", tr("Reverse"))),
            tr("Reverse when the scanner fed the film last frame first: the strip then shows the frames in the order you shot them"),
            data=("forward", "reverse"),
        )
        self.direction_choice.setCurrentIndex(1 if self.plan.reverse else 0)
        self.direction_choice.currentChanged.connect(self._on_direction_changed)
        row.addSpacing(THEME.space_2xl)
        row.addWidget(self.direction_choice)
        return row

    def _build_strip_header(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.addWidget(section_subheader(tr("Film Strip")))
        self.strip_note = hint_label()
        row.addWidget(self.strip_note, 1)
        return row

    def _build_strip(self) -> QScrollArea:
        self.strip = CaptureStrip()
        self.strip.frameClicked.connect(self._on_frame_clicked)
        self.strip.cutClicked.connect(self._on_cut_clicked)
        self.strip.photoClicked.connect(self._on_strip_photo_clicked)
        self.strip.slotClicked.connect(self._on_slot_clicked)
        self.strip.segmentClicked.connect(self._on_segment_clicked)
        self.strip.photoDropped.connect(self._on_photo_dropped)
        self.strip_scroll = QScrollArea()
        self.strip_scroll.setWidget(self.strip)
        self.strip_scroll.setWidgetResizable(False)
        self.strip_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.strip_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.strip_scroll.setFixedHeight(STRIP_HEIGHT + self.strip_scroll.horizontalScrollBar().sizeHint().height() + THEME.space_sm)
        return self.strip_scroll

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

    # ── state to view ─────────────────────────────────────────────────────
    def _refresh(self) -> None:
        self._results, self._issues = self.plan.compute(self._scan_times)
        self._flags = self._flags_by_id()
        self._fill_strip()
        self._fill_action_bar()
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
                icon = self._reference_icons.get(photo.path)
                if icon is not None and not icon.isNull():
                    self.strip.set_photo_pixmap(photo.path, icon.pixmap(QSize(_REFERENCE_THUMB, _REFERENCE_THUMB)))
                elif photo.path not in self._reference_icons:
                    self._request_thumbnail(photo.path)
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
            item = QListWidgetItem(self._reference_icons.get(photo.path, QIcon()), f"{card['time']}\n{card['name']}")
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

    def _request_thumbnail(self, path: str) -> None:
        if path not in self._reference_icons:
            self._reference_icons[path] = QIcon()
            if Path(path).parent != Path("."):
                self._loader.request(path, _REFERENCE_THUMB)

    def _load_frame_pixmaps(self) -> None:
        for i, frame in enumerate(self.plan.frames):
            icon = self._thumbnails.get(frame.key)
            if icon is not None and not icon.isNull():
                self.strip.set_frame_pixmap(i, icon.pixmap(_FRAME_PIXMAP))

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
        icon = QIcon(QPixmap.fromImage(image))
        self._reference_icons[path] = icon
        for i in range(self.references.count()):
            item = self.references.item(i)
            if item.data(Qt.ItemDataRole.UserRole) == path:
                item.setIcon(icon)
        if path in self.plan.placed:
            self.strip.set_photo_pixmap(path, icon.pixmap(QSize(_REFERENCE_THUMB, _REFERENCE_THUMB)))

    def dragEnterEvent(self, event) -> None:  # noqa: N802
        if any(url.toLocalFile().lower().endswith(_PHOTO_SUFFIXES) for url in event.mimeData().urls()):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event) -> None:  # noqa: N802
        self._add_photo_files([url.toLocalFile() for url in event.mimeData().urls() if url.toLocalFile().lower().endswith(_PHOTO_SUFFIXES)])
        event.acceptProposedAction()

    def keyPressEvent(self, event) -> None:  # noqa: N802
        if event.key() == Qt.Key.Key_Escape and self._armed:
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

    def plan_frame(idx: int) -> PlanFrame:
        asset = files[idx]
        config = state.config if idx == state.selected_file_idx else session.config_for_asset(asset)
        return PlanFrame(
            key=asset["hash"],
            name=Path(asset.get("name") or asset["path"]).stem,
            metadata=config.metadata,
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
    scope = "selection" if len(frames_by_scope["selection"]) > 1 else "roll"
    dialog = AssignCaptureTimesDialog(parent, frames_by_scope, thumbnails, scope=scope, scan_times=scan_times, repo=session.repo)
    if dialog.exec() != QDialog.DialogCode.Accepted:
        return 0
    return session.assign_capture_times(dialog.changes())
