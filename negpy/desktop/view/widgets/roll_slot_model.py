"""Qt model for scanner-addressable roll preview slots.

The model deliberately has no content-based filtering or sorting.  A preview is
an index of the scanner's physical slots, not a claim about how many exposures
are on the roll, so slot 40 remains slot 40 even when it is blank or partial.
"""

from dataclasses import dataclass, replace
from typing import Any, Sequence

from PyQt6.QtCore import QAbstractListModel, QByteArray, QModelIndex, QSize, Qt
from PyQt6.QtGui import QIcon, QImage, QPixmap

from negpy.infrastructure.scanners.settings import (
    DEFAULT_PREVIEW_METER_INSET_PERCENT,
    MAX_PREVIEW_METER_INSET_PERCENT,
)
from negpy.services.scanning.roll_preview_controls import validate_boundary_offset


_BOUNDARY_WARNING_TEXT = {
    "outside-index-raster": "falls outside the captured roll preview.",
    "broad-clear-region": "is in a clear region wider than a normal frame gap.",
    "narrow-gap-evidence": "has clear-gap evidence that is narrower than expected.",
    "no-local-gap-run": "has no clear film-gap evidence near its expected position.",
    "low-gap-evidence": "has weak film-gap evidence.",
}

_WARNING_DISPLAY_TEXT = {
    "low-content-support": "The preview contains too little image content to confirm this slot automatically.",
    "partial-index-coverage": "Only part of this slot is present in the captured roll preview.",
    "spacing-outlier": "The distance between this slot's boundaries differs from the roll's regular frame spacing.",
    "transport-origin-inferred": (
        "The scanner position was inferred from the roll's spacing and transport data rather than a clear local gap."
    ),
    "ambiguous-content-tail-boundary": (
        "NegPy found more than one plausible point where photographed content ends; this slot is in that uncertain range."
    ),
    "beyond-advisory-content-end": "This slot is past the last likely photographed frame detected in the preview.",
}

_BLOCKING_REVIEW_WARNINGS = frozenset({"transport-origin-inferred"})


def _display_warning(warning: str) -> str:
    for prefix, position in (("start-", "starting"), ("end-", "ending")):
        if warning.startswith(prefix):
            boundary_text = _BOUNDARY_WARNING_TEXT.get(warning.removeprefix(prefix))
            if boundary_text is not None:
                return f"The frame's {position} boundary {boundary_text}"
    boundary_text = _BOUNDARY_WARNING_TEXT.get(warning)
    if boundary_text is not None:
        return f"The boundary used to position the scanner {boundary_text}"
    display = _WARNING_DISPLAY_TEXT.get(warning)
    if display is not None:
        return display
    return f"Review required: {warning.replace('-', ' ')}."


def _display_warnings(warnings: tuple[str, ...]) -> tuple[str, ...]:
    """Translate current review codes and suppress duplicated boundary evidence."""

    contextual_boundary_reasons = {
        warning.removeprefix(prefix) for warning in warnings for prefix in ("start-", "end-") if warning.startswith(prefix)
    }
    return tuple(_display_warning(warning) for warning in warnings if warning not in contextual_boundary_reasons)


def _warning_state(
    warnings: tuple[str, ...],
    blocking_review_approved: bool = False,
) -> str:
    if any(warning in _BLOCKING_REVIEW_WARNINGS for warning in warnings):
        return "reviewed" if blocking_review_approved else "blocking"
    return "advisory" if warnings else "none"


@dataclass(frozen=True)
class RollPreviewSlot:
    """One 1-based, scanner-addressable preview slot.

    ``warnings`` are review annotations (for example, low coverage or a likely
    blank tail slot). They never hide a slot. An inferred transport origin
    requires an explicit thumbnail review before it can enter a scan request.
    ``boundary_offset`` is a scanner transport lookup adjustment retained for
    an explicit thumbnail reload; it does not crop the current image.
    """

    slot_id: int
    thumbnail: QIcon | QImage | QPixmap | None = None
    warnings: tuple[str, ...] = ()
    boundary_offset: int = 0
    confirmed_boundary_offset: int | None = None
    blocking_review_approved: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.slot_id, bool) or not isinstance(self.slot_id, int) or self.slot_id < 1:
            raise ValueError("slot_id must be a positive 1-based integer")

        warnings = tuple(self.warnings)
        if any(not isinstance(warning, str) or not warning.strip() for warning in warnings):
            raise ValueError("warnings must contain non-empty strings")
        object.__setattr__(self, "warnings", warnings)
        if type(self.blocking_review_approved) is not bool:
            raise ValueError("blocking_review_approved must be a boolean")
        if self.blocking_review_approved and not _BLOCKING_REVIEW_WARNINGS.intersection(warnings):
            raise ValueError("only a blocking preview warning can be reviewed")
        validate_boundary_offset(self.slot_id, self.boundary_offset)
        confirmed = self.confirmed_boundary_offset
        if confirmed is None:
            confirmed = self.boundary_offset
        validate_boundary_offset(self.slot_id, confirmed)
        object.__setattr__(self, "confirmed_boundary_offset", confirmed)


class RollSlotModel(QAbstractListModel):
    """Fixed-order model whose rows map exactly to physical scanner slots."""

    SLOT_ID_ROLE = int(Qt.ItemDataRole.UserRole) + 1
    WARNINGS_ROLE = SLOT_ID_ROLE + 1
    HAS_WARNINGS_ROLE = WARNINGS_ROLE + 1
    WARNING_STATE_ROLE = HAS_WARNINGS_ROLE + 1
    BOUNDARY_OFFSET_ROLE = WARNING_STATE_ROLE + 1
    BOUNDARY_OFFSET_CONFIRMED_ROLE = BOUNDARY_OFFSET_ROLE + 1
    METER_INSET_PERCENT_ROLE = BOUNDARY_OFFSET_CONFIRMED_ROLE + 1

    _CELL_SIZE = QSize(142, 112)

    def __init__(self, slots: Sequence[RollPreviewSlot] = (), parent=None) -> None:
        super().__init__(parent)
        self._slots: tuple[RollPreviewSlot, ...] = ()
        self._meter_inset_percent = DEFAULT_PREVIEW_METER_INSET_PERCENT
        self.replace_slots(slots)

    def replace_slots(self, slots: Sequence[RollPreviewSlot]) -> None:
        """Replace the preview while preserving the physical 1…N contract."""

        next_slots = tuple(slots)
        actual_ids = [slot.slot_id for slot in next_slots]
        expected_ids = list(range(1, len(next_slots) + 1))
        if actual_ids != expected_ids:
            raise ValueError("roll preview slots must be contiguous and ordered from 1 through N")

        self.beginResetModel()
        self._slots = next_slots
        self.endResetModel()

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        return 0 if parent.isValid() else len(self._slots)

    def data(self, index: QModelIndex, role: int = int(Qt.ItemDataRole.DisplayRole)) -> Any:
        if not index.isValid() or index.row() < 0 or index.row() >= len(self._slots):
            return None

        slot = self._slots[index.row()]
        if role == Qt.ItemDataRole.DisplayRole:
            return f"{slot.slot_id:02d}"
        if role == Qt.ItemDataRole.DecorationRole:
            return self._as_icon(slot.thumbnail)
        if role == Qt.ItemDataRole.ToolTipRole:
            offset_line = f"Film Spacing Offset: {slot.boundary_offset:+d}"
            if slot.boundary_offset != slot.confirmed_boundary_offset:
                offset_line += " (reload required)"
            if not slot.warnings:
                return f"Slot {slot.slot_id}\n{offset_line}\nNo preview warnings"
            warning_lines = "\n".join(f"• {warning}" for warning in _display_warnings(slot.warnings))
            state = _warning_state(slot.warnings, slot.blocking_review_approved)
            if state == "blocking":
                heading = "Blocking review:"
                consequence = (
                    "Review this inferred position before adding it to a scan request. "
                    "The live safety recheck still runs and may stop before full-quality capture."
                )
            elif state == "reviewed":
                heading = "Reviewed blocking warning:"
                consequence = (
                    "The operator reviewed this thumbnail. The live safety recheck still runs "
                    "and may stop before full-quality capture."
                )
            else:
                heading = "Advisory preview warning:"
                consequence = "This slot remains selectable; check the thumbnail before scanning."
            return f"Slot {slot.slot_id}\n{offset_line}\n{heading}\n{warning_lines}\n\n{consequence}"
        if role == Qt.ItemDataRole.AccessibleTextRole:
            state = _warning_state(slot.warnings, slot.blocking_review_approved)
            warning_suffix = {
                "none": "",
                "advisory": ", advisory preview warning",
                "blocking": ", blocking review required",
                "reviewed": ", blocking warning reviewed",
            }[state]
            return f"Roll slot {slot.slot_id}{warning_suffix}"
        if role == Qt.ItemDataRole.SizeHintRole:
            return self._CELL_SIZE
        if role == self.SLOT_ID_ROLE:
            return slot.slot_id
        if role == self.WARNINGS_ROLE:
            return slot.warnings
        if role == self.HAS_WARNINGS_ROLE:
            return bool(slot.warnings)
        if role == self.WARNING_STATE_ROLE:
            return _warning_state(slot.warnings, slot.blocking_review_approved)
        if role == self.BOUNDARY_OFFSET_ROLE:
            return slot.boundary_offset
        if role == self.BOUNDARY_OFFSET_CONFIRMED_ROLE:
            return slot.boundary_offset == slot.confirmed_boundary_offset
        if role == self.METER_INSET_PERCENT_ROLE:
            return self._meter_inset_percent
        return None

    def flags(self, index: QModelIndex) -> Qt.ItemFlag:
        if not index.isValid():
            return Qt.ItemFlag.NoItemFlags
        return Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable

    def roleNames(self) -> dict[int, QByteArray]:  # noqa: N802
        names = super().roleNames()
        names[self.SLOT_ID_ROLE] = QByteArray(b"slotId")
        names[self.WARNINGS_ROLE] = QByteArray(b"warnings")
        names[self.HAS_WARNINGS_ROLE] = QByteArray(b"hasWarnings")
        names[self.WARNING_STATE_ROLE] = QByteArray(b"warningState")
        names[self.BOUNDARY_OFFSET_ROLE] = QByteArray(b"boundaryOffset")
        names[self.BOUNDARY_OFFSET_CONFIRMED_ROLE] = QByteArray(b"boundaryOffsetConfirmed")
        names[self.METER_INSET_PERCENT_ROLE] = QByteArray(b"meterInsetPercent")
        return names

    def set_meter_inset_percent(self, value: int) -> None:
        """Set one display-only metering guide for the complete contact sheet."""

        if type(value) is not int or not 0 <= value <= MAX_PREVIEW_METER_INSET_PERCENT:
            raise ValueError("metering crop must be an integer from 0 to 30 percent per edge")
        if value == self._meter_inset_percent:
            return
        self._meter_inset_percent = value
        if not self._slots:
            return
        self.dataChanged.emit(
            self.index(0, 0),
            self.index(len(self._slots) - 1, 0),
            [self.METER_INSET_PERCENT_ROLE],
        )

    def slot_id_for_row(self, row: int) -> int:
        if row < 0 or row >= len(self._slots):
            raise IndexError("slot row is outside the preview")
        return self._slots[row].slot_id

    def row_for_slot_id(self, slot_id: int) -> int:
        """Return the zero-based row for a physical slot ID."""

        if isinstance(slot_id, bool) or not isinstance(slot_id, int) or not 1 <= slot_id <= len(self._slots):
            raise ValueError(f"unknown roll slot: {slot_id!r}")
        return slot_id - 1

    def boundary_offset_for_slot_id(self, slot_id: int) -> int:
        """Return the persisted thumbnail boundary offset for one slot."""

        return self._slots[self.row_for_slot_id(slot_id)].boundary_offset

    def set_boundary_offset(self, slot_id: int, boundary_offset: int) -> None:
        """Persist one slot offset without resetting model or view identity."""

        row = self.row_for_slot_id(slot_id)
        boundary_offset = validate_boundary_offset(slot_id, boundary_offset)
        slot = self._slots[row]
        if slot.boundary_offset == boundary_offset:
            return
        approval_was_reset = slot.blocking_review_approved
        self._replace_slot(
            row,
            replace(
                slot,
                boundary_offset=boundary_offset,
                blocking_review_approved=False,
            ),
        )
        index = self.index(row, 0)
        roles = [self.BOUNDARY_OFFSET_ROLE, int(Qt.ItemDataRole.ToolTipRole)]
        if approval_was_reset:
            roles.extend(
                [
                    self.WARNING_STATE_ROLE,
                    int(Qt.ItemDataRole.AccessibleTextRole),
                ]
            )
        self.dataChanged.emit(
            index,
            index,
            roles,
        )

    def set_blocking_review_approved(self, slot_id: int, approved: bool) -> None:
        """Record an operator review for one inferred-position thumbnail."""

        if type(approved) is not bool:
            raise ValueError("approved must be a boolean")
        row = self.row_for_slot_id(slot_id)
        slot = self._slots[row]
        if not _BLOCKING_REVIEW_WARNINGS.intersection(slot.warnings):
            raise ValueError(f"slot {slot_id} has no blocking preview warning")
        if slot.blocking_review_approved == approved:
            return
        self._replace_slot(
            row,
            replace(slot, blocking_review_approved=approved),
        )
        index = self.index(row, 0)
        self.dataChanged.emit(
            index,
            index,
            [
                self.WARNING_STATE_ROLE,
                int(Qt.ItemDataRole.ToolTipRole),
                int(Qt.ItemDataRole.AccessibleTextRole),
            ],
        )

    def boundary_offset_is_confirmed(self, slot_id: int) -> bool:
        slot = self._slots[self.row_for_slot_id(slot_id)]
        return slot.boundary_offset == slot.confirmed_boundary_offset

    def confirm_thumbnail(
        self,
        slot_id: int,
        boundary_offset: int,
        thumbnail: QIcon | QImage | QPixmap | None,
    ) -> None:
        """Install only the thumbnail matching the slot's current offset."""

        row = self.row_for_slot_id(slot_id)
        slot = self._slots[row]
        if boundary_offset != slot.boundary_offset:
            raise ValueError(
                f"slot {slot_id} thumbnail is for stale Film Spacing Offset {boundary_offset:+d}; "
                f"current value is {slot.boundary_offset:+d}"
            )
        self._as_icon(thumbnail)
        approval_was_reset = slot.blocking_review_approved
        self._replace_slot(
            row,
            replace(
                slot,
                thumbnail=thumbnail,
                confirmed_boundary_offset=boundary_offset,
                blocking_review_approved=False,
            ),
        )
        index = self.index(row, 0)
        roles = [
            int(Qt.ItemDataRole.DecorationRole),
            self.BOUNDARY_OFFSET_CONFIRMED_ROLE,
            int(Qt.ItemDataRole.ToolTipRole),
        ]
        if approval_was_reset:
            roles.extend(
                [
                    self.WARNING_STATE_ROLE,
                    int(Qt.ItemDataRole.AccessibleTextRole),
                ]
            )
        self.dataChanged.emit(
            index,
            index,
            roles,
        )

    def update_thumbnail(self, slot_id: int, thumbnail: QIcon | QImage | QPixmap | None) -> None:
        """Replace only one preview image, preserving its physical slot row."""

        row = self.row_for_slot_id(slot_id)
        self._as_icon(thumbnail)
        self._replace_slot(row, replace(self._slots[row], thumbnail=thumbnail))
        index = self.index(row, 0)
        self.dataChanged.emit(index, index, [int(Qt.ItemDataRole.DecorationRole)])

    def _replace_slot(self, row: int, slot: RollPreviewSlot) -> None:
        slots = list(self._slots)
        slots[row] = slot
        self._slots = tuple(slots)

    @staticmethod
    def _as_icon(thumbnail: QIcon | QImage | QPixmap | None) -> QIcon:
        if thumbnail is None:
            return QIcon()
        if isinstance(thumbnail, QIcon):
            return thumbnail
        if isinstance(thumbnail, QImage):
            return QIcon(QPixmap.fromImage(thumbnail))
        if isinstance(thumbnail, QPixmap):
            return QIcon(thumbnail)
        raise TypeError(f"unsupported preview thumbnail: {type(thumbnail).__name__}")
