"""Find Duplicates' review: one card per group of frames scanned more than once, where the
user picks the frame to keep and what happens to the rest."""

import os
from typing import Callable, List, Optional, Sequence, Tuple

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from negpy.desktop.view.styles.templates import field_label, hint_label, pin_dialog_default, section_subheader, wrap_tooltip
from negpy.desktop.view.styles.theme import THEME
from negpy.desktop.view.widgets.dialog_geometry import remember_dialog_geometry
from negpy.kernel.system.text import count_of
from negpy.services.assets.duplicates import DuplicateGroup

_TILE_H = 120

# (label, action) in menu order.
_ACTIONS = (
    ("Stack", "stack"),
    ("Reject Others", "reject"),
    ("Move Others to Trash", "trash"),
    ("Skip", "skip"),
)
_ACTION_TIP = (
    "What happens to the frames not kept. Stack averages them with the kept one into one frame, "
    "for less scanner noise; the kept frame sets the framing. Reject Others marks them rejected, "
    "so filters and batch export skip them; the files stay. Move Others to Trash moves their files "
    "to the Trash and unloads them. Skip leaves the group as it is."
)


def default_keep(paths: Sequence[str]) -> str:
    """The largest file: the full-size, least compressed of the copies, and the scan with the
    most film on it to stack the others onto."""
    return max(paths, key=lambda p: (os.path.getsize(p) if os.path.exists(p) else 0, p))


def default_action(group: DuplicateGroup) -> str:
    """Stack for rescans, which average to less noise. Copies of one scan, or a mix of file
    types (a scan beside its exported JPEGs), average to nothing, so the rest are rejected."""
    kinds = {os.path.splitext(p)[1].lower() for p in group.paths}
    return "reject" if group.same_scan or len(kinds) > 1 else "stack"


class _GroupCard(QWidget):
    def __init__(self, group: DuplicateGroup, pixmap_for: Callable[[str], Optional[QPixmap]], parent=None) -> None:
        super().__init__(parent)
        self.group = group
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, THEME.space_md)
        what = "copies of one scan" if group.same_scan else "scans of one frame"
        layout.addWidget(section_subheader(f"{len(group.paths)} {what}"))

        tiles = QHBoxLayout()
        self.keep_group = QButtonGroup(self)
        keep = default_keep(group.paths)
        self._radios: List[Tuple[QRadioButton, str]] = []
        for path in group.paths:
            tile = QVBoxLayout()
            image = QLabel()
            image.setFixedHeight(_TILE_H)
            image.setAlignment(Qt.AlignmentFlag.AlignCenter)
            pix = pixmap_for(path)
            if pix is not None and not pix.isNull():
                image.setPixmap(pix.scaledToHeight(_TILE_H, Qt.TransformationMode.SmoothTransformation))
            image.setToolTip(path)
            tile.addWidget(image)
            name = hint_label(os.path.basename(path))
            name.setAlignment(Qt.AlignmentFlag.AlignCenter)
            tile.addWidget(name)
            radio = QRadioButton("Keep")
            radio.setToolTip(wrap_tooltip("The frame that stays. With Stack it is the one the others are aligned to"))
            radio.setChecked(path == keep)
            self.keep_group.addButton(radio)
            self._radios.append((radio, path))
            tile.addWidget(radio, 0, Qt.AlignmentFlag.AlignHCenter)
            tiles.addLayout(tile)
        tiles.addStretch()
        layout.addLayout(tiles)

        row = QHBoxLayout()
        row.addWidget(field_label("Others"))
        self.action = QComboBox()
        for label, key in _ACTIONS:
            self.action.addItem(label, key)
        self.action.setCurrentIndex([k for _l, k in _ACTIONS].index(default_action(group)))
        self.action.setToolTip(wrap_tooltip(_ACTION_TIP))
        row.addWidget(self.action)
        row.addStretch()
        layout.addLayout(row)

    def decision(self) -> Tuple[Tuple[str, ...], str, str]:
        keep = next((p for r, p in self._radios if r.isChecked()), self.group.paths[0])
        return self.group.paths, keep, str(self.action.currentData())


class DuplicatesDialog(QDialog):
    def __init__(self, groups: Sequence[DuplicateGroup], pixmap_for: Callable[[str], Optional[QPixmap]], repo=None, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Find Duplicates")
        self.resize(760, 560)
        layout = QVBoxLayout(self)
        frames = sum(len(g.paths) for g in groups)
        layout.addWidget(
            hint_label(
                f"{count_of(len(groups), 'group')} of frames scanned more than once ({count_of(frames, 'frame')}). "
                "Pick the frame to keep in each, and what happens to the others."
            )
        )

        body = QWidget()
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        self.cards = [_GroupCard(g, pixmap_for) for g in groups]
        for card in self.cards:
            body_layout.addWidget(card)
        body_layout.addStretch()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(body)
        layout.addWidget(scroll, 1)

        btns = QHBoxLayout()
        btns.addStretch()
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        self.apply_btn = QPushButton("Apply")
        self.apply_btn.clicked.connect(self.accept)
        btns.addWidget(self.cancel_btn)
        btns.addWidget(self.apply_btn)
        layout.addLayout(btns)
        pin_dialog_default(self.apply_btn, self.cancel_btn)
        remember_dialog_geometry(self, repo, "find_duplicates")

    def decisions(self) -> List[Tuple[Tuple[str, ...], str, str]]:
        return [c.decision() for c in self.cards]
