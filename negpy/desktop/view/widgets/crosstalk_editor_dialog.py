from negpy.kernel.system.i18n import mark, tr
from typing import List, Optional

import qtawesome as qta
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from negpy.desktop.view.confirm import confirm_delete_named
from negpy.desktop.view.sidebar.tone import _CH_COLORS
from negpy.desktop.view.styles.templates import (
    dialog_pane_qss,
    field_row,
    hint_label,
    icon_button,
    labeled_action,
    pane_header_qss,
    pin_dialog_default,
    section_subheader,
)
from negpy.desktop.view.styles.theme import THEME
from negpy.desktop.view.widgets.sliders import CompactSlider, align_slider_columns
from negpy.desktop.view.widgets.step_combo import StepCombo
from negpy.features.process.models import DEFAULT_CROSSTALK_MATRIX, ProcessMode
from negpy.services.assets.crosstalk import CrosstalkProfiles, CrosstalkType
from negpy.desktop.view.widgets.dialog_geometry import remember_dialog_geometry
from negpy.desktop.view.widgets.floating_panel import float_over_app

#: Selectable provenances, in dropdown group order. "Other" is not offered: it exists to
#: keep a hand-written type loadable, not as something to choose.
#: Plain str values, like _PROCESS_CHOICES below: combo item data round-trips through
#: QVariant, which does not match an enum member against the equal string from disk.
_TYPE_CHOICES: tuple[tuple[str, str], ...] = (
    (str(CrosstalkType.TUNED), tr("Tuned on a rig")),
    (str(CrosstalkType.MEASURED), tr("Measured")),
    (str(CrosstalkType.SPECSHEET), tr("From spec sheets (approx)")),
)

_CHANNELS = (mark("Red"), mark("Green"), mark("Blue"))
#: A term's slider range. Rows are normalized, so a term past ±1 only rescales the row.
_TERM_RANGE = 1.0

#: Film processes a matrix can describe. A B&W negative has one emulsion, so there is
#: nothing to unmix.
_PROCESS_CHOICES: tuple[tuple[str, str], ...] = (
    (str(ProcessMode.C41), tr("Color Negative (C-41)")),
    (str(ProcessMode.E6), tr("Transparency (E-6)")),
)


def flat_to_grid(flat: List[float]) -> List[List[float]]:
    return [list(flat[i * 3 : i * 3 + 3]) for i in range(3)]


def grid_to_flat(grid: List[List[float]]) -> List[float]:
    return [float(v) for row in grid for v in row]


def unique_copy_name(base: str, existing) -> str:
    """ "<base> Copy", then "<base> Copy 2", 3, ... skipping names already taken."""
    taken = set(existing)
    candidate = tr("{name} Copy", name=base)
    if candidate not in taken:
        return candidate
    i = 2
    while f"{candidate} {i}" in taken:
        i += 1
    return f"{candidate} {i}"


class CrosstalkEditorDialog(QDialog):
    """Modeless editor for spectral-crosstalk density matrices.

    Bundled matrices and Default are read-only (view + copy); user profiles live
    as TOMLs in the docs folder. Emits live previews as sliders move; the sidebar
    renders them and decides whether to apply or restore on close.
    """

    matrix_previewed = pyqtSignal(object, float, str)  # (flat 9-float matrix, strength, film process)
    profiles_changed = pyqtSignal()

    def __init__(self, current_profile: str, current_strength: float, process_mode: Optional[str] = None, parent=None, *, repo=None):
        super().__init__(parent)
        self._selected_name: Optional[str] = None
        self._updating = False
        # What a new profile is for: the process being worked in, not a fixed default.
        self._default_process = str(process_mode or ProcessMode.C41)

        self.setWindowTitle(tr("Crosstalk Matrices"))
        float_over_app(self)
        self.resize(760, 640)
        self.setMinimumSize(520, 560)
        self._init_ui()

        self._reload_list(
            select=current_profile if current_profile in CrosstalkProfiles.list_profiles() else CrosstalkProfiles.DEFAULT_NAME
        )
        self.preview_strength_slider.setValue(current_strength if current_strength > 0 else 1.0)
        remember_dialog_geometry(self, repo, "crosstalk_editor")

    # ------------------------------------------------------------------ UI

    def _init_ui(self) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left: profile list + new / copy / delete
        left = QWidget()
        left.setMinimumWidth(180)
        left.setObjectName("dialog_pane")
        left.setStyleSheet(dialog_pane_qss())
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(THEME.space_lg, THEME.space_lg, THEME.space_lg, THEME.space_lg)
        left_layout.setSpacing(THEME.space_md)

        header = QLabel(tr("PROFILES"))
        header.setStyleSheet(pane_header_qss())
        left_layout.addWidget(header)

        self.profile_list = QListWidget()
        self.profile_list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.profile_list.setTextElideMode(Qt.TextElideMode.ElideRight)
        self.profile_list.currentRowChanged.connect(self._on_row_changed)
        left_layout.addWidget(self.profile_list)

        btns = QHBoxLayout()
        self.new_btn = icon_button("fa5s.plus", tr("New matrix, starting from identity"))
        self.copy_btn = icon_button("fa5s.copy", tr("Make an editable copy of the selected profile"))
        self.delete_btn = icon_button("fa5s.trash-alt", tr("Delete the selected profile"))
        self.new_btn.clicked.connect(self._on_new)
        self.copy_btn.clicked.connect(self._on_copy)
        self.delete_btn.clicked.connect(self._on_delete)
        btns.addWidget(self.new_btn)
        btns.addWidget(self.copy_btn)
        btns.addWidget(self.delete_btn)
        btns.addStretch()
        left_layout.addLayout(btns)

        splitter.addWidget(left)

        # Right: name + matrix grid + preview strength
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(THEME.space_2xl, THEME.space_2xl, THEME.space_2xl, THEME.space_2xl)
        rl.setSpacing(THEME.space_xl)

        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText(tr("Profile name"))
        self.name_edit.setToolTip(tr("The name in the Matrix dropdown, such as the film and the rig."))
        self.name_edit.textChanged.connect(self._on_name_changed)
        rl.addLayout(field_row(tr("Name"), self.name_edit))

        self.type_combo = StepCombo()
        for value, label in _TYPE_CHOICES:
            self.type_combo.addItem(label, value)
        self.type_combo.setToolTip(
            tr("Where these numbers came from: measured on a reference, tuned by eye on a rig, or read from film data sheets.")
        )
        rl.addLayout(field_row(tr("Type"), self.type_combo))

        self.process_combo = StepCombo()
        for value, label in _PROCESS_CHOICES:
            self.process_combo.addItem(label, value)
        self.process_combo.setToolTip(tr("The film type these numbers describe, and the only one the matrix shows and works for."))
        self.process_combo.currentIndexChanged.connect(lambda _i: self._emit_preview())
        rl.addLayout(field_row(tr("Process"), self.process_combo))

        info = hint_label(
            tr(
                "Each channel's density takes in a share of the other two. A negative value removes "
                "that channel's leak. Every channel keeps its own density, because each row is "
                "normalized to sum to 1. Strength on the Calibration card applies the matrix."
            )
        )
        info.setWordWrap(True)
        rl.addWidget(info)

        self.readonly_hint = hint_label(tr("Bundled matrix, read-only. Make an editable copy to change it."))
        rl.addWidget(self.readonly_hint)

        rl.addWidget(self._build_matrix_rows())

        self.preview_strength_slider = CompactSlider(tr("Preview Strength"), 0.0, 1.0, 1.0, has_neutral=False)
        self.preview_strength_slider.setToolTip(
            tr("How strongly the matrix previews here, while Strength on the Calibration card applies it.")
        )
        self.preview_strength_slider.valueChanged.connect(lambda _v: self._emit_preview())
        rl.addWidget(self.preview_strength_slider)

        rl.addStretch()

        save_row = QHBoxLayout()
        save_row.addStretch()
        self.save_btn = labeled_action(
            "fa5s.save", tr("Save to Disk"), tr("Write this profile as a .toml in the NegPy/crosstalk folder, so it is reusable")
        )
        self.save_btn.clicked.connect(self._on_save)
        save_row.addWidget(self.save_btn)
        rl.addLayout(save_row)

        close_row = QHBoxLayout()
        cancel_btn = QPushButton(tr("Cancel"))
        cancel_btn.clicked.connect(self.reject)
        apply_btn = QPushButton(tr("Apply"))
        apply_btn.clicked.connect(self.accept)
        close_row.addStretch()
        close_row.addWidget(cancel_btn)
        close_row.addWidget(apply_btn)
        rl.addLayout(close_row)

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([250, 510])
        root.addWidget(splitter)
        align_slider_columns(right)
        pin_dialog_default(apply_btn, scope=self)
        # The sliders polish the pane while it is built, before the button is marked primary.
        apply_btn.style().unpolish(apply_btn)
        apply_btn.style().polish(apply_btn)

    def _build_matrix_rows(self) -> QWidget:
        """Two sliders per channel, one for each channel it takes density from.

        self._cells is 3x3, row = output channel, with None on the diagonal: row normalization
        makes that term redundant."""
        self._cells: List[List[Optional[CompactSlider]]] = [[None] * 3 for _ in range(3)]
        self._diag = [1.0, 1.0, 1.0]
        container = QWidget()
        col = QVBoxLayout(container)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(THEME.space_sm)
        for r, out in enumerate(_CHANNELS):
            col.addWidget(section_subheader(out))
            for c, src in enumerate(_CHANNELS):
                if r == c:
                    continue
                sld = CompactSlider(
                    tr("From {value}", value=tr(src)),
                    -_TERM_RANGE,
                    _TERM_RANGE,
                    0.0,
                    step=0.001,
                    precision=1000,
                    color=_CH_COLORS[c],
                    has_neutral=True,
                )
                sld.spin.setDecimals(3)
                sld.setToolTip(
                    tr(
                        "How much of the {value} density goes into {value1}, where a negative value removes the leak.",
                        value=tr(src).lower(),
                        value1=tr(out).lower(),
                    )
                )
                sld.valueChanged.connect(lambda _v: self._emit_preview())
                col.addWidget(sld)
                self._cells[r][c] = sld
        return container

    # ------------------------------------------------------------- helpers

    def working_matrix(self) -> List[float]:
        return [
            self._diag[r] if r == c else self._cells[r][c].value()  # type: ignore[union-attr]
            for r in range(3)
            for c in range(3)
        ]

    def preview_strength(self) -> float:
        return self.preview_strength_slider.value()

    def selected_name(self) -> Optional[str]:
        return self._selected_name

    def _matrix_for(self, name: str) -> List[float]:
        if name == CrosstalkProfiles.DEFAULT_NAME:
            return list(DEFAULT_CROSSTALK_MATRIX)
        return CrosstalkProfiles.get_matrix(name) or list(DEFAULT_CROSSTALK_MATRIX)

    def _all_names(self) -> list:
        return CrosstalkProfiles.list_profiles()

    def selected_type(self) -> str:
        return self.type_combo.currentData() or CrosstalkType.TUNED

    def selected_process(self) -> str:
        return self.process_combo.currentData() or str(ProcessMode.C41)

    def _set_process(self, value: str) -> None:
        idx = self.process_combo.findData(value)
        self.process_combo.setCurrentIndex(idx if idx >= 0 else 0)

    def _set_type(self, value: str) -> None:
        """Select `value`, falling back to Tuned for a built-in or hand-written type.

        Not the first entry: saving must not relabel an unknown type as a spec-sheet claim."""
        idx = self.type_combo.findData(str(value))
        self.type_combo.setCurrentIndex(idx if idx >= 0 else self.type_combo.findData(str(CrosstalkType.TUNED)))

    def _set_grid(self, flat: List[float]) -> None:
        grid = flat_to_grid(flat)
        for r in range(3):
            self._diag[r] = grid[r][r]
            for c in range(3):
                if r != c:
                    self._cells[r][c].setValue(grid[r][c])  # type: ignore[union-attr]

    def _set_grid_enabled(self, enabled: bool) -> None:
        for r, row in enumerate(self._cells):
            for c, cell in enumerate(row):
                if r != c:
                    cell.setEnabled(enabled)  # type: ignore[union-attr]

    def _emit_preview(self) -> None:
        if self._updating:
            return
        self.matrix_previewed.emit(self.working_matrix(), self.preview_strength(), self.selected_process())

    # ------------------------------------------------------------- list

    def _reload_list(self, select: Optional[str] = None) -> None:
        self._updating = True
        self.profile_list.blockSignals(True)
        self.profile_list.clear()
        names = [*sorted(CrosstalkProfiles.scan_user()), CrosstalkProfiles.DEFAULT_NAME, *sorted(CrosstalkProfiles.scan_bundled())]
        for name in names:
            item = QListWidgetItem(name)
            item.setData(Qt.ItemDataRole.UserRole, name)
            if CrosstalkProfiles.is_bundled(name):
                item.setForeground(QColor(THEME.text_muted))
                item.setIcon(qta.icon("fa5s.lock", color=THEME.text_muted))
            self.profile_list.addItem(item)
        self.profile_list.blockSignals(False)
        self._updating = False

        target = select if select in names else (names[0] if names else None)
        if target is not None:
            self.profile_list.setCurrentRow(names.index(target))

    def _on_row_changed(self, row: int) -> None:
        item = self.profile_list.item(row)
        if item is None:
            return
        name = item.data(Qt.ItemDataRole.UserRole)
        self._selected_name = name
        editable = not CrosstalkProfiles.is_bundled(name)

        self._updating = True
        self._set_grid(self._matrix_for(name))
        self.name_edit.setText(name)
        self._set_type(CrosstalkProfiles.get_type(name))
        self._set_process(CrosstalkProfiles.get_process(name))
        self._updating = False

        self.name_edit.setEnabled(editable)
        self.type_combo.setEnabled(editable)
        self.process_combo.setEnabled(editable)
        self._set_grid_enabled(editable)
        self.save_btn.setEnabled(editable)
        self.delete_btn.setEnabled(editable)
        self.readonly_hint.setVisible(not editable)
        self._emit_preview()

    # ------------------------------------------------------------- actions

    def _on_name_changed(self, _text: str) -> None:
        if self._updating:
            return
        # A name colliding with a bundled/Default profile would be shadowed in the combo.
        name = self.name_edit.text().strip()
        self.save_btn.setEnabled(bool(name) and not CrosstalkProfiles.is_bundled(name))

    def _on_new(self) -> None:
        existing = set(self._all_names())
        name, i = tr("New Matrix"), 2
        while name in existing:
            name = tr("New Matrix {value}", value=i)
            i += 1
        identity = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        # For the process being worked in: a new matrix the user cannot then select is the whole
        # reason this key is here.
        CrosstalkProfiles.save(name, identity, process=self._default_process)
        self.profiles_changed.emit()
        self._reload_list(select=name)

    def _on_copy(self) -> None:
        if self._selected_name is None:
            return
        new_name = unique_copy_name(self._selected_name, self._all_names())
        # Takes the `tuned` default rather than inheriting a datasheet provenance claim. The
        # process IS inherited, because it says which dye set the numbers describe.
        CrosstalkProfiles.save(new_name, self.working_matrix(), process=self.selected_process())
        self.profiles_changed.emit()
        self._reload_list(select=new_name)

    def _on_save(self) -> None:
        name = self.name_edit.text().strip()
        if not name or CrosstalkProfiles.is_bundled(name):
            return
        old = self._selected_name
        if old and old != name and not CrosstalkProfiles.is_bundled(old):
            CrosstalkProfiles.delete(old)
        CrosstalkProfiles.save(name, self.working_matrix(), self.selected_type(), self.selected_process())
        self.profiles_changed.emit()
        self._reload_list(select=name)

    def accept(self) -> None:
        # Apply-and-close persists the edited profile too (bundled/Default are read-only).
        if self._selected_name is not None and not CrosstalkProfiles.is_bundled(self._selected_name):
            self._on_save()
        super().accept()

    def _on_delete(self) -> None:
        if self._selected_name is None or CrosstalkProfiles.is_bundled(self._selected_name):
            return
        if not confirm_delete_named(self, tr("Matrix"), self._selected_name):
            return
        CrosstalkProfiles.delete(self._selected_name)
        self.profiles_changed.emit()
        self._reload_list(select=CrosstalkProfiles.DEFAULT_NAME)
