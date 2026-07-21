from typing import List, Optional

import qtawesome as qta
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPen
from PyQt6.QtWidgets import (
    QDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from negpy.desktop.view.sidebar.tone import _CH_COLORS
from negpy.desktop.view.styles.templates import dialog_pane_qss, hint_label, pane_header_qss
from negpy.desktop.view.styles.theme import THEME
from negpy.desktop.view.widgets.sliders import CompactSlider
from negpy.features.process.models import DEFAULT_CROSSTALK_MATRIX
from negpy.services.assets.crosstalk import CrosstalkProfiles


def flat_to_grid(flat: List[float]) -> List[List[float]]:
    return [list(flat[i * 3 : i * 3 + 3]) for i in range(3)]


def grid_to_flat(grid: List[List[float]]) -> List[float]:
    return [float(v) for row in grid for v in row]


def unique_copy_name(base: str, existing) -> str:
    """ "<base> Copy", then "<base> Copy 2", 3, ... skipping names already taken."""
    taken = set(existing)
    candidate = f"{base} Copy"
    if candidate not in taken:
        return candidate
    i = 2
    while f"{candidate} {i}" in taken:
        i += 1
    return f"{candidate} {i}"


class _MatrixGridWidget(QWidget):
    """Hosts the 3×3 slider grid and paints subtle separators between cells."""

    def __init__(self, cells, parent=None):
        super().__init__(parent)
        self._cells = cells

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        # Column/row centers from the slider cells (the diagonal cells are None).
        ref = None
        col_x: list = [None, None, None]
        row_y: list = [None, None, None]
        for r, row in enumerate(self._cells):
            for c, cell in enumerate(row):
                if cell is None:
                    continue
                g = cell.geometry()
                ref = g
                col_x[c] = (g.left() + g.right()) // 2
                row_y[r] = (g.top() + g.bottom()) // 2
        if ref is None or None in col_x or None in row_y:
            return
        p = QPainter(self)
        p.setPen(QPen(QColor(255, 255, 255, 22), 1))
        hw, hh = ref.width() // 2, ref.height() // 2
        # Overhang a bit into the column/row header labels.
        top, bottom = row_y[0] - hh - 18, row_y[2] + hh
        left, right = col_x[0] - hw - 26, col_x[2] + hw
        for j in (1, 2):
            p.drawLine((col_x[j - 1] + col_x[j]) // 2, top, (col_x[j - 1] + col_x[j]) // 2, bottom)
        for i in (1, 2):
            p.drawLine(left, (row_y[i - 1] + row_y[i]) // 2, right, (row_y[i - 1] + row_y[i]) // 2)


class CrosstalkEditorDialog(QDialog):
    """Modeless editor for spectral-crosstalk density matrices.

    Bundled matrices and Default are read-only (view + copy); user profiles live
    as TOMLs in the docs folder. Emits live previews as sliders move; the sidebar
    renders them and decides whether to apply or restore on close.
    """

    matrix_previewed = pyqtSignal(object, float)  # (flat 9-float matrix, preview strength)
    profiles_changed = pyqtSignal()

    def __init__(self, current_profile: str, current_strength: float, parent=None):
        super().__init__(parent)
        self._selected_name: Optional[str] = None
        self._updating = False

        self.setWindowTitle("Crosstalk Matrices")
        self.resize(680, 620)
        self.setMinimumSize(520, 560)
        self._init_ui()

        self._reload_list(
            select=current_profile if current_profile in CrosstalkProfiles.list_profiles() else CrosstalkProfiles.DEFAULT_NAME
        )
        self.preview_strength_slider.setValue(current_strength if current_strength > 0 else 1.0)

    # ------------------------------------------------------------------ UI

    def _init_ui(self) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left: profile list + new / copy / delete
        left = QWidget()
        left.setMinimumWidth(180)
        left.setStyleSheet(dialog_pane_qss())
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(8, 8, 8, 8)
        left_layout.setSpacing(6)

        header = QLabel("PROFILES")
        header.setStyleSheet(pane_header_qss())
        left_layout.addWidget(header)

        self.profile_list = QListWidget()
        self.profile_list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.profile_list.setTextElideMode(Qt.TextElideMode.ElideRight)
        self.profile_list.currentRowChanged.connect(self._on_row_changed)
        left_layout.addWidget(self.profile_list)

        btns = QHBoxLayout()
        self.new_btn = self._tool_btn("fa5s.plus", "New matrix (starts from identity)", self._on_new)
        self.copy_btn = self._tool_btn("fa5s.copy", "Make an editable copy of the selected profile", self._on_copy)
        self.delete_btn = self._tool_btn("fa5s.trash-alt", "Delete the selected profile", self._on_delete)
        btns.addWidget(self.new_btn)
        btns.addWidget(self.copy_btn)
        btns.addWidget(self.delete_btn)
        btns.addStretch()
        left_layout.addLayout(btns)

        splitter.addWidget(left)

        # Right: name + matrix grid + preview strength
        right = QWidget()
        right.setStyleSheet(f"background: {THEME.bg_dark};")
        rl = QVBoxLayout(right)
        rl.setContentsMargins(16, 16, 16, 16)
        rl.setSpacing(12)

        name_row = QHBoxLayout()
        name_row.addWidget(QLabel("Name"))
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("Profile name")
        self.name_edit.textChanged.connect(self._on_name_changed)
        name_row.addWidget(self.name_edit, 1)
        rl.addLayout(name_row)

        info = QLabel(
            "<b>Spectral crosstalk unmix</b><br>"
            "Film dyes leak a little density into the channels they shouldn't, muddying colour.<br>"
            "<br>"
            "• <b>IN</b> columns are the source channel; each row is the output channel it feeds.<br>"
            "• Each off-diagonal slider subtracts one channel's leak from another — e.g. column "
            "green, row red removes green's contamination from red.<br>"
            "• The diagonal is fixed (rows are re-normalized).<br>"
            "• Raise <b>Separation</b> in the sidebar to dial the effect in."
        )
        info.setWordWrap(True)
        info.setStyleSheet(
            f"background: rgba(255, 255, 255, 0.04); border: 1px solid rgba(255, 255, 255, 0.08); "
            f"border-radius: 6px; padding: 8px; color: {THEME.text_secondary};"
        )
        rl.addWidget(info)

        self.readonly_hint = hint_label("Bundled matrix — read-only. Make an editable copy to change it.")
        rl.addWidget(self.readonly_hint)

        rl.addWidget(self._build_matrix_grid())

        self.preview_strength_slider = CompactSlider("Preview strength", 0.0, 1.0, 1.0, has_neutral=False)
        self.preview_strength_slider.setToolTip(
            "How strongly the matrix previews here (view-only — set Separation in the sidebar to apply)"
        )
        self.preview_strength_slider.valueChanged.connect(lambda _v: self._emit_preview())
        rl.addWidget(self.preview_strength_slider)

        rl.addStretch()

        save_row = QHBoxLayout()
        save_row.addStretch()
        self.save_btn = QPushButton(" Save to disk")
        self.save_btn.setIcon(qta.icon("fa5s.save", color=THEME.text_primary))
        self.save_btn.setToolTip("Write this profile as a .toml in the NegPy/crosstalk folder so it's reusable")
        self.save_btn.clicked.connect(self._on_save)
        save_row.addWidget(self.save_btn)
        rl.addLayout(save_row)

        close_row = QHBoxLayout()
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        apply_btn = QPushButton("Apply and close")
        apply_btn.setDefault(True)
        apply_btn.clicked.connect(self.accept)
        close_row.addStretch()
        close_row.addWidget(cancel_btn)
        close_row.addWidget(apply_btn)
        rl.addLayout(close_row)

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([210, 450])
        root.addWidget(splitter)

    def _build_matrix_grid(self) -> QWidget:
        # Diagonal is pinned by row-normalization, so only off-diagonal terms are
        # sliders; self._cells is 3×3 with None on the diagonal.
        self._cells: List[List[Optional[CompactSlider]]] = []
        self._diag = [1.0, 1.0, 1.0]
        container = _MatrixGridWidget(self._cells)
        grid = QGridLayout(container)
        grid.setSpacing(10)
        grid.setContentsMargins(2, 4, 2, 4)
        # Axis-title (0) and colour-box (1) columns stay fixed; slider columns absorb resize.
        grid.setColumnStretch(0, 0)
        grid.setColumnStretch(1, 0)
        for j in (2, 3, 4):
            grid.setColumnStretch(j, 1)

        in_title = QLabel("IN")
        in_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        in_title.setStyleSheet(f"color: {THEME.text_secondary}; font-weight: bold; letter-spacing: 3px;")
        in_title.setToolTip("Columns are the input channel a slider mixes in; each row is the output channel.")
        grid.addWidget(in_title, 0, 2, 1, 3)

        for c in range(3):
            col = QLabel()
            col.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            col.setFixedHeight(22)
            col.setStyleSheet(f"background: {_CH_COLORS[c]}; border-radius: 4px;")
            grid.addWidget(col, 1, c + 2)

        for r in range(3):
            row_lbl = QLabel()
            row_lbl.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
            row_lbl.setFixedWidth(22)
            row_lbl.setStyleSheet(f"background: {_CH_COLORS[r]}; border-radius: 4px;")
            grid.addWidget(row_lbl, r + 2, 1)
            row_cells: List[Optional[CompactSlider]] = []
            for c in range(3):
                if r == c:
                    dash = QLabel("—")
                    dash.setAlignment(Qt.AlignmentFlag.AlignCenter)
                    dash.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
                    dash.setStyleSheet(f"color: {THEME.text_muted};")
                    dash.setToolTip(
                        "Diagonal is fixed — this channel keeps itself (row normalization makes it redundant). Edit the off-diagonal mixing terms."
                    )
                    grid.addWidget(dash, r + 2, c + 2)
                    row_cells.append(None)
                    continue
                sld = CompactSlider("", -0.5, 0.5, 0.0, step=0.001, precision=1000, has_neutral=True)
                sld.spin.setDecimals(3)
                sld.valueChanged.connect(lambda _v: self._emit_preview())
                grid.addWidget(sld, r + 2, c + 2)
                row_cells.append(sld)
            self._cells.append(row_cells)
        return container

    def _tool_btn(self, icon: str, tooltip: str, slot) -> QPushButton:
        btn = QPushButton()
        btn.setIcon(qta.icon(icon, color=THEME.text_primary, color_disabled=THEME.text_muted))
        btn.setToolTip(tooltip)
        btn.setFixedWidth(34)
        btn.clicked.connect(slot)
        return btn

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
        self.matrix_previewed.emit(self.working_matrix(), self.preview_strength())

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
        self._updating = False

        self.name_edit.setEnabled(editable)
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
        name, i = "New Matrix", 2
        while name in existing:
            name = f"New Matrix {i}"
            i += 1
        identity = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        CrosstalkProfiles.save(name, identity)
        self.profiles_changed.emit()
        self._reload_list(select=name)

    def _on_copy(self) -> None:
        if self._selected_name is None:
            return
        new_name = unique_copy_name(self._selected_name, self._all_names())
        CrosstalkProfiles.save(new_name, self.working_matrix())
        self.profiles_changed.emit()
        self._reload_list(select=new_name)

    def _on_save(self) -> None:
        name = self.name_edit.text().strip()
        if not name or CrosstalkProfiles.is_bundled(name):
            return
        old = self._selected_name
        if old and old != name and not CrosstalkProfiles.is_bundled(old):
            CrosstalkProfiles.delete(old)
        CrosstalkProfiles.save(name, self.working_matrix())
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
        CrosstalkProfiles.delete(self._selected_name)
        self.profiles_changed.emit()
        self._reload_list(select=CrosstalkProfiles.DEFAULT_NAME)
