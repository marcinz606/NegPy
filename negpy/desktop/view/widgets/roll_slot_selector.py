"""Nikon-style candidate-slot picker for roll previews.

This widget stops at the UI boundary: it emits explicit physical slot IDs and
does not start a scanner itself.  That keeps preview selection testable without
hardware and lets a workflow decide how each selected slot is captured.
"""

from collections.abc import Iterable, Sequence

from PyQt6.QtCore import (
    QAbstractItemModel,
    QEvent,
    QItemSelection,
    QItemSelectionModel,
    QModelIndex,
    QPoint,
    QRect,
    QSignalBlocker,
    QSize,
    Qt,
    pyqtSignal,
)
from PyQt6.QtGui import (
    QColor,
    QFont,
    QFontDatabase,
    QIcon,
    QImage,
    QKeySequence,
    QMouseEvent,
    QPainter,
    QPen,
    QPixmap,
    QShortcut,
)
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QListView,
    QPushButton,
    QRubberBand,
    QSlider,
    QSpinBox,
    QStyle,
    QStyledItemDelegate,
    QStyleOptionViewItem,
    QVBoxLayout,
    QWidget,
)

from negpy.desktop.view.styles.theme import THEME
from negpy.desktop.view.widgets.roll_slot_model import RollPreviewSlot, RollSlotModel
from negpy.services.scanning.roll_preview_controls import ScanMaterial, boundary_offset_range


class _SignedOffsetSpinBox(QSpinBox):
    """Compact, stable-width display for a signed film-spacing offset."""

    def textFromValue(self, value: int) -> str:  # noqa: N802
        return "0" if value == 0 else f"{value:+d}"


class _RollSlotDelegate(QStyledItemDelegate):
    """Film-frame cells with a persistent slot number and advisory badge."""

    _CELL_MARGIN = 6
    _LABEL_HEIGHT = 22
    _FRAME_RADIUS = 3
    _WARNING = QColor("#D6A84B")

    def paint(self, painter: QPainter | None, option: QStyleOptionViewItem, index: QModelIndex) -> None:
        if painter is None:
            return
        selected = bool(option.state & QStyle.StateFlag.State_Selected)
        hovered = bool(option.state & QStyle.StateFlag.State_MouseOver)
        cell = option.rect.adjusted(self._CELL_MARGIN, self._CELL_MARGIN, -self._CELL_MARGIN, -self._CELL_MARGIN)
        frame = QRect(cell.left(), cell.top(), cell.width(), max(1, cell.height() - self._LABEL_HEIGHT))

        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(THEME.bg_header if (selected or hovered) else THEME.bg_dark))
        painter.drawRoundedRect(cell, THEME.radius_md, THEME.radius_md)

        icon = index.data(Qt.ItemDataRole.DecorationRole)
        if icon is not None and not icon.isNull():
            icon.paint(
                painter,
                frame.adjusted(2, 2, -2, -2),
                Qt.AlignmentFlag.AlignCenter,
                mode=QIcon.Mode.Normal,
                state=QIcon.State.On if selected else QIcon.State.Off,
            )
        else:
            painter.setBrush(QColor("#111111"))
            painter.drawRect(frame.adjusted(2, 2, -2, -2))
            painter.setPen(QColor(THEME.text_muted))
            font = painter.font()
            font.setPixelSize(THEME.font_size_xs)
            font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 1.0)
            painter.setFont(font)
            painter.drawText(frame, Qt.AlignmentFlag.AlignCenter, "NO PREVIEW")

        border_color = THEME.accent_primary if selected else (THEME.text_secondary if hovered else THEME.border_color)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(QColor(border_color), 2 if selected else 1))
        painter.drawRoundedRect(frame.adjusted(1, 1, -1, -1), self._FRAME_RADIUS, self._FRAME_RADIUS)

        slot_id = index.data(RollSlotModel.SLOT_ID_ROLE)
        painter.setPen(QColor(THEME.text_primary if selected else THEME.text_secondary))
        label_font = painter.font()
        label_font.setBold(selected)
        label_font.setPixelSize(THEME.font_size_small)
        painter.setFont(label_font)
        label_rect = QRect(cell.left() + 5, frame.bottom() + 1, cell.width() - 10, self._LABEL_HEIGHT - 1)
        painter.drawText(label_rect, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, f"SLOT {slot_id:02d}")

        if index.data(RollSlotModel.HAS_WARNINGS_ROLE):
            badge_size = 18
            badge = QRect(frame.right() - badge_size - 5, frame.top() + 5, badge_size, badge_size)
            painter.setPen(QPen(QColor("#201A0D"), 1))
            painter.setBrush(self._WARNING)
            painter.drawEllipse(badge)
            badge_font = painter.font()
            badge_font.setBold(True)
            badge_font.setPixelSize(12)
            painter.setFont(badge_font)
            painter.setPen(QColor("#201A0D"))
            painter.drawText(badge, Qt.AlignmentFlag.AlignCenter, "!")

        painter.restore()

    def sizeHint(self, option: QStyleOptionViewItem, index: QModelIndex) -> QSize:  # noqa: N802
        size = index.data(Qt.ItemDataRole.SizeHintRole)
        return size if isinstance(size, QSize) else super().sizeHint(option, index)


class RollSlotGridView(QListView):
    """Icon grid with desktop-style additive, range, and marquee selection."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._pointer_down = False
        self._drag_active = False
        self._drag_anchor_row: int | None = None
        self._drag_origin = QPoint()
        self._drag_base_rows: set[int] = set()
        self._drag_additive = False
        self._rubber_band = QRubberBand(QRubberBand.Shape.Rectangle, self.viewport())
        self.setObjectName("roll_slot_grid")
        self.setViewMode(QListView.ViewMode.IconMode)
        self.setFlow(QListView.Flow.LeftToRight)
        self.setWrapping(True)
        self.setResizeMode(QListView.ResizeMode.Adjust)
        self.setMovement(QListView.Movement.Static)
        self.setUniformItemSizes(True)
        self.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectItems)
        self.setSelectionRectVisible(True)
        self.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.setDragEnabled(False)
        self.setMouseTracking(True)
        self.setSpacing(2)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)

    def selectionCommand(self, index: QModelIndex, event: QEvent | None = None) -> QItemSelectionModel.SelectionFlag:  # noqa: N802
        """Treat both Control and Command clicks as additive toggles.

        Qt's platform style normally maps the native modifier, but recognizing
        both explicitly also makes remote-desktop and cross-platform behavior
        predictable.
        """

        if isinstance(event, QMouseEvent) and event.type() == QEvent.Type.MouseButtonPress and index.isValid():
            modifiers = event.modifiers()
            if modifiers & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.MetaModifier):
                return QItemSelectionModel.SelectionFlag.Toggle
        return super().selectionCommand(index, event)

    def mousePressEvent(self, e: QMouseEvent | None) -> None:  # noqa: N802
        if e is None or e.button() != Qt.MouseButton.LeftButton:
            super().mousePressEvent(e)
            return

        self._pointer_down = True
        self._drag_active = False
        self._drag_origin = e.position().toPoint()
        anchor = self.indexAt(self._drag_origin)
        self._drag_anchor_row = anchor.row() if anchor.isValid() else None
        self._drag_base_rows = {index.row() for index in self._selection_model().selectedIndexes()}
        modifiers = e.modifiers()
        self._drag_additive = bool(modifiers & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.MetaModifier))
        super().mousePressEvent(e)

    def mouseMoveEvent(self, e: QMouseEvent | None) -> None:  # noqa: N802
        if e is None:
            super().mouseMoveEvent(e)
            return
        if not self._pointer_down:
            super().mouseMoveEvent(e)
            return

        point = e.position().toPoint()
        if not self._drag_active:
            distance = (point - self._drag_origin).manhattanLength()
            if distance < QApplication.startDragDistance():
                return
            self._drag_active = True

        rows = set(self._drag_base_rows) if self._drag_additive else set()
        if self._drag_anchor_row is not None:
            target = self.indexAt(point)
            if target.isValid():
                first, last = sorted((self._drag_anchor_row, target.row()))
                rows.update(range(first, last + 1))
        else:
            rectangle = QRect(self._drag_origin, point).normalized()
            self._rubber_band.setGeometry(rectangle)
            self._rubber_band.show()
            model = self._item_model()
            rows.update(row for row in range(model.rowCount()) if self.visualRect(model.index(row, 0)).intersects(rectangle))

        self._replace_selection(rows)
        e.accept()

    def mouseReleaseEvent(self, e: QMouseEvent | None) -> None:  # noqa: N802
        was_dragging = self._drag_active
        self._pointer_down = False
        self._drag_active = False
        self._drag_anchor_row = None
        self._rubber_band.hide()
        if was_dragging and e is not None and e.button() == Qt.MouseButton.LeftButton:
            e.accept()
            return
        super().mouseReleaseEvent(e)

    def _replace_selection(self, rows: set[int]) -> None:
        selection = QItemSelection()
        model = self._item_model()
        for row in sorted(rows):
            index = model.index(row, 0)
            selection.select(index, index)
        self._selection_model().select(selection, QItemSelectionModel.SelectionFlag.ClearAndSelect)

    def _item_model(self) -> QAbstractItemModel:
        model = self.model()
        if model is None:
            raise RuntimeError("roll slot grid has no model")
        return model

    def _selection_model(self) -> QItemSelectionModel:
        selection_model = self.selectionModel()
        if selection_model is None:
            raise RuntimeError("roll slot grid has no selection model")
        return selection_model


class RollSlotSelector(QWidget):
    """Select scanner slots from a complete, fixed-order roll preview."""

    selection_changed = pyqtSignal(list)
    scan_requested = pyqtSignal(list)
    thumbnail_reload_requested = pyqtSignal(int, int)
    scan_material_changed = pyqtSignal(object)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("roll_slot_selector")
        self.model = RollSlotModel(parent=self)
        self.grid = RollSlotGridView(self)
        self.grid.setModel(self.model)
        self.grid.setItemDelegate(_RollSlotDelegate(self.grid))
        self.grid.setAccessibleName("Roll preview slots")
        self.grid.setToolTip("Select physical scanner slots. Preview warnings never prevent selection.")

        self.title_label = QLabel("ROLL PREVIEW")
        self.title_label.setStyleSheet(
            f"font-size: {THEME.font_size_xs}px; color: {THEME.text_muted}; font-weight: {THEME.weight_semibold}; letter-spacing: 1px;"
        )
        self.summary_label = QLabel("No preview loaded")
        self.summary_label.setStyleSheet(f"color: {THEME.text_secondary};")

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.addWidget(self.title_label)
        header.addStretch()
        header.addWidget(self.summary_label)

        self.help_label = QLabel(
            "All scanner slots stay visible. Use Ctrl/Cmd-click, Shift-click, or drag to choose frames; warnings are advisory."
        )
        self.help_label.setWordWrap(True)
        self.help_label.setStyleSheet(f"color: {THEME.text_muted}; font-size: {THEME.font_size_xs}px;")

        self.scan_material_label = QLabel("Scan material")
        self.scan_material_label.setAccessibleName("Scan material")
        self.scan_material_combo = QComboBox()
        self.scan_material_combo.setObjectName("roll_scan_material")
        self.scan_material_combo.setAccessibleName("Scan material")
        for material in ScanMaterial:
            self.scan_material_combo.addItem(material.value, material.value)
        self.scan_material_status_label = QLabel()
        self.scan_material_status_label.setObjectName("roll_scan_material_status")
        self.scan_material_status_label.setAccessibleName("Scan material capture behavior")
        self.scan_material_status_label.setStyleSheet(f"color: {THEME.text_muted}; font-size: {THEME.font_size_xs}px;")

        material_controls = QHBoxLayout()
        material_controls.setContentsMargins(0, 0, 0, 0)
        material_controls.addWidget(self.scan_material_label)
        material_controls.addWidget(self.scan_material_combo)
        material_controls.addWidget(self.scan_material_status_label)
        material_controls.addStretch()

        offset_tooltip = (
            "Move left (negative) or right (positive), then reload the thumbnail. "
            "Keyboard: Alt+Left / Alt+Right"
        )
        self.boundary_offset_label = QLabel("Film Spacing Offset")
        self.boundary_offset_label.setAccessibleName("Film Spacing Offset")
        self.boundary_offset_label.setToolTip(offset_tooltip)

        self.boundary_offset_spin = _SignedOffsetSpinBox()
        self.boundary_offset_spin.setObjectName("roll_boundary_offset")
        self.boundary_offset_spin.setAccessibleName("Film Spacing Offset")
        self.boundary_offset_spin.setSuffix(" rows")
        self.boundary_offset_spin.setRange(0, 144)
        self.boundary_offset_spin.setValue(0)
        self.boundary_offset_spin.setMinimumHeight(40)
        self.boundary_offset_spin.setMinimumWidth(104)
        offset_font = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
        offset_font.setStyleHint(QFont.StyleHint.Monospace)
        offset_font.setFixedPitch(True)
        self.boundary_offset_spin.setFont(offset_font)
        self.boundary_offset_spin.setEnabled(False)
        self.boundary_offset_spin.setToolTip(offset_tooltip)

        self.boundary_offset_slider = QSlider(Qt.Orientation.Horizontal)
        self.boundary_offset_slider.setObjectName("roll_boundary_offset_slider")
        self.boundary_offset_slider.setAccessibleName("Film Spacing Offset slider")
        self.boundary_offset_slider.setRange(0, 144)
        self.boundary_offset_slider.setValue(0)
        self.boundary_offset_slider.setSingleStep(1)
        self.boundary_offset_slider.setPageStep(12)
        self.boundary_offset_slider.setTickInterval(24)
        self.boundary_offset_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.boundary_offset_slider.setMinimumSize(180, 40)
        self.boundary_offset_slider.setEnabled(False)
        self.boundary_offset_slider.setToolTip(offset_tooltip)

        self.offset_shortcut_label = QLabel("Alt+← / Alt+→")
        self.offset_shortcut_label.setObjectName("roll_boundary_offset_shortcuts")
        self.offset_shortcut_label.setAccessibleName("Film Spacing Offset keyboard shortcuts: Alt Left and Alt Right")
        self.offset_shortcut_label.setToolTip(offset_tooltip)
        self.offset_shortcut_label.setStyleSheet(
            f"color: {THEME.text_muted}; font-size: {THEME.font_size_xs}px;"
        )
        self.reload_thumbnail_button = QPushButton("Reload Thumbnail")
        self.reload_thumbnail_button.setObjectName("reload_roll_thumbnail")
        self.reload_thumbnail_button.setAccessibleName("Reload Thumbnail for current roll slot")
        self.reload_thumbnail_button.setMinimumHeight(40)
        self.reload_thumbnail_button.setEnabled(False)

        thumbnail_controls = QHBoxLayout()
        thumbnail_controls.setContentsMargins(0, 0, 0, 0)
        thumbnail_controls.addWidget(self.boundary_offset_label)
        thumbnail_controls.addWidget(self.boundary_offset_spin)
        thumbnail_controls.addWidget(self.boundary_offset_slider, 1)
        thumbnail_controls.addWidget(self.offset_shortcut_label)
        thumbnail_controls.addWidget(self.reload_thumbnail_button)

        self.select_all_button = QPushButton("Select all")
        self.select_all_button.setAccessibleName("Select all roll slots")
        self.clear_button = QPushButton("Clear")
        self.clear_button.setAccessibleName("Clear selected roll slots")
        self.selected_count_label = QLabel("0 of 0 selected")
        self.selected_count_label.setAccessibleName("Selected roll slot count")
        self.scan_button = QPushButton("Scan selected")
        self.scan_button.setProperty("primary", True)
        self.scan_button.setEnabled(False)
        self.scan_button.setAccessibleName("Scan selected roll slots")

        actions = QHBoxLayout()
        actions.setContentsMargins(0, 0, 0, 0)
        actions.addWidget(self.select_all_button)
        actions.addWidget(self.clear_button)
        actions.addWidget(self.selected_count_label)
        actions.addStretch()
        actions.addWidget(self.scan_button)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(THEME.space_xl, THEME.space_xl, THEME.space_xl, THEME.space_xl)
        layout.setSpacing(THEME.space_lg)
        layout.addLayout(header)
        layout.addWidget(self.help_label)
        layout.addLayout(material_controls)
        layout.addWidget(self.grid, 1)
        layout.addLayout(thumbnail_controls)
        layout.addLayout(actions)

        selection_model = self._selection_model()
        selection_model.selectionChanged.connect(self._on_selection_changed)
        selection_model.currentChanged.connect(self._on_current_slot_changed)
        self.select_all_button.clicked.connect(self.select_all)
        self.clear_button.clicked.connect(self.clear_selection)
        self.scan_button.clicked.connect(self.request_scan)
        self.boundary_offset_spin.valueChanged.connect(self._on_boundary_offset_changed)
        self.boundary_offset_slider.valueChanged.connect(self.boundary_offset_spin.setValue)
        self.reload_thumbnail_button.clicked.connect(self.request_thumbnail_reload)
        self.scan_material_combo.currentIndexChanged.connect(self._on_scan_material_changed)

        self._select_all_shortcut = QShortcut(QKeySequence(QKeySequence.StandardKey.SelectAll), self.grid)
        self._select_all_shortcut.setContext(Qt.ShortcutContext.WidgetShortcut)
        self._select_all_shortcut.activated.connect(self.select_all)
        self._clear_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Escape), self.grid)
        self._clear_shortcut.setContext(Qt.ShortcutContext.WidgetShortcut)
        self._clear_shortcut.activated.connect(self.clear_selection)
        self._offset_decrease_shortcut = QShortcut(QKeySequence("Alt+Left"), self)
        self._offset_decrease_shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self._offset_decrease_shortcut.activated.connect(lambda: self._nudge_boundary_offset(-1))
        self._offset_increase_shortcut = QShortcut(QKeySequence("Alt+Right"), self)
        self._offset_increase_shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self._offset_increase_shortcut.activated.connect(lambda: self._nudge_boundary_offset(1))
        self._sync_scan_material_ui()

    def set_slots(self, slots: Sequence[RollPreviewSlot]) -> None:
        """Display every scanner slot and start with a deliberate empty choice."""

        self.model.replace_slots(slots)
        self.clear_selection()
        count = self.model.rowCount()
        self.summary_label.setText("No preview loaded" if count == 0 else f"{count} scanner slots")
        self._sync_selection_ui()
        self._sync_boundary_offset_ui()

    def selected_slot_ids(self) -> list[int]:
        """Return explicit physical slot IDs in display order, not click order."""

        rows = sorted(index.row() for index in self._selection_model().selectedIndexes())
        return [self.model.slot_id_for_row(row) for row in rows]

    def set_selected_slot_ids(self, slot_ids: Iterable[int]) -> None:
        """Replace the selection using physical 1-based slot IDs."""

        rows = sorted({self.model.row_for_slot_id(slot_id) for slot_id in slot_ids})
        selection_model = self._selection_model()
        selection_model.clearSelection()
        if not rows:
            return

        for row in rows:
            index = self.model.index(row, 0)
            selection_model.select(index, QItemSelectionModel.SelectionFlag.Select)
        selection_model.setCurrentIndex(self.model.index(rows[-1], 0), QItemSelectionModel.SelectionFlag.NoUpdate)

    def select_all(self) -> None:
        if self.model.rowCount() == 0:
            return
        first = self.model.index(0, 0)
        last = self.model.index(self.model.rowCount() - 1, 0)
        self._selection_model().select(QItemSelection(first, last), QItemSelectionModel.SelectionFlag.ClearAndSelect)

    def clear_selection(self) -> None:
        self._selection_model().clearSelection()

    def request_scan(self) -> None:
        """Emit the selected physical slots; scanner execution lives elsewhere."""

        slot_ids = self.selected_slot_ids()
        if slot_ids:
            self.scan_requested.emit(slot_ids)

    def scan_material(self) -> ScanMaterial:
        """Return the capture material currently chosen by the user."""

        return ScanMaterial(self.scan_material_combo.currentData())

    def set_scan_material(self, material: ScanMaterial | str) -> None:
        """Choose a material by enum or persisted display value."""

        material = ScanMaterial(material)
        index = self.scan_material_combo.findData(material.value)
        if index < 0:
            raise ValueError(f"scan material is unavailable: {material.value}")
        self.scan_material_combo.setCurrentIndex(index)

    def update_slot_thumbnail(self, slot_id: int, thumbnail: QIcon | QImage | QPixmap | None) -> None:
        """Install a replacement preview without disturbing slot selection."""

        self.model.update_thumbnail(slot_id, thumbnail)

    def confirm_slot_thumbnail(
        self,
        slot_id: int,
        boundary_offset: int,
        thumbnail: QIcon | QImage | QPixmap | None,
    ) -> None:
        """Confirm that the displayed recrop matches the pending scan offset."""

        self.model.confirm_thumbnail(slot_id, boundary_offset, thumbnail)
        self._sync_selection_ui()

    def request_thumbnail_reload(self) -> None:
        """Emit the current physical slot and its persisted boundary offset."""

        current = self._selection_model().currentIndex()
        if not current.isValid():
            return
        slot_id = self.model.slot_id_for_row(current.row())
        self.thumbnail_reload_requested.emit(slot_id, self.model.boundary_offset_for_slot_id(slot_id))

    def _on_selection_changed(self, _selected: QItemSelection, _deselected: QItemSelection) -> None:
        self._sync_selection_ui()
        self.selection_changed.emit(self.selected_slot_ids())

    def _on_current_slot_changed(self, _current: QModelIndex, _previous: QModelIndex) -> None:
        self._sync_boundary_offset_ui()

    def _on_boundary_offset_changed(self, boundary_offset: int) -> None:
        slider_blocker = QSignalBlocker(self.boundary_offset_slider)
        self.boundary_offset_slider.setValue(boundary_offset)
        del slider_blocker
        current = self._selection_model().currentIndex()
        if current.isValid():
            self.model.set_boundary_offset(self.model.slot_id_for_row(current.row()), boundary_offset)
            self._sync_selection_ui()

    def _nudge_boundary_offset(self, delta: int) -> None:
        """Move the current selected frame by one valid transport row."""

        current = self._selection_model().currentIndex()
        if not current.isValid() or not self._selection_model().isSelected(current):
            return
        next_value = max(
            self.boundary_offset_spin.minimum(),
            min(self.boundary_offset_spin.maximum(), self.boundary_offset_spin.value() + delta),
        )
        self.boundary_offset_spin.setValue(next_value)

    def _on_scan_material_changed(self, _index: int) -> None:
        self._sync_scan_material_ui()
        self.scan_material_changed.emit(self.scan_material())

    def _sync_selection_ui(self) -> None:
        selected_ids = self.selected_slot_ids()
        selected = len(selected_ids)
        total = self.model.rowCount()
        self.selected_count_label.setText(f"{selected} of {total} selected")
        unconfirmed = [slot_id for slot_id in selected_ids if not self.model.boundary_offset_is_confirmed(slot_id)]
        self.scan_button.setEnabled(selected > 0 and not unconfirmed)
        self.scan_button.setToolTip(
            "Reload each edited thumbnail before scanning"
            if unconfirmed
            else "Scan the selected slots at the material's full-quality recipe"
        )

    def _sync_boundary_offset_ui(self) -> None:
        current = self._selection_model().currentIndex()
        enabled = current.isValid()
        self.boundary_offset_spin.setEnabled(enabled)
        self.boundary_offset_slider.setEnabled(enabled)
        self.reload_thumbnail_button.setEnabled(enabled)
        if not enabled:
            self.boundary_offset_label.setText("Film Spacing Offset")
            return

        slot_id = self.model.slot_id_for_row(current.row())
        minimum, maximum = boundary_offset_range(slot_id)
        spin_blocker = QSignalBlocker(self.boundary_offset_spin)
        slider_blocker = QSignalBlocker(self.boundary_offset_slider)
        self.boundary_offset_spin.setRange(minimum, maximum)
        self.boundary_offset_spin.setValue(self.model.boundary_offset_for_slot_id(slot_id))
        self.boundary_offset_slider.setRange(minimum, maximum)
        self.boundary_offset_slider.setValue(self.model.boundary_offset_for_slot_id(slot_id))
        del spin_blocker, slider_blocker
        self.boundary_offset_label.setText(f"Film Spacing Offset: Slot {slot_id:02d}")

    def _sync_scan_material_ui(self) -> None:
        if self.scan_material().captures_infrared:
            self.scan_material_status_label.setText("Full quality: 4000 dpi · 16-bit · RGB 4× + IR · IR dust repair available")
        else:
            self.scan_material_status_label.setText("Full quality: 4000 dpi · 16-bit · RGB only, 4× · IR/dust repair off")

    def _selection_model(self) -> QItemSelectionModel:
        selection_model = self.grid.selectionModel()
        if selection_model is None:
            raise RuntimeError("roll slot selector has no selection model")
        return selection_model
