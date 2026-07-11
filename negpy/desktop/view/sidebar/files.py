import os

import qtawesome as qta
from PyQt6.QtCore import Qt, QItemSelectionModel, QModelIndex, QRect, QRectF, QSize, QTimer, pyqtSignal
from PyQt6.QtGui import QActionGroup, QColor, QKeySequence, QPainter, QPainterPath, QPen, QShortcut
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListView,
    QMenu,
    QPushButton,
    QStyle,
    QStyledItemDelegate,
    QStyleOptionViewItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from negpy.desktop.controller import AppController
from negpy.desktop.view.confirm import confirm_unload
from negpy.desktop.view.styles.theme import THEME
from negpy.desktop.view.widgets.sync_settings_dialog import SyncSettingsDialog
from negpy.infrastructure.filesystem.watcher import FolderWatchService
from negpy.infrastructure.loaders.helpers import get_supported_raw_wildcards


class _ThumbnailDelegate(QStyledItemDelegate):
    """Contact-sheet rendering: scales each cached ~120px thumbnail into its cell and
    draws a subtle 1px border hugging the image outline (no cell box). The selected
    image is shown full-brightness with a white frame while the others are dimmed; a
    dirty active file gets an accent line along the image's bottom edge."""

    _MARGIN = 3
    _RADIUS = 4  # = button border-radius (modern_dark.qss)

    def paint(self, painter: QPainter, option: QStyleOptionViewItem, index: QModelIndex) -> None:
        icon = index.data(Qt.ItemDataRole.DecorationRole)
        if icon is None or icon.isNull():
            return
        base = icon.pixmap(QSize(4096, 4096))  # largest available pixmap (~120px)
        if base.isNull():
            return

        painter.save()
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        area = option.rect.adjusted(self._MARGIN, self._MARGIN, -self._MARGIN, -self._MARGIN)
        scaled = base.scaled(
            area.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        x = area.x() + (area.width() - scaled.width()) // 2
        y = area.y() + (area.height() - scaled.height()) // 2
        img_rect = QRect(x, y, scaled.width(), scaled.height())

        # Selected image full-brightness with the armed-red frame; others dimmed.
        selected = bool(option.state & QStyle.StateFlag.State_Selected)
        hover = bool(option.state & QStyle.StateFlag.State_MouseOver)

        clip = QPainterPath()
        clip.addRoundedRect(QRectF(img_rect), self._RADIUS, self._RADIUS)
        painter.setClipPath(clip)
        painter.setOpacity(1.0 if (selected or hover) else 0.5)
        painter.drawPixmap(img_rect.topLeft(), scaled)
        painter.setOpacity(1.0)
        painter.setClipping(False)

        if selected:
            pen = QPen(QColor(THEME.accent_primary), 2)
        elif hover:
            pen = QPen(QColor(THEME.text_muted), 1)
        else:
            pen = QPen(QColor(THEME.border_color), 1)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRoundedRect(img_rect.adjusted(0, 0, -1, -1), self._RADIUS, self._RADIUS)

        painter.restore()


class ThumbnailGridView(QListView):
    """
    Icon-mode grid that justifies thumbnails to the panel width. It fits as many
    MIN_CELL-wide columns as possible, then scales the cell up (to MAX_CELL) to fill
    the width; once there's room for another MIN_CELL column it adds one and the cells
    snap back down. e.g. with MIN 120 / MAX 180: 2 columns grow 120→180, and at ~3×120
    of width a 3rd column appears.
    """

    MIN_CELL = 120
    MAX_CELL = 180
    SPACING = 2

    def __init__(self, parent=None):
        super().__init__(parent)
        self._last_cell = -1
        # Reserve the vertical scrollbar permanently so the viewport width is stable —
        # otherwise scaling toggles the scrollbar, which changes the width and flips the
        # column count back, causing flicker.
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setSpacing(self.SPACING)
        self._apply_cell(self.MIN_CELL)

    def _apply_cell(self, cell: int) -> None:
        if cell == self._last_cell:
            return
        self._last_cell = cell
        self.setGridSize(QSize(cell + self.SPACING, cell + self.SPACING))
        self.setIconSize(QSize(cell, cell))

    def _relayout(self) -> None:
        vw = self.viewport().width()
        columns = max(1, (vw - self.SPACING) // (self.MIN_CELL + self.SPACING))
        cell = (vw - (columns + 1) * self.SPACING) // columns
        cell = max(self.MIN_CELL, min(self.MAX_CELL, cell))
        self._apply_cell(cell)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._relayout()

    def wheelEvent(self, event) -> None:
        pixel = event.pixelDelta()
        if not pixel.isNull() and pixel.y() != 0:
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - pixel.y())
            event.accept()
        else:
            super().wheelEvent(event)


class FileBrowser(QWidget):
    """
    Asset management panel for loading and selecting images.
    """

    file_selected = pyqtSignal(str)

    def __init__(self, controller: AppController):
        super().__init__()
        self.controller = controller
        self.session = controller.session

        self.scan_timer = QTimer(self)
        self.scan_timer.setInterval(2000)
        self.scan_timer.timeout.connect(self._scan_folder)

        self.selection_timer = QTimer(self)
        self.selection_timer.setSingleShot(True)
        self.selection_timer.setInterval(200)
        self.selection_timer.timeout.connect(self._commit_selection)

        self.filter_timer = QTimer(self)
        self.filter_timer.setSingleShot(True)
        self.filter_timer.setInterval(200)
        self.filter_timer.timeout.connect(self._apply_filter)

        self._init_ui()
        self._connect_signals()

    def _create_separator(self) -> QFrame:
        line = QFrame()
        line.setFrameShape(QFrame.Shape.VLine)
        line.setFrameShadow(QFrame.Shadow.Plain)
        line.setObjectName("toolbar_separator")
        line.setFixedWidth(1)
        return line

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(6)

        icon_size = QSize(16, 16)
        btn_height = 28

        toolbar_row = QHBoxLayout()
        toolbar_row.setSpacing(4)

        self.add_files_btn = QToolButton()
        self.add_files_btn.setIcon(qta.icon("fa5s.file-import", color=THEME.text_primary))
        self.add_files_btn.setToolTip("Add files")
        self.add_folder_btn = QToolButton()
        self.add_folder_btn.setIcon(qta.icon("fa5s.folder-plus", color=THEME.text_primary))
        self.add_folder_btn.setToolTip("Add folder")
        self.unload_btn = QToolButton()
        self.unload_btn.setIcon(qta.icon("fa5s.times-circle", color=THEME.text_primary))
        self.unload_btn.setToolTip("Clear all")

        self.hot_folder_btn = QToolButton()
        self.hot_folder_btn.setCheckable(True)
        self.hot_folder_btn.setIcon(qta.icon("fa5s.fire", color=THEME.text_primary))
        self.hot_folder_btn.setToolTip("Hot Folder — automatically load new images from the current folder")
        self._update_hot_folder_style(False)

        self.rgb_scan_btn = QToolButton()
        self.rgb_scan_btn.setCheckable(True)
        self.rgb_scan_btn.setIcon(qta.icon("mdi.google-circles-communities", color=THEME.text_primary))
        self.rgb_scan_btn.setToolTip("RGB Scan — assemble each frame from red/green/blue exposures; groups a folder into triplets on load")
        self.rgb_scan_btn.setChecked(bool(self.session.repo.get_global_setting("rgbscan_mode", False)))
        self._update_rgb_scan_style(self.rgb_scan_btn.isChecked())

        self.apply_btn = QToolButton()
        self.apply_btn.setIcon(qta.icon("fa5s.clone", color=THEME.text_primary))
        self.apply_btn.setToolTip("Apply settings from the current frame to selected frames or the whole roll")
        self.apply_btn.clicked.connect(self._open_apply_dialog)

        # Sort dropdown
        self.sort_btn = QToolButton()
        self.sort_btn.setIcon(qta.icon("fa5s.sort", color=THEME.text_primary))
        self.sort_btn.setToolTip("Sort")
        self.sort_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)

        sort_menu = QMenu(self.sort_btn)
        self._order_group = QActionGroup(self)
        self._order_group.setExclusive(True)
        self.act_sort_name = sort_menu.addAction("Name")
        self.act_sort_date = sort_menu.addAction("Date")
        for act in (self.act_sort_name, self.act_sort_date):
            act.setCheckable(True)
            self._order_group.addAction(act)
        sort_menu.addSeparator()
        self._dir_group = QActionGroup(self)
        self._dir_group.setExclusive(True)
        self.act_sort_asc = sort_menu.addAction("Ascending")
        self.act_sort_desc = sort_menu.addAction("Descending")
        for act in (self.act_sort_asc, self.act_sort_desc):
            act.setCheckable(True)
            self._dir_group.addAction(act)
        self.act_sort_name.triggered.connect(lambda: self._apply_sort_order("name"))
        self.act_sort_date.triggered.connect(lambda: self._apply_sort_order("date"))
        self.act_sort_asc.triggered.connect(lambda: self._apply_sort_direction(False))
        self.act_sort_desc.triggered.connect(lambda: self._apply_sort_direction(True))
        self.sort_btn.setMenu(sort_menu)

        for btn in (
            self.add_files_btn,
            self.add_folder_btn,
            self.unload_btn,
            self.hot_folder_btn,
            self.rgb_scan_btn,
            self.apply_btn,
            self.sort_btn,
        ):
            btn.setIconSize(icon_size)
            btn.setFixedHeight(btn_height)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)

        toolbar_row.addWidget(self.add_files_btn)
        toolbar_row.addWidget(self.add_folder_btn)
        toolbar_row.addWidget(self.unload_btn)
        toolbar_row.addWidget(self._create_separator())
        toolbar_row.addWidget(self.hot_folder_btn)
        toolbar_row.addWidget(self.rgb_scan_btn)
        toolbar_row.addWidget(self.apply_btn)
        toolbar_row.addStretch()
        toolbar_row.addWidget(self._create_separator())
        toolbar_row.addWidget(self.sort_btn)
        layout.addLayout(toolbar_row)

        saved_sort = self.session.repo.get_global_setting("file_sort_order") or "name"
        saved_desc = self.session.repo.get_global_setting("file_sort_descending") or False
        self._apply_sort_order(str(saved_sort), save=False)
        self._apply_sort_direction(bool(saved_desc), save=False)

        search_row = QHBoxLayout()
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Filter by filename...")
        self.search_input.setClearButtonEnabled(True)
        self.search_input.addAction(
            qta.icon("fa5s.search", color=THEME.text_secondary),
            QLineEdit.ActionPosition.LeadingPosition,
        )
        self.regex_btn = QPushButton(".*")
        self.regex_btn.setCheckable(True)
        self.regex_btn.setFixedWidth(36)
        self.regex_btn.setToolTip("Regex mode")
        search_row.addWidget(self.search_input)
        search_row.addWidget(self.regex_btn)
        layout.addLayout(search_row)

        self.list_view = ThumbnailGridView()
        self.list_view.setModel(self.session.asset_model)
        self.list_view.setItemDelegate(_ThumbnailDelegate(self.list_view))
        self.list_view.setViewMode(QListView.ViewMode.IconMode)
        self.list_view.setResizeMode(QListView.ResizeMode.Adjust)
        self.list_view.setSelectionMode(QListView.SelectionMode.ExtendedSelection)
        self.list_view.setAlternatingRowColors(False)
        self.list_view.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)

        layout.addWidget(self.list_view)

    def _connect_signals(self) -> None:
        self.add_files_btn.clicked.connect(self._on_add_files)
        self.add_folder_btn.clicked.connect(self._on_add_folder)
        self.unload_btn.clicked.connect(self._on_unload_clicked)
        self.list_view.clicked.connect(self._on_item_clicked)
        self.list_view.doubleClicked.connect(self._on_item_double_clicked)
        self.list_view.customContextMenuRequested.connect(self._show_context_menu)
        self.list_view.selectionModel().selectionChanged.connect(self._on_selection_changed)
        self.hot_folder_btn.toggled.connect(self._on_hot_folder_toggled)
        self.rgb_scan_btn.toggled.connect(self._on_rgb_scan_toggled)
        self.session.state_changed.connect(self.sync_ui)
        self.session.files_changed.connect(self.sync_ui)
        self.search_input.textChanged.connect(lambda _: self.filter_timer.start())
        self.regex_btn.toggled.connect(lambda _: self.filter_timer.start())

        # Delete key unloads the selected frame(s) — scoped to the thumbnail list so it
        # doesn't fire while typing in the filter box or editing elsewhere.
        del_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Delete), self.list_view)
        del_shortcut.setContext(Qt.ShortcutContext.WidgetShortcut)
        del_shortcut.activated.connect(self._on_delete_key)

    def _on_unload_clicked(self) -> None:
        count = len(self.session.state.selected_indices)
        if count > 1:
            if confirm_unload(self, count=count):
                self.session.remove_selected_files()
        else:
            if confirm_unload(self, clear_all=True):
                self.session.clear_files()

    def _update_unload_button(self) -> None:
        if len(self.session.state.selected_indices) > 1:
            self.unload_btn.setToolTip("Clear selected")
        else:
            self.unload_btn.setToolTip("Clear all")

    def sync_ui(self) -> None:
        """Updates list selection to match session state."""
        model = self.session.asset_model
        selection_model = self.list_view.selectionModel()
        self._update_unload_button()

        current_actual = {
            model.display_to_actual(idx.row()) for idx in selection_model.selectedIndexes() if model.display_to_actual(idx.row()) >= 0
        }
        target_actual = set(self.session.state.selected_indices)

        # Repaint for dirty underline
        self.list_view.viewport().update()

        if current_actual == target_actual:
            return

        selection_model.blockSignals(True)
        try:
            selection_model.clearSelection()
            for actual_idx in self.session.state.selected_indices:
                display_row = model.actual_to_display(actual_idx)
                if display_row >= 0:
                    qt_idx = model.index(display_row, 0)
                    selection_model.select(qt_idx, QItemSelectionModel.SelectionFlag.Select)

            active_idx = self.session.state.selected_file_idx
            if active_idx >= 0:
                display_row = model.actual_to_display(active_idx)
                if display_row >= 0:
                    qt_idx = model.index(display_row, 0)
                    selection_model.setCurrentIndex(qt_idx, QItemSelectionModel.SelectionFlag.NoUpdate)
                    self.list_view.scrollTo(qt_idx)
        finally:
            selection_model.blockSignals(False)

    def _on_selection_changed(self, selected, deselected) -> None:
        self.selection_timer.start()

    def _commit_selection(self) -> None:
        """Sends current UI selection to the session after debounce."""
        model = self.session.asset_model
        actual_indices = [a for idx in self.list_view.selectionModel().selectedIndexes() if (a := model.display_to_actual(idx.row())) >= 0]
        if set(actual_indices) != set(self.session.state.selected_indices):
            self.session.update_selection(actual_indices)

    def _apply_filter(self) -> None:
        text = self.search_input.text().strip()
        regex = self.regex_btn.isChecked()
        ok = self.session.asset_model.set_filter(text, regex)
        self._set_search_error(not ok)
        if ok:
            self._prune_selection_to_visible()
            self.sync_ui()

    def _set_search_error(self, error: bool) -> None:
        if error:
            self.search_input.setStyleSheet(f"border: 1px solid {THEME.accent_primary};")
        else:
            self.search_input.setStyleSheet("")

    def _prune_selection_to_visible(self) -> None:
        visible = self.session.asset_model.visible_actual_indices()
        state = self.session.state
        new_selection = [i for i in state.selected_indices if i in visible]
        if state.selected_file_idx in visible:
            new_active = state.selected_file_idx
        elif new_selection:
            new_active = new_selection[0]
        else:
            new_active = -1

        selection_changed = new_selection != state.selected_indices
        active_changed = new_active != state.selected_file_idx

        if active_changed and new_active >= 0:
            self.session.select_file(new_active, selection_override=new_selection)
            return

        if selection_changed:
            self.session.update_selection(new_selection)
        if active_changed and new_active == -1:
            state.selected_file_idx = -1
            self.session.state_changed.emit()

    def _apply_sort_order(self, order: str, save: bool = True) -> None:
        self.act_sort_name.setChecked(order == "name")
        self.act_sort_date.setChecked(order == "date")
        self.session.asset_model.set_sort_order(order)
        if save:
            self.session.repo.save_global_setting("file_sort_order", order)

    def _apply_sort_direction(self, descending: bool, save: bool = True) -> None:
        self.act_sort_asc.setChecked(not descending)
        self.act_sort_desc.setChecked(descending)
        self.session.asset_model.set_sort_descending(descending)
        if save:
            self.session.repo.save_global_setting("file_sort_descending", descending)

    def _on_hot_folder_toggled(self, checked: bool) -> None:
        self._update_hot_folder_style(checked)
        if checked:
            self.scan_timer.start()
        else:
            self.scan_timer.stop()

    def _update_hot_folder_style(self, checked: bool) -> None:
        icon_color = "white" if checked else THEME.text_primary
        self.hot_folder_btn.setIcon(qta.icon("fa5s.fire", color=icon_color))

    def _update_rgb_scan_style(self, checked: bool) -> None:
        icon_color = "white" if checked else THEME.text_primary
        self.rgb_scan_btn.setIcon(qta.icon("mdi.google-circles-communities", color=icon_color))

    def _on_rgb_scan_toggled(self, checked: bool) -> None:
        self._update_rgb_scan_style(checked)
        self.controller.set_rgb_scan_mode(checked)

    def _scan_folder(self) -> None:
        if not self.session.state.uploaded_files:
            return

        last_file = self.session.state.uploaded_files[-1]
        folder_path = os.path.dirname(last_file["path"])
        existing = {f["path"] for f in self.session.state.uploaded_files}

        new_files = FolderWatchService.scan_for_new_files(folder_path, existing)
        if new_files:
            self.controller.request_asset_discovery(new_files)

    def _on_add_files(self) -> None:
        wildcards = get_supported_raw_wildcards()
        start_dir = self.session.repo.get_global_setting("last_open_folder", "") or ""
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Images",
            start_dir,
            f"Supported Images ({wildcards})",
        )
        if files:
            self.session.repo.save_global_setting("last_open_folder", os.path.dirname(files[0]))
            self.controller.request_asset_discovery(files, auto_open=True)

    def _on_add_folder(self) -> None:
        start_dir = self.session.repo.get_global_setting("last_open_folder", "") or ""
        folder = QFileDialog.getExistingDirectory(self, "Select Folder", start_dir)
        if folder:
            self.session.repo.save_global_setting("last_open_folder", os.path.dirname(folder))
            self.controller.request_asset_discovery([folder], auto_open=True)

    def _activate_file(self, index) -> None:
        """Load a thumbnail into the main viewer, skipping a redundant reload of the
        already-active frame."""
        actual = self.session.asset_model.display_to_actual(index.row())
        if actual >= 0 and actual != self.session.state.selected_file_idx:
            self.session.select_file(actual)

    def _on_item_clicked(self, index) -> None:
        # Plain single click sets the active frame instantly. Ctrl/Shift clicks build a
        # multi-selection for batch actions and are left to the selectionChanged handler.
        if QApplication.keyboardModifiers() & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier):
            return
        self._activate_file(index)

    def _on_item_double_clicked(self, index) -> None:
        self._activate_file(index)

    def _show_context_menu(self, pos) -> None:
        index = self.list_view.indexAt(pos)
        if not index.isValid():
            return
        actual = self.session.asset_model.display_to_actual(index.row())
        if actual < 0:
            return

        # Right-clicking outside the current selection re-selects just that file;
        # within a multi-selection, keep the selection and make the clicked file active.
        state = self.session.state
        if actual not in state.selected_indices:
            self.session.select_file(actual)
        elif actual != state.selected_file_idx:
            self.session.select_file(actual, selection_override=list(state.selected_indices))

        menu = self._build_context_menu()
        menu.exec(self.list_view.viewport().mapToGlobal(pos))

    def _source_name(self) -> str:
        idx = self.session.state.selected_file_idx
        files = self.session.state.uploaded_files
        return os.path.basename(files[idx]["path"]) if 0 <= idx < len(files) else ""

    def _open_apply_dialog(self) -> None:
        state = self.session.state
        src = state.selected_file_idx
        if src == -1:
            return
        # "Whole roll" means the visible (filtered) frames, not every loaded file —
        # a filename filter is a non-destructive view, so hidden files aren't counted.
        visible = self.session.asset_model.visible_actual_indices()
        sel_targets = len([i for i in set(state.selected_indices) if i != src and i in visible])
        roll_targets = len([i for i in visible if i != src])

        dlg = SyncSettingsDialog(self, self._source_name(), sel_targets, roll_targets)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self.session.sync_selected_settings(dlg.aspects(), dlg.scope())

    def _build_context_menu(self) -> QMenu:
        state = self.session.state
        multi = len(state.selected_indices) > 1

        menu = QMenu(self)
        if multi:
            menu.addAction("Export selected frames").triggered.connect(lambda: self.controller.request_export_selected())
        else:
            menu.addAction("Export current frame").triggered.connect(lambda: self.controller.request_export())
        menu.addSeparator()
        menu.addAction("Copy Settings  Ctrl+C").triggered.connect(self.session.copy_settings)
        menu.addAction("Copy Settings + Bounds  Ctrl+Shift+C").triggered.connect(self.session.copy_settings_with_bounds)
        act_paste = menu.addAction("Paste Settings  Ctrl+V")
        act_paste.triggered.connect(self.session.paste_settings)
        act_paste.setEnabled(state.clipboard is not None)
        menu.addAction("Reset Settings").triggered.connect(self.session.reset_settings)
        menu.addSeparator()
        menu.addAction("Apply settings…").triggered.connect(self._open_apply_dialog)
        if not multi:
            menu.addSeparator()
            menu.addAction("Edit RGB Triplet…").triggered.connect(self._on_edit_triplet)
        menu.addSeparator()
        unload_label = "Unload Selected" if multi else "Unload"
        menu.addAction(unload_label).triggered.connect(self._on_remove_from_menu)
        return menu

    def _on_edit_triplet(self) -> None:
        idx = self.session.state.selected_file_idx
        files = self.session.state.uploaded_files
        if not (0 <= idx < len(files)):
            return
        info = files[idx]
        dlg = _RgbTripletDialog(self, info["path"], info.get("green_path", ""), info.get("blue_path", ""), info.get("align", True))
        if dlg.exec():
            red, green, blue = dlg.paths()
            if red and green and blue:
                self.session.set_triplet(idx, red, green, blue, dlg.align())

    def _on_remove_from_menu(self) -> None:
        count = len(self.session.state.selected_indices)
        if count > 1:
            if confirm_unload(self, count=count):
                self.session.remove_selected_files()
        else:
            if confirm_unload(self):
                self.session.remove_current_file()

    def _on_delete_key(self) -> None:
        """Delete key in the thumbnail list unloads the selected frame(s)."""
        state = self.session.state
        if not state.uploaded_files or state.selected_file_idx < 0:
            return
        self._on_remove_from_menu()


class _RgbTripletDialog(QDialog):
    """Manually assign the red/green/blue exposure files for one RGB-scan frame."""

    def __init__(self, parent, red: str, green: str, blue: str, align: bool = True) -> None:
        super().__init__(parent)
        self.setWindowTitle("Edit RGB Triplet")
        layout = QVBoxLayout(self)
        self._edits: dict[str, QLineEdit] = {}
        for label, path in (("Red", red), ("Green", green), ("Blue", blue)):
            row = QHBoxLayout()
            row.addWidget(QLabel(label, minimumWidth=48))
            edit = QLineEdit(path)
            row.addWidget(edit, 1)
            browse = QPushButton("Browse…")
            browse.clicked.connect(lambda _=False, e=edit: self._browse(e))
            row.addWidget(browse)
            layout.addLayout(row)
            self._edits[label] = edit

        self._align = QCheckBox("Align channels (sub-pixel)")
        self._align.setChecked(align)
        self._align.setToolTip("Register green/blue to the red exposure to remove fringing from capture drift.")
        layout.addWidget(self._align)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _browse(self, edit: QLineEdit) -> None:
        start = os.path.dirname(edit.text()) if edit.text() else ""
        path, _ = QFileDialog.getOpenFileName(self, "Select exposure", start, f"Supported Images ({get_supported_raw_wildcards()})")
        if path:
            edit.setText(path)

    def paths(self) -> tuple[str, str, str]:
        return (self._edits["Red"].text(), self._edits["Green"].text(), self._edits["Blue"].text())

    def align(self) -> bool:
        return self._align.isChecked()
