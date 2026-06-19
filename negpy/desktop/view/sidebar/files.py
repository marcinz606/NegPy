import os

import qtawesome as qta
from PyQt6.QtCore import Qt, QItemSelectionModel, QModelIndex, QRect, QSize, QTimer, pyqtSignal
from PyQt6.QtGui import QActionGroup, QColor, QPainter, QPen
from PyQt6.QtWidgets import (
    QFileDialog,
    QFrame,
    QHBoxLayout,
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
from negpy.desktop.view.styles.theme import THEME
from negpy.infrastructure.filesystem.watcher import FolderWatchService
from negpy.infrastructure.loaders.helpers import get_supported_raw_wildcards


class _ThumbnailDelegate(QStyledItemDelegate):
    """Contact-sheet rendering: scales each cached ~120px thumbnail into its cell and
    draws a subtle 1px border hugging the image outline (no cell box). The selected
    image is shown full-brightness with a white frame while the others are dimmed; a
    dirty active file gets an accent line along the image's bottom edge."""

    _MARGIN = 3

    def paint(self, painter: QPainter, option: QStyleOptionViewItem, index: QModelIndex) -> None:
        icon = index.data(Qt.ItemDataRole.DecorationRole)
        if icon is None or icon.isNull():
            return
        base = icon.pixmap(QSize(4096, 4096))  # largest available pixmap (~120px)
        if base.isNull():
            return

        painter.save()
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)

        area = option.rect.adjusted(self._MARGIN, self._MARGIN, -self._MARGIN, -self._MARGIN)
        scaled = base.scaled(
            area.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        x = area.x() + (area.width() - scaled.width()) // 2
        y = area.y() + (area.height() - scaled.height()) // 2
        img_rect = QRect(x, y, scaled.width(), scaled.height())

        # Selected image full-brightness with a white frame; others dimmed.
        selected = bool(option.state & QStyle.StateFlag.State_Selected)
        hover = bool(option.state & QStyle.StateFlag.State_MouseOver)

        painter.setOpacity(1.0 if (selected or hover) else 0.5)
        painter.drawPixmap(img_rect.topLeft(), scaled)
        painter.setOpacity(1.0)

        if selected:
            pen = QPen(QColor(THEME.accent_edited), 2)
        elif hover:
            pen = QPen(QColor(THEME.text_muted), 1)
        else:
            pen = QPen(QColor(THEME.border_color), 1)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(img_rect.adjusted(0, 0, -1, -1))

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

        self.sync_btn = QToolButton()
        self.sync_btn.setIcon(qta.icon("fa5s.sync", color=THEME.text_primary))
        self.sync_btn.setToolTip("Sync Edits — apply exposure / lab / toning to selected images (preserves their crop and rotation)")

        self.sync_crop_btn = QToolButton()
        self.sync_crop_btn.setIcon(qta.icon("fa5s.crop", color=THEME.text_primary))
        self.sync_crop_btn.setToolTip("Sync Crop — apply current crop and rotation to selected images")

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
            self.sync_btn,
            self.sync_crop_btn,
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
        toolbar_row.addWidget(self.sync_btn)
        toolbar_row.addWidget(self.sync_crop_btn)
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
        self.list_view.doubleClicked.connect(self._on_item_double_clicked)
        self.list_view.customContextMenuRequested.connect(self._show_context_menu)
        self.list_view.selectionModel().selectionChanged.connect(self._on_selection_changed)
        self.hot_folder_btn.toggled.connect(self._on_hot_folder_toggled)
        self.sync_btn.clicked.connect(lambda *_: self.session.sync_selected_settings("edits"))
        self.sync_crop_btn.clicked.connect(lambda *_: self.session.sync_selected_settings("geometry_only"))
        self.session.state_changed.connect(self.sync_ui)
        self.session.files_changed.connect(self.sync_ui)
        self.search_input.textChanged.connect(lambda _: self.filter_timer.start())
        self.regex_btn.toggled.connect(lambda _: self.filter_timer.start())

    def _on_unload_clicked(self) -> None:
        if len(self.session.state.selected_indices) > 1:
            self.session.remove_selected_files()
        else:
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
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Images",
            "",
            f"Supported Images ({wildcards})",
        )
        if files:
            self.controller.request_asset_discovery(files, auto_open=True)

    def _on_add_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder:
            self.controller.request_asset_discovery([folder], auto_open=True)

    def _on_item_double_clicked(self, index) -> None:
        actual = self.session.asset_model.display_to_actual(index.row())
        if actual >= 0:
            self.session.select_file(actual)

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

    def _build_context_menu(self) -> QMenu:
        state = self.session.state
        multi = len(state.selected_indices) > 1

        menu = QMenu(self)
        if multi:
            menu.addAction("Export Selected").triggered.connect(lambda: self.controller.request_export_selected())
        else:
            menu.addAction("Export").triggered.connect(lambda: self.controller.request_export())
        menu.addSeparator()
        menu.addAction("Copy Settings  Ctrl+C").triggered.connect(self.session.copy_settings)
        menu.addAction("Copy Settings + Bounds  Ctrl+Shift+C").triggered.connect(self.session.copy_settings_with_bounds)
        act_paste = menu.addAction("Paste Settings  Ctrl+V")
        act_paste.triggered.connect(self.session.paste_settings)
        act_paste.setEnabled(state.clipboard is not None)
        menu.addAction("Reset Settings").triggered.connect(self.session.reset_settings)
        if multi:
            menu.addSeparator()
            menu.addAction("Sync Edits to Selection").triggered.connect(lambda: self.session.sync_selected_settings("edits"))
        menu.addSeparator()
        unload_label = "Unload Selected" if multi else "Unload"
        menu.addAction(unload_label).triggered.connect(self._on_remove_from_menu)
        return menu

    def _on_remove_from_menu(self) -> None:
        if len(self.session.state.selected_indices) > 1:
            self.session.remove_selected_files()
        else:
            self.session.remove_current_file()
