import os

import qtawesome as qta
from PyQt6.QtCore import (
    Qt,
    QEasingCurve,
    QItemSelectionModel,
    QModelIndex,
    QPropertyAnimation,
    QRect,
    QRectF,
    QSize,
    QTimer,
    pyqtSignal,
)
from PyQt6.QtGui import QActionGroup, QColor, QKeySequence, QPainter, QPainterPath, QPen, QShortcut
from PyQt6.QtWidgets import (
    QAbstractItemView,
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
    QMessageBox,
    QPushButton,
    QSlider,
    QStyle,
    QStyledItemDelegate,
    QStyleOptionViewItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from negpy.kernel.system.text import count_of
from negpy.desktop.controller import AppController
from negpy.desktop.session import _source_effective_bounds
from negpy.desktop.view.confirm import confirm_unload
from negpy.features.hdr.logic import anchor_choices
from negpy.features.hdr.models import hdr_frame_paths
from negpy.desktop.view.widgets.overflow_bar import OverflowBar
from negpy.desktop.view.styles.theme import THEME
from negpy.desktop.view.widgets.granular_settings_dialog import GranularSettingsDialog, open_paste_dialog
from negpy.infrastructure.filesystem.watcher import FolderWatchService
from negpy.infrastructure.loaders.helpers import get_supported_raw_wildcards
from negpy.desktop.view.sidebar.library_tree import LibraryTree
from negpy.desktop.view.widgets.collapsible import CollapsibleSection
from negpy.services.assets.library import folder_counts


_UNBOUNDED_HEIGHT = 16777215  # QWIDGETSIZE_MAX — Qt's "no maximum"
# With both sections open the panel splits 40/60: the tree is for finding a roll, the
# sheet is where the work happens, so the frames get the larger half.
_LIBRARY_SHARE, _FRAMES_SHARE = 2, 3


def _folder_label(path: str) -> str:
    return os.path.basename(path.rstrip(os.sep)) or path


class _ThumbnailDelegate(QStyledItemDelegate):
    """Contact-sheet rendering: scales each cached ~120px thumbnail into its cell and
    draws a subtle 1px border hugging the image outline (no cell box). The selected
    image is shown full-brightness with a white frame while the others are dimmed; a
    dirty active file gets an accent line along the image's bottom edge. Triage marks
    are small bottom-right badges: check = keeper, cross + heavy dim = rejected; the
    top-right badge is reserved for decode failures."""

    _MARGIN = 3
    _RADIUS = 4  # = button border-radius (modern_dark.qss)
    _MARK = QColor(183, 28, 28, 150)  # THEME.accent_primary at ~60% alpha

    def _draw_mark_badge(self, painter: QPainter, img_rect: QRect, check: bool) -> None:
        r = 9
        cx, cy = img_rect.right() - r - 4, img_rect.bottom() - r - 4
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(self._MARK)
        painter.drawEllipse(QRect(cx - r, cy - r, 2 * r, 2 * r))
        painter.setPen(QPen(QColor(255, 255, 255, 230), 2, cap=Qt.PenCapStyle.RoundCap))
        if check:
            painter.drawLine(cx - 4, cy, cx - 1, cy + 3)
            painter.drawLine(cx - 1, cy + 3, cx + 4, cy - 3)
        else:
            painter.drawLine(cx - 3, cy - 3, cx + 3, cy + 3)
            painter.drawLine(cx + 3, cy - 3, cx - 3, cy + 3)

    def _draw_failed_badge(self, painter: QPainter, img_rect: QRect) -> None:
        r = 9
        cx, cy = img_rect.right() - r - 4, img_rect.top() + r + 4
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(THEME.accent_primary))
        painter.drawEllipse(QRect(cx - r, cy - r, 2 * r, 2 * r))
        painter.setPen(QPen(QColor("#FFFFFF"), 2))
        painter.drawLine(cx, cy - 4, cx, cy + 1)
        painter.drawPoint(cx, cy + 4)

    def paint(self, painter: QPainter, option: QStyleOptionViewItem, index: QModelIndex) -> None:
        file_info = index.data(Qt.ItemDataRole.UserRole) or {}
        failed = bool(file_info.get("decode_failed"))

        icon = index.data(Qt.ItemDataRole.DecorationRole)
        if icon is None or icon.isNull():
            if failed:
                painter.save()
                painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
                area = option.rect.adjusted(self._MARGIN, self._MARGIN, -self._MARGIN, -self._MARGIN)
                painter.setPen(QPen(QColor(THEME.border_color), 1))
                painter.setBrush(QColor(20, 20, 20))
                painter.drawRoundedRect(area, self._RADIUS, self._RADIUS)
                self._draw_failed_badge(painter, area)
                painter.restore()
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
        rejected = bool(file_info.get("excluded"))
        keeper = bool(file_info.get("keeper"))

        clip = QPainterPath()
        clip.addRoundedRect(QRectF(img_rect), self._RADIUS, self._RADIUS)
        painter.setClipPath(clip)
        base_opacity = 1.0 if (selected or hover) else 0.5
        painter.setOpacity(0.25 if rejected else base_opacity)
        painter.drawPixmap(img_rect.topLeft(), scaled)
        painter.setOpacity(1.0)

        if rejected:
            self._draw_mark_badge(painter, img_rect, check=False)
        elif keeper:
            self._draw_mark_badge(painter, img_rect, check=True)
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

        if failed:
            self._draw_failed_badge(painter, img_rect)

        painter.restore()


# Thumbnail size preference (px), as set by the filmstrip's size slider. The default is
# chosen so one column fills the session sidebar at a typical narrow width (~240px viewport),
# where a single full-width frame is the most legible use of it. Dropping toward
# THUMB_CELL_MIN fits a second column at that same width. (The panel can now be dragged
# narrower still — the toolbar overflows — and the grid simply keeps one column.)
#
# The maximum is the default: the slider only shrinks frames. Anything larger just
# holds a widened sidebar at one column — cells fill the panel, so a 500px-wide
# sidebar became a single ~500px cell, and since cells are square a 3:2 frame in one
# leaves ~165px of empty space above and below. Splitting into two columns is both
# denser and larger-in-practice, so there is nothing above the default worth offering.
THUMB_CELL_MIN = 100
THUMB_CELL_DEFAULT = 220
THUMB_CELL_MAX = THUMB_CELL_DEFAULT


class ThumbnailGridView(QListView):
    """
    Icon-mode grid that justifies thumbnails to the panel width. It fits as many
    columns of at least ``target_cell`` as the viewport allows, then grows the cell to
    fill the leftover width exactly; once another target-wide column fits it adds one
    and the cells snap back down. So ``target_cell`` sets thumbnail size and the panel
    width sets how many fit — e.g. at the default 220 a 240px-wide panel shows one
    236px column, and widening past ~444px splits into two.

    Cells are not capped directly — at one column the frame is meant to fill the panel
    — but capping the *target* at the default bounds them in practice: a target above
    it only delays the split to two columns, which is what produced oversized cells
    (and, since cells are square, large empty bands around a 3:2 frame) on a widened
    sidebar. Cells can still exceed APP_CONFIG.thumbnail_size and upscale on a very
    wide panel; the frames stay legible because the canvas is the place for detail.
    """

    SPACING = 2
    # One notch scrolls one row of thumbnails. Qt's default (ScrollPerItem, three
    # "lines" a notch) advanced 4-5 frames at a time in a single-column panel.
    WHEEL_ROWS_PER_NOTCH = 1.0
    SCROLL_ANIM_MS = 160

    def __init__(self, parent=None, target_cell: int = THUMB_CELL_DEFAULT):
        super().__init__(parent)
        self._last_cell = -1
        self._target_cell = self._clamp_target(target_cell)
        # Reserve the vertical scrollbar permanently so the viewport width is stable —
        # otherwise scaling toggles the scrollbar, which changes the width and flips the
        # column count back, causing flicker.
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setSpacing(self.SPACING)
        # Per-pixel is what makes a partial-row offset representable at all; under
        # ScrollPerItem the view snaps to whole rows and no easing is possible.
        self.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self._scroll_anim = QPropertyAnimation(self.verticalScrollBar(), b"value", self)
        self._scroll_anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._scroll_anim.setDuration(self.SCROLL_ANIM_MS)
        self._scroll_target = 0
        self._apply_cell(self._target_cell)

    @staticmethod
    def _clamp_target(cell: int) -> int:
        return max(THUMB_CELL_MIN, min(THUMB_CELL_MAX, int(cell)))

    @property
    def target_cell(self) -> int:
        return self._target_cell

    def set_target_cell(self, cell: int) -> None:
        """Set the preferred thumbnail size and re-justify to the current width."""
        cell = self._clamp_target(cell)
        if cell == self._target_cell:
            return
        self._target_cell = cell
        self._relayout()

    def columns_for_width(self, vw: int) -> int:
        return max(1, (vw - self.SPACING) // (self._target_cell + self.SPACING))

    def cell_for_width(self, vw: int) -> int:
        columns = self.columns_for_width(vw)
        return max(1, (vw - (columns + 1) * self.SPACING) // columns)

    def _apply_cell(self, cell: int) -> None:
        if cell == self._last_cell:
            return
        self._last_cell = cell
        self.setGridSize(QSize(cell + self.SPACING, cell + self.SPACING))
        self.setIconSize(QSize(cell, cell))
        # ScrollPerPixel leaves singleStep at 1px, which makes the scrollbar arrows
        # and arrow keys crawl; a quarter row is a usable step.
        self.verticalScrollBar().setSingleStep(max(1, (cell + self.SPACING) // 4))

    def _relayout(self) -> None:
        self._apply_cell(self.cell_for_width(self.viewport().width()))

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._relayout()

    def _row_step(self) -> int:
        """Pixels one row of thumbnails occupies."""
        return max(1, self.gridSize().height())

    def wheelEvent(self, event) -> None:
        bar = self.verticalScrollBar()
        pixel = event.pixelDelta()
        if not pixel.isNull() and pixel.y() != 0:
            # Trackpads already deliver continuous deltas; easing them adds lag.
            self._scroll_anim.stop()
            bar.setValue(bar.value() - pixel.y())
            event.accept()
            return

        notches = event.angleDelta().y() / 120.0
        if not notches:
            super().wheelEvent(event)
            return

        # Accumulate onto the in-flight target so a fast spin covers the whole
        # distance instead of restarting from wherever the easing had reached.
        running = self._scroll_anim.state() == QPropertyAnimation.State.Running
        base = self._scroll_target if running else bar.value()
        target = int(round(base - notches * self.WHEEL_ROWS_PER_NOTCH * self._row_step()))
        self._scroll_target = max(bar.minimum(), min(bar.maximum(), target))

        self._scroll_anim.stop()
        self._scroll_anim.setStartValue(bar.value())
        self._scroll_anim.setEndValue(self._scroll_target)
        self._scroll_anim.start()
        event.accept()


class FileBrowser(QWidget):
    """
    Asset management panel for loading and selecting images.
    """

    file_selected = pyqtSignal(str)
    library_requested = pyqtSignal(bool)  # reveal the library (arg: ask for a folder if unset)
    browse_requested = pyqtSignal(str)  # reveal this folder in the tree
    sort_changed = pyqtSignal()  # the folder tree follows the sheet's sort

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

        toolbar_row = OverflowBar(height=btn_height, spacing=4)

        self.library_btn = QToolButton()
        self.library_btn.setIcon(qta.icon("fa5s.book-open", color=THEME.text_primary))
        self.library_btn.setToolTip("Library — browse the folder your scans live in")

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

        self.half_frame_btn = QToolButton()
        self.half_frame_btn.setCheckable(True)
        self.half_frame_btn.setIcon(qta.icon("mdi.view-split-vertical", color=THEME.text_primary))
        self.half_frame_btn.setToolTip("Half Frame — split each scan into two frames, edited and measured separately")
        self.half_frame_btn.setChecked(bool(self.session.repo.get_global_setting("half_frame_mode", False)))
        self._update_half_frame_style(self.half_frame_btn.isChecked())

        self.half_frame_adjust_btn = QToolButton()
        self.half_frame_adjust_btn.setIcon(qta.icon("mdi.tune-variant", color=THEME.text_primary))
        self.half_frame_adjust_btn.setToolTip("Adjust Half Frame split — reposition the crop rectangle and split line for the current scan")
        self.half_frame_adjust_btn.clicked.connect(self._on_half_frame_adjust)

        self.apply_btn = QToolButton()
        self.apply_btn.setIcon(qta.icon("fa5s.clone", color=THEME.text_primary))
        self.apply_btn.setToolTip("Apply settings from the current frame to selected frames or the whole roll")
        self.apply_btn.clicked.connect(self._open_apply_dialog)

        # Sheet filter dropdown
        self.sheet_btn = QToolButton()
        self.sheet_btn.setToolTip("Sheet — filter the contact sheet by triage mark")
        self.sheet_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        sheet_menu = QMenu(self.sheet_btn)
        self._sheet_group = QActionGroup(self)
        self._sheet_group.setExclusive(True)
        self.act_sheet_all = sheet_menu.addAction("All frames")
        self.act_sheet_keepers = sheet_menu.addAction("Keepers only")
        self.act_sheet_unrejected = sheet_menu.addAction("Hide rejected")
        for act in (self.act_sheet_all, self.act_sheet_keepers, self.act_sheet_unrejected):
            act.setCheckable(True)
            self._sheet_group.addAction(act)
        self.act_sheet_all.triggered.connect(lambda: self._apply_sheet_filter("all"))
        self.act_sheet_keepers.triggered.connect(lambda: self._apply_sheet_filter("keepers"))
        self.act_sheet_unrejected.triggered.connect(lambda: self._apply_sheet_filter("unrejected"))
        self.sheet_btn.setMenu(sheet_menu)

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
            self.library_btn,
            self.add_files_btn,
            self.add_folder_btn,
            self.unload_btn,
            self.hot_folder_btn,
            self.rgb_scan_btn,
            self.half_frame_btn,
            self.half_frame_adjust_btn,
            self.apply_btn,
            self.sheet_btn,
            self.sort_btn,
        ):
            btn.setIconSize(icon_size)
            btn.setFixedHeight(btn_height)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)

        # OverflowBar rather than a QHBoxLayout: a plain row made the whole session panel
        # unshrinkable below every button laid end to end, so each new tool widened it for good.
        for widget, label in (
            (self.library_btn, "Library"),
            (self.add_files_btn, "Add files"),
            (self.add_folder_btn, "Add folder"),
            (self.unload_btn, "Clear all"),
            (None, None),
            (self.hot_folder_btn, "Hot Folder"),
            (self.rgb_scan_btn, "RGB Scan"),
            (self.half_frame_btn, "Half Frame"),
            (self.half_frame_adjust_btn, "Adjust Half Frame"),
            (self.apply_btn, "Apply settings"),
            (None, None),
            (self.sheet_btn, "Sheet filter"),
            (self.sort_btn, "Sort"),
        ):
            if widget is None:
                toolbar_row.add_separator(self._create_separator())
            else:
                toolbar_row.add_button(widget, label)
        layout.addWidget(toolbar_row)

        saved_sort = self.session.repo.get_global_setting("file_sort_order") or "name"
        saved_desc = self.session.repo.get_global_setting("file_sort_descending") or False
        self._apply_sort_order(str(saved_sort), save=False)
        self._apply_sort_direction(bool(saved_desc), save=False)

        search_row = QHBoxLayout()
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Filter — name, film:portra, iso:>=400…")
        self.search_input.setToolTip(
            "Filter the sheet. A bare word matches the filename; terms are combined with AND.\n"
            "Fields: film, camera, lens, developer, format, scanning, roll, frame, iso, push,\n"
            "name, path, ext, date, keeper, rejected, edited.\n"
            'Examples:  film:portra iso:>=400   ·   camera:"Nikon F3" -rejected:   ·   date:>=2024-03'
        )
        self.search_input.setClearButtonEnabled(True)
        self.search_input.addAction(
            qta.icon("fa5s.search", color=THEME.text_secondary),
            QLineEdit.ActionPosition.LeadingPosition,
        )
        # The elastic item in this row: its natural minimum is what now keeps the panel from
        # narrowing further, and the other two here are fixed-width by design.
        self.search_input.setMinimumWidth(40)
        self.regex_btn = QPushButton(".*")
        self.regex_btn.setCheckable(True)
        self.regex_btn.setFixedWidth(36)
        self.regex_btn.setToolTip("Regex mode")

        # Same query text, wider net: the box above filters what is loaded, this runs
        # it against every library folder and opens what it finds.
        self.library_search_btn = QToolButton()
        self.library_search_btn.setIcon(qta.icon("mdi.folder-search-outline", color=THEME.text_primary))
        self.library_search_btn.setFixedSize(28, 28)
        self.library_search_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.library_search_btn.setToolTip("Search the whole library — runs this search across your library folders and loads the matches")

        search_row.addWidget(self.search_input)
        search_row.addWidget(self.regex_btn)
        search_row.addWidget(self.library_search_btn)

        # Thumbnail size lives here rather than the toolbar row above, to keep that row's
        # overflow menu to file actions.
        saved_cell = self.session.repo.get_global_setting("thumbnail_cell_size") or THUMB_CELL_DEFAULT
        self.thumb_size_slider = QSlider(Qt.Orientation.Horizontal)
        self.thumb_size_slider.setRange(THUMB_CELL_MIN, THUMB_CELL_MAX)
        self.thumb_size_slider.setValue(ThumbnailGridView._clamp_target(int(saved_cell)))
        self.thumb_size_slider.setFixedWidth(72)
        self.thumb_size_slider.setToolTip("Thumbnail size — smaller fits more columns in the panel")
        search_row.addWidget(self.thumb_size_slider)
        # Above both sections: one box that filters the frames and searches the library,
        # so it belongs to neither and stays reachable when either is folded away.
        layout.addLayout(search_row)

        self.tally_label = QLabel("")
        self.tally_label.setStyleSheet(f"color: {THEME.text_secondary}; font-size: 10px;")
        self.tally_label.setVisible(False)

        self.list_view = ThumbnailGridView(target_cell=self.thumb_size_slider.value())
        self.list_view.setModel(self.session.asset_model)
        self.list_view.setItemDelegate(_ThumbnailDelegate(self.list_view))
        self.list_view.setViewMode(QListView.ViewMode.IconMode)
        self.list_view.setResizeMode(QListView.ResizeMode.Adjust)
        self.list_view.setSelectionMode(QListView.SelectionMode.ExtendedSelection)
        self.list_view.setAlternatingRowColors(False)
        self.list_view.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)

        self.library_tree = LibraryTree(self.controller)
        self.library_section = self._make_section("Library", "fa5s.folder-open", self.library_tree, "library_section_expanded")

        frames = QWidget()
        frames_layout = QVBoxLayout(frames)
        frames_layout.setContentsMargins(0, 0, 0, 0)
        frames_layout.setSpacing(4)
        frames_layout.addWidget(self.tally_label)
        frames_layout.addWidget(self.list_view, 1)
        self.frames_section = self._make_section("Film Strip", "fa5s.film", frames, "frames_section_expanded")

        layout.addWidget(self.library_section)
        layout.addWidget(self.frames_section)
        # Absorbs the surplus when every section is collapsed — otherwise Qt spreads it
        # into the gaps between the rows above. Stretch 0 leaves an open section its share.
        layout.addStretch(0)
        self._rebalance_sections()

        # Applied after list_view exists — the filter prunes selection against the view.
        saved_sheet = self.session.repo.get_global_setting("sheet_filter") or "all"
        self._apply_sheet_filter(str(saved_sheet), save=False)

    def _make_section(self, title: str, icon: str, content: QWidget, setting: str) -> CollapsibleSection:
        saved = self.session.repo.get_global_setting(setting)
        section = CollapsibleSection(title, expanded=True if saved is None else bool(saved), icon=qta.icon(icon, color=THEME.text_muted))
        section.set_content(content)
        section.expanded_changed.connect(lambda on, key=setting: self._on_section_toggled(key, on))
        return section

    def _on_section_toggled(self, setting: str, expanded: bool) -> None:
        self.session.repo.save_global_setting(setting, expanded)
        self._rebalance_sections()

    def _rebalance_sections(self) -> None:
        """Expanded sections share the panel; a collapsed one keeps only its header.

        Stretch alone is not enough — a collapsed section would still be handed leftover
        space — so its height is pinned to the header until it opens again.
        """
        layout = self.layout()
        for section, share in ((self.library_section, _LIBRARY_SHARE), (self.frames_section, _FRAMES_SHARE)):
            expanded = section.toggle_button.isChecked()
            layout.setStretchFactor(section, share if expanded else 0)
            section.setMaximumHeight(_UNBOUNDED_HEIGHT if expanded else section.toggle_button.height())

    def _connect_signals(self) -> None:
        self.library_btn.clicked.connect(lambda: self.library_requested.emit(True))
        self.add_files_btn.clicked.connect(self.prompt_add_files)
        self.add_folder_btn.clicked.connect(self.prompt_add_folder)
        self.unload_btn.clicked.connect(self._on_unload_clicked)
        self.list_view.clicked.connect(self._on_item_clicked)
        self.list_view.doubleClicked.connect(self._on_item_double_clicked)
        self.list_view.customContextMenuRequested.connect(self._show_context_menu)
        self.list_view.selectionModel().selectionChanged.connect(self._on_selection_changed)
        self.hot_folder_btn.toggled.connect(self._on_hot_folder_toggled)
        self.rgb_scan_btn.toggled.connect(self._on_rgb_scan_toggled)
        self.half_frame_btn.toggled.connect(self._on_half_frame_toggled)
        self.session.state_changed.connect(self.sync_ui)
        self.session.files_changed.connect(self._on_files_changed)
        # Unloading the last frame leaves nothing to show — fall back to the library
        # rather than an empty panel. Never prompts: the user asked to unload, not to load.
        self.session.session_emptied.connect(lambda: self.library_requested.emit(False))
        self.search_input.textChanged.connect(lambda _: self.filter_timer.start())
        self.search_input.returnPressed.connect(self.search_library)
        self.regex_btn.toggled.connect(lambda _: self.filter_timer.start())
        self.library_search_btn.clicked.connect(self.search_library)
        # Relayout live while dragging, but only write the setting on release —
        # a drag crosses dozens of values and each save is a DB round-trip.
        self.thumb_size_slider.valueChanged.connect(self.list_view.set_target_cell)
        self.thumb_size_slider.sliderReleased.connect(self._save_thumb_size)

        # Delete key unloads the selected frame(s) — scoped to the thumbnail list so it
        # doesn't fire while typing in the filter box or editing elsewhere.
        del_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Delete), self.list_view)
        del_shortcut.setContext(Qt.ShortcutContext.WidgetShortcut)
        del_shortcut.activated.connect(self._on_delete_key)

    def load_folder(self, path: str, add_to_session: bool = False) -> None:
        """Load one folder's images, asking first.

        Listing a folder is free and belongs to the library tree; this is the expensive
        half, and an accepted prompt is the only thing that starts the hashing pass.
        """
        images, _ = folder_counts(path)
        if not images:
            self.controller.set_status(f"No images directly in “{_folder_label(path)}”", 4000)
            return
        if not self._confirm_load(images, _folder_label(path)):
            return
        self.controller.open_library_folder(path, add_to_session=add_to_session)

    def load_folders(self, paths, add_to_session: bool = False) -> None:
        """Load one folder, or a whole selection of them at once."""
        paths = [p for p in paths if p]
        if len(paths) == 1:
            self.load_folder(paths[0], add_to_session=add_to_session)
            return
        if not paths:
            return

        counted = [(p, folder_counts(p)[0]) for p in paths]
        loadable = [p for p, n in counted if n]
        total = sum(n for _, n in counted)
        if not loadable:
            self.controller.set_status("Those folders have no images in them", 4000)
            return
        if not self._confirm_load(total, f"{len(loadable)} folders"):
            return
        self.controller.open_library_folders(loadable, add_to_session=add_to_session)

    def _confirm_load(self, image_count: int, label: str) -> bool:
        if self.session.repo.get_global_setting("library_autoload_folders", False):
            return True

        n = image_count
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Question)
        box.setWindowTitle("Load roll")
        box.setText(f"Load {n} image{'s' if n != 1 else ''} from “{label}”?")
        box.setInformativeText("They are hashed and thumbnailed on load, which takes a moment on a large roll.")
        remember = QCheckBox("Always load without asking")
        box.setCheckBox(remember)
        load = box.addButton("Load", QMessageBox.ButtonRole.AcceptRole)
        box.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
        box.exec()
        if box.clickedButton() is not load:
            return False
        if remember.isChecked():
            self.session.repo.save_global_setting("library_autoload_folders", True)
        return True

    def search_library(self) -> None:
        """Run the box's query against the library folders instead of the loaded roll."""
        self.controller.request_library_search(self.search_input.text())

    def focus_search(self) -> None:
        self.search_input.setFocus()
        self.search_input.selectAll()

    def _save_thumb_size(self) -> None:
        self.session.repo.save_global_setting("thumbnail_cell_size", self.thumb_size_slider.value())

    def _on_files_changed(self) -> None:
        # A mark toggle can hide the active frame under a Sheet filter — pruning then
        # auto-advances the selection to the next visible frame (reject → move on).
        if self.session.asset_model.sheet_filter != "all":
            self._prune_selection_to_visible()
        self.sync_ui()

    def _on_unload_clicked(self) -> None:
        count = len(self.session.state.selected_indices)
        if count > 1:
            if confirm_unload(self, count=count):
                self.session.remove_selected_files()
        else:
            self._on_clear_all()

    def _on_clear_all(self) -> None:
        """Drop every loaded frame. The empty-space menu always means *all*, unlike
        the toolbar button, which clears the selection when one is active."""
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
        self._update_tally()

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
        self.sort_changed.emit()
        if save:
            self.session.repo.save_global_setting("file_sort_order", order)

    def _apply_sort_direction(self, descending: bool, save: bool = True) -> None:
        self.act_sort_asc.setChecked(not descending)
        self.act_sort_desc.setChecked(descending)
        self.session.asset_model.set_sort_descending(descending)
        self.sort_changed.emit()
        if save:
            self.session.repo.save_global_setting("file_sort_descending", descending)

    def sort_choice(self) -> tuple[str, bool]:
        return ("date" if self.act_sort_date.isChecked() else "name", self.act_sort_desc.isChecked())

    def _apply_sheet_filter(self, mode: str, save: bool = True) -> None:
        self.act_sheet_all.setChecked(mode == "all")
        self.act_sheet_keepers.setChecked(mode == "keepers")
        self.act_sheet_unrejected.setChecked(mode == "unrejected")
        icon_color = "white" if mode != "all" else THEME.text_primary
        self.sheet_btn.setIcon(qta.icon("fa5s.filter", color=icon_color))
        self.session.asset_model.set_sheet_filter(mode)
        if save:
            self.session.repo.save_global_setting("sheet_filter", mode)
        self._prune_selection_to_visible()
        self.sync_ui()

    def _update_tally(self) -> None:
        files = self.session.state.uploaded_files
        if not files:
            self.tally_label.setVisible(False)
            return
        keepers = sum(1 for f in files if f.get("keeper"))
        rejected = sum(1 for f in files if f.get("excluded"))
        n = len(files)
        text = f"{n} frame{'s' if n != 1 else ''}"
        if keepers:
            text += f" · {keepers} keeper{'s' if keepers != 1 else ''}"
        if rejected:
            text += f" · {rejected} rejected"
        self.tally_label.setText(text)
        self.tally_label.setVisible(True)

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

    def _update_half_frame_style(self, checked: bool) -> None:
        icon_color = "white" if checked else THEME.text_primary
        self.half_frame_btn.setIcon(qta.icon("mdi.view-split-vertical", color=icon_color))

    def _on_half_frame_toggled(self, checked: bool) -> None:
        self._update_half_frame_style(checked)
        if checked and self.session.state.uploaded_files:
            # Offer the rectangle editor on the current frame; the saved profile
            # is applied to every half-frame split from then on.
            current = self.session.state.current_file_path
            path = None
            if current:
                for f in self.session.state.uploaded_files:
                    if f.get("path") == current:
                        path = current
                        break
            if path is None:
                path = self.session.state.uploaded_files[0].get("path")
            if path:
                self.controller.open_half_frame_dialog(path)
        self.controller.set_half_frame_mode(checked)

    def _on_half_frame_adjust(self) -> None:
        """Re-open the half-frame rectangle editor on the current image."""
        current = self.session.state.current_file_path
        path = None
        if current:
            for f in self.session.state.uploaded_files:
                if f.get("path") == current:
                    path = current
                    break
        if path is None and self.session.state.uploaded_files:
            path = self.session.state.uploaded_files[0].get("path")
        if not path:
            return
        profile = self.controller.open_half_frame_dialog(path)
        if profile is not None:
            # Re-discover so the new profile takes effect immediately.
            files = self.session.state.uploaded_files
            self.controller.request_asset_discovery(
                [f["path"] for f in files if "path" in f],
                replace_existing=True,
                reselect_path=self.session.state.current_file_path,
            )

    def _scan_folder(self) -> None:
        if not self.session.state.uploaded_files:
            return

        last_file = self.session.state.uploaded_files[-1]
        folder_path = os.path.dirname(last_file["path"])
        existing = {f["path"] for f in self.session.state.uploaded_files}

        new_files = FolderWatchService.scan_for_new_files(folder_path, existing)
        if new_files:
            self.controller.request_asset_discovery(new_files)

    def prompt_add_files(self) -> None:
        """Public entry point: also driven by the canvas empty state."""
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

    def prompt_add_folder(self) -> None:
        """Public entry point: also driven by the canvas empty state."""
        start_dir = self.session.repo.get_global_setting("last_open_folder", "") or ""
        folder = QFileDialog.getExistingDirectory(self, "Select Folder", start_dir)
        if folder:
            self.session.repo.save_global_setting("last_open_folder", os.path.dirname(folder))
            self.open_or_browse(folder)

    def open_or_browse(self, folder: str) -> None:
        """Load a folder's images, or — when it only holds subfolders — reveal it in the
        library tree so its subfolders are one click away.

        Picking the one directory everything lives under used to dead-end on "no
        supported assets found", because the importer looks in that folder and not
        through it.
        """
        images, subfolders = folder_counts(folder)
        if images:
            self.controller.request_asset_discovery([folder], auto_open=True)
        elif subfolders:
            self.browse_requested.emit(folder)
            self.controller.set_status(f"No images directly in that folder — showing its {subfolders} subfolders", 5000)
        else:
            self.controller.set_status("That folder has no images in it", 4000)

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
            # Empty space carries the session-level tools, so they stay reachable
            # without travelling back to the toolbar at the top of the panel — and
            # are discoverable at all when the session is empty.
            self._build_session_menu().exec(self.list_view.viewport().mapToGlobal(pos))
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

        source_cfg = self.session.state.config
        show_bounds = _source_effective_bounds(source_cfg.process) is not None
        dlg = GranularSettingsDialog(
            self,
            source_cfg,
            self._source_name(),
            show_scope=True,
            show_bounds=show_bounds,
            sel_count=sel_targets,
            roll_count=roll_targets,
        )
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self.session.sync_selected_settings(dlg.selected(), dlg.bounds_flags(), dlg.scope())

    def _build_session_menu(self) -> QMenu:
        """Mirrors the panel toolbar's add/clear tools, for a right click on empty space."""
        icon_color = THEME.text_primary
        menu = QMenu(self)
        menu.addAction(qta.icon("fa5s.file-import", color=icon_color), "Add files…").triggered.connect(self.prompt_add_files)
        menu.addAction(qta.icon("fa5s.folder-plus", color=icon_color), "Add folder…").triggered.connect(self.prompt_add_folder)
        menu.addSeparator()
        clear = menu.addAction(qta.icon("fa5s.times-circle", color=icon_color), "Clear all")
        clear.triggered.connect(self._on_clear_all)
        clear.setEnabled(bool(self.session.state.uploaded_files))
        return menu

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
        act_paste.triggered.connect(lambda: open_paste_dialog(self, self.controller))
        act_paste.setEnabled(state.clipboard is not None)
        menu.addAction("Reset Settings").triggered.connect(self.session.reset_settings)
        menu.addSeparator()
        targets = [i for i in (state.selected_indices or [state.selected_file_idx]) if 0 <= i < len(state.uploaded_files)]
        n = len(targets)
        act_keep = menu.addAction(f"Keep {count_of(n, 'frame')}" if multi else "Keep")
        act_keep.setCheckable(True)
        act_keep.setChecked(bool(targets) and all(state.uploaded_files[i].get("keeper") for i in targets))
        act_keep.triggered.connect(lambda: self.session.toggle_mark("keeper"))
        act_reject = menu.addAction(f"Reject {count_of(n, 'frame')}" if multi else "Reject")
        act_reject.setCheckable(True)
        act_reject.setChecked(bool(targets) and all(state.uploaded_files[i].get("excluded") for i in targets))
        act_reject.triggered.connect(lambda: self.session.toggle_mark("excluded"))
        menu.addSeparator()
        menu.addAction("Apply settings…").triggered.connect(self._open_apply_dialog)
        if multi:
            menu.addSeparator()
            menu.addAction("Stitch selected frames").triggered.connect(lambda: self.controller.request_stitch_selected())
            self._add_hdr_merge_action(menu, state)
        else:
            menu.addSeparator()
            menu.addAction("Edit RGB Triplet…").triggered.connect(self._on_edit_triplet)
            active = state.uploaded_files[state.selected_file_idx] if 0 <= state.selected_file_idx < len(state.uploaded_files) else {}
            if active.get("stitch_paths"):
                menu.addAction("Unstitch").triggered.connect(lambda: self.controller.request_unstitch())
            if active.get("hdr_paths"):
                self._add_hdr_anchor_menu(menu, active)
                menu.addAction("Unmerge exposures").triggered.connect(lambda: self.controller.request_unmerge_hdr())
        menu.addSeparator()
        unload_label = "Unload Selected" if multi else "Unload"
        menu.addAction(unload_label).triggered.connect(self._on_remove_from_menu)
        return menu

    def _add_hdr_merge_action(self, menu, state) -> None:
        """Merging is for transparencies, so the action follows the film process.

        A colour negative holds about 5-6 stops between base and Dmax, and an ordinary
        black-and-white negative nearer 4 — both inside a single capture, so a bracket buys
        nothing. A transparency runs to 10-12, which is what the merge exists for.

        Hidden on C-41, disabled with a reason on B&W: reversal-processed monochrome
        (Scala, dr5, Fomapan R) *is* a transparency and does have the range, it is simply
        not wired yet, and a missing menu entry would leave nobody anything to ask about.
        """
        from negpy.features.process.models import ProcessMode

        idx = state.selected_file_idx
        assets = state.uploaded_files
        mode = str(self.controller.state.config.process.process_mode)
        if 0 <= idx < len(assets):
            mode = str(assets[idx].get("process_mode") or mode)
        if mode == ProcessMode.C41:
            return
        act = menu.addAction("Merge exposures (HDR)")
        if mode == ProcessMode.BW:
            act.setEnabled(False)
            act.setToolTip("Merging is for transparencies; black-and-white reversal film is not supported yet")
            return
        act.triggered.connect(lambda: self.controller.request_hdr_merge_selected())

    def _add_hdr_anchor_menu(self, menu, asset: dict) -> None:
        """ "Render exposure": which frame of the bracket the merged result opens at.

        The merge is computed in the *reference* frame's units — the longest exposure that
        does not clip — but that is a radiometric choice, and which exposure looks right is
        the photographer's. See `hdr.logic.output_scale`.

        Only exposures the render can actually sit at are offered (`anchor_choices`); a
        frame longer than the reference clamps back to it and would be an entry that
        provably cannot change the picture.
        """
        paths = hdr_frame_paths(asset)
        ratios = [float(r) for r in (asset.get("hdr_ratios") or ())]
        if len(paths) != len(ratios):
            return
        choices = anchor_choices(paths, ratios)
        if len(choices) < 2:
            return  # only the reference is reachable: every entry would be the same picture
        current = str(asset.get("hdr_anchor", "") or "")
        sub = menu.addMenu("Render exposure")
        auto = sub.addAction("Bracket middle (auto)")
        auto.setCheckable(True)
        auto.setChecked(not current)
        auto.triggered.connect(lambda: self.controller.set_hdr_anchor(""))
        sub.addSeparator()
        for path, stops in choices:
            label = f"{os.path.basename(path)}   {stops:+.1f} EV"
            if stops == 0.0:
                label += "  (as captured)"
            act = sub.addAction(label)
            act.setCheckable(True)
            act.setChecked(path == current)
            act.triggered.connect(lambda _=False, p=path: self.controller.set_hdr_anchor(p))

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
