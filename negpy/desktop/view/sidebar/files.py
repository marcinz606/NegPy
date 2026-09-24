import os
from typing import List, Optional

import qtawesome as qta
from PyQt6.QtCore import (
    Qt,
    QEasingCurve,
    QItemSelection,
    QItemSelectionModel,
    QModelIndex,
    QPersistentModelIndex,
    QPoint,
    QPropertyAnimation,
    QRect,
    QRectF,
    QSize,
    QTimer,
    pyqtSignal,
    pyqtSlot,
)
from PyQt6.QtGui import QActionGroup, QColor, QKeySequence, QPainter, QPainterPath, QPen, QShortcut
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListView,
    QMenu,
    QSlider,
    QSplitter,
    QStyle,
    QStyledItemDelegate,
    QStyleOptionViewItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from negpy.kernel.system.text import count_of
from negpy.desktop.controller import AppController
from negpy.desktop.session import AppState, composite_kind
from negpy.desktop.view.confirm import (
    prompt_delete_scene,
    confirm_reset_frames,
    confirm_undiptych,
    confirm_unfork_edit,
    confirm_unload,
    warn_invalid_roll_name,
)
from negpy.desktop.view.keyboard_shortcuts import _reset_roll, _reset_selected
from negpy.features.hdr.logic import anchor_choices
from negpy.features.hdr.models import hdr_frame_paths
from negpy.desktop.view.widgets.elided_label import ElidedLabel
from negpy.desktop.view.widgets.sort_button import SortButton
from negpy.desktop.view.widgets.overflow_bar import OverflowBar
from negpy.desktop.view.shortcut_registry import label_with_shortcut
from negpy.desktop.view.styles.templates import (
    ICON_BUTTON_WIDTH,
    TOOLBAR_BUTTON_HEIGHT,
    TOOLBAR_ICON_SIZE,
    tool_toggle,
    wrap_tooltip,
)
from negpy.desktop.view.styles.theme import THEME, scene_color
from negpy.desktop.view.widgets.granular_settings_dialog import open_apply_dialog, open_paste_dialog, open_sync_bounds_dialog
from negpy.desktop.view.widgets.rgb_triplet_dialog import open_triplet_dialog
from negpy.desktop.view.widgets.roll_settings_dialog import RollSettingsDialog
from negpy.services.assets import rolls
from negpy.services.assets.gear import GearProfiles
from negpy.services.assets.gear_match import GearMatch, folder_name_for_active_context, match_gear_for_folder
from negpy.services.assets.presets import is_valid_preset_name
from negpy.infrastructure.filesystem.watcher import FolderWatchService
from negpy.infrastructure.loaders.helpers import get_supported_raw_wildcards
from negpy.desktop.view.sidebar.library_tree import LibraryTree
from negpy.desktop.view.widgets.collapsible import CollapsibleSection, make_section
from negpy.desktop.view.widgets.file_dialogs import last_open_folder
from negpy.services.assets.library import folder_counts, folder_label
from negpy.services.assets.thumbnails import asset_thumbnail_key


_UNBOUNDED_HEIGHT = 16777215  # QWIDGETSIZE_MAX — Qt's "no maximum"
# With both sections open the panel starts 20/80: the tree is for finding a roll, glanced
# at occasionally, while the sheet is where the work happens and wants the room. The
# splitter's handle can move this default any time.
_LIBRARY_SHARE, _FRAMES_SHARE = 1, 4


class _ThumbnailDelegate(QStyledItemDelegate):
    """Contact-sheet rendering: scales each cached ~120px thumbnail into its cell and
    draws a subtle 1px border hugging the image outline (no cell box), 2px in the frame's
    scene color while Show Scenes is on. The selected image is shown full-brightness
    with the accent ring just outside that border, so both show at once; the others are
    dimmed. A dirty active file gets an accent line along the image's bottom edge.
    Triage marks are small bottom-right badges: check = keeper, cross + heavy dim =
    rejected; the bottom-left badge says the frame was built from several files
    (stitch, HDR merge, RGB triplet, half-frame split). Top-left holds the
    decode-failure badge, else a small dot saying the bitmap shown predates a settings
    change (a bulk apply reaches the file before a render reaches its thumbnail)."""

    _MARGIN = 5  # room for the selection ring outside the picture
    _SELECTION_OUTSET = 4  # the ring's outer edge, outside the picture edge
    _RADIUS = 4  # = button border-radius (modern_dark.qss)
    _MARK = QColor(183, 28, 28, 150)  # THEME.accent_primary at ~60% alpha
    # Neutral, not the triage red: red already means "you marked this" and "this failed".
    # What a frame is built from is a fact about the asset, not a state the user set.
    _COMPOSITE_CHIP = QColor(20, 20, 20, 190)
    _COMPOSITE_RING = QColor(255, 255, 255, 90)
    _COMPOSITE_GLYPH = QColor(255, 255, 255, 235)
    _DIRTY_PX = 2
    _STALE_DOT_RADIUS = 4
    _ACTIVITY_INTERVAL_MS = 40
    _ACTIVITY_STEP = 0.035

    def __init__(self, parent=None, state: Optional[AppState] = None) -> None:
        super().__init__(parent)
        self._state = state
        self._show_scenes = False
        self._placeholder_icon = qta.icon("fa5s.image", color=THEME.text_muted)
        self._activity_icon = qta.icon("fa5s.image", color=THEME.text_secondary)
        self._activity_key = ""
        self._activity_index = QPersistentModelIndex()
        self._activity_phase = 0.0
        self._activity_timer = QTimer(self)
        self._activity_timer.setInterval(self._ACTIVITY_INTERVAL_MS)
        self._activity_timer.timeout.connect(self._advance_activity)

    def set_show_scenes(self, on: bool) -> None:
        self._show_scenes = on

    @pyqtSlot(str)
    def set_activity(self, key: str) -> None:
        if key == self._activity_key:
            return
        previous = self._activity_index
        self._activity_key = key
        self._activity_index = self._find_activity_index()
        self._activity_phase = 0.0
        if key:
            self._activity_timer.start()
        else:
            self._activity_timer.stop()
        self._repaint_index(previous)
        self._repaint_view()

    def _advance_activity(self) -> None:
        self._activity_phase = (self._activity_phase + self._ACTIVITY_STEP) % 1.0
        self._repaint_view()

    def _repaint_view(self) -> None:
        if not self._index_matches_activity(self._activity_index):
            self._activity_index = self._find_activity_index()
        self._repaint_index(self._activity_index)

    def _find_activity_index(self) -> QPersistentModelIndex:
        view = self.parent()
        if not isinstance(view, QListView) or not self._activity_key:
            return QPersistentModelIndex()
        model = view.model()
        if model is None:
            return QPersistentModelIndex()
        for row in range(model.rowCount()):
            index = model.index(row, 0)
            file_info = index.data(Qt.ItemDataRole.UserRole) or {}
            if file_info.get("hash") and asset_thumbnail_key(file_info) == self._activity_key:
                return QPersistentModelIndex(index)
        return QPersistentModelIndex()

    def _index_matches_activity(self, index: QPersistentModelIndex) -> bool:
        if not index.isValid() or not self._activity_key:
            return False
        file_info = index.data(Qt.ItemDataRole.UserRole) or {}
        return bool(file_info.get("hash") and asset_thumbnail_key(file_info) == self._activity_key)

    def _repaint_index(self, index: QPersistentModelIndex) -> None:
        view = self.parent()
        if isinstance(view, QListView) and index.isValid():
            view.viewport().update(view.visualRect(QModelIndex(index)))

    def _is_dirty(self, file_info: dict) -> bool:
        """Only the active file can carry unsaved edits; every other frame is on disk."""
        state = self._state
        return bool(state and state.is_dirty and state.current_file_path and file_info.get("path") == state.current_file_path)

    def _is_stale_thumbnail(self, file_info: dict) -> bool:
        """True while the cached bitmap predates a settings write a render hasn't caught up to."""
        state = self._state
        if not state or not file_info.get("hash"):
            return False
        return asset_thumbnail_key(file_info) in state.stale_thumbnails

    def _draw_stale_dot(self, painter: QPainter, img_rect: QRect) -> None:
        r = self._STALE_DOT_RADIUS
        cx, cy = img_rect.left() + r + 4, img_rect.top() + r + 4
        painter.setPen(QPen(QColor(0, 0, 0, 140), 1))
        painter.setBrush(QColor(THEME.warn_amber))
        painter.drawEllipse(QRect(cx - r, cy - r, 2 * r, 2 * r))

    @staticmethod
    def _fit_rect(area: QRect, source_size: QSize) -> QRect:
        size = source_size.scaled(area.size(), Qt.AspectRatioMode.KeepAspectRatio)
        x = area.x() + (area.width() - size.width()) // 2
        y = area.y() + (area.height() - size.height()) // 2
        return QRect(x, y, size.width(), size.height())

    def picture_rect(self, cell: QRect, index: QModelIndex) -> QRect:
        """Where paint() puts the picture inside *cell*: the thumbnail fitted to the area
        within the margin, or that whole area while it has none."""
        area = cell.adjusted(self._MARGIN, self._MARGIN, -self._MARGIN, -self._MARGIN)
        icon = index.data(Qt.ItemDataRole.DecorationRole)
        base = icon.pixmap(QSize(4096, 4096)) if icon is not None and not icon.isNull() else None
        if base is None or base.isNull():
            return area
        return self._fit_rect(area, base.size().scaled(area.size(), Qt.AspectRatioMode.KeepAspectRatio))

    def sizeHint(self, option: QStyleOptionViewItem, index: QModelIndex) -> QSize:
        view = self.parent()
        if isinstance(view, QListView) and view.iconSize().isValid():
            return view.iconSize()
        return super().sizeHint(option, index)

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
        cx, cy = img_rect.left() + r + 4, img_rect.top() + r + 4
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(THEME.error))
        painter.drawEllipse(QRect(cx - r, cy - r, 2 * r, 2 * r))
        painter.setPen(QPen(QColor(THEME.text_on_accent), 2))
        painter.drawLine(cx, cy - 4, cx, cy + 1)
        painter.drawPoint(cx, cy + 4)

    def _draw_outline(self, painter: QPainter, img_rect: QRect, file_info: dict, selected: bool, hover: bool) -> None:
        """The picture's edge line, 2px in its scene's color while Show Scenes is on, and
        the selection ring outside it."""
        scene = file_info.get("scene") if self._show_scenes else None
        painter.setBrush(Qt.BrushStyle.NoBrush)
        if scene:
            painter.setPen(QPen(QColor(scene_color(scene[0])), 2))
            painter.drawRoundedRect(QRectF(img_rect).adjusted(1, 1, -2, -2), self._RADIUS, self._RADIUS)
        else:
            painter.setPen(self._border_pen(hover))
            painter.drawRoundedRect(img_rect.adjusted(0, 0, -1, -1), self._RADIUS, self._RADIUS)
        if selected:
            out = self._SELECTION_OUTSET
            painter.setPen(QPen(QColor(THEME.accent_primary), 2))
            painter.drawRoundedRect(QRectF(img_rect).adjusted(1 - out, 1 - out, out - 2, out - 2), self._RADIUS + out, self._RADIUS + out)

    def _draw_composite_badge(self, painter: QPainter, img_rect: QRect, kind: str, half: int) -> None:
        """Bottom-left mark: this frame was assembled from more than one file.

        One glyph per kind, so a merge is told from a stitch without opening the menu.
        The chip carries a faint ring because a flat dark disc vanishes on a dense frame."""
        r = 9
        cx, cy = img_rect.left() + r + 4, img_rect.bottom() - r - 4
        painter.setPen(QPen(self._COMPOSITE_RING, 1))
        painter.setBrush(self._COMPOSITE_CHIP)
        painter.drawEllipse(QRect(cx - r, cy - r, 2 * r, 2 * r))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(self._COMPOSITE_GLYPH, 1.5))
        if kind == "stitch":  # two overlapping panes — a divided box reads as the half glyph
            painter.drawRect(QRect(cx - 6, cy - 5, 8, 7))
            front = QRect(cx - 2, cy - 2, 8, 7)
            painter.fillRect(front, self._COMPOSITE_CHIP)
            painter.drawRect(front)
        elif kind == "hdr":  # a bracket: stacked exposures
            for dy, width in ((-3, 11), (0, 8), (3, 5)):
                painter.drawLine(cx - 5, cy + dy, cx - 5 + width, cy + dy)
        elif kind == "rgb":  # the three narrowband exposures
            painter.setPen(Qt.PenStyle.NoPen)
            for dx, color in ((-4, THEME.channel_red), (0, THEME.channel_green), (4, THEME.channel_blue)):
                painter.setBrush(QColor(color))
                painter.drawEllipse(QRect(cx + dx - 2, cy - 2, 4, 4))
            painter.setBrush(Qt.BrushStyle.NoBrush)
        elif kind == "half":  # a split frame, this asset's own half filled
            painter.drawRect(QRect(cx - 6, cy - 4, 12, 8))
            painter.fillRect(QRect(cx - 5 if half == 1 else cx + 1, cy - 3, 5, 7), self._COMPOSITE_GLYPH)
        elif kind == "diptych":  # the same split frame with both halves filled
            painter.drawRect(QRect(cx - 6, cy - 4, 12, 8))
            for left in (cx - 5, cx + 1):
                painter.fillRect(QRect(left, cy - 3, 5, 7), self._COMPOSITE_GLYPH)

    @staticmethod
    def _border_pen(hover: bool) -> QPen:
        if hover:
            return QPen(QColor(THEME.text_muted), 1)
        return QPen(QColor(THEME.border_color), 1)

    def _paint_placeholder(self, painter: QPainter, option: QStyleOptionViewItem, file_info: dict, failed: bool, kind: str) -> None:
        """A cell whose thumbnail hasn't decoded yet (or failed to). Still carries the
        selection/hover border and the triage marks, so a multi-selection or a rejected frame
        stays visible while thumbnails are still loading in the background instead of looking
        selective or broken."""
        selected = bool(option.state & QStyle.StateFlag.State_Selected)
        hover = bool(option.state & QStyle.StateFlag.State_MouseOver)
        rejected = bool(file_info.get("excluded"))
        keeper = bool(file_info.get("keeper"))
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        area = option.rect.adjusted(self._MARGIN, self._MARGIN, -self._MARGIN, -self._MARGIN)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(20, 20, 20))
        painter.drawRoundedRect(area, self._RADIUS, self._RADIUS)
        self._draw_outline(painter, area, file_info, selected, hover)
        if rejected:
            self._draw_mark_badge(painter, area, check=False)
        elif keeper:
            self._draw_mark_badge(painter, area, check=True)
        if failed:
            self._draw_failed_badge(painter, area)
        if kind:
            self._draw_composite_badge(painter, area, kind, int(file_info.get("half") or 0))
        painter.restore()

    def paint(self, painter: QPainter, option: QStyleOptionViewItem, index: QModelIndex) -> None:
        file_info = index.data(Qt.ItemDataRole.UserRole) or {}
        failed = bool(file_info.get("decode_failed"))
        kind = composite_kind(file_info)

        icon = index.data(Qt.ItemDataRole.DecorationRole)
        if icon is None or icon.isNull():
            painter.save()
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            area = option.rect.adjusted(self._MARGIN, self._MARGIN, -self._MARGIN, -self._MARGIN)
            img_rect = area
            selected = bool(option.state & QStyle.StateFlag.State_Selected)
            hover = bool(option.state & QStyle.StateFlag.State_MouseOver)
            rejected = bool(file_info.get("excluded"))
            keeper = bool(file_info.get("keeper"))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(THEME.bg_header))
            painter.drawRoundedRect(img_rect, self._RADIUS, self._RADIUS)
            self._draw_outline(painter, img_rect, file_info, selected, hover)
            glyph_side = min(32, min(img_rect.width(), img_rect.height()) // 3)
            glyph_rect = QRect(0, 0, glyph_side, glyph_side)
            glyph_rect.moveCenter(img_rect.center())
            active = bool(self._activity_key and file_info.get("hash") and asset_thumbnail_key(file_info) == self._activity_key)
            if active:
                painter.save()
                painter.setOpacity(0.2)
                self._placeholder_icon.paint(painter, glyph_rect)
                painter.restore()
                painter.save()
                reveal = QRect(glyph_rect)
                reveal.setWidth(max(1, round(glyph_rect.width() * self._activity_phase)))
                painter.setClipRect(reveal)
                self._activity_icon.paint(painter, glyph_rect)
                painter.restore()
            else:
                self._placeholder_icon.paint(painter, glyph_rect)
            if rejected:
                self._draw_mark_badge(painter, img_rect, check=False)
            elif keeper:
                self._draw_mark_badge(painter, img_rect, check=True)
            if failed:
                self._draw_failed_badge(painter, img_rect)
            if kind:
                self._draw_composite_badge(painter, img_rect, kind, int(file_info.get("half") or 0))
            painter.restore()
            return
        base = icon.pixmap(QSize(4096, 4096))  # largest available pixmap (~120px)
        if base.isNull():
            self._paint_placeholder(painter, option, file_info, failed, kind)
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
        img_rect = self._fit_rect(area, scaled.size())

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
        if kind:
            self._draw_composite_badge(painter, img_rect, kind, int(file_info.get("half") or 0))
        if not failed and self._is_stale_thumbnail(file_info):
            self._draw_stale_dot(painter, img_rect)
        painter.setClipping(False)

        self._draw_outline(painter, img_rect, file_info, selected, hover)

        if self._is_dirty(file_info):
            # Over the frame line, so it reads as the accent and not a blend with the border.
            painter.setClipPath(clip)
            painter.fillRect(
                QRect(img_rect.left(), img_rect.bottom() - self._DIRTY_PX + 1, img_rect.width(), self._DIRTY_PX),
                QColor(THEME.accent_primary),
            )
            painter.setClipping(False)

        if failed:
            self._draw_failed_badge(painter, img_rect)

        painter.restore()


# Thumbnail size preference (px), as set by the filmstrip's size slider. The default fills
# one column of the session sidebar at a typical narrow width, where a single full-width
# frame is the most legible use of it. Dropping toward THUMB_CELL_MIN fits a second column
# at that width. A narrower panel overflows the toolbar and the grid keeps one column.
#
# The maximum is the default: the slider only shrinks frames. Anything larger holds a
# widened sidebar at one column, and since cells are square a 3:2 frame leaves a wide band
# of empty space above and below. Two columns are both denser and larger in practice, so
# there is nothing above the default worth offering.
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
    # Scene sort: the gap between two scenes' blocks, and each block's tinted band.
    SCENE_GAP = THEME.space_lg
    SCENE_BAND_ALPHA = 0.18
    SCENE_BAND_RADIUS = 6
    # One notch scrolls one row of thumbnails. Qt's default, three "lines" a notch, advanced
    # several frames at a time in a single-column panel.
    WHEEL_ROWS_PER_NOTCH = 1.0
    SCROLL_ANIM_MS = 160

    def __init__(self, parent=None, target_cell: int = THUMB_CELL_DEFAULT):
        super().__init__(parent)
        self._last_cell = -1
        self._target_cell = self._clamp_target(target_cell)
        self._pending_click_row: Optional[int] = None
        self._pending_click_modifiers = Qt.KeyboardModifier.NoModifier
        self._pending_mode: Optional[str] = None  # "range" | "ctrl"
        self._range_anchor_row: Optional[int] = None
        self._ctrl_target: set[QPersistentModelIndex] = set()
        self._pre_press_selection: set[QPersistentModelIndex] = set()
        self._placing_scenes = False
        self._scene_bands: list[tuple[int, int, int, QRect]] = []  # (ordinal, first row, last row, band)
        # Reserve the vertical scrollbar permanently so the viewport width is stable. Otherwise
        # scaling toggles the scrollbar, which changes the width, flips the column count back
        # and flickers.
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setSpacing(self.SPACING)
        # Per-pixel is what makes a partial-row offset representable at all. Under ScrollPerItem
        # the view snaps to whole rows and no easing is possible.
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
        # ScrollPerPixel leaves singleStep at 1px, which makes the scrollbar arrows and arrow
        # keys crawl. A quarter row is a usable step.
        self.verticalScrollBar().setSingleStep(max(1, (cell + self.SPACING) // 4))

    def _relayout(self) -> None:
        self._apply_cell(self.cell_for_width(self.viewport().width()))

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._relayout()

    def updateGeometries(self) -> None:
        super().updateGeometries()
        if self._placing_scenes:
            return
        self._placing_scenes = True
        try:
            self._place_scene_runs()
        finally:
            self._placing_scenes = False

    def _place_scene_runs(self) -> None:
        """Scene sort: each scene's frames as their own block from a new row, SCENE_GAP apart,
        frames in no scene last. Qt re-flows every cell on each layout, so this runs after
        each one; every other order keeps Qt's own flow."""
        model = self.model()
        runs = model.scene_runs() if hasattr(model, "scene_runs") else []
        self._scene_bands = []
        if not runs:
            return
        grid, cell = self.gridSize(), self.iconSize()
        cols = self.columns_for_width(self.viewport().width())
        inset = (grid.width() - cell.width()) // 2  # Qt centers a cell in its grid square
        pad = self.SCENE_GAP // 2 - 1
        y = self.SCENE_GAP // 2
        for ordinal, first, last in runs:
            count = last - first + 1
            for n in range(count):
                position = QPoint(inset + (n % cols) * grid.width(), y + (n // cols) * grid.height())
                self.setPositionForIndex(position, model.index(first + n, 0))
            lines = (count - 1) // cols + 1
            if ordinal is not None:
                height = (lines - 1) * grid.height() + cell.height() + 2 * pad
                self._scene_bands.append((ordinal, first, last, QRect(0, y - pad, min(count, cols) * grid.width(), height)))
            y += lines * grid.height() + self.SCENE_GAP

    def paintEvent(self, event) -> None:
        if self._scene_bands:
            self._paint_scene_bands()
        super().paintEvent(event)

    def _paint_scene_bands(self) -> None:
        """Each scene's band, cut away under every picture: a dimmed or letterboxed
        thumbnail would otherwise show the tint through it."""
        model, delegate = self.model(), self.itemDelegate()
        painter = QPainter(self.viewport())
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setPen(Qt.PenStyle.NoPen)
        for ordinal, first, last, rect in self._scene_bands:
            band = QPainterPath()
            band.addRoundedRect(QRectF(rect.translated(0, -self.verticalOffset())), self.SCENE_BAND_RADIUS, self.SCENE_BAND_RADIUS)
            if isinstance(delegate, _ThumbnailDelegate):
                pictures = QPainterPath()
                for row in range(first, last + 1):
                    index = model.index(row, 0)
                    picture = QRectF(delegate.picture_rect(self.visualRect(index), index))
                    pictures.addRoundedRect(picture, delegate._RADIUS, delegate._RADIUS)
                band = band.subtracted(pictures)
            color = QColor(scene_color(ordinal))
            color.setAlphaF(self.SCENE_BAND_ALPHA)
            painter.fillPath(band, color)
        painter.end()

    def _begin_click_selection(self, pre_press_current: QModelIndex) -> None:
        """Decides the gesture's mode and range anchor exactly once, from the modifiers and
        current index at press; nothing later in the gesture reads either again.
        `pre_press_current` is passed in, rather than read here, because `super().mousePressEvent()`
        (already run by this point) moves Qt's own current index to the just-pressed row."""
        model = self.model()
        sel_model = self.selectionModel()
        row = self._pending_click_row
        if model is None or sel_model is None or row is None:
            return
        index = model.index(row, 0)
        shift = bool(self._pending_click_modifiers & Qt.KeyboardModifier.ShiftModifier)
        ctrl = bool(self._pending_click_modifiers & Qt.KeyboardModifier.ControlModifier)
        self._range_anchor_row = pre_press_current.row() if (shift and pre_press_current.isValid()) else row

        if ctrl:
            target = set(self._pre_press_selection)
            if shift:
                # Ctrl+Shift extends the existing selection with the anchor-to-row range,
                # additively, rather than replacing it the way a plain Shift-range does.
                top, bottom = sorted((self._range_anchor_row, row))
                for r in range(top, bottom + 1):
                    target.add(QPersistentModelIndex(model.index(r, 0)))
            else:
                pindex = QPersistentModelIndex(index)
                if pindex in target:
                    target.discard(pindex)
                else:
                    target.add(pindex)
            self._pending_mode = "ctrl"
            self._ctrl_target = target
            self._apply_ctrl_target()
            sel_model.setCurrentIndex(index, QItemSelectionModel.SelectionFlag.NoUpdate)
            return

        self._pending_mode = "range"
        self._extend_click_selection(row)

    def _apply_ctrl_target(self) -> None:
        """Reapplies the gesture's whole target selection in one call, not just the toggled
        index — Qt's own release-time recompute can `ClearAndSelect` the clicked item alone
        before this runs, and adjusting just that index against a cleared selection wouldn't
        restore what it cleared."""
        sel_model = self.selectionModel()
        if sel_model is None:
            return
        combined = QItemSelection()
        for pindex in self._ctrl_target:
            if pindex.isValid():
                idx = QModelIndex(pindex)
                combined.merge(QItemSelection(idx, idx), QItemSelectionModel.SelectionFlag.Select)
        sel_model.select(combined, QItemSelectionModel.SelectionFlag.ClearAndSelect)

    def _extend_click_selection(self, current_row: int) -> None:
        """Selects the range from the gesture's anchor to `current_row`. Called again for every
        move/release while the button stays held, reading only the row under the cursor, so a
        click-drag extends the range live."""
        model = self.model()
        sel_model = self.selectionModel()
        if model is None or sel_model is None or self._range_anchor_row is None:
            return
        top, bottom = sorted((self._range_anchor_row, current_row))
        sel_model.select(QItemSelection(model.index(top, 0), model.index(bottom, 0)), QItemSelectionModel.SelectionFlag.ClearAndSelect)
        sel_model.setCurrentIndex(model.index(current_row, 0), QItemSelectionModel.SelectionFlag.NoUpdate)

    def _row_near(self, pos) -> Optional[int]:
        """The row at `pos`, or the nearest end row when a drag has gone past the first/last
        cell — a drag over that dead space would otherwise fall back to Qt's own geometric,
        modifier-reading selection update for that move."""
        index = self.indexAt(pos)
        if index.isValid():
            return index.row()
        model = self.model()
        if model is None or model.rowCount() == 0:
            return None
        if pos.y() >= self.visualRect(model.index(model.rowCount() - 1, 0)).bottom():
            return model.rowCount() - 1
        if pos.y() <= self.visualRect(model.index(0, 0)).top():
            return 0
        return None

    def _apply_pending_row(self, row: int) -> None:
        if self._pending_mode == "ctrl":
            self._apply_ctrl_target()
        else:
            self._extend_click_selection(row)

    # press/move/release below reassert the gesture's decision after Qt's own handling: it
    # recomputes its own selection command at each stage, reading modifiers fresh every time.
    def mousePressEvent(self, event) -> None:
        self._end_click_gesture()
        # Captured before super(), which runs Qt's own press handling and would otherwise
        # already have moved currentIndex() and ctrl-toggled the selection by the time this reads it.
        sel_model = self.selectionModel()
        pre_press_current = sel_model.currentIndex() if sel_model is not None else QModelIndex()
        if event.button() == Qt.MouseButton.LeftButton:
            index = self.indexAt(event.position().toPoint())
            if index.isValid():
                self._pending_click_row = index.row()
                self._pending_click_modifiers = event.modifiers()
                if event.modifiers() & Qt.KeyboardModifier.ControlModifier and sel_model is not None:
                    self._pre_press_selection = {QPersistentModelIndex(i) for i in sel_model.selectedIndexes()}
        super().mousePressEvent(event)
        if self._pending_click_row is not None:
            self._begin_click_selection(pre_press_current)

    def mouseMoveEvent(self, event) -> None:
        # Row read before super(): its own autoscroll-to-current can jump the viewport to fit
        # the cell under the cursor, and hitting this same screen point again afterward would
        # then resolve to whatever row the scroll left there instead of the one dragged to.
        row = (
            self._row_near(event.position().toPoint())
            if (self._pending_click_row is not None and event.buttons() & Qt.MouseButton.LeftButton)
            else None
        )
        super().mouseMoveEvent(event)
        if row is not None:
            self._apply_pending_row(row)

    def mouseReleaseEvent(self, event) -> None:
        row = (
            self._row_near(event.position().toPoint())
            if (self._pending_click_row is not None and event.button() == Qt.MouseButton.LeftButton)
            else None
        )
        super().mouseReleaseEvent(event)
        if row is not None:
            self._apply_pending_row(row)
        self._end_click_gesture()

    def focusOutEvent(self, event) -> None:
        super().focusOutEvent(event)
        # Insurance against a press with no matching release (focus stolen mid-drag, e.g. by
        # a modal dialog) leaving a gesture's state to apply to some unrelated later click.
        self._end_click_gesture()

    def _end_click_gesture(self) -> None:
        self._pending_click_row = None
        self._pending_mode = None
        self._range_anchor_row = None
        self._ctrl_target = set()
        self._pre_press_selection = set()

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

        # Accumulate onto the in-flight target, so a fast spin covers the whole distance instead
        # of restarting from wherever the easing had reached.
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
    library_requested = pyqtSignal(bool)  # reveal the library (arg: import a first roll if unset)

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
        # session.state.selected_indices as of the start of the debounce currently pending —
        # lets sync_ui tell a stale echo of the in-flight click from a genuinely newer command.
        self._debounce_baseline_selected: List[int] = []

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

        icon_size = QSize(TOOLBAR_ICON_SIZE, TOOLBAR_ICON_SIZE)
        btn_height = TOOLBAR_BUTTON_HEIGHT

        # No top-level toolbar: every action lives in the row of the section it acts on --
        # Library's own +/refresh corner, or film_strip_toolbar next to the loaded frames.
        self.film_strip_toolbar = OverflowBar(height=btn_height, spacing=4)

        # One button for both: Add Files and Add Folder are two pickers for the same job
        # (put pictures in this session), not two different actions worth their own icons.
        self.add_btn = QToolButton()
        self.add_btn.setIcon(qta.icon("fa5s.file-import", color=THEME.text_primary))
        self.add_btn.setToolTip("Add pictures or a folder to this session")
        self.add_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        add_menu = QMenu(self.add_btn)
        add_menu.addAction("Add Files…").triggered.connect(self.prompt_add_files)
        add_menu.addAction("Add Folder…").triggered.connect(self.prompt_add_folder)
        self.add_btn.setMenu(add_menu)
        self.unload_btn = QToolButton()
        self.unload_btn.setIcon(qta.icon("fa5s.times-circle", color=THEME.text_primary))
        self.unload_btn.setToolTip("Unload…")

        self.hot_folder_btn = QToolButton()
        self.hot_folder_btn.setCheckable(True)
        self.hot_folder_btn.setIcon(qta.icon("fa5s.fire", color=THEME.text_primary))
        self.hot_folder_btn.setToolTip("Hot Folder — automatically load new images from the current folder")
        self._update_hot_folder_style(False)

        self.apply_btn = QToolButton()
        self.apply_btn.setIcon(qta.icon("fa5s.clone", color=THEME.text_primary))
        self.apply_btn.setToolTip("Apply settings from the current frame to selected frames or the whole roll")
        self.apply_btn.clicked.connect(self._open_apply_dialog)

        self.roll_settings_btn = QToolButton()
        self.roll_settings_btn.setIcon(qta.icon("fa5s.tags", color=THEME.text_primary))
        self.roll_settings_btn.setToolTip(
            wrap_tooltip("Roll Settings — tag gear, capture and process metadata across the current frame, a selection or the whole roll")
        )
        self.roll_settings_btn.clicked.connect(self._open_roll_settings_dialog)

        self.save_roll_btn = QToolButton()
        self.save_roll_btn.setIcon(qta.icon("fa5s.search", color=THEME.text_primary))
        self.save_roll_btn.setToolTip(wrap_tooltip("Save these frames as a roll — a named, reopenable group, not tied to a folder"))
        self.save_roll_btn.clicked.connect(self._on_save_roll_clicked)
        self.update_thumbnails_btn = QToolButton()
        self.update_thumbnails_btn.setIcon(qta.icon("fa5s.sync-alt", color=THEME.text_primary))
        self.update_thumbnails_btn.setToolTip("Update Thumbnails — re-render every stale thumbnail in the roll")
        self.update_thumbnails_btn.clicked.connect(self._on_update_thumbnails_clicked)

        self.scenes_btn = QToolButton()
        self.scenes_btn.setCheckable(True)
        self.scenes_btn.setToolTip(wrap_tooltip("Show Scenes — frame each picture in its scene's color"))
        self.scenes_btn.toggled.connect(self._apply_show_scenes)

        # Sheet filter dropdown
        self.sheet_btn = QToolButton()
        self.sheet_btn.setToolTip("Sheet — filter the contact sheet by triage mark")
        self.sheet_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        sheet_menu = QMenu(self.sheet_btn)
        self._sheet_group = QActionGroup(self)
        self._sheet_group.setExclusive(True)
        self.act_sheet_all = sheet_menu.addAction("All Frames")
        self.act_sheet_keepers = sheet_menu.addAction("Keepers Only")
        self.act_sheet_unrejected = sheet_menu.addAction("Hide Rejected")
        for act in (self.act_sheet_all, self.act_sheet_keepers, self.act_sheet_unrejected):
            act.setCheckable(True)
            self._sheet_group.addAction(act)
        self.act_sheet_all.triggered.connect(lambda: self._apply_sheet_filter("all"))
        self.act_sheet_keepers.triggered.connect(lambda: self._apply_sheet_filter("keepers"))
        self.act_sheet_unrejected.triggered.connect(lambda: self._apply_sheet_filter("unrejected"))
        self.sheet_btn.setMenu(sheet_menu)

        # The frames' own order; the Library's roll list has a Sort of its own.
        self.sort_btn = SortButton((("name", "Name"), ("date", "Date"), ("scene", "Scene")), "Sort the frames in the Film Strip")
        self.act_sort_name, self.act_sort_date, self.act_sort_scene = (self.sort_btn.order_action(k) for k in ("name", "date", "scene"))
        self.act_sort_asc, self.act_sort_desc = self.sort_btn.ascending_action, self.sort_btn.descending_action
        self.act_sort_scene.setVisible(False)
        self.sort_btn.order_selected.connect(self._apply_sort_order)
        self.sort_btn.direction_selected.connect(self._apply_sort_direction)

        for btn in (
            self.add_btn,
            self.unload_btn,
            self.hot_folder_btn,
            self.apply_btn,
            self.roll_settings_btn,
            self.save_roll_btn,
            self.update_thumbnails_btn,
            self.scenes_btn,
            self.sort_btn,
            self.sheet_btn,
        ):
            btn.setIconSize(icon_size)
            btn.setFixedHeight(btn_height)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)

        for widget, label in (
            (self.save_roll_btn, "Save as Roll…"),
            (None, None),
            (self.add_btn, "Add"),
            (None, None),
            (self.hot_folder_btn, "Hot Folder"),
            (None, None),
            (self.apply_btn, "Apply settings"),
            (self.roll_settings_btn, "Roll Settings"),
            (self.update_thumbnails_btn, "Update thumbnails"),
            (None, None),
            (self.unload_btn, "Unload…"),
            (self.scenes_btn, "Show Scenes"),
            (self.sort_btn, "Sort"),
            (self.sheet_btn, "Sheet filter"),
        ):
            if widget is None:
                self.film_strip_toolbar.add_separator(self._create_separator())
            else:
                self.film_strip_toolbar.add_button(widget, label)

        saved_sort = self.session.repo.get_global_setting("file_sort_order") or "name"
        saved_desc = self.session.repo.get_global_setting("file_sort_descending") or False

        search_row = QHBoxLayout()
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Filter — name, film:portra, iso:>=400…")
        self.search_input.setToolTip(
            "Filter the sheet. A bare word matches the filename; terms are combined with AND.\n"
            "Fields: film, camera, lens, developer, format, scanning, roll, frame, iso, push,\n"
            "shot, place, scene, name, path, ext, date, keeper, rejected, edited.\n"
            'Examples:  film:portra iso:>=400   ·   camera:"Nikon F3" -rejected:   ·   shot:>=1998-07   ·   place:tokyo'
        )
        self.search_input.setClearButtonEnabled(True)
        self.search_input.addAction(
            qta.icon("fa5s.search", color=THEME.text_secondary),
            QLineEdit.ActionPosition.LeadingPosition,
        )
        # The elastic item in this row. Its natural minimum is what keeps the panel from narrowing
        # further, and the other two here are fixed-width by design.
        self.search_input.setMinimumWidth(40)
        self.regex_btn = tool_toggle("", ".*", "Regex mode")
        self.regex_btn.setFixedWidth(ICON_BUTTON_WIDTH)

        # Same query text, wider net: the box above filters what is loaded, and this runs it
        # against every library folder and opens what it finds.
        self.library_search_btn = QToolButton()
        self.library_search_btn.setIcon(qta.icon("mdi.folder-search-outline", color=THEME.text_primary))
        self.library_search_btn.setFixedSize(28, 28)
        self.library_search_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.library_search_btn.setToolTip("Search the whole library — runs this search across your library folders and loads the matches")

        # Opt-in (Preferences); hidden until then. Mutually exclusive with regex/the
        # structured query language -- ranks this session's frames by meaning instead.
        self.semantic_btn = tool_toggle(
            "mdi.image-search-outline", "", "Search by meaning — describe what you're looking for instead of using field:value terms"
        )
        self.semantic_btn.setFixedWidth(ICON_BUTTON_WIDTH)
        self.semantic_btn.setVisible(False)

        search_row.addWidget(self.search_input)
        search_row.addWidget(self.regex_btn)
        search_row.addWidget(self.semantic_btn)
        search_row.addWidget(self.library_search_btn)

        # Built here (Film Strip's thumbnail grid needs a starting value below) but added to
        # the Film Strip section's own tally row, since it only ever affects that grid.
        saved_cell = self.session.repo.get_global_setting("thumbnail_cell_size") or THUMB_CELL_DEFAULT
        self.thumb_size_slider = QSlider(Qt.Orientation.Horizontal)
        self.thumb_size_slider.setRange(THUMB_CELL_MIN, THUMB_CELL_MAX)
        self.thumb_size_slider.setValue(ThumbnailGridView._clamp_target(int(saved_cell)))
        self.thumb_size_slider.setFixedWidth(72)
        self.thumb_size_slider.setToolTip("Thumbnail size — smaller fits more columns in the panel")
        # Above both sections: one box that filters the frames and searches the library, so it
        # belongs to neither and stays reachable when either is folded away.
        layout.addLayout(search_row)

        self.tally_label = ElidedLabel("")
        self.tally_label.setStyleSheet(f"color: {THEME.text_secondary}; font-size: {THEME.font_size_small}px;")
        self.tally_label.setVisible(False)

        self.list_view = ThumbnailGridView(target_cell=self.thumb_size_slider.value())
        self.list_view.setModel(self.session.asset_model)
        self._thumbnail_delegate = _ThumbnailDelegate(self.list_view, state=self.session.state)
        self.list_view.setItemDelegate(self._thumbnail_delegate)
        self.list_view.setViewMode(QListView.ViewMode.IconMode)
        self.list_view.setResizeMode(QListView.ResizeMode.Adjust)
        self.list_view.setSelectionMode(QListView.SelectionMode.ExtendedSelection)
        self.list_view.setAlternatingRowColors(False)
        self.list_view.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._apply_sort_order(str(saved_sort), save=False)
        self._apply_sort_direction(bool(saved_desc), save=False)

        # Takes the strip's place when a filter hides every frame: a blank panel under a
        # full tally reads as a load failure.
        self.empty_label = QLabel("")
        self.empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.empty_label.setWordWrap(True)
        self.empty_label.setTextFormat(Qt.TextFormat.RichText)
        self.empty_label.setOpenExternalLinks(False)
        self.empty_label.setStyleSheet(f"color: {THEME.text_secondary}; font-size: {THEME.font_size_small}px;")
        self.empty_label.setVisible(False)
        self.empty_label.linkActivated.connect(lambda _: self._clear_frame_filters())

        self.library_tree = LibraryTree(self.controller)
        self.library_section = self._make_section("Library", "library", "fa5s.folder-open", self.library_tree)

        frames = QWidget()
        frames_layout = QVBoxLayout(frames)
        frames_layout.setContentsMargins(0, 0, 0, 0)
        frames_layout.setSpacing(4)
        frames_layout.addWidget(self.film_strip_toolbar)
        tally_row = QHBoxLayout()
        tally_row.addWidget(self.tally_label, 1)
        tally_row.addWidget(self.thumb_size_slider)
        frames_layout.addLayout(tally_row)
        frames_layout.addWidget(self.list_view, 1)
        frames_layout.addWidget(self.empty_label, 1)
        self.frames_section = self._make_section("Film Strip", "frames", "fa5s.film", frames)

        # Clearing the strip is how you start a roll you will build entirely by drag-drop,
        # so it lives on the section header rather than its own toolbar button -- the
        # header has room a wrapping toolbar row does not.
        frames_menu = QMenu(self.frames_section)
        frames_menu.addAction("New Roll…").triggered.connect(self._on_clear_all)
        frames_menu.addAction(label_with_shortcut("Reset Roll to Defaults…", "reset_roll")).triggered.connect(self._on_reset_roll)
        self.frames_section.set_actions_menu(
            frames_menu,
            "New Roll clears the film strip so you can drag in a fresh batch of frames. "
            "Reset Roll to Defaults undoes every loaded frame's edit at once.",
        )

        # A splitter, like the right panel's Analysis/Tabs one, so the boundary can be
        # dragged; expanded sections still share it by _LIBRARY_SHARE/_FRAMES_SHARE.
        self.sections_splitter = QSplitter(Qt.Orientation.Vertical)
        self.sections_splitter.addWidget(self.library_section)
        self.sections_splitter.addWidget(self.frames_section)
        self.sections_splitter.setCollapsible(0, False)
        self.sections_splitter.setCollapsible(1, False)

        saved_sizes = self.session.repo.get_global_setting("session_sections_splitter_sizes")
        if isinstance(saved_sizes, list) and len(saved_sizes) == 2:
            self.sections_splitter.setSizes([int(s) for s in saved_sizes])
        else:
            self.sections_splitter.setSizes([120, 480])
        self.sections_splitter.splitterMoved.connect(self._on_sections_splitter_moved)
        self._section_sizes = self.sections_splitter.sizes()

        for index, section in enumerate((self.library_section, self.frames_section)):
            section.expanded_changed.connect(lambda expanded, i=index: self._on_section_toggled(i, expanded))
            self._on_section_toggled(index, section.toggle_button.isChecked())

        # Absorbs the surplus when both sections are collapsed, or Qt spreads it above the
        # splitter instead of below it (the splitter's own stretch factor, set to 0 in that
        # case by _on_section_toggled, leaves this the only claimant on the leftover space).
        layout.addWidget(self.sections_splitter, 1)
        layout.addStretch(0)

        # Applied after list_view exists: the filter prunes the selection against the view.
        saved_sheet = self.session.repo.get_global_setting("sheet_filter") or "all"
        self._apply_sheet_filter(str(saved_sheet), save=False)
        show_scenes = bool(self.session.repo.get_global_setting("show_scenes") or False)
        self.scenes_btn.blockSignals(True)
        self.scenes_btn.setChecked(show_scenes)
        self.scenes_btn.blockSignals(False)
        self._apply_show_scenes(show_scenes, save=False)

    def _make_section(self, title: str, key: str, icon: str, content: QWidget) -> CollapsibleSection:
        return make_section(self.session.repo, title, key, content, icon, default_expanded=True)

    def _on_sections_splitter_moved(self, *_args) -> None:
        self._section_sizes = self.sections_splitter.sizes()
        self.session.repo.save_global_setting("session_sections_splitter_sizes", self._section_sizes)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if hasattr(self, "sections_splitter"):
            self._rebalance_splitter_sizes()

    def _apply_section_constraints(self, index: int, expanded: bool) -> None:
        """A collapsed pane is fixed to exactly its header height, so nothing -- a drag,
        a resize -- can hand it more or less than that; an expanded pane is freed back to
        its natural range."""
        section = (self.library_section, self.frames_section)[index]
        share = (_LIBRARY_SHARE, _FRAMES_SHARE)[index]
        if expanded:
            section.setMinimumHeight(0)
            section.setMaximumHeight(_UNBOUNDED_HEIGHT)
        else:
            section.setFixedHeight(section.toggle_button.height())
        self.sections_splitter.setStretchFactor(index, share if expanded else 0)

    def _on_section_toggled(self, index: int, expanded: bool) -> None:
        """Pin a collapsed section's pane to its header and hand its space to the other
        pane, restoring the last size when it reopens — same pattern as the right panel's
        Analysis/Tabs splitter, generalized to two panes that can each collapse."""
        sections = (self.library_section, self.frames_section)
        other = 1 - index
        header = sections[index].toggle_button.height()
        sizes = self.sections_splitter.sizes()
        total = sum(sizes) or self.sections_splitter.height()

        if not expanded:
            self._section_sizes[index] = sizes[index]
        self._apply_section_constraints(index, expanded)

        # A stretch factor sizes the widget, not its content, so a splitter whose panes are
        # both pinned small still takes the layout's leftover space. Its own maximum height
        # is what keeps that space out.
        other_header = sections[other].toggle_button.height()
        if expanded or sections[other].toggle_button.isChecked():
            self.sections_splitter.setMaximumHeight(_UNBOUNDED_HEIGHT)
        else:
            self.sections_splitter.setMaximumHeight(header + self.sections_splitter.handleWidth() + other_header)
        if total <= 0:
            return

        want = min(max(self._section_sizes[index], header), max(header, total - header)) if expanded else header
        other_want = max(other_header, total - want) if sections[other].toggle_button.isChecked() else other_header
        new_sizes = [0, 0]
        new_sizes[index] = want
        new_sizes[other] = other_want
        self.sections_splitter.setSizes(new_sizes)
        self._section_sizes = self.sections_splitter.sizes()

    def _rebalance_splitter_sizes(self) -> None:
        """Reassert the split on a window resize: a collapsed pane stays pinned to its
        header; expanded panes keep their current size ratio, scaled to the new total.
        Needed because a QSplitter's own resize handling redistributes new space by
        stretch factor, and with two independently-collapsible panes that factor is 0
        for both whenever both happen to be collapsed.
        """
        sections = (self.library_section, self.frames_section)
        total = self.sections_splitter.height() or sum(self.sections_splitter.sizes())
        if total <= 0:
            return
        current = self.sections_splitter.sizes()
        headers = [s.toggle_button.height() for s in sections]
        expanded = [s.toggle_button.isChecked() for s in sections]
        collapsed_total = sum(h for h, e in zip(headers, expanded) if not e)
        remaining = max(0, total - collapsed_total)
        expanded_weight_total = sum(w for w, e in zip(current, expanded) if e)
        sizes = []
        for h, e, w in zip(headers, expanded, current):
            if e and expanded_weight_total:
                sizes.append(round(remaining * w / expanded_weight_total))
            elif e:
                sizes.append(remaining)
            else:
                sizes.append(h)
        self.sections_splitter.setSizes(sizes)

    def _connect_signals(self) -> None:
        self.library_tree.folder_roll_created.connect(self._maybe_suggest_gear)
        self.unload_btn.clicked.connect(self._on_unload_clicked)
        self.list_view.clicked.connect(self._on_item_clicked)
        self.list_view.doubleClicked.connect(self._on_item_double_clicked)
        self.list_view.customContextMenuRequested.connect(self._show_context_menu)
        self.list_view.selectionModel().selectionChanged.connect(self._on_selection_changed)
        self.hot_folder_btn.toggled.connect(self._on_hot_folder_toggled)
        self.controller.thumbnail_refresh_state_changed.connect(self._on_thumbnail_refresh_state_changed)
        self.session.state_changed.connect(self.sync_ui)
        self.session.files_changed.connect(self._on_files_changed)
        self.controller.first_scene_created.connect(lambda: self._apply_sort_order("scene"))
        self.controller.thumbnail_activity_changed.connect(self._thumbnail_delegate.set_activity)
        # Unloading the last frame leaves nothing to show, so fall back to the library rather
        # than an empty panel. Never prompts: the user asked to unload, not to load.
        self.session.session_emptied.connect(lambda: self.library_requested.emit(False))
        self.search_input.textChanged.connect(lambda _: self.filter_timer.start())
        self.search_input.returnPressed.connect(self.search_library)
        self.regex_btn.toggled.connect(lambda _: self.filter_timer.start())
        self.semantic_btn.toggled.connect(self._on_semantic_toggled)
        self.library_search_btn.clicked.connect(self.search_library)
        # Relayout live while dragging, but write the setting only on release: a drag crosses
        # dozens of values and each save is a DB round-trip.
        self.thumb_size_slider.valueChanged.connect(self.list_view.set_target_cell)
        self.thumb_size_slider.sliderReleased.connect(self._save_thumb_size)

        # Delete unloads the selected frames, scoped to the thumbnail list so it does not fire
        # while typing in the filter box or editing elsewhere.
        del_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Delete), self.list_view)
        del_shortcut.setContext(Qt.ShortcutContext.WidgetShortcut)
        del_shortcut.activated.connect(self._on_delete_key)

    def search_library(self) -> None:
        """Run the box's query against the library folders instead of the loaded roll."""
        # A keystroke just before Enter leaves the live-filter debounce pending, and it
        # would fire _apply_filter over whatever the hand-off loaded, putting back the
        # in-session query the hand-off clears. One action, one rebuild.
        self.filter_timer.stop()
        if self.semantic_btn.isChecked():
            self.controller.request_library_semantic_search(self.search_input.text())
        else:
            self.controller.request_library_search(self.search_input.text())

    def focus_search(self) -> None:
        self.search_input.setFocus()
        self.search_input.selectAll()

    def _save_thumb_size(self) -> None:
        self.session.repo.save_global_setting("thumbnail_cell_size", self.thumb_size_slider.value())

    def _on_files_changed(self) -> None:
        # A mark toggle can hide the active frame under a Sheet filter. Pruning then auto-advances
        # the selection to the next visible frame.
        if self.session.asset_model.sheet_filter != "all":
            self._prune_selection_to_visible()
        self._sync_sort_menu()
        self.sync_ui()

    def _on_unload_clicked(self) -> None:
        """Same as the context menu's Unload…: always targets the selection -- at least
        the active frame, ordinarily -- never the whole roll. Opening a different roll
        already replaces the film strip, so wiping everything is not something this
        button needs to reach for; Clear All for the rare "go back to empty" case lives
        in the empty-space context menu instead."""
        self._on_remove_from_menu()

    def _on_clear_all(self) -> None:
        """Drop every loaded frame, from the empty-space context menu."""
        if confirm_unload(self, clear_all=True):
            self.session.clear_files()

    def _on_reset_roll(self) -> None:
        """Reset every visible frame back to its own defaults, from the Film Strip
        header's ⋮ menu."""
        count = len(self.session.asset_model.visible_actual_indices_ordered())
        if count and confirm_reset_frames(self, count, roll=True):
            self.controller.request_reset_roll()

    def _on_save_roll_clicked(self) -> None:
        """Save whatever the Film Strip currently holds as a named, reopenable roll --
        a library search's results, a hand-picked selection, or a folder roll's extras."""
        name, ok = QInputDialog.getText(self, "Save as Roll", "Name:")
        name = name.strip()
        if not ok or not name:
            return
        if not is_valid_preset_name(name):
            warn_invalid_roll_name(self, "Save as Roll")
            return
        if self.controller.create_roll_from_session(name):
            self.library_tree.reload()
            self._update_tally()

    def _update_unload_button(self) -> None:
        multi = len(self.session.state.selected_indices) > 1
        self.unload_btn.setToolTip("Unload Selected…" if multi else "Unload…")

    def sync_ui(self) -> None:
        """Updates list selection to match session state."""
        model = self.session.asset_model
        selection_model = self.list_view.selectionModel()
        self.semantic_btn.setVisible(self.session.state.semantic_search_enabled)
        if not self.session.state.semantic_search_enabled and self.semantic_btn.isChecked():
            self.semantic_btn.setChecked(False)  # reverts to the plain filter via _on_semantic_toggled
        self.library_tree.sync_ui()
        self._update_unload_button()
        self._update_tally()
        self._update_empty_state()

        current_actual = {
            model.display_to_actual(idx.row()) for idx in selection_model.selectedIndexes() if model.display_to_actual(idx.row()) >= 0
        }
        target_actual = set(self.session.state.selected_indices)

        # Repaint for dirty underline
        self.list_view.viewport().update()

        if current_actual == target_actual:
            return
        if self.selection_timer.isActive():
            if target_actual == set(self._debounce_baseline_selected):
                # A stale echo of the still-pending click, not a newer command — let the
                # debounce commit it normally rather than clobbering the live click.
                return
            # A newer command changed state while the click was pending; it wins, so the
            # click's own eventual commit must not overwrite it with a now-stale value.
            self.selection_timer.stop()

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
        if not self.selection_timer.isActive():
            self._debounce_baseline_selected = list(self.session.state.selected_indices)
        self.selection_timer.start()

    def _commit_selection(self) -> None:
        """Sends current UI selection to the session after debounce."""
        model = self.session.asset_model
        actual_indices = [a for idx in self.list_view.selectionModel().selectedIndexes() if (a := model.display_to_actual(idx.row())) >= 0]
        if set(actual_indices) != set(self.session.state.selected_indices):
            self.session.update_selection(actual_indices)

    def _apply_filter(self) -> None:
        text = self.search_input.text().strip()
        if self.semantic_btn.isChecked():
            embedding = self.controller.embed_search_query(text)
            self.session.asset_model.set_semantic_query(embedding)
            self._set_search_error(bool(text) and embedding is None)
            self._prune_selection_to_visible()
            self.sync_ui()
            return

        if self.session.asset_model.semantic_query_active:
            self.session.asset_model.set_semantic_query(None)
        regex = self.regex_btn.isChecked()
        ok = self.session.asset_model.set_filter(text, regex)
        self._set_search_error(not ok)
        if ok:
            self._prune_selection_to_visible()
            self.sync_ui()

    def _on_semantic_toggled(self, checked: bool) -> None:
        self.regex_btn.setEnabled(not checked)
        self.search_input.setPlaceholderText("Describe what you're looking for…" if checked else "Filter — name, film:portra, iso:>=400…")
        self.filter_timer.start()

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
        # AssetListModel's own reindex remaps every persistent index (Qt's selection and
        # current-index among them), so the view's selection needs no separate resync here.
        self.session.asset_model.set_sort_order(order)
        self._sync_sort_menu()
        if save:
            self.session.repo.save_global_setting("file_sort_order", order)

    def _apply_sort_direction(self, descending: bool, save: bool = True) -> None:
        self.act_sort_asc.setChecked(not descending)
        self.act_sort_desc.setChecked(descending)
        self.session.asset_model.set_sort_descending(descending)
        if save:
            self.session.repo.save_global_setting("file_sort_descending", descending)

    def _sync_sort_menu(self) -> None:
        """Scene shows once the loaded roll has a scene. The ticks show the order the frames
        are in, which is Name while a Scene choice waits for a roll with scenes."""
        model = self.session.asset_model
        self.act_sort_scene.setVisible(model.has_scenes)
        order = model.effective_sort_order
        self.act_sort_name.setChecked(order == "name")
        self.act_sort_date.setChecked(order == "date")
        self.act_sort_scene.setChecked(order == "scene")

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

    def _active_frame_filters(self) -> list:
        """Names of the filters currently hiding frames, in the order they are applied."""
        model = self.session.asset_model
        names = []
        if model.filter_text:
            names.append("search")
        if model.sheet_filter == "keepers":
            names.append("Keepers")
        elif model.sheet_filter == "unrejected":
            names.append("Hide Rejected")
        return names

    def _update_tally(self) -> None:
        files = self.session.state.uploaded_files
        if not files:
            self.tally_label.setVisible(False)
            return
        keepers = sum(1 for f in files if f.get("keeper"))
        rejected = sum(1 for f in files if f.get("excluded"))
        n = len(files)
        # The strip shows the model, the tally counts the session, so a filter that hides
        # every frame reads as an empty panel under a full count unless it is named here.
        visible = self.session.asset_model.rowCount()
        text = f"{visible} of {n} frames" if visible != n else count_of(n, "frame")
        for name in self._active_frame_filters():
            text += f" · {name} filter"
        if keepers:
            text += f" · {count_of(keepers, 'keeper')}"
        if rejected:
            text += f" · {rejected} rejected"
        roll_name = self._active_roll_name()
        # The prefix slot holds the roll's name. Frames that are not one roll get named
        # as what they are instead: an edit here reaches the roll each frame came from,
        # which a strip that looks identical either way gives no sign of.
        text = f"{roll_name or 'No roll'} — {text}"
        self.tally_label.setText(text)
        self.tally_label.setToolTip(
            ""
            if roll_name
            else wrap_tooltip(
                "These frames are not one roll. An edit changes the photo itself, so it also shows in the roll the frame came from."
            )
        )
        self.tally_label.setVisible(True)

    def _active_roll_name(self) -> str:
        """The roll the Film Strip's frames came from, if any -- shown ahead of the
        tally so it stays visible without opening Library to check."""
        roll_id = self.session.state.active_roll_id
        if not roll_id:
            return ""
        entry = rolls.roll_for_id(self.session.repo, roll_id)
        return entry.get("name", "") if entry else ""

    def _update_empty_state(self) -> None:
        """Swap the strip for a message when a filter leaves it with nothing to show."""
        files = self.session.state.uploaded_files
        filters = self._active_frame_filters()
        empty = bool(files) and self.session.asset_model.rowCount() == 0 and bool(filters)
        if empty:
            named = " and ".join(filters)
            self.empty_label.setText(
                f'No frames match the {named} filter.<br><a href="#clear" style="color: {THEME.accent_primary};">Show all frames</a>'
            )
        self.empty_label.setVisible(empty)
        self.list_view.setVisible(not empty)

    def _clear_frame_filters(self) -> None:
        """Clear both filters from the empty state, so the strip cannot be a dead end."""
        self.search_input.clear()
        # Ahead of the debounce the clear would otherwise start, so one click is one rebuild.
        self.filter_timer.stop()
        self._apply_filter()
        self._apply_sheet_filter("all")

    def _on_hot_folder_toggled(self, checked: bool) -> None:
        self._update_hot_folder_style(checked)
        if checked:
            self.scan_timer.start()
        else:
            self.scan_timer.stop()

    def _apply_show_scenes(self, on: bool, save: bool = True) -> None:
        self.scenes_btn.setIcon(qta.icon("fa5s.layer-group", color="white" if on else THEME.text_primary))
        self._thumbnail_delegate.set_show_scenes(on)
        self.list_view.viewport().update()
        if save:
            self.session.repo.save_global_setting("show_scenes", on)

    def _update_hot_folder_style(self, checked: bool) -> None:
        icon_color = "white" if checked else THEME.text_primary
        self.hot_folder_btn.setIcon(qta.icon("fa5s.fire", color=icon_color))

    def _on_adjust_half_frame_split(self, path: str, base_hash: str) -> None:
        """Open the rectangle editor for one file, defaulting Apply to just that
        frame — for the odd frame the roll-wide split still gets wrong."""
        result = self.controller.open_half_frame_dialog(path, base_hash, initial_scope="current")
        if result is not None:
            self.controller.reload_after_half_frame_change()

    def _on_reset_half_frame_split(self, base_hash: str) -> None:
        self.controller.clear_half_frame_override(base_hash)
        self.controller.reload_after_half_frame_change()

    def _scan_folder(self) -> None:
        if not self.session.state.uploaded_files:
            return

        last_file = self.session.state.uploaded_files[-1]
        folder_path = os.path.dirname(last_file["path"])
        # A duplicate of a loaded frame never reaches uploaded_files, so counting only
        # what is loaded would re-offer it every poll: hash it, turn it away, repeat.
        existing = {f["path"] for f in self.session.state.uploaded_files} | self.session.state.duplicate_paths

        new_files = FolderWatchService.scan_for_new_files(folder_path, existing)
        if new_files:
            self.controller.request_asset_discovery(new_files, hot_folder=True)

    def prompt_add_files(self) -> None:
        """Public entry point: also driven by the canvas empty state."""
        wildcards = get_supported_raw_wildcards()
        start_dir = last_open_folder(self.session.repo)
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Images",
            start_dir,
            f"Supported Images ({wildcards})",
        )
        if files:
            self.session.repo.save_global_setting("last_open_folder", os.path.dirname(files[0]))
            self.controller.request_asset_discovery(files, auto_open=True, announce_rgb=True)

    def prompt_add_folder(self) -> None:
        """Public entry point: also driven by the canvas empty state."""
        start_dir = last_open_folder(self.session.repo)
        folder = QFileDialog.getExistingDirectory(self, "Select Folder", start_dir)
        if folder:
            self.session.repo.save_global_setting("last_open_folder", os.path.dirname(folder))
            self.open_or_browse(folder)

    def open_or_browse(self, folder: str) -> None:
        """Load a folder's images into the session, or point at Library's own import
        when it only holds subfolders: the importer looks in the folder, not through it."""
        images, subfolders = folder_counts(folder)
        if images:
            self.controller.request_asset_discovery([folder], auto_open=True, announce_rgb=True)
        elif subfolders:
            self.controller.set_status(
                f"No images directly in “{folder_label(folder)}” — use Library's Import Subfolders as Rolls for its "
                f"{count_of(subfolders, 'subfolder')}",
                5000,
            )
        else:
            self.controller.set_status("That folder has no images in it", 4000)

    def _activate_file(self, index) -> None:
        """Load a thumbnail into the main viewer, skipping a redundant reload of the
        already-active frame."""
        actual = self.session.asset_model.display_to_actual(index.row())
        if actual >= 0 and actual != self.session.state.selected_file_idx:
            self.session.select_file(actual)

    def _on_item_clicked(self, index) -> None:
        # `clicked` fires from inside Qt's own mouseReleaseEvent, before ThumbnailGridView's
        # reapply runs — force it now so a plain click is told apart from a Shift/Ctrl one
        # (left to the selectionChanged handler) by the gesture's final, decided selection.
        self.list_view._apply_pending_row(index.row())
        selected = self.list_view.selectionModel().selectedIndexes()
        if len(selected) == 1 and selected[0].row() == index.row():
            self._activate_file(index)

    def _on_item_double_clicked(self, index) -> None:
        self._activate_file(index)

    def _show_context_menu(self, pos) -> None:
        index = self.list_view.indexAt(pos)
        if not index.isValid():
            # Empty space carries the session-level tools, so they stay reachable without travelling
            # back to the toolbar at the top of the panel, and are discoverable at all when the
            # session is empty.
            self._build_session_menu().exec(self.list_view.viewport().mapToGlobal(pos))
            return
        actual = self.session.asset_model.display_to_actual(index.row())
        if actual < 0:
            return

        # Right-clicking outside the current selection re-selects just that file. Within a
        # multi-selection, keep the selection and make the clicked file active.
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
        applied = open_apply_dialog(self, self.session)
        if applied and applied[1] == "roll":
            self.controller.record_roll_apply(applied[0])

    def _open_roll_settings_dialog(self) -> None:
        """The tag-icon button: always opens, and silently pre-fills a gear match too
        (only when Gear is not already set) -- the automatic suggestion at import time
        is one moment among several this same guess is useful in."""
        detected = self._detect_gear_suggestion(self._folder_name_for_gear_suggestion())
        dlg = self._build_roll_settings_dialog()
        if dlg is None:
            return
        if detected is not None:
            dlg.apply_detected_gear(camera_id=detected.camera_id, film_stock_id=detected.film_stock_id)
        self._exec_roll_settings_dialog(dlg)

    def _build_roll_settings_dialog(self) -> Optional[RollSettingsDialog]:
        state = self.session.state
        src = state.selected_file_idx
        if src == -1:
            return None
        visible = self.session.asset_model.visible_actual_indices()
        sel_targets = len([i for i in set(state.selected_indices) if i != src and i in visible])
        roll_targets = len([i for i in visible if i != src])
        return RollSettingsDialog(self, state.config, GearProfiles.load_library(), sel_count=sel_targets, roll_count=roll_targets)

    def _exec_roll_settings_dialog(self, dlg: RollSettingsDialog) -> None:
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        rows = dlg.selected_rows()
        if not rows:
            return
        if self.controller.session.apply_preset_fields(dlg.selected_config(), rows, dlg.scope()):
            self.controller.request_render()

    def _folder_name_for_gear_suggestion(self) -> str:
        return folder_name_for_active_context(self.session.state, self.session.repo)

    def _detect_gear_suggestion(self, folder_name: str) -> Optional[GearMatch]:
        """The gear match for *folder_name*, restricted to whichever of camera/film
        stock the current frame does not already carry -- checked independently, so an
        unrelated camera already set (carried from another frame, tagged by hand) does
        not also block a film-stock match that is otherwise free to suggest. None when
        there is nothing left to offer."""
        if not folder_name:
            return None
        meta = self.session.state.config.metadata
        detected = match_gear_for_folder(folder_name, GearProfiles.load_library())
        result = GearMatch(
            camera_id="" if meta.camera_id else detected.camera_id,
            film_stock_id="" if meta.film_stock_id else detected.film_stock_id,
        )
        return result if result.any() else None

    def _maybe_suggest_gear(self, folder_path: str) -> None:
        """A folder just became a roll for the first time: offer Roll Settings pre-filled
        from a folder-name match against the gear library, ticked but never applied
        without the user pressing Apply. Silent when nothing matches -- checked before
        building the dialog, so a folder with nothing to suggest never pops one up."""
        detected = self._detect_gear_suggestion(folder_label(folder_path))
        if detected is None:
            return
        dlg = self._build_roll_settings_dialog()
        if dlg is None:
            return
        dlg.apply_detected_gear(camera_id=detected.camera_id, film_stock_id=detected.film_stock_id)
        self._exec_roll_settings_dialog(dlg)

    def _on_update_thumbnails_clicked(self) -> None:
        if self.controller.thumbnail_refresh_running:
            self.controller.cancel_thumbnail_refresh()
        else:
            self.controller.request_thumbnail_refresh("roll")

    def _on_thumbnail_refresh_state_changed(self, running: bool) -> None:
        """Same button starts and stops it: a refresh over a very large folder needs a
        way out that isn't waiting for it to finish."""
        if running:
            self.update_thumbnails_btn.setIcon(qta.icon("fa5s.stop-circle", color=THEME.text_primary))
            self.update_thumbnails_btn.setToolTip("Cancel Thumbnail Update — stop the background refresh in progress")
        else:
            self.update_thumbnails_btn.setIcon(qta.icon("fa5s.sync-alt", color=THEME.text_primary))
            self.update_thumbnails_btn.setToolTip("Update Thumbnails — re-render every stale thumbnail in the roll")

    def _build_session_menu(self) -> QMenu:
        """Mirrors the panel toolbar's add/clear tools, for a right click on empty space."""
        icon_color = THEME.text_primary
        menu = QMenu(self)
        menu.addAction(qta.icon("fa5s.file-import", color=icon_color), "Add Files…").triggered.connect(self.prompt_add_files)
        menu.addAction(qta.icon("fa5s.folder-plus", color=icon_color), "Add Folder…").triggered.connect(self.prompt_add_folder)
        menu.addSeparator()
        clear = menu.addAction(qta.icon("fa5s.times-circle", color=icon_color), "Clear All…")
        clear.triggered.connect(self._on_clear_all)
        clear.setEnabled(bool(self.session.state.uploaded_files))
        return menu

    def _build_context_menu(self) -> QMenu:
        state = self.session.state
        multi = len(state.selected_indices) > 1

        menu = QMenu(self)
        if multi:
            menu.addAction("Export Selected Frames").triggered.connect(lambda: self.controller.request_export_selected())
        else:
            menu.addAction("Export Current Frame").triggered.connect(lambda: self.controller.request_export())
        menu.addSeparator()
        menu.addAction(label_with_shortcut("Copy Settings", "copy")).triggered.connect(self.session.copy_settings)
        menu.addAction(label_with_shortcut("Copy Settings + Bounds", "copy_with_bounds")).triggered.connect(
            self.session.copy_settings_with_bounds
        )
        act_paste = menu.addAction(label_with_shortcut("Paste Settings", "paste"))
        act_paste.triggered.connect(lambda: open_paste_dialog(self, self.controller))
        act_paste.setEnabled(state.clipboard is not None)
        targets = [i for i in (state.selected_indices or [state.selected_file_idx]) if 0 <= i < len(state.uploaded_files)]
        n = len(targets)
        if multi:
            menu.addAction(f"Reset {count_of(n, 'frame')}").triggered.connect(lambda: _reset_selected(self, self.controller))
        else:
            menu.addAction("Reset Settings").triggered.connect(self.session.reset_settings)
            act_roll = menu.addAction(label_with_shortcut("Reset to Roll Settings", "reset_to_roll"))
            act_roll.triggered.connect(self.controller.revert_frame_to_roll)
            act_roll.setEnabled(self.controller.can_revert_frame_to_roll())
        menu.addSeparator()
        act_keep = menu.addAction(f"Keep {count_of(n, 'frame')}" if multi else "Keep")
        act_keep.setCheckable(True)
        act_keep.setChecked(bool(targets) and all(state.uploaded_files[i].get("keeper") for i in targets))
        act_keep.triggered.connect(lambda: self.session.toggle_mark("keeper"))
        act_reject = menu.addAction(f"Reject {count_of(n, 'frame')}" if multi else "Reject")
        act_reject.setCheckable(True)
        act_reject.setChecked(bool(targets) and all(state.uploaded_files[i].get("excluded") for i in targets))
        act_reject.triggered.connect(lambda: self.session.toggle_mark("excluded"))
        menu.addSeparator()
        menu.addAction("Apply Settings…").triggered.connect(self._open_apply_dialog)
        menu.addAction(label_with_shortcut("Sync Bounds…", "sync_bounds")).triggered.connect(
            lambda: open_sync_bounds_dialog(self, self.session)
        )
        if self.controller.thumbnail_refresh_running:
            menu.addAction("Cancel Thumbnail Update").triggered.connect(lambda: self.controller.cancel_thumbnail_refresh())
        else:
            menu.addAction(f"Update {count_of(n, 'thumbnail')}" if multi else "Update Thumbnail").triggered.connect(
                lambda: self.controller.request_thumbnail_refresh("selection")
            )
        menu.addAction(label_with_shortcut("Reset Roll to Defaults…", "reset_roll")).triggered.connect(
            lambda: _reset_roll(self, self.controller)
        )
        if multi:
            menu.addSeparator()
            menu.addAction("Stitch Selected Frames").triggered.connect(lambda: self.controller.request_stitch_selected())
            self._add_hdr_merge_action(menu, state)
        else:
            menu.addSeparator()
            menu.addAction("Edit RGB Triplet…").triggered.connect(self._on_edit_triplet)
            active = state.uploaded_files[state.selected_file_idx] if 0 <= state.selected_file_idx < len(state.uploaded_files) else {}
            if active.get("stitch_paths"):
                menu.addAction("Unstitch").triggered.connect(lambda: self.controller.request_unstitch())
            if active.get("hdr_paths"):
                self._add_hdr_anchor_menu(menu, active)
                menu.addAction("Unmerge Exposures").triggered.connect(lambda: self.controller.request_unmerge_hdr())
            if active.get("diptych"):
                menu.addAction("Unsplit Diptych").triggered.connect(self.prompt_undiptych)
            if active.get("half"):
                from negpy.services.assets.half_frame import base_hash

                base = base_hash(active.get("hash"))
                menu.addAction("Adjust Split for This Frame…").triggered.connect(
                    lambda: self._on_adjust_half_frame_split(active["path"], base)
                )
                if base and self.controller.half_frame_override(base) is not None:
                    menu.addAction("Reset Split to Roll Default").triggered.connect(lambda: self._on_reset_half_frame_split(base))
            if state.active_roll_id and active.get("path"):
                if rolls.is_forked(self.session.repo, state.active_roll_id, active.get("hash") or ""):
                    menu.addAction("Use the Shared Edit Again…").triggered.connect(self.prompt_unfork_edit)
                elif len(rolls.rolls_containing_path(self.session.repo, active["path"])) >= 2:
                    menu.addAction("Edit Independently in This Roll").triggered.connect(
                        lambda: self.controller.request_fork_edit_for_roll()
                    )
        if state.active_roll_id:
            self._add_scene_menu(menu, [state.uploaded_files[i] for i in targets], multi)
        menu.addSeparator()
        unload_label = "Unload Selected…" if multi else "Unload…"
        menu.addAction(unload_label).triggered.connect(self._on_remove_from_menu)
        return menu

    def _add_scene_menu(self, menu: QMenu, frames: List[dict], multi: bool) -> None:
        scenes = rolls.roll_scenes(self.session.repo, self.session.state.active_roll_id)
        # Scene id per frame, None for a frame outside every scene.
        of_frames = {(f.get("scene") or (0, None))[1] for f in frames}
        scene_menu = menu.addMenu("Scene")
        if multi:
            scene_menu.addAction("Group as Scene…").triggered.connect(self._on_group_as_scene)
        for scene_id, entry in scenes:
            if of_frames != {scene_id}:
                scene_menu.addAction(f"Add to {entry['name']}").triggered.connect(
                    lambda _=False, sid=scene_id: self.controller.request_add_to_scene(sid)
                )
        if of_frames - {None}:
            scene_menu.addAction("Remove from Scene").triggered.connect(lambda: self.controller.request_remove_from_scene())
        if len(of_frames) == 1 and None not in of_frames:
            (sid,) = of_frames
            scene_menu.addSeparator()
            scene_menu.addAction("Analyze Scene…").triggered.connect(lambda: self.controller.request_scene_analysis(sid))
            scene_menu.addAction("Rename Scene…").triggered.connect(lambda: self._on_rename_scene(sid))
            scene_menu.addAction("Delete Scene…").triggered.connect(lambda: prompt_delete_scene(self, self.controller, sid))
        scene_menu.setEnabled(not scene_menu.isEmpty())

    def _on_group_as_scene(self) -> None:
        default = rolls.next_scene_name(self.session.repo, self.session.state.active_roll_id)
        name, ok = QInputDialog.getText(self, "Group as Scene", "Name:", text=default)
        if ok and name.strip():
            self.controller.request_group_as_scene(name.strip())

    def _on_rename_scene(self, scene_id: str) -> None:
        current = dict(rolls.roll_scenes(self.session.repo, self.session.state.active_roll_id)).get(scene_id, {}).get("name", "")
        name, ok = QInputDialog.getText(self, "Rename Scene", "Name:", text=current)
        if ok and name.strip():
            self.controller.request_rename_scene(scene_id, name.strip())

    def prompt_undiptych(self) -> None:
        if confirm_undiptych(self):
            self.controller.request_undiptych()

    def prompt_unfork_edit(self) -> None:
        if confirm_unfork_edit(self):
            self.controller.request_unfork_edit_for_roll()

    def _add_hdr_merge_action(self, menu, state) -> None:
        """Merging is for transparencies, so the action follows the film process.

        A color negative holds about 5-6 stops between base and Dmax, and an ordinary
        black-and-white negative nearer 4 — both inside a single capture, so a bracket buys
        nothing. A transparency runs to 10-12, which is what the merge exists for.

        Hidden on Color Negative, disabled with a reason on B&W Negative: reversal-processed monochrome
        (Scala, dr5, Fomapan R) *is* a transparency and does have the range, it is simply
        not wired yet, and a missing menu entry would leave nobody anything to ask about.
        """
        from negpy.features.process.models import ProcessMode

        idx = state.selected_file_idx
        assets = state.uploaded_files
        mode = self.controller.state.config.process.process_mode
        if 0 <= idx < len(assets):
            # Coerced, not compared raw: a session blob written before the mode rename still carries
            # the old names.
            mode = ProcessMode(assets[idx].get("process_mode") or mode)
        if mode == ProcessMode.C41:
            return
        act = menu.addAction("Merge Exposures (HDR)")
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
        auto = sub.addAction("Bracket Middle (Auto)")
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
        open_triplet_dialog(self, self.session)

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
