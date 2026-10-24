from negpy.kernel.system.i18n import tr
import math
import os
from typing import Any, List, Optional

import negpy_core
import numpy as np

import qtawesome as qta
from PyQt6.QtCore import (
    Qt,
    QEvent,
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
from PyQt6.QtGui import (
    QAction,
    QActionGroup,
    QColor,
    QIcon,
    QImage,
    QKeySequence,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QShortcut,
)
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QDialog,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListView,
    QMenu,
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
from negpy.desktop.view.canvas.toolbar import CANVAS_COLORS
from negpy.desktop.view.confirm import (
    prompt_delete_scene,
    confirm_reset_frames,
    confirm_undiptych,
    confirm_unfork_edit,
    confirm_unload,
    warn_invalid_roll_name,
)
from negpy.desktop.view.frame_merge_action import (
    ACTION_IDS,
    LABELS,
    SCOPE_FRAME,
    SCOPE_SELECTION,
    merge_to_tiff,
    mergeable_in,
)
from negpy.desktop.view.keyboard_shortcuts import _close_roll, _reset_roll, _reset_selected, close_roll_label, remove_from_roll
from negpy.features.hdr.logic import anchor_choices
from negpy.features.hdr.models import hdr_frame_paths
from negpy.desktop.view.widgets.elided_label import ElidedLabel
from negpy.desktop.view.widgets.marks import CHIP_FILL, CHIP_RING, draw_mark_badge
from negpy.desktop.view.widgets.sort_button import SortButton
from negpy.desktop.view.shortcut_registry import label_with_shortcut, tooltip_with_shortcut
from negpy.desktop.view.styles.templates import BAR_BUTTON_HEIGHT, BAR_ICON_SIZE, hint_label, with_alpha, wrap_tooltip
from negpy.desktop.view.styles.theme import THEME, scene_color
from negpy.desktop.view.widgets.granular_settings_dialog import open_apply_dialog, open_paste_dialog, open_sync_bounds_dialog
from negpy.desktop.view.sidecar_action import LABEL as SIDECAR_LABEL, load_edit_from_sidecar
from negpy.desktop.view.file_actions import SHOW_LABEL, TRASH_LABEL, move_selected_to_trash, show_current_in_folder
from negpy.desktop.view.widgets.rgb_triplet_dialog import open_triplet_dialog
from negpy.desktop.view.widgets.roll_settings_dialog import RollSettingsDialog
from negpy.services.assets import rolls
from negpy.services.assets.gear import GearProfiles
from negpy.services.assets.gear_match import GearMatch, folder_name_for_active_context, match_gear_for_folder
from negpy.services.assets.presets import is_valid_preset_name
from negpy.infrastructure.filesystem.watcher import FolderWatchService
from negpy.infrastructure.loaders.helpers import get_supported_raw_wildcards
from negpy.desktop.view.sidebar.library_tree import LibraryTree
from negpy.desktop.view.widgets.file_dialogs import last_open_folder
from negpy.services.assets.library import folder_counts, folder_label
from negpy.services.assets.thumbnails import asset_thumbnail_key
from negpy.domain.models import WorkspaceConfig
from negpy.kernel.image.logic import THUMBNAIL_PAD
from negpy.services.export.contact_sheet import palette_for
from negpy.services.export.contact_sheet_roll import SheetFrame, infer_format, sheet_look
from negpy.services.film_edge.geometry import FilmGeometry, film_geometry


class _ThumbnailDelegate(QStyledItemDelegate):
    """Scales each cached thumbnail into its cell; a frame not selected or under the pointer is
    dimmed. Triage marks are small bottom-right badges: check = keeper, cross + heavy dim =
    rejected; the bottom-left badge says the frame was built from several files
    (stitch, HDR merge, RGB triplet, half-frame split). Top-left holds the
    decode-failure badge, else a small dot saying the bitmap shown predates a settings
    change (a bulk apply reaches the file before a render reaches its thumbnail)."""

    _RADIUS = 4  # = button border-radius (modern_dark.qss)
    _COMPOSITE_GLYPH = with_alpha(THEME.mark_photo, 235)
    _IDLE_OPACITY = 0.5
    _STALE_DOT_RADIUS = 4
    _ACTIVITY_INTERVAL_MS = 40
    _ACTIVITY_STEP = 0.035

    def __init__(self, parent=None, state: Optional[AppState] = None) -> None:
        super().__init__(parent)
        self._state = state
        self._placeholder_icon = qta.icon("fa5s.image", color=THEME.text_muted)
        self._activity_icon = qta.icon("fa5s.image", color=THEME.text_secondary)
        self._activity_key = ""
        self._activity_index = QPersistentModelIndex()
        self._activity_phase = 0.0
        self._activity_timer = QTimer(self)
        self._activity_timer.setInterval(self._ACTIVITY_INTERVAL_MS)
        self._activity_timer.timeout.connect(self._advance_activity)

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
        rows = model.rows_showing(self._activity_key) if model is not None else []
        return QPersistentModelIndex(model.index(rows[0], 0)) if rows else QPersistentModelIndex()

    def _index_matches_activity(self, index: QPersistentModelIndex) -> bool:
        if not index.isValid() or not self._activity_key:
            return False
        file_info = index.data(Qt.ItemDataRole.UserRole) or {}
        return bool(file_info.get("hash") and asset_thumbnail_key(file_info) == self._activity_key)

    def _repaint_index(self, index: QPersistentModelIndex) -> None:
        # A hidden view needs no repaint, and visualRect would run its pending layout of every frame.
        view = self.parent()
        if isinstance(view, QListView) and view.isVisible() and index.isValid():
            view.viewport().update(view.visualRect(QModelIndex(index)))

    def is_active(self, file_info: dict) -> bool:
        """The open frame, by hash: the two halves of a half-frame scan share one path."""
        state = self._state
        return bool(state and state.current_file_hash and file_info.get("hash") == state.current_file_hash)

    def _is_stale_thumbnail(self, file_info: dict) -> bool:
        """True while the cached bitmap predates a settings write a render hasn't caught up to."""
        state = self._state
        if not state or not file_info.get("hash"):
            return False
        return asset_thumbnail_key(file_info) in state.stale_thumbnails

    def _draw_stale_dot(self, painter: QPainter, img_rect: QRect) -> None:
        r = self._STALE_DOT_RADIUS
        cx, cy = img_rect.left() + r + 4, img_rect.top() + r + 4
        painter.setPen(QPen(with_alpha(THEME.mark_shadow, 140), 1))
        painter.setBrush(QColor(THEME.warn_amber))
        painter.drawEllipse(QRect(cx - r, cy - r, 2 * r, 2 * r))

    @staticmethod
    def _fit_rect(area: QRect, source_size: QSize) -> QRect:
        size = source_size.scaled(area.size(), Qt.AspectRatioMode.KeepAspectRatio)
        x = area.x() + (area.width() - size.width()) // 2
        y = area.y() + (area.height() - size.height()) // 2
        return QRect(x, y, size.width(), size.height())

    def picture_rect(self, cell: QRect, index: QModelIndex) -> QRect:
        """Where paint() puts the picture inside *cell*: the thumbnail fitted to it, or all of it
        while it has none."""
        area = cell
        icon = index.data(Qt.ItemDataRole.DecorationRole)
        base = self._thumbnail(icon, index) if icon is not None and not icon.isNull() else None
        if base is None or base.isNull():
            return area
        return self._fit_rect(area, base.size().scaled(area.size(), Qt.AspectRatioMode.KeepAspectRatio))

    def _thumbnail(self, icon: QIcon, index: QModelIndex) -> QPixmap:
        return icon.pixmap(QSize(4096, 4096))  # largest available pixmap (~120px)

    def sizeHint(self, option: QStyleOptionViewItem, index: QModelIndex) -> QSize:
        view = self.parent()
        if isinstance(view, QListView) and view.iconSize().isValid():
            return view.iconSize()
        return super().sizeHint(option, index)

    def _draw_mark_badge(self, painter: QPainter, img_rect: QRect, check: bool) -> None:
        draw_mark_badge(painter, img_rect, check)

    def _draw_failed_badge(self, painter: QPainter, img_rect: QRect) -> None:
        r = 9
        cx, cy = img_rect.left() + r + 4, img_rect.top() + r + 4
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(THEME.error))
        painter.drawEllipse(QRect(cx - r, cy - r, 2 * r, 2 * r))
        painter.setPen(QPen(QColor(THEME.text_on_accent), 2))
        painter.drawLine(cx, cy - 4, cx, cy + 1)
        painter.drawPoint(cx, cy + 4)

    def _draw_composite_badge(self, painter: QPainter, img_rect: QRect, kind: str, half: int, split_axis: str = "x") -> None:
        """Bottom-left mark: this frame was assembled from more than one file.

        One glyph per kind, so a merge is told from a stitch without opening the menu.
        The chip carries a faint ring because a flat dark disc vanishes on a dense frame."""
        r = 9
        cx, cy = img_rect.left() + r + 4, img_rect.bottom() - r - 4
        painter.setPen(QPen(CHIP_RING, 1))
        painter.setBrush(CHIP_FILL)
        painter.drawEllipse(QRect(cx - r, cy - r, 2 * r, 2 * r))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(self._COMPOSITE_GLYPH, 1.5))
        if kind == "stitch":  # two overlapping panes — a divided box reads as the half glyph
            painter.drawRect(QRect(cx - 6, cy - 5, 8, 7))
            front = QRect(cx - 2, cy - 2, 8, 7)
            painter.fillRect(front, CHIP_FILL)
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
            panes = self._half_badge_panes(cx, cy, split_axis)
            painter.fillRect(panes[0] if half == 1 else panes[1], self._COMPOSITE_GLYPH)
        elif kind == "diptych":  # the same split frame with both halves filled
            painter.drawRect(QRect(cx - 6, cy - 4, 12, 8))
            for pane in self._half_badge_panes(cx, cy, split_axis):
                painter.fillRect(pane, self._COMPOSITE_GLYPH)

    @staticmethod
    def _half_badge_panes(cx: int, cy: int, split_axis: str) -> tuple[QRect, QRect]:
        """(half 1, half 2) glyph panes: left/right for an "x" split, top/bottom for "y"."""
        if split_axis == "y":
            return QRect(cx - 5, cy - 3, 11, 3), QRect(cx - 5, cy + 1, 11, 3)
        return QRect(cx - 5, cy - 3, 5, 7), QRect(cx + 1, cy - 3, 5, 7)

    def _paint_placeholder(self, painter: QPainter, option: QStyleOptionViewItem, file_info: dict, failed: bool, kind: str) -> None:
        """A cell whose thumbnail hasn't decoded yet (or failed to). Still carries the triage
        marks, so a rejected frame stays visible while thumbnails are still loading in the
        background instead of looking broken."""
        rejected = bool(file_info.get("excluded"))
        keeper = bool(file_info.get("keeper"))
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        area = option.rect
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(THEME.bg_header))
        painter.drawRoundedRect(area, self._RADIUS, self._RADIUS)
        if rejected:
            self._draw_mark_badge(painter, area, check=False)
        elif keeper:
            self._draw_mark_badge(painter, area, check=True)
        if failed:
            self._draw_failed_badge(painter, area)
        if kind:
            self._draw_composite_badge(painter, area, kind, int(file_info.get("half") or 0), str(file_info.get("split_axis") or "x"))
        painter.restore()

    def paint(self, painter: QPainter, option: QStyleOptionViewItem, index: QModelIndex) -> None:
        file_info = index.data(Qt.ItemDataRole.UserRole) or {}
        failed = bool(file_info.get("decode_failed"))
        kind = composite_kind(file_info)

        icon = index.data(Qt.ItemDataRole.DecorationRole)
        if icon is None or icon.isNull():
            painter.save()
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            img_rect = option.rect
            rejected = bool(file_info.get("excluded"))
            keeper = bool(file_info.get("keeper"))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(THEME.bg_header))
            painter.drawRoundedRect(img_rect, self._RADIUS, self._RADIUS)
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
                self._draw_composite_badge(
                    painter, img_rect, kind, int(file_info.get("half") or 0), str(file_info.get("split_axis") or "x")
                )
            painter.restore()
            return
        base = self._thumbnail(icon, index)
        if base.isNull():
            self._paint_placeholder(painter, option, file_info, failed, kind)
            return

        painter.save()
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        area = option.rect
        scaled = base.scaled(
            area.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        img_rect = self._fit_rect(area, scaled.size())

        selected = bool(option.state & QStyle.StateFlag.State_Selected)
        hover = bool(option.state & QStyle.StateFlag.State_MouseOver)
        rejected = bool(file_info.get("excluded"))
        keeper = bool(file_info.get("keeper"))

        clip = QPainterPath()
        clip.addRoundedRect(QRectF(img_rect), self._RADIUS, self._RADIUS)
        painter.setClipPath(clip)
        base_opacity = 1.0 if (selected or hover) else self._IDLE_OPACITY
        painter.setOpacity(0.25 if rejected else base_opacity)
        painter.drawPixmap(img_rect.topLeft(), scaled)
        painter.setOpacity(1.0)

        if rejected:
            self._draw_mark_badge(painter, img_rect, check=False)
        elif keeper:
            self._draw_mark_badge(painter, img_rect, check=True)
        if kind:
            self._draw_composite_badge(painter, img_rect, kind, int(file_info.get("half") or 0), str(file_info.get("split_axis") or "x"))
        if not failed and self._is_stale_thumbnail(file_info):
            self._draw_stale_dot(painter, img_rect)
        painter.setClipping(False)

        if failed:
            self._draw_failed_badge(painter, img_rect)

        painter.restore()


# A grid cell's target width, one frame pitch on the Light Table; columns justify to the width.
THUMB_CELL_DEFAULT = 220
THUMB_CELL_MIN = 110


class ThumbnailGridView(QListView):
    """Icon-mode grid that justifies cells to the view width: as many columns of at least
    ``target_cell`` as fit, each grown to fill the width exactly."""

    SPACING = 2
    # One notch scrolls one row of thumbnails. Qt's default, three "lines" a notch, advanced
    # several frames at a time in a single-column panel.
    WHEEL_ROWS_PER_NOTCH = 1.0
    SCROLL_ANIM_MS = 160
    # A held press sweeps a range, or a box from empty space; off where a drag moves frames instead.
    DRAG_SELECTS = True

    def __init__(self, parent=None, target_cell: int = THUMB_CELL_DEFAULT):
        super().__init__(parent)
        self._last_cell = -1
        self._target_cell = target_cell
        self._pending_click_row: Optional[int] = None
        self._pending_click_modifiers = Qt.KeyboardModifier.NoModifier
        self._pending_mode: Optional[str] = None  # "range" | "ctrl"
        self._range_anchor_row: Optional[int] = None
        self._ctrl_target: set[QPersistentModelIndex] = set()
        self._pre_press_selection: set[QPersistentModelIndex] = set()
        # Hover tracks the pointer, not only a scroll. The macOS style turns on neither, so a
        # plain move never reaches the view.
        self.setMouseTracking(True)
        self.viewport().setAttribute(Qt.WidgetAttribute.WA_Hover, True)
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
        if not self.DRAG_SELECTS and event.buttons() & Qt.MouseButton.LeftButton:
            return
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
            (self._row_near(event.position().toPoint()) if self.DRAG_SELECTS else self._pending_click_row)
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


def table_color(index: int) -> str:
    """The surface of the Film Strip and the Light Table: the canvas background."""
    return CANVAS_COLORS[index][0]


# Share of a print's darkest pixels whose mean color is its D-max.
_DMAX_SHARE = 0.01
# Frames spread over the roll that stand for its D-max; measuring every frame stalls a long roll.
_DMAX_SAMPLES = 5
# A new thumbnail this close to the shape a frame shows keeps the layout: about 2 px on a strip frame.
_SHAPE_TOLERANCE = 0.02
_RESHAPE_MS = 50


def roll_film(
    asset: dict,
    config: WorkspaceConfig,
    library: Any,
    half_frame: bool,
    rebate: Optional[tuple[int, int, int]] = None,
) -> tuple[FilmGeometry, tuple[int, int, int]]:
    """The roll's film: the active frame's format and the rebate color of the contact sheet's palette;
    `rebate`, the roll's D-max, replaces that color."""
    frames = [SheetFrame(asset, config)]
    fmt, frame_size = infer_format(frames, half_frame)
    palette = palette_for(sheet_look(frames, fmt, library=library))
    if rebate is None:
        rebate = palette.rebate
    return film_geometry(fmt, frame_size), rebate


# The film until the active frame's roll gives one.
_DEFAULT_FILM = roll_film({}, WorkspaceConfig(), None, half_frame=False)


def _rgb_pixels(image: QImage) -> np.ndarray:
    rgb = image.convertToFormat(QImage.Format.Format_RGB888)
    width, height = rgb.width(), rgb.height()
    rows = np.frombuffer(rgb.constBits().asstring(rgb.sizeInBytes()), np.uint8).reshape(height, rgb.bytesPerLine())
    return rows[:, : width * 3].reshape(height, width, 3)


def print_dmax(image: QImage) -> Optional[tuple[int, int, int]]:
    """A print's D-max tone: the mean color of the darkest pixels of a thumbnail, its padding left out."""
    pixels = np.ascontiguousarray(_rgb_pixels(image.copy(picture_in_thumbnail(image))))
    return negpy_core.print_dmax(pixels, _DMAX_SHARE)


def picture_in_thumbnail(image: QImage) -> QRect:
    """The picture inside a thumbnail: all of it, or in a square one from an older cache, the part
    inside the THUMBNAIL_PAD padding on one axis."""
    if image.width() != image.height():
        return image.rect()
    pixels = _rgb_pixels(image).astype(np.int16)
    height, width = pixels.shape[:2]
    live = np.abs(pixels - np.array(THUMBNAIL_PAD, np.int16)).max(axis=2) > 10

    def pad(lines: np.ndarray) -> int:
        found = np.flatnonzero(lines)
        return int(min(found[0], len(lines) - 1 - found[-1])) if found.size else 0

    pad_y, pad_x = pad(live.any(axis=1)), pad(live.any(axis=0))
    if pad_y >= pad_x:
        return QRect(0, pad_y, width, height - 2 * pad_y)
    return QRect(pad_x, 0, width - 2 * pad_x, height)


def justified_rows(aspects: list[float], width: int, height: int, gap: int) -> list[tuple[int, list[tuple[int, int]]]]:
    """Frames `gap` apart in rows that fill `width`: per row, the picture height and each frame's (x, width).
    A row takes frames until they reach the width at `height`, then shrinks to fit; the last row keeps `height`."""
    rows: list[tuple[int, list[tuple[int, int]]]] = []
    start, reach = 0, 0.0
    for end in range(1, len(aspects) + 1):
        reach += aspects[end - 1] * height + gap
        last = end == len(aspects)
        if reach < width and not last:
            continue
        row = aspects[start:end]
        h = (width - gap * len(row)) / sum(row) if reach >= width else float(height)
        x, frames = 0.0, []
        for aspect in row:
            right = x + aspect * h + gap
            frames.append((round(x), round(right) - round(x)))
            x = right
        rows.append((round(h), frames))
        start, reach = end, 0.0
    return rows


# The corner radius of a scene's name chip; a selection mat's is half a pixel more.
_CHIP_RADIUS = 3


def _scene_mark(index: QModelIndex) -> Optional[tuple]:
    """A frame's scene, ``(ordinal, id, name, found)``: found is True for a group Detect Scenes
    proposed that waits for Accept or Discard."""
    data = index.data(Qt.ItemDataRole.UserRole) or {}
    if data.get("scene"):
        return (*data["scene"], False)
    if data.get("found_scene"):
        return (*data["found_scene"], True)
    return None


class _FilmDelegate(_ThumbnailDelegate):
    """Each frame stands upright in a cell as wide as its picture, a thin rebate border around it."""

    _CACHE_LIMIT = 512
    # Every layout reads the picture bounds of every frame, so their small cache holds a long roll. The strip's
    # and the Light Table's delegates share it, as they show the same thumbnails.
    _BOUNDS_LIMIT = 1 << 16
    _bounds: dict[int, QRect] = {}
    _IDLE_OPACITY = 0.75
    _BORDER = THEME.space_sm

    def __init__(self, parent=None, state: Optional[AppState] = None) -> None:
        super().__init__(parent, state)
        self._film = _DEFAULT_FILM
        self._pictures: dict[int, QPixmap] = {}

    def set_film(self, geometry: FilmGeometry, rebate: tuple[int, int, int]) -> None:
        self._film = (geometry, rebate)

    def picture_aspect(self, index: QModelIndex) -> float:
        """The upright picture's shape."""
        icon = index.data(Qt.ItemDataRole.DecorationRole)
        base = super()._thumbnail(icon, index) if isinstance(icon, QIcon) and not icon.isNull() else None
        bounds = self._picture_bounds(base) if base is not None and not base.isNull() else QRect()
        return bounds.width() / bounds.height() if bounds.height() else self.placeholder_aspect

    @property
    def placeholder_aspect(self) -> float:
        return self._film[0].window_aspect

    def _picture_bounds(self, base: QPixmap) -> QRect:
        bounds = self._bounds.get(base.cacheKey())
        if bounds is None:
            if len(self._bounds) > self._BOUNDS_LIMIT:
                self._bounds.clear()
            bounds = self._bounds[base.cacheKey()] = picture_in_thumbnail(base.toImage())
        return bounds

    def sizeHint(self, option: QStyleOptionViewItem, index: QModelIndex) -> QSize:
        view = self.parent()
        if isinstance(view, _FilmGridView):
            return view.slim_cell(index)
        return super().sizeHint(option, index)

    def _thumbnail(self, icon: QIcon, index: QModelIndex) -> QPixmap:
        base = super()._thumbnail(icon, index)
        if base.isNull():
            return base
        picture = self._pictures.get(base.cacheKey())
        if picture is None:
            # A re-rendered thumbnail is a new pixmap, so the cache only grows until cleared.
            if len(self._pictures) > self._CACHE_LIMIT:
                self._pictures.clear()
            picture = self._pictures[base.cacheKey()] = base.copy(self._picture_bounds(base))
        return picture

    def band(self, cell: QRect) -> QRect:
        """The film across a cell, without the view's space above and below it."""
        view = self.parent()
        gap = view.STRIP_GAP if isinstance(view, _FilmGridView) else 0
        return cell.adjusted(0, gap // 2, 0, -(gap - gap // 2))

    def _window(self, cell: QRect) -> QRect:
        b = self._BORDER
        return self.band(cell).adjusted(b // 2, b, -(b - b // 2), -b)

    def picture_rect(self, cell: QRect, index: QModelIndex) -> QRect:
        return super().picture_rect(self._window(cell), index)

    def paint(self, painter: QPainter, option: QStyleOptionViewItem, index: QModelIndex) -> None:
        cell = option.rect
        painter.fillRect(self.band(cell), QColor(*self._film[1]))
        framed = QStyleOptionViewItem(option)
        framed.rect = self._window(cell)
        super().paint(painter, framed, index)


class _LightTableDelegate(_FilmDelegate):
    """The Light Table shows every picture at full strength; only a rejected frame dims."""

    _IDLE_OPACITY = 1.0


class _FilmGridView(ThumbnailGridView):
    """Frames in justified rows, the cells of a row joined into one strip; each run of a scene's frames
    on one row starts with the scene's name."""

    SPACING = 0
    STRIP_GAP = THEME.space_xl
    # Scene sort: the gap between two scenes' blocks.
    SCENE_GAP = THEME.space_lg
    _film = _DEFAULT_FILM[0]
    cell_zoomed = pyqtSignal(int)
    # A layout that may change the frames to a row or their range: a zoom, a resize or a new film.
    columns_changed = pyqtSignal()
    # Selected frames dropped on a found scene: its index, or -1 for a frame in no scene.
    found_scene_drop = pyqtSignal(int)
    # The Light Table's drag only moves found scenes' frames; the Film Strip keeps its drag selection.
    DRAG_SELECTS = False

    def __init__(self, parent=None, target_cell: int = THUMB_CELL_DEFAULT) -> None:
        # A press held back from Qt while it may become a drag of found scenes' frames.
        self._scene_move: Optional[dict] = None
        self._move_was_click = False
        self._slim_cells: Optional[dict[int, QRect]] = None
        # Each row's picture shape as the layout read it, kept until the frames take new shapes.
        self._shown_aspects: dict[int, float] = {}
        # scene_runs_on_rows in content coordinates, kept until the next layout or data change.
        self._scene_runs: Optional[list[tuple[tuple, QModelIndex, QRect, QRect]]] = None
        self._placing_scenes = False
        super().__init__(parent, target_cell)
        # Thumbnails arrive faster than a layout of every frame; new shapes in one burst share one layout.
        self._reshape_timer = QTimer(self)
        self._reshape_timer.setSingleShot(True)
        self._reshape_timer.setInterval(_RESHAPE_MS)
        self._reshape_timer.timeout.connect(self._invalidate_slim)
        self._zoom_angle = 0
        # Margins, not QSS padding, which would inset the scroll bar too; the film runs off the top and bottom.
        self.setViewportMargins(THEME.space_2xl, 0, THEME.space_2xl, 0)
        self.set_table_color(table_color(AppState.canvas_bg_index))

    def set_table_color(self, color: str) -> None:
        self._table_color = color
        # A new stylesheet lays every cell out again, even when it is the same.
        qss = self._table_qss(color)
        if qss != self.styleSheet():
            self.setStyleSheet(qss)

    def _table_qss(self, color: str) -> str:
        return f"QListView {{ background-color: {color}; border: none; }}"

    def set_film(self, geometry: FilmGeometry) -> None:
        """Lays the cells out again for a new film."""
        self._film = geometry
        self._last_cell = -1
        self._slim_cells = None
        self._relayout()

    def setModel(self, model) -> None:  # noqa: N802
        super().setModel(model)
        for signal in (model.modelReset, model.layoutChanged, model.rowsInserted, model.rowsRemoved):
            signal.connect(self._on_rows_changed)
        model.dataChanged.connect(self._on_data_changed)

    def _on_rows_changed(self, *_args) -> None:
        self._invalidate_slim()

    def _invalidate_slim(self) -> None:
        self._slim_cells = None
        self._shown_aspects = {}
        self._scene_runs = None
        self.scheduleDelayedItemsLayout()

    def _on_data_changed(self, top_left: QModelIndex, bottom_right: QModelIndex, *_roles) -> None:
        # Qt does not move the frames after one whose width a new thumbnail changed. A layout of every frame
        # per thumbnail stalls a long roll, so a thumbnail within _SHAPE_TOLERANCE of the shape it replaces keeps it.
        self._scene_runs = None
        # A row no layout has read yet takes its new shape from the pending one.
        delegate = self.itemDelegate()
        for row in range(top_left.row(), bottom_right.row() + 1):
            before = self._shown_aspects.get(row)
            if before is not None and abs(delegate.picture_aspect(top_left.siblingAtRow(row)) / before - 1) > _SHAPE_TOLERANCE:
                if not self._reshape_timer.isActive():
                    self._reshape_timer.start()
                return

    def _slim_height(self) -> int:
        """The picture height of a full row: a landscape frame as wide as one cell."""
        return max(1, round(self._last_cell / self._film.window_aspect))

    @property
    def _aspect(self) -> float:
        """The film's width over a frame's pitch along it."""
        return self._film.width / self._film.pitch

    def slim_cell(self, index: QModelIndex) -> QSize:
        return self._slim_layout().get(index.row(), QRect()).size()

    def _layout_aspect(self, index: QModelIndex) -> float:
        row = index.row()
        if row not in self._shown_aspects:
            self._shown_aspects[row] = self.itemDelegate().picture_aspect(index)
        return self._shown_aspects[row]

    def _slim_layout(self) -> dict[int, QRect]:
        """Each frame's cell in justified rows, the gap between rows split above and below its film; a
        scene sort starts each scene's block on a new row."""
        if self._slim_cells is None:
            model, delegate = self.model(), self.itemDelegate()
            runs = model.scene_runs() if hasattr(model, "scene_runs") else []
            blocks = [(first, last) for _ordinal, first, last in runs] or [(0, model.rowCount() - 1)]
            border, cells, y = delegate._BORDER, {}, 0
            for first, last in blocks:
                aspects = [self._layout_aspect(model.index(row, 0)) for row in range(first, last + 1)]
                row = first
                for height, frames in justified_rows(aspects, self.viewport().width(), self._slim_height(), border):
                    for x, width in frames:
                        cells[row] = QRect(x, y, width, height + 2 * border + self.STRIP_GAP)
                        row += 1
                    y += height + 2 * border + self.STRIP_GAP
                y += self.SCENE_GAP
            self._slim_cells = cells
        return self._slim_cells

    def cell_for_width(self, vw: int) -> int:
        # QListView wraps a row whose right edge reaches the viewport's last pixel.
        return max(1, (vw - 1) // self.columns_for_width(vw))

    def column_range(self) -> tuple[int, int, int]:
        """(fewest, default, most) frames to a row: one row of film as tall as the view, frames of
        THUMB_CELL_DEFAULT, and the smallest frames."""
        vw, vh = self.viewport().width(), self.viewport().height()
        fewest = max(1, math.ceil(vw * self._aspect / max(1, vh)))
        most = max(fewest, vw // THUMB_CELL_MIN)
        return fewest, min(most, max(fewest, vw // THUMB_CELL_DEFAULT)), most

    def columns(self) -> int:
        return self.columns_for_width(self.viewport().width())

    def set_columns(self, columns: int) -> None:
        fewest, _default, most = self.column_range()
        self._target_cell = self.viewport().width() // min(most, max(fewest, columns))
        self._relayout()
        self.scrollTo(self.currentIndex())
        self.cell_zoomed.emit(self._target_cell)

    def zoom(self, steps: int) -> None:
        """One frame fewer a row per step."""
        self.set_columns(self.columns() - steps)

    def _relayout(self) -> None:
        super()._relayout()
        self.columns_changed.emit()

    def wheelEvent(self, event) -> None:
        if not event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            super().wheelEvent(event)
            return
        # A trackpad sends one notch as many small deltas.
        self._zoom_angle += event.angleDelta().y()
        steps = int(self._zoom_angle / 120)
        if steps:
            self._zoom_angle -= steps * 120
            self.zoom(steps)
        event.accept()

    def _apply_cell(self, cell: int) -> None:
        if cell == self._last_cell:
            return
        self._last_cell = cell
        # No grid: each cell takes the size the delegate asks of slim_cell.
        self._slim_cells = None
        self.setGridSize(QSize())
        self.scheduleDelayedItemsLayout()
        self.verticalScrollBar().setSingleStep(max(1, self._row_step() // 4))

    def resizeEvent(self, event) -> None:
        self._slim_cells = None
        super().resizeEvent(event)

    def _row_step(self) -> int:
        return self._slim_height() + 2 * _FilmDelegate._BORDER + self.STRIP_GAP

    def _place_scene_runs(self) -> None:
        """Each frame at its slim cell. Qt re-flows every cell on each layout, so this runs after each one."""
        model = self.model()
        if model is None:
            return
        for row, cell in self._slim_layout().items():
            self.setPositionForIndex(cell.topLeft(), model.index(row, 0))

    def updateGeometries(self) -> None:
        self._scene_runs = None
        super().updateGeometries()
        if self._placing_scenes:
            return
        self._placing_scenes = True
        try:
            self._place_scene_runs()
        finally:
            self._placing_scenes = False

    def scene_runs_on_rows(self) -> list[tuple[tuple, QModelIndex, QRect, QRect]]:
        """(scene, first index, first cell, last cell) per run of one scene's consecutive frames on one row."""
        model = self.model()
        if model is None or not getattr(model, "has_scenes", True):
            return []
        if self._scene_runs is None:
            self._scene_runs = self._find_scene_runs()
        dx, dy = -self.horizontalOffset(), -self.verticalOffset()
        return [(scene, index, first.translated(dx, dy), last.translated(dx, dy)) for scene, index, first, last in self._scene_runs]

    def _find_scene_runs(self) -> list[tuple[tuple, QModelIndex, QRect, QRect]]:
        model = self.model()
        runs: list[tuple[tuple, QModelIndex, QRect, QRect]] = []
        current = None
        for row in range(model.rowCount()):
            index = model.index(row, 0)
            scene = _scene_mark(index)
            rect = self.rectForIndex(index)
            if scene and current and current[0] == scene and current[3].top() == rect.top():
                current = (scene, current[1], current[2], rect)
                runs[-1] = current
            elif scene:
                current = (scene, index, rect, rect)
                runs.append(current)
            else:
                current = None
        return runs

    def _starts_scene_move(self, event, index: QModelIndex) -> bool:
        """A plain press on a frame while found scenes wait for review. Qt's own press would collapse the
        selection to this frame, so the press waits until the gesture moves or not."""
        return (
            not self.DRAG_SELECTS
            and event.button() == Qt.MouseButton.LeftButton
            and event.modifiers() == Qt.KeyboardModifier.NoModifier
            and index.isValid()
            and bool(getattr(self.model(), "has_found_scenes", False))
        )

    def _lift(self, move: dict) -> None:
        """Starts a drag: the pressed frame alone unless it is selected, and pictures of the first two frames
        to carry under the pointer."""
        model, sel = self.model(), self.selectionModel()
        pressed = model.index(move["row"], 0)
        if not sel.isSelected(pressed):
            sel.setCurrentIndex(pressed, QItemSelectionModel.SelectionFlag.ClearAndSelect)
        rows = sorted(index.row() for index in sel.selectedIndexes())
        carried = [move["row"], *(row for row in rows if row != move["row"])][:2]
        move.update(
            rows=rows, grab=move["press"] - self.visualRect(pressed).topLeft(), ghosts=[self._ghost(model.index(row, 0)) for row in carried]
        )

    def _ghost(self, index: QModelIndex) -> QPixmap:
        """The frame as its cell shows it, unselected."""
        size, ratio = self.visualRect(index).size(), self.devicePixelRatioF()
        pixmap = QPixmap(size * ratio)
        pixmap.setDevicePixelRatio(ratio)
        pixmap.fill(Qt.GlobalColor.transparent)
        option = QStyleOptionViewItem()
        self.initViewItemOption(option)
        option.rect = QRect(0, 0, size.width(), size.height())
        painter = QPainter(pixmap)
        self.itemDelegate().paint(painter, option, index)
        painter.end()
        return pixmap

    @staticmethod
    def _drop_target(index: QModelIndex) -> Optional[int]:
        """The found scene a drop on *index* adds to, -1 for a frame in no scene, None where nothing drops."""
        if not index.isValid():
            return None
        mark = _scene_mark(index)
        if mark is None:
            return -1
        return mark[1] if mark[3] else None

    def mousePressEvent(self, event) -> None:
        self._move_was_click = False
        index = self.indexAt(event.position().toPoint())
        if self._starts_scene_move(event, index):
            self._scene_move = {"press": event.position().toPoint(), "row": index.row(), "dragging": False, "pos": None, "target": None}
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        move = self._scene_move
        if move is None:
            super().mouseMoveEvent(event)
            return
        pos = event.position().toPoint()
        if move["dragging"] or (pos - move["press"]).manhattanLength() >= QApplication.startDragDistance():
            if not move["dragging"]:
                self._lift(move)
            move.update(dragging=True, pos=pos, target=self._drop_target(self.indexAt(pos)))
            self.viewport().update()

    def mouseReleaseEvent(self, event) -> None:
        move, self._scene_move = self._scene_move, None
        if move is None:
            super().mouseReleaseEvent(event)
            return
        if move["dragging"]:
            self.viewport().update()
            if move["target"] is not None:
                self.found_scene_drop.emit(move["target"])
            return
        # It stayed put: a plain click, this frame alone selected.
        self._move_was_click = True
        self.selectionModel().setCurrentIndex(self.model().index(move["row"], 0), QItemSelectionModel.SelectionFlag.ClearAndSelect)

    def mouseDoubleClickEvent(self, event) -> None:
        # Qt never saw this double click's first press, so it would take the second for a new press.
        index = self.indexAt(event.position().toPoint())
        if self._move_was_click and index.isValid():
            self._move_was_click = False
            self.doubleClicked.emit(index)
            return
        super().mouseDoubleClickEvent(event)

    def focusOutEvent(self, event) -> None:
        super().focusOutEvent(event)
        if self._scene_move is not None:
            self._scene_move = None
            self.viewport().update()

    def _paint_scene_move(self, painter: QPainter) -> None:
        """While frames are dragged: their places faded to the table, the target scene's frames outlined in
        its color, the frames under the pointer, and a chip below them that says where the frames go."""
        move, model = self._scene_move, self.model()
        if move is None or not move["dragging"] or model is None:
            return
        delegate = self.itemDelegate()
        painter.setPen(QPen(QColor(THEME.border_checked), 2, Qt.PenStyle.DashLine))
        painter.setBrush(with_alpha(self._table_color, 215))
        for row in move["rows"]:
            painter.drawRoundedRect(QRectF(delegate._window(self.visualRect(model.index(row, 0)))), _CHIP_RADIUS, _CHIP_RADIUS)
        target, name, color = move["target"], "", THEME.text_hint
        lines: list[QRect] = []
        last_row = None
        for row in range(model.rowCount()):
            index = model.index(row, 0)
            if target is None or self._drop_target(index) != target:
                continue
            mark = _scene_mark(index)
            if mark is not None:
                name, color = mark[2], scene_color(mark[0])
            rect = self.visualRect(index)
            if last_row == row - 1 and lines and lines[-1].top() == rect.top():
                lines[-1] = lines[-1].united(rect)
            else:
                lines.append(rect)
            last_row = row
        painter.setPen(QPen(QColor(color), 2))
        painter.setBrush(with_alpha(color, 30))
        for rect in lines:
            painter.drawRoundedRect(QRectF(rect).adjusted(1, 1, -1, -1), _CHIP_RADIUS, _CHIP_RADIUS)

        scale, grab, pos = 0.9, move["grab"], move["pos"]
        size = move["ghosts"][0].deviceIndependentSize()
        for k in reversed(range(len(move["ghosts"]))):
            painter.save()
            painter.translate(pos + QPoint(THEME.space_md, THEME.space_sm) * k)
            painter.rotate(3 if k else -2)
            painter.scale(scale, scale)
            painter.setOpacity(0.85 if k else 0.95)
            painter.fillRect(QRectF(-grab.x(), THEME.space_sm - grab.y(), size.width(), size.height()), with_alpha(THEME.mark_shadow, 120))
            painter.drawPixmap(-grab, move["ghosts"][k])
            painter.restore()

        frames = count_of(len(move["rows"]), "frame")
        if target is None:
            text = frames
        elif target < 0:
            text = tr("Remove {value} from the scene", value=frames)
        else:
            text = tr("Add {value} to {name}", value=frames, name=name)
        font = painter.font()
        font.setPixelSize(THEME.font_size_small)
        font.setBold(True)
        painter.setFont(font)
        metrics, pad = painter.fontMetrics(), THEME.space_sm
        if len(move["rows"]) > 1:
            side = metrics.height() + pad
            corner = QRectF(pos.x() + (size.width() - grab.x()) * scale - side / 2, pos.y() - grab.y() * scale - side / 2, side, side)
            painter.setPen(QPen(QColor(THEME.border_checked_hover), 1))
            painter.setBrush(QColor(THEME.bg_primary))
            painter.drawEllipse(corner)
            painter.setPen(QColor(THEME.text_primary))
            painter.drawText(corner, Qt.AlignmentFlag.AlignCenter, str(len(move["rows"])))
        chip = QRectF(
            pos.x() + THEME.space_xl,
            pos.y() + (size.height() - grab.y()) * scale + THEME.space_md,
            metrics.horizontalAdvance(text) + 2 * pad,
            metrics.height() + pad,
        )
        painter.setPen(QPen(with_alpha(THEME.mark_photo, 55), 1))
        painter.setBrush(with_alpha(THEME.mark_shadow, 225))
        painter.drawRoundedRect(chip, _CHIP_RADIUS, _CHIP_RADIUS)
        painter.setPen(QColor(THEME.text_primary))
        painter.drawText(chip, Qt.AlignmentFlag.AlignCenter, text)
        painter.setBrush(Qt.BrushStyle.NoBrush)

    def selectionChanged(self, selected, deselected) -> None:  # noqa: N802
        super().selectionChanged(selected, deselected)
        # A mark reaches past its cell, where Qt repaints nothing.
        self.viewport().update()

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        delegate = self.itemDelegate()
        if not isinstance(delegate, _FilmDelegate):
            return
        painter = QPainter(self.viewport())
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        # A scene's line and name chip lie on its frames' top border, over their selection mats.
        self._paint_marks(painter, delegate, event.rect())
        self._paint_scenes(painter, delegate, event.rect())
        self._paint_scene_move(painter)
        painter.end()

    def _paint_marks(self, painter: QPainter, delegate: _FilmDelegate, area: QRect) -> None:
        """A muted red mat over a selected frame's border, the lighter red on the open frame."""
        model, selection = self.model(), self.selectionModel()
        rows = {index.row() for index in selection.selectedIndexes()} if selection is not None else set()
        if self._scene_move is not None and self._scene_move["dragging"]:
            rows -= set(self._scene_move["rows"])
        # As wide on every side: the border around the picture, then one pixel at 90% and one at 80%; on
        # the left and right it reaches over the neighbors' border.
        rings = ((delegate._BORDER, 1.0), (delegate._BORDER + 1, 0.9), (delegate._BORDER + 2, 0.8))
        reach = rings[-1][0]
        marks = []
        for row in sorted(rows):
            index = model.index(row, 0)
            cell = self.visualRect(index)
            if cell.adjusted(-reach, 0, reach, 0).intersects(area):
                marks.append((delegate.is_active(index.data(Qt.ItemDataRole.UserRole) or {}), delegate._window(cell)))
        # The open frame's mat goes over a selected neighbor's.
        for active, window in sorted(marks, key=lambda mark: mark[0]):
            color = QColor(THEME.border_checked_hover if active else THEME.border_checked)
            inside = QPainterPath()
            inside.addRect(QRectF(window))
            for grow, alpha in rings:
                edge = QPainterPath()
                radius = _CHIP_RADIUS + 0.5 + grow - delegate._BORDER
                edge.addRoundedRect(QRectF(window).adjusted(-grow, -grow, grow, grow), radius, radius)
                color.setAlphaF(alpha)
                painter.fillPath(edge.subtracted(inside), color)
                inside = edge

    def _paint_scenes(self, painter: QPainter, delegate: _FilmDelegate, area: QRect) -> None:
        """A scene's line across the top border of each run of its frames on one row, its name in a chip at the run's
        start. A found scene's line is dashed, its chip outlined with a dashed edge and a dot of its color."""
        font = painter.font()
        font.setPixelSize(THEME.font_size_small)
        font.setBold(True)
        painter.setFont(font)
        metrics = painter.fontMetrics()
        height, pad = metrics.height(), THEME.space_sm
        for scene, _index, first, last in self.scene_runs_on_rows():
            if not first.united(last).adjusted(0, -height, 0, 0).intersects(area):
                continue
            band = delegate.band(first)
            color = QColor(scene_color(scene[0]))
            left, right = band.left(), delegate.band(last).right() + 1
            if not scene[3]:
                painter.fillRect(left, band.top(), right - left, delegate._BORDER, color)
            else:
                # Dashes on a fixed grid line up across the runs of a row.
                dash, period = 2 * delegate._BORDER, 3 * delegate._BORDER
                for x in range(left - left % period, right, period):
                    start, end = max(x, left), min(x + dash, right)
                    if end > start:
                        painter.fillRect(start, band.top(), end - start, delegate._BORDER, color)
            dot = height // 3 if scene[3] else 0
            lead = dot + pad if scene[3] else 0
            name = metrics.elidedText(scene[2], Qt.TextElideMode.ElideRight, last.right() - first.left() - 2 * pad - lead)
            chip = QRectF(band.left(), band.top() - height / 2, metrics.horizontalAdvance(name) + 2 * pad + lead, height)
            if scene[3]:
                painter.setPen(QPen(color, 1, Qt.PenStyle.DashLine))
                painter.setBrush(QColor(THEME.bg_dark))
                painter.drawRoundedRect(chip.adjusted(0.5, 0.5, -0.5, -0.5), _CHIP_RADIUS, _CHIP_RADIUS)
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(color)
                painter.drawEllipse(QRectF(chip.left() + pad, chip.center().y() - dot / 2, dot, dot))
                painter.setPen(QColor(THEME.text_primary))
                painter.drawText(chip.adjusted(pad + lead, 0, 0, 0), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, name)
                continue
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(color)
            painter.drawRoundedRect(chip, _CHIP_RADIUS, _CHIP_RADIUS)
            luma = 0.299 * color.red() + 0.587 * color.green() + 0.114 * color.blue()
            painter.setPen(QColor(THEME.bg_dark if luma > 140 else THEME.text_primary))
            painter.drawText(chip, Qt.AlignmentFlag.AlignCenter, name)
        painter.setBrush(Qt.BrushStyle.NoBrush)


class _FilmStripView(_FilmGridView):
    """The Film Strip: one row of film as tall as the view, scrolling sideways."""

    # The space above and below the film lies inside the cells, as the Light Table's row gap does,
    # so a scene's name chip on the film's top edge fits.
    STRIP_GAP = 2 * THEME.space_xl
    DRAG_SELECTS = True

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        # The film runs off the left and right edges.
        self.setViewportMargins(0, 0, 0, 0)
        self.setFlow(QListView.Flow.LeftToRight)
        self.setWrapping(False)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        # Always on, so the viewport height that sizes the frames never changes as the row grows.
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.setHorizontalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self._wheel_axis: Optional[str] = None

    def _table_qss(self, color: str) -> str:
        return (
            super()._table_qss(color)
            + f"QScrollBar:horizontal {{ background: {color}; }}"
            + "QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal { background: none; }"
        )

    def slim_cell(self, index: QModelIndex) -> QSize:
        height, border = self.viewport().height(), _FilmDelegate._BORDER
        return QSize(round((height - self.STRIP_GAP - 2 * border) * self._layout_aspect(index)) + border, height)

    def cell_for_width(self, vw: int) -> int:
        return max(1, int((self.viewport().height() - self.STRIP_GAP - 1) / self._aspect))

    def _apply_cell(self, cell: int) -> None:
        super()._apply_cell(cell)
        self.horizontalScrollBar().setSingleStep(max(1, cell // 4))

    def _place_scene_runs(self) -> None:
        # One row: scenes keep their order, without the Light Table's blocks.
        return

    def wheelEvent(self, event) -> None:
        """The wheel moves the row a frame a notch; a trackpad moves it by its own pixels."""
        bar = self.horizontalScrollBar()
        phase = event.phase()
        if phase in (Qt.ScrollPhase.ScrollBegin, Qt.ScrollPhase.NoScrollPhase):
            self._wheel_axis = None
        pixel = event.pixelDelta()
        delta = pixel if not pixel.isNull() else event.angleDelta()
        # A trackpad swipe drifts off its axis: a gesture keeps the axis it starts on, momentum included.
        axis = self._wheel_axis
        if axis is None and (delta.x() or delta.y()):
            axis = "x" if abs(delta.x()) >= abs(delta.y()) else "y"
            if phase != Qt.ScrollPhase.NoScrollPhase:
                self._wheel_axis = axis
        step = delta.x() if axis == "x" else delta.y()
        if not pixel.isNull():
            bar.setValue(bar.value() - step)
        else:
            frame = self.gridSize().width() if self.gridSize().isValid() else self.viewport().height()
            bar.setValue(bar.value() - int(round(step / 120 * frame)))
        if phase == Qt.ScrollPhase.ScrollEnd:
            self._wheel_axis = None
        event.accept()


# The Roll row: the widest its roll name and tally get, and the filter box's width below which
# Save as Roll moves to the ⋮ menu.
_ROLL_TITLE_MAX_WIDTH = 280
_FILTER_ROOM = 220


class FileBrowser(QWidget):
    """
    Asset management panel for loading and selecting images.
    """

    file_selected = pyqtSignal(str)
    library_requested = pyqtSignal(bool)  # reveal the library (arg: import a first roll if unset)
    light_table_opened = pyqtSignal()

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
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        icon_size = QSize(BAR_ICON_SIZE, BAR_ICON_SIZE)
        btn_height = BAR_BUTTON_HEIGHT

        self.apply_btn = QToolButton()
        self.apply_btn.setIcon(qta.icon("fa5s.clone", color=THEME.text_primary))
        self.apply_btn.setToolTip(tr("Apply settings from the current frame to selected frames or the whole roll"))
        self.apply_btn.clicked.connect(self._open_apply_dialog)

        # Sheet filter dropdown
        self.sheet_btn = QToolButton()
        self.sheet_btn.setIcon(qta.icon("fa5s.filter", color=THEME.text_primary))
        self.sheet_btn.setToolTip(tr("Sheet filter: show frames by their keep and reject marks"))
        self.sheet_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        sheet_menu = QMenu(self.sheet_btn)
        self._sheet_group = QActionGroup(self)
        self._sheet_group.setExclusive(True)
        self.act_sheet_all = sheet_menu.addAction(tr("All Frames"))
        self.act_sheet_keepers = sheet_menu.addAction(tr("Keepers Only"))
        self.act_sheet_unrejected = sheet_menu.addAction(tr("Hide Rejected"))
        self.act_sheet_unmarked = sheet_menu.addAction(tr("Unmarked Only"))
        for act in (self.act_sheet_all, self.act_sheet_keepers, self.act_sheet_unrejected, self.act_sheet_unmarked):
            act.setCheckable(True)
            self._sheet_group.addAction(act)
        self.act_sheet_all.triggered.connect(lambda: self._apply_sheet_filter("all"))
        self.act_sheet_keepers.triggered.connect(lambda: self._apply_sheet_filter("keepers"))
        self.act_sheet_unrejected.triggered.connect(lambda: self._apply_sheet_filter("unrejected"))
        self.act_sheet_unmarked.triggered.connect(lambda: self._apply_sheet_filter("unmarked"))
        sheet_menu.addSeparator()
        self.act_advance_after_mark = sheet_menu.addAction(tr("Advance After Marking"))
        self.act_advance_after_mark.setCheckable(True)
        self.act_advance_after_mark.setChecked(bool(self.session.repo.get_global_setting("advance_after_mark", False)))
        self.act_advance_after_mark.setToolTip(tr("Marking a frame Keeper or Reject moves on to the next frame"))
        self.act_advance_after_mark.toggled.connect(lambda on: self.session.repo.save_global_setting("advance_after_mark", on))
        sheet_menu.setToolTipsVisible(True)
        self.sheet_btn.setMenu(sheet_menu)

        # The frames' own order; the Library's roll list has a Sort of its own.
        self.sort_btn = SortButton(
            (("name", tr("Name")), ("date", tr("Date")), ("scene", tr("Scene"))), tr("Sort the frames in the Film Strip")
        )
        self.act_sort_name, self.act_sort_date, self.act_sort_scene = (self.sort_btn.order_action(k) for k in ("name", "date", "scene"))
        self.act_sort_asc, self.act_sort_desc = self.sort_btn.ascending_action, self.sort_btn.descending_action
        self.act_sort_scene.setVisible(False)
        self.sort_btn.order_selected.connect(self._apply_sort_order)
        self.sort_btn.direction_selected.connect(self._apply_sort_direction)

        self.roll_settings_btn = QToolButton()
        self.roll_settings_btn.setIcon(qta.icon("mdi6.tag-outline", color=THEME.text_primary))
        self.roll_settings_btn.setToolTip(
            tooltip_with_shortcut(
                tr("Roll Settings: tag gear, film and process details on this frame, the selection or the whole roll."), "roll_settings"
            )
        )
        self.roll_settings_btn.clicked.connect(self._open_roll_settings_dialog)

        for btn in (self.sort_btn, self.sheet_btn):
            btn.setProperty("dropdown", True)
        # Icons alone: the tooltips name them, and the filter box keeps the room.
        for btn in (self.sort_btn, self.sheet_btn, self.roll_settings_btn, self.apply_btn):
            btn.setIconSize(icon_size)
            btn.setFixedHeight(btn_height)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
        # 1 while Save as Roll waits in the ⋮ menu for room in the row.
        self._roll_fit = 0
        self._save_roll_wanted = False

        saved_sort = self.session.repo.get_global_setting("file_sort_order") or "name"
        saved_desc = self.session.repo.get_global_setting("file_sort_descending") or False

        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText(tr("Filter: name, film:portra, iso:>=400…"))
        self.search_input.setToolTip(
            tr("Filter the frames: a word matches the file name, and terms such as film:portra iso:>=400 -rejected: match frame data.")
        )
        self.search_input.setClearButtonEnabled(True)
        self.search_input.addAction(
            qta.icon("fa5s.search", color=THEME.text_secondary),
            QLineEdit.ActionPosition.LeadingPosition,
        )
        # The elastic item in the row: _fit_roll_row reads its width.
        self.search_input.setMinimumWidth(40)
        self.search_input.setFixedHeight(btn_height)
        self.search_input.installEventFilter(self)
        # Same query text, wider net: the box filters what is loaded, and this runs it
        # against every library folder and opens what it finds.
        self.library_search_action = QAction(qta.icon("mdi.folder-search-outline", color=THEME.text_hint), "", self)
        self.search_input.addAction(self.library_search_action, QLineEdit.ActionPosition.TrailingPosition)
        self.library_search_action.setToolTip(tr("Search the whole library and load the matches."))
        self.regex_action = self._box_toggle("mdi6.regex", tr("Regex mode"))
        # Opt-in (Preferences); hidden until then. Mutually exclusive with regex/the
        # structured query language -- ranks this session's frames by meaning instead.
        self.semantic_action = self._box_toggle(
            "mdi.image-search-outline", tr("Search by meaning: describe what you are looking for, in place of field:value terms")
        )
        self.semantic_action.setVisible(False)

        self._refresh_progress = ""
        # First in the Roll row: the roll's name over the tally of its frames.
        self.roll_name_label = QLabel("")
        self.roll_name_label.setStyleSheet(
            f"color: {THEME.text_primary}; font-size: {THEME.font_size_header}px; font-weight: {THEME.weight_semibold};"
        )
        self.tally_label = ElidedLabel("")
        self.tally_label.setStyleSheet(f"color: {THEME.text_hint}; font-size: {THEME.font_size_small}px;")
        self.roll_title = QWidget()
        self.roll_title.setMaximumWidth(_ROLL_TITLE_MAX_WIDTH)
        title_col = QVBoxLayout(self.roll_title)
        title_col.setContentsMargins(THEME.space_sm, 0, THEME.space_sm, 0)
        title_col.setSpacing(0)
        title_col.addWidget(self.roll_name_label)
        title_col.addWidget(self.tally_label)
        self.roll_title.setVisible(False)

        self.list_view = _FilmStripView()
        self._thumbnail_delegate = _FilmDelegate(self.list_view, state=self.session.state)
        self._make_grid(self.list_view, self._thumbnail_delegate)
        # The Light Table; MainWindow lays it out in the canvas's place.
        self.light_table_view = _FilmGridView(
            target_cell=int(self.session.repo.get_global_setting("light_table_cell") or THUMB_CELL_DEFAULT)
        )
        self.light_table_view.cell_zoomed.connect(lambda cell: self.session.repo.save_global_setting("light_table_cell", cell))
        self._light_table_delegate = _LightTableDelegate(self.light_table_view, state=self.session.state)
        self._make_grid(self.light_table_view, self._light_table_delegate)
        self.light_table_view.setSelectionModel(self.list_view.selectionModel())
        self._film_key: Optional[tuple] = None
        self._film_geometry: Optional[FilmGeometry] = None
        # The rebate prints as the roll's mean print D-max, measured once per thumbnail pixmap.
        self._rebate_tone: Optional[tuple[int, int, int]] = None
        self._dmax_by_pixmap: dict[int, Optional[tuple[int, int, int]]] = {}
        self._rebate_timer = QTimer(self)
        self._rebate_timer.setSingleShot(True)
        self._rebate_timer.setInterval(300)
        self._rebate_timer.timeout.connect(self._measure_rebate)
        self._apply_sort_order(str(saved_sort), save=False)
        self._apply_sort_direction(bool(saved_desc), save=False)

        # Takes the strip's place when a filter hides every frame: a blank panel under a
        # full tally reads as a load failure.
        self.empty_label = hint_label()
        self.empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.empty_label.setTextFormat(Qt.TextFormat.RichText)
        self.empty_label.setOpenExternalLinks(False)
        self.empty_label.setVisible(False)
        self.empty_label.linkActivated.connect(lambda _: self._clear_frame_filters())

        # Shown in the right panel's Library tab; the strip reloads it and reads its state.
        self.library_tree = LibraryTree(self.controller)

        self.save_roll_btn = QToolButton()
        self.save_roll_btn.setIcon(qta.icon("fa5s.save", color=THEME.text_primary))
        self.save_roll_btn.setText(tr("Save as Roll…"))
        self.save_roll_btn.setToolTip(wrap_tooltip(tr("Save these frames as a roll you can open again.")))
        self.save_roll_btn.clicked.connect(self._on_save_roll_clicked)

        self.hot_folder_action = QAction(tr("Hot Folder"), self)
        self.hot_folder_action.setCheckable(True)
        self.hot_folder_action.setToolTip(tr("Hot Folder: load new images from the current folder as they appear"))
        # The ⋮ menu holds the switch; the row shows this while it is on, since a watched
        # folder keeps loading files. A click turns it off.
        self.hot_folder_btn = QToolButton()
        self.hot_folder_btn.setCheckable(True)
        self.hot_folder_btn.setIcon(qta.icon("fa5s.fire", color=THEME.text_hint, color_on=THEME.text_primary))
        self.hot_folder_btn.setToolTip(tr("Hot Folder is on: click to stop loading new images from the current folder."))
        self.hot_folder_btn.clicked.connect(lambda: self.hot_folder_action.setChecked(False))
        self.hot_folder_btn.setVisible(False)

        for btn in (self.save_roll_btn, self.hot_folder_btn):
            btn.setIconSize(icon_size)
            btn.setFixedHeight(btn_height)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.save_roll_btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.save_roll_btn.setVisible(False)

        self.roll_menu_btn = QToolButton()
        self.roll_menu_btn.setIcon(qta.icon("fa5s.ellipsis-v", color=THEME.text_primary))
        self.roll_menu_btn.setIconSize(icon_size)
        self.roll_menu_btn.setFixedHeight(btn_height)
        self.roll_menu_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.roll_menu_btn.setToolTip(tr("Add frames, the Hot Folder, and start, close or reset the roll"))
        self.roll_menu_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.roll_menu_btn.setObjectName("toolbar_overflow")
        roll_menu = QMenu(self.roll_menu_btn)
        roll_menu.setToolTipsVisible(True)
        roll_menu.addAction(tr("Add Files…")).triggered.connect(self.prompt_add_files)
        roll_menu.addAction(tr("Add Folder…")).triggered.connect(self.prompt_add_folder)
        self.save_roll_action = roll_menu.addAction(label_with_shortcut(tr("Save as Roll…"), "save_as_roll"))
        self.save_roll_action.triggered.connect(self._on_save_roll_clicked)
        self.update_thumbnails_action = roll_menu.addAction("")
        self.update_thumbnails_action.triggered.connect(self._on_update_thumbnails_clicked)
        self._on_thumbnail_refresh_state_changed(False)
        roll_menu.addAction(self.hot_folder_action)
        self.unload_action = roll_menu.addAction(tr("Unload…"))
        self.unload_action.triggered.connect(self._on_unload_clicked)
        roll_menu.addSeparator()
        roll_menu.addAction(tr("New Roll…")).triggered.connect(self._on_clear_all)
        self.close_roll_action = roll_menu.addAction(label_with_shortcut(tr("Close Roll…"), "close_roll"))
        self.close_roll_action.triggered.connect(lambda: _close_roll(self, self.controller))
        roll_menu.aboutToShow.connect(self._sync_close_roll_action)
        roll_menu.addAction(label_with_shortcut(tr("Reset Roll to Defaults…"), "reset_roll")).triggered.connect(self._on_reset_roll)
        roll_menu.addSeparator()
        roll_menu.addAction(tr("Film Strip Guide…")).triggered.connect(self._show_guide)
        self.roll_menu_btn.setMenu(roll_menu)

        # The toolbar's Roll mode shows this row: what the strip holds, the box that filters the
        # frames and searches the library, how they are sorted and shown, Apply and the ⋮ menu.
        self.roll_row = QWidget()
        roll_layout = QHBoxLayout(self.roll_row)
        roll_layout.setContentsMargins(0, 0, 0, 0)
        roll_layout.setSpacing(THEME.space_sm)
        self._title_separator = self._create_separator()
        self._title_separator.setVisible(False)
        for widget in (self.roll_title, self.save_roll_btn, self.hot_folder_btn, self._title_separator):
            roll_layout.addWidget(widget)
        roll_layout.addWidget(self.search_input, 1)
        roll_layout.addWidget(self.sort_btn)
        roll_layout.addWidget(self.sheet_btn)
        roll_layout.addWidget(self._create_separator())
        roll_layout.addWidget(self.roll_settings_btn)
        roll_layout.addWidget(self.apply_btn)
        roll_layout.addWidget(self.roll_menu_btn)

        layout.addWidget(self.list_view, 1)
        layout.addWidget(self.empty_label, 1)

        # Applied after list_view exists: the filter prunes the selection against the view.
        saved_sheet = self.session.repo.get_global_setting("sheet_filter") or "all"
        self._apply_sheet_filter(str(saved_sheet), save=False)

    def _connect_signals(self) -> None:
        self.library_tree.folder_roll_created.connect(self._maybe_suggest_gear)
        self.list_view.clicked.connect(self._on_item_clicked)
        self.list_view.doubleClicked.connect(self._on_item_double_clicked)
        self.list_view.customContextMenuRequested.connect(self._show_context_menu)
        self.light_table_view.doubleClicked.connect(self._open_from_light_table)
        self.light_table_view.found_scene_drop.connect(self._on_found_scene_drop)
        open_key = QShortcut(QKeySequence(Qt.Key.Key_Return), self.light_table_view)
        open_key.setContext(Qt.ShortcutContext.WidgetShortcut)
        open_key.activated.connect(lambda: self._open_from_light_table(self.light_table_view.currentIndex()))
        self.light_table_view.customContextMenuRequested.connect(lambda pos: self._show_context_menu(pos, self.light_table_view))
        self.list_view.selectionModel().selectionChanged.connect(self._on_selection_changed)
        self.hot_folder_action.toggled.connect(self._on_hot_folder_toggled)
        self.controller.thumbnail_refresh_state_changed.connect(self._on_thumbnail_refresh_state_changed)
        self.session.canvas_bg_changed.connect(self._sync_film)
        model = self.session.asset_model
        for signal in (model.dataChanged, model.layoutChanged, model.modelReset, model.rowsInserted, model.rowsRemoved):
            signal.connect(lambda *_args: self._rebate_timer.start())
        self.controller.thumbnail_refresh_progress.connect(self._on_thumbnail_refresh_progress)
        self.session.state_changed.connect(self.sync_ui)
        self.session.files_changed.connect(self._on_files_changed)
        self.controller.first_scene_created.connect(lambda: self._apply_sort_order("scene"))
        self.controller.scenes_detected.connect(lambda: self._apply_sort_order("scene"))
        self.controller.thumbnail_activity_changed.connect(self._thumbnail_delegate.set_activity)
        self.controller.thumbnail_activity_changed.connect(self._light_table_delegate.set_activity)
        # Unloading the last frame leaves nothing to show, so fall back to the library rather
        # than an empty panel. Never prompts: the user asked to unload, not to load.
        self.session.session_emptied.connect(lambda: self.library_requested.emit(False))
        self.search_input.textChanged.connect(lambda _: self.filter_timer.start())
        self.search_input.returnPressed.connect(self.search_library)
        self.regex_action.toggled.connect(lambda _: self.filter_timer.start())
        self.semantic_action.toggled.connect(self._on_semantic_toggled)
        self.library_search_action.triggered.connect(self.search_library)

        # Delete unloads the selected frames, scoped to the thumbnail list so it does not fire
        # while typing in the filter box or editing elsewhere.
        for view in (self.list_view, self.light_table_view):
            del_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Delete), view)
            del_shortcut.setContext(Qt.ShortcutContext.WidgetShortcut)
            del_shortcut.activated.connect(self._on_delete_key)

    def _measure_rebate(self) -> None:
        # Frames in roll order, so a sort or a filter keeps the tone.
        state = self.session.state
        files, tones = state.uploaded_files, []
        last = len(files) - 1
        for row in sorted({round(i * last / (_DMAX_SAMPLES - 1)) for i in range(_DMAX_SAMPLES)} if last >= 0 else ()):
            icon = state.thumbnails.get(asset_thumbnail_key(files[row]))
            pixmap = icon.pixmap(QSize(4096, 4096)) if isinstance(icon, QIcon) else None
            if pixmap is None or pixmap.isNull():
                continue
            key = pixmap.cacheKey()
            if key not in self._dmax_by_pixmap:
                if len(self._dmax_by_pixmap) > _FilmDelegate._CACHE_LIMIT:
                    self._dmax_by_pixmap.clear()
                self._dmax_by_pixmap[key] = print_dmax(pixmap.toImage())
            if (tone := self._dmax_by_pixmap[key]) is not None:
                tones.append(tone)
        mean = np.mean(tones, axis=0) if tones else None
        rebate = None if mean is None else (int(round(mean[0])), int(round(mean[1])), int(round(mean[2])))
        if rebate != self._rebate_tone:
            self._rebate_tone = rebate
            self._sync_film()

    def _sync_film(self) -> None:
        """The film of the strip and the Light Table, from the active frame; the gear library is read
        only when an input changes."""
        state = self.session.state
        files = state.uploaded_files
        asset = files[state.selected_file_idx] if 0 <= state.selected_file_idx < len(files) else {}
        config = state.config
        meta = config.metadata
        half_frame = bool(self.controller.half_frame_mode_for_roll(state.active_roll_id)) if state.active_roll_id else False
        table = table_color(state.canvas_bg_index)
        key = (
            (meta.film_stock_id, meta.film, meta.film_manufacturer, meta.film_color_type, meta.format, meta.format_other),
            (config.process.process_mode, config.process.positive_source, config.exposure.paper_black),
            (half_frame, asset.get("half")),
            table,
            self._rebate_tone,
        )
        if key == self._film_key:
            return
        self._film_key = key
        try:
            library = self.controller.load_gear_library()
        except Exception:
            library = None
        film = roll_film(asset, config, library, half_frame, self._rebate_tone)
        geometry = film[0]
        new_shape, self._film_geometry = geometry != self._film_geometry, geometry
        for delegate, view in ((self._thumbnail_delegate, self.list_view), (self._light_table_delegate, self.light_table_view)):
            delegate.set_film(*film)
            if new_shape:
                view.set_film(geometry)
            else:
                view.viewport().update()
            view.set_table_color(table)

    def _open_from_light_table(self, index) -> None:
        if index.isValid():
            self._activate_file(index)
            self.light_table_opened.emit()

    def _make_grid(self, view: ThumbnailGridView, delegate: _ThumbnailDelegate) -> None:
        view.setModel(self.session.asset_model)
        view.setItemDelegate(delegate)
        view.setViewMode(QListView.ViewMode.IconMode)
        view.setResizeMode(QListView.ResizeMode.Adjust)
        view.setSelectionMode(QListView.SelectionMode.ExtendedSelection)
        view.setAlternatingRowColors(False)
        view.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)

    def search_library(self) -> None:
        """Run the box's query against the library folders instead of the loaded roll."""
        # A keystroke just before Enter leaves the live-filter debounce pending, and it
        # would fire _apply_filter over whatever the hand-off loaded, putting back the
        # in-session query the hand-off clears. One action, one rebuild.
        self.filter_timer.stop()
        if self.semantic_action.isChecked():
            self.controller.request_library_semantic_search(self.search_input.text())
        else:
            self.controller.request_library_search(self.search_input.text())

    def _show_guide(self) -> None:
        from negpy.desktop.view.widgets.section_help_dialog import SectionHelpDialog

        SectionHelpDialog("frames", tr("Film Strip"), self, repo=self.session.repo).exec()

    def focus_search(self) -> None:
        self.search_input.setFocus()
        self.search_input.selectAll()

    def _on_files_changed(self) -> None:
        # A mark toggle can hide the active frame under a Sheet filter. Pruning then auto-advances
        # the selection to the next visible frame.
        if self.session.asset_model.sheet_filter != "all":
            self._prune_selection_to_visible()
        self._sync_sort_menu()
        self.sync_ui()

    def _on_unload_clicked(self) -> None:
        """Same as the context menu's Unload…: the selection, never the whole roll."""
        self._on_remove_from_menu()

    def _sync_close_roll_action(self) -> None:
        state = self.session.state
        self.close_roll_action.setText(label_with_shortcut(close_roll_label(state), "close_roll"))
        self.close_roll_action.setEnabled(bool(state.uploaded_files))

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
        name, ok = QInputDialog.getText(self, tr("Save as Roll"), tr("Name:"))
        name = name.strip()
        if not ok or not name:
            return
        if not is_valid_preset_name(name):
            warn_invalid_roll_name(self, tr("Save as Roll"))
            return
        if self.controller.create_roll_from_session(name):
            self.library_tree.reload()
            self._update_tally()

    def _update_unload_button(self) -> None:
        on_selection = not self.session.selection_is_open_frame()
        self.unload_action.setText(tr("Unload Selected…") if on_selection else tr("Unload…"))

    def sync_ui(self) -> None:
        """Updates list selection to match session state."""
        model = self.session.asset_model
        selection_model = self.list_view.selectionModel()
        self.semantic_action.setVisible(self.session.state.semantic_search_enabled)
        if not self.session.state.semantic_search_enabled and self.semantic_action.isChecked():
            self.semantic_action.setChecked(False)  # reverts to the plain filter via _on_semantic_toggled
        self.library_tree.sync_ui()
        self._update_unload_button()
        self._update_tally()
        self._update_empty_state()

        current_actual = {
            model.display_to_actual(idx.row()) for idx in selection_model.selectedIndexes() if model.display_to_actual(idx.row()) >= 0
        }
        target_actual = set(self.session.state.selected_indices)

        # The open frame's mat follows the session, not the selection.
        self.list_view.viewport().update()
        self._sync_film()
        self.light_table_view.viewport().update()

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
        if self.semantic_action.isChecked():
            embedding = self.controller.embed_search_query(text)
            self.session.asset_model.set_semantic_query(embedding)
            self._set_search_error(bool(text) and embedding is None)
            self._prune_selection_to_visible()
            self.sync_ui()
            return

        if self.session.asset_model.semantic_query_active:
            self.session.asset_model.set_semantic_query(None)
        regex = self.regex_action.isChecked()
        ok = self.session.asset_model.set_filter(text, regex)
        self._set_search_error(not ok)
        if ok:
            self._prune_selection_to_visible()
            self.sync_ui()

    def _on_semantic_toggled(self, checked: bool) -> None:
        self.regex_action.setEnabled(not checked)
        self.search_input.setPlaceholderText(tr("Describe the photo to find…") if checked else tr("Filter: name, film:portra, iso:>=400…"))
        self.filter_timer.start()

    def _set_search_error(self, error: bool) -> None:
        if error:
            self.search_input.setStyleSheet(f"border: 1px solid {THEME.error};")
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
        model = self.session.asset_model
        model.set_sheet_filter(mode)
        mode = model.sheet_filter
        self.act_sheet_all.setChecked(mode == "all")
        self.act_sheet_keepers.setChecked(mode == "keepers")
        self.act_sheet_unrejected.setChecked(mode == "unrejected")
        self.act_sheet_unmarked.setChecked(mode == "unmarked")
        # A filter that hides frames takes the chosen-item chip (QSS [filtering]).
        self.sheet_btn.setProperty("filtering", mode != "all")
        self.sheet_btn.style().unpolish(self.sheet_btn)
        self.sheet_btn.style().polish(self.sheet_btn)
        if save:
            self.session.repo.save_global_setting("sheet_filter", mode)
        self._prune_selection_to_visible()
        self.sync_ui()

    def _active_frame_filters(self) -> list:
        """Names of the filters currently hiding frames, in the order they are applied."""
        model = self.session.asset_model
        names = []
        if model.filter_text:
            names.append(tr("search"))
        if model.sheet_filter == "keepers":
            names.append(tr("Keepers"))
        elif model.sheet_filter == "unrejected":
            names.append(tr("Hide Rejected"))
        elif model.sheet_filter == "unmarked":
            names.append(tr("Unmarked Only"))
        return names

    def _on_thumbnail_refresh_progress(self, text: str) -> None:
        self._refresh_progress = text
        self._update_tally()

    def _update_tally(self) -> None:
        files = self.session.state.uploaded_files
        if not files:
            self.roll_title.setVisible(False)
            self._title_separator.setVisible(False)
            self._show_save_roll(False)
            return
        keepers = sum(1 for f in files if f.get("keeper"))
        rejected = sum(1 for f in files if f.get("excluded"))
        n = len(files)
        # The strip shows the model, the tally counts the session; the filter button's chip and the
        # filter box say which filter hides the rest.
        visible = self.session.asset_model.rowCount()
        text = tr("{visible} of {n} frames", visible=visible, n=n) if visible != n else count_of(n, "frame")
        if keepers:
            text += f" · {count_of(keepers, 'keeper')}"
        if rejected:
            text += f" · {count_of(rejected, 'rejected', 'rejected')}"
        selected = len(self.session.state.selected_indices)
        if selected > 1:
            text += " · " + tr("{count} selected", count=selected)
        # First, so a narrow row elides the counts rather than the progress.
        if self._refresh_progress:
            text = f"{self._refresh_progress} · {text}"
        roll_name = self._active_roll_name()
        # Frames that are not one roll get named as what they are: an edit here reaches the
        # roll each frame came from, which a strip that looks identical either way gives no sign of.
        self.roll_name_label.setText(roll_name or tr("No roll"))
        self.tally_label.setText(text)
        self.roll_title.setToolTip(
            ""
            if roll_name
            else wrap_tooltip(tr("These frames are not one roll, so an edit here also shows in the roll the frame came from."))
        )
        self.roll_title.setVisible(True)
        self._title_separator.setVisible(True)
        self._show_save_roll(not roll_name)

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
            named = filters[0] if len(filters) == 1 else tr("{value} and {value1}", value=filters[0], value1=filters[1])
            self.empty_label.setText(tr('No frames match the {named} filter.<br><a href="#clear">Show all frames</a>', named=named))
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
        self.hot_folder_btn.setChecked(checked)
        self.hot_folder_btn.setVisible(checked)
        if checked:
            self.scan_timer.start()
        else:
            self.scan_timer.stop()

    def _box_toggle(self, icon_name: str, tooltip: str) -> QAction:
        """A switch inside the filter box, its icon lit while on."""

        def icon(on: bool) -> QIcon:
            return qta.icon(icon_name, color=THEME.text_primary if on else THEME.text_hint, color_disabled=THEME.text_muted)

        action = QAction(icon(False), "", self)
        self.search_input.addAction(action, QLineEdit.ActionPosition.TrailingPosition)
        action.setCheckable(True)
        action.setToolTip(tooltip)
        action.toggled.connect(lambda on: action.setIcon(icon(on)))
        return action

    def _show_save_roll(self, wanted: bool) -> None:
        self._save_roll_wanted = wanted
        self.save_roll_btn.setVisible(wanted and self._roll_fit == 0)

    def eventFilter(self, obj, event) -> bool:  # noqa: N802
        # Events still reach the filter while the browser is torn down, after its attributes are gone.
        if event.type() == QEvent.Type.Resize and obj is getattr(self, "search_input", None):
            self._fit_roll_row()
        return False

    def _fit_roll_row(self) -> None:
        """Save as Roll moves to the ⋮ menu while the filter box would be narrower than _FILTER_ROOM with it."""
        save = self.save_roll_btn.sizeHint().width() + THEME.space_sm if self._save_roll_wanted else 0
        room = self.search_input.width() - (save if self._roll_fit else 0)
        # A step back needs a margin: a width right at the edge must not flicker.
        fit = 0 if room >= _FILTER_ROOM + (THEME.space_2xl if self._roll_fit else 0) else 1
        if fit != self._roll_fit:
            self._roll_fit = fit
            self._show_save_roll(self._save_roll_wanted)

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

        removed = rolls.excluded_paths(self.session.repo, [folder_path])
        new_files = [p for p in FolderWatchService.scan_for_new_files(folder_path, existing) if not rolls.is_excluded_key(p, removed)]
        if new_files:
            self.controller.request_asset_discovery(new_files, hot_folder=True)

    def prompt_add_files(self) -> None:
        """Public entry point: also driven by the canvas empty state."""
        wildcards = get_supported_raw_wildcards()
        start_dir = last_open_folder(self.session.repo)
        files, _ = QFileDialog.getOpenFileNames(
            self,
            tr("Select Images"),
            start_dir,
            tr("Supported Images ({wildcards})", wildcards=wildcards),
        )
        if files:
            self.session.repo.save_global_setting("last_open_folder", os.path.dirname(files[0]))
            self.controller.request_asset_discovery(files, auto_open=True, announce_rgb=True)

    def prompt_add_folder(self) -> None:
        """Public entry point: also driven by the canvas empty state."""
        start_dir = last_open_folder(self.session.repo)
        folder = QFileDialog.getExistingDirectory(self, tr("Select Folder"), start_dir)
        if folder:
            self.session.repo.save_global_setting("last_open_folder", os.path.dirname(folder))
            self.open_or_browse(folder)

    def open_or_browse(self, folder: str) -> None:
        """Load a folder's images into the session, or import the roll folders under it
        into Library when it holds none directly."""
        images, _ = folder_counts(folder)
        if images:
            self.controller.request_asset_discovery([folder], auto_open=True, announce_rgb=True)
        elif self.library_tree.import_subfolders(folder):
            self.library_requested.emit(False)

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

    def _show_context_menu(self, pos, view=None) -> None:
        view = view or self.list_view
        index = view.indexAt(pos)
        if not index.isValid():
            # Empty space carries the session-level tools, so they stay reachable without travelling
            # back to the toolbar at the top of the panel, and are discoverable at all when the
            # session is empty.
            self._build_session_menu().exec(view.viewport().mapToGlobal(pos))
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

        menu = self.build_frame_menu()
        menu.exec(view.viewport().mapToGlobal(pos))

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
        # The dialog's fields are the source, so the active frame is a target in both scopes
        # (apply_preset_fields, not sync_selected_settings).
        visible = self.session.asset_model.visible_actual_indices()
        sel_count = len([i for i in set(state.selected_indices) if i in visible])
        return RollSettingsDialog(
            self, state.config, GearProfiles.load_library(), sel_count=sel_count, roll_count=len(visible), repo=self.session.repo
        )

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
        from a folder-name match against the gear library, never applied without the user
        pressing Apply. Silent when nothing matches -- checked before
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
            self.update_thumbnails_action.setText(tr("Cancel Thumbnail Update"))
            self.update_thumbnails_action.setToolTip(tr("Cancel Thumbnail Update: stop the background refresh in progress"))
        else:
            self.update_thumbnails_action.setText(tr("Update Thumbnails"))
            self.update_thumbnails_action.setToolTip(tr("Update Thumbnails: re-render every out-of-date thumbnail in the roll"))

    def _build_session_menu(self) -> QMenu:
        """Mirrors the panel toolbar's add/clear tools, for a right click on empty space."""
        icon_color = THEME.text_primary
        menu = QMenu(self)
        menu.addAction(qta.icon("fa5s.file-import", color=icon_color), tr("Add Files…")).triggered.connect(self.prompt_add_files)
        menu.addAction(qta.icon("fa5s.folder-plus", color=icon_color), tr("Add Folder…")).triggered.connect(self.prompt_add_folder)
        menu.addSeparator()
        clear = menu.addAction(qta.icon("fa5s.times-circle", color=icon_color), close_roll_label(self.session.state))
        clear.triggered.connect(lambda: _close_roll(self, self.controller))
        clear.setEnabled(bool(self.session.state.uploaded_files))
        return menu

    def build_frame_menu(self) -> QMenu:
        state = self.session.state
        # The selection, not the open frame, once it is anything but the open frame alone.
        on_selection = not self.session.selection_is_open_frame()
        multi = len(state.selected_indices) > 1

        menu = QMenu(self)
        if on_selection:
            menu.addAction(tr("Export Selected Frames")).triggered.connect(lambda: self.controller.request_export_selected())
        else:
            menu.addAction(tr("Export Current Frame")).triggered.connect(lambda: self.controller.request_export())
        menu.addSeparator()
        menu.addAction(label_with_shortcut(tr("Copy Settings"), "copy")).triggered.connect(self.session.copy_settings)
        menu.addAction(label_with_shortcut(tr("Copy Settings + Bounds"), "copy_with_bounds")).triggered.connect(
            self.session.copy_settings_with_bounds
        )
        act_paste = menu.addAction(label_with_shortcut(tr("Paste Settings…"), "paste"))
        act_paste.triggered.connect(lambda: open_paste_dialog(self, self.controller))
        act_paste.setEnabled(state.clipboard is not None)
        targets = [i for i in (state.selected_indices or [state.selected_file_idx]) if 0 <= i < len(state.uploaded_files)]
        n = len(targets)
        if on_selection:
            menu.addAction(tr("Reset {value}…", value=count_of(n, "frame").title())).triggered.connect(
                lambda: _reset_selected(self, self.controller)
            )
        else:
            menu.addAction(tr("Reset Settings")).triggered.connect(self.session.reset_settings)
            act_roll = menu.addAction(label_with_shortcut(tr("Reset to Roll Settings"), "reset_to_roll"))
            act_roll.triggered.connect(self.controller.revert_frame_to_roll)
            act_roll.setEnabled(self.controller.can_revert_frame_to_roll())
            menu.addAction(label_with_shortcut(SIDECAR_LABEL, "load_sidecar")).triggered.connect(
                lambda: load_edit_from_sidecar(self, self.controller)
            )
        menu.addSeparator()
        act_keep = menu.addAction(tr("Keep {value}", value=count_of(n, "frame")) if on_selection else tr("Keep"))
        act_keep.setCheckable(True)
        act_keep.setChecked(bool(targets) and all(state.uploaded_files[i].get("keeper") for i in targets))
        act_keep.triggered.connect(lambda: self.session.toggle_mark("keeper"))
        act_reject = menu.addAction(tr("Reject {value}", value=count_of(n, "frame")) if on_selection else tr("Reject"))
        act_reject.setCheckable(True)
        act_reject.setChecked(bool(targets) and all(state.uploaded_files[i].get("excluded") for i in targets))
        act_reject.triggered.connect(lambda: self.session.toggle_mark("excluded"))
        menu.addSeparator()
        menu.addAction(tr("Apply Settings…")).triggered.connect(self._open_apply_dialog)
        menu.addAction(label_with_shortcut(tr("Sync Bounds…"), "sync_bounds")).triggered.connect(
            lambda: open_sync_bounds_dialog(self, self.session)
        )
        if self.controller.thumbnail_refresh_running:
            menu.addAction(tr("Cancel Thumbnail Update")).triggered.connect(lambda: self.controller.cancel_thumbnail_refresh())
        else:
            menu.addAction(
                tr("Update {value}", value=count_of(n, "thumbnail")) if on_selection else tr("Update Thumbnail")
            ).triggered.connect(lambda: self.controller.request_thumbnail_refresh("selection"))
        menu.addAction(label_with_shortcut(tr("Reset Roll to Defaults…"), "reset_roll")).triggered.connect(
            lambda: _reset_roll(self, self.controller)
        )
        if on_selection:
            menu.addSeparator()
            if multi:
                menu.addAction(tr("Stitch Selected Frames")).triggered.connect(lambda: self.controller.request_stitch_selected())
                self._add_hdr_merge_action(menu, state)
            self._add_merge_to_tiff_action(menu, state, SCOPE_SELECTION)
        else:
            menu.addSeparator()
            menu.addAction(tr("Edit RGB Triplet…")).triggered.connect(self._on_edit_triplet)
            self._add_merge_to_tiff_action(menu, state, SCOPE_FRAME)
            active = state.uploaded_files[state.selected_file_idx] if 0 <= state.selected_file_idx < len(state.uploaded_files) else {}
            if active.get("stitch_paths"):
                menu.addAction(tr("Unstitch")).triggered.connect(lambda: self.controller.request_unstitch())
            if active.get("hdr_paths"):
                self._add_hdr_anchor_menu(menu, active)
                menu.addAction(tr("Unmerge Exposures")).triggered.connect(lambda: self.controller.request_unmerge_hdr())
            if active.get("diptych"):
                menu.addAction(tr("Unsplit Diptych…")).triggered.connect(self.prompt_undiptych)
            if active.get("half"):
                from negpy.services.assets.half_frame import base_hash

                base = base_hash(active.get("hash"))
                menu.addAction(tr("Adjust Split for This Frame…")).triggered.connect(
                    lambda: self._on_adjust_half_frame_split(active["path"], base)
                )
                if base and self.controller.half_frame_override(base) is not None:
                    menu.addAction(tr("Reset Split to Roll Default")).triggered.connect(lambda: self._on_reset_half_frame_split(base))
            if state.active_roll_id and active.get("path"):
                if rolls.is_forked(self.session.repo, state.active_roll_id, rolls.unforked_hash(active.get("hash") or "")):
                    menu.addAction(tr("Use the Shared Edit Again…")).triggered.connect(self.prompt_unfork_edit)
                elif len(rolls.rolls_containing_path(self.session.repo, active["path"])) >= 2:
                    menu.addAction(tr("Edit Independently in This Roll")).triggered.connect(
                        lambda: self.controller.request_fork_edit_for_roll()
                    )
        if state.active_roll_id:
            self._add_scene_menu(menu, [state.uploaded_files[i] for i in targets], multi)
        menu.addSeparator()
        if not on_selection:
            menu.addAction(label_with_shortcut(SHOW_LABEL, "show_in_folder")).triggered.connect(
                lambda: show_current_in_folder(self.controller)
            )
        menu.addAction(label_with_shortcut(TRASH_LABEL, "move_to_trash")).triggered.connect(
            lambda: move_selected_to_trash(self, self.controller)
        )
        menu.addSeparator()
        unload_label = tr("Unload Selected…") if on_selection else tr("Unload…")
        menu.addAction(unload_label).triggered.connect(self._on_remove_from_menu)
        if state.active_roll_id:
            menu.addAction(label_with_shortcut(tr("Remove from Roll…"), "remove_from_roll")).triggered.connect(
                lambda: remove_from_roll(self, self.controller)
            )
        return menu

    def _add_scene_menu(self, menu: QMenu, frames: List[dict], multi: bool) -> None:
        scenes = rolls.roll_scenes(self.session.repo, self.session.state.active_roll_id)
        # Scene id per frame, None for a frame outside every scene.
        of_frames = {(f.get("scene") or (0, None))[1] for f in frames}
        scene_menu = menu.addMenu(tr("Scene"))
        if multi:
            scene_menu.addAction(tr("Group as Scene…")).triggered.connect(self._on_group_as_scene)
        for scene_id, entry in scenes:
            if of_frames != {scene_id}:
                scene_menu.addAction(tr("Add to {value}", value=entry["name"])).triggered.connect(
                    lambda _=False, sid=scene_id: self.controller.request_add_to_scene(sid)
                )
        # Detect Scenes' groups waiting for review take the same two actions, after the scenes.
        found = self.session.state.found_scenes
        of_found = {(f.get("found_scene") or (0, None))[1] for f in frames}
        if found and not scene_menu.isEmpty():
            scene_menu.addSeparator()
        for k, entry in enumerate(found):
            if of_found != {k}:
                scene_menu.addAction(tr("Add to {value}", value=entry["name"])).triggered.connect(
                    lambda _=False, k=k: self.controller.add_to_found_scene(k)
                )
        in_scene, in_found = bool(of_frames - {None}), bool(of_found - {None})
        if in_scene or in_found:
            scene_menu.addAction(tr("Remove from Scene")).triggered.connect(lambda: self._remove_from_scene(in_scene, in_found))
        if len(of_frames) == 1 and None not in of_frames:
            (sid,) = of_frames
            scene_menu.addSeparator()
            scene_menu.addAction(tr("Analyze Scene…")).triggered.connect(lambda: self.controller.request_scene_analysis(sid))
            scene_menu.addAction(tr("Rename Scene…")).triggered.connect(lambda: self._on_rename_scene(sid))
            scene_menu.addAction(tr("Delete Scene…")).triggered.connect(lambda: prompt_delete_scene(self, self.controller, sid))
        scene_menu.setEnabled(not scene_menu.isEmpty())

    def _on_found_scene_drop(self, target: int) -> None:
        # A drag can follow a click before the debounce commits it; the drop moves what the view shows.
        if self.selection_timer.isActive():
            self.selection_timer.stop()
            self._commit_selection()
        if target < 0:
            self.controller.remove_from_found_scene()
        else:
            self.controller.add_to_found_scene(target)

    def _remove_from_scene(self, in_scene: bool, in_found: bool) -> None:
        if in_scene:
            self.controller.request_remove_from_scene()
        if in_found:
            self.controller.remove_from_found_scene()

    def _on_group_as_scene(self) -> None:
        default = rolls.next_scene_name(self.session.repo, self.session.state.active_roll_id)
        name, ok = QInputDialog.getText(self, tr("Group as Scene"), tr("Name:"), text=default)
        if ok and name.strip():
            self.controller.request_group_as_scene(name.strip())

    def _on_rename_scene(self, scene_id: str) -> None:
        current = dict(rolls.roll_scenes(self.session.repo, self.session.state.active_roll_id)).get(scene_id, {}).get("name", "")
        name, ok = QInputDialog.getText(self, tr("Rename Scene"), tr("Name:"), text=current)
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
        act = menu.addAction(tr("Merge Exposures (HDR)"))
        if mode == ProcessMode.BW:
            act.setEnabled(False)
            act.setToolTip(tr("Merging is for transparencies; black-and-white reversal film is not supported yet"))
            return
        act.triggered.connect(lambda: self.controller.request_hdr_merge_selected())

    def _add_merge_to_tiff_action(self, menu, state, scope: str) -> None:
        """Hidden when nothing in scope can merge; the confirm dialog explains a refused frame."""
        if not mergeable_in(state, scope):
            return
        menu.addAction(label_with_shortcut(LABELS[scope], ACTION_IDS[scope])).triggered.connect(
            lambda: merge_to_tiff(self, self.controller, scope)
        )

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
        sub = menu.addMenu(tr("Render Exposure"))
        auto = sub.addAction(tr("Bracket Middle (Auto)"))
        auto.setCheckable(True)
        auto.setChecked(not current)
        auto.triggered.connect(lambda: self.controller.set_hdr_anchor(""))
        sub.addSeparator()
        for path, stops in choices:
            name = os.path.basename(path)
            if stops == 0.0:
                label = tr("{value}   {stops:+.1f} EV  (as captured)", value=name, stops=stops)
            else:
                label = tr("{value}   {stops:+.1f} EV", value=name, stops=stops)
            act = sub.addAction(label)
            act.setCheckable(True)
            act.setChecked(path == current)
            act.triggered.connect(lambda _=False, p=path: self.controller.set_hdr_anchor(p))

    def _on_edit_triplet(self) -> None:
        open_triplet_dialog(self, self.session)

    def _on_remove_from_menu(self) -> None:
        count = len(self.session.state.selected_indices)
        if not self.session.selection_is_open_frame():
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
