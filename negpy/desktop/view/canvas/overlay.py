import math
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import qtawesome as qta
from PyQt6.QtCore import QPointF, QRectF, QSize, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QCursor, QImage, QKeySequence, QMouseEvent, QPainter, QPainterPath, QPen, QPixmap, QPolygonF, QShortcut
from PyQt6.QtWidgets import QWidget

from negpy.desktop.converters import ImageConverter
from negpy.desktop.session import AppState, ToolMode
from negpy.desktop.view.canvas.crop_guides import CropGuide, guide_shapes
from negpy.desktop.view.styles.theme import THEME
from negpy.features.geometry.logic import rotation_drag_angle, smooth_polyline, straighten_delta_degrees, translate_manual_crop_rect
from negpy.features.local.logic import _rasterise_mask
from negpy.features.retouch.models import HEAL_SIZE_REF

_LASSO_SNAP_PX = 12.0
_CROP_HANDLE_PX = 10.0
_CROP_MIN_SCREEN_PX = 24.0
# Drag distance required before an outside-the-rect press starts redrawing an
# existing crop (stray-click guard).
_CROP_REDRAW_SLOP_PX = 16.0
_ROT_HANDLE_RADIUS_PX = 11.0  # hit + draw radius of the edge rotation handles
_ROT_HANDLE_OFFSET_PX = 24.0  # gap between crop edge and handle center (outside the box)
_ROT_FINE_SENSITIVITY = 0.2  # Shift-drag sensitivity, like the crop-move fine drag
_ROTATION_GRID_DIVISIONS = 10
_GRID_ALPHA = 70
_MASK_RASTER_MAX = 384  # px cap for feathered overlay rasters


def grid_interior_fractions(divisions: int) -> List[float]:
    """Interior division fractions, e.g. 3 -> [1/3, 2/3], 10 -> [.1 .. .9]."""
    return [i / divisions for i in range(1, divisions)]


def _distance_to_polyline(pos: QPointF, pts: List[QPointF]) -> float:
    """Shortest screen distance from `pos` to a polyline (a single point counts)."""
    if not pts:
        return float("inf")
    if len(pts) == 1:
        return math.hypot(pos.x() - pts[0].x(), pos.y() - pts[0].y())
    best = float("inf")
    for a, b in zip(pts, pts[1:]):
        abx, aby = b.x() - a.x(), b.y() - a.y()
        apx, apy = pos.x() - a.x(), pos.y() - a.y()
        denom = abx * abx + aby * aby
        t = 0.0 if denom <= 1e-12 else max(0.0, min(1.0, (apx * abx + apy * aby) / denom))
        cx, cy = a.x() + t * abx, a.y() + t * aby
        best = min(best, math.hypot(pos.x() - cx, pos.y() - cy))
    return best


def feathered_mask_image(local_pts: List[Tuple[float, float]], w: int, h: int, sigma_px: float, color: QColor, max_alpha: int) -> QImage:
    """Tinted premultiplied-alpha QImage of a feathered polygon.

    `local_pts` in raster pixel coords; `sigma_px` in raster pixels.
    """
    norm = [(x / w, y / h) for x, y in local_pts]
    alpha = _rasterise_mask(norm, h, w, sigma_px)
    a = alpha * (max_alpha / 255.0)
    buf = np.empty((h, w, 4), dtype=np.uint8)
    buf[..., 0] = (color.red() * a).astype(np.uint8)
    buf[..., 1] = (color.green() * a).astype(np.uint8)
    buf[..., 2] = (color.blue() * a).astype(np.uint8)
    buf[..., 3] = (a * 255.0).astype(np.uint8)
    img = QImage(buf.data, w, h, w * 4, QImage.Format.Format_RGBA8888_Premultiplied)
    return img.copy()  # QImage-from-buffer does not own the memory


class CanvasOverlay(QWidget):
    """
    Transparent overlay for image interaction (crop, guides) and CPU rendering fallback.
    """

    clicked = pyqtSignal(float, float)
    crop_rect_changed = pyqtSignal(float, float, float, float, bool)
    crop_rotation_changed = pyqtSignal(float, bool)  # (fine_rotation_deg, persist)
    crop_confirmed = pyqtSignal()
    analysis_rect_changed = pyqtSignal(float, float, float, float, bool)
    analysis_confirmed = pyqtSignal()
    cursor_moved = pyqtSignal(float, float)
    cursor_left = pyqtSignal()
    lasso_completed = pyqtSignal(list)
    scratch_completed = pyqtSignal(list)
    local_mask_selected = pyqtSignal(int)
    local_mask_edited = pyqtSignal(int, list)  # (mask index, viewport-normalized vertices)
    local_vertex_deleted = pyqtSignal(int, int)  # (mask index, vertex index)
    straighten_completed = pyqtSignal(float)  # fine-rotation delta, stored convention (CCW+)

    def __init__(self, state: AppState, parent=None):
        super().__init__(parent)
        self.state = state
        self._qimage: Optional[QImage] = None
        self._current_size: Optional[Tuple[int, int]] = None
        self._content_rect: Optional[Tuple[int, int, int, int]] = None

        # Crop tool interaction state: corner-resize, interior move, edge-handle
        # rotate, or fresh draw (when the click lands outside the existing rect).
        self._crop_rect_norm: Optional[Tuple[float, float, float, float]] = None
        self._crop_drag_mode: Optional[str] = None  # "corner" | "move" | "rotate" | "draw"
        self._crop_anchor_screen: Optional[QPointF] = None
        self._crop_press_norm: Optional[Tuple[float, float]] = None
        self._crop_orig_rect: Optional[Tuple[float, float, float, float]] = None
        self._crop_draw_armed: bool = False
        self._crop_redraw_hint_shown: bool = False
        self._crop_draw_p1: Optional[QPointF] = None
        self._crop_draw_p2: Optional[QPointF] = None

        # Rotation-handle drag state (writes geometry.fine_rotation live).
        self._rotate_center: Optional[QPointF] = None
        self._rotate_press: Optional[QPointF] = None
        self._rotate_start_fine: float = 0.0
        self._rotate_current: Optional[float] = None
        self._rot_handle_pixmap: Optional[QPixmap] = None
        self._rotate_cursor: Optional[QCursor] = None

        # Freehand analysis-region interaction (transformed-normalized, like the crop rect).
        self._analysis_rect_norm: Optional[Tuple[float, float, float, float]] = None
        self._analysis_drag_mode: Optional[str] = None  # "move" | "draw"
        self._analysis_press_norm: Optional[Tuple[float, float]] = None
        self._analysis_orig_rect: Optional[Tuple[float, float, float, float]] = None
        self._analysis_draw_p1: Optional[QPointF] = None
        self._analysis_draw_p2: Optional[QPointF] = None

        self._tool_mode: ToolMode = ToolMode.NONE
        self._mouse_pos: QPointF = QPointF()

        # Lasso (polygon mask) interaction state
        self._lasso_pts: List[QPointF] = []
        self._lasso_drawing: bool = False

        # Scratch heal (open polyline) interaction state
        self._scratch_pts: List[QPointF] = []
        self._heal_drag_pts: List[QPointF] = []
        self._local_mask_screen_polys: List[List[QPointF]] = []
        self._mask_img_cache: Dict[tuple, QImage] = {}

        # Working screen points while a selected-mask vertex is dragged/added.
        self._local_edit_verts: Optional[List[QPointF]] = None
        self._local_drag_vertex: Optional[int] = None

        # Straighten tool: reference-line drag (press -> drag -> release applies).
        self._straighten_p1: Optional[QPointF] = None
        self._straighten_p2: Optional[QPointF] = None

        self.zoom_level: float = 1.0
        self.pan_x: float = 0.0
        self.pan_y: float = 0.0

        self._view_rect: QRectF = QRectF()

        self._buffer_overlay_ratio: float = 0.0
        self._buffer_overlay_visible: bool = False
        self._buffer_hide_timer = QTimer(self)
        self._buffer_hide_timer.setSingleShot(True)
        self._buffer_hide_timer.timeout.connect(self._hide_buffer_overlay)

        self._rotation_grid_visible: bool = False
        self._rotation_grid_timer = QTimer(self)
        self._rotation_grid_timer.setSingleShot(True)
        self._rotation_grid_timer.timeout.connect(self._hide_rotation_grid)

        self.setMouseTracking(True)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        # Widget-context shortcuts (Enter finish, Backspace take-back) need focus to
        # fire; clicking the canvas to draw grants it. No widget-scope Esc here — a
        # second Esc binding is ambiguous against the window-scope cancel_tool one
        # (only activatedAmbiguously fires) and the key goes dead mid-draw; that
        # handler owns the Esc ladder via cancel_in_progress().
        self.setFocusPolicy(Qt.FocusPolicy.ClickFocus)

        # Enter finishes an in-progress scratch/lasso polyline or confirms the
        # crop, same as double-click.
        for key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            sc = QShortcut(QKeySequence(key), self)
            sc.setContext(Qt.ShortcutContext.WidgetShortcut)
            sc.activated.connect(self._finish_draw_if_active)

        # Backspace steps back one click-point of the in-progress scratch polyline.
        self._backspace_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Backspace), self)
        self._backspace_shortcut.setContext(Qt.ShortcutContext.WidgetShortcut)
        self._backspace_shortcut.activated.connect(self.undo_last_scratch_point)

        if sys.platform == "win32":
            self.setAttribute(Qt.WidgetAttribute.WA_StaticContents, False)

    def set_transform(self, zoom: float, px: float, py: float) -> None:
        self.zoom_level = zoom
        self.pan_x = px
        self.pan_y = py
        self._recalc_view_rect()
        self.update()

    def show_analysis_buffer(self, ratio: float) -> None:
        self._buffer_overlay_ratio = max(0.0, min(ratio, 0.3))
        self._buffer_overlay_visible = True
        self._buffer_hide_timer.start(1000)
        self.update()

    def _hide_buffer_overlay(self) -> None:
        self._buffer_overlay_visible = False
        self.update()

    def show_rotation_grid(self) -> None:
        """Show the rule-of-thirds alignment grid while Fine Rot is adjusted; lingers 1s."""
        self._rotation_grid_visible = True
        self._rotation_grid_timer.start(1000)
        self.update()

    def _hide_rotation_grid(self) -> None:
        self._rotation_grid_visible = False
        self.update()

    def set_tool_mode(self, mode: ToolMode) -> None:
        self._tool_mode = mode
        if mode == ToolMode.CROP_MANUAL:
            self._crop_rect_norm = self.state.config.geometry.manual_crop_rect
        else:
            self._crop_rect_norm = None
            self._end_crop_drag()
            # Drop any contextual crop cursor so the widget inherits the tool default.
            self.unsetCursor()
        if mode == ToolMode.ANALYSIS_DRAW:
            self._analysis_rect_norm = self.state.config.process.analysis_rect
        else:
            self._analysis_rect_norm = None
            self._end_analysis_drag()
        if mode != ToolMode.LOCAL_DRAW:
            self._lasso_pts = []
            self._lasso_drawing = False
            self._end_local_edit()
        if mode != ToolMode.SCRATCH_PICK:
            self._scratch_pts = []
        if mode != ToolMode.DUST_PICK:
            self._heal_drag_pts = []
        if mode != ToolMode.STRAIGHTEN:
            self._straighten_p1 = None
            self._straighten_p2 = None
        self.update()

    def _end_local_edit(self) -> None:
        self._local_edit_verts = None
        self._local_drag_vertex = None

    def _end_crop_drag(self) -> None:
        self._crop_drag_mode = None
        self._crop_anchor_screen = None
        self._crop_press_norm = None
        self._crop_orig_rect = None
        self._crop_draw_armed = False
        self._crop_draw_p1 = None
        self._crop_draw_p2 = None
        self._rotate_center = None
        self._rotate_press = None
        self._rotate_current = None

    def _end_analysis_drag(self) -> None:
        self._analysis_drag_mode = None
        self._analysis_press_norm = None
        self._analysis_orig_rect = None
        self._analysis_draw_p1 = None
        self._analysis_draw_p2 = None

    def cancel_in_progress(self) -> bool:
        """First rung of the Esc ladder: clear in-progress tool geometry (lasso
        points, scratch polyline, straighten line). Returns True when something was
        cleared — the caller only puts the tool down when nothing was in progress."""
        if self._tool_mode == ToolMode.LOCAL_DRAW and self._lasso_drawing:
            self._lasso_pts = []
            self._lasso_drawing = False
            self.update()
            return True
        if self._tool_mode == ToolMode.SCRATCH_PICK and self._scratch_pts:
            self._scratch_pts = []
            self.update()
            return True
        if self._tool_mode == ToolMode.STRAIGHTEN and self._straighten_p1 is not None:
            self._straighten_p1 = None
            self._straighten_p2 = None
            self.update()
            return True
        return False

    def update_buffer(
        self,
        buffer: Optional[np.ndarray],
        color_space: str,
        content_rect: Optional[Tuple[int, int, int, int]] = None,
        gpu_size: Optional[Tuple[int, int]] = None,
        monitor_icc_bytes: Optional[bytes] = None,
    ) -> None:
        self._content_rect = content_rect
        if buffer is not None:
            self._qimage = ImageConverter.to_qimage(buffer, color_space, monitor_icc_bytes)
            self._current_size = (self._qimage.width(), self._qimage.height())
        else:
            self._qimage = None
            self._current_size = gpu_size

        if self._tool_mode == ToolMode.CROP_MANUAL and self._crop_drag_mode is None:
            self._crop_rect_norm = self.state.config.geometry.manual_crop_rect
        if self._tool_mode == ToolMode.ANALYSIS_DRAW and self._analysis_drag_mode is None:
            self._analysis_rect_norm = self.state.config.process.analysis_rect

        self._recalc_view_rect()
        self.update()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._recalc_view_rect()
        self.update()

    def _recalc_view_rect(self) -> None:
        old_rect = self._view_rect
        size = None
        if self._qimage:
            size = self._qimage.size()
        elif self._current_size:
            size = QSize(self._current_size[0], self._current_size[1])

        if size is None or size.isNull():
            self._view_rect = QRectF()
            return

        # No margins - use full widget dimensions
        w, h = self.width(), self.height()
        img_w, img_h = size.width(), size.height()

        scale_fit = min(w / img_w, h / img_h)
        total_scale = scale_fit * self.zoom_level

        final_w = img_w * total_scale
        final_h = img_h * total_scale

        center_x = (w / 2) + (self.pan_x * w)
        center_y = (h / 2) + (self.pan_y * h)

        self._view_rect = QRectF(center_x - (final_w / 2), center_y - (final_h / 2), final_w, final_h)
        self._remap_inflight_points(old_rect)

    def _remap_inflight_points(self, old: QRectF) -> None:
        """Repin in-progress screen points to the image when the view rect changes
        (zoom/pan/resize mid-draw), so the preview tracks the image, not the screen."""
        new = self._view_rect
        if old.isEmpty() or new.isEmpty() or old == new:
            return

        def remap(p: QPointF) -> QPointF:
            nx = (p.x() - old.x()) / old.width()
            ny = (p.y() - old.y()) / old.height()
            return QPointF(new.x() + nx * new.width(), new.y() + ny * new.height())

        if self._lasso_pts:
            self._lasso_pts = [remap(p) for p in self._lasso_pts]
        if self._scratch_pts:
            self._scratch_pts = [remap(p) for p in self._scratch_pts]
        if self._heal_drag_pts:
            self._heal_drag_pts = [remap(p) for p in self._heal_drag_pts]
        if self._local_edit_verts is not None:
            self._local_edit_verts = [remap(p) for p in self._local_edit_verts]
        if self._straighten_p1 is not None:
            self._straighten_p1 = remap(self._straighten_p1)
        if self._straighten_p2 is not None:
            self._straighten_p2 = remap(self._straighten_p2)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)

        parent_bg = getattr(self.parent(), "_bg_color", QColor("#050505"))
        if not getattr(self.parent(), "gpu_widget", None) or not self.parent().gpu_widget.isVisible():
            painter.fillRect(event.rect(), parent_bg)

        if sys.platform in ("darwin", "win32"):
            painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Source)
            if getattr(self.parent(), "gpu_widget", None) and self.parent().gpu_widget.isVisible():
                painter.fillRect(event.rect(), Qt.GlobalColor.transparent)
            else:
                painter.fillRect(event.rect(), parent_bg)
            painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)

        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

        if not self._view_rect.isEmpty():
            if self._qimage:
                painter.drawImage(self._view_rect, self._qimage)

        self._draw_ui(painter)

    def _draw_ui(self, painter: QPainter) -> None:
        if self._view_rect.isEmpty():
            return

        visible_rect = self._view_rect

        if self._tool_mode == ToolMode.CROP_MANUAL:
            self._draw_crop_tool(painter)

        if self._tool_mode == ToolMode.ANALYSIS_DRAW:
            self._draw_analysis_tool(painter)

        if (
            self._buffer_overlay_visible
            and self._buffer_overlay_ratio > 1e-4
            and self._tool_mode not in (ToolMode.CROP_MANUAL, ToolMode.ANALYSIS_DRAW)
        ):
            d = visible_rect
            margin_w = d.width() * self._buffer_overlay_ratio
            margin_h = d.height() * self._buffer_overlay_ratio
            inner = QRectF(d.x() + margin_w, d.y() + margin_h, d.width() - 2 * margin_w, d.height() - 2 * margin_h)

            painter.setBrush(QColor(0, 0, 0, 140))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRect(QRectF(d.x(), d.y(), d.width(), margin_h))
            painter.drawRect(QRectF(d.x(), inner.bottom(), d.width(), margin_h))
            painter.drawRect(QRectF(d.x(), inner.y(), margin_w, inner.height()))
            painter.drawRect(QRectF(inner.right(), inner.y(), margin_w, inner.height()))

            pen = QPen(QColor(THEME.accent_primary), 1, Qt.PenStyle.DashLine)
            pen.setCosmetic(True)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(pen)
            painter.drawRect(inner)

        if self._tool_mode != ToolMode.NONE and visible_rect.contains(self._mouse_pos):
            if self._tool_mode in (ToolMode.DUST_PICK, ToolMode.SCRATCH_PICK):
                self._draw_brush(painter)
            elif self._tool_mode != ToolMode.LOCAL_DRAW:
                pen = QPen(QColor(255, 255, 255, 80), 1, Qt.PenStyle.DotLine)
                pen.setCosmetic(True)
                painter.setPen(pen)
                painter.drawLine(QPointF(visible_rect.x(), self._mouse_pos.y()), QPointF(visible_rect.right(), self._mouse_pos.y()))
                painter.drawLine(QPointF(self._mouse_pos.x(), visible_rect.top()), QPointF(self._mouse_pos.x(), visible_rect.bottom()))

        if self.state.config.local.masks:
            self._draw_local_masks(painter)
        if self._tool_mode == ToolMode.LOCAL_DRAW:
            self._draw_lasso_in_progress(painter)
        if self._tool_mode in (ToolMode.DUST_PICK, ToolMode.SCRATCH_PICK):
            self._draw_placed_heals(painter)
        if self._tool_mode == ToolMode.SCRATCH_PICK:
            self._draw_scratch_in_progress(painter)
        if self._tool_mode == ToolMode.DUST_PICK:
            self._draw_heal_drag_in_progress(painter)
        if self._tool_mode == ToolMode.STRAIGHTEN:
            self._draw_straighten_line(painter)

        if self._rotation_grid_visible:
            self._draw_rotation_grid(painter, visible_rect)

        if getattr(self.state, "compare_mode", False):
            self._draw_compare_badge(painter, visible_rect)

    def _draw_grid(self, painter: QPainter, rect: QRectF, divisions: int, alpha: int) -> None:
        """Even N×N reference grid (interior lines only) across `rect`, screen-aligned."""
        pen = QPen(QColor(255, 255, 255, alpha), 1, Qt.PenStyle.SolidLine)
        pen.setCosmetic(True)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(pen)
        for f in grid_interior_fractions(divisions):
            x = rect.left() + rect.width() * f
            y = rect.top() + rect.height() * f
            painter.drawLine(QPointF(x, rect.top()), QPointF(x, rect.bottom()))
            painter.drawLine(QPointF(rect.left(), y), QPointF(rect.right(), y))

    def _draw_rotation_grid(self, painter: QPainter, visible_rect: QRectF) -> None:
        """Dense leveling grid shown while Fine Rot is adjusted (Lightroom-style)."""
        self._draw_grid(painter, visible_rect, _ROTATION_GRID_DIVISIONS, _GRID_ALPHA)

    def _draw_crop_guides(self, painter: QPainter, rect: QRectF) -> None:
        """Selected composition guide (thirds, phi, spiral, ...) inside the crop rect."""
        shapes = guide_shapes(CropGuide(self.state.crop_guide), rect.width(), rect.height(), self.state.crop_guide_orientation)
        if not shapes:
            return
        pen = QPen(QColor(255, 255, 255, _GRID_ALPHA), 1, Qt.PenStyle.SolidLine)
        pen.setCosmetic(True)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(pen)
        ox, oy = rect.left(), rect.top()
        for poly in shapes:
            painter.drawPolyline(QPolygonF([QPointF(ox + x, oy + y) for x, y in poly]))

    def _draw_compare_badge(self, painter: QPainter, visible_rect: QRectF) -> None:
        badge = QRectF(visible_rect.x() + 12, visible_rect.y() + 12, 78, 22)
        painter.setBrush(QColor(0, 0, 0, 170))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(badge, 4, 4)
        painter.setPen(QColor(THEME.accent_primary))
        painter.drawText(badge, Qt.AlignmentFlag.AlignCenter, "BEFORE")

    def _draw_brush(self, painter: QPainter) -> None:
        radius = self._brush_screen_radius(self.state.config.retouch.manual_dust_size)

        painter.setBrush(Qt.BrushStyle.NoBrush)
        pen = QPen(Qt.GlobalColor.white, 1.0, Qt.PenStyle.SolidLine)
        pen.setCosmetic(True)
        painter.setPen(pen)
        painter.drawEllipse(self._mouse_pos, radius, radius)

        accent = QColor(THEME.accent_primary)
        accent.setAlpha(60)
        painter.setBrush(accent)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(self._mouse_pos, radius, radius)

    def _brush_screen_radius(self, size: float) -> float:
        max_screen_dim = max(self._view_rect.width(), self._view_rect.height())
        return (size / (2.0 * HEAL_SIZE_REF)) * max_screen_dim

    def _preview_curve_path(self, pts: List[QPointF]) -> QPainterPath:
        """Smoothed path through the placed points plus the live cursor."""
        scr = [(p.x(), p.y()) for p in pts]
        if self._view_rect.contains(self._mouse_pos):
            scr.append((self._mouse_pos.x(), self._mouse_pos.y()))
        if len(scr) >= 3:
            scr = smooth_polyline(scr, closed=False)
        path = QPainterPath(QPointF(*scr[0]))
        for x, y in scr[1:]:
            path.lineTo(QPointF(x, y))
        return path

    def _draw_scratch_in_progress(self, painter: QPainter) -> None:
        if not self._scratch_pts:
            return
        width = max(1.5, 2.0 * self._brush_screen_radius(self.state.config.retouch.manual_dust_size))

        band = QColor(THEME.accent_primary)
        band.setAlpha(60)
        pen = QPen(band, width, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        path = self._preview_curve_path(self._scratch_pts)
        painter.drawPath(path)

        centerline = QPen(Qt.GlobalColor.white, 1.0, Qt.PenStyle.SolidLine)
        centerline.setCosmetic(True)
        painter.setPen(centerline)
        painter.drawPath(path)
        painter.setBrush(QColor(255, 255, 255, 180))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(self._scratch_pts[0], 3.0, 3.0)

    @staticmethod
    def _heal_region_path(pts: List[QPointF], radius: float) -> QPainterPath:
        """Union of brush dabs swept along `pts` — the mask silhouette of a heal
        stroke (capsule chain). Drawn as one filled region, never a stroked
        polyline: the line rendering reads jagged at drag-sample spacing."""
        region = QPainterPath()
        region.setFillRule(Qt.FillRule.WindingFill)
        step = max(1.0, radius * 0.5)
        for a, b in zip(pts, pts[1:]):
            seg = math.hypot(b.x() - a.x(), b.y() - a.y())
            n = max(1, int(seg / step))
            for i in range(n):
                t = i / n
                region.addEllipse(QPointF(a.x() + (b.x() - a.x()) * t, a.y() + (b.y() - a.y()) * t), radius, radius)
        region.addEllipse(pts[-1], radius, radius)
        return region

    def _draw_heal_drag_in_progress(self, painter: QPainter) -> None:
        """Translucent mask of the area being painted with the heal tool (click-drag)."""
        if len(self._heal_drag_pts) < 2:
            return
        radius = max(1.5, self._brush_screen_radius(self.state.config.retouch.manual_dust_size))
        fill = QColor(THEME.accent_primary)
        fill.setAlpha(60)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(fill)
        painter.drawPath(self._heal_region_path(self._heal_drag_pts, radius))

    def _draw_straighten_line(self, painter: QPainter) -> None:
        """Reference line being dragged with the straighten tool, plus a badge
        previewing the correction (display convention: positive = clockwise)."""
        if self._straighten_p1 is None or self._straighten_p2 is None:
            return
        p1, p2 = self._straighten_p1, self._straighten_p2

        pen = QPen(Qt.GlobalColor.white, 1.5, Qt.PenStyle.SolidLine)
        pen.setCosmetic(True)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawLine(p1, p2)
        painter.setBrush(QColor(255, 255, 255, 200))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(p1, 3.0, 3.0)
        painter.drawEllipse(p2, 3.0, 3.0)

        dx, dy = p2.x() - p1.x(), p2.y() - p1.y()
        if math.hypot(dx, dy) < 8.0:
            return
        delta = straighten_delta_degrees(dx, dy)
        vertical = abs(abs(math.degrees(math.atan2(dy, dx))) - 90.0) < 45.0
        label = f"{'Plumb' if vertical else 'Level'}  {-delta:+.2f}°"
        mid = QPointF((p1.x() + p2.x()) / 2.0, (p1.y() + p2.y()) / 2.0)
        badge = QRectF(mid.x() - 52, mid.y() - 26, 104, 22)
        painter.setBrush(QColor(0, 0, 0, 170))
        painter.drawRoundedRect(badge, 4, 4)
        painter.setPen(QColor(THEME.accent_primary))
        painter.drawText(badge, Qt.AlignmentFlag.AlignCenter, label)

    def _draw_placed_heals(self, painter: QPainter) -> None:
        """Thin outlines of committed heals (strokes + legacy spots) while a retouch tool is active."""
        conf = self.state.config.retouch
        if not (conf.manual_heal_strokes or conf.manual_dust_spots):
            return
        with self.state.metrics_lock:
            uv_grid = self.state.last_metrics.get("uv_grid")
        if uv_grid is None:
            return

        pen = QPen(QColor(THEME.accent_primary), 1.0, Qt.PenStyle.SolidLine)
        pen.setCosmetic(True)
        painter.setBrush(Qt.BrushStyle.NoBrush)

        for stroke in conf.manual_heal_strokes:
            points, size = stroke[0], stroke[1]
            screen_pts = [self._raw_to_screen(px, py, uv_grid) for px, py in points]
            radius = max(2.0, self._brush_screen_radius(size))
            if len(screen_pts) == 1:
                painter.setPen(pen)
                painter.drawEllipse(screen_pts[0], radius, radius)
            else:
                if len(screen_pts) >= 3:
                    screen_pts = [QPointF(x, y) for x, y in smooth_polyline([(p.x(), p.y()) for p in screen_pts], closed=False)]
                # Masked area only — no centerline, no outline.
                fill = QColor(THEME.accent_primary)
                fill.setAlpha(40)
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(fill)
                painter.drawPath(self._heal_region_path(screen_pts, radius))
                painter.setBrush(Qt.BrushStyle.NoBrush)

        painter.setPen(pen)
        for rx, ry, size in conf.manual_dust_spots:
            center = self._raw_to_screen(rx, ry, uv_grid)
            radius = max(2.0, self._brush_screen_radius(size))
            painter.drawEllipse(center, radius, radius)

    def _raw_to_screen(self, rx: float, ry: float, uv_grid: np.ndarray, buckets: int = 100) -> QPointF:
        """
        Inverse UV-grid lookup: raw-normalised (0-1) -> screen position.

        Two-stage nearest-neighbour: a coarse pass over a `buckets`-decimated grid
        locates the neighbourhood cheaply, then a full-resolution pass over that
        bucket's window pins the exact pixel. The coarse pass alone snapped results
        to bucket centres (± step/2 grid pixels ≈ 3-20px depending on preview size,
        magnified by zoom) — enough to draw a heal outline entirely off the healed
        spot even though the heal itself landed exactly where clicked.
        """
        h_uv, w_uv = uv_grid.shape[:2]
        step = max(1, h_uv // buckets)
        small = uv_grid[::step, ::step]
        dist = (small[..., 0] - rx) ** 2 + (small[..., 1] - ry) ** 2
        idx = int(np.argmin(dist))
        h_s, w_s = small.shape[:2]
        vy, vx = divmod(idx, w_s)

        # Refine: exact search across the coarse cell and its neighbours.
        py, px = vy * step, vx * step
        y0, y1 = max(0, py - step), min(h_uv, py + step + 1)
        x0, x1 = max(0, px - step), min(w_uv, px + step + 1)
        window = uv_grid[y0:y1, x0:x1]
        wdist = (window[..., 0] - rx) ** 2 + (window[..., 1] - ry) ** 2
        widx = int(np.argmin(wdist))
        wy, wx = divmod(widx, window.shape[1])

        nx = min((x0 + wx + 0.5) / w_uv, 1.0)
        ny = min((y0 + wy + 0.5) / h_uv, 1.0)
        return QPointF(
            self._view_rect.x() + nx * self._view_rect.width(),
            self._view_rect.y() + ny * self._view_rect.height(),
        )

    def _norm_to_screen(self, nx: float, ny: float) -> QPointF:
        """Transformed-image normalized coords (0-1) -> screen position."""
        return QPointF(
            self._view_rect.x() + nx * self._view_rect.width(),
            self._view_rect.y() + ny * self._view_rect.height(),
        )

    def _screen_to_norm(self, screen_pos: QPointF) -> Tuple[float, float]:
        """Screen position -> transformed-image normalized coords, clamped to 0-1."""
        if self._view_rect.isEmpty():
            return 0.0, 0.0
        nx = (screen_pos.x() - self._view_rect.x()) / self._view_rect.width()
        ny = (screen_pos.y() - self._view_rect.y()) / self._view_rect.height()
        return float(np.clip(nx, 0.0, 1.0)), float(np.clip(ny, 0.0, 1.0))

    def _crop_corner_screen_points(self) -> Optional[Dict[str, QPointF]]:
        """Screen positions of the crop rect's four corners.

        The rect is stored in the transformed (display) image's normalized coords — the
        same space it is drawn on — so it maps linearly through the view rect and stays a
        true axis-aligned rectangle. The box shown is exactly the box `CropProcessor`
        slices (no fine-rotation bounding-box inflation).
        """
        if self._crop_rect_norm is None or self._view_rect.isEmpty():
            return None
        x1, y1, x2, y2 = self._crop_rect_norm
        return {
            "tl": self._norm_to_screen(x1, y1),
            "tr": self._norm_to_screen(x2, y1),
            "br": self._norm_to_screen(x2, y2),
            "bl": self._norm_to_screen(x1, y2),
        }

    def _hit_test_crop_corner(self, pos: QPointF, corners: Dict[str, QPointF]) -> Optional[str]:
        for name, pt in corners.items():
            dx, dy = pos.x() - pt.x(), pos.y() - pt.y()
            if dx * dx + dy * dy <= _CROP_HANDLE_PX * _CROP_HANDLE_PX:
                return name
        return None

    def _crop_rotation_handle_points(self) -> Optional[Dict[str, QPointF]]:
        """Screen positions of the four rotation handles: one per crop-box edge,
        centered on the edge midpoint and offset outward (outside the crop area).
        Clamped to the widget so they stay reachable when the box touches an edge."""
        if self._crop_rect_norm is None or self._view_rect.isEmpty():
            return None
        x1, y1, x2, y2 = self._crop_rect_norm
        tl = self._norm_to_screen(x1, y1)
        br = self._norm_to_screen(x2, y2)
        cx, cy = (tl.x() + br.x()) / 2.0, (tl.y() + br.y()) / 2.0
        off = _ROT_HANDLE_OFFSET_PX
        pts = {
            "top": QPointF(cx, tl.y() - off),
            "bottom": QPointF(cx, br.y() + off),
            "left": QPointF(tl.x() - off, cy),
            "right": QPointF(br.x() + off, cy),
        }
        m = _ROT_HANDLE_RADIUS_PX + 2.0
        return {
            name: QPointF(
                float(np.clip(p.x(), m, self.width() - m)),
                float(np.clip(p.y(), m, self.height() - m)),
            )
            for name, p in pts.items()
        }

    def _hit_test_rotation_handle(self, pos: QPointF) -> bool:
        handles = self._crop_rotation_handle_points()
        if handles is None:
            return False
        for pt in handles.values():
            dx, dy = pos.x() - pt.x(), pos.y() - pt.y()
            if dx * dx + dy * dy <= _ROT_HANDLE_RADIUS_PX * _ROT_HANDLE_RADIUS_PX:
                return True
        return False

    def _rotation_cursor(self) -> QCursor:
        """A rotate-icon cursor for hovering the crop rotation handles."""
        if self._rotate_cursor is None:
            pix = qta.icon("fa5s.sync-alt", color="white").pixmap(22, 22)
            self._rotate_cursor = QCursor(pix)
        return self._rotate_cursor

    def _update_crop_hover_cursor(self, pos: QPointF) -> None:
        """Set a contextual cursor while hovering the crop tool (not dragging), so the
        available action — rotate, resize, move, or draw — is obvious without clicking."""
        if self._crop_rect_norm is None or self._view_rect.isEmpty():
            self.unsetCursor()
            return
        if self._hit_test_rotation_handle(pos):
            self.setCursor(self._rotation_cursor())
            return
        corners = self._crop_corner_screen_points()
        corner = self._hit_test_crop_corner(pos, corners) if corners else None
        if corner is not None:
            self.setCursor(Qt.CursorShape.SizeFDiagCursor if corner in ("tl", "br") else Qt.CursorShape.SizeBDiagCursor)
            return
        if corners is not None and QPolygonF(list(corners.values())).containsPoint(pos, Qt.FillRule.OddEvenFill):
            self.setCursor(Qt.CursorShape.OpenHandCursor)
            return
        # Outside the box: a fresh rectangle would be drawn — keep the crosshair.
        if self._view_rect.contains(pos):
            self.setCursor(Qt.CursorShape.CrossCursor)
        else:
            self.unsetCursor()

    def _rotation_handle_pixmap(self) -> QPixmap:
        if self._rot_handle_pixmap is None:
            # Rendered at 2x the drawn size so it stays crisp on hi-DPI screens.
            size = int((_ROT_HANDLE_RADIUS_PX - 3.0) * 4)
            self._rot_handle_pixmap = qta.icon("fa5s.sync-alt", color="white").pixmap(size, size)
        return self._rot_handle_pixmap

    def _oriented_target_ratio(self, dx: float, dy: float) -> Optional[float]:
        """Aspect-ratio constraint (w/h) for a crop drag, or None when unconstrained.

        The configured ratio string (e.g. "4:3") names a shape, not an orientation.
        After a 90° canvas rotation the crop box is portrait, so the same constraint
        must act as 3:4 — applying it as fixed landscape collapsed the box into a
        small sideways one on the first corner-adjust (#442). Orient by the drag's
        dominant axis: dragging taller than wide yields the portrait variant, wider
        yields landscape, so adjusting an existing box keeps its orientation.
        """
        ratio_str = self.state.config.geometry.autocrop_ratio
        if ratio_str == "Free":
            return None
        try:
            w_r, h_r = map(float, ratio_str.split(":"))
            ratio = w_r / h_r
        except (ValueError, ZeroDivisionError):
            return None
        if ratio <= 0.0:
            return None
        if abs(dy) > abs(dx):
            return min(ratio, 1.0 / ratio)
        return max(ratio, 1.0 / ratio)

    def _apply_aspect_and_min(self, anchor_screen: QPointF, cur_screen: QPointF) -> Tuple[float, float, float, float]:
        """Resizes a rect anchored at `anchor_screen` towards `cur_screen`, honoring the
        configured aspect ratio (if any) and a minimum rect size.

        Done entirely in screen-pixel space: normalised (0-1) fractions only equal
        physical aspect ratio when the displayed image is square, so applying a target
        ratio to normalised deltas distorts it by the image's actual width/height ratio.
        Screen pixels reflect the image as displayed, so ratios computed there are correct.
        """
        ax, ay = anchor_screen.x(), anchor_screen.y()
        nx, ny = cur_screen.x(), cur_screen.y()

        dx = nx - ax
        dy = ny - ay
        target_ratio = self._oriented_target_ratio(dx, dy)

        if target_ratio:
            if abs(dx) > abs(dy) * target_ratio:
                dx = abs(dy) * target_ratio * (1 if dx >= 0 else -1)
            else:
                dy = abs(dx) / target_ratio * (1 if dy >= 0 else -1)
            # Enforce the minimum size by scaling dx/dy up together so the
            # locked ratio survives even on the tiny first move of a drag
            # (clamping each axis independently here would distort the ratio).
            scale = max(_CROP_MIN_SCREEN_PX / max(abs(dx), 1e-6), _CROP_MIN_SCREEN_PX / max(abs(dy), 1e-6), 1.0)
            dx *= scale
            dy *= scale
        else:
            if abs(dx) < _CROP_MIN_SCREEN_PX:
                dx = _CROP_MIN_SCREEN_PX if dx >= 0 else -_CROP_MIN_SCREEN_PX
            if abs(dy) < _CROP_MIN_SCREEN_PX:
                dy = _CROP_MIN_SCREEN_PX if dy >= 0 else -_CROP_MIN_SCREEN_PX

        end_screen = QPointF(ax + dx, ay + dy)
        c1 = self._screen_to_norm(anchor_screen)
        c2 = self._screen_to_norm(end_screen)
        x1, x2 = sorted((c1[0], c2[0]))
        y1, y2 = sorted((c1[1], c2[1]))
        return (x1, y1, x2, y2)

    def _draw_crop_tool(self, painter: QPainter) -> None:
        if self._crop_drag_mode == "draw" and self._crop_draw_p1 is not None and self._crop_draw_armed:
            rect = QRectF(self._crop_draw_p1, self._crop_draw_p2 or self._crop_draw_p1).normalized().intersected(self._view_rect)
            pen = QPen(Qt.GlobalColor.white, 1, Qt.PenStyle.DashLine)
            pen.setCosmetic(True)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(pen)
            painter.drawRect(rect)
            self._draw_crop_guides(painter, rect)
            return

        corners = self._crop_corner_screen_points()
        if corners is None:
            return
        poly = QPolygonF([corners["tl"], corners["tr"], corners["br"], corners["bl"]])

        # Dim everything outside the crop rect: full view rect minus the crop polygon.
        outer = QPainterPath()
        outer.addRect(self._view_rect)
        inner = QPainterPath()
        inner.addPolygon(poly)
        painter.setBrush(QColor(0, 0, 0, 180))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawPath(outer.subtracted(inner))

        pen = QPen(Qt.GlobalColor.white, 1, Qt.PenStyle.DashLine)
        pen.setCosmetic(True)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(pen)
        painter.drawPolygon(poly)

        self._draw_crop_guides(painter, QRectF(corners["tl"], corners["br"]))

        handle_pen = QPen(Qt.GlobalColor.white, 1.5, Qt.PenStyle.SolidLine)
        handle_pen.setCosmetic(True)
        painter.setPen(handle_pen)
        painter.setBrush(QColor(THEME.accent_primary))
        for pt in corners.values():
            painter.drawRect(QRectF(pt.x() - 5, pt.y() - 5, 10, 10))

        self._draw_rotation_handles(painter, corners)

    def _draw_rotation_handles(self, painter: QPainter, corners: Dict[str, QPointF]) -> None:
        """Edge rotation handles (outside the crop box) + live angle badge while dragging."""
        handles = self._crop_rotation_handle_points()
        if handles is None:
            return

        # Thin ticks connecting each edge midpoint to its handle.
        edge_mids = {
            "top": QPointF((corners["tl"].x() + corners["tr"].x()) / 2.0, corners["tl"].y()),
            "bottom": QPointF((corners["bl"].x() + corners["br"].x()) / 2.0, corners["bl"].y()),
            "left": QPointF(corners["tl"].x(), (corners["tl"].y() + corners["bl"].y()) / 2.0),
            "right": QPointF(corners["tr"].x(), (corners["tr"].y() + corners["br"].y()) / 2.0),
        }
        tick_pen = QPen(QColor(255, 255, 255, 120), 1, Qt.PenStyle.SolidLine)
        tick_pen.setCosmetic(True)
        painter.setPen(tick_pen)
        for name, pt in handles.items():
            painter.drawLine(edge_mids[name], pt)

        circle_pen = QPen(Qt.GlobalColor.white, 1.5, Qt.PenStyle.SolidLine)
        circle_pen.setCosmetic(True)
        painter.setPen(circle_pen)
        painter.setBrush(QColor(THEME.accent_primary))
        pix = self._rotation_handle_pixmap()
        icon_r = _ROT_HANDLE_RADIUS_PX - 3.0
        for pt in handles.values():
            painter.drawEllipse(pt, _ROT_HANDLE_RADIUS_PX - 1.0, _ROT_HANDLE_RADIUS_PX - 1.0)
            painter.drawPixmap(QRectF(pt.x() - icon_r, pt.y() - icon_r, 2 * icon_r, 2 * icon_r), pix, QRectF(pix.rect()))

        if self._crop_drag_mode == "rotate" and self._rotate_current is not None:
            cx = (corners["tl"].x() + corners["br"].x()) / 2.0
            cy = (corners["tl"].y() + corners["br"].y()) / 2.0
            badge = QRectF(cx - 36, cy - 12, 72, 24)
            painter.setBrush(QColor(0, 0, 0, 170))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRoundedRect(badge, 4, 4)
            painter.setPen(QColor(THEME.accent_primary))
            # Badge shows the display convention (positive = clockwise on screen),
            # matching the Fine Rotation slider; _rotate_current is stored-convention.
            painter.drawText(badge, Qt.AlignmentFlag.AlignCenter, f"{-self._rotate_current:+.2f}°")

    def _analysis_rect_screen(self) -> Optional[QRectF]:
        """Screen rect for the current analysis region, or None if unset."""
        if self._analysis_drag_mode == "draw" and self._analysis_draw_p1 is not None:
            return QRectF(self._analysis_draw_p1, self._analysis_draw_p2 or self._analysis_draw_p1).normalized()
        if self._analysis_rect_norm is None or self._view_rect.isEmpty():
            return None
        x1, y1, x2, y2 = self._analysis_rect_norm
        return QRectF(self._norm_to_screen(x1, y1), self._norm_to_screen(x2, y2)).normalized()

    def _draw_analysis_tool(self, painter: QPainter) -> None:
        """Green-dashed analysis region: the exact area the exposure meters read."""
        rect = self._analysis_rect_screen()
        if rect is None:
            return
        rect = rect.intersected(self._view_rect)
        fill = QColor(THEME.channel_green)
        fill.setAlpha(28)
        painter.setBrush(fill)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRect(rect)

        pen = QPen(QColor(THEME.channel_green), 1.5, Qt.PenStyle.DashLine)
        pen.setCosmetic(True)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(pen)
        painter.drawRect(rect)

    def _draw_local_masks(self, painter: QPainter) -> None:
        if self._view_rect.isEmpty():
            return
        masks = self.state.config.local.masks
        self._local_mask_screen_polys = []
        if not masks:
            return

        with self.state.metrics_lock:
            uv_grid = self.state.last_metrics.get("uv_grid")
        if uv_grid is None:
            return

        selected = getattr(self.state, "local_selected_mask", -1)
        fresh_cache: Dict[tuple, QImage] = {}
        for i, mask in enumerate(masks):
            if len(mask.vertices) < 3:
                self._local_mask_screen_polys.append([])
                continue
            ctrl = [self._raw_to_screen(rx, ry, uv_grid) for rx, ry in mask.vertices]
            self._local_mask_screen_polys.append(ctrl)

            is_selected = i == selected
            if i in getattr(self.state, "local_hidden_masks", ()):
                continue
            working = self._local_edit_verts if is_selected else None
            drag_this = working is not None
            draw_ctrl = working if working is not None else ctrl
            curve = smooth_polyline([(p.x(), p.y()) for p in draw_ctrl], closed=True)
            outline = QColor(232, 200, 74) if mask.strength >= 0 else QColor(74, 143, 232)
            max_alpha = 70 if is_selected else 32

            # Skip the feathered fill mid-drag; it re-rasters every frame.
            if not drag_this:
                sigma_screen = mask.feather * min(self._view_rect.width(), self._view_rect.height())
                pad = 3.0 * sigma_screen + 2.0
                xs = [x for x, _ in curve]
                ys = [y for _, y in curve]
                x0, y0 = min(xs) - pad, min(ys) - pad
                bw, bh = max(xs) + pad - x0, max(ys) + pad - y0
                scale = min(1.0, _MASK_RASTER_MAX / max(bw, bh, 1.0))
                rw, rh = max(int(bw * scale), 2), max(int(bh * scale), 2)
                # Bbox-relative points are pan-invariant, so panning reuses the cache.
                local = tuple((round((x - x0) * scale, 1), round((y - y0) * scale, 1)) for x, y in curve)

                key = (local, rw, rh, round(sigma_screen * scale, 2), outline.rgb(), max_alpha)
                img = self._mask_img_cache.get(key)
                if img is None:
                    img = feathered_mask_image(local, rw, rh, sigma_screen * scale, outline, max_alpha)
                fresh_cache[key] = img
                painter.drawImage(QRectF(x0, y0, bw, bh), img)

            if is_selected:
                outline_color = QColor(outline)
                outline_color.setAlpha(200)
                pen = QPen(outline_color, 2.6, Qt.PenStyle.SolidLine)
            else:
                pen = QPen(QColor(255, 255, 255, 110), 1.4, Qt.PenStyle.SolidLine)
            pen.setCosmetic(True)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawPolygon(QPolygonF([QPointF(x, y) for x, y in curve]))

            if is_selected and self._tool_mode in (ToolMode.NONE, ToolMode.LOCAL_DRAW) and not self._lasso_drawing:
                self._draw_local_handles(painter, draw_ctrl, outline)
        self._mask_img_cache = fresh_cache

    def _draw_local_handles(self, painter: QPainter, ctrl_pts: List[QPointF], color: QColor) -> None:
        """Draggable vertices + '+' discs on edge midpoints for the selected mask."""
        n = len(ctrl_pts)
        if n < 2:
            return

        # Edge-midpoint "add point" handles: white disc with a plus glyph.
        plus_pen = QPen(QColor(35, 35, 35, 235), 1.5)
        plus_pen.setCosmetic(True)
        for i in range(n):
            a, b = ctrl_pts[i], ctrl_pts[(i + 1) % n]
            m = QPointF((a.x() + b.x()) / 2.0, (a.y() + b.y()) / 2.0)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(0, 0, 0, 130))
            painter.drawEllipse(m, 6.0, 6.0)
            painter.setBrush(QColor(255, 255, 255, 220))
            painter.drawEllipse(m, 5.0, 5.0)
            painter.setPen(plus_pen)
            painter.drawLine(QPointF(m.x() - 2.6, m.y()), QPointF(m.x() + 2.6, m.y()))
            painter.drawLine(QPointF(m.x(), m.y() - 2.6), QPointF(m.x(), m.y() + 2.6))

        # White halo behind a solid coloured core so vertices read on any image.
        core = QColor(color)
        core.setAlpha(255)
        for p in ctrl_pts:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(255, 255, 255, 240))
            painter.drawEllipse(p, 7.0, 7.0)
            painter.setBrush(core)
            painter.drawEllipse(p, 5.0, 5.0)

    def _draw_lasso_in_progress(self, painter: QPainter) -> None:
        if not self._lasso_drawing or not self._lasso_pts:
            return

        pen = QPen(Qt.GlobalColor.white, 1.5, Qt.PenStyle.SolidLine)
        pen.setCosmetic(True)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)

        painter.drawPath(self._preview_curve_path(self._lasso_pts))

        first = self._lasso_pts[0]
        near_close = len(self._lasso_pts) >= 3 and (self._mouse_pos - first).manhattanLength() < _LASSO_SNAP_PX * 2
        accent = QColor(THEME.accent_primary) if near_close else QColor(255, 255, 255, 180)
        painter.setBrush(accent)
        painter.setPen(Qt.PenStyle.NoPen)
        r = 5.0 if near_close else 3.0
        painter.drawEllipse(first, r, r)

    def _map_to_image_coords(self, screen_pos: QPointF) -> Optional[Tuple[float, float]]:
        if self._view_rect.isEmpty() or not self._view_rect.contains(screen_pos):
            return None

        nb_x = (screen_pos.x() - self._view_rect.x()) / self._view_rect.width()
        nb_y = (screen_pos.y() - self._view_rect.y()) / self._view_rect.height()

        return float(np.clip(nb_x, 0, 1)), float(np.clip(nb_y, 0, 1))

    def mousePressEvent(self, event: QMouseEvent) -> None:
        # A selected mask is editable even without the Draw Mask tool (grab a handle).
        if (
            event.button() == Qt.MouseButton.LeftButton
            and self._tool_mode == ToolMode.NONE
            and self._try_start_vertex_edit(event.position())
        ):
            event.accept()
            return

        if event.button() == Qt.MouseButton.MiddleButton or (
            event.button() == Qt.MouseButton.LeftButton and self.zoom_level > 1.0 and self._tool_mode == ToolMode.NONE
        ):
            self.parent()._is_panning = True
            self.parent()._last_mouse_pos = event.position()
            self.parent().setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return

        # Tool placements are left-click only: right-click must fall through to the
        # context menu without dropping a lasso vertex, scratch point, or heal.
        if event.button() != Qt.MouseButton.LeftButton:
            super().mousePressEvent(event)
            return

        if self._tool_mode == ToolMode.LOCAL_DRAW:
            self._handle_lasso_press(event.position())
            event.accept()
            return

        if self._tool_mode == ToolMode.SCRATCH_PICK:
            if self._view_rect.contains(event.position()):
                self._scratch_pts.append(event.position())
                self.update()
            event.accept()
            return

        if self._tool_mode == ToolMode.DUST_PICK:
            # Heal commits on release: a plain click heals the spot, a drag paints a
            # continuous stroke healed as one region (one undo step, one render).
            if self._view_rect.contains(event.position()):
                self._heal_drag_pts = [event.position()]
                self.update()
            event.accept()
            return

        if self._tool_mode == ToolMode.STRAIGHTEN:
            # Left-click draws the reference line; other buttons pass through.
            if event.button() == Qt.MouseButton.LeftButton:
                if self._view_rect.contains(event.position()):
                    self._straighten_p1 = event.position()
                    self._straighten_p2 = event.position()
                    self.update()
                event.accept()
                return

        if self._tool_mode == ToolMode.ANALYSIS_DRAW:
            self._start_analysis_drag(event.position())
            self.update()
            event.accept()
            return

        coords = self._map_to_image_coords(event.position())
        if coords:
            self.clicked.emit(*coords)
            if self._tool_mode == ToolMode.CROP_MANUAL:
                self._start_crop_drag(event.position())
            self.update()

    def _start_analysis_drag(self, pos: QPointF) -> None:
        if self._view_rect.isEmpty():
            return
        rect = self._analysis_rect_screen()
        if rect is not None and rect.contains(pos):
            self._analysis_drag_mode = "move"
            self._analysis_press_norm = self._screen_to_norm(pos)
            self._analysis_orig_rect = self._analysis_rect_norm
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            return
        # Fresh region: drag out a new rectangle.
        px = np.clip(pos.x(), self._view_rect.left(), self._view_rect.right())
        py = np.clip(pos.y(), self._view_rect.top(), self._view_rect.bottom())
        self._analysis_drag_mode = "draw"
        self._analysis_draw_p1 = QPointF(px, py)
        self._analysis_draw_p2 = QPointF(px, py)

    def _start_crop_drag(self, pos: QPointF) -> None:
        if self._view_rect.isEmpty():
            return

        # Rotation handles live outside the crop box, so they can't collide with
        # the corner/move hit areas — but test them first anyway.
        if self._hit_test_rotation_handle(pos) and self._crop_rect_norm is not None:
            x1, y1, x2, y2 = self._crop_rect_norm
            self._crop_drag_mode = "rotate"
            self._rotate_center = self._norm_to_screen((x1 + x2) / 2.0, (y1 + y2) / 2.0)
            self._rotate_press = pos
            self._rotate_start_fine = self.state.config.geometry.fine_rotation
            self._rotate_current = None
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            return

        corners = self._crop_corner_screen_points()
        corner = self._hit_test_crop_corner(pos, corners) if corners else None
        if corner is not None and corners is not None:
            anchor_name = {"tl": "br", "tr": "bl", "br": "tl", "bl": "tr"}[corner]
            self._crop_drag_mode = "corner"
            self._crop_anchor_screen = corners[anchor_name]
            self.setCursor(Qt.CursorShape.SizeFDiagCursor if corner in ("tl", "br") else Qt.CursorShape.SizeBDiagCursor)
            return

        if corners is not None and QPolygonF(list(corners.values())).containsPoint(pos, Qt.FillRule.OddEvenFill):
            self._crop_drag_mode = "move"
            self._crop_press_norm = self._screen_to_norm(pos)
            self._crop_orig_rect = self._crop_rect_norm
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            return

        # Clicked outside the existing rect: draw a fresh one from scratch (disarmed
        # until slop travel when a rect already exists).
        px = np.clip(pos.x(), self._view_rect.left(), self._view_rect.right())
        py = np.clip(pos.y(), self._view_rect.top(), self._view_rect.bottom())
        self._crop_drag_mode = "draw"
        self._crop_draw_armed = self._crop_rect_norm is None
        self._crop_draw_p1 = QPointF(px, py)
        self._crop_draw_p2 = QPointF(px, py)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        self._mouse_pos = event.position()

        coords = self._map_to_image_coords(event.position())
        if coords is not None:
            self.cursor_moved.emit(*coords)
        else:
            self.cursor_left.emit()

        # Placement tools carry special cursors (blank brush, pen nib, WB picker) that
        # read as broken over the empty canvas around the image — fall back to the
        # normal arrow there and restore the tool cursor over the image itself.
        if self._tool_mode in (ToolMode.DUST_PICK, ToolMode.SCRATCH_PICK, ToolMode.WB_PICK):
            if coords is None:
                self.setCursor(Qt.CursorShape.ArrowCursor)
            else:
                self.unsetCursor()

        if self.parent()._is_panning:
            delta = event.position() - self.parent()._last_mouse_pos
            self.parent()._last_mouse_pos = event.position()
            self.parent().pan_offset += QPointF(delta.x() / self.width(), delta.y() / self.height())
            self.parent()._sync_transform()
            event.accept()
            return

        if self._local_drag_vertex is not None and self._local_edit_verts is not None and not self._view_rect.isEmpty():
            px = float(np.clip(event.position().x(), self._view_rect.left(), self._view_rect.right()))
            py = float(np.clip(event.position().y(), self._view_rect.top(), self._view_rect.bottom()))
            self._local_edit_verts[self._local_drag_vertex] = QPointF(px, py)
            self.update()
            event.accept()
            return

        # Painting a heal stroke: accumulate the drag path (spaced by half the brush
        # radius so long drags stay a sane number of capsule segments), clamped to
        # the image so the stroke can't run off into the border.
        if self._tool_mode == ToolMode.DUST_PICK and self._heal_drag_pts and event.buttons() & Qt.MouseButton.LeftButton:
            pos = QPointF(
                float(np.clip(event.position().x(), self._view_rect.left(), self._view_rect.right())),
                float(np.clip(event.position().y(), self._view_rect.top(), self._view_rect.bottom())),
            )
            spacing = max(6.0, self._brush_screen_radius(self.state.config.retouch.manual_dust_size) * 0.5)
            if (pos - self._heal_drag_pts[-1]).manhattanLength() >= spacing:
                self._heal_drag_pts.append(pos)
            self.update()
            event.accept()
            return

        # Hovering the crop tool (no drag in progress): reflect the action under the
        # cursor — rotate handle, corner resize, interior move, or draw — right away.
        if self._tool_mode == ToolMode.CROP_MANUAL and self._crop_drag_mode is None:
            self._update_crop_hover_cursor(event.position())

        if self._analysis_drag_mode == "move" and self._analysis_press_norm is not None and self._analysis_orig_rect is not None:
            curr_norm = self._screen_to_norm(event.position())
            dx = curr_norm[0] - self._analysis_press_norm[0]
            dy = curr_norm[1] - self._analysis_press_norm[1]
            new_rect = translate_manual_crop_rect(self._analysis_orig_rect, dx, dy)
            if any(abs(a - b) > 5e-4 for a, b in zip(new_rect, self._analysis_rect_norm or new_rect)):
                self._analysis_rect_norm = new_rect
                self.analysis_rect_changed.emit(*new_rect, False)
                self.update()
            event.accept()
            return

        if self._analysis_drag_mode == "draw" and self._analysis_draw_p1 is not None:
            mx = np.clip(event.position().x(), self._view_rect.left(), self._view_rect.right())
            my = np.clip(event.position().y(), self._view_rect.top(), self._view_rect.bottom())
            self._analysis_draw_p2 = QPointF(mx, my)
            self.update()
            event.accept()
            return

        if self._crop_drag_mode == "rotate" and self._rotate_center is not None and self._rotate_press is not None:
            fine = bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier)
            angle = rotation_drag_angle(
                self._rotate_start_fine,
                (self._rotate_center.x(), self._rotate_center.y()),
                (self._rotate_press.x(), self._rotate_press.y()),
                (event.position().x(), event.position().y()),
                sensitivity=_ROT_FINE_SENSITIVITY if fine else 1.0,
            )
            if self._rotate_current is None or abs(angle - self._rotate_current) > 5e-3:
                self._rotate_current = angle
                self.crop_rotation_changed.emit(angle, False)
                self.update()
            event.accept()
            return

        if self._crop_drag_mode == "corner" and self._crop_anchor_screen is not None:
            cur_screen = QPointF(
                float(np.clip(event.position().x(), self._view_rect.left(), self._view_rect.right())),
                float(np.clip(event.position().y(), self._view_rect.top(), self._view_rect.bottom())),
            )
            rect = self._apply_aspect_and_min(self._crop_anchor_screen, cur_screen)
            self._crop_rect_norm = rect
            self.crop_rect_changed.emit(*rect, False)
            self.update()
            event.accept()
            return

        if self._crop_drag_mode == "move" and self._crop_press_norm is not None and self._crop_orig_rect is not None:
            curr_norm = self._screen_to_norm(event.position())
            # Normalized coords track the cursor 1:1, so a plain drag moves the crop
            # exactly with the mouse; Shift halves it for fine placement.
            fine = bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier)
            sensitivity = 0.5 if fine else 1.0
            dx = (curr_norm[0] - self._crop_press_norm[0]) * sensitivity
            dy = (curr_norm[1] - self._crop_press_norm[1]) * sensitivity
            new_rect = translate_manual_crop_rect(self._crop_orig_rect, dx, dy)
            if any(abs(a - b) > 5e-4 for a, b in zip(new_rect, self._crop_rect_norm or new_rect)):
                self._crop_rect_norm = new_rect
                self.crop_rect_changed.emit(*new_rect, False)
                self.update()
            event.accept()
            return

        if self._tool_mode == ToolMode.STRAIGHTEN and self._straighten_p1 is not None:
            self._straighten_p2 = QPointF(
                float(np.clip(event.position().x(), self._view_rect.left(), self._view_rect.right())),
                float(np.clip(event.position().y(), self._view_rect.top(), self._view_rect.bottom())),
            )
            self.update()
            event.accept()
            return

        if self._crop_drag_mode == "draw" and self._crop_draw_p1 is not None:
            mx = np.clip(event.position().x(), self._view_rect.left(), self._view_rect.right())
            my = np.clip(event.position().y(), self._view_rect.top(), self._view_rect.bottom())

            if not self._crop_draw_armed:
                if (QPointF(mx, my) - self._crop_draw_p1).manhattanLength() < _CROP_REDRAW_SLOP_PX:
                    return
                self._crop_draw_armed = True

            dx = mx - self._crop_draw_p1.x()
            dy = my - self._crop_draw_p1.y()
            target_ratio = self._oriented_target_ratio(dx, dy)
            if target_ratio is None:
                self._crop_draw_p2 = QPointF(mx, my)
            else:
                if abs(dx) > abs(dy) * target_ratio:
                    dx = abs(dy) * target_ratio * (1 if dx >= 0 else -1)
                else:
                    dy = abs(dx) / target_ratio * (1 if dy >= 0 else -1)

                self._crop_draw_p2 = QPointF(self._crop_draw_p1.x() + dx, self._crop_draw_p1.y() + dy)
            self.update()
            return

        self.update()

    def _selected_mask_screen_pts(self) -> Optional[List[QPointF]]:
        idx = getattr(self.state, "local_selected_mask", -1)
        if 0 <= idx < len(self._local_mask_screen_polys):
            pts = self._local_mask_screen_polys[idx]
            return pts if len(pts) >= 3 else None
        return None

    def _hit_local_vertex(self, pos: QPointF, pts: List[QPointF]) -> Optional[int]:
        for i, p in enumerate(pts):
            dx, dy = pos.x() - p.x(), pos.y() - p.y()
            if dx * dx + dy * dy <= _CROP_HANDLE_PX * _CROP_HANDLE_PX:
                return i
        return None

    def _hit_local_edge_midpoint(self, pos: QPointF, pts: List[QPointF]) -> Optional[int]:
        """Index i of the edge (i, i+1) whose midpoint handle is under `pos`."""
        n = len(pts)
        for i in range(n):
            a, b = pts[i], pts[(i + 1) % n]
            mx, my = (a.x() + b.x()) / 2.0, (a.y() + b.y()) / 2.0
            dx, dy = pos.x() - mx, pos.y() - my
            if dx * dx + dy * dy <= _CROP_HANDLE_PX * _CROP_HANDLE_PX:
                return i
        return None

    def try_delete_local_vertex(self, pos: QPointF) -> bool:
        """Right-click on a selected-mask vertex removes it. Returns True if handled."""
        pts = self._selected_mask_screen_pts()
        if pts is None:
            return False
        vi = self._hit_local_vertex(pos, pts)
        if vi is None:
            return False
        self.local_vertex_deleted.emit(getattr(self.state, "local_selected_mask", -1), vi)
        return True

    def _try_start_vertex_edit(self, pos: QPointF) -> bool:
        """Grab a selected-mask vertex, or insert one at an edge midpoint; True if started."""
        pts = self._selected_mask_screen_pts()
        if pts is None:
            return False
        vi = self._hit_local_vertex(pos, pts)
        if vi is not None:
            self._local_edit_verts = list(pts)
            self._local_drag_vertex = vi
            self.update()
            return True
        ei = self._hit_local_edge_midpoint(pos, pts)
        if ei is not None:
            work = list(pts)
            a, b = work[ei], work[(ei + 1) % len(work)]
            work.insert(ei + 1, QPointF((a.x() + b.x()) / 2.0, (a.y() + b.y()) / 2.0))
            self._local_edit_verts = work
            self._local_drag_vertex = ei + 1
            self.update()
            return True
        return False

    def _handle_lasso_press(self, pos: QPointF) -> None:
        if not self._view_rect.contains(pos):
            return

        if not self._lasso_drawing:
            if self._try_start_vertex_edit(pos):
                return
            for i, poly_pts in enumerate(self._local_mask_screen_polys):
                if len(poly_pts) < 3:
                    continue
                if QPolygonF(poly_pts).containsPoint(pos, Qt.FillRule.OddEvenFill):
                    self.local_mask_selected.emit(i)
                    return
            self._lasso_drawing = True
            self._lasso_pts = [pos]
            self.update()
            return

        first = self._lasso_pts[0]
        if len(self._lasso_pts) >= 3 and (pos - first).manhattanLength() < _LASSO_SNAP_PX * 2:
            self._finish_lasso()
            return

        self._lasso_pts.append(pos)
        self.update()

    def _finish_lasso(self) -> None:
        pts = self._lasso_pts
        self._lasso_pts = []
        self._lasso_drawing = False
        if len(pts) < 3:
            self.update()
            return
        vertices = []
        for pt in pts:
            coords = self._map_to_image_coords(pt)
            if coords is None:
                self.update()
                return
            vertices.append(coords)
        self.lasso_completed.emit(vertices)
        self.update()

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        if self._tool_mode == ToolMode.LOCAL_DRAW and self._lasso_drawing:
            self._finish_lasso()
            event.accept()
            return
        if self._tool_mode == ToolMode.SCRATCH_PICK and self._scratch_pts:
            self._finish_scratch()
            event.accept()
            return
        # Double-clicking inside the crop box confirms the crop and closes the tool, so
        # the user never has to leave the canvas to press the Crop button again.
        if self._tool_mode == ToolMode.CROP_MANUAL:
            corners = self._crop_corner_screen_points()
            if corners is not None and QPolygonF(list(corners.values())).containsPoint(event.position(), Qt.FillRule.OddEvenFill):
                self._end_crop_drag()
                self.crop_confirmed.emit()
                event.accept()
                return
        # Double-clicking inside the analysis region confirms it and closes the tool.
        if self._tool_mode == ToolMode.ANALYSIS_DRAW:
            rect = self._analysis_rect_screen()
            if rect is not None and rect.contains(event.position()):
                self._end_analysis_drag()
                self.analysis_confirmed.emit()
                event.accept()
                return
        super().mouseDoubleClickEvent(event)

    def _finish_draw_if_active(self) -> None:
        if self._tool_mode == ToolMode.SCRATCH_PICK and self._scratch_pts:
            self._finish_scratch()
        elif self._tool_mode == ToolMode.LOCAL_DRAW and self._lasso_drawing and len(self._lasso_pts) >= 3:
            self._finish_lasso()
        elif self._tool_mode == ToolMode.CROP_MANUAL:
            self._end_crop_drag()
            self.crop_confirmed.emit()

    def has_scratch_points(self) -> bool:
        return bool(self._scratch_pts)

    def confirm_scratch(self) -> None:
        """Commit the in-progress scratch polyline (same as double-click / Enter)."""
        if self._tool_mode == ToolMode.SCRATCH_PICK and self._scratch_pts:
            self._finish_scratch()

    def undo_last_scratch_point(self) -> None:
        """Step back one click-point of the in-progress scratch polyline."""
        if self._tool_mode == ToolMode.SCRATCH_PICK and self._scratch_pts:
            self._scratch_pts.pop()
            self.update()

    def heal_hit_test(self, pos: QPointF) -> Optional[Tuple[str, int]]:
        """Placed heal under `pos`, as ("stroke"|"spot", index), or None.

        Mirrors the geometry `_draw_placed_heals` renders: raw-normalized points
        mapped to screen through the uv grid, hit within the brush band radius
        (plus a small slop so thin strokes stay clickable).
        """
        conf = self.state.config.retouch
        if not (conf.manual_heal_strokes or conf.manual_dust_spots):
            return None
        with self.state.metrics_lock:
            uv_grid = self.state.last_metrics.get("uv_grid")
        if uv_grid is None:
            return None

        slop = 4.0
        best: Optional[Tuple[str, int]] = None
        best_dist = float("inf")
        for i, (points, size, _dx, _dy) in enumerate(conf.manual_heal_strokes):
            screen_pts = [self._raw_to_screen(px, py, uv_grid) for px, py in points]
            radius = max(2.0, self._brush_screen_radius(size)) + slop
            d = _distance_to_polyline(pos, screen_pts)
            if d <= radius and d < best_dist:
                best = ("stroke", i)
                best_dist = d
        for i, (rx, ry, size) in enumerate(conf.manual_dust_spots):
            center = self._raw_to_screen(rx, ry, uv_grid)
            radius = max(2.0, self._brush_screen_radius(size)) + slop
            d = math.hypot(pos.x() - center.x(), pos.y() - center.y())
            if d <= radius and d < best_dist:
                best = ("spot", i)
                best_dist = d
        return best

    def _finish_scratch(self) -> None:
        pts = self._scratch_pts
        self._scratch_pts = []
        # The double-click lands as an extra press at the previous point — drop near-duplicates.
        deduped: List[QPointF] = []
        for pt in pts:
            if not deduped or (pt - deduped[-1]).manhattanLength() > 2.0:
                deduped.append(pt)
        vertices = []
        for pt in deduped:
            coords = self._map_to_image_coords(pt)
            if coords is None:
                self.update()
                return
            vertices.append(coords)
        if vertices:
            self.scratch_completed.emit(vertices)
        self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self.parent()._is_panning:
            self.parent()._is_panning = False
            self.parent().reset_tool_cursor()
            event.accept()
            return

        if self._tool_mode == ToolMode.DUST_PICK and self._heal_drag_pts and event.button() == Qt.MouseButton.LeftButton:
            pts = self._heal_drag_pts
            self._heal_drag_pts = []
            end = QPointF(
                float(np.clip(event.position().x(), self._view_rect.left(), self._view_rect.right())),
                float(np.clip(event.position().y(), self._view_rect.top(), self._view_rect.bottom())),
            )
            if (end - pts[-1]).manhattanLength() > 2.0:
                pts.append(end)
            vertices = [c for c in (self._map_to_image_coords(p) for p in pts) if c is not None]
            if len(vertices) == 1:
                # Plain click: the classic single-spot heal.
                self.clicked.emit(*vertices[0])
            elif len(vertices) > 1:
                # Drag: the painted path becomes one multi-point heal stroke.
                self.scratch_completed.emit(vertices)
            self.update()
            event.accept()
            return

        if self._local_drag_vertex is not None:
            verts = self._local_edit_verts or []
            selected = getattr(self.state, "local_selected_mask", -1)
            self._end_local_edit()
            if verts and selected >= 0 and not self._view_rect.isEmpty():
                w, h = self._view_rect.width(), self._view_rect.height()
                vp = [
                    (
                        float(np.clip((p.x() - self._view_rect.x()) / w, 0.0, 1.0)),
                        float(np.clip((p.y() - self._view_rect.y()) / h, 0.0, 1.0)),
                    )
                    for p in verts
                ]
                self.local_mask_edited.emit(selected, vp)
            self.update()
            event.accept()
            return

        if self._tool_mode == ToolMode.STRAIGHTEN and self._straighten_p1 is not None:
            p1, p2 = self._straighten_p1, self._straighten_p2 or self._straighten_p1
            self._straighten_p1 = None
            self._straighten_p2 = None
            dx, dy = p2.x() - p1.x(), p2.y() - p1.y()
            # Ignore accidental clicks — a reference line needs some length.
            if math.hypot(dx, dy) >= 8.0:
                self.straighten_completed.emit(straighten_delta_degrees(dx, dy))
            self.update()
            event.accept()
            return

        if self._crop_drag_mode == "rotate":
            if self._rotate_current is not None:
                self.crop_rotation_changed.emit(self._rotate_current, True)
            self._end_crop_drag()
            self.unsetCursor()
            event.accept()
            return

        if self._crop_drag_mode in ("corner", "move"):
            if self._crop_rect_norm is not None:
                self.crop_rect_changed.emit(*self._crop_rect_norm, True)
            self._end_crop_drag()
            self.unsetCursor()
            event.accept()
            return

        if self._crop_drag_mode == "draw":
            if not self._crop_draw_armed:
                hud = getattr(self.parent(), "hud", None)
                if hud is not None and not self._crop_redraw_hint_shown:
                    self._crop_redraw_hint_shown = True
                    hud.showMessage("drag outside the box to redraw the crop", timeout=2500)
                self._end_crop_drag()
                self.update()
                event.accept()
                return
            r = QRectF(self._crop_draw_p1, self._crop_draw_p2 or self._crop_draw_p1).normalized()
            r = r.intersected(self._view_rect)
            if r.width() > 5 and r.height() > 5:
                c1 = self._screen_to_norm(r.topLeft())
                c2 = self._screen_to_norm(r.bottomRight())
                rect = (min(c1[0], c2[0]), min(c1[1], c2[1]), max(c1[0], c2[0]), max(c1[1], c2[1]))
                self._crop_rect_norm = rect
                self.crop_rect_changed.emit(*rect, True)
            self._end_crop_drag()
            self.update()

        if self._analysis_drag_mode == "move":
            if self._analysis_rect_norm is not None:
                self.analysis_rect_changed.emit(*self._analysis_rect_norm, True)
            self._end_analysis_drag()
            self.unsetCursor()
            event.accept()
            return

        if self._analysis_drag_mode == "draw":
            r = QRectF(self._analysis_draw_p1, self._analysis_draw_p2 or self._analysis_draw_p1).normalized()
            r = r.intersected(self._view_rect)
            if r.width() > 5 and r.height() > 5:
                c1 = self._screen_to_norm(r.topLeft())
                c2 = self._screen_to_norm(r.bottomRight())
                rect = (min(c1[0], c2[0]), min(c1[1], c2[1]), max(c1[0], c2[0]), max(c1[1], c2[1]))
                self._analysis_rect_norm = rect
                self.analysis_rect_changed.emit(*rect, True)
            self._end_analysis_drag()
            self.update()

    def leaveEvent(self, event) -> None:
        self.cursor_left.emit()
        super().leaveEvent(event)

    def update_overlay(self, filename: str, res: str, colorspace: str, extra: str, edits: int = 0) -> None:
        self.update()
