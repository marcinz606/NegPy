import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
from PyQt6.QtCore import QPointF, QRectF, QSize, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QImage, QKeySequence, QMouseEvent, QPainter, QPainterPath, QPen, QPolygonF, QShortcut
from PyQt6.QtWidgets import QWidget

from negpy.desktop.converters import ImageConverter
from negpy.desktop.session import AppState, ToolMode
from negpy.desktop.view.styles.theme import THEME
from negpy.features.geometry.logic import translate_manual_crop_rect
from negpy.features.local.logic import _rasterise_mask
from negpy.kernel.system.config import APP_CONFIG
from negpy.services.view.coordinate_mapping import CoordinateMapping

_LASSO_SNAP_PX = 12.0
_CROP_HANDLE_PX = 10.0
_CROP_MIN_SCREEN_PX = 24.0
_ROTATION_GRID_DIVISIONS = 10
_GRID_ALPHA = 70
_MASK_RASTER_MAX = 384  # px cap for feathered overlay rasters


def grid_interior_fractions(divisions: int) -> List[float]:
    """Interior division fractions, e.g. 3 -> [1/3, 2/3], 10 -> [.1 .. .9]."""
    return [i / divisions for i in range(1, divisions)]


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
    cursor_moved = pyqtSignal(float, float)
    cursor_left = pyqtSignal()
    lasso_completed = pyqtSignal(list)
    local_mask_selected = pyqtSignal(int)

    def __init__(self, state: AppState, parent=None):
        super().__init__(parent)
        self.state = state
        self._qimage: Optional[QImage] = None
        self._current_size: Optional[Tuple[int, int]] = None
        self._content_rect: Optional[Tuple[int, int, int, int]] = None

        # Crop tool interaction state: corner-resize, interior move, or fresh draw
        # (when the click lands outside the existing rect).
        self._crop_rect_raw: Optional[Tuple[float, float, float, float]] = None
        self._crop_drag_mode: Optional[str] = None  # "corner" | "move" | "draw"
        self._crop_anchor_screen: Optional[QPointF] = None
        self._crop_press_raw: Optional[Tuple[float, float]] = None
        self._crop_orig_rect: Optional[Tuple[float, float, float, float]] = None
        self._crop_uv_grid: Optional[np.ndarray] = None
        self._crop_draw_p1: Optional[QPointF] = None
        self._crop_draw_p2: Optional[QPointF] = None
        self._tool_mode: ToolMode = ToolMode.NONE
        self._mouse_pos: QPointF = QPointF()

        # Lasso (polygon mask) interaction state
        self._lasso_pts: List[QPointF] = []
        self._lasso_drawing: bool = False
        self._local_mask_screen_polys: List[List[QPointF]] = []
        self._mask_img_cache: Dict[tuple, QImage] = {}

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

        self._escape_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Escape), self)
        self._escape_shortcut.setContext(Qt.ShortcutContext.WidgetShortcut)
        self._escape_shortcut.activated.connect(self._cancel_lasso)

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
            self._crop_rect_raw = self.state.config.geometry.manual_crop_rect
        else:
            self._crop_rect_raw = None
            self._end_crop_drag()
        if mode != ToolMode.LOCAL_DRAW:
            self._lasso_pts = []
            self._lasso_drawing = False
        self.update()

    def _end_crop_drag(self) -> None:
        self._crop_drag_mode = None
        self._crop_anchor_screen = None
        self._crop_press_raw = None
        self._crop_orig_rect = None
        self._crop_uv_grid = None
        self._crop_draw_p1 = None
        self._crop_draw_p2 = None

    def _cancel_lasso(self) -> None:
        if self._tool_mode == ToolMode.LOCAL_DRAW and self._lasso_drawing:
            self._lasso_pts = []
            self._lasso_drawing = False
            self.update()

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
            self._crop_rect_raw = self.state.config.geometry.manual_crop_rect

        self._recalc_view_rect()
        self.update()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._recalc_view_rect()
        self.update()

    def _recalc_view_rect(self) -> None:
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

        if self._buffer_overlay_visible and self._buffer_overlay_ratio > 1e-4 and self._tool_mode != ToolMode.CROP_MANUAL:
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
            if self._tool_mode == ToolMode.DUST_PICK:
                self._draw_brush(painter)
            elif self._tool_mode != ToolMode.LOCAL_DRAW:
                pen = QPen(QColor(255, 255, 255, 80), 1, Qt.PenStyle.DotLine)
                pen.setCosmetic(True)
                painter.setPen(pen)
                painter.drawLine(QPointF(visible_rect.x(), self._mouse_pos.y()), QPointF(visible_rect.right(), self._mouse_pos.y()))
                painter.drawLine(QPointF(self._mouse_pos.x(), visible_rect.top()), QPointF(self._mouse_pos.x(), visible_rect.bottom()))

        show_masks = getattr(self.state, "show_local_overlay", False) or self._tool_mode == ToolMode.LOCAL_DRAW
        if show_masks:
            self._draw_local_masks(painter)
        if self._tool_mode == ToolMode.LOCAL_DRAW:
            self._draw_lasso_in_progress(painter)

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

    def _draw_compare_badge(self, painter: QPainter, visible_rect: QRectF) -> None:
        badge = QRectF(visible_rect.x() + 12, visible_rect.y() + 12, 78, 22)
        painter.setBrush(QColor(0, 0, 0, 170))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(badge, 4, 4)
        painter.setPen(QColor(THEME.accent_primary))
        painter.drawText(badge, Qt.AlignmentFlag.AlignCenter, "BEFORE")

    def _draw_brush(self, painter: QPainter) -> None:
        conf = self.state.config.retouch
        max_screen_dim = max(self._view_rect.width(), self._view_rect.height())
        radius = (conf.manual_dust_size / (2.0 * APP_CONFIG.preview_render_size)) * max_screen_dim

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

    def _raw_to_screen(self, rx: float, ry: float, uv_grid: np.ndarray, buckets: int = 100) -> QPointF:
        """
        Inverse UV-grid lookup: raw-normalised (0-1) -> screen position.

        Downsamples the grid before the nearest-neighbour search so this stays
        fast even for large preview buffers. `buckets` trades precision for
        speed: the default is fine for one-shot mask-vertex rendering, but
        anything redrawn continuously while being dragged (e.g. crop handles)
        needs a much finer grid or the on-screen point visibly snaps between
        buckets instead of tracking the cursor.
        """
        h_uv, w_uv = uv_grid.shape[:2]
        step = max(1, h_uv // buckets)
        small = uv_grid[::step, ::step]
        dist = (small[..., 0] - rx) ** 2 + (small[..., 1] - ry) ** 2
        idx = int(np.argmin(dist))
        h_s, w_s = small.shape[:2]
        vy, vx = divmod(idx, w_s)
        nx = min((vx * step + step // 2) / w_uv, 1.0)
        ny = min((vy * step + step // 2) / h_uv, 1.0)
        return QPointF(
            self._view_rect.x() + nx * self._view_rect.width(),
            self._view_rect.y() + ny * self._view_rect.height(),
        )

    def _crop_corner_screen_points(self) -> Optional[Dict[str, QPointF]]:
        """Maps the current raw crop rect to its on-screen axis-aligned bounding box.

        The actual crop (`get_manual_rect_coords` / `CropProcessor`) always takes the
        axis-aligned bounding box of the rect's transformed corners - it's a plain
        array slice, never a rotated one. So under fine rotation the 4 raw corners
        land at a tilted quadrilateral, but we deliberately collapse that to its AABB
        here rather than drawing the tilt, so the overlay shows what will actually be
        cropped instead of a shape the pipeline can't produce.
        """
        if self._crop_rect_raw is None:
            return None
        with self.state.metrics_lock:
            uv_grid = self.state.last_metrics.get("uv_grid")
        if uv_grid is None:
            return None
        x1, y1, x2, y2 = self._crop_rect_raw
        h_uv, w_uv = uv_grid.shape[:2]
        buckets = max(h_uv, w_uv)
        pts = [
            self._raw_to_screen(x1, y1, uv_grid, buckets),
            self._raw_to_screen(x2, y1, uv_grid, buckets),
            self._raw_to_screen(x2, y2, uv_grid, buckets),
            self._raw_to_screen(x1, y2, uv_grid, buckets),
        ]
        left = min(p.x() for p in pts)
        right = max(p.x() for p in pts)
        top = min(p.y() for p in pts)
        bottom = max(p.y() for p in pts)
        return {
            "tl": QPointF(left, top),
            "tr": QPointF(right, top),
            "br": QPointF(right, bottom),
            "bl": QPointF(left, bottom),
        }

    def _hit_test_crop_corner(self, pos: QPointF, corners: Dict[str, QPointF]) -> Optional[str]:
        for name, pt in corners.items():
            dx, dy = pos.x() - pt.x(), pos.y() - pt.y()
            if dx * dx + dy * dy <= _CROP_HANDLE_PX * _CROP_HANDLE_PX:
                return name
        return None

    def _apply_aspect_and_min(self, anchor_screen: QPointF, cur_screen: QPointF, uv_grid: np.ndarray) -> Tuple[float, float, float, float]:
        """Resizes a rect anchored at `anchor_screen` towards `cur_screen`, honoring the
        configured aspect ratio (if any) and a minimum rect size.

        Done entirely in screen-pixel space: raw-normalised (0-1) fractions only equal
        physical aspect ratio when the source image is square, so applying a target
        ratio to raw-space deltas distorts it by the image's actual width/height ratio.
        Screen pixels reflect the image as displayed, so ratios computed there are correct.
        """
        ax, ay = anchor_screen.x(), anchor_screen.y()
        nx, ny = cur_screen.x(), cur_screen.y()

        ratio_str = self.state.config.geometry.autocrop_ratio
        target_ratio: Optional[float] = None
        if ratio_str != "Free":
            try:
                w_r, h_r = map(float, ratio_str.split(":"))
                target_ratio = w_r / h_r
            except Exception:
                target_ratio = None

        dx = nx - ax
        dy = ny - ay

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
        c1 = self._raw_from_screen_with_grid(anchor_screen, uv_grid)
        c2 = self._raw_from_screen_with_grid(end_screen, uv_grid)
        if c1 is None or c2 is None:
            return self._crop_rect_raw or (0.0, 0.0, 1.0, 1.0)
        x1, x2 = sorted((c1[0], c2[0]))
        y1, y2 = sorted((c1[1], c2[1]))
        return (max(0.0, x1), max(0.0, y1), min(1.0, x2), min(1.0, y2))

    def _draw_crop_tool(self, painter: QPainter) -> None:
        if self._crop_drag_mode == "draw" and self._crop_draw_p1 is not None:
            rect = QRectF(self._crop_draw_p1, self._crop_draw_p2 or self._crop_draw_p1).normalized().intersected(self._view_rect)
            pen = QPen(Qt.GlobalColor.white, 1, Qt.PenStyle.DashLine)
            pen.setCosmetic(True)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(pen)
            painter.drawRect(rect)
            self._draw_grid(painter, rect, 3, _GRID_ALPHA)
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

        self._draw_grid(painter, QRectF(corners["tl"], corners["br"]), 3, _GRID_ALPHA)

        handle_pen = QPen(Qt.GlobalColor.white, 1.5, Qt.PenStyle.SolidLine)
        handle_pen.setCosmetic(True)
        painter.setPen(handle_pen)
        painter.setBrush(QColor(THEME.accent_primary))
        for pt in corners.values():
            painter.drawRect(QRectF(pt.x() - 5, pt.y() - 5, 10, 10))

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
            screen_pts = [self._raw_to_screen(rx, ry, uv_grid) for rx, ry in mask.vertices]
            self._local_mask_screen_polys.append(screen_pts)

            is_selected = i == selected
            outline = QColor(232, 200, 74) if mask.strength >= 0 else QColor(74, 143, 232)
            max_alpha = 70 if is_selected else 32

            sigma_screen = mask.feather * min(self._view_rect.width(), self._view_rect.height())
            pad = 3.0 * sigma_screen + 2.0
            xs = [p.x() for p in screen_pts]
            ys = [p.y() for p in screen_pts]
            x0, y0 = min(xs) - pad, min(ys) - pad
            bw, bh = max(xs) + pad - x0, max(ys) + pad - y0
            scale = min(1.0, _MASK_RASTER_MAX / max(bw, bh, 1.0))
            rw, rh = max(int(bw * scale), 2), max(int(bh * scale), 2)
            # Bbox-relative points are pan-invariant, so panning reuses the cache.
            local = tuple((round((p.x() - x0) * scale, 1), round((p.y() - y0) * scale, 1)) for p in screen_pts)

            key = (local, rw, rh, round(sigma_screen * scale, 2), outline.rgb(), max_alpha)
            img = self._mask_img_cache.get(key)
            if img is None:
                img = feathered_mask_image(local, rw, rh, sigma_screen * scale, outline, max_alpha)
            fresh_cache[key] = img
            painter.drawImage(QRectF(x0, y0, bw, bh), img)

            if is_selected:
                outline_color = QColor(outline)
                outline_color.setAlpha(160)
                pen = QPen(outline_color, 2.0, Qt.PenStyle.SolidLine)
            else:
                pen = QPen(QColor(255, 255, 255, 100), 1.0, Qt.PenStyle.SolidLine)
            pen.setCosmetic(True)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawPolygon(QPolygonF(screen_pts))
        self._mask_img_cache = fresh_cache

    def _draw_lasso_in_progress(self, painter: QPainter) -> None:
        if not self._lasso_drawing or not self._lasso_pts:
            return

        pen = QPen(Qt.GlobalColor.white, 1.5, Qt.PenStyle.SolidLine)
        pen.setCosmetic(True)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)

        path = QPainterPath(self._lasso_pts[0])
        for pt in self._lasso_pts[1:]:
            path.lineTo(pt)
        if self._view_rect.contains(self._mouse_pos):
            path.lineTo(self._mouse_pos)
        painter.drawPath(path)

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

    def _raw_from_screen_with_grid(self, screen_pos: QPointF, uv_grid: np.ndarray) -> Optional[Tuple[float, float]]:
        if self._view_rect.isEmpty():
            return None
        nb_x = float(np.clip((screen_pos.x() - self._view_rect.x()) / self._view_rect.width(), 0.0, 1.0))
        nb_y = float(np.clip((screen_pos.y() - self._view_rect.y()) / self._view_rect.height(), 0.0, 1.0))
        return CoordinateMapping.map_click_to_raw(nb_x, nb_y, uv_grid)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.MiddleButton or (
            event.button() == Qt.MouseButton.LeftButton and self.zoom_level > 1.0 and self._tool_mode == ToolMode.NONE
        ):
            self.parent()._is_panning = True
            self.parent()._last_mouse_pos = event.position()
            self.parent().setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return

        if self._tool_mode == ToolMode.LOCAL_DRAW:
            self._handle_lasso_press(event.position())
            event.accept()
            return

        coords = self._map_to_image_coords(event.position())
        if coords:
            self.clicked.emit(*coords)
            if self._tool_mode == ToolMode.CROP_MANUAL:
                self._start_crop_drag(event.position())
            self.update()

    def _start_crop_drag(self, pos: QPointF) -> None:
        with self.state.metrics_lock:
            uv_grid = self.state.last_metrics.get("uv_grid")
        if uv_grid is None:
            return

        corners = self._crop_corner_screen_points()
        corner = self._hit_test_crop_corner(pos, corners) if corners else None
        if corner is not None and corners is not None:
            anchor_name = {"tl": "br", "tr": "bl", "br": "tl", "bl": "tr"}[corner]
            self._crop_drag_mode = "corner"
            self._crop_anchor_screen = corners[anchor_name]
            self._crop_uv_grid = uv_grid
            self.setCursor(Qt.CursorShape.SizeFDiagCursor if corner in ("tl", "br") else Qt.CursorShape.SizeBDiagCursor)
            return

        if corners is not None and QPolygonF(list(corners.values())).containsPoint(pos, Qt.FillRule.OddEvenFill):
            press_raw = self._raw_from_screen_with_grid(pos, uv_grid)
            if press_raw is not None:
                self._crop_drag_mode = "move"
                self._crop_press_raw = press_raw
                self._crop_orig_rect = self._crop_rect_raw
                self._crop_uv_grid = uv_grid
                self.setCursor(Qt.CursorShape.ClosedHandCursor)
            return

        # Clicked outside the existing rect: draw a fresh one from scratch.
        px = np.clip(pos.x(), self._view_rect.left(), self._view_rect.right())
        py = np.clip(pos.y(), self._view_rect.top(), self._view_rect.bottom())
        self._crop_drag_mode = "draw"
        self._crop_draw_p1 = QPointF(px, py)
        self._crop_draw_p2 = QPointF(px, py)
        self._crop_uv_grid = uv_grid

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        self._mouse_pos = event.position()

        coords = self._map_to_image_coords(event.position())
        if coords is not None:
            self.cursor_moved.emit(*coords)
        else:
            self.cursor_left.emit()

        if self.parent()._is_panning:
            delta = event.position() - self.parent()._last_mouse_pos
            self.parent()._last_mouse_pos = event.position()
            self.parent().pan_offset += QPointF(delta.x() / self.width(), delta.y() / self.height())
            self.parent()._sync_transform()
            event.accept()
            return

        if self._crop_drag_mode == "corner" and self._crop_anchor_screen is not None and self._crop_uv_grid is not None:
            cur_screen = QPointF(
                float(np.clip(event.position().x(), self._view_rect.left(), self._view_rect.right())),
                float(np.clip(event.position().y(), self._view_rect.top(), self._view_rect.bottom())),
            )
            rect = self._apply_aspect_and_min(self._crop_anchor_screen, cur_screen, self._crop_uv_grid)
            self._crop_rect_raw = rect
            self.crop_rect_changed.emit(*rect, False)
            self.update()
            event.accept()
            return

        if (
            self._crop_drag_mode == "move"
            and self._crop_press_raw is not None
            and self._crop_orig_rect is not None
            and self._crop_uv_grid is not None
        ):
            curr_raw = self._raw_from_screen_with_grid(event.position(), self._crop_uv_grid)
            if curr_raw is not None:
                fine = bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier)
                sensitivity = 0.2 if fine else 0.5
                dx = (curr_raw[0] - self._crop_press_raw[0]) * sensitivity
                dy = (curr_raw[1] - self._crop_press_raw[1]) * sensitivity
                new_rect = translate_manual_crop_rect(self._crop_orig_rect, dx, dy)
                if any(abs(a - b) > 5e-4 for a, b in zip(new_rect, self._crop_rect_raw or new_rect)):
                    self._crop_rect_raw = new_rect
                    self.crop_rect_changed.emit(*new_rect, False)
                    self.update()
            event.accept()
            return

        if self._crop_drag_mode == "draw" and self._crop_draw_p1 is not None:
            mx = np.clip(event.position().x(), self._view_rect.left(), self._view_rect.right())
            my = np.clip(event.position().y(), self._view_rect.top(), self._view_rect.bottom())

            ratio_str = self.state.config.geometry.autocrop_ratio
            if ratio_str == "Free":
                self._crop_draw_p2 = QPointF(mx, my)
            else:
                try:
                    w_r, h_r = map(float, ratio_str.split(":"))
                    target_ratio = w_r / h_r

                    dx = mx - self._crop_draw_p1.x()
                    dy = my - self._crop_draw_p1.y()

                    if abs(dx) > abs(dy) * target_ratio:
                        dx = abs(dy) * target_ratio * (1 if dx >= 0 else -1)
                    else:
                        dy = abs(dx) / target_ratio * (1 if dy >= 0 else -1)

                    self._crop_draw_p2 = QPointF(self._crop_draw_p1.x() + dx, self._crop_draw_p1.y() + dy)
                except Exception:
                    self._crop_draw_p2 = QPointF(mx, my)
            self.update()
            return

        self.update()

    def _handle_lasso_press(self, pos: QPointF) -> None:
        if not self._view_rect.contains(pos):
            return

        if not self._lasso_drawing:
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
        super().mouseDoubleClickEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self.parent()._is_panning:
            self.parent()._is_panning = False
            self.parent().reset_tool_cursor()
            event.accept()
            return

        if self._crop_drag_mode in ("corner", "move"):
            if self._crop_rect_raw is not None:
                self.crop_rect_changed.emit(*self._crop_rect_raw, True)
            self._end_crop_drag()
            self.unsetCursor()
            event.accept()
            return

        if self._crop_drag_mode == "draw":
            r = QRectF(self._crop_draw_p1, self._crop_draw_p2 or self._crop_draw_p1).normalized()
            r = r.intersected(self._view_rect)
            uv_grid = self._crop_uv_grid
            if r.width() > 5 and r.height() > 5 and uv_grid is not None:
                c1 = self._raw_from_screen_with_grid(r.topLeft(), uv_grid)
                c2 = self._raw_from_screen_with_grid(r.bottomRight(), uv_grid)
                if c1 and c2:
                    rect = (min(c1[0], c2[0]), min(c1[1], c2[1]), max(c1[0], c2[0]), max(c1[1], c2[1]))
                    self._crop_rect_raw = rect
                    self.crop_rect_changed.emit(*rect, True)
            self._end_crop_drag()
            self.update()

    def leaveEvent(self, event) -> None:
        self.cursor_left.emit()
        super().leaveEvent(event)

    def update_overlay(self, filename: str, res: str, colorspace: str, extra: str, edits: int = 0) -> None:
        self.update()
