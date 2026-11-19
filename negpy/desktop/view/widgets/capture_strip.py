"""The film strip of Assign Capture Times: frames in film order, the reference photos placed in the
gaps between frames, the cuts between segments, and a bar under each segment. Every picture shows
whole, as wide as its shape at the chosen height."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import qtawesome as qta
from PyQt6.QtCore import QEvent, QPoint, QRect, QRectF, QSize, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QFontMetrics, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import QListWidget, QToolTip, QWidget

from negpy.desktop.view.styles.theme import THEME
from negpy.kernel.system.i18n import tr

THUMB_H = 54
MIN_W = 40  # a frame or photo is at least this wide, so its name and time fit
LINE_H = 15
CUT_W = 12
SLOT_W = 24
BAR_H = 20
PAD = THEME.space_md
TOP = THEME.space_sm
FRAME_ASPECT = 3 / 2
PHOTO_ASPECT = 4 / 3
_BADGE = 17


def strip_height(thumb_h: int) -> int:
    """The strip's height for a picture height: picture, name, time and the segment bar."""
    return TOP + thumb_h + THEME.space_xs + 2 * LINE_H + THEME.space_sm + BAR_H + THEME.space_sm


HEIGHT = strip_height(THUMB_H)


@dataclass(frozen=True)
class StripFrame:
    name: str
    time: str = ""
    tooltip: str = ""
    segment: int = 0
    rejected: bool = False
    typed: bool = False
    state: str = ""  # "", "warning" or "error"
    own: bool = False  # the frame has values of its own
    muted: bool = False  # outside the frames that Apply writes


@dataclass(frozen=True)
class StripPhoto:
    path: str
    time: str = ""
    name: str = ""
    tooltip: str = ""
    segment: int = 0
    error: bool = False
    ghost: bool = False  # the photo under Compare, drawn dashed where it would go


@dataclass(frozen=True)
class _Hit:
    kind: str  # "frame", "photo", "ghost", "cut", "slot" or "segment"
    rect: QRect
    index: int  # frame or segment index, else the gap
    sub: int = 0  # for a cut: the photos of the gap before it
    path: str = ""
    tooltip: str = ""


class CaptureStrip(QWidget):
    """Paints the strip and reports clicks. Gap g sits between frame g and frame g + 1; gap -1 is
    before the first frame. Each gap between two frames has a cut line before and after each of its
    photos. While a photo is chosen or dragged, each gap where it keeps the time order shows a +."""

    frameClicked = pyqtSignal(int)
    cutClicked = pyqtSignal(int, int)
    photoClicked = pyqtSignal(str)
    slotClicked = pyqtSignal(int)
    segmentClicked = pyqtSignal(int)
    photoDropped = pyqtSignal(str, int)
    quickLookRequested = pyqtSignal(str, str)  # "frame" and its index, or "photo" and its path
    sizeStepped = pyqtSignal(int)  # +1 or -1 from Ctrl and the wheel

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._frames: list[StripFrame] = []
        self._gaps: dict[int, list[StripPhoto]] = {}
        self._cuts: dict[int, int] = {}
        self._segments: list[str] = []
        self._frame_pixmaps: dict[int, QPixmap] = {}
        self._photo_pixmaps: dict[str, QPixmap] = {}
        self._scaled: dict[tuple, QPixmap] = {}
        self._selected_frame = -1
        self._selected_photo = ""
        self._selected_segment = -1
        self._armed = False
        self._moving = ""
        self._dragging = False
        self._slot_range: Optional[tuple[int, int]] = None
        self._drag_range: Optional[tuple[int, int]] = None
        self._gap_rule: Optional[Callable[[str], tuple[int, int]]] = None
        self._thumb_h = THUMB_H
        self._hits: list[_Hit] = []
        self._hover: Optional[_Hit] = None
        self._drop_gap: Optional[int] = None
        self.setMouseTracking(True)
        self.setAcceptDrops(True)
        self.setFixedHeight(HEIGHT)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._phone_icon = qta.icon("fa5s.mobile-alt", color=THEME.warn_amber)
        self._clock_icon = qta.icon("fa5s.clock", color=THEME.warn_amber)

    # ── content ───────────────────────────────────────────────────────────
    def set_content(
        self,
        frames: list[StripFrame],
        gaps: dict[int, list[StripPhoto]],
        cuts: dict[int, int],
        segments: list[str],
        *,
        selected_frame: int = -1,
        selected_photo: str = "",
        selected_segment: int = -1,
        armed: bool = False,
        moving: str = "",
        slot_range: Optional[tuple[int, int]] = None,
    ) -> None:
        """`cuts` maps a cut gap to the number of its photos before the cut; `segments` holds a label
        per segment. `slot_range` limits the + slots to the gaps first..last."""
        self._frames, self._gaps, self._cuts, self._segments = frames, gaps, cuts, segments
        self._selected_frame, self._selected_photo, self._selected_segment = selected_frame, selected_photo, selected_segment
        self._armed, self._moving, self._slot_range = armed, moving, slot_range
        self._relayout()

    def set_gap_rule(self, rule: Callable[[str], tuple[int, int]]) -> None:
        """The (first, last) gaps a dragged photo may go to."""
        self._gap_rule = rule

    def set_thumb_height(self, height: int) -> None:
        if height != self._thumb_h:
            self._thumb_h = height
            self._scaled.clear()
            self.setFixedHeight(strip_height(height))
            self._relayout()

    def thumb_height(self) -> int:
        return self._thumb_h

    def set_frame_pixmap(self, index: int, pixmap: QPixmap) -> None:
        reshaped = _aspect(self._frame_pixmaps.get(index)) != _aspect(pixmap)
        self._frame_pixmaps[index] = pixmap
        self._forget_scaled(("frame", index))
        self._relayout() if reshaped else self.update()

    def set_photo_pixmap(self, path: str, pixmap: QPixmap) -> None:
        reshaped = _aspect(self._photo_pixmaps.get(path)) != _aspect(pixmap)
        self._photo_pixmaps[path] = pixmap
        self._forget_scaled(("photo", path))
        self._relayout() if reshaped else self.update()

    def rect_of(self, kind: str, index: int = -2, path: str = "") -> Optional[QRect]:
        """Where a frame or a placed photo is, for scrolling it into view."""
        hit = next((h for h in self._hits if h.kind == kind and (h.path == path if path else h.index == index)), None)
        return hit.rect if hit else None

    def hits(self) -> list[_Hit]:
        return list(self._hits)

    def hovered_item(self) -> Optional[tuple[str, str]]:
        """("frame", index) or ("photo", path) under the pointer, for Quick Look."""
        hit = self._hover
        if hit is None:
            return None
        if hit.kind == "frame":
            return "frame", str(hit.index)
        if hit.kind in ("photo", "ghost"):
            return "photo", hit.path
        return None

    # ── layout ────────────────────────────────────────────────────────────
    def _width(self, pixmap: Optional[QPixmap], default: float) -> int:
        ratio = _aspect(pixmap) or default
        return max(MIN_W, round(self._thumb_h * ratio))

    def _allowed(self, gap: int) -> bool:
        rule = self._drag_range if self._dragging else self._slot_range
        if rule is None:
            return True
        first, last = rule
        return first > last or first <= gap <= last

    def _relayout(self) -> None:
        hits: list[_Hit] = []
        n = len(self._frames)
        slots = self._armed or self._dragging
        spans: dict[int, list[int]] = {}
        bar_y = self._bar_y()
        x = PAD

        def span(segment: int, left: int, right: int) -> None:
            lo_hi = spans.setdefault(segment, [left, right])
            lo_hi[0], lo_hi[1] = min(lo_hi[0], left), max(lo_hi[1], right)

        def cut(g: int, k: int) -> None:
            nonlocal x
            hits.append(_Hit("cut", QRect(x, TOP, CUT_W, self._thumb_h), g, k, tooltip=self._cut_tip(g, k)))
            x += CUT_W + PAD

        def gap(g: int) -> None:
            nonlocal x
            x += PAD
            inner = 0 <= g < n - 1
            k = 0
            for photo in self._gaps.get(g, []):
                width = self._width(self._photo_pixmaps.get(photo.path), PHOTO_ASPECT)
                if photo.ghost:
                    hits.append(_Hit("ghost", QRect(x, 0, width, bar_y), g, k, photo.path, photo.tooltip))
                    x += width + PAD
                    continue
                if inner:
                    cut(g, k)
                hits.append(_Hit("photo", QRect(x, 0, width, bar_y), g, k, photo.path, photo.tooltip))
                span(photo.segment, x, x + width)
                x += width + PAD
                k += 1
            if inner:
                cut(g, k)
            if slots and self._allowed(g):
                hits.append(_Hit("slot", QRect(x, TOP, SLOT_W, self._thumb_h), g, tooltip=self._slot_tip(g)))
                x += SLOT_W + PAD

        gap(-1)
        for i, frame in enumerate(self._frames):
            width = self._width(self._frame_pixmaps.get(i), FRAME_ASPECT)
            hits.append(_Hit("frame", QRect(x, 0, width, bar_y), i, tooltip=frame.tooltip))
            span(frame.segment, x, x + width)
            x += width
            gap(i)
        for segment, (left, right) in sorted(spans.items()):
            label = self._segments[segment] if segment < len(self._segments) else ""
            hits.append(_Hit("segment", QRect(left, bar_y, right - left, BAR_H), segment, tooltip=label))
        self._hits = hits
        self._hover = None
        self.setFixedWidth(x + PAD)
        self.update()

    def _name_y(self) -> int:
        return TOP + self._thumb_h + THEME.space_xs

    def _bar_y(self) -> int:
        return self._name_y() + 2 * LINE_H + THEME.space_sm

    def _cut_tip(self, g: int, k: int) -> str:
        if g not in self._cuts:
            return tr("Start a new segment here")
        if self._cuts[g] == k:
            return tr("Join the segments here")
        return tr("Move the start of the segment here")

    def _slot_tip(self, g: int) -> str:
        names = [f.name for f in self._frames]
        if not names:
            return ""
        if g < 0:
            return tr("Place the photo before {frame}", frame=names[0])
        if g >= len(names) - 1:
            return tr("Place the photo after {frame}", frame=names[-1])
        return tr("Place the photo between {first} and {second}", first=names[g], second=names[g + 1])

    # ── painting ──────────────────────────────────────────────────────────
    def paintEvent(self, _event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        font = QFont(self.font())
        font.setPixelSize(THEME.font_size_small)
        p.setFont(font)
        painters = {
            "frame": self._paint_frame,
            "photo": self._paint_photo,
            "ghost": self._paint_photo,
            "cut": self._paint_cut,
            "slot": self._paint_slot,
            "segment": self._paint_segment,
        }
        for hit in self._hits:
            painters[hit.kind](p, hit)
        p.end()

    def _forget_scaled(self, key: tuple) -> None:
        for cached in [k for k in self._scaled if k[:2] == key]:
            del self._scaled[cached]

    def _thumb(self, p: QPainter, rect: QRect, pixmap: Optional[QPixmap], key: tuple) -> None:
        """The whole picture, fitted into the box."""
        inner = rect.adjusted(2, 2, -2, -2)
        p.fillRect(inner, QColor(THEME.bg_input))
        if pixmap is None or pixmap.isNull():
            return
        ratio = self.devicePixelRatioF()
        size = QSize(round(inner.width() * ratio), round(inner.height() * ratio))
        cache_key = (*key, size.width(), size.height())
        scaled = self._scaled.get(cache_key)
        if scaled is None:
            scaled = pixmap.scaled(size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            scaled.setDevicePixelRatio(ratio)
            self._scaled[cache_key] = scaled
        w, h = round(scaled.width() / ratio), round(scaled.height() / ratio)
        p.drawPixmap(QRect(inner.x() + (inner.width() - w) // 2, inner.y() + (inner.height() - h) // 2, w, h), scaled)

    def _border(self, p: QPainter, rect: QRect, color: str, width: int, dashed: bool = False) -> None:
        pen = QPen(QColor(color), width)
        if dashed:
            pen.setStyle(Qt.PenStyle.DashLine)
        p.setPen(pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        half = width / 2
        p.drawRoundedRect(QRectF(rect).adjusted(half, half, -half, -half), THEME.radius_sm, THEME.radius_sm)

    def _badge(self, p: QPainter, x: int, y: int, icon) -> None:
        shade = QColor(THEME.mark_shadow)
        shade.setAlpha(150)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(shade)
        p.drawEllipse(x, y, _BADGE, _BADGE)
        icon.paint(p, QRect(x + 3, y + 3, _BADGE - 6, _BADGE - 6))

    def _text(self, p: QPainter, x: int, y: int, width: int, text: str, color: str) -> None:
        p.setPen(QColor(color))
        elided = QFontMetrics(p.font()).elidedText(text, Qt.TextElideMode.ElideMiddle, width)
        p.drawText(QRect(x, y, width, LINE_H), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, elided)

    def _hovered(self, hit: _Hit) -> bool:
        return self._hover is not None and self._hover.kind == hit.kind and self._hover.index == hit.index and self._hover.sub == hit.sub

    def _paint_frame(self, p: QPainter, hit: _Hit) -> None:
        frame = self._frames[hit.index]
        thumb = QRect(hit.rect.x(), TOP, hit.rect.width(), self._thumb_h)
        p.setOpacity(0.35 if frame.muted else (0.45 if frame.rejected else 1.0))
        self._thumb(p, thumb, self._frame_pixmaps.get(hit.index), ("frame", hit.index))
        selected = hit.index == self._selected_frame and not self._selected_photo
        color = THEME.border_checked_hover if selected else (THEME.border_focus if self._hovered(hit) else THEME.border_input)
        self._border(p, thumb, color, 2 if selected else 1)
        if frame.typed:
            self._badge(p, thumb.right() - _BADGE - 2, thumb.y() + 3, self._clock_icon)
        if frame.own:
            p.setPen(QPen(QColor(THEME.mark_shadow), 1))
            p.setBrush(QColor(THEME.mark_edited))
            p.drawEllipse(QPoint(thumb.x() + 8, thumb.bottom() - 7), 3, 3)
        p.setOpacity(1.0)
        name_y = self._name_y()
        self._text(p, hit.rect.x() + 1, name_y, hit.rect.width() - 2, frame.name, THEME.text_secondary)
        time_color = {"error": THEME.error_text, "warning": THEME.warn_amber}.get(frame.state, THEME.text_primary)
        self._text(p, hit.rect.x() + 1, name_y + LINE_H, hit.rect.width() - 2, frame.time, time_color)

    def _paint_photo(self, p: QPainter, hit: _Hit) -> None:
        photo = next((ph for ph in self._gaps.get(hit.index, []) if ph.path == hit.path and ph.ghost == (hit.kind == "ghost")), None)
        if photo is None:
            return
        thumb = QRect(hit.rect.x(), TOP, hit.rect.width(), self._thumb_h)
        p.setOpacity(0.4 if hit.path == self._moving else (0.55 if photo.ghost else 1.0))
        self._thumb(p, thumb, self._photo_pixmaps.get(hit.path), ("photo", hit.path))
        self._border(p, thumb, THEME.error if photo.error else THEME.warn_amber, 2, dashed=photo.ghost)
        if hit.path == self._selected_photo and not photo.ghost:
            self._border(p, thumb.adjusted(-3, -3, 3, 3), THEME.text_primary, 1)
        self._badge(p, thumb.x() + 3, thumb.y() + 3, self._phone_icon)
        p.setOpacity(1.0)
        name_y = self._name_y()
        self._text(p, hit.rect.x() + 1, name_y, hit.rect.width() - 2, photo.time, THEME.error_text if photo.error else THEME.warn_amber)
        self._text(p, hit.rect.x() + 1, name_y + LINE_H, hit.rect.width() - 2, photo.name, THEME.text_hint)

    def _paint_cut(self, p: QPainter, hit: _Hit) -> None:
        x = hit.rect.center().x()
        if self._cuts.get(hit.index) == hit.sub:
            p.setPen(QPen(QColor(THEME.border_checked_hover), 3))
            p.drawLine(x, TOP, x, self._bar_y() + BAR_H)
            p.setBrush(QColor(THEME.bg_card))
            p.drawEllipse(QPoint(x, TOP + self._thumb_h // 2), 4, 4)
            return
        p.setPen(QPen(QColor(THEME.border_focus if self._hovered(hit) else THEME.border_input), 2))
        p.drawLine(x, TOP + 6, x, TOP + self._thumb_h - 6)

    def _paint_slot(self, p: QPainter, hit: _Hit) -> None:
        if self._drop_gap == hit.index or self._hovered(hit):
            wash = QColor(THEME.warn_amber)
            wash.setAlpha(60)
            p.fillRect(hit.rect.adjusted(1, 1, -1, -1), wash)
        self._border(p, hit.rect, THEME.warn_amber, 1, dashed=True)
        p.setPen(QColor(THEME.warn_amber))
        p.drawText(hit.rect, Qt.AlignmentFlag.AlignCenter, "+")

    def _paint_segment(self, p: QPainter, hit: _Hit) -> None:
        selected = hit.index == self._selected_segment
        p.setPen(QPen(QColor(THEME.border_checked if selected else (THEME.border_focus if self._hovered(hit) else THEME.border_input)), 1))
        p.setBrush(QColor(THEME.bg_checked if selected else THEME.bg_input))
        p.drawRoundedRect(QRectF(hit.rect).adjusted(0.5, 0.5, -0.5, -0.5), THEME.radius_md, THEME.radius_md)
        p.setPen(QColor(THEME.text_primary if selected else THEME.text_secondary))
        # The label stays in view while a long segment scrolls past.
        visible = self.visibleRegion().boundingRect()
        left = max(hit.rect.left(), visible.left() if not visible.isEmpty() else hit.rect.left()) + THEME.space_md
        inner = QRect(left, hit.rect.y(), max(0, hit.rect.right() - THEME.space_md - left), hit.rect.height())
        p.drawText(
            inner,
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            QFontMetrics(p.font()).elidedText(hit.tooltip, Qt.TextElideMode.ElideRight, inner.width()),
        )

    # ── mouse ─────────────────────────────────────────────────────────────
    def _hit_at(self, pos: QPoint) -> Optional[_Hit]:
        return next((h for h in reversed(self._hits) if h.rect.contains(pos)), None)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        hit = self._hit_at(event.position().toPoint())
        if hit != self._hover:
            self._hover = hit
            self.setCursor(Qt.CursorShape.PointingHandCursor if hit else Qt.CursorShape.ArrowCursor)
            self.update()

    def leaveEvent(self, _event) -> None:  # noqa: N802
        self._hover = None
        self.update()

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() != Qt.MouseButton.LeftButton:
            return
        hit = self._hit_at(event.position().toPoint())
        if hit is None or hit.kind == "ghost":
            return
        if hit.kind == "frame":
            self.frameClicked.emit(hit.index)
        elif hit.kind == "cut":
            self.cutClicked.emit(hit.index, hit.sub)
        elif hit.kind == "photo":
            self.photoClicked.emit(hit.path)
        elif hit.kind == "segment":
            self.segmentClicked.emit(hit.index)
        else:
            self.slotClicked.emit(hit.index)

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802
        if event.button() != Qt.MouseButton.LeftButton:
            return
        hit = self._hit_at(event.position().toPoint())
        if hit is not None and hit.kind == "frame":
            self.quickLookRequested.emit("frame", str(hit.index))
        elif hit is not None and hit.kind in ("photo", "ghost"):
            self.quickLookRequested.emit("photo", hit.path)

    def wheelEvent(self, event) -> None:  # noqa: N802
        if event.modifiers() & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.MetaModifier) and event.angleDelta().y():
            self.sizeStepped.emit(1 if event.angleDelta().y() > 0 else -1)
            event.accept()
            return
        event.ignore()

    def event(self, event) -> bool:
        if event.type() == QEvent.Type.ToolTip:
            hit = self._hit_at(event.pos())
            if hit is not None and hit.tooltip:
                QToolTip.showText(event.globalPos(), hit.tooltip, self)
            else:
                QToolTip.hideText()
            return True
        return super().event(event)

    # ── drag a photo from the reference list ─────────────────────────────
    @staticmethod
    def _dragged_path(event) -> str:
        source = event.source()
        if isinstance(source, QListWidget) and source.currentItem() is not None:
            return source.currentItem().data(Qt.ItemDataRole.UserRole) or ""
        return ""

    def _nearest_slot(self, pos: QPoint) -> Optional[int]:
        slots = [h for h in self._hits if h.kind == "slot"]
        if not slots:
            return None
        return min(slots, key=lambda h: abs(h.rect.center().x() - pos.x())).index

    def dragEnterEvent(self, event) -> None:  # noqa: N802
        path = self._dragged_path(event)
        if not path:
            event.ignore()
            return
        self._drag_range = self._gap_rule(path) if self._gap_rule is not None else None
        self._dragging = True
        self._relayout()
        event.acceptProposedAction()

    def dragMoveEvent(self, event) -> None:  # noqa: N802
        self._drop_gap = self._nearest_slot(event.position().toPoint())
        self.update()
        event.acceptProposedAction()

    def dragLeaveEvent(self, _event) -> None:  # noqa: N802
        self._end_drag()

    def dropEvent(self, event) -> None:  # noqa: N802
        path, gap = self._dragged_path(event), self._nearest_slot(event.position().toPoint())
        self._end_drag()
        if path and gap is not None:
            event.acceptProposedAction()
            self.photoDropped.emit(path, gap)

    def _end_drag(self) -> None:
        self._dragging = False
        self._drag_range = None
        self._drop_gap = None
        self._relayout()


def _aspect(pixmap: Optional[QPixmap]) -> float:
    """Width over height, or 0 for no picture."""
    if pixmap is None or pixmap.isNull() or pixmap.height() == 0:
        return 0.0
    return pixmap.width() / pixmap.height()
