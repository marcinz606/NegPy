"""The film strip of Assign Capture Times: frames in film order, the reference photos placed in the
gaps between frames, the cuts between segments, and a bar under each segment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import qtawesome as qta
from PyQt6.QtCore import QEvent, QPoint, QRect, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QFontMetrics, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import QListWidget, QToolTip, QWidget

from negpy.desktop.view.styles.theme import THEME
from negpy.kernel.system.i18n import tr

FRAME_W = 80
PHOTO_W = 64
THUMB_H = 54
LINE_H = 15
CUT_W = 12
SLOT_W = 24
BAR_H = 20
PAD = THEME.space_md
TOP = THEME.space_sm
NAME_Y = TOP + THUMB_H + THEME.space_xs
TIME_Y = NAME_Y + LINE_H
BAR_Y = TIME_Y + LINE_H + THEME.space_sm
HEIGHT = BAR_Y + BAR_H + THEME.space_sm
_BADGE = 17


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


@dataclass(frozen=True)
class _Hit:
    kind: str  # "frame", "photo", "cut", "slot" or "segment"
    rect: QRect
    index: int  # frame or segment index, else the gap
    sub: int = 0  # for a cut: the photos of the gap before it
    path: str = ""
    tooltip: str = ""


class CaptureStrip(QWidget):
    """Paints the strip and reports clicks. Gap g sits between frame g and frame g + 1; gap -1 is
    before the first frame. Each gap between two frames has a cut line before and after each of its
    photos. While a photo is chosen or dragged, every gap shows a + to drop it in."""

    frameClicked = pyqtSignal(int)
    cutClicked = pyqtSignal(int, int)
    photoClicked = pyqtSignal(str)
    slotClicked = pyqtSignal(int)
    segmentClicked = pyqtSignal(int)
    photoDropped = pyqtSignal(str, int)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._frames: list[StripFrame] = []
        self._gaps: dict[int, list[StripPhoto]] = {}
        self._cuts: dict[int, int] = {}
        self._segments: list[str] = []
        self._frame_pixmaps: dict[int, QPixmap] = {}
        self._photo_pixmaps: dict[str, QPixmap] = {}
        self._selected_frame = -1
        self._selected_photo = ""
        self._selected_segment = -1
        self._armed = False
        self._moving = ""
        self._dragging = False
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
    ) -> None:
        """`cuts` maps a cut gap to the number of its photos before the cut; `segments` holds a label
        per segment."""
        self._frames, self._gaps, self._cuts, self._segments = frames, gaps, cuts, segments
        self._selected_frame, self._selected_photo, self._selected_segment = selected_frame, selected_photo, selected_segment
        self._armed, self._moving = armed, moving
        self._relayout()

    def set_frame_pixmap(self, index: int, pixmap: QPixmap) -> None:
        self._frame_pixmaps[index] = pixmap
        self.update()

    def set_photo_pixmap(self, path: str, pixmap: QPixmap) -> None:
        self._photo_pixmaps[path] = pixmap
        self.update()

    def rect_of(self, kind: str, index: int = -2, path: str = "") -> Optional[QRect]:
        """Where a frame or a placed photo is, for scrolling it into view."""
        hit = next((h for h in self._hits if h.kind == kind and (h.path == path if path else h.index == index)), None)
        return hit.rect if hit else None

    def hits(self) -> list[_Hit]:
        return list(self._hits)

    # ── layout ────────────────────────────────────────────────────────────
    def _relayout(self) -> None:
        hits: list[_Hit] = []
        n = len(self._frames)
        slots = self._armed or self._dragging
        spans: dict[int, list[int]] = {}
        x = PAD

        def span(segment: int, left: int, right: int) -> None:
            lo_hi = spans.setdefault(segment, [left, right])
            lo_hi[0], lo_hi[1] = min(lo_hi[0], left), max(lo_hi[1], right)

        def cut(g: int, k: int) -> None:
            nonlocal x
            hits.append(_Hit("cut", QRect(x, TOP, CUT_W, THUMB_H), g, k, tooltip=self._cut_tip(g, k)))
            x += CUT_W + PAD

        def gap(g: int) -> None:
            nonlocal x
            x += PAD
            inner = 0 <= g < n - 1
            photos = self._gaps.get(g, [])
            for k, photo in enumerate(photos):
                if inner:
                    cut(g, k)
                hits.append(_Hit("photo", QRect(x, 0, PHOTO_W, BAR_Y), g, k, photo.path, photo.tooltip))
                span(photo.segment, x, x + PHOTO_W)
                x += PHOTO_W + PAD
            if inner:
                cut(g, len(photos))
            if slots:
                hits.append(_Hit("slot", QRect(x, TOP, SLOT_W, THUMB_H), g, tooltip=self._slot_tip(g)))
                x += SLOT_W + PAD

        gap(-1)
        for i, frame in enumerate(self._frames):
            hits.append(_Hit("frame", QRect(x, 0, FRAME_W, BAR_Y), i, tooltip=frame.tooltip))
            span(frame.segment, x, x + FRAME_W)
            x += FRAME_W
            gap(i)
        for segment, (left, right) in sorted(spans.items()):
            label = self._segments[segment] if segment < len(self._segments) else ""
            hits.append(_Hit("segment", QRect(left, BAR_Y, right - left, BAR_H), segment, tooltip=label))
        self._hits = hits
        self._hover = None
        self.setFixedWidth(x + PAD)
        self.update()

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
            "cut": self._paint_cut,
            "slot": self._paint_slot,
            "segment": self._paint_segment,
        }
        for hit in self._hits:
            painters[hit.kind](p, hit)
        p.end()

    def _thumb(self, p: QPainter, rect: QRect, pixmap: Optional[QPixmap]) -> None:
        inner = rect.adjusted(2, 2, -2, -2)
        if pixmap is None or pixmap.isNull():
            p.fillRect(inner, QColor(THEME.bg_input))
            return
        scaled = pixmap.scaled(inner.size(), Qt.AspectRatioMode.KeepAspectRatioByExpanding, Qt.TransformationMode.SmoothTransformation)
        sx, sy = (scaled.width() - inner.width()) // 2, (scaled.height() - inner.height()) // 2
        p.drawPixmap(inner, scaled, QRect(sx, sy, inner.width(), inner.height()))

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
        thumb = QRect(hit.rect.x(), TOP, FRAME_W, THUMB_H)
        p.setOpacity(0.35 if frame.muted else (0.45 if frame.rejected else 1.0))
        self._thumb(p, thumb, self._frame_pixmaps.get(hit.index))
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
        self._text(p, hit.rect.x() + 1, NAME_Y, FRAME_W - 2, frame.name, THEME.text_secondary)
        time_color = {"error": THEME.error_text, "warning": THEME.warn_amber}.get(frame.state, THEME.text_primary)
        self._text(p, hit.rect.x() + 1, TIME_Y, FRAME_W - 2, frame.time, time_color)

    def _paint_photo(self, p: QPainter, hit: _Hit) -> None:
        photo = next((ph for ph in self._gaps.get(hit.index, []) if ph.path == hit.path), None)
        if photo is None:
            return
        thumb = QRect(hit.rect.x(), TOP, PHOTO_W, THUMB_H)
        p.setOpacity(0.4 if hit.path == self._moving else 1.0)
        self._thumb(p, thumb, self._photo_pixmaps.get(hit.path))
        self._border(p, thumb, THEME.error if photo.error else THEME.warn_amber, 2)
        if hit.path == self._selected_photo:
            self._border(p, thumb.adjusted(-3, -3, 3, 3), THEME.text_primary, 1)
        self._badge(p, thumb.x() + 3, thumb.y() + 3, self._phone_icon)
        p.setOpacity(1.0)
        self._text(p, hit.rect.x() + 1, NAME_Y, PHOTO_W - 2, photo.time, THEME.error_text if photo.error else THEME.warn_amber)
        self._text(p, hit.rect.x() + 1, TIME_Y, PHOTO_W - 2, photo.name, THEME.text_hint)

    def _paint_cut(self, p: QPainter, hit: _Hit) -> None:
        x = hit.rect.center().x()
        if self._cuts.get(hit.index) == hit.sub:
            p.setPen(QPen(QColor(THEME.border_checked_hover), 3))
            p.drawLine(x, TOP, x, BAR_Y + BAR_H)
            p.setBrush(QColor(THEME.bg_card))
            p.drawEllipse(QPoint(x, TOP + THUMB_H // 2), 4, 4)
            return
        p.setPen(QPen(QColor(THEME.border_focus if self._hovered(hit) else THEME.border_input), 2))
        p.drawLine(x, TOP + 6, x, TOP + THUMB_H - 6)

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
        inner = hit.rect.adjusted(THEME.space_md, 0, -THEME.space_md, 0)
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
        if hit is None:
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
        if not self._dragged_path(event):
            event.ignore()
            return
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
        self._drop_gap = None
        self._relayout()
