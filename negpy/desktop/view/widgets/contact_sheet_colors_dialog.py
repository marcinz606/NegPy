import colorsys

import numpy as np
from PyQt6.QtCore import QPointF, Qt, QRectF, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QImage, QMouseEvent, QPainter, QPainterPath, QPen, QPixmap
from PyQt6.QtWidgets import QDialog, QHBoxLayout, QPushButton, QVBoxLayout, QWidget

from negpy.desktop.view.styles.theme import THEME


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    if len(h) != 6:
        return (0, 0, 0)
    try:
        return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))
    except ValueError:
        return (0, 0, 0)


def _rgb_to_hex(r: int, g: int, b: int) -> str:
    return f"#{r:02x}{g:02x}{b:02x}"


def _hex_to_hsv(hex_color: str) -> tuple[float, float, float]:
    r, g, b = _hex_to_rgb(hex_color)
    h, s, v = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
    return h * 360.0, s, v


def _hsv_to_hex(h: float, s: float, v: float) -> str:
    r, g, b = colorsys.hsv_to_rgb(h / 360.0, s, v)
    return _rgb_to_hex(int(round(r * 255)), int(round(g * 255)), int(round(b * 255)))


def _sv_pixmap(hue: float, width: int, height: int) -> QPixmap:
    s = np.linspace(0.0, 1.0, width, dtype=np.float32)
    v = np.linspace(1.0, 0.0, height, dtype=np.float32)
    ss, vv = np.meshgrid(s, v)
    hh = np.full_like(ss, hue / 360.0)
    i = (hh * 6.0).astype(np.int32)
    f = hh * 6.0 - i
    i = i % 6
    p = vv * (1.0 - ss)
    q = vv * (1.0 - f * ss)
    t = vv * (1.0 - (1.0 - f) * ss)
    r = np.choose(i, [vv, q, p, p, t, vv])
    g = np.choose(i, [t, vv, vv, q, p, p])
    b = np.choose(i, [p, p, t, vv, vv, q])
    buf = np.ascontiguousarray(np.clip(np.stack((r, g, b), axis=-1) * 255.0, 0, 255).astype(np.uint8))
    img = QImage(buf.data, width, height, width * 3, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(img.copy())


def _hue_pixmap(width: int, height: int) -> QPixmap:
    hues = np.linspace(0.0, 1.0, width, dtype=np.float32)
    rgb_row = np.empty((width, 3), dtype=np.uint8)
    for x, h in enumerate(hues):
        r, g, b = colorsys.hsv_to_rgb(float(h), 1.0, 1.0)
        rgb_row[x] = (int(r * 255), int(g * 255), int(b * 255))
    strip = np.broadcast_to(rgb_row[np.newaxis, :, :], (height, width, 3))
    buf = np.ascontiguousarray(strip)
    img = QImage(buf.data, width, height, width * 3, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(img.copy())


class _SvField(QWidget):
    changed = pyqtSignal()

    _W = 280
    _H = 168
    _R = 6.0

    def __init__(self, parent=None):
        super().__init__(parent)
        self._hue = 0.0
        self._sat = 0.0
        self._val = 1.0
        self._dragging = False
        self._pixmap = QPixmap()
        self.setFixedSize(self._W, self._H)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self._rebuild_pixmap()

    def set_hsv(self, hue: float, sat: float, val: float) -> None:
        hue = max(0.0, min(360.0, hue))
        sat = max(0.0, min(1.0, sat))
        val = max(0.0, min(1.0, val))
        if abs(hue - self._hue) > 0.01 or self._pixmap.isNull():
            self._hue = hue
            self._rebuild_pixmap()
        else:
            self._hue = hue
        self._sat = sat
        self._val = val
        self.update()

    def hex_color(self) -> str:
        return _hsv_to_hex(self._hue, self._sat, self._val)

    def hsv(self) -> tuple[float, float, float]:
        return self._hue, self._sat, self._val

    def _rebuild_pixmap(self) -> None:
        self._pixmap = _sv_pixmap(self._hue, self._W, self._H)

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        path = QPainterPath()
        path.addRoundedRect(QRectF(0.5, 0.5, self._W - 1, self._H - 1), self._R, self._R)
        painter.setClipPath(path)
        if self._pixmap.isNull():
            painter.fillRect(self.rect(), QColor(THEME.bg_header))
        else:
            painter.drawPixmap(0, 0, self._pixmap)
        painter.setClipping(False)
        painter.setPen(QPen(QColor(THEME.border_primary), 1))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPath(path)

        x = int(round(self._sat * (self._W - 1)))
        y = int(round((1.0 - self._val) * (self._H - 1)))
        # Dual ring so the cursor stays readable on any hue.
        for color, width, radius in ((QColor("#000000"), 3, 7), (QColor("#ffffff"), 1.5, 7)):
            painter.setPen(QPen(color, width))
            painter.drawEllipse(QPointF(x, y), radius, radius)

    def _pick(self, pos) -> None:
        x = max(0, min(self._W - 1, pos.x()))
        y = max(0, min(self._H - 1, pos.y()))
        self._sat = x / max(1, self._W - 1)
        self._val = 1.0 - y / max(1, self._H - 1)
        self.update()
        self.changed.emit()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = True
            self._pick(event.position().toPoint())

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._dragging:
            self._pick(event.position().toPoint())

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = False


class _HueBar(QWidget):
    changed = pyqtSignal(float)

    _W = 280
    _H = 14
    _R = 7.0

    def __init__(self, parent=None):
        super().__init__(parent)
        self._hue = 0.0
        self._dragging = False
        self._pixmap = _hue_pixmap(self._W, self._H)
        self.setFixedSize(self._W, self._H + 8)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def set_hue(self, hue: float) -> None:
        self._hue = max(0.0, min(360.0, hue))
        self.update()

    def hue(self) -> float:
        return self._hue

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        y0 = 4.0
        path = QPainterPath()
        path.addRoundedRect(QRectF(0.5, y0, self._W - 1, self._H - 1), self._R, self._R)
        painter.setClipPath(path)
        if not self._pixmap.isNull():
            painter.drawPixmap(0, int(y0), self._pixmap)
        painter.setClipping(False)

        x = self._hue / 360.0 * (self._W - 1)
        # Soft white thumb.
        painter.setPen(QPen(QColor(0, 0, 0, 140), 1))
        painter.setBrush(QColor("#ffffff"))
        painter.drawRoundedRect(QRectF(x - 3, 1, 6, self._H + 6), 2, 2)

    def _pick(self, pos) -> None:
        x = max(0, min(self._W - 1, pos.x()))
        self._hue = x / max(1, self._W - 1) * 360.0
        self.update()
        self.changed.emit(self._hue)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = True
            self._pick(event.position().toPoint())

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._dragging:
            self._pick(event.position().toPoint())

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = False


class _TargetRow(QWidget):
    """Selectable row: swatch + name + hex readout. Click selects the edit target."""

    clicked = pyqtSignal()

    _H = 42

    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self._title = title
        self._hex = "#000000"
        self._selected = False
        self.setFixedHeight(self._H)
        self.setMinimumWidth(132)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def set_color(self, hex_color: str) -> None:
        self._hex = hex_color
        self.update()

    def set_selected(self, selected: bool) -> None:
        self._selected = selected
        self.update()

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = QRectF(0.5, 0.5, self.width() - 1, self.height() - 1)

        # Row background: subtle lift when selected, accent border to mark focus.
        painter.setBrush(QColor(THEME.bg_header if self._selected else THEME.bg_dark))
        painter.setPen(QPen(QColor(THEME.accent_primary if self._selected else THEME.border_primary), 2 if self._selected else 1))
        painter.drawRoundedRect(rect, THEME.radius_md, THEME.radius_md)

        # Swatch square
        sw = 22.0
        sx = 11.0
        sy = (self.height() - sw) / 2
        swatch = QRectF(sx, sy, sw, sw)
        painter.setBrush(QColor(self._hex))
        painter.setPen(QPen(QColor(THEME.border_color), 1))
        painter.drawRoundedRect(swatch, THEME.radius_sm, THEME.radius_sm)

        # Title
        painter.setPen(QColor(THEME.text_primary))
        title_font = QFont(painter.font())
        title_font.setPixelSize(THEME.font_size_small)
        title_font.setWeight(THEME.weight_medium if self._selected else THEME.weight_regular)
        painter.setFont(title_font)
        painter.drawText(QRectF(42, 0, self.width() - 46, self.height()), int(Qt.AlignmentFlag.AlignVCenter), self._title)

        # Hex readout (right-aligned, read-only)
        painter.setPen(QColor(THEME.text_secondary))
        hex_font = QFont("Consolas, Menlo, monospace")
        hex_font.setPixelSize(THEME.font_size_xs)
        painter.setFont(hex_font)
        painter.drawText(
            QRectF(0, 0, self.width() - 12, self.height()),
            int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter),
            self._hex.upper(),
        )

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()


class _MiniSheetPreview(QWidget):
    """Contact-sheet mock so both colours read in context."""

    _W = 280
    _H = 96

    def __init__(self, parent=None):
        super().__init__(parent)
        self._bg = "#000000"
        self._fg = "#ffffff"
        self.setFixedSize(self._W, self._H)

    def set_colors(self, background: str, label_color: str) -> None:
        self._bg = background
        self._fg = label_color
        self.update()

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)

        outer = QPainterPath()
        outer.addRoundedRect(QRectF(0.5, 0.5, self._W - 1, self._H - 1), 6, 6)
        painter.setClipPath(outer)
        painter.fillRect(self.rect(), QColor(self._bg))
        painter.setClipping(False)
        painter.setPen(QPen(QColor(THEME.border_primary), 1))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPath(outer)

        cell_w, cell_h = 108, 46
        band_h = 15
        gap = 16
        margin_x = (self._W - (2 * cell_w + gap)) / 2
        margin_y = 10
        font = QFont(painter.font())
        font.setPixelSize(9)
        painter.setFont(font)

        # Caption band tint mirrors the export: bg blended 15% toward label colour.
        bg = QColor(self._bg)
        fg = QColor(self._fg)
        band = QColor(
            round(0.85 * bg.red() + 0.15 * fg.red()),
            round(0.85 * bg.green() + 0.15 * fg.green()),
            round(0.85 * bg.blue() + 0.15 * fg.blue()),
        )

        for i in range(2):
            x = margin_x + i * (cell_w + gap)
            y = margin_y
            # Soft photo placeholder with a subtle gradient feel (two fills).
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor("#4a4a4a"))
            painter.drawRect(QRectF(x, y, cell_w, cell_h))
            painter.setBrush(QColor("#3a3a3a"))
            painter.drawRect(QRectF(x, y + cell_h * 0.55, cell_w, cell_h * 0.45))
            painter.setBrush(QColor(255, 255, 255, 18))
            painter.drawEllipse(QRectF(x + cell_w * 0.15, y + 5, cell_w * 0.35, cell_h * 0.35))

            # Caption band hugging the photo (card feel), then filename.
            band_y = y + cell_h + 3
            painter.setBrush(band)
            painter.drawRoundedRect(QRectF(x, band_y, cell_w, band_h), 2, 2)
            painter.setPen(QColor(self._fg))
            painter.drawText(
                QRectF(x, band_y, cell_w, band_h),
                int(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter),
                f"IMG_{1001 + i}.NEF",
            )


class ContactSheetColorsDialog(QDialog):
    """Background + label colours — visual HSV picker, no typed values."""

    def __init__(self, background: str, label_color: str, parent=None):
        super().__init__(parent)
        self._background = (background or "#000000").lower()
        self._label_color = (label_color or "#ffffff").lower()
        self._editing_background = True
        self._syncing = False

        self.setWindowTitle("Contact Sheet Colours")
        self.setStyleSheet(f"QDialog {{ background: {THEME.bg_dark}; }}")
        self.setFixedWidth(328)

        root = QVBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 20)
        root.setSpacing(16)

        self._mini = _MiniSheetPreview(self)
        root.addWidget(self._mini, alignment=Qt.AlignmentFlag.AlignHCenter)

        targets = QVBoxLayout()
        targets.setSpacing(6)
        self._bg_row = _TargetRow("Background", self)
        self._label_row = _TargetRow("Labels", self)
        self._bg_row.clicked.connect(lambda: self._select_target(True))
        self._label_row.clicked.connect(lambda: self._select_target(False))
        targets.addWidget(self._bg_row)
        targets.addWidget(self._label_row)
        root.addLayout(targets)

        self._sv = _SvField(self)
        root.addWidget(self._sv, alignment=Qt.AlignmentFlag.AlignHCenter)

        self._hue = _HueBar(self)
        root.addWidget(self._hue, alignment=Qt.AlignmentFlag.AlignHCenter)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)
        btn_row.addStretch()
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        ok = QPushButton("OK")
        ok.setProperty("primary", True)
        ok.setDefault(True)
        ok.clicked.connect(self.accept)
        ok.style().polish(ok)
        btn_row.addWidget(cancel)
        btn_row.addWidget(ok)
        root.addLayout(btn_row)

        self._sv.changed.connect(self._on_picker_changed)
        self._hue.changed.connect(self._on_hue_changed)

        self._refresh_targets()
        self._mini.set_colors(self._background, self._label_color)
        self._select_target(True)

    def _select_target(self, background: bool) -> None:
        self._editing_background = background
        self._bg_row.set_selected(background)
        self._label_row.set_selected(not background)
        self._load_active_into_picker()

    def _active_hex(self) -> str:
        return self._background if self._editing_background else self._label_color

    def _set_active_hex(self, hex_color: str) -> None:
        if self._editing_background:
            self._background = hex_color
        else:
            self._label_color = hex_color

    def _refresh_targets(self) -> None:
        self._bg_row.set_color(self._background)
        self._label_row.set_color(self._label_color)

    def _load_active_into_picker(self) -> None:
        self._syncing = True
        try:
            h, s, v = _hex_to_hsv(self._active_hex())
            if s < 1e-6:
                h = self._hue.hue()
            self._hue.set_hue(h)
            self._sv.set_hsv(h, s, v)
        finally:
            self._syncing = False

    def _on_hue_changed(self, hue: float) -> None:
        if self._syncing:
            return
        self._syncing = True
        try:
            _, s, v = self._sv.hsv()
            self._sv.set_hsv(hue, s, v)
        finally:
            self._syncing = False
        self._commit_picker()

    def _on_picker_changed(self) -> None:
        if self._syncing:
            return
        h, _, _ = self._sv.hsv()
        self._hue.set_hue(h)
        self._commit_picker()

    def _commit_picker(self) -> None:
        self._set_active_hex(self._sv.hex_color())
        self._refresh_targets()
        self._mini.set_colors(self._background, self._label_color)

    def colors(self) -> tuple[str, str]:
        return self._background, self._label_color
