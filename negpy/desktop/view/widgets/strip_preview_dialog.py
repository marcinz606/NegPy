"""Modal pop-up: preview each frame of a strip, set a per-frame window and pick
which frames to scan.

Read after ``exec()`` via ``selected_frames()`` / ``frame_windows()`` /
``frame_offset()``.
"""

import numpy as np
import qtawesome as qta
from PyQt6.QtCore import Qt, pyqtSlot
from PyQt6.QtGui import QPixmap, QTransform
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from negpy.desktop.converters import ImageConverter
from negpy.desktop.view.styles.theme import THEME
from negpy.desktop.view.widgets.scan_window_label import ScanWindowLabel
from negpy.desktop.workers.scan_worker import RollPreviewRequest
from negpy.infrastructure.scanners.base import ScannerDevice
from negpy.infrastructure.scanners.params import clamp_frame_offset_mm
from negpy.infrastructure.scanners.roll import effective_pitch_mm

_CLAMP_NOTICE = "Offset held at the frame pitch"
_CUT_NOTICE = "Offset cuts into the frame"
# 135 full frame. Delivery ends one pitch past the frame start, so offset beyond
# (pitch - frame) discards that much picture off the frame tail.
_FRAME_LEN_MM = 36.0
_PREVIEW_FALLBACK_DPI = 500  # only when the device reports no DPI list at all
_TILE_H = 140  # constant tile height; width follows the device aspect
_TILES_PER_ROW = 6  # one SA-21 strip per row; roll adapters (up to 40 frames) wrap below

# The LS-50 raster is portrait (feed axis vertical); rotate each preview -90° so the
# frame reads landscape. QTransform().rotate(-90) maps a scan point (fx, fy) →
# display (fy, 1 - fx) — pinned against Qt — so crop rects round-trip exactly and the
# feed-axis start lands on the display's LEFT edge: tiles 1..N laid left-to-right read
# continuously, like the physical strip (+90 mirrors the feed axis within each tile).
_DISPLAY_ROTATION_DEG = -90


def _preview_positive(rgb: np.ndarray) -> np.ndarray:
    """Cheap negative→positive for the strip preview: per-channel invert + auto-level.

    Not the real develop pipeline — just enough to read the scene through the
    orange mask. Each channel is inverted and stretched between its 1st/99th
    percentiles, which both flips the negative and neutralizes the base cast.
    """
    a = rgb.astype(np.float32)
    if a.ndim == 2:
        a = a[:, :, None]
    out = np.empty_like(a)
    for c in range(a.shape[2]):
        ch = a[..., c]
        lo, hi = np.percentile(ch, 1), np.percentile(ch, 99)
        out[..., c] = 0.0 if hi <= lo else np.clip((hi - ch) / (hi - lo), 0.0, 1.0) * 255.0
    return out.astype(np.uint8)


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


def _order(a: float, b: float) -> tuple[float, float]:
    return (a, b) if a <= b else (b, a)


def _scan_to_display_rect(rect):
    """Scan-space window (fx, fy) → the rotated (landscape) display's coordinates."""
    sx1, sy1, sx2, sy2 = rect
    dx1, dx2 = _order(sy1, sy2)
    dy1, dy2 = _order(1 - sx1, 1 - sx2)
    return (_clamp01(dx1), _clamp01(dy1), _clamp01(dx2), _clamp01(dy2))


def _display_to_scan_rect(rect):
    """Rotated (landscape) display window → scan-space (what the backend crops with)."""
    dx1, dy1, dx2, dy2 = rect
    sx1, sx2 = _order(1 - dy1, 1 - dy2)
    sy1, sy2 = _order(dx1, dx2)
    return (_clamp01(sx1), _clamp01(sy1), _clamp01(sx2), _clamp01(sy2))


class _ResetSlider(QSlider):
    """Horizontal QSlider that resets to a default on double-click (matches BaseSlider UX)."""

    def __init__(self, default: int = 0) -> None:
        super().__init__(Qt.Orientation.Horizontal)
        self._default = default

    def mouseDoubleClickEvent(self, _event) -> None:
        self.setValue(self._default)


class _Tile:
    """One strip position: its preview label and include box."""

    def __init__(self, frame: int, label: ScanWindowLabel, checkbox: QCheckBox, preview_btn: QPushButton, widget: QWidget) -> None:
        self.frame = frame
        self.previewed_offset: float | None = None  # offset the shown preview was scanned at
        self.label = label
        self.checkbox = checkbox
        self.preview_btn = preview_btn
        self.widget = widget


class StripPreviewDialog(QDialog):
    """Preview each frame of a strip; set a per-frame window and frame selection."""

    def __init__(
        self,
        controller,
        device: ScannerDevice,
        initial_windows=None,
        initial_selected=None,
        initial_offset: float = 0.0,
        initial_offset_modifier: float = 0.0,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._controller = controller
        self._device = device
        self._caps = device.capabilities
        self._capacity = max(1, self._caps.adapter_frame_capacity or 1)
        # Landscape tile aspect (W/H) from the rotated raster: the feed axis (max_area_mm[1])
        # becomes horizontal. Tiles are constant-size at this aspect.
        mm = self._caps.max_area_mm
        self._tile_aspect = (mm[1] / mm[0]) if (mm and len(mm) > 1 and mm[0]) else 1.5
        self._previewing = False
        self._failed_frames: list[int] = []
        self._scan_now = False  # set when the user chooses "Scan" over "Use"
        initial_windows = initial_windows or {}
        initial_selected = tuple(initial_selected or ())
        self.setWindowTitle("Preview strip — set a window per frame")
        self.setModal(True)
        tile_w, tile_h = self._tile_size()
        cols = min(self._capacity, _TILES_PER_ROW)
        rows = -(-self._capacity // _TILES_PER_ROW)
        self.resize(cols * (tile_w + 4) + 36, min(rows, 3) * (tile_h + 4) + 260)

        layout = QVBoxLayout(self)

        help_lbl = QLabel(
            "Preview each frame (the eye button on a tile, or Preview all). Drag on a previewed "
            "frame to crop it — a corner to resize, inside to move; each frame keeps its own window. "
            "Offset slides every frame along the film to clear the inter-frame gap — frames shift "
            "left as it grows, live; the shaded band on the right is film past the frame boundary "
            "the transport cannot deliver (offset past the gap costs frame tail). Drift adds "
            "progressively more (or less) offset per frame position; re-preview to refresh the pixels. "
            "Tick the frames to scan, then Use (apply and return) "
            "or Scan (start scanning now)."
        )
        help_lbl.setWordWrap(True)
        help_lbl.setStyleSheet(
            f"color: {THEME.text_secondary}; font-size: {THEME.font_size_small}px;"
            f" background: rgba(255,255,255,0.04); border-radius: 6px; padding: 6px 8px;"
        )
        layout.addWidget(help_lbl)

        top = QHBoxLayout()
        top.addWidget(QLabel("Offset"))
        self.offset_slider = _ResetSlider()
        self.offset_slider.setRange(0, 100)  # tenths of a mm → 0..10.0 mm
        self.offset_slider.setSingleStep(1)
        self.offset_slider.setPageStep(5)
        self.offset_slider.setFixedWidth(160)
        self.offset_slider.setValue(int(round(max(0.0, float(initial_offset)) * 10)))
        self.offset_slider.setToolTip("Feed-axis offset applied to every frame (the transport cannot back up)")
        top.addWidget(self.offset_slider)
        self.offset_label = QLabel()
        top.addWidget(self.offset_label)
        top.addSpacing(16)
        top.addWidget(QLabel("Drift"))
        self.drift_slider = _ResetSlider()
        self.drift_slider.setRange(-250, 250)  # hundredths of a mm → ±2.50 mm/frame
        self.drift_slider.setSingleStep(1)
        self.drift_slider.setPageStep(10)
        self.drift_slider.setFixedWidth(160)
        self.drift_slider.setValue(int(round(float(initial_offset_modifier) * 100)))
        self.drift_slider.setToolTip(
            "Extra offset added per frame position (mm/frame) — corrects progressive frame-gap drift along the strip"
        )
        top.addWidget(self.drift_slider)
        self.drift_label = QLabel()
        top.addWidget(self.drift_label)
        top.addSpacing(16)
        top.addWidget(QLabel("Preview DPI"))
        self.preview_dpi_combo = QComboBox()
        for dpi in sorted(self._caps.supported_dpi) or [_PREVIEW_FALLBACK_DPI]:
            self.preview_dpi_combo.addItem(str(dpi), dpi)
        self.preview_dpi_combo.setCurrentIndex(0)  # lowest: fastest, framing only
        self.preview_dpi_combo.setToolTip("Resolution used for the preview scans")
        top.addWidget(self.preview_dpi_combo)
        top.addStretch()
        self.preview_all_btn = QPushButton(qta.icon("fa5s.eye", color=THEME.text_primary), " Preview all")
        self.preview_all_btn.clicked.connect(self._on_preview_all)
        top.addWidget(self.preview_all_btn)
        layout.addLayout(top)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        container = QWidget()
        strip = QGridLayout(container)
        strip.setContentsMargins(2, 2, 2, 2)
        strip.setSpacing(4)
        self._tiles: dict[int, _Tile] = {}
        for frame in range(1, self._capacity + 1):
            checked = (frame in initial_selected) if initial_selected else True
            tile = self._build_tile(frame, initial_windows.get(frame), checked)
            self._tiles[frame] = tile
            strip.addWidget(tile.widget, (frame - 1) // _TILES_PER_ROW, (frame - 1) % _TILES_PER_ROW)
        # Pin the grid top-left so a partial last row doesn't spread across the viewport.
        strip.setColumnStretch(cols, 1)
        strip.setRowStretch(rows, 1)
        self._scroll.setWidget(container)
        layout.addWidget(self._scroll, 1)

        self.status = QLabel("")  # live status only (previewing / errors); help moved to the top box
        self.status.setWordWrap(True)
        self.status.setStyleSheet(f"color: {THEME.text_muted}; font-size: {THEME.font_size_small}px;")
        layout.addWidget(self.status)

        btns = QHBoxLayout()
        self.clear_btn = QPushButton("Clear all")
        self.clear_btn.setToolTip("Remove every window (scan full frames)")
        self.clear_btn.clicked.connect(self._on_clear_all)
        btns.addWidget(self.clear_btn)
        btns.addStretch()
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btns.addWidget(cancel_btn)
        self.ok_btn = QPushButton("Use")
        self.ok_btn.setDefault(True)
        self.ok_btn.clicked.connect(self.accept)
        btns.addWidget(self.ok_btn)
        self.scan_btn = QPushButton(qta.icon("fa5s.play", color=THEME.text_primary), " Scan")
        self.scan_btn.setToolTip("Scan the ticked frames now with the current settings")
        self.scan_btn.clicked.connect(self._on_scan_clicked)
        btns.addWidget(self.scan_btn)
        layout.addLayout(btns)

        # Connect after ok_btn exists — setChecked during tile build must not fire
        # the enable-check before the button is there.
        for tile in self._tiles.values():
            tile.checkbox.toggled.connect(self._update_ok_enabled)
        self.offset_slider.valueChanged.connect(self._on_offset_changed)
        self.drift_slider.valueChanged.connect(self._on_offset_changed)
        self._on_offset_changed(self.offset_slider.value())
        self._update_ok_enabled()

        controller.scan_roll_preview_ready.connect(self._on_preview_ready)
        controller.scan_roll_preview_finished.connect(self._on_preview_finished)
        controller.scan_error.connect(self._on_error)
        controller.scan_cancelled.connect(self._on_cancelled)

    def _build_tile(self, frame: int, initial_window, checked: bool) -> _Tile:
        """A big landscape preview with a subtle overlay box (frame checkbox + preview)."""
        widget = QWidget()
        grid = QGridLayout(widget)
        grid.setContentsMargins(0, 0, 0, 0)

        label = ScanWindowLabel()
        label.setFixedSize(*self._tile_size())
        label.set_window(_scan_to_display_rect(initial_window) if initial_window else None)
        grid.addWidget(label, 0, 0)

        overlay = QFrame()
        overlay.setObjectName("frameOverlay")
        overlay.setStyleSheet(
            "#frameOverlay { background: rgba(13, 13, 15, 0.55); border-radius: 6px; }"
            f"#frameOverlay QCheckBox {{ color: {THEME.text_primary}; font-size: {THEME.font_size_small}px; }}"
        )
        oh = QHBoxLayout(overlay)
        oh.setContentsMargins(6, 3, 6, 3)
        oh.setSpacing(6)
        checkbox = QCheckBox(str(frame))
        checkbox.setChecked(checked)
        checkbox.setToolTip(f"Scan frame {frame}")
        oh.addWidget(checkbox)
        preview_btn = QPushButton(qta.icon("fa5s.eye", color=THEME.text_secondary), "")
        preview_btn.setToolTip(f"Preview frame {frame}")
        preview_btn.setFlat(True)
        preview_btn.setFixedSize(24, 20)
        preview_btn.clicked.connect(lambda _checked=False, f=frame: self._on_preview_one(f))
        oh.addWidget(preview_btn)
        grid.addWidget(overlay, 0, 0, Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)

        return _Tile(frame, label, checkbox, preview_btn, widget)

    def _tile_size(self) -> tuple[int, int]:
        return int(_TILE_H * self._tile_aspect), _TILE_H

    # ── result getters ────────────────────────────────────────────────

    def selected_frames(self) -> tuple[int, ...]:
        return tuple(f for f in range(1, self._capacity + 1) if self._tiles[f].checkbox.isChecked())

    def frame_windows(self) -> dict:
        return {f: _display_to_scan_rect(t.label.window()) for f, t in self._tiles.items() if t.label.window() is not None}

    def frame_offset(self) -> float:
        return self.offset_slider.value() / 10.0

    def frame_offset_modifier(self) -> float:
        return self.drift_slider.value() / 100.0

    def _frame_pitch(self) -> float:
        """Feed-axis frame pitch (mm) — the length a tile represents. 0.0 when unknown."""
        return effective_pitch_mm(self._caps)

    def _raw_offset_for_frame(self, frame: int) -> float:
        return self.frame_offset() + (frame - 1) * self.frame_offset_modifier()

    def _offset_for_frame(self, frame: int) -> float:
        """Effective offset for a frame position: base + (N-1)·drift, floored at 0 and
        held short of one pitch (the scan blacks out at the frame boundary — below 0 is
        unreachable, past one pitch there is nothing left to scan)."""
        return clamp_frame_offset_mm(self._raw_offset_for_frame(frame), self._frame_pitch())

    def scan_requested(self) -> bool:
        """True when the dialog was accepted via Scan (start now), not Use."""
        return self._scan_now

    # ── ui state ──────────────────────────────────────────────────────

    def _on_scan_clicked(self) -> None:
        self._scan_now = True
        self.accept()

    def _update_ok_enabled(self, *_args) -> None:
        enabled = any(t.checkbox.isChecked() for t in self._tiles.values())
        self.ok_btn.setEnabled(enabled)
        self.scan_btn.setEnabled(enabled)

    def _on_clear_all(self) -> None:
        for tile in self._tiles.values():
            tile.label.clear_window()

    def _set_previewing(self, busy: bool) -> None:
        self.preview_all_btn.setEnabled(not busy)
        for tile in self._tiles.values():
            tile.preview_btn.setEnabled(not busy)

    def _on_offset_changed(self, _value: int) -> None:
        self.offset_label.setText(f"{self.frame_offset():.1f} mm")
        self.drift_label.setText(f"{self.frame_offset_modifier():+.2f} mm/frame")
        self._refresh_offset_indicators()

    def _tile_coverage(self, tile: _Tile) -> tuple[float, float]:
        """Span a raster previewed at x occupies when the slider reads y: (x − y, 1 − y).
        Tile coords are the next scan's raster, so content slides left as the offset
        grows and every raster ends at the blackout boundary (a fixed film position)."""
        pitch = self._frame_pitch()
        y = (self._offset_for_frame(tile.frame) / pitch) if pitch else 0.0
        x = tile.previewed_offset or 0.0
        return (x - y, 1.0 - y)

    def _refresh_offset_indicators(self) -> None:
        pitch = self._frame_pitch()
        clamped: list[int] = []
        cut: list[tuple[int, float]] = []
        for tile in self._tiles.values():
            # The band is the absolute effective offset, from the RIGHT: film past
            # the frame boundary the transport cannot deliver at this offset (the
            # scan blacks out one pitch past every frame start — LS-50 measured).
            # A frame floored at 0 by negative drift pins the line at the edge so
            # the slider visibly acts. Stale rasters re-place per _tile_coverage,
            # so content slides live while the band stays at the raster end.
            indicators: list[tuple[float, str]] = []
            if pitch:
                raw = self._raw_offset_for_frame(tile.frame)
                offset = self._offset_for_frame(tile.frame)
                if offset != raw:
                    clamped.append(tile.frame)
                loss = offset - (pitch - _FRAME_LEN_MM)
                if loss > 0.05:
                    cut.append((tile.frame, loss))
                if offset > 0 or raw < 0:
                    indicators.append((offset / pitch, "right"))
            tile.label.set_offset_indicators(indicators)
            if tile.label.has_frame():
                tile.label.set_coverage(self._tile_coverage(tile))
        if clamped:
            frames = ", ".join(str(f) for f in clamped)
            self.status.setText(f"{_CLAMP_NOTICE} on frame(s) {frames} — reduce Offset or Drift.")
        elif cut:
            frames = ", ".join(str(f) for f, _ in cut)
            worst = max(loss for _, loss in cut)
            self.status.setText(
                f"{_CUT_NOTICE} on frame(s) {frames} — up to {worst:.1f} mm of picture lost off the "
                f"frame tail; reduce Offset, or re-feed the strip for a better registration."
            )
        elif self.status.text().startswith((_CLAMP_NOTICE, _CUT_NOTICE)):
            self.status.clear()

    # ── preview flow (single-flight chain) ────────────────────────────

    def _preview_dpi(self) -> int:
        return int(self.preview_dpi_combo.currentData() or _PREVIEW_FALLBACK_DPI)

    def _on_preview_one(self, frame: int) -> None:
        self._start_preview((frame,))

    def _on_preview_all(self) -> None:
        self._start_preview(tuple(range(1, self._capacity + 1)))

    def _start_preview(self, slots: tuple[int, ...]) -> None:
        if self._previewing:
            return
        self._failed_frames = []
        pitch = self._frame_pitch()
        req = RollPreviewRequest(
            device=self._device,
            slots=slots,
            dpi=self._preview_dpi(),
            # Raw, not clamped: the session holds the transport's own limits and
            # reports back the offset it actually reached.
            offsets={f: (self._raw_offset_for_frame(f) / pitch if pitch else 0.0) for f in slots},
        )
        try:
            self._controller.start_roll_preview(req)
        except Exception as e:
            self.status.setText(f"Scanner busy — {e}")
            return
        self._previewing = True
        self._set_previewing(True)
        self.status.setText(f"Previewing {'frame ' + str(slots[0]) if len(slots) == 1 else f'{len(slots)} frames'}…")

    @pyqtSlot(object)
    def _on_preview_ready(self, preview) -> None:
        """One slot landed. Slot number and effective offset ride on the preview,
        so results need no in-flight bookkeeping and may arrive in any order."""
        tile = self._tiles.get(preview.slot)
        if tile is None:
            return
        if preview.error is not None:
            # One frame glitched (the backend already retried it); the rest of the
            # strip is still coming.
            self._failed_frames.append(preview.slot)
            self.status.setText(f"Frame {preview.slot} failed — continuing…")
            return
        try:
            positive = _preview_positive(preview.rgb)
            pixmap = QPixmap.fromImage(ImageConverter.to_qimage(positive)).transformed(QTransform().rotate(_DISPLAY_ROTATION_DEG))
        except Exception as e:
            self.status.setText(f"Could not display frame {preview.slot}: {e}")
            return
        tile.previewed_offset = preview.offset
        # Anchor the tile to the next scan: a current raster sits flush left and ends
        # at the blackout boundary, so tile fractions are exactly the window fractions
        # the batch applies to an offset scan.
        tile.label.set_frame(pixmap, self._tile_coverage(tile))
        self._refresh_offset_indicators()

    @pyqtSlot()
    def _on_preview_finished(self) -> None:
        self._previewing = False
        self._set_previewing(False)
        if self._failed_frames:
            failed = ", ".join(str(f) for f in self._failed_frames)
            self.status.setText(f"Preview done. Failed frame(s): {failed}")
        else:
            self.status.clear()

    @pyqtSlot(str)
    def _on_error(self, msg) -> None:
        if not self._previewing:
            return
        self._previewing = False
        self._set_previewing(False)
        self.status.setText(f"Preview failed: {msg}")

    @pyqtSlot()
    def _on_cancelled(self) -> None:
        if not self._previewing:
            return
        self._previewing = False
        self._set_previewing(False)
        self.status.setText("Preview cancelled.")

    def closeEvent(self, ev) -> None:
        for signal, slot in (
            (self._controller.scan_roll_preview_ready, self._on_preview_ready),
            (self._controller.scan_roll_preview_finished, self._on_preview_finished),
            (self._controller.scan_error, self._on_error),
            (self._controller.scan_cancelled, self._on_cancelled),
        ):
            try:
                signal.disconnect(slot)
            except (TypeError, RuntimeError):
                pass
        super().closeEvent(ev)
