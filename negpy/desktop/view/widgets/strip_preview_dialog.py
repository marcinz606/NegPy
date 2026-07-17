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
from negpy.desktop.workers.scan_worker import ScanRequest
from negpy.infrastructure.scanners.base import ScannerDevice
from negpy.infrastructure.scanners.params import ScanParams

_PREVIEW_FALLBACK_DPI = 500  # only when the device reports no DPI list at all
_TILE_MIN_H = 140  # floor for tile height before scaling to the window

# The LS-50 raster is portrait (feed axis vertical); rotate each preview 90° so the
# frame reads landscape, as it sits on the strip. QTransform().rotate(90) maps a scan
# point (fx, fy) → display (1 - fy, fx) — pinned against Qt — so a crop rect drawn in
# the rotated view round-trips back to scan geometry exactly, and the feed-axis offset
# (scan top) lands on the display's right edge.
_DISPLAY_ROTATION_DEG = 90


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
    dx1, dx2 = _order(1 - sy1, 1 - sy2)
    dy1, dy2 = _order(sx1, sx2)
    return (_clamp01(dx1), _clamp01(dy1), _clamp01(dx2), _clamp01(dy2))


def _display_to_scan_rect(rect):
    """Rotated (landscape) display window → scan-space (what the backend crops with)."""
    dx1, dy1, dx2, dy2 = rect
    sx1, sx2 = _order(dy1, dy2)
    sy1, sy2 = _order(1 - dx1, 1 - dx2)
    return (_clamp01(sx1), _clamp01(sy1), _clamp01(sx2), _clamp01(sy2))


class _Tile:
    """One strip position: its preview label, include box, and the offset the
    shown preview was scanned at (for the live cut indicator)."""

    def __init__(self, frame: int, label: ScanWindowLabel, checkbox: QCheckBox, preview_btn: QPushButton, widget: QWidget) -> None:
        self.frame = frame
        self.label = label
        self.checkbox = checkbox
        self.preview_btn = preview_btn
        self.widget = widget
        self.previewed_offset = 0.0
        self.previewed_offset_pending = 0.0


class StripPreviewDialog(QDialog):
    """Preview each frame of a strip; set a per-frame window and frame selection."""

    def __init__(
        self,
        controller,
        device: ScannerDevice,
        initial_windows=None,
        initial_selected=None,
        initial_offset: float = 0.0,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._controller = controller
        self._device = device
        self._caps = device.capabilities
        self._capacity = max(1, self._caps.adapter_frame_capacity or 1)
        # Landscape tile aspect (W/H) from the rotated raster: the feed axis (max_area_mm[1])
        # becomes horizontal. Tiles are sized to the window height at this aspect.
        mm = self._caps.max_area_mm
        self._tile_aspect = (mm[1] / mm[0]) if (mm and len(mm) > 1 and mm[0]) else 1.5
        self._inflight_frame: int | None = None
        self._preview_queue: list[int] = []
        self._failed_frames: list[int] = []
        self._scan_now = False  # set when the user chooses "Scan" over "Use"
        initial_windows = initial_windows or {}
        initial_selected = tuple(initial_selected or ())
        self.setWindowTitle("Preview strip — set a window per frame")
        self.setModal(True)
        self.resize(1280, 420)

        layout = QVBoxLayout(self)

        help_lbl = QLabel(
            "Preview each frame (the eye button on a tile, or Preview all). Drag on a previewed "
            "frame to crop it — a corner to resize, inside to move; each frame keeps its own window. "
            "Offset nudges every frame along the feed axis to clear the inter-frame gap — re-preview "
            "after changing it to see the effect. Tick the frames to scan, then Use (apply and return) "
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
        self.offset_slider = QSlider(Qt.Orientation.Horizontal)
        self.offset_slider.setRange(0, 40)  # tenths of a mm → 0..4.0 mm
        self.offset_slider.setSingleStep(1)
        self.offset_slider.setPageStep(5)
        self.offset_slider.setFixedWidth(160)
        self.offset_slider.setValue(int(round(max(0.0, float(initial_offset)) * 10)))
        self.offset_slider.setToolTip("Feed-axis offset applied to every frame")
        top.addWidget(self.offset_slider)
        self.offset_label = QLabel()
        top.addWidget(self.offset_label)
        top.addStretch()
        self.preview_all_btn = QPushButton(qta.icon("fa5s.eye", color=THEME.text_primary), " Preview all")
        self.preview_all_btn.clicked.connect(self._on_preview_all)
        top.addWidget(self.preview_all_btn)
        layout.addLayout(top)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        container = QWidget()
        strip = QHBoxLayout(container)
        strip.setContentsMargins(2, 2, 2, 2)
        strip.setSpacing(4)
        self._tiles: dict[int, _Tile] = {}
        for frame in range(1, self._capacity + 1):
            checked = (frame in initial_selected) if initial_selected else True
            tile = self._build_tile(frame, initial_windows.get(frame), checked)
            self._tiles[frame] = tile
            strip.addWidget(tile.widget)
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
        self._on_offset_changed(self.offset_slider.value())
        self._update_ok_enabled()

        controller.scan_preview_ready.connect(self._on_preview_ready)
        controller.scan_error.connect(self._on_error)
        controller.scan_cancelled.connect(self._on_cancelled)

    def _build_tile(self, frame: int, initial_window, checked: bool) -> _Tile:
        """A big landscape preview with a subtle overlay box (frame checkbox + preview)."""
        widget = QWidget()
        grid = QGridLayout(widget)
        grid.setContentsMargins(0, 0, 0, 0)

        label = ScanWindowLabel()
        label.setMinimumSize(160, 107)  # placeholder; _rescale_tiles fits it to the window height
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

    # ── tile sizing (scale to window height) ──────────────────────────

    def _tile_dims(self, available_h: int) -> tuple[int, int]:
        h = max(_TILE_MIN_H, available_h - 8)  # small padding inside the scroll viewport
        return int(h * self._tile_aspect), h

    def _rescale_tiles(self) -> None:
        w, h = self._tile_dims(self._scroll.viewport().height())
        for tile in self._tiles.values():
            tile.label.setFixedSize(w, h)

    def resizeEvent(self, ev) -> None:
        super().resizeEvent(ev)
        self._rescale_tiles()

    def showEvent(self, ev) -> None:
        super().showEvent(ev)
        self._rescale_tiles()

    # ── result getters ────────────────────────────────────────────────

    def selected_frames(self) -> tuple[int, ...]:
        return tuple(f for f in range(1, self._capacity + 1) if self._tiles[f].checkbox.isChecked())

    def frame_windows(self) -> dict:
        return {f: _display_to_scan_rect(t.label.window()) for f, t in self._tiles.items() if t.label.window() is not None}

    def frame_offset(self) -> float:
        return self.offset_slider.value() / 10.0

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
        self._refresh_offset_indicators()

    def _refresh_offset_indicators(self) -> None:
        extent = self._caps.max_area_mm[1] if self._caps.max_area_mm and len(self._caps.max_area_mm) > 1 else 0.0
        offset = self.frame_offset()
        for tile in self._tiles.values():
            delta = offset - tile.previewed_offset
            # +offset shifts content toward the raster top → the rotated preview's
            # right; new content enters from the LEFT, so the cut band grows from
            # the left edge rightward, tracking the slider (verified on an LS-50).
            tile.label.set_offset_indicator(delta / extent if (extent and delta > 0) else None, edge="left")

    # ── preview flow (single-flight chain) ────────────────────────────

    def _preview_dpi(self) -> int:
        # Lowest supported DPI: previews are for framing only, and the smallest
        # raster is fastest and least prone to transient device I/O on a flaky link.
        dpis = self._caps.supported_dpi
        return min(dpis) if dpis else _PREVIEW_FALLBACK_DPI

    def _on_preview_one(self, frame: int) -> None:
        self._failed_frames = []
        self._preview_queue = [frame]
        self._pump()

    def _on_preview_all(self) -> None:
        self._failed_frames = []
        self._preview_queue = list(range(1, self._capacity + 1))
        self._pump()

    def _pump(self) -> None:
        if self._inflight_frame is not None:
            return
        if not self._preview_queue:
            self._set_previewing(False)
            return
        frame = self._preview_queue.pop(0)
        req = ScanRequest(
            device_id=self._device.id,
            params=ScanParams(
                dpi=self._preview_dpi(),
                depth=8,
                capture_ir=False,
                autofocus=False,
                auto_exposure=False,
                window=None,
                frame_offset_mm=self.frame_offset(),
                frame=frame,
            ),
            output_folder="",
            filename_pattern="",
            output_format="TIFF",
        )
        try:
            self._controller.start_preview(req)
        except Exception as e:
            self._inflight_frame = None
            self._preview_queue.clear()
            self._set_previewing(False)
            self.status.setText(f"Scanner busy — {e}")
            return
        self._inflight_frame = frame
        self._tiles[frame].previewed_offset_pending = self.frame_offset()
        self._set_previewing(True)
        self.status.setText(f"Previewing frame {frame}…")

    @pyqtSlot(object)
    def _on_preview_ready(self, rgb) -> None:
        frame = self._inflight_frame
        if frame is None:
            return
        self._inflight_frame = None
        try:
            positive = _preview_positive(rgb)
            pixmap = QPixmap.fromImage(ImageConverter.to_qimage(positive)).transformed(QTransform().rotate(_DISPLAY_ROTATION_DEG))
        except Exception as e:
            self._preview_queue.clear()
            self._set_previewing(False)
            self.status.setText(f"Could not display frame {frame}: {e}")
            return
        tile = self._tiles[frame]
        tile.label.set_frame(pixmap)
        tile.previewed_offset = tile.previewed_offset_pending
        self._refresh_offset_indicators()
        self._pump()

    @pyqtSlot(str)
    def _on_error(self, msg) -> None:
        if self._inflight_frame is None and not self._preview_queue:
            return
        frame = self._inflight_frame
        self._inflight_frame = None
        if frame is not None:
            self._failed_frames.append(frame)
        if self._preview_queue:
            # One frame glitched (the backend already retried it); don't abort the
            # whole strip — carry on with the rest.
            self.status.setText(f"Frame {frame} failed — continuing…")
            self._pump()
            return
        self._set_previewing(False)
        if self._failed_frames:
            failed = ", ".join(str(f) for f in self._failed_frames)
            self.status.setText(f"Preview done. Failed frame(s): {failed} ({msg})")
        else:
            self.status.setText(f"Preview failed: {msg}")

    @pyqtSlot()
    def _on_cancelled(self) -> None:
        if self._inflight_frame is None and not self._preview_queue:
            return
        self._inflight_frame = None
        self._preview_queue.clear()
        self._set_previewing(False)
        self.status.setText("Preview cancelled.")

    def closeEvent(self, ev) -> None:
        for signal, slot in (
            (self._controller.scan_preview_ready, self._on_preview_ready),
            (self._controller.scan_error, self._on_error),
            (self._controller.scan_cancelled, self._on_cancelled),
        ):
            try:
                signal.disconnect(slot)
            except (TypeError, RuntimeError):
                pass
        super().closeEvent(ev)
