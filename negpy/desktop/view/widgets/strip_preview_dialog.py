"""Modal pop-up: preview each frame of a strip, set a per-frame window and pick
which frames to scan.

Read after ``exec()`` via ``selected_frames()`` / ``frame_windows()`` /
``frame_offset()``.
"""

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

from negpy.kernel.system.text import count_of, plural
from negpy.desktop.converters import ImageConverter
from negpy.desktop.view.styles.templates import StatusStrip, pin_dialog_default
from negpy.desktop.view.styles.theme import THEME
from negpy.desktop.view.widgets.scan_preview_common import RollPreviewSignalsMixin, preview_positive
from negpy.desktop.view.widgets.section_help_dialog import SectionHelpDialog, has_guide
from negpy.desktop.view.widgets.scan_window_label import ScanWindowLabel
from negpy.desktop.workers.scan_worker import RollPreviewRequest
from negpy.infrastructure.scanners.base import ScannerDevice
from negpy.infrastructure.scanners.params import clamp_frame_offset_mm
from negpy.infrastructure.scanners.roll import effective_pitch_mm

_GUIDE_KEY = "scan_strip"  # the <!-- panel: --> marker its ⓘ reads out of the user guide
_CLAMP_NOTICE = "Offset held at the frame pitch"
_CUT_NOTICE = "Offset cuts into the frame"
# 135 full frame. Delivery ends one pitch past the frame start, so an offset beyond
# (pitch - frame) discards that much picture off the frame tail.
_FRAME_LEN_MM = 36.0
_PREVIEW_FALLBACK_DPI = 500  # only when the device reports no DPI list at all
_MAX_MEASURED_OFFSET_TENTHS = 25  # ±2.5 mm, in the slider's tenths of a millimetre
_TILE_H = 140  # constant tile height; width follows the device aspect
_TILE_SLIDER_H = 18  # the per-frame offset slider under each tile
_TILES_PER_ROW = 6  # one SA-21 strip per row; roll adapters (up to 40 frames) wrap below
# A transport that measures the strip reports its frame count only as previews arrive, so ask
# for a roll's worth and keep the tiles it answers with.
_DISCOVERY_SLOTS = 40

# A coolscan3 raster is portrait, with the feed axis vertical, so rotate each preview -90°
# and the frame reads landscape. QTransform().rotate(-90) maps a scan point (fx, fy) to
# display (fy, 1 - fx), pinned against Qt, so crop rects round-trip exactly and the
# feed-axis start lands on the display's LEFT edge. Tiles 1..N laid left to right then
# read continuously, like the physical strip. +90 mirrors the feed axis within each tile.
_DISPLAY_ROTATION_DEG = -90


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


# One line of orientation. Offset and Drift explain themselves on their own sliders, where
# the hand already is, and the ⓘ carries the rest.
_FEEDER_HELP = "Preview a frame, drag on it to crop, and tick the frames to scan."
_DISCOVERY_HELP = "Detect the frames, untick what you do not want, drag on a tile to crop it."

_OFFSET_TIP = (
    "Slides every frame along the film to clear the inter-frame gap. Frames shift left as it "
    "grows; the shaded band is film past the frame boundary the transport cannot deliver, so "
    "offset past the gap costs frame tail."
)
_DRIFT_TIP = "Adds progressively more (or less) offset per frame position, for a strip whose gaps creep. Re-preview to refresh the pixels."
_TILE_OFFSET_TIP = "Corrects this frame alone, on top of Offset and Drift. Double-click to reset."


class _ResetSlider(QSlider):
    """Horizontal QSlider that resets to a default on double-click (matches BaseSlider UX)."""

    def __init__(self, default: int = 0) -> None:
        super().__init__(Qt.Orientation.Horizontal)
        self._default = default

    def mouseDoubleClickEvent(self, _event) -> None:
        self.setValue(self._default)


class _Tile:
    """One strip position: its preview label and include box."""

    def __init__(
        self,
        frame: int,
        label: ScanWindowLabel,
        checkbox: QCheckBox,
        preview_btn: QPushButton,
        offset_slider: "_ResetSlider",
        widget: QWidget,
    ) -> None:
        self.frame = frame
        self.previewed_offset: float | None = None  # offset the shown preview was scanned at
        self.label = label
        self.checkbox = checkbox
        self.preview_btn = preview_btn
        self.offset_slider = offset_slider
        self.widget = widget


class StripPreviewDialog(RollPreviewSignalsMixin, QDialog):
    """Preview each frame of a strip; set a per-frame window and frame selection."""

    def __init__(
        self,
        controller,
        device: ScannerDevice,
        initial_windows=None,
        initial_selected=None,
        initial_offset: float = 0.0,
        initial_offset_modifier: float = 0.0,
        initial_frame_offsets: dict[int, float] | None = None,
        film_format: str | None = None,
        film_type: str = "negative",
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._controller = controller
        self._device = device
        self._film_format = film_format
        self._film_type = film_type
        self._caps = device.capabilities
        # A measured strip has no capacity: tiles grow from what the preview finds.
        self._discovers = self._caps.adapter_frame_capacity is None and self._caps.roll_discovery
        # A measured strip arrives feed-axis-horizontal already, so only a portrait raster turns.
        self._rotation = 0 if self._discovers else _DISPLAY_ROTATION_DEG
        self._capacity = 0 if self._discovers else max(1, self._caps.adapter_frame_capacity or 1)
        # Landscape tile aspect (W/H) from the rotated raster: the feed axis (max_area_mm[1])
        # becomes horizontal. Tiles are constant-size at this aspect.
        mm = self._caps.max_area_mm
        self._tile_aspect = (mm[1] / mm[0]) if (mm and len(mm) > 1 and mm[0]) else 1.5
        self._previewing = False
        self._failed_frames: list[int] = []
        self._scan_now = False  # set when the user chooses "Scan" over "Use"
        initial_windows = initial_windows or {}
        initial_selected = tuple(initial_selected or ())
        self._initial_frame_offsets = dict(initial_frame_offsets or {})
        self.setWindowTitle("Preview strip — set a window per frame")
        self.setModal(True)
        tile_w, tile_h = self._tile_size()
        cols = min(self._capacity or _TILES_PER_ROW, _TILES_PER_ROW)
        rows = max(1, -(-self._capacity // _TILES_PER_ROW))
        self.resize(cols * (tile_w + 4) + 36, min(rows, 3) * (tile_h + _TILE_SLIDER_H + 4) + 260)

        layout = QVBoxLayout(self)

        help_row = QHBoxLayout()
        self.help_lbl = QLabel(_DISCOVERY_HELP if self._discovers else _FEEDER_HELP)
        self.help_lbl.setWordWrap(True)
        self.help_lbl.setStyleSheet(f"color: {THEME.text_secondary}; font-size: {THEME.font_size_small}px;")
        help_row.addWidget(self.help_lbl)
        help_row.addStretch()
        self.help_btn = QPushButton(qta.icon("fa5s.info-circle", color=THEME.text_muted), "")
        self.help_btn.setToolTip("Offset, Drift and cropping, in full")
        self.help_btn.setFlat(True)
        self.help_btn.setFixedSize(24, 22)
        self.help_btn.setVisible(has_guide(_GUIDE_KEY))
        self.help_btn.clicked.connect(lambda: SectionHelpDialog(_GUIDE_KEY, "Strip preview", self).exec())
        help_row.addWidget(self.help_btn)
        layout.addLayout(help_row)

        top = QHBoxLayout()
        top.setSpacing(THEME.space_2xl)

        self.offset_slider = _ResetSlider()
        # A measured strip re-addresses the frame, so its offset may go either way; a feeder
        # cannot back up and blacks out one pitch past the frame start. Both are a correction to
        # a boundary, not a way to reach the next frame, so a measured strip gets the same
        # ±2.5 mm span as Drift.
        self.offset_slider.setRange(
            -_MAX_MEASURED_OFFSET_TENTHS if self._discovers else 0, _MAX_MEASURED_OFFSET_TENTHS if self._discovers else 100
        )
        self.offset_slider.setSingleStep(1)
        self.offset_slider.setPageStep(5)
        self.offset_slider.setMinimumWidth(160)
        # Not floored at 0: a measured strip's saved offset may be negative, and the range
        # clamps it either way.
        self.offset_slider.setValue(int(round(float(initial_offset) * 10)))
        self.offset_slider.setToolTip(_OFFSET_TIP if self._discovers else f"{_OFFSET_TIP} This transport cannot back up.")
        self.offset_label = QLabel()

        self.drift_slider = _ResetSlider()
        self.drift_slider.setRange(-250, 250)  # hundredths of a mm → ±2.50 mm/frame
        self.drift_slider.setSingleStep(1)
        self.drift_slider.setPageStep(10)
        self.drift_slider.setMinimumWidth(160)
        self.drift_slider.setValue(int(round(float(initial_offset_modifier) * 100)))
        self.drift_slider.setToolTip(_DRIFT_TIP)
        self.drift_label = QLabel()

        # Name left, reading right, groove underneath — the panel sliders' shape. Beside the
        # groove the reading either clips or steals the width it is measuring.
        for name, slider, value in (("Offset", self.offset_slider, self.offset_label), ("Drift", self.drift_slider, self.drift_label)):
            block = QVBoxLayout()
            block.setSpacing(0)
            head = QHBoxLayout()
            head.setSpacing(THEME.space_md)
            head.addWidget(QLabel(name))
            head.addStretch()
            value.setStyleSheet(f"color: {THEME.text_secondary};")
            head.addWidget(value)
            block.addLayout(head)
            block.addWidget(slider)
            top.addLayout(block, 1)

        # A measured strip previews out of its own pass, whose resolution nothing chooses.
        self.preview_dpi_label = QLabel("Preview DPI")
        self.preview_dpi_combo = QComboBox()
        for dpi in sorted(self._caps.supported_dpi) or [_PREVIEW_FALLBACK_DPI]:
            self.preview_dpi_combo.addItem(str(dpi), dpi)
        self.preview_dpi_combo.setCurrentIndex(0)  # lowest: fastest, framing only
        self.preview_dpi_combo.setToolTip("Resolution used for the preview scans")
        self.preview_dpi_label.setVisible(not self._discovers)
        self.preview_dpi_combo.setVisible(not self._discovers)
        top.addWidget(self.preview_dpi_label)
        top.addWidget(self.preview_dpi_combo)
        label = " Detect frames" if self._discovers else " Preview all"
        self.preview_all_btn = QPushButton(qta.icon("fa5s.eye", color=THEME.text_primary), label)
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
        self._tiles_wired = False
        self._strip = strip
        self._empty_hint = QLabel("Press Detect frames to measure the strip" if self._discovers else "Preview a frame to set its window")
        self._empty_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty_hint.setStyleSheet(f"color: {THEME.text_hint}; font-size: {THEME.font_size_base}px; padding: 48px;")
        strip.addWidget(self._empty_hint, 0, 0, 1, _TILES_PER_ROW)
        self._initial_windows = initial_windows
        self._initial_selected = initial_selected
        for frame in range(1, self._capacity + 1):
            self._ensure_tile(frame)
        # Pin the grid top-left so a partial last row doesn't spread across the viewport.
        strip.setColumnStretch(cols, 1)
        strip.setRowStretch(rows, 1)
        self._scroll.setWidget(container)
        layout.addWidget(self._scroll, 1)

        # One reserved row: the pass that is running, or the message it left behind.
        self.status_strip = StatusStrip(lines=1)
        layout.addWidget(self.status_strip)

        btns = QHBoxLayout()
        self.select_all_btn = QPushButton("All")
        self.select_all_btn.setFixedWidth(48)
        self.select_all_btn.setToolTip("Scan every frame on the strip")
        self.select_all_btn.clicked.connect(lambda: self._set_all_checked(True))
        self.select_none_btn = QPushButton("None")
        self.select_none_btn.setFixedWidth(56)
        self.select_none_btn.setToolTip("Untick every frame")
        self.select_none_btn.clicked.connect(lambda: self._set_all_checked(False))
        self.selection_label = QLabel()
        self.selection_label.setStyleSheet(f"color: {THEME.text_secondary}; font-size: {THEME.font_size_small}px;")
        btns.addWidget(QLabel("Frames to scan"))
        btns.addWidget(self.select_all_btn)
        btns.addWidget(self.select_none_btn)
        btns.addWidget(self.selection_label)
        btns.addSpacing(16)
        # Not "Clear all": it clears crops, and it sits next to the selection buttons.
        self.clear_btn = QPushButton("Clear crops")
        self.clear_btn.setToolTip("Remove every window (scan full frames)")
        self.clear_btn.clicked.connect(self._on_clear_all)
        btns.addWidget(self.clear_btn)
        btns.addStretch()
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self._on_cancel_clicked)
        btns.addWidget(self.cancel_btn)
        self.ok_btn = QPushButton("Apply framing")
        self.ok_btn.setToolTip("Keep this framing and selection, and return to the Scan panel")
        self.ok_btn.clicked.connect(self.accept)
        btns.addWidget(self.ok_btn)
        self.scan_btn = QPushButton(qta.icon("fa5s.play", color=THEME.text_primary), " Scan")
        self.scan_btn.setToolTip("Scan the ticked frames now with the current settings")
        self.scan_btn.setProperty("primary", True)
        self.scan_btn.clicked.connect(self._on_scan_clicked)
        pin_dialog_default(None, self.clear_btn, self.cancel_btn, self.ok_btn, self.scan_btn)
        btns.addWidget(self.scan_btn)
        layout.addLayout(btns)

        # Connect after ok_btn exists: setChecked during the tile build must not fire the
        # enable-check before the button is there. Tiles built later connect themselves.
        for tile in self._tiles.values():
            tile.checkbox.toggled.connect(self._update_ok_enabled)
        self._tiles_wired = True
        self.offset_slider.valueChanged.connect(self._on_offset_changed)
        self.drift_slider.valueChanged.connect(self._on_offset_changed)
        self._on_offset_changed(self.offset_slider.value())
        self._update_ok_enabled()

        self._connect_preview_signals()

    def _ensure_tile(self, frame: int) -> _Tile:
        """The tile for a strip position, built and placed on first sight of it."""
        tile = self._tiles.get(frame)
        if tile is not None:
            return tile
        checked = (frame in self._initial_selected) if self._initial_selected else True
        tile = self._build_tile(frame, self._initial_windows.get(frame), checked)
        self._tiles[frame] = tile
        self._capacity = max(self._capacity, frame)
        self._empty_hint.setVisible(False)
        self._strip.addWidget(tile.widget, (frame - 1) // _TILES_PER_ROW, (frame - 1) % _TILES_PER_ROW)
        if self._tiles_wired:
            tile.checkbox.toggled.connect(self._update_ok_enabled)
            self._update_ok_enabled()
        return tile

    def _build_tile(self, frame: int, initial_window, checked: bool) -> _Tile:
        """A big landscape preview with a subtle overlay box (frame checkbox + preview)."""
        widget = QWidget()
        grid = QGridLayout(widget)
        grid.setContentsMargins(0, 0, 0, 0)

        label = ScanWindowLabel()
        label.setFixedSize(*self._tile_size())
        label.set_window(self._to_display(initial_window) if initial_window else None)
        grid.addWidget(label, 0, 0)

        overlay = QFrame()
        overlay.setObjectName("frameOverlay")
        # Opaque, not a wash: the tick sits over the picture and has to read on a bright frame
        # as well as a dark one.
        overlay.setStyleSheet(
            "#frameOverlay { background: rgba(13, 13, 15, 0.82); border-radius: 6px; }"
            f"#frameOverlay QCheckBox {{ color: {THEME.text_primary}; font-size: {THEME.font_size_base}px;"
            " font-weight: 600; spacing: 6px; }"
            "#frameOverlay QCheckBox::indicator { width: 16px; height: 16px; }"
        )
        oh = QHBoxLayout(overlay)
        oh.setContentsMargins(7, 4, 7, 4)
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

        offset_slider = _ResetSlider()
        offset_slider.setRange(-_MAX_MEASURED_OFFSET_TENTHS, _MAX_MEASURED_OFFSET_TENTHS)
        offset_slider.setFixedSize(self._tile_size()[0], _TILE_SLIDER_H)
        # Seeded before the connection, so building a tile never runs the refresh against a
        # dialog that is still assembling itself.
        offset_slider.setValue(int(round(self._initial_frame_offsets.get(frame, 0.0) * 10)))
        offset_slider.valueChanged.connect(lambda _v, f=frame: self._on_tile_offset_changed(f))
        grid.addWidget(offset_slider, 1, 0)

        tile = _Tile(frame, label, checkbox, preview_btn, offset_slider, widget)
        self._set_tile_offset_tooltip(tile)
        return tile

    def _tile_size(self) -> tuple[int, int]:
        return int(_TILE_H * self._tile_aspect), _TILE_H

    # ── result getters ────────────────────────────────────────────────

    def selected_frames(self) -> tuple[int, ...]:
        return tuple(sorted(f for f, t in self._tiles.items() if t.checkbox.isChecked()))

    def frame_windows(self) -> dict:
        return {f: self._to_scan(t.label.window()) for f, t in self._tiles.items() if t.label.window() is not None}

    def _to_display(self, rect):
        return _scan_to_display_rect(rect) if self._rotation else rect

    def _to_scan(self, rect):
        return _display_to_scan_rect(rect) if self._rotation else rect

    def frame_offsets(self) -> dict[int, float]:
        """Per-frame corrections, non-zero entries only."""
        return {f: t.offset_slider.value() / 10.0 for f, t in self._tiles.items() if t.offset_slider.value()}

    def frame_offset(self) -> float:
        return self.offset_slider.value() / 10.0

    def frame_offset_modifier(self) -> float:
        return self.drift_slider.value() / 100.0

    def _frame_pitch(self) -> float:
        """Feed-axis frame pitch (mm) — the length a tile represents. 0.0 when unknown."""
        return effective_pitch_mm(self._caps)

    def _frame_delta(self, frame: int) -> float:
        """This frame's own correction. A slot with no tile yet contributes nothing."""
        tile = self._tiles.get(frame)
        return tile.offset_slider.value() / 10.0 if tile else self._initial_frame_offsets.get(frame, 0.0)

    def _raw_offset_for_frame(self, frame: int) -> float:
        return self.frame_offset() + (frame - 1) * self.frame_offset_modifier() + self._frame_delta(frame)

    def _offset_for_frame(self, frame: int) -> float:
        """Effective offset for a frame position: base + (N-1)·drift + the frame's own correction.

        A feeder is floored at 0 and held short of one pitch: it cannot back up, and the scan
        blacks out at the frame boundary. A measured strip re-addresses the frame instead, so
        its offset stands as asked, either way.
        """
        raw = self._raw_offset_for_frame(frame)
        return raw if self._discovers else clamp_frame_offset_mm(raw, self._frame_pitch())

    def scan_requested(self) -> bool:
        """True when the dialog was accepted via Scan (start now), not Use."""
        return self._scan_now

    # ── ui state ──────────────────────────────────────────────────────

    def _on_scan_clicked(self) -> None:
        self._scan_now = True
        self.accept()

    def _on_cancel_clicked(self) -> None:
        """Stop the pass in flight, or leave when there is none: the tiles already in
        hand are worth keeping, so a stopped preview does not close the dialog."""
        if self._previewing:
            self.stop_preview()
            return
        self.reject()

    def _update_ok_enabled(self, *_args) -> None:
        picked = sum(1 for t in self._tiles.values() if t.checkbox.isChecked())
        ready = bool(picked) and not self._previewing
        self.ok_btn.setEnabled(ready)
        self.scan_btn.setEnabled(ready)
        self.scan_btn.setText(f" Scan {count_of(picked, 'frame')}" if picked else " Scan")
        # Enter follows the intent: once frames are measured and ticked, that is scanning them,
        # not walking back to the panel.
        self.scan_btn.setDefault(ready)
        self.ok_btn.setDefault(not ready)
        self.selection_label.setText(f"{picked} of {count_of(len(self._tiles), 'frame')}" if self._tiles else "none yet")

    def _set_all_checked(self, checked: bool) -> None:
        for tile in self._tiles.values():
            tile.checkbox.setChecked(checked)

    def _on_clear_all(self) -> None:
        for tile in self._tiles.values():
            tile.label.clear_window()

    def _set_previewing(self, busy: bool) -> None:
        self.preview_all_btn.setEnabled(not busy)
        for tile in self._tiles.values():
            tile.preview_btn.setEnabled(not busy)
        # Committing mid-pass would hand the batch a unit the preview still holds.
        self.cancel_btn.setText("Stop preview" if busy else "Cancel")
        if busy:
            self.status_strip.start_progress("Previewing… %p%")
        else:
            self.status_strip.stop_progress()
        self._update_ok_enabled()

    def _set_tile_offset_tooltip(self, tile: _Tile) -> None:
        tile.offset_slider.setToolTip(f"Frame {tile.frame}: {tile.offset_slider.value() / 10.0:+.1f} mm. {_TILE_OFFSET_TIP}")

    def _on_tile_offset_changed(self, frame: int) -> None:
        tile = self._tiles.get(frame)
        if tile is not None:
            self._set_tile_offset_tooltip(tile)
        self._on_offset_changed(0)

    def _on_offset_changed(self, _value: int) -> None:
        self.offset_label.setText(f"{self.frame_offset():.1f} mm")
        self.drift_label.setText(f"{self.frame_offset_modifier():+.2f} mm/frame")
        self._refresh_offset_indicators()

    def _tile_coverage(self, tile: _Tile) -> tuple[float, float]:
        """Span a raster previewed at x occupies when the slider reads y: (x − y, 1 − y).
        Tile coords are the next scan's raster, so content slides left as the offset
        grows and every raster ends at the blackout boundary (a fixed film position).

        A measured strip re-addresses the whole frame instead of losing its tail, so its raster
        keeps full length and simply slides: the gap that opens at one edge is the film the next
        scan takes in.
        """
        pitch = self._frame_pitch()
        if self._discovers:
            y = (self._offset_for_frame(tile.frame) / pitch) if pitch else 0.0
            x = tile.previewed_offset or 0.0
            return (x - y, x - y + 1.0)
        y = (self._offset_for_frame(tile.frame) / pitch) if pitch else 0.0
        x = tile.previewed_offset or 0.0
        return (x - y, 1.0 - y)

    def _refresh_offset_indicators(self) -> None:
        if self._discovers:
            # Nothing is cut off, so there is no band to shade: the frame boundary itself is the
            # line, drawn where the raster the operator is looking at now ends.
            for tile in self._tiles.values():
                start, end = self._tile_coverage(tile)
                edge = [(end, "right")] if tile.label.has_frame() and end < 1.0 else []
                edge += [(start, "left")] if tile.label.has_frame() and start > 0.0 else []
                tile.label.set_offset_indicators(edge)
                if tile.label.has_frame():
                    tile.label.set_coverage((start, end))
            return
        pitch = self._frame_pitch()
        clamped: list[int] = []
        cut: list[tuple[int, float]] = []
        for tile in self._tiles.values():
            # The band is the absolute effective offset, from the RIGHT: film past the frame boundary
            # that the transport cannot deliver at this offset, because the scan blacks out one pitch
            # past every frame start. A frame floored at 0 by negative drift pins the line at the edge
            # so the slider visibly acts. Stale rasters re-place per _tile_coverage, so content slides
            # live while the band stays at the raster end.
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
            self.status_strip.set_message(f"{_CLAMP_NOTICE} on {plural(len(clamped), 'frame')} {frames} — reduce Offset or Drift.")
        elif cut:
            frames = ", ".join(str(f) for f, _ in cut)
            worst = max(loss for _, loss in cut)
            self.status_strip.set_message(
                f"{_CUT_NOTICE} on {plural(len(cut), 'frame')} {frames} — up to {worst:.1f} mm of picture lost off the "
                f"frame tail; reduce Offset, or re-feed the strip for a better registration."
            )
        elif self.status_strip.message().startswith((_CLAMP_NOTICE, _CUT_NOTICE)):
            self.status_strip.set_message("")

    # ── preview flow (single-flight chain) ────────────────────────────

    def _preview_dpi(self) -> int:
        return int(self.preview_dpi_combo.currentData() or _PREVIEW_FALLBACK_DPI)

    def _on_preview_one(self, frame: int) -> None:
        self._start_preview((frame,))

    def _on_preview_all(self) -> None:
        # A measured strip answers with the frames it found and ignores the rest.
        slots = _DISCOVERY_SLOTS if self._discovers else self._capacity
        self._start_preview(tuple(range(1, slots + 1)))

    def _start_preview(self, slots: tuple[int, ...]) -> None:
        if self._previewing:
            return
        self._failed_frames = []
        pitch = self._frame_pitch()
        req = RollPreviewRequest(
            device=self._device,
            slots=slots,
            dpi=self._preview_dpi(),
            # Raw, not clamped: the session holds the transport's own limits and reports back the
            # offset it actually reached.
            offsets={f: (self._raw_offset_for_frame(f) / pitch if pitch else 0.0) for f in slots},
            film_format=self._film_format,
            film_type=self._film_type,
        )
        try:
            self._controller.start_roll_preview(req)
        except Exception as e:
            self.status_strip.set_message(f"Scanner busy — {e}")
            return
        self._previewing = True
        self._set_previewing(True)
        if self._discovers and len(slots) > 1:
            # The slot count asked for is a roll's worth, not what the strip holds.
            self.status_strip.set_message("Measuring the strip…")
        else:
            self.status_strip.set_message(f"Previewing {'frame ' + str(slots[0]) if len(slots) == 1 else f'{len(slots)} frames'}…")

    @pyqtSlot(object)
    def _on_preview_ready(self, preview) -> None:
        """One slot landed. Slot number and effective offset ride on the preview,
        so results need no in-flight bookkeeping and may arrive in any order."""
        tile = self._ensure_tile(preview.slot) if self._discovers else self._tiles.get(preview.slot)
        if tile is None:
            return
        if preview.error is not None:
            # One frame glitched, and the backend already retried it. The rest of the strip is still
            # coming.
            self._failed_frames.append(preview.slot)
            self.status_strip.set_message(f"Frame {preview.slot} failed — continuing…")
            return
        try:
            positive = preview_positive(preview.rgb, self._film_type)
            pixmap = QPixmap.fromImage(ImageConverter.to_qimage(positive))
            if self._rotation:
                pixmap = pixmap.transformed(QTransform().rotate(self._rotation))
        except Exception as e:
            self.status_strip.set_message(f"Could not display frame {preview.slot}: {e}")
            return
        tile.previewed_offset = preview.offset
        # Anchor the tile to the next scan: a current raster sits flush left and ends at the
        # blackout boundary, so tile fractions are exactly the window fractions the batch
        # applies to an offset scan.
        tile.label.set_frame(pixmap, self._tile_coverage(tile))
        self._refresh_offset_indicators()

    @pyqtSlot()
    def _on_preview_finished(self) -> None:
        self._previewing = False
        self._set_previewing(False)
        if self._discovers and not self._failed_frames:
            found = len(self._tiles)
            self.status_strip.set_message(
                f"{count_of(found, 'frame')} detected — check the framing before scanning."
                if found
                else "No frames were detected on the loaded film."
            )
            return
        if self._failed_frames:
            failed = ", ".join(str(f) for f in self._failed_frames)
            self.status_strip.set_message(f"Preview done. Failed {plural(len(self._failed_frames), 'frame')}: {failed}")
        else:
            self.status_strip.set_message("")

    @pyqtSlot(str)
    def _on_error(self, msg) -> None:
        if not self._previewing:
            return
        self._previewing = False
        self._set_previewing(False)
        self.status_strip.set_message(f"Preview failed: {msg}")

    @pyqtSlot()
    def _on_cancelled(self) -> None:
        if not self._previewing:
            return
        self._previewing = False
        self._set_previewing(False)
        self.status_strip.set_message("Preview cancelled.")
