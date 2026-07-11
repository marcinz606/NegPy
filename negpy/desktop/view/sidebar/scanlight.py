"""Scanlight RGB-scan capture sidebar.

Live R/G/B light sliders + per-channel shutter, one-button triplet capture,
film-stock presets, and a live-view preview for framing/focus. Captured
exposures land in the hot folder and are handed to NegPy's RGB-Scan mode, which
aligns + merges + inverts them.
"""

import json
import os
import re
from dataclasses import asdict, fields, replace

import qtawesome as qta
from PyQt6.QtCore import QEvent, QObject, Qt, QTimer, pyqtSlot
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from negpy.desktop.view.sidebar.calibration_window import CalibrationWindow
from negpy.desktop.view.sidebar.live_view_window import LiveViewWindow
from negpy.desktop.view.styles.templates import section_subheader
from negpy.desktop.view.styles.theme import THEME
from negpy.infrastructure.capture.gphoto import default_settings_path
from negpy.infrastructure.capture.settings import ScanlightSettings
from negpy.services.capture.calibration import shutter_seconds
from negpy.services.capture.presets import PresetStore, ScanlightPreset

_CHANNEL_COLORS = {"R": "#E24B4A", "G": "#639922", "B": "#378ADD", "W": "#B4B2A9"}

# Built-in white-light preset (no calibration needed): name → process mode.
# Selecting it switches the panel to a single white-light exposure. B&W and slide/E-6
# share the *same* light (plain white), so they're one preset; which process to run is
# left to NegPy's autodetect ("auto") — the user can still force it in NegPy if needed.
_BUILTIN_WHITE_PRESETS = {"White Light (B&W or Slide Film)": "auto"}


# LED settle before each exposure. Narrowband PWM LEDs reach full brightness in <10 ms
# and the serial set_color round-trip is ~5-20 ms, so 150 ms is a safe margin (the old
# 400 ms was conservative). A fixed tuning constant, not a user/persisted setting.
_LED_SETTLE_S = 0.15


class _NoWheel(QObject):
    """Event filter that swallows wheel events so scrolling can't change a value."""

    def eventFilter(self, obj, event) -> bool:
        return event.type() == QEvent.Type.Wheel or super().eventFilter(obj, event)


class ScanlightSidebar(QWidget):
    """Trichromatic RGB-scan capture panel."""

    def __init__(self, controller) -> None:
        super().__init__()
        self.controller = controller
        self._settings: ScanlightSettings = self._load_settings()
        self._presets = PresetStore(self.controller.session.repo)
        self._scanning = False
        self._camera_verified = False  # "Live View & Scan" is gated until Check confirms camera…
        self._light_verified = False  # …and light, plus a folder + a selected preset
        self._rgb_mode = True  # Scanlight present → RGB (presets + sliders); else normal white-light scan
        self._calibrating_preset = ""  # non-empty while the "+" calibration flow is saving a new preset
        self._magnifier_on = False  # camera focus magnifier state (driven by clicks on the live image)
        self._settings_loaded = False  # have the live camera-setting dropdowns been populated yet?
        self._slider_readouts: dict = {}  # slider → its value label (updated on preset apply, where signals are blocked)
        self._no_wheel = _NoWheel(self)

        self.lv_window = LiveViewWindow(self)
        self.lv_window.closed.connect(self._on_live_view_window_closed)
        self.lv_image = self.lv_window.image

        # Dedicated pop-up for creating a preset by calibration (independent of the
        # scan cockpit, so the very first preset can be made). Live frames are routed
        # to whichever ROI image is active via self._lv_target.
        self.calib_window = CalibrationWindow(self)
        self.calib_window.closed.connect(self._on_calib_window_closed)
        self.calib_window.calibrateRequested.connect(self._on_calibrate_new_preset)
        self._lv_target = self.lv_image  # RoiImageLabel currently fed by the live-view poll

        self._light_debounce = QTimer()
        self._light_debounce.setSingleShot(True)
        self._light_debounce.setInterval(60)
        self._light_debounce.timeout.connect(self._push_light)

        # Live-view: the camera's preview thread rewrites a JPEG; this timer polls it. It
        # runs a bit faster than the frame interval and skips re-decoding an unchanged
        # frame (mtime guard), so new frames show promptly without wasting CPU.
        self._lv_jpeg_path = ""
        self._lv_polls = 0
        self._lv_frames_seen = 0
        self._lv_last_mtime = 0.0
        self._lv_timer = QTimer()
        self._lv_timer.setInterval(80)
        self._lv_timer.timeout.connect(self._refresh_live_view)

        # Auto-connect: poll for a USB camera + the light every few seconds while the panel
        # is visible (paused during live-view/scan — the body grants a single PTP claim).
        self._conn_poll_inflight = False
        self._conn_poll_timer = QTimer()
        self._conn_poll_timer.setInterval(3000)
        self._conn_poll_timer.timeout.connect(self._poll_connection_tick)
        self._conn_poll_timer.start()

        self._init_ui()
        self._connect_signals()
        self._reload_presets()

    # ── settings persistence ──────────────────────────────────────────

    def _load_settings(self) -> ScanlightSettings:
        data = self.controller.session.repo.get_global_setting("scanlight_settings", default={})
        if isinstance(data, dict) and data:
            try:
                # Filter to known fields so a dropped/renamed persisted setting doesn't
                # blow up construction and silently reset everything to defaults.
                known = {f.name for f in fields(ScanlightSettings)}
                return ScanlightSettings(**{k: v for k, v in data.items() if k in known})
            except Exception:
                pass
        return ScanlightSettings.defaults()

    def _save_settings(self) -> None:
        self.controller.session.repo.save_global_setting("scanlight_settings", asdict(self._settings))

    def _gphoto_available(self) -> bool:
        """True when python-gphoto2 is importable. It is an optional dependency (and has no
        Windows build), so this drives the one-time setup hint."""
        import importlib.util

        return importlib.util.find_spec("gphoto2") is not None

    def _refresh_setup_hint(self) -> None:
        """Show the setup note only while python-gphoto2 is missing."""
        self._setup_hint.setVisible(not self._gphoto_available())

    # ── UI construction ───────────────────────────────────────────────

    def _slider_row(self, letter: str, value: int) -> QSlider:
        row = QHBoxLayout()
        tag = QLabel(letter)
        tag.setFixedWidth(14)
        tag.setStyleSheet(f"color: {_CHANNEL_COLORS[letter]}; font-weight: bold;")
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(0, 255)
        slider.setValue(value)
        readout = QLabel(str(value))
        readout.setFixedWidth(28)
        readout.setStyleSheet(f"color: {THEME.text_muted}; font-size: {THEME.font_size_small}px;")
        row.addWidget(tag)
        row.addWidget(slider, 1)
        row.addWidget(readout)
        self._light_layout.addLayout(row)
        self._slider_readouts[slider] = readout  # so preset apply (signals blocked) can refresh it
        slider.valueChanged.connect(lambda v, lbl=readout: lbl.setText(str(v)))
        slider.valueChanged.connect(lambda _v: self._light_debounce.start())
        return slider

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 0, 5, 5)
        layout.setSpacing(10)

        # ── LIVE VIEW & SCAN (primary action — top, gated) ───
        self.lv_btn = QPushButton(qta.icon("fa5s.video", color=THEME.text_primary), " Scan")
        self.lv_btn.setObjectName("scan_btn")
        self.lv_btn.setCheckable(True)
        self.lv_btn.setFixedHeight(44)
        _lv_font = self.lv_btn.font()
        _lv_font.setBold(True)  # make the primary action stand out
        self.lv_btn.setFont(_lv_font)
        layout.addWidget(self.lv_btn)

        # Persistent hint listing what's still missing before you can scan (task 5).
        self.gate_hint = QLabel("")
        self.gate_hint.setStyleSheet(f"color: #C8922E; font-size: {THEME.font_size_small}px;")
        self.gate_hint.setWordWrap(True)
        layout.addWidget(self.gate_hint)

        # ── CAMERA (auto-connect over USB) ─────────────────────────────────
        layout.addWidget(section_subheader("CAMERA"))
        # python-gphoto2 is an optional dependency, so show a one-time setup note while it
        # is missing; it hides once installed and never nags an already-equipped user.
        self._setup_hint = QLabel(
            "Camera scanning needs python-gphoto2, an optional dependency: "
            "`pip install gphoto2` (macOS and Linux — libgphoto2 has no Windows build). "
            "See docs/CAMERA_SCANNING.md."
        )
        self._setup_hint.setWordWrap(True)
        self._setup_hint.setStyleSheet(f"color: #C8922E; font-size: {THEME.font_size_small}px;")
        self._setup_hint.setVisible(not self._gphoto_available())
        layout.addWidget(self._setup_hint)
        conn_hint = QLabel("Connect the camera by USB, in PC Remote mode — it's detected automatically.")
        conn_hint.setWordWrap(True)
        conn_hint.setStyleSheet(f"color: {THEME.text_muted}; font-size: {THEME.font_size_small}px;")
        layout.addWidget(conn_hint)
        status_row = QHBoxLayout()
        self.cam_status = QLabel()
        self.light_status = QLabel()
        self.light_temp = QLabel()  # live LED temperature next to the light status (heat monitoring)
        self.light_temp.setStyleSheet(f"color: {THEME.text_muted}; font-size: {THEME.font_size_small}px;")
        self.light_temp.hide()  # stay hidden until a reading arrives — an empty label still paints a dark #0D0D0D box
        status_row.addWidget(self.cam_status)
        status_row.addWidget(self.light_status)
        status_row.addWidget(self.light_temp)
        status_row.addStretch()
        layout.addLayout(status_row)
        self._set_conn_status(self.cam_status, None, "Cam")
        self._set_conn_status(self.light_status, None, "Light")
        # RGB scanning needs the Scanlight; when it's absent (normal white-light mode) this hint
        # sits with the connection status. Hidden while in RGB mode (the light poll flips it).
        self._rgb_hint = QLabel("You can also connect the Scanlight to scan in RGB.")
        self._rgb_hint.setWordWrap(True)
        self._rgb_hint.setStyleSheet(f"color: {THEME.text_muted}; font-size: {THEME.font_size_small}px;")
        self._rgb_hint.setVisible(False)
        layout.addWidget(self._rgb_hint)
        # Connection / scan status + progress live with the connection area (not as a strip
        # between the Live-View button and the gate hint).
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setFormat("Capturing… %p%")
        layout.addWidget(self.progress_bar)
        self.status_label = QLabel("")
        self.status_label.setStyleSheet(f"color: {THEME.text_muted}; font-size: {THEME.font_size_small}px;")
        self.status_label.setWordWrap(True)
        self.status_label.setVisible(False)
        layout.addWidget(self.status_label)

        # ── OUTPUT (above presets so the folder is noticed) ──
        layout.addWidget(section_subheader("OUTPUT"))
        out_form = QFormLayout()
        out_form.setSpacing(6)
        folder_row = QHBoxLayout()
        self.folder_edit = QLineEdit(self._settings.output_folder)
        self.folder_edit.setPlaceholderText("Hot folder…")
        self.folder_browse = QPushButton("…")
        self.folder_browse.setFixedWidth(32)
        folder_row.addWidget(self.folder_edit)
        folder_row.addWidget(self.folder_browse)
        out_form.addRow("Folder", folder_row)
        self.roll_edit = QLineEdit(self._settings.roll_name)
        self.roll_edit.setToolTip("Roll name — one folder/file name (no / or \\); the frame number is assigned automatically per roll")
        out_form.addRow("Roll", self.roll_edit)
        layout.addLayout(out_form)

        # ── RGB section (Scanlight only) — presets + level sliders + calibration ──
        # Shown when the Scanlight is connected (narrowband RGB scanning); hidden for normal
        # white-light camera scanning, where only Camera + Output are needed (_set_rgb_mode).
        self._rgb_section = QWidget()
        rgb = QVBoxLayout(self._rgb_section)
        rgb.setContentsMargins(0, 0, 0, 0)
        rgb.setSpacing(10)

        rgb.addWidget(section_subheader("PRESET  ·  film stock / light"))
        preset_row = QHBoxLayout()
        self.preset_combo = QComboBox()
        self.preset_combo.setToolTip("Film-stock preset (RGB levels + shutter), or a built-in white-light mode")
        self.preset_new_btn = QPushButton(qta.icon("fa5s.plus", color=THEME.text_secondary), "")
        self.preset_new_btn.setFixedWidth(32)
        self.preset_new_btn.setToolTip("Create a new preset by calibrating on the film base")
        self.preset_save_btn = QPushButton(qta.icon("fa5s.save", color=THEME.text_secondary), "")
        self.preset_save_btn.setFixedWidth(32)
        self.preset_save_btn.setToolTip("Save the current levels + shutters as a preset")
        self.preset_del_btn = QPushButton(qta.icon("fa5s.trash", color=THEME.text_secondary), "")
        self.preset_del_btn.setFixedWidth(32)
        self.preset_del_btn.setToolTip("Delete the selected preset")
        preset_row.addWidget(self.preset_combo, 1)
        preset_row.addWidget(self.preset_new_btn)
        preset_row.addWidget(self.preset_save_btn)
        preset_row.addWidget(self.preset_del_btn)
        rgb.addLayout(preset_row)
        # A one-line note about the current preset (e.g. white-light mode), right under the
        # dropdown — not up in the camera status line. Hidden when it has nothing to say.
        self.preset_hint = QLabel("")
        self.preset_hint.setStyleSheet(f"color: {THEME.text_muted}; font-size: {THEME.font_size_small}px;")
        self.preset_hint.setWordWrap(True)
        self.preset_hint.setVisible(False)
        rgb.addWidget(self.preset_hint)

        rgb.addWidget(section_subheader("LIGHT  ·  level / shutter"))
        self._light_layout = QVBoxLayout()
        self._light_layout.setSpacing(6)
        rgb.addLayout(self._light_layout)
        self.r_slider = self._slider_row("R", self._settings.r_level)
        self.g_slider = self._slider_row("G", self._settings.g_level)
        self.b_slider = self._slider_row("B", self._settings.b_level)
        self.w_slider = self._slider_row("W", self._settings.w_level)
        # ONE shutter, shared across R/G/B/W (set by calibration; the LED sliders do the
        # per-channel balancing) — a single field instead of one per channel.
        shutter_row = QHBoxLayout()
        _sh_tag = QLabel("Shutter")
        _sh_tag.setStyleSheet(f"color: {THEME.text_muted}; font-size: {THEME.font_size_small}px;")
        self.shutter_edit = QLineEdit(self._settings.shutter_r)
        self.shutter_edit.setPlaceholderText("1/100")
        self.shutter_edit.setToolTip("Camera shutter, shared across channels (e.g. 1/100). Blank = leave the camera as set.")
        self.shutter_edit.setFixedWidth(70)
        self.shutter_edit.editingFinished.connect(self._update_settings_from_ui)
        shutter_row.addWidget(_sh_tag)
        shutter_row.addStretch(1)
        shutter_row.addWidget(self.shutter_edit)
        self._light_layout.addLayout(shutter_row)
        # White-light modes (B&W / slide) are built-in presets now — no separate toggle.
        # The Scanlight is auto-detected by its Raspberry Pi Pico USB VID (no port picker).
        self.off_btn = QPushButton("Light off")
        self.off_btn.setToolTip("Turn all Scanlight channels off")
        rgb.addWidget(self.off_btn)
        layout.addWidget(self._rgb_section)

        self._disable_wheel()
        self._apply_gating()
        layout.addStretch()

    def _connect_signals(self) -> None:
        self.off_btn.clicked.connect(self._on_light_off)
        self.folder_browse.clicked.connect(self._on_browse_folder)
        self.lv_btn.toggled.connect(self._on_live_view_toggled)
        self.preset_combo.activated.connect(self._on_preset_selected)
        self.preset_new_btn.clicked.connect(self._on_preset_new)
        self.preset_save_btn.clicked.connect(self._on_preset_save)
        self.preset_del_btn.clicked.connect(self._on_preset_delete)
        for w in (self.roll_edit, self.folder_edit):
            w.editingFinished.connect(self._update_settings_from_ui)

        self.controller.capture_light_set.connect(self._on_light_set)
        self.controller.capture_progress.connect(self._on_progress)
        self.controller.capture_finished.connect(self._on_finished)
        self.controller.capture_cancelled.connect(self._on_cancelled)
        self.controller.capture_error.connect(self._on_error)
        self.controller.capture_status.connect(self._on_status)
        self.controller.capture_live_view_started.connect(self._on_live_view_started)
        self.controller.capture_calibration_progress.connect(self._on_calibration_progress)
        self.controller.capture_calibration_finished.connect(self._on_calibration_finished)
        self.controller.connection_polled.connect(self._on_poll_status)
        self.controller.light_temp_polled.connect(self._on_light_temp)
        # Pop-up toolbar mirrors the panel actions (scan a roll without tab-switching).
        self.lv_window.scanRequested.connect(self._on_scan)
        self.lv_window.retakeRequested.connect(self._on_retake)
        self.lv_image.clicked.connect(self._on_magnifier_click)
        for which, stepper in (
            ("iso", self.lv_window.iso_stepper),
            ("shutter", self.lv_window.shutter_stepper),
            ("aperture", self.lv_window.aperture_stepper),
        ):
            stepper.activated.connect(lambda _i, w=which, c=stepper: self._on_camera_setting(w, c))

    # ── activation hook ───────────────────────────────────────────────

    def on_activated(self) -> None:
        """Called when the Scan tab is switched to — kick an immediate connection poll."""
        self._refresh_setup_hint()  # re-check whether python-gphoto2 is installed
        self._apply_gating()  # refresh the "what's still missing to scan" hint
        self._poll_connection_tick()

    def _disable_wheel(self) -> None:
        """Stop the mouse wheel from changing values (avoids accidental scroll edits)."""
        for widget in (
            self.r_slider,
            self.g_slider,
            self.b_slider,
            self.w_slider,
            self.preset_combo,
        ):
            widget.installEventFilter(self._no_wheel)

    # ── light ─────────────────────────────────────────────────────────

    def _white_framing(self) -> bool:
        """White light for a white-mode preset, or when framing/focusing (live view / calibration)."""
        return self._settings.white_mode or self.lv_btn.isChecked() or self.calib_window.isVisible()

    def _push_light(self) -> None:
        if not self._rgb_mode:
            return  # normal white-light scanning has no Scanlight to control
        if self._white_framing():
            self.controller.set_scanlight_color(0, 0, 0, self.w_slider.value(), self._settings.port)
        else:
            self.controller.set_scanlight_color(self.r_slider.value(), self.g_slider.value(), self.b_slider.value(), 0, self._settings.port)
        self._update_settings_from_ui()

    def _on_light_off(self) -> None:
        self.controller.set_scanlight_color(0, 0, 0, 0, self._settings.port)

    @pyqtSlot(int, int, int, int)
    def _on_light_set(self, r: int, g: int, b: int, w: int) -> None:
        self._set_status(f"Light: W{w}" if w else f"Light: R{r} G{g} B{b}")

    # ── presets ───────────────────────────────────────────────────────

    def _reload_presets(self, select: str = "") -> None:
        self.preset_combo.blockSignals(True)
        self.preset_combo.clear()
        self.preset_combo.addItem("— Select preset —", None)
        for name in _BUILTIN_WHITE_PRESETS:
            self.preset_combo.addItem(name, name)  # built-in white-light modes
        for name in self._presets.names():
            self.preset_combo.addItem(name, name)  # user film-stock (RGB) presets
        if select:
            idx = self.preset_combo.findData(select)
            if idx >= 0:
                self.preset_combo.setCurrentIndex(idx)
        self.preset_combo.blockSignals(False)
        self._refresh_preset_hint()
        self._apply_gating()

    def _preset_selected(self) -> bool:
        return bool(self.preset_combo.currentData())

    def _on_preset_selected(self, _index: int) -> None:
        name = self.preset_combo.currentData()
        if not name:
            self._refresh_preset_hint()  # deselected → clear the note
            self._apply_gating()
            return
        if name in _BUILTIN_WHITE_PRESETS:
            # Built-in white-light mode (single white exposure → B&W or slide/E-6).
            self._settings = replace(self._settings, white_mode=True, white_process_mode=_BUILTIN_WHITE_PRESETS[name])
        else:
            preset = self._presets.get(name)
            if preset is None:
                self._refresh_preset_hint()
                self._apply_gating()
                return
            for slider, value in (
                (self.r_slider, preset.r_level),
                (self.g_slider, preset.g_level),
                (self.b_slider, preset.b_level),
            ):
                slider.blockSignals(True)
                slider.setValue(value)
                slider.blockSignals(False)
                self._slider_readouts[slider].setText(str(value))  # valueChanged was suppressed; refresh the label
            self.shutter_edit.setText(preset.shutter_r)  # one shared shutter (r/g/b are equal)
            self._settings = replace(self._settings, white_mode=False)
        self._refresh_preset_hint()  # note (e.g. white-light) sits under the preset row now
        self._push_light()  # apply the recalled light + persist
        self._apply_gating()

    def _refresh_preset_hint(self) -> None:
        """One-line note under the preset row for the current selection — white-light presets
        do a single exposure. Empty/hidden for RGB film-stock presets or no selection."""
        name = self.preset_combo.currentData()
        self.preset_hint.setText("Single white-light exposure — for B&W or slide film." if name in _BUILTIN_WHITE_PRESETS else "")
        self.preset_hint.setVisible(bool(self.preset_hint.text()))

    def _on_preset_save(self) -> None:
        name, ok = QInputDialog.getText(self, "Save preset", "Film stock name:")
        name = name.strip()
        if not ok or not name or name in _BUILTIN_WHITE_PRESETS:
            return
        self._save_current_as_preset(name)
        self._set_status(f"Saved preset “{name}”.")

    def _save_current_as_preset(self, name: str) -> None:
        self._update_settings_from_ui()
        s = self._settings
        self._presets.save(
            name,
            ScanlightPreset(
                r_level=s.r_level,
                g_level=s.g_level,
                b_level=s.b_level,
                shutter_r=s.shutter_r,
                shutter_g=s.shutter_g,
                shutter_b=s.shutter_b,
            ),
        )
        self._reload_presets(select=name)

    def _on_preset_delete(self) -> None:
        name = self.preset_combo.currentData()
        if not name or name in _BUILTIN_WHITE_PRESETS:
            return
        self._presets.delete(name)
        self._reload_presets()
        self._set_status(f"Deleted preset “{name}”.")

    # ── new preset via calibration (dedicated pop-up) ─────────────────

    def _on_preset_new(self) -> None:
        """Open the dedicated calibration pop-up to make a new preset from the film base."""
        if self.lv_btn.isChecked():
            self.lv_btn.setChecked(False)  # stop the scan live-view (one SDK session)
        self._update_settings_from_ui()
        self._lv_target = self.calib_window.image
        self.calib_window.start()
        self._start_live_view_worker()  # white-light framing for the crosshair
        self._push_light()
        self._set_status("Calibrating a new preset — see the pop-up.")

    def _available_shutters(self) -> tuple[str, ...]:
        """The camera's writable shutter labels (from the live-view settings JSON), fastest-first
        and ≤ 1 s, so calibration solves on *this* body's ladder. Empty → built-in fallback."""
        try:
            with open(default_settings_path()) as f:
                data = json.load(f)
        except (OSError, ValueError):
            return ()
        by_seconds: dict[str, float] = {}
        for o in (data.get("shutter") or {}).get("options", []):
            label = str(o.get("label", "")).strip()
            try:
                seconds = shutter_seconds(label)
            except (TypeError, ValueError):
                continue
            if 0.0 < seconds <= 1.0:
                by_seconds[label] = seconds
        return tuple(sorted(by_seconds, key=by_seconds.__getitem__))

    def _on_calibrate_new_preset(self, name: str) -> None:
        if self._scanning:
            self.calib_window.set_status("A scan is running — wait for it to finish.")
            return
        name = name.strip()
        if not name:
            self.calib_window.set_status("Enter a film-stock name first.")
            return
        roi = self.calib_window.image.roi()
        if roi is None:
            self.calib_window.set_status("Click the clear film base (crosshair) first.")
            return
        # Live view stays up — calibration captures within it (no ~4 s reconnect), like a scan.
        # It's torn down when calibration finishes/fails (_stop_calibration_live_view).
        self._calibrating_preset = name
        self._apply_gating()  # a running calibration locks Scan / Retake
        self._update_settings_from_ui()
        from negpy.desktop.workers.capture_worker import CalibrationRequest

        s = self._settings
        self.calib_window.set_progress(0.0)
        self.controller.start_calibration(
            CalibrationRequest(
                roi=roi,
                output_folder=s.output_folder or "",
                port=s.port,
                settle_s=_LED_SETTLE_S,
                shutter_candidates=self._available_shutters(),
            )
        )

    def _on_calib_window_closed(self) -> None:
        """Cancel: abort any in-progress calibration, stop the calib live-view, and route
        frames back to the scan pop-up."""
        if self._lv_target is self.calib_window.image:
            calibration_running = bool(self._calibrating_preset)
            if calibration_running:
                self.controller.cancel_capture()  # calibration runs in this live view → abort it cleanly
            self.controller.stop_live_view()
            self._lv_timer.stop()
            self._lv_target = self.lv_image
            # Keep the job marker until the worker acknowledges a terminal outcome. The
            # shared capture thread is still occupied, so re-enabling Scan here could queue
            # another frame behind work that has not actually stopped yet.
            if not calibration_running:
                self._calibrating_preset = ""
            self._apply_gating()
            self._push_light()

    # ── live view ─────────────────────────────────────────────────────

    def _on_live_view_toggled(self, on: bool) -> None:
        if on and self.calib_window.isVisible() and not self._calibrating_preset:
            self.calib_window.close()  # only one live-view window at a time
        if on:
            self._settings_loaded = False  # repopulate the camera-setting dropdowns
            self._update_settings_from_ui()
            self._start_live_view_worker()
            self._push_light()  # white light on for focusing under Live View
            self.lv_window.show()
            self.lv_window.raise_()
            self._set_status("Starting live view…")
        else:
            self.controller.stop_live_view()
            self._lv_timer.stop()
            self._lv_target.set_loading(False)  # drop the buffering spinner
            self._reset_magnifier()
            self.lv_window.hide()
            self._push_light()  # back to the capture light (RGB unless white mode)
            self._set_status("")  # clear the "Live view running." line once the stream stops

    def _start_live_view_worker(self) -> None:
        """Spawn the live-view stream subprocess (shared by toggle-on and resume)."""
        self._lv_target.set_loading(True)  # buffering spinner until the first frame lands
        from negpy.desktop.workers.capture_worker import LiveViewRequest

        self.controller.start_live_view(LiveViewRequest())

    @pyqtSlot(str)
    def _on_live_view_started(self, jpeg_path: str) -> None:
        self._lv_jpeg_path = jpeg_path
        self._lv_polls = 0
        self._lv_frames_seen = 0
        # Blank the view and ignore the leftover JPEG from the previous session: pin
        # _lv_last_mtime to the stale file so only a *fresh* frame (newer mtime) is shown.
        self._lv_target.clear_frame()
        try:
            self._lv_last_mtime = os.stat(jpeg_path).st_mtime
        except OSError:
            self._lv_last_mtime = 0.0
        if self._lv_target is self.lv_image:  # scan cockpit (not the calibration pop-up)
            self.lv_window.show()
            self.lv_window.raise_()
        self._lv_timer.start()
        self._set_status("Live view running.")

    def _on_live_view_window_closed(self) -> None:
        if self.lv_btn.isChecked():
            self.lv_btn.setChecked(False)  # stops live view via _on_live_view_toggled(False)

    def _refresh_live_view(self) -> None:
        if not self._lv_jpeg_path:
            return
        # Skip the redundant decode+repaint when the preview thread hasn't written a new
        # frame since the last poll (the poll runs a little faster than frames arrive).
        try:
            mtime = os.stat(self._lv_jpeg_path).st_mtime
        except OSError:
            mtime = 0.0
        if mtime and mtime == self._lv_last_mtime:
            return
        pixmap = QPixmap(self._lv_jpeg_path)
        if pixmap.isNull():
            self._lv_polls += 1
            if self._lv_polls == 50 and self._lv_frames_seen == 0:  # ~4s without a frame
                self._set_status(
                    "No live-view image — is the camera in PC Remote? "
                    "On macOS, quit Preview / Photos / Image Capture — they hold the camera."
                )
                self._lv_target.set_loading(False)  # stop the spinner; the hint explains why
            return
        self._lv_last_mtime = mtime
        self._lv_frames_seen += 1
        self._lv_target.set_frame(pixmap)  # scan pop-up or the calibration window
        if self._lv_target is self.lv_image and self._lv_frames_seen % 12 == 0:
            self._refresh_camera_settings()  # ~1×/s: keep the ISO/shutter/aperture dropdowns fresh

    def _after_capture_live_view(self) -> None:
        """Re-light the preview after a scan. An in-session capture leaves the Scanlight
        off (capture_triplet turns it off in its finally) while the live-view stream keeps
        running, so just push the framing light back. No-op when live view is off."""
        if self.lv_btn.isChecked():
            self._push_light()

    def _stop_calibration_live_view(self) -> None:
        """Tear down the live view a calibration captured inside (Step-1-style, no reconnect)
        once it's done or failed — restores the pre-migration state: LV off, re-enable Scan to
        continue. The calibration window's stream isn't tied to the Scan button, so no gate."""
        self.controller.stop_live_view()
        self._lv_timer.stop()
        self._lv_target.set_loading(False)
        self._reset_magnifier()

    def _reset_magnifier(self) -> None:
        """Forget the magnifier state when the stream stops (the camera resets it too)."""
        self._magnifier_on = False

    def _on_magnifier_click(self, fx: float, fy: float) -> None:
        """Click the live view to magnify at that spot; click again for the full frame.
        Only while the stream is running."""
        if not self.lv_btn.isChecked():
            return
        if self._magnifier_on:
            self._on_magnifier_off()
            return
        x = max(0, min(639, round(fx * 640)))  # 640×480 grid → valid indices 0..639 / 0..479
        y = max(0, min(479, round(fy * 480)))
        self.controller.set_focus_magnifier_pos(x, y)
        self._magnifier_on = True

    def _on_magnifier_off(self) -> None:
        """Back to the full frame."""
        if self._magnifier_on:
            self.controller.set_focus_magnifier(False)
            self._magnifier_on = False
            self._set_status("Full frame — click the image to magnify")

    # ── live camera settings (ISO / shutter / aperture) ──────────

    def _on_camera_setting(self, which: str, combo) -> None:
        raw = combo.currentData()
        if raw is not None:
            self.controller.set_camera_setting(which, int(raw))

    def _refresh_camera_settings(self) -> None:
        """Poll the stream's settings JSON → populate/refresh the ISO/Shutter/aperture steppers."""
        try:
            with open(default_settings_path()) as f:
                data = json.load(f)
        except (OSError, ValueError):
            return
        steppers = {
            "iso": self.lv_window.iso_stepper,
            "shutter": self.lv_window.shutter_stepper,
            "aperture": self.lv_window.aperture_stepper,
        }
        for key, stepper in steppers.items():
            info = data.get(key)
            if not info:  # property unavailable (e.g. aperture on a manual lens)
                stepper.setEnabled(False)
                if not self._settings_loaded:
                    stepper.blockSignals(True)
                    stepper.clear()
                    stepper.addItem("—", None)
                    stepper.blockSignals(False)
                continue
            stepper.setEnabled(bool(info.get("writable", False)))
            if stepper.hasFocus():
                continue  # don't snap the value back while the user is stepping
            options = info.get("options", [])
            if not self._settings_loaded or stepper.count() != len(options):
                stepper.blockSignals(True)
                stepper.clear()
                for o in options:
                    stepper.addItem(o["label"], o["raw"])
                stepper.blockSignals(False)
            idx = stepper.findData(info.get("cur"))
            if idx >= 0 and idx != stepper.currentIndex():
                stepper.blockSignals(True)
                stepper.setCurrentIndex(idx)
                stepper.blockSignals(False)
        self._settings_loaded = True

    # ── calibration (drives the new-preset pop-up) ────────────────────

    @pyqtSlot(float, str)
    def _on_calibration_progress(self, frac: float, msg: str) -> None:
        if self._calibrating_preset:
            self.calib_window.set_progress(frac)
            self.calib_window.set_status(msg)
        else:
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(int(frac * 100))
            self._set_status(msg)

    @pyqtSlot(object)
    def _on_calibration_finished(self, result) -> None:
        self.progress_bar.setVisible(False)
        levels, shutters = result.levels, result.shutters
        self.r_slider.setValue(int(levels[0]))
        self.g_slider.setValue(int(levels[1]))
        self.b_slider.setValue(int(levels[2]))
        self.shutter_edit.setText(shutters[0])  # one shared shutter (all three are equal)
        self._settings = replace(self._settings, white_mode=False)
        self._update_settings_from_ui()
        self._save_settings()
        if self._calibrating_preset:
            name = self._calibrating_preset
            self._calibrating_preset = ""
            self._save_current_as_preset(name)  # persist + reload + select + re-gate
            self._lv_target = self.lv_image
            self.calib_window.hide()
            self._set_status(f"Saved preset “{name}”.")
        else:
            self._set_status("Calibrated — review, then Save as a preset.")
        self._stop_calibration_live_view()  # calibration ran inside live view → tear it down

    # ── browse ────────────────────────────────────────────────────────

    def _on_browse_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select Hot Folder")
        if folder:
            self.folder_edit.setText(folder)
            self._update_settings_from_ui()

    # ── scan ──────────────────────────────────────────────────────────

    def _on_scan(self) -> None:
        if self._scanning:
            self.controller.cancel_capture()
            return
        self._start_capture(retake=False)

    def _on_retake(self) -> None:
        if not self._scanning:
            self._start_capture(retake=True)

    def _last_frame_number(self, folder: str, roll: str) -> int:
        """Highest existing Frame### for `roll` in `folder` (0 if none / unreadable).

        The folder is the source of truth for numbering — a fresh scan takes the next
        number, a retake re-uses this one. Matches the capture filename
        `{roll}_Frame{n:03d}[_R/_G/_B].<raw>`, so the R/G/B triplet counts as one frame.
        """
        pat = re.compile(re.escape(roll) + r"_Frame(\d+)", re.IGNORECASE)
        hi = 0
        try:
            for name in os.listdir(folder):
                m = pat.match(name)
                if m:
                    hi = max(hi, int(m.group(1)))
        except OSError:
            return 0
        return hi

    def _capture_roll_name(self) -> str | None:
        roll = self.roll_edit.text().strip() or "Roll001"
        if roll in {".", ".."} or any(separator in roll for separator in ("/", "\\", "\0")):
            self._set_status('Roll name must be a single safe name (not "." or "..", and no path separators).')
            return None
        return roll

    def _start_capture(self, retake: bool) -> None:
        if self._calibrating_preset:
            # Both ride one worker thread, so this would merely queue — and then fire with
            # the exposure the calibration is in the middle of replacing.
            self._set_status("A calibration is running — wait for it to finish.")
            return
        if self._scanning:
            return  # already capturing; a second click must not queue another frame
        output_folder = self.folder_edit.text().strip()
        if not output_folder:
            self._on_browse_folder()
            output_folder = self.folder_edit.text().strip()
            if not output_folder:
                return

        roll = self._capture_roll_name()
        if roll is None:
            return
        if self.roll_edit.text() != roll:
            self.roll_edit.setText(roll)

        # Capture happens *inside* the live-view session — the body grants one PTP claim, so
        # the preview simply pauses for the shot and resumes. No teardown, no reconnect.
        self._update_settings_from_ui()
        self._save_settings()

        from negpy.desktop.workers.capture_worker import CaptureRequest

        s = self._settings
        roll_folder = os.path.join(output_folder, roll)  # one subfolder per roll
        # Frame numbers are derived from the roll's folder (no manual field): a fresh scan
        # takes the next free number, a retake re-shoots the last one (overwrite). The
        # service creates the subfolder (os.makedirs) before writing.
        last = self._last_frame_number(roll_folder, roll)
        frame_number = max(1, last if retake else last + 1)
        rgb = self._rgb_mode
        req = CaptureRequest(
            roll_name=roll,
            frame_number=frame_number,
            output_folder=roll_folder,
            levels=(s.r_level, s.g_level, s.b_level),
            settle_s=_LED_SETTLE_S,
            port=s.port,
            # Normal mode: no calibrated shutter/white — the operator sets the exposure via the
            # live-view steppers, so leave the shutter blank (the camera keeps its current value).
            shutters=(s.shutter_r, s.shutter_g, s.shutter_b) if rgb else ("", "", ""),
            white_mode=s.white_mode if rgb else False,
            w_level=s.w_level,
            shutter_w=s.shutter_w,
            white_process_mode=s.white_process_mode,
            is_retake=retake,
            rgb_mode=rgb,
        )
        self.set_scanning(True)
        self.controller.start_capture(req)

    @pyqtSlot(float)
    def _on_progress(self, progress: float) -> None:
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(int(progress * 100))

    @pyqtSlot(list)
    def _on_finished(self, paths: list) -> None:
        self.set_scanning(False)
        self.progress_bar.setVisible(False)
        frame = paths[0].split("_Frame")[-1][:3] if paths else ""
        self._set_status(f"Captured frame {frame} — inverting in NegPy…")
        self._after_capture_live_view()  # re-light the still-running preview

    @pyqtSlot()
    def _on_cancelled(self) -> None:
        self.set_scanning(False)
        self.progress_bar.setVisible(False)
        if self._calibrating_preset:
            self._finish_calibration_terminal("Calibration cancelled.")
            self._set_status("Calibration cancelled.")
            return
        self._set_status("Capture cancelled.")
        self._after_capture_live_view()

    def _finish_calibration_terminal(self, status: str) -> None:
        """Restore the scan UI after calibration stops without producing a preset."""
        self._calibrating_preset = ""
        self.calib_window.set_status(status)
        self.calib_window.progress.setVisible(False)
        self._lv_target = self.lv_image
        self._stop_calibration_live_view()
        self._apply_gating()

    @pyqtSlot(str)
    def _on_error(self, msg: str) -> None:
        self.set_scanning(False)
        self.progress_bar.setVisible(False)
        if self._calibrating_preset:
            # New-preset calibration failed: report in the pop-up, drop back to the scan target.
            self._finish_calibration_terminal(f"Calibration failed: {msg}")
        else:
            if self.lv_btn.isChecked():
                # CaptureWorker discards its camera session on errors. Close the frozen
                # preview honestly; the operator can reopen it to establish a fresh session.
                self.lv_btn.setChecked(False)
            self._set_status(f"Error: {msg}")

    @pyqtSlot(str)
    def _on_status(self, msg: str) -> None:
        self._set_status(msg)

    def _set_status(self, text: str) -> None:
        """Show a status line on the panel and mirror it into the live-view pop-up."""
        self.status_label.setText(text)
        self.status_label.setVisible(bool(text))  # collapse the strip when there's no message
        self.lv_window.set_status(text)

    def _set_conn_status(self, label, state, short: str, detail: str = "") -> None:
        """Compact colour-coded dot: green=ok, red=fail, grey=unknown (detail in tooltip)."""
        color = "#1D9E75" if state else ("#E24B4A" if state is False else "#888780")
        label.setText(f"● {short}")
        label.setStyleSheet(f"color: {color}; font-size: {THEME.font_size_small}px;")
        label.setToolTip(detail or short)

    def _poll_connection_tick(self) -> None:
        """Timer tick: while the panel is visible, refresh the camera + light status in the
        background (the auto-connect that replaced 'Check'). Enumerating the USB bus does
        not claim the camera, so this keeps running through live view — that is the only
        way an unplug is noticed while the preview is up."""
        if not self.isVisible():
            return
        self.controller.poll_light_temp(self._settings.port)  # cheap light-only read
        if self._conn_poll_inflight or self._scanning:
            return  # a scan owns the worker thread; the poll would only queue behind it
        self._update_settings_from_ui()
        self._conn_poll_inflight = True
        self.controller.poll_connection(self._settings.port)

    @pyqtSlot(dict)
    def _on_poll_status(self, status: dict) -> None:
        self._conn_poll_inflight = False
        was_verified = self._camera_verified
        self._set_conn_status(self.light_status, status["light_ok"], "Light", f"Scanlight: {status['light_detail']}")
        self._light_verified = status["light_ok"]
        self._set_rgb_mode(status["light_ok"])  # Scanlight present → RGB scanning; absent → normal white-light
        self._camera_verified = bool(status["usb_ok"])
        self._set_cam_status(self._camera_verified, status["usb_model"])
        if self._camera_verified and not was_verified:
            self._set_status("")  # just connected → drop any stale failure line
        elif was_verified and not self._camera_verified and self.lv_btn.isChecked():
            # The body went away mid-stream: close the preview rather than leave the last
            # frame on screen looking live.
            self.lv_btn.setChecked(False)  # → _on_live_view_toggled(False) tears it down
            self._set_status("Camera disconnected.")
        self._apply_gating()

    @pyqtSlot(object)
    def _on_light_temp(self, temp) -> None:
        """Show the live Scanlight LED temperature next to the Light status (amber when warm)."""
        if isinstance(temp, (int, float)):
            color = "#C8922E" if temp >= 55 else THEME.text_muted  # amber once it's getting warm
            self.light_temp.setStyleSheet(f"color: {color}; font-size: {THEME.font_size_small}px;")
            self.light_temp.setText(f"{temp:.0f} °C")
            self.light_temp.show()
        else:
            self.light_temp.setText("")  # no light / no telemetry yet
            self.light_temp.hide()  # hide the widget entirely so no dark placeholder box lingers

    def _set_cam_status(self, ok: bool, model: str) -> None:
        """Camera dot: '● Cam (USB)' when a body answered, '● Cam' when none did."""
        short = "Cam (USB)" if ok else "Cam"
        if ok:
            detail = f"Camera: {model} (USB)" if model else "Camera connected (USB)"
        else:
            detail = "no camera — plug it in over USB, in PC Remote mode"
        self._set_conn_status(self.cam_status, ok, short, detail)

    def _missing_requirements(self) -> list[str]:
        """What still blocks scanning — drives both the gate and the hint. Normal white-light
        scanning needs only camera + folder; RGB scanning additionally needs the Scanlight and
        a film-stock preset."""
        m = []
        # The worker runs one job at a time, so a scan clicked mid-calibration would only
        # queue — and then fire with the exposure the calibration was about to replace.
        if self._calibrating_preset:
            m.append("wait for the calibration to finish")
        if not self._camera_verified:
            m.append("connect the camera")
        if self._rgb_mode:
            if not self._light_verified:
                m.append("connect the Scanlight")
            if not self._preset_selected():
                m.append("select or create a preset")
        if not self.folder_edit.text().strip():
            m.append("choose an output folder")
        return m

    def _apply_gating(self) -> None:
        """“Live View & Scan” needs camera+light+folder+preset; the new-preset (+)
        button only needs camera+light. When scanning is blocked, say why (task 5)."""
        missing = self._missing_requirements()
        can_scan = not missing
        # keep enabled while live view is open so it can be toggled off
        self.lv_btn.setEnabled(can_scan or self.lv_btn.isChecked())
        # Calibration needs camera + light and an idle capture worker.
        can_calibrate = self._camera_verified and self._light_verified and not self._scanning and not self._calibrating_preset
        self.preset_new_btn.setEnabled(can_calibrate)
        self.calib_window.calibrate_btn.setEnabled(can_calibrate)  # the pop-up may already be open
        for btn in (self.lv_window.scan_btn, self.lv_window.retake_btn):
            btn.setEnabled(can_scan)
        if missing:
            self.lv_btn.setToolTip("Can't scan yet — " + "; ".join(missing))
            self.gate_hint.setText("⚠ To scan: " + ", ".join(missing) + ".")
            self.gate_hint.setVisible(True)
        else:
            self.lv_btn.setToolTip("Open the live view to frame, focus and scan")
            self.gate_hint.setText("")
            self.gate_hint.setVisible(False)  # collapse the strip when nothing is missing

    def _set_rgb_mode(self, on: bool) -> None:
        """Switch between RGB (Scanlight) and normal white-light scanning, driven by the
        Scanlight's presence: connected → show presets + level sliders (narrowband triplet);
        absent → hide them + show the hint (one plain white-light shot, only camera + output)."""
        if on == self._rgb_mode:
            return
        self._rgb_mode = on
        self._rgb_section.setVisible(on)
        self._rgb_hint.setVisible(not on)
        if not on:
            self._set_status("")  # drop a lingering "Light: R… G… B…" — there's no Scanlight now
        self._apply_gating()

    # ── state helpers ─────────────────────────────────────────────────

    def set_scanning(self, active: bool) -> None:
        self._scanning = active
        if active:
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
        self.lv_window.set_scanning(active)
        self._apply_gating()  # a running scan locks the "+" calibration button

    def _update_settings_from_ui(self) -> None:
        # white_mode / white_process_mode are set by preset selection, not widgets.
        shutter = self.shutter_edit.text().strip()  # one shutter shared across R/G/B/W
        updated = replace(
            self._settings,
            r_level=self.r_slider.value(),
            g_level=self.g_slider.value(),
            b_level=self.b_slider.value(),
            w_level=self.w_slider.value(),
            shutter_r=shutter,
            shutter_g=shutter,
            shutter_b=shutter,
            shutter_w=shutter,
            roll_name=self.roll_edit.text().strip() or "Roll001",
            output_folder=self.folder_edit.text().strip(),
        )
        if updated == self._settings:
            return  # nothing changed → skip the disk write + re-gate (the 3 s poll calls this each tick)
        self._settings = updated
        self._save_settings()
        self._apply_gating()
