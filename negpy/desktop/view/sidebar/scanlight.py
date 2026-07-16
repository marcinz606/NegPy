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
from PyQt6.QtGui import QPixmap, QStandardItemModel
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
from negpy.desktop.view.sidebar.live_view_window import LiveViewWindow, SettingStepper
from negpy.desktop.view.styles.templates import section_subheader
from negpy.desktop.view.styles.theme import THEME
from negpy.infrastructure.capture.gphoto import default_settings_path
from negpy.infrastructure.capture.settings import ScanlightSettings
from negpy.services.capture.calibration import REFERENCE_LEVELS, SHUTTER_CANDIDATES, normalize_start_point, shutter_seconds, usable_ladder
from negpy.services.capture.presets import PresetStore, ScanlightPreset

_CHANNEL_COLORS = {"R": "#E24B4A", "G": "#639922", "B": "#378ADD", "W": "#B4B2A9"}

# Built-in white-light preset (no calibration needed): name → process mode.
# Selecting it switches the panel to a single white-light exposure. B&W and slide/E-6
# share the *same* light (plain white), so they're one preset; which process to run is
# left to NegPy's autodetect ("auto") — the user can still force it in NegPy if needed.
_BUILTIN_WHITE_PRESETS = {"White Light (B&W or Slide Film)": "auto"}

# A dropdown sentinel (not a real preset name — user names are stripped, so a NUL can't collide):
# picking it unlocks the sliders + exposure steppers to build a preset by hand, then Save bakes it.
_MANUAL_PRESET = "\x00create-manual"


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
        self._manual_mode = False  # True while building a preset by hand (sliders + exposure editable)
        self._manual_populate_pending = False  # seed the sidebar exposure steppers from the body once, then let the user drive
        self._calibrating_preset = ""  # non-empty while the "+" calibration flow is saving a new preset
        self._status_pinned = False  # a pinned status (calibration outcome) outranks the light echo
        self._magnifier_on = False  # camera focus magnifier state (driven by clicks on the live image)
        self._settings_loaded = False  # have the live camera-setting dropdowns been populated yet?
        self._slider_readouts: dict = {}  # slider → its value label (updated on preset apply, where signals are blocked)
        self._slider_rows: dict = {}  # slider → its row widget, so the W row can be hidden on an RGB-only Scanlight
        self._light_has_white = True  # does the connected Scanlight have a white LED? (False = v1-v3, RGB-only)
        self._suppress_camera_release = False  # true only mid hand-off between lv_window/calib_window
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

        # Coalesce rapid ISO/shutter/aperture stepping into one verified camera write per setting
        # (each write is ~1-2 s, so writing every intermediate step made 1/5 → 1/125 take ~30 s).
        self._cam_setting_debounce = QTimer()
        self._cam_setting_debounce.setSingleShot(True)
        self._cam_setting_debounce.setInterval(250)
        self._cam_setting_debounce.timeout.connect(self._flush_camera_settings)
        self._cam_pending: dict[str, int] = {}

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
        row_widget = QWidget()  # wrapped so the whole row (W) can be hidden on an RGB-only Scanlight
        row = QHBoxLayout(row_widget)
        row.setContentsMargins(0, 0, 0, 0)
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
        self._light_layout.addWidget(row_widget)
        self._slider_readouts[slider] = readout  # so preset apply (signals blocked) can refresh it
        self._slider_rows[slider] = row_widget
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
        self._conn_hint = QLabel("Connect the camera by USB, in PC Remote mode — it's detected automatically.")
        self._conn_hint.setWordWrap(True)
        self._conn_hint.setStyleSheet(f"color: {THEME.text_muted}; font-size: {THEME.font_size_small}px;")
        layout.addWidget(self._conn_hint)
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
        self._set_conn_status(self.cam_status, None, "Camera")
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
        self.preset_combo.setToolTip(
            "Pick a saved film-stock preset (RGB levels + ISO + shutter + aperture, shown read-only), a "
            "built-in white-light mode, or “Create a manual preset…” to build one by hand"
        )
        self.preset_new_btn = QPushButton(qta.icon("fa5s.plus", color=THEME.text_secondary), "")
        self.preset_new_btn.setFixedWidth(32)
        self.preset_new_btn.setToolTip("Create a preset by calibrating on the film base (auto-meters the exposure)")
        self.preset_save_btn = QPushButton(qta.icon("fa5s.save", color=THEME.text_secondary), "")
        self.preset_save_btn.setFixedWidth(32)
        self.preset_save_btn.setToolTip("Name and save the manual preset you're building (only while in manual-preset mode)")
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
        self.w_slider.setToolTip("White LED — used only by the white-light preset; the Scanlight can't light it together with RGB")
        # ISO / shutter / aperture — the preset's exposure, wrapped so it hides as a unit for a
        # white-light preset (set those in the live view). Each is a ‹ value › stepper: disabled
        # (read-only) while a preset is active — the scan forces these on the body, so a drifted
        # setting can't falsify it — and enabled only in "manual preset" mode, where it steps through
        # this body's own choices (no invalid values). The shutter is normally solved by calibration.
        self._exposure_widget = QWidget()
        _exp = QVBoxLayout(self._exposure_widget)
        _exp.setContentsMargins(0, 0, 0, 0)
        _exp.setSpacing(6)
        self.iso_stepper = SettingStepper()
        self.shutter_stepper = SettingStepper()
        self.aperture_stepper = SettingStepper()
        for _tag_text, _which, _stepper in (
            ("ISO", "iso", self.iso_stepper),
            ("Shutter", "shutter", self.shutter_stepper),
            ("Aperture", "aperture", self.aperture_stepper),
        ):
            _stepper.setEnabled(False)  # read-only until "create a manual preset" unlocks it
            _stepper.setToolTip("Locked to the preset — pick “Create a manual preset” to set it by hand.")
            _stepper.activated.connect(lambda _i, w=_which, s=_stepper: self._on_sidebar_exposure_changed(w, s))
            _row = QHBoxLayout()
            _tag = QLabel(_tag_text)
            _tag.setStyleSheet(f"color: {THEME.text_muted}; font-size: {THEME.font_size_small}px;")
            _row.addWidget(_tag)
            _row.addStretch(1)
            _row.addWidget(_stepper)
            _exp.addLayout(_row)
        self._light_layout.addWidget(self._exposure_widget)
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
            ("iso", self.calib_window.iso_stepper),  # calibration pop-up drives the same camera
            ("aperture", self.calib_window.aperture_stepper),
        ):
            stepper.activated.connect(lambda _i, w=which, c=stepper: self._on_camera_setting(w, c))
        self._apply_gating()  # now that every widget exists, put the preset area in its read-only state

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

    def _push_light(self) -> None:
        if not self._rgb_mode:
            return  # normal white-light scanning has no Scanlight to control
        if self._settings.white_mode:
            # A white-light preset: the W slider is the scan light itself.
            self.controller.set_scanlight_color(0, 0, 0, self.w_slider.value(), self._settings.port)
        else:
            # Any RGB state — a selected preset, manual building, or framing/focusing in live view —
            # lights the R/G/B mix from the sliders. The preset's own values are the framing light too
            # (one light for scanning and focusing), so it works on every Scanlight and never assumes a
            # white LED an RGB-only body lacks. White stays off: the Scanlight can't mix white with RGB.
            self.controller.set_scanlight_color(self.r_slider.value(), self.g_slider.value(), self.b_slider.value(), 0, self._settings.port)
        self._update_settings_from_ui()

    def _on_light_off(self) -> None:
        self.controller.set_scanlight_color(0, 0, 0, 0, self._settings.port)

    @pyqtSlot(int, int, int, int)
    def _on_light_set(self, r: int, g: int, b: int, w: int) -> None:
        # Ambient echo, not a user action: _on_calibration_finished sets the R/G/B sliders, each of
        # which starts the light debounce, and this slot fires ~60 ms later — right after the
        # calibration outcome ("Saved preset … ⚠ over-exposed …") was written. Declining to replace
        # a pinned outcome is what keeps that advice on screen; the slider readouts echo the levels.
        if self._status_pinned:
            return
        self._set_status(f"Light: W{w}" if w else f"Light: R{r} G{g} B{b}")

    # ── presets ───────────────────────────────────────────────────────

    def _reload_presets(self, select: str = "") -> None:
        self.preset_combo.blockSignals(True)
        self.preset_combo.clear()
        self.preset_combo.addItem("— Select preset —", None)
        self.preset_combo.addItem("＋ Create a manual preset…", _MANUAL_PRESET)  # build one by hand
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
        data = self.preset_combo.currentData()
        return bool(data) and data != _MANUAL_PRESET  # the manual-preset action isn't a scannable preset

    def _set_slider(self, slider, value: int) -> None:
        """Set a light slider + its readout without firing valueChanged — preset apply drives the
        sliders itself, and the sliders reflect the *preset*, not the live LED level."""
        slider.blockSignals(True)
        slider.setValue(value)
        slider.blockSignals(False)
        self._slider_readouts[slider].setText(str(value))

    def _show_lone(self, stepper, label: str) -> None:
        """Make a stepper display one fixed value (a preset's baked setting): no options to step
        through, just the recalled label — or blank (shown as “—”) when the preset stores none."""
        stepper.blockSignals(True)
        stepper.clear()
        if label:
            stepper.addItem(label, None)
        stepper.blockSignals(False)

    @staticmethod
    def _stepper_label(stepper) -> str:
        """The stepper's current label as a clean value ('' for the empty “—” placeholder)."""
        text = stepper.currentText().strip()
        return "" if text == "—" else text

    def _apply_preset_exposure(self, iso: str, aperture: str) -> None:
        """Point the active exposure (settings + the read-only ISO/f steppers) at a preset's baked
        values, so a scan forces them. Blank for white-light / no preset — the camera stays free."""
        self._settings = replace(self._settings, iso=iso, aperture=aperture)
        self._show_lone(self.iso_stepper, iso)
        self._show_lone(self.aperture_stepper, aperture)

    def _apply_preset(self, preset) -> None:
        """Show a stored preset read-only: recall its levels, shutter and exposure onto the (disabled)
        sliders + steppers, and point settings at them so a scan reproduces the exact recipe."""
        for slider, value in (
            (self.r_slider, preset.r_level),
            (self.g_slider, preset.g_level),
            (self.b_slider, preset.b_level),
            (self.w_slider, preset.w_level),  # RGB presets store 0 → the white LED stays off
        ):
            self._set_slider(slider, value)
        self._show_lone(self.shutter_stepper, preset.shutter_r)  # one shared shutter (r/g/b are equal)
        self._settings = replace(
            self._settings,
            white_mode=False,
            shutter_r=preset.shutter_r,
            shutter_g=preset.shutter_r,
            shutter_b=preset.shutter_r,
            shutter_w=preset.shutter_r,
        )
        self._apply_preset_exposure(preset.iso, preset.aperture)  # a scan forces these on the body
        self._apply_preset_camera_settings(preset)  # and reflect them in the live view now

    def _set_manual_mode(self, on: bool) -> None:
        """Enter/leave manual-preset mode. On: unlock the sliders + exposure steppers and Save, and
        fill the steppers with the body's real ISO/shutter/aperture choices (so only valid values can
        be picked). Off returns everything to the read-only, preset-driven state. Editability + the
        Save button follow `_manual_mode` in `_refresh_preset_ui` (via `_apply_gating`)."""
        self._manual_mode = on
        if on:
            self._settings = replace(self._settings, white_mode=False)
            self._set_slider(self.w_slider, 0)  # RGB preset → white LED off (the Scanlight can't combine them)
            self._settings_loaded = False  # force a fresh repopulate of the sidebar exposure steppers
            self._manual_populate_pending = True  # fill them from the body once (below), then the user owns them
            self._refresh_camera_settings()
            self._update_settings_from_ui()  # seed settings from the freshly populated steppers + sliders
        self._apply_gating()

    def _on_sidebar_exposure_changed(self, which: str, stepper) -> None:
        """A manual-preset exposure stepper moved: copy its label into settings (what Save bakes and a
        scan forces) and push it to the body via the debounce, so the live view shows the change."""
        label = self._stepper_label(stepper)
        if which == "shutter":
            self._settings = replace(self._settings, shutter_r=label, shutter_g=label, shutter_b=label, shutter_w=label)
        else:
            self._settings = replace(self._settings, **{which: label})
        self._on_camera_setting(which, stepper)  # debounced verified write via the stepper's raw index

    def _on_preset_selected(self, _index: int) -> None:
        name = self.preset_combo.currentData()
        if name == _MANUAL_PRESET:  # the "build one by hand" action, not a stored preset
            if not self._camera_verified:
                # Defensive: the dropdown item is greyed without a camera, but refuse here too — a
                # manual preset's exposure steppers need the body's own ISO/shutter/aperture choices.
                self._set_status("Connect the camera first — a manual preset uses the camera's ISO / shutter / aperture choices.")
                self.preset_combo.setCurrentIndex(0)
                return
            self._set_manual_mode(True)
            self._refresh_preset_hint()
            self._push_light()  # white light to frame by while dialling it in
            return
        if self._manual_mode:
            self._set_manual_mode(False)  # picking a real preset (or nothing) leaves manual mode
        if not name:
            self._apply_preset_exposure("", "")  # no preset → the camera exposure is free again
            self._refresh_preset_hint()  # deselected → clear the note
            self._apply_gating()
            return
        if name in _BUILTIN_WHITE_PRESETS:
            # Built-in white-light mode (single white exposure → B&W or slide/E-6): white on, RGB off.
            self._settings = replace(self._settings, white_mode=True, white_process_mode=_BUILTIN_WHITE_PRESETS[name])
            for slider, value in ((self.r_slider, 0), (self.g_slider, 0), (self.b_slider, 0), (self.w_slider, 255)):
                self._set_slider(slider, value)
            self._show_lone(self.shutter_stepper, "")
            self._apply_preset_exposure("", "")  # white-light doesn't lock exposure — the steppers do
        else:
            preset = self._presets.get(name)
            if preset is None:
                self._apply_preset_exposure("", "")
                self._refresh_preset_hint()
                self._apply_gating()
                return
            self._apply_preset(preset)
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
        if not self._manual_mode:
            return  # Save only stores a hand-built preset — the button is greyed out otherwise
        name, ok = QInputDialog.getText(self, "Save manual preset", "Film stock name:")
        name = name.strip()
        if not ok or not name or name in _BUILTIN_WHITE_PRESETS:
            return
        self._update_settings_from_ui()  # capture the final slider + stepper values
        self._manual_mode = False  # leaving manual mode → the saved preset becomes the read-only selection
        self._save_current_as_preset(name)  # persist + reload + select + re-gate
        saved = self._presets.get(name)
        if saved is not None:
            self._apply_preset(saved)  # show it read-only (lone steppers, disabled sliders)
        self._push_light()
        self._apply_gating()
        self._set_status(f"Saved preset “{name}”.")

    def _save_current_as_preset(self, name: str) -> None:
        self._update_settings_from_ui()
        s = self._settings
        # Bake the active recipe from settings — set by calibration (metered) or the manual-mode
        # steppers. A later scan reproduces it; aperture is blank on a manual lens (set by hand).
        self._presets.save(
            name,
            ScanlightPreset(
                r_level=s.r_level,
                g_level=s.g_level,
                b_level=s.b_level,
                w_level=s.w_level,
                shutter_r=s.shutter_r,
                shutter_g=s.shutter_g,
                shutter_b=s.shutter_b,
                iso=s.iso,
                aperture=s.aperture,
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
        if self._manual_mode:
            self._set_manual_mode(False)  # calibrating supersedes a half-built manual preset
        if self.lv_btn.isChecked():
            self.lv_btn.setChecked(False)  # stop the scan live-view (one SDK session)
        self._update_settings_from_ui()
        self._lv_target = self.calib_window.image
        self.calib_window.start()
        self._start_live_view_worker()  # live-view stream for the crosshair
        # Framing light = the calibration's own fixed start point (REFERENCE_LEVELS), so the crosshair
        # is placed under the very light the probe will begin from — not the previously selected
        # preset's leftover RGB, nor an arbitrary neutral grey. Pushed DIRECTLY, leaving the shared
        # R/G/B/W sliders on the selected preset, so cancelling the calibration (or handing off to the
        # scan window) restores the preset's own light. Calibration overwrites R/G/B on success.
        self.controller.set_scanlight_color(*REFERENCE_LEVELS, 0, self._settings.port)
        self._set_status("Calibrating a new preset — see the pop-up.")

    def _settings_json(self) -> dict:
        """The live-view settings JSON the stream publishes (ISO/shutter/aperture options + current),
        or {} if the stream hasn't written it yet."""
        try:
            with open(default_settings_path()) as f:
                data = json.load(f)
        except (OSError, ValueError):
            return {}
        return data if isinstance(data, dict) else {}

    def _current_setting_label(self, key: str, require_writable: bool = False) -> str:
        """The current label for a live camera setting (from the stream JSON), '' if unavailable.
        `require_writable` skips a read-only property — aperture on a manual lens isn't baked."""
        info = self._settings_json().get(key)
        if not isinstance(info, dict) or (require_writable and not info.get("writable", False)):
            return ""
        cur = info.get("cur")
        for o in info.get("options", []):
            if o.get("raw") == cur:
                return str(o.get("label", ""))
        return ""

    def _apply_active_preset_camera_settings(self) -> None:
        """Push the active RGB preset's baked ISO/aperture to the body (no-op for white/built-in
        presets or no selection). Used when the scan live view comes up after a preset was picked."""
        name = self.preset_combo.currentData()
        if not name or name in _BUILTIN_WHITE_PRESETS:
            return
        preset = self._presets.get(name)
        if preset is not None:
            self._apply_preset_camera_settings(preset)

    def _apply_preset_camera_settings(self, preset) -> None:
        """Set the body's ISO + aperture to the values baked into the RGB preset, so the scan
        matches the calibration. Labels are resolved against the live options (a no-op if the value
        is absent, the body lacks the option, or no camera session is open to receive the write)."""
        data = self._settings_json()
        for key, label in (("iso", preset.iso), ("aperture", preset.aperture)):
            if not label:
                continue
            info = data.get(key)
            if not isinstance(info, dict):
                continue
            for o in info.get("options", []):
                if str(o.get("label", "")) == label:
                    self.controller.set_camera_setting(key, int(o["raw"]))
                    break

    def _available_shutters(self) -> tuple[str, ...]:
        """The camera's writable shutter labels (from the live-view settings JSON), fastest-first,
        clamped to the solver's own ladder span, so calibration solves on *this* body's ladder.

        The bounds are read from SHUTTER_CANDIDATES rather than repeated here: this per-body ladder
        takes precedence over the built-in fallback, so any limit the solver relies on must hold for
        it too, or the two silently disagree depending on whether live view had published a ladder.
        Both ends carry a reason — the ceiling keeps the under-exposure cure reachable (a stopped-down
        aperture needs the slow end), and the floor keeps the PWM-lit LED out of banding: at the
        Scanlight's 40 kHz a 1/250 s frame integrates ~160 pulses, but a body's 1/8000 s catches ~5
        and meters noise instead of light."""
        data = self._settings_json()
        floor, ceiling = shutter_seconds(SHUTTER_CANDIDATES[0]), shutter_seconds(SHUTTER_CANDIDATES[-1])
        labels = tuple(str(o.get("label", "")).strip() for o in (data.get("shutter") or {}).get("options", []))
        # usable_ladder drops the unparseables (bulb-like "1/0", "Bulb", "") and returns ascending =
        # fastest-first; the range clamp on top is this UI's own policy, not the solver's.
        return tuple(label for label in usable_ladder(labels) if floor <= shutter_seconds(label) <= ceiling)

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
        self.calib_window.set_inputs_locked(True)  # freeze name / ROI / ISO / aperture while it meters
        self._update_settings_from_ui()
        from negpy.desktop.workers.capture_worker import CalibrationRequest

        s = self._settings
        self.calib_window.set_progress(0.0)
        # Phase 1: scale the fixed reference start point to the body's live ISO/aperture so the probe
        # begins near target (fewer captures). Levels stay fixed; only the shutter is corrected. On a
        # manual lens the aperture reads blank → ISO-only correction, and the probe absorbs the rest.
        candidates = self._available_shutters()
        start_levels, start_shutter = normalize_start_point(
            self._current_setting_label("iso"),
            self._current_setting_label("aperture", require_writable=True),
            candidates=candidates,
        )
        self.controller.start_calibration(
            CalibrationRequest(
                roi=roi,
                output_folder=s.output_folder or "",
                port=s.port,
                settle_s=_LED_SETTLE_S,
                shutter_candidates=candidates,
                start_levels=start_levels,
                start_shutter=start_shutter,
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
        self._maybe_release_camera_session(closing=self.calib_window)

    def _maybe_release_camera_session(self, *, closing=None) -> None:
        """Once neither camera pop-up is open and nothing is mid-capture, release the held
        PTP session rather than leaving it open indefinitely. Some bodies (Fuji in
        particular) get stuck in a tethered-capture state on the camera side until the
        session is cleanly exited — holding it open past the last window that uses it
        makes the next connection attempt hang instead of reconnecting.

        `closing`, when given, names the window whose closeEvent just fired: `closed` is
        emitted from inside closeEvent, before Qt actually hides the widget, so
        `closing.isVisible()` would still (wrongly) read True here — treat it as already
        gone instead. Omit it when called after the fact (e.g. once a cancelled
        calibration actually stops), when both windows' visibility is already accurate.
        """
        if self._suppress_camera_release:
            return  # a hand-off between the two pop-ups, not a real exit
        lv_open = self.lv_window.isVisible() and closing is not self.lv_window
        calib_open = self.calib_window.isVisible() and closing is not self.calib_window
        if lv_open or calib_open:
            return
        if self._scanning or self._calibrating_preset:
            return
        self.controller.close_camera_session()

    # ── live view ─────────────────────────────────────────────────────

    def _on_live_view_toggled(self, on: bool) -> None:
        if on and self.calib_window.isVisible() and not self._calibrating_preset:
            # Hand-off, not an exit: closing calib_window here would otherwise look like
            # the last camera window going away and release the session, only to have
            # _start_live_view_worker() below immediately reopen it (~2-4 s reconnect).
            self._suppress_camera_release = True
            try:
                self.calib_window.close()  # only one live-view window at a time
            finally:
                self._suppress_camera_release = False
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
        # Blank the previous session's frame *before* the window is shown, so reopening live view
        # goes straight to black + the buffering spinner instead of flashing the stale image until
        # the first fresh frame lands (`_on_live_view_started` re-blanks + pins the mtime).
        self._lv_target.clear_frame()
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
            # The body may have drifted since a preset was picked with the stream down — the write
            # only lands once a session is open, which it now is. Re-assert the preset's exposure.
            self._apply_active_preset_camera_settings()
        self._lv_timer.start()
        self._set_status("Live view running.")

    def _on_live_view_window_closed(self) -> None:
        if self.lv_btn.isChecked():
            self.lv_btn.setChecked(False)  # stops live view via _on_live_view_toggled(False)
        self._maybe_release_camera_session(closing=self.lv_window)

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
        if self._lv_frames_seen % 12 == 0:
            # ~1×/s: keep the ISO/shutter/aperture dropdowns fresh in whichever pop-up is streaming.
            # This was gated to the scan window, so opening the calibration pop-up straight after
            # camera idle left its ISO/aperture greyed out (the body reports them non-writable for
            # the first frames and nothing re-read the flag) until the scan window was opened once —
            # Bug 1. _refresh_camera_settings still skips the calib steppers while a run locks them.
            self._refresh_camera_settings()

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
            # Buffer the latest value and write once the user pauses (debounced), so rapid
            # stepping doesn't queue a ~1-2 s verified write per intermediate step.
            self._cam_pending[which] = int(raw)
            self._cam_setting_debounce.start()

    def _flush_camera_settings(self) -> None:
        """Apply the buffered ISO/shutter/aperture changes — one verified write per setting."""
        pending, self._cam_pending = self._cam_pending, {}
        for which, raw in pending.items():
            self.controller.set_camera_setting(which, raw)

    def _refresh_camera_settings(self) -> None:
        """Poll the stream's settings JSON → refresh the ISO/Shutter/aperture steppers in both the
        live-view and the calibration pop-up (the calibration one carries no shutter — the
        calibration solves that; it does carry ISO + aperture, which the base is metered at)."""
        try:
            with open(default_settings_path()) as f:
                data = json.load(f)
        except (OSError, ValueError):
            return
        steppers = {
            "iso": [self.lv_window.iso_stepper],
            "shutter": [self.lv_window.shutter_stepper],
            "aperture": [self.lv_window.aperture_stepper],
        }
        # The calibration pop-up's ISO/aperture are frozen while it meters at them (set_inputs_locked);
        # only mirror the body onto them when no calibration is running, or the refresh would re-enable
        # the disabled steppers mid-run.
        if not self._calibrating_preset:
            steppers["iso"].append(self.calib_window.iso_stepper)
            steppers["aperture"].append(self.calib_window.aperture_stepper)
        # The sidebar steppers are *controllers* in manual mode (the body follows them), not mirrors,
        # so seed them from the body's choices only once — otherwise this periodic refresh would snap
        # the user's picks back to the body whenever a write hasn't landed (e.g. no live session yet).
        include_sidebar = self._manual_mode and self._manual_populate_pending
        if include_sidebar:
            steppers["iso"].append(self.iso_stepper)
            steppers["shutter"].append(self.shutter_stepper)
            steppers["aperture"].append(self.aperture_stepper)
        for key, group in steppers.items():
            info = data.get(key)
            for stepper in group:
                self._apply_setting_to_stepper(stepper, info)
        self._settings_loaded = True
        if include_sidebar:
            self._manual_populate_pending = False  # populated once; the user now owns the sidebar steppers
            self._update_settings_from_ui()  # seed settings from the freshly filled steppers

    def _apply_setting_to_stepper(self, stepper, info) -> None:
        """Reflect one property's value + options onto a ‹ value › stepper (both pop-ups share it)."""
        if not info:  # property unavailable (e.g. aperture on a manual lens)
            stepper.setEnabled(False)
            if not self._settings_loaded:
                stepper.blockSignals(True)
                stepper.clear()
                stepper.addItem("—", None)
                stepper.blockSignals(False)
            return
        stepper.setEnabled(bool(info.get("writable", False)))
        if stepper.hasFocus():
            return  # don't snap the value back while the user is stepping
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
        self._manual_mode = False  # a calibrated preset is read-only, never left in manual-edit mode
        levels, shutters = result.levels, result.shutters
        self.r_slider.setValue(int(levels[0]))
        self.g_slider.setValue(int(levels[1]))
        self.b_slider.setValue(int(levels[2]))
        # RGB preset → white LED off. Without this the W slider keeps a prior white preset's 255,
        # which _update_settings_from_ui below would then bake into the saved preset as w_level=255.
        self._set_slider(self.w_slider, 0)
        shutter = shutters[0]  # one shared shutter (all three are equal)
        self._show_lone(self.shutter_stepper, shutter)
        self._settings = replace(
            self._settings, white_mode=False, shutter_r=shutter, shutter_g=shutter, shutter_b=shutter, shutter_w=shutter
        )
        # The body is sitting at the metered ISO/aperture — capture them so the preset bakes and,
        # later, forces them; the fields show them read-only.
        self._apply_preset_exposure(self._current_setting_label("iso"), self._current_setting_label("aperture", require_writable=True))
        self._update_settings_from_ui()
        self._save_settings()
        # Graceful exposure warning: a channel that couldn't reach ETTR at the hardware limits is
        # saved anyway (best effort) but the user is told which way to move the aperture.
        warn = {
            "over": " ⚠ over-exposed — close the aperture (e.g. f/11) and recalibrate.",
            "under": " ⚠ under-exposed — open the aperture (e.g. f/5.6) and recalibrate.",
        }.get(getattr(result, "status", "target"), "")
        # A finished run always carries its preset name: the flow refuses to start without one, and
        # the marker is cleared only on a terminal outcome (the worker emits exactly one of
        # finished/cancelled/error). The guard remains because saving under an empty name would
        # write a nameless preset into the store — but there is deliberately no else-branch to
        # "review and save later"; that flow no longer exists.
        if self._calibrating_preset:
            name = self._calibrating_preset
            self._calibrating_preset = ""
            self._save_current_as_preset(name)  # persist + reload + select + re-gate (bakes settings.iso/aperture)
            self._lv_target = self.lv_image
            self.calib_window.hide()
            # Pinned: the slider writes above armed the 60 ms light debounce, whose light_set echo
            # lands right after this line — without the pin it replaced this outcome (and its
            # aperture advice) with "Light: R… G… B…" before anyone could read it.
            self._set_status(f"Saved preset “{name}”.{warn}", pinned=True)
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
            # Only the RGB triplet forces the preset's ISO/aperture; white-light and normal
            # scanning leave the body free (the operator sets those in the live view).
            iso=s.iso if rgb and not s.white_mode else "",
            aperture=s.aperture if rgb and not s.white_mode else "",
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
        self.calib_window.set_inputs_locked(False)  # re-enable name / ROI / ISO / aperture for a retry
        self.calib_window.set_status(status)
        self.calib_window.progress.setVisible(False)
        self._lv_target = self.lv_image
        self._stop_calibration_live_view()
        self._apply_gating()
        self._maybe_release_camera_session()  # covers a calibration that was still running when its window closed

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

    def _set_status(self, text: str, *, pinned: bool = False) -> None:
        """Show a status line on the panel and mirror it into the live-view pop-up.

        `pinned` marks a message the user must not miss (the calibration outcome, possibly carrying
        the over/under aperture advice): the ambient light echo (_on_light_set) declines to replace
        it. Any other write — every one of them user-driven — replaces and unpins as usual."""
        self._status_pinned = pinned and bool(text)
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
        self._light_has_white = status.get("light_has_white", True)  # RGB-only Scanlights (v1-v3) have no white LED
        self._set_rgb_mode(status["light_ok"])  # Scanlight present → RGB scanning; absent → normal white-light
        self._refresh_light_channels()  # show/hide the W slider + white preset for the connected model
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
        """Show the live Scanlight LED temperature next to the Light status (amber when warm).
        RGB-only bodies (v1-v3) have no temperature sensor and report a bogus 0 °C, so hide it there
        (no white channel is our proxy for those models)."""
        if isinstance(temp, (int, float)) and self._light_has_white:
            color = "#C8922E" if temp >= 55 else THEME.text_muted  # amber once it's getting warm
            self.light_temp.setStyleSheet(f"color: {color}; font-size: {THEME.font_size_small}px;")
            self.light_temp.setText(f"{temp:.0f} °C")
            self.light_temp.show()
        else:
            self.light_temp.setText("")  # no light / no telemetry yet / RGB-only (no sensor)
            self.light_temp.hide()  # hide the widget entirely so no dark placeholder box lingers

    def _set_cam_status(self, ok: bool, model: str) -> None:
        """Camera dot: '● Camera (USB)' when a body answered, '● Camera' when none did."""
        short = "Camera (USB)" if ok else "Camera"
        if ok:
            detail = f"Camera: {model} (USB)" if model else "Camera connected (USB)"
        else:
            detail = "no camera — plug it in over USB, in PC Remote mode"
        self._set_conn_status(self.cam_status, ok, short, detail)
        self._conn_hint.setVisible(not ok)  # the "connect the camera" nudge is only useful until it is

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
        self._refresh_preset_ui()

    def _refresh_preset_ui(self) -> None:
        """Sync the preset-area widgets to the current mode. The scan live-view exposure steppers hide
        for a calibrated RGB scan (locked to the preset; they stay for white-light and camera-only
        modes). The sidebar exposure fields hide for a white-light preset. And the sliders + exposure
        steppers + Save are editable only while building a manual preset — a selected preset is a
        fixed recipe."""
        locked = self._rgb_mode and not self._settings.white_mode
        self.lv_window.settings_widget.setVisible(not locked)
        if not hasattr(self, "_exposure_widget"):
            return  # first call lands during __init__, before the RGB section is built
        self._exposure_widget.setVisible(not self._settings.white_mode)
        editable = self._manual_mode
        for slider in (self.r_slider, self.g_slider, self.b_slider):
            slider.setEnabled(editable)
        self.w_slider.setEnabled(False)  # the Scanlight can't light white with RGB → a manual RGB preset keeps W off
        tip = (
            "Set it for this preset — steps through the camera's own values."
            if editable
            else "Locked to the preset — pick “Create a manual preset” to set it by hand."
        )
        for stepper in (self.iso_stepper, self.shutter_stepper, self.aperture_stepper):
            stepper.setEnabled(editable)
            stepper.setToolTip(tip)
        self.preset_save_btn.setEnabled(editable)  # the floppy only stores a hand-built preset
        name = self.preset_combo.currentData()  # trash only deletes a stored user preset
        self.preset_del_btn.setEnabled(bool(name) and name != _MANUAL_PRESET and name not in _BUILTIN_WHITE_PRESETS)
        # A manual preset steps through the *camera's* own ISO/shutter/aperture choices, so grey the
        # option out with no camera — NegPy has no idea what values that body offers (it differs per model).
        manual_idx = self.preset_combo.findData(_MANUAL_PRESET)
        model = self.preset_combo.model()
        if manual_idx >= 0 and isinstance(model, QStandardItemModel):
            item = model.item(manual_idx)
            if item is not None:
                item.setEnabled(self._camera_verified)

    def _refresh_light_channels(self) -> None:
        """Adapt the light controls to the connected Scanlight. An RGB-only body (v1-v3, no white
        LED) hides the W slider and greys out the white-light preset, since neither can do anything;
        the live-view framing then lights all three RGB channels instead of a missing white one
        (`_push_light`). A body with a white LED (v4 / Big) keeps them."""
        has_white = self._light_has_white
        w_row = self._slider_rows.get(self.w_slider)
        if w_row is not None:
            w_row.setVisible(has_white)
        white_name = next(iter(_BUILTIN_WHITE_PRESETS))  # the built-in white-light preset
        white_idx = self.preset_combo.findData(white_name)
        model = self.preset_combo.model()
        if white_idx >= 0 and isinstance(model, QStandardItemModel):
            item = model.item(white_idx)
            if item is not None:
                item.setEnabled(has_white)
        # If a white-light preset was somehow selected on an RGB-only body, drop it — it can't run.
        if not has_white and self.preset_combo.currentData() in _BUILTIN_WHITE_PRESETS:
            self.preset_combo.setCurrentIndex(0)
            self._on_preset_selected(0)

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
        # white_mode / white_process_mode are set by preset selection, not widgets. Exposure comes
        # from the steppers only while building a manual preset; otherwise it's the preset's /
        # calibration's value already in settings — not whatever the disabled steppers happen to show.
        if self._manual_mode:
            shutter = self._stepper_label(self.shutter_stepper)
            iso = self._stepper_label(self.iso_stepper)
            aperture = self._stepper_label(self.aperture_stepper)
        else:
            shutter, iso, aperture = self._settings.shutter_r, self._settings.iso, self._settings.aperture
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
            iso=iso,
            aperture=aperture,
            roll_name=self.roll_edit.text().strip() or "Roll001",
            output_folder=self.folder_edit.text().strip(),
        )
        if updated == self._settings:
            return  # nothing changed → skip the disk write + re-gate (the 3 s poll calls this each tick)
        self._settings = updated
        self._save_settings()
        self._apply_gating()
