"""Scanlight RGB-scan capture sidebar.

Live R/G/B light sliders + per-channel shutter, one-button triplet capture,
film-stock presets, and a live-view preview for framing/focus. Captured
exposures land in the hot folder and are handed to NegPy's RGB-Scan mode, which
aligns + merges + inverts them.
"""

import json
import os
import re
from dataclasses import asdict, dataclass, fields, replace

import cv2
import numpy as np
import qtawesome as qta
from PyQt6.QtCore import QTimer, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QImage, QPixmap, QStandardItemModel
from PyQt6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QInputDialog,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from negpy.desktop.view.sidebar.calibration_window import CAPTURE_MODE_TOOLTIP, CAPTURE_MODES, CalibrationWindow
from negpy.desktop.view.sidebar.live_view_window import LiveViewWindow, SettingStepper
from negpy.desktop.view.sidebar.roi_image import zebra_image
from negpy.desktop.view.sidebar.scan_output import ScanOutputPanel
from negpy.desktop.view.styles.templates import (
    field_row,
    header_row,
    hint_label,
    icon_button,
    labeled_action,
    labeled_toggle,
    SCAN_BUTTON_HEIGHT,
    section_subheader,
    set_hint_kind,
    StatusStrip,
    wrap_tooltip,
)
from negpy.desktop.view.styles.theme import THEME
from negpy.desktop.view.widgets.choice_button import ChoiceButton
from negpy.desktop.view.widgets.sliders import CompactSlider
from negpy.infrastructure import simulated
from negpy.infrastructure.capture.gphoto import default_settings_path
from negpy.infrastructure.capture.settings import ScanlightSettings, WhiteCaptureMode
from negpy.features.exposure.analysis import output_histogram
from negpy.services.capture.focus_meter import FocusMeter
from negpy.services.capture.live_stats import clip_masks
from negpy.services.capture.meter import raw_clip_mask
from negpy.services.capture.calibration import (
    CLIP_CEILING,
    REFERENCE_LEVELS,
    normalize_start_point,
    scan_ladder,
)
from negpy.services.assets.sensor import SensorProfiles
from negpy.services.capture.presets import PresetStore, ScanlightPreset, framing_levels

_CHANNEL_COLORS = {"R": THEME.channel_red_text, "G": THEME.channel_green_text, "B": THEME.channel_blue_text, "W": THEME.text_secondary}

# One source for the over/under advice, shown on both surfaces of an aborted calibration:
# the calibration window's status line and the pop-up.
# (label, cause: a full sentence for the pop-up, fix: lowercase so the strip can inline it).
_EXPOSURE_WARNINGS = {
    "over": (
        "over-exposed",
        "Even the fastest shutter with the LEDs at their minimum still clips the film base.",
        "close the aperture (e.g. f/11) or lower the ISO, then recalibrate",
    ),
    "under": (
        "under-exposed",
        "Even the slowest shutter with the LEDs at their maximum stays below the exposure target.",
        "open the aperture (e.g. f/5.6) or raise the ISO, then recalibrate",
    ),
}

# Built-in white-light preset, which needs no calibration: name -> process mode.
# Selecting it switches the panel to a single white-light exposure. B&W and slide film share
# the *same* plain white light, so they are one preset, and autodetect picks the process.
_BUILTIN_WHITE_PRESETS = {
    "White Light (B&W or Slide Film)": WhiteCaptureMode.AUTO,
    "White Light, B&W Negative": WhiteCaptureMode.BW,
    "White Light, Slide": WhiteCaptureMode.E6,
}

# A dropdown sentinel, not a real preset name: user names are stripped, so a NUL cannot
# collide. Picking it unlocks the sliders and steppers to build a preset by hand.
_MANUAL_PRESET = "\x00create-manual"


# LED settle before each exposure. Narrowband PWM LEDs reach full brightness in under
# 10 ms and set_color is a fire-and-forget serial write, so 50 ms keeps an
# order-of-magnitude margin. A fixed tuning constant, not a persisted setting.
_LED_SETTLE_S = 0.05
#: Time a body takes to switch between its full and magnified view.
_MAGNIFIER_SETTLE_MS = 2500


def _gray_array(pixmap: QPixmap) -> np.ndarray:
    """HxW uint8 gray copy of a live frame."""
    image = pixmap.toImage().convertToFormat(QImage.Format.Format_Grayscale8)
    bits = image.constBits()
    bits.setsize(image.sizeInBytes())
    rows = np.frombuffer(bits, dtype=np.uint8).reshape(image.height(), image.bytesPerLine())
    return rows[:, : image.width()].copy()


#: Normalized correlation of two previews' structure above which they are the same picture.
_SAME_PICTURE = 0.9


@dataclass
class _RawClip:
    mask: np.ndarray  # raw_clip_mask, on the probe's peak map
    exposure: tuple  # _exposure_key at the metered shutter
    reference: np.ndarray | None = None  # _structure of the first preview at that exposure


def _structure(rgb: np.ndarray) -> np.ndarray:
    """A preview's picture at 64 px, zero-mean and unit-variance, so exposure drops out of a
    comparison and framing does not."""
    gray = cv2.resize(rgb.mean(axis=2).astype(np.float32), (64, 43), interpolation=cv2.INTER_AREA)
    gray = gray - gray.mean()
    return gray / max(float(gray.std()), 1e-6)


def _rgb_array(pixmap: QPixmap) -> np.ndarray:
    """HxWx3 uint8 RGB copy of a live frame."""
    image = pixmap.toImage().convertToFormat(QImage.Format.Format_RGB888)
    bits = image.constBits()
    bits.setsize(image.sizeInBytes())
    rows = np.frombuffer(bits, dtype=np.uint8).reshape(image.height(), image.bytesPerLine())
    return rows[:, : 3 * image.width()].reshape(image.height(), image.width(), 3).copy()


class ScanlightSidebar(QWidget):
    """Trichromatic RGB-scan capture panel."""

    cards_changed = pyqtSignal()

    def __init__(self, controller, output: ScanOutputPanel | None = None) -> None:
        super().__init__()
        self.controller = controller
        self.output = output if output is not None else ScanOutputPanel(controller.session.repo)
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
        self._exposure_popup = None  # the over/under pop-up (kept referenced; replaced per calibration)
        self._magnifier_on = False  # camera focus magnifier state (driven by clicks on the live image)
        self._magnifier_seen = False  # the state the stream last published, or a click's own
        self._metering = False  # a Meter probe is on the worker; Scan, Retake and Meter wait for it
        self._meter_raw: dict[str, int] = {}  # the body's shutter labels → raw indexes, read once per probe
        # The last probe's predicted RAW clipping, the exposure it holds for, and the preview it
        # was metered on (_current_raw_clip).
        self._raw_clip: _RawClip | None = None
        self._magnifier_available = True
        self._focus_meter = FocusMeter()
        # Full and magnified views do not share a sharpness scale: reset the peak once the body has switched.
        self._focus_settle_timer = QTimer()
        self._focus_settle_timer.setSingleShot(True)
        self._focus_settle_timer.setInterval(_MAGNIFIER_SETTLE_MS)
        self._focus_settle_timer.timeout.connect(self._reset_focus_meter)
        self._settings_loaded = False  # have the live camera-setting dropdowns been populated yet?
        self._light_has_white = True  # does the connected Scanlight have a white LED? (False = v1-v3, RGB-only)
        # Advertised camera abilities (issue #621). Optimistic until a body has been opened:
        # an unlisted body matches the generic PTP entry that claims everything, so only a
        # *missing* bit is evidence. Never grey a control out on an uninspected camera.
        self._camera_has_preview = True
        self._camera_has_config = True
        self._suppress_camera_release = False  # true only mid hand-off between lv_window/calib_window
        # Preset-exposure writes still in flight on the worker. Scan stays gated until each is
        # confirmed, so a capture cannot race the body's async programming.
        self._pending_exposure_writes = 0

        self.lv_window = LiveViewWindow(self, repo=self.controller.session.repo)
        self.lv_window.closed.connect(self._on_live_view_window_closed)
        self.lv_image = self.lv_window.image

        # Dedicated pop-up for creating a preset by calibration, independent of the scan
        # cockpit so the first preset can be made. Live frames route to whichever ROI image
        # is active via self._lv_target.
        self.calib_window = CalibrationWindow(self, repo=self.controller.session.repo)
        self.calib_window.closed.connect(self._on_calib_window_closed)
        self.calib_window.calibrateRequested.connect(self._on_calibrate_new_preset)
        self._lv_target = self.lv_image  # RoiImageLabel currently fed by the live-view poll

        self._light_debounce = QTimer()
        self._light_debounce.setSingleShot(True)
        self._light_debounce.setInterval(60)
        self._light_debounce.timeout.connect(self._push_light)

        # Coalesce rapid ISO/shutter/aperture stepping into one verified camera write per
        # setting. Each write takes a second or two, so writing every intermediate step made
        # a few clicks take half a minute.
        self._cam_setting_debounce = QTimer()
        self._cam_setting_debounce.setSingleShot(True)
        self._cam_setting_debounce.setInterval(250)
        self._cam_setting_debounce.timeout.connect(self._flush_camera_settings)
        self._cam_pending: dict[str, int] = {}

        # Live view: the camera's preview thread rewrites a JPEG and this timer polls it. It
        # runs a little faster than the frame interval and skips re-decoding an unchanged
        # frame, so new frames show promptly without wasting CPU.
        self._lv_jpeg_path = ""
        self._lv_polls = 0
        self._lv_frames_seen = 0
        self._lv_last_mtime = 0.0
        self._lv_timer = QTimer()
        self._lv_timer.setInterval(80)
        self._lv_timer.timeout.connect(self._refresh_live_view)

        # Auto-connect: poll for a USB camera and the light every few seconds while the panel
        # is visible. Paused during live view and scanning, since the body grants one PTP claim.
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
                # Filter to known fields, so a dropped or renamed persisted setting cannot
                # break construction and reset everything to defaults.
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

        return simulated.enabled() or importlib.util.find_spec("gphoto2") is not None

    def _refresh_setup_hint(self) -> None:
        """Show the setup note only while python-gphoto2 is missing."""
        self._setup_hint.setVisible(not self._gphoto_available())

    # ── UI construction ───────────────────────────────────────────────

    def _light_slider(self, name: str, letter: str, value: int, default: int) -> CompactSlider:
        slider = CompactSlider(name, 0, 255, default, step=1, precision=1, color=_CHANNEL_COLORS[letter])
        slider.setValue(value)
        slider.valueChanged.connect(lambda _v: self._light_debounce.start())
        return slider

    def _init_ui(self) -> None:
        """Three bodies that the Scan tab puts in cards; a standalone panel stacks them."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(THEME.space_lg)

        self.camera_body = QWidget()
        cam = QVBoxLayout(self.camera_body)
        cam.setContentsMargins(0, 0, 0, 0)
        cam.setSpacing(THEME.space_md)
        # python-gphoto2 is optional, so show a setup note while it is missing. It hides once
        # installed and never nags an equipped user.
        self._setup_hint = hint_label(
            "Camera scanning needs python-gphoto2, an optional dependency: "
            "`pip install gphoto2` (macOS and Linux — libgphoto2 has no Windows build). "
            "See docs/CAMERA_SCANNING.md.",
            "warning",
        )
        cam.addWidget(self._setup_hint)
        self._setup_hint.setVisible(not self._gphoto_available())
        self._conn_hint = hint_label("Connect the camera by USB, in PC Remote mode — it's detected automatically.")
        cam.addWidget(self._conn_hint)
        status_row = QHBoxLayout()
        status_row.setSpacing(THEME.space_lg)
        self.cam_status = hint_label()
        self.light_status = hint_label()
        self.light_temp = hint_label()  # live LED temperature next to the light status (heat monitoring)
        self.light_temp.hide()  # stay hidden until a reading arrives — an empty label still paints a panel-dark box
        for label in (self.cam_status, self.light_status, self.light_temp):
            label.setWordWrap(False)
            status_row.addWidget(label)
        status_row.addStretch()
        cam.addLayout(status_row)
        self._set_conn_status(self.cam_status, None, "Camera")
        self._set_conn_status(self.light_status, None, "Light")
        # RGB scanning needs the Scanlight. Without it, in normal white-light mode, this hint
        # sits with the connection status. The light poll hides it in RGB mode.
        self._rgb_hint = hint_label("You can also connect the Scanlight to scan in RGB.")
        self._rgb_hint.setVisible(False)
        cam.addWidget(self._rgb_hint)
        layout.addWidget(self.camera_body)

        # Scanlight only: _set_rgb_mode hides it for white-light scanning.
        self.light_body = QWidget()
        rgb = QVBoxLayout(self.light_body)
        rgb.setContentsMargins(0, 0, 0, 0)
        rgb.setSpacing(THEME.space_md)

        self.preset_combo = QComboBox()
        self.preset_combo.setToolTip(
            wrap_tooltip(
                "Pick a saved film-stock preset (RGB levels + ISO + shutter + aperture, shown read-only), a "
                "built-in white-light mode, or “Create a manual preset…” to build one by hand"
            )
        )
        self.preset_new_btn = icon_button("fa5s.plus", "Create a preset by calibrating on the film base (auto-meters the exposure)")
        self.preset_save_btn = icon_button(
            "fa5s.save", "Name and save the manual preset you're building (only while in manual-preset mode)"
        )
        self.preset_del_btn = icon_button("fa5s.trash", "Delete the selected preset")
        rgb.addLayout(header_row(section_subheader("PRESET"), self.preset_new_btn, self.preset_save_btn, self.preset_del_btn))
        rgb.addWidget(self.preset_combo)
        self.preset_hint = hint_label("")
        self.preset_hint.setVisible(False)
        rgb.addWidget(self.preset_hint)

        rgb.addWidget(section_subheader("LIGHT"))
        self.r_slider = self._light_slider("Red", "R", self._settings.r_level, 255)
        self.g_slider = self._light_slider("Green", "G", self._settings.g_level, 255)
        self.b_slider = self._light_slider("Blue", "B", self._settings.b_level, 255)
        self.w_slider = self._light_slider("White", "W", self._settings.w_level, 0)
        self.r_slider.setToolTip("Red LED level")
        self.g_slider.setToolTip("Green LED level")
        self.b_slider.setToolTip("Blue LED level")
        self.w_slider.setToolTip("White LED — used only by the white-light preset; the Scanlight can't light it together with RGB")
        for slider in (self.r_slider, self.g_slider, self.b_slider, self.w_slider):
            rgb.addWidget(slider)
        self.off_btn = labeled_action("fa5s.power-off", " Light Off", "Turn all Scanlight channels off")
        rgb.addWidget(self.off_btn)

        # ISO / shutter / aperture: the preset's exposure, wrapped so it hides as a unit for a
        # white-light preset, whose exposure is set in the live view. Each is a stepper, read-only
        # while a preset is active because the scan forces these onto the body, and writable only
        # in "manual preset" mode, where it steps through this body's own choices. Calibration
        # normally solves the shutter.
        self._exposure_widget = QWidget()
        self._exposure_widget.setObjectName("collapsible_content_body")
        _exp = QVBoxLayout(self._exposure_widget)
        _exp.setContentsMargins(0, 0, 0, 0)
        _exp.setSpacing(THEME.space_md)
        _exp.addWidget(section_subheader("EXPOSURE"))
        self.capture_btn = ChoiceButton(CAPTURE_MODES, CAPTURE_MODE_TOOLTIP)
        self.capture_btn.setEnabled(False)  # the preset's mode, editable only while building a manual preset
        _exp.addLayout(field_row("Capture mode", self.capture_btn))
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
            _exp.addLayout(field_row(_tag_text, _stepper))
        rgb.addWidget(self._exposure_widget)
        self.inter_exposure_delay_slider = CompactSlider("Channel Delay", 0, 5000, 0, step=100, precision=1, unit=" ms")
        self.inter_exposure_delay_slider.setValue(self._settings.inter_exposure_delay_ms)
        self.inter_exposure_delay_slider.setToolTip(
            "Wait this long between successive trichrome channel exposures so the camera can flush the previous shot."
        )
        rgb.addWidget(self.inter_exposure_delay_slider)
        layout.addWidget(self.light_body)

        self.footer = QWidget()
        foot = QVBoxLayout(self.footer)
        foot.setContentsMargins(0, 0, 0, 0)
        foot.setSpacing(THEME.space_md)
        self.gate_hint = hint_label("", "warning")
        foot.addWidget(self.gate_hint)
        self.status_strip = StatusStrip()
        foot.addWidget(self.status_strip)
        self.lv_btn = labeled_toggle("fa5s.video", " Live View", False, "Open the live view to frame and focus")
        self.retake_btn = labeled_action("fa5s.redo", " Retake", "Capture the last frame again, without advancing the frame number")
        actions = QHBoxLayout()
        actions.addWidget(self.lv_btn, 1)
        actions.addWidget(self.retake_btn, 1)
        foot.addLayout(actions)
        self.scan_btn = QPushButton(" Scan")
        self.scan_btn.setObjectName("scan_btn")
        self.scan_btn.setFixedHeight(SCAN_BUTTON_HEIGHT)
        self.scan_btn.setProperty("scanning", "false")
        self.scan_btn.setIcon(qta.icon("fa5s.camera", color=THEME.text_on_accent))
        foot.addWidget(self.scan_btn)
        layout.addWidget(self.footer)

        self._apply_gating()
        layout.addStretch()

    def _connect_signals(self) -> None:
        self.off_btn.clicked.connect(self._on_light_off)
        self.lv_btn.toggled.connect(self._on_live_view_toggled)
        self.scan_btn.clicked.connect(self._on_scan)
        self.retake_btn.clicked.connect(self._on_retake)
        self.preset_combo.activated.connect(self._on_preset_selected)
        self.capture_btn.currentChanged.connect(lambda _i: self._push_light())
        self.preset_new_btn.clicked.connect(self._on_preset_new)
        self.preset_save_btn.clicked.connect(self._on_preset_save)
        self.preset_del_btn.clicked.connect(self._on_preset_delete)
        # Typing an invalid path greys Scan immediately (with the reason), not on the next click.
        self.output.changed.connect(self._apply_gating)
        self.inter_exposure_delay_slider.valueChanged.connect(self._update_settings_from_ui)

        self.controller.capture_light_set.connect(self._on_light_set)
        self.controller.capture_progress.connect(self._on_progress)
        self.controller.capture_channel.connect(self._on_channel)
        self.controller.capture_camera_setting_applied.connect(self._on_camera_setting_applied)
        self.controller.capture_live_view_failed.connect(self._on_live_view_failed)
        self.controller.capture_live_view_unsupported.connect(self._on_live_view_unsupported)
        self.controller.capture_focus_magnifier_unavailable.connect(self._on_magnifier_unavailable)
        self.controller.capture_finished.connect(self._on_finished)
        self.controller.capture_cancelled.connect(self._on_cancelled)
        self.controller.capture_error.connect(self._on_error)
        self.controller.capture_status.connect(self._on_status)
        self.controller.capture_live_view_started.connect(self._on_live_view_started)
        self.controller.capture_calibration_progress.connect(self._on_calibration_progress)
        self.controller.capture_calibration_finished.connect(self._on_calibration_finished)
        self.controller.capture_calibration_exposure.connect(self._on_calibration_exposure)
        self.controller.connection_polled.connect(self._on_poll_status)
        self.controller.light_temp_polled.connect(self._on_light_temp)
        self.controller.batch_started.connect(self._keep_scan_windows_on_top)
        # The pop-up toolbar mirrors the panel actions, so a roll scans without tab-switching.
        self.lv_window.scanRequested.connect(self._on_scan)
        self.lv_window.retakeRequested.connect(self._on_retake)
        self.lv_window.meterRequested.connect(self._on_meter)
        self.controller.capture_exposure_probe_finished.connect(self._on_meter_finished)
        self.controller.capture_exposure_probe_failed.connect(self._on_meter_failed)
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

    # ── light ─────────────────────────────────────────────────────────

    def _push_light(self) -> None:
        if not self._rgb_mode:
            return  # normal white-light scanning has no Scanlight to control
        if self._settings.white_mode:
            # A white-light preset: the W slider is the scan light itself.
            self.controller.set_scanlight_color(0, 0, 0, int(self.w_slider.value()), self._settings.port)
        else:
            # Any RGB state (a selected preset, manual building, or framing in live view) lights
            # the R/G/B mix from the sliders, so it works on every Scanlight and never assumes a
            # white LED an RGB-only body lacks. White stays off: the Scanlight cannot mix both.
            r, g, b = (int(slider.value()) for slider in (self.r_slider, self.g_slider, self.b_slider))
            if not self._manual_mode and self._preset_selected() and not self._settings.single_capture:
                # A stored triplet preset frames by its own mix, but dimmed (issue #573): all three
                # channels burn at once against a single-channel scan exposure, so full levels
                # blow the live view out. The scan never reads this, since capture levels come
                # from the preset. Manual building keeps full levels for the operator, and a
                # single-capture preset frames under its own scan light.
                r, g, b = framing_levels(r, g, b)
            self.controller.set_scanlight_color(r, g, b, 0, self._settings.port)
        self._update_settings_from_ui()

    def _on_light_off(self) -> None:
        self.controller.set_scanlight_color(0, 0, 0, 0, self._settings.port)

    @pyqtSlot(int, int, int, int)
    def _on_light_set(self, r: int, g: int, b: int, w: int) -> None:
        # Ambient echo, not a user action. _on_calibration_finished sets the R/G/B sliders, each
        # starts the light debounce, and this slot fires just after the calibration outcome was
        # written. Refusing to replace a pinned outcome keeps that advice on screen.
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
        slider.setValue(value)

    def _set_capture_mode(self, single: bool) -> None:
        """Point the Capture mode button and the settings at a preset's mode, without the push
        a user's own pick triggers."""
        self.capture_btn.blockSignals(True)
        self.capture_btn.setCurrentIndex(int(single))
        self.capture_btn.blockSignals(False)
        self._settings = replace(self._settings, single_capture=single)

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
        self._set_capture_mode(preset.single_capture)
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
                # Defensive: the dropdown item is greyed without a camera, but refuse here too.
                # A manual preset's steppers need the body's own ISO/shutter/aperture choices.
                self._set_status("Connect the camera first — a manual preset uses the camera's ISO / shutter / aperture choices")
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
            # Built-in white-light mode, one white exposure for B&W or slide: white on, RGB off.
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
        if name in _BUILTIN_WHITE_PRESETS:
            text = "Single white-light exposure — for B&W or slide film."
        elif self._preset_selected() and getattr(self._presets.get(name), "single_capture", False):
            text = "Single exposure with red, green and blue lit together."
        else:
            text = ""
        self.preset_hint.setText(text)
        self.preset_hint.setVisible(bool(self.preset_hint.text()))

    def _on_preset_save(self) -> None:
        if not self._manual_mode:
            return  # Save only stores a hand-built preset — the button is greyed out otherwise
        name, ok = QInputDialog.getText(self, "Save Manual Preset", "Film stock name:")
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
        self._set_status(f"Saved preset “{name}”")

    def _save_current_as_preset(self, name: str, sensor_profile: str = "") -> None:
        self._update_settings_from_ui()
        s = self._settings
        # Bake the active recipe from settings, set either by calibration or by the manual-mode
        # steppers. A later scan reproduces it. Aperture is blank on a manual lens.
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
                single_capture=s.single_capture,
                sensor_profile=sensor_profile,
            ),
        )
        self._reload_presets(select=name)

    def _on_preset_delete(self) -> None:
        name = self.preset_combo.currentData()
        if not name or name in _BUILTIN_WHITE_PRESETS:
            return
        self._presets.delete(name)
        self._reload_presets()
        self._set_status(f"Deleted preset “{name}”")

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
        # The framing light is the calibration's own start point (REFERENCE_LEVELS), so the
        # crosshair is placed under the light the probe begins from, not under leftover RGB or
        # an arbitrary grey. Pushed DIRECTLY, leaving the shared sliders on the selected preset,
        # so cancelling restores the preset's own light. Calibration overwrites R/G/B on success.
        self.controller.set_scanlight_color(*REFERENCE_LEVELS, 0, self._settings.port)
        self._set_status("Calibrating a new preset — see the pop-up")

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
        """Push the active RGB preset's baked exposure (shutter/ISO/aperture) to the body (no-op for
        white/built-in presets or no selection). Used when the scan live view comes up after a preset
        was picked."""
        name = self.preset_combo.currentData()
        if not name or name in _BUILTIN_WHITE_PRESETS:
            return
        preset = self._presets.get(name)
        if preset is not None:
            self._apply_preset_camera_settings(preset)

    def _apply_preset_camera_settings(self, preset) -> None:
        """Set the body's shutter + ISO + aperture to the values baked into the RGB preset, so the
        scan matches the calibration and the framing brightness is deterministic: the dimmed framing
        light (see framing_levels) is sized against the preset's shutter, so a body left on some
        other speed would frame too dark or too bright. Shutter first — it dominates that brightness.
        Labels are resolved against the live options (a no-op if the value is absent, the body lacks
        the option, or no camera session is open to receive the write)."""
        data = self._settings_json()
        issued = 0
        for key, label in (("shutter", preset.shutter_r), ("iso", preset.iso), ("aperture", preset.aperture)):
            if not label:
                continue
            info = data.get(key)
            if not isinstance(info, dict):
                continue
            for o in info.get("options", []):
                if str(o.get("label", "")) == label:
                    self.controller.set_camera_setting(key, int(o["raw"]))
                    issued += 1
                    break
        if issued:
            self._pending_exposure_writes += issued
            self._apply_gating()  # Scan greys out until the worker confirms every write

    @pyqtSlot(str)
    def _on_camera_setting_applied(self, _which: str) -> None:
        if self._pending_exposure_writes == 0:
            return  # a stepper write from manual/white mode, not one of the gated preset writes
        self._pending_exposure_writes -= 1
        if self._pending_exposure_writes == 0:
            self._apply_gating()

    def _available_shutters(self, info: dict | None = None) -> tuple[str, ...]:
        """The camera's writable shutter labels (from the live-view settings JSON, or `info`,
        that JSON's shutter entry) as the ladder calibration solves on, so it solves on this
        body's own speeds."""
        if info is None:
            info = self._settings_json().get("shutter") or {}
        if not info.get("writable", True):  # absent = unknown, so only an explicit read-only blocks
            # A body that reports the shutter read-only ignores every write, and the only symptom
            # is a read-back that never settles (issue #768). Publishing no ladder makes the caller
            # refuse instead of solving against speeds it cannot set.
            return ()
        return scan_ladder(o.get("label", "") for o in info.get("options", []))

    def _on_calibrate_new_preset(self, name: str) -> None:
        if self._scanning:
            self.calib_window.set_status("A scan is running — wait for it to finish")
            return
        name = name.strip()
        if not name:
            self.calib_window.set_status("Enter a film-stock name first")
            return
        roi = self.calib_window.image.roi()
        if roi is None:
            self.calib_window.set_status("Click the clear film base (crosshair) first")
            return
        candidates = self._available_shutters()
        if not candidates:
            # Without the body's own ladder the only speeds we can name are the built-in ones,
            # spelled in one vendor's vocabulary. Another body ignores them instead of refusing,
            # which reads as "camera rejected it" (issue #768). Say what is actually wrong.
            self.calib_window.set_status(
                "This camera is not reporting settable shutter speeds. Set it to Manual (M), "
                "make sure live view is running, then try again."
            )
            return
        # Live view stays up: calibration captures within it, like a scan, so there is no
        # reconnect. _stop_calibration_live_view tears it down when the run ends.
        self._calibrating_preset = name
        self._apply_gating()  # a running calibration locks Scan / Retake
        self.calib_window.set_inputs_locked(True)  # freeze name / ROI / ISO / aperture while it meters
        self._update_settings_from_ui()
        from negpy.desktop.workers.capture_worker import CalibrationRequest

        s = self._settings
        self.calib_window.set_progress(0.0)
        # Phase 1: scale the fixed reference start point to the body's live ISO/aperture, so the
        # probe begins near target and needs fewer captures. Levels stay fixed and only the
        # shutter is corrected. A manual lens reads a blank aperture, so the correction is
        # ISO-only and the probe absorbs the rest.
        start_levels, start_shutter = normalize_start_point(
            self._current_setting_label("iso"),
            self._current_setting_label("aperture", require_writable=True),
            candidates=candidates,
        )
        self.controller.start_calibration(
            CalibrationRequest(
                roi=roi,
                output_folder=self.output.folder(),
                port=s.port,
                settle_s=_LED_SETTLE_S,
                shutter_candidates=candidates,
                start_levels=start_levels,
                start_shutter=start_shutter,
                single_capture=self.calib_window.capture_btn.currentIndex() == 1,
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
            # Keep the job marker until the worker acknowledges a terminal outcome. The shared
            # capture thread is still occupied, so re-enabling Scan here would queue another
            # frame behind work that has not stopped.
            if not calibration_running:
                self._calibrating_preset = ""
            self._apply_gating()
            self._push_light()
        self._maybe_release_camera_session(closing=self.calib_window)

    def _maybe_release_camera_session(self, *, closing=None) -> None:
        """Release the held PTP session once neither camera pop-up is open and nothing is mid-capture.

        Some bodies (Fuji) stay in tethered capture until the session exits; one held open hangs the next connect.
        `closing` names a window whose `closed` fired inside done(), where its isVisible() still reads True.
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
            # A hand-off, not an exit. Closing calib_window here looks like the last camera
            # window going away, releases the session, and _start_live_view_worker() below then
            # reopens it at the cost of a reconnect.
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
            self._metering = False  # a probe still in flight reports to a stopped stream
            self.lv_window.hide()
            self._push_light()  # back to the capture light (RGB unless white mode)
            self._set_status("")  # clear the "Live view running" line once the stream stops

    def _start_live_view_worker(self) -> None:
        """Spawn the live-view stream subprocess (shared by toggle-on and resume)."""
        # Blank the previous session's frame *before* the window is shown, so reopening live view
        # goes straight to black and the buffering spinner instead of flashing the stale image.
        # `_on_live_view_started` re-blanks and pins the mtime.
        self._lv_target.clear_frame()
        self.lv_window.set_histogram(None)  # the row follows the frame: empty until one lands
        self._raw_clip = None  # a new session may hold a different exposure
        self._lv_target.set_loading(True)  # buffering spinner until the first frame lands
        from negpy.desktop.workers.capture_worker import LiveViewRequest

        self.controller.start_live_view(LiveViewRequest())

    @pyqtSlot(str)
    def _on_live_view_started(self, jpeg_path: str) -> None:
        self._lv_jpeg_path = jpeg_path
        self._lv_polls = 0
        self._lv_frames_seen = 0
        self.lv_window.set_preview_available(True)  # restore the pane after a no-preview body
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
            # The body may have drifted since a preset was picked with the stream down, because
            # the write lands only once a session is open. Re-assert the preset's exposure.
            self._apply_active_preset_camera_settings()
        self._magnifier_available = True
        self._reset_focus_meter()
        self._lv_timer.start()
        self._set_status("Live view running")

    def _on_live_view_window_closed(self) -> None:
        if self.lv_btn.isChecked():
            self.lv_btn.setChecked(False)  # stops live view via _on_live_view_toggled(False)
        self._maybe_release_camera_session(closing=self.lv_window)

    def _keep_scan_windows_on_top(self, _title: str, _abortable: bool) -> None:
        """The batch progress popup shows itself with raise_() on every batch — including the
        per-frame "Hashing files" imports after a capture — which puts
        it over the live-view pop-up and reads as "wait here" mid-roll. Re-raise the operator's
        open scan windows one event-loop turn later (the popup's own raise_() runs first)."""
        for window in (self.lv_window, self.calib_window):
            if window.isVisible():
                QTimer.singleShot(0, window.raise_)

    def _refresh_live_view(self) -> None:
        if not self._lv_jpeg_path:
            return
        # Skip the decode and repaint when the preview thread has written no new frame since
        # the last poll. The poll runs a little faster than frames arrive.
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
                    "On macOS a background app such as a cloud sync client can be holding it; unplugging frees it."
                )
                self._lv_target.set_loading(False)  # stop the spinner; the hint explains why
            return
        self._lv_last_mtime = mtime
        self._lv_frames_seen += 1
        self._lv_target.set_frame(pixmap)  # scan pop-up or the calibration window
        if self._lv_target is self.lv_image:
            self.lv_window.set_focus(self._focus_meter.update(_gray_array(pixmap)))
            self._update_exposure_readouts(pixmap)
        if self._lv_frames_seen % 12 == 0:
            # About once a second: keep the ISO/shutter/aperture dropdowns fresh in whichever
            # pop-up is streaming. Gated to the scan window, this left the calibration pop-up's
            # ISO/aperture greyed out after camera idle, because the body reports them
            # non-writable for the first frames and nothing re-read the flag.
            # _refresh_camera_settings still skips the calib steppers while a run locks them.
            self._refresh_camera_settings()

    def _on_meter(self) -> None:
        """One RAW probe at the body's current exposure, under the light as lit. The reading
        lands in `_on_meter_finished`. Only while the stream runs."""
        if not self.lv_btn.isChecked() or self._scanning or self._calibrating_preset or self._metering:
            return
        if self._rgb_mode and not self._settings.white_mode:
            return  # the button is hidden under a calibrated RGB preset; its key is not
        # One read of the stream's JSON: the ladder, the probe's shutter and the labels the
        # reading is written back through all come from the same publish.
        info = self._settings_json().get("shutter") or {}
        candidates = self._available_shutters(info)
        options = {str(o.get("label", "")): int(o["raw"]) for o in info.get("options", []) if "raw" in o}
        shutter = next((label for label, raw in options.items() if raw == info.get("cur")), "")
        if not candidates:
            self._set_status("Meter: this body publishes no writable shutter")
            return
        if shutter not in candidates:  # Bulb, a bad label, or past the ladder's span
            self._set_status(f"Meter: set the shutter between {candidates[0]} and {candidates[-1]} first")
            return
        from negpy.desktop.workers.capture_worker import ExposureProbeRequest

        # The preset's medium, as the import reads it; AUTO leaves the probe to the classifier.
        medium = {WhiteCaptureMode.BW: "negative", WhiteCaptureMode.E6: "positive"}.get(WhiteCaptureMode(self._settings.white_process_mode))
        self._meter_raw = options
        self._raw_clip = None
        self._metering = True
        self._apply_gating()
        self._set_status("Metering…")
        self.controller.start_exposure_probe(ExposureProbeRequest(shutter=shutter, shutter_candidates=candidates, medium=medium))

    @pyqtSlot(object)
    def _on_meter_finished(self, reading) -> None:
        """Write the metered shutter to the body as a counted exposure write, so Scan waits
        for the body's confirmation. A white-light or camera-only scan shoots at the body's
        shutter."""
        self._metering = False
        where = f"{reading.medium}, {reading.region}"
        level = f"{reading.measured / CLIP_CEILING:.0%}"
        label = reading.recommended
        raw = self._meter_raw.get(label) if label else None
        if label is None:
            self._apply_gating()
            self._set_status(f"Meter: {where} at {level} needs a faster shutter than this body offers; close the aperture or lower the ISO")
            return
        if raw is None:
            self._apply_gating()
            self._set_status(f"Meter: {where} at {level}; {label} is not a step this body offers")
            return
        self.controller.set_camera_setting("shutter", raw)
        self._pending_exposure_writes += 1
        mask = raw_clip_mask(reading)
        if mask is not None:
            self._raw_clip = _RawClip(mask, self._exposure_key(shutter=raw))
        index = self.lv_window.shutter_stepper.findData(raw)
        if index >= 0:
            self.lv_window.shutter_stepper.setCurrentIndex(index)
        self._apply_gating()
        if reading.clipped and reading.region == "base":
            self._set_status(f"Meter: {where} clipped at {reading.shutter}; set {label} ({reading.stops:+.1f} st), meter again")
        elif reading.clipped:
            self._set_status(
                f"Meter: {where} clipped on {reading.clipped_fraction:.1%} of the picture at {reading.shutter}; "
                f"set {label} ({reading.stops:+.1f} st), meter again"
            )
        else:
            self._set_status(f"Meter: {where} at {level} → {label} ({reading.stops:+.1f} st)")

    @pyqtSlot(str)
    def _on_meter_failed(self, reason: str) -> None:
        """The probe decoded but could not be metered; the stream and the session stand."""
        self._metering = False
        self._apply_gating()
        self._set_status(f"Meter: {reason}")

    def _update_exposure_readouts(self, pixmap: QPixmap) -> None:
        """The histogram row and the zebras, from the frame on screen, only while either shows."""
        want_histogram = self.lv_window.histogram_btn.isChecked()
        want_zebra = self.lv_window.zebra_btn.isChecked()
        if not (want_histogram or want_zebra):
            return
        rgb = _rgb_array(pixmap)
        if want_histogram:
            self.lv_window.set_histogram(output_histogram(rgb))
        if want_zebra:
            highlights, blacks = clip_masks(rgb)
            raw = self._current_raw_clip(rgb)
            if raw is not None:
                h, w = highlights.shape
                highlights = cv2.resize(raw.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
            self.lv_image.set_zebra(zebra_image(highlights, blacks))

    def _exposure_key(self, shutter: object = None) -> tuple:
        """What sets the RAW level apart from the film: the body's ISO, shutter and aperture and
        the light. `shutter` stands in for the stepper's while a metered write is on its way."""
        lv = self.lv_window
        return (
            lv.iso_stepper.currentData(),
            lv.shutter_stepper.currentData() if shutter is None else shutter,
            lv.aperture_stepper.currentData(),
            self._settings,
        )

    def _current_raw_clip(self, rgb: np.ndarray) -> np.ndarray | None:
        """The Meter probe's RAW clipping, while it still describes the frame on screen: the same
        exposure and the same picture. The picture is the first preview at the metered exposure,
        compared by normalized correlation, so a film advance, a reframe or the focus magnifier
        hands the zebras back to the preview; a brighter preview of the same frame does not."""
        clip = self._raw_clip
        if clip is None:
            return None
        if self._pending_exposure_writes:
            return clip.mask  # the metered shutter is on its way; the stepper may still lag
        if self._exposure_key() != clip.exposure:
            return None
        thumb = _structure(rgb)
        if clip.reference is None:
            clip.reference = thumb
            return clip.mask
        return clip.mask if float(np.mean(thumb * clip.reference)) >= _SAME_PICTURE else None

    def _after_capture_live_view(self) -> None:
        """Re-light the preview after a scan. An in-session capture leaves the Scanlight
        off (capture_triplet turns it off in its finally) while the live-view stream keeps
        running, so just push the framing light back. No-op when live view is off."""
        if self.lv_btn.isChecked():
            self._push_light()

    @pyqtSlot(str)
    def _on_live_view_unsupported(self, reason: str) -> None:
        """No stream was started because the body advertises none (issue #621). Not an error:
        the scan window stays open and usable with the preview pane replaced by the reason,
        since it holds the only Scan button. Calibration is gated separately in _apply_gating."""
        self._camera_has_preview = False
        self._lv_timer.stop()
        self._lv_target.set_loading(False)  # nothing is buffering; the spinner would lie
        self.lv_window.set_preview_available(False, reason)
        if self._calibrating_preset:
            # The stream died mid-run (a Fujifilm giving up its preview, issue #658). The ROI is
            # placed and metering needs only stills, so let the run finish instead of cancelling
            # work that is about to succeed. The window reports why the image froze.
            self.calib_window.set_status(f"⚠ {reason} The calibration continues without the preview.")
            self._apply_gating()
            return
        if self.calib_window.isVisible():
            self.calib_window.close()  # calibration cannot aim at the base without a stream
        # The refusal published the settings JSON on its way out, normally the preview loop's
        # job. Do what _on_live_view_started would have: fill the exposure steppers from it, then
        # put the selected preset's exposure on the body. Without this the steppers stay empty
        # and the preset is applied only per shot, inside the capture.
        self._settings_loaded = False
        self._refresh_camera_settings()
        self._apply_active_preset_camera_settings()
        self._set_status(reason, pinned=True)
        self._apply_gating()

    @pyqtSlot(str)
    def _on_live_view_failed(self, reason: str) -> None:
        """The preview thread died after its retries and the session was dropped (issue #617:
        the GFX50S II times out with [-110] three times) — without this, whichever pop-up was
        streaming spins on "loading live view" forever. Covers the scan window and the idle
        calibration window (which streams for crosshair placement before a run and as the
        exposure-abort retry surface); a death during a *running* calibration is left to that
        run's own capture error. Both pop-ups close and the same pinned panel warning names
        the reason — reopening is the retry."""
        if self._calibrating_preset:
            return
        if self.lv_btn.isChecked():
            self.lv_btn.setChecked(False)  # full teardown via _on_live_view_toggled(False)
        elif self.calib_window.isVisible():
            self.calib_window.close()  # its closed-handler stops the calib stream + routes the session
        else:
            return
        self._set_status(
            f"⚠ Live view failed — the camera stopped answering ({reason}). "
            "Reconnect or power-cycle the camera, then start Live View again.",
            pinned=True,
        )

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
        self._magnifier_seen = False
        self._focus_settle_timer.stop()

    def _track_magnifier(self, on: bool) -> None:
        """Follow the body's published magnifier state. A change from any cause restarts the
        focus peak and realigns the click toggle; a click's own echo is no change."""
        if on == self._magnifier_seen:
            return
        self._magnifier_seen = on
        self._magnifier_on = on
        self._reset_focus_meter()

    def _reset_focus_meter(self) -> None:
        self._focus_meter.reset()
        self.lv_window.set_focus(None)

    @pyqtSlot(str)
    def _on_magnifier_unavailable(self, reason: str) -> None:
        self._magnifier_available = False
        self._magnifier_on = False
        self._focus_settle_timer.stop()
        self._set_status(reason)

    def _on_magnifier_click(self, fx: float, fy: float) -> None:
        """Click the live view to magnify at that spot; click again for the full frame.
        Only while the stream is running."""
        if not self.lv_btn.isChecked():
            return
        self._reset_focus_meter()
        if not self._magnifier_available:
            return
        self._focus_settle_timer.start()
        if self._magnifier_on:
            self._on_magnifier_off()
            return
        x = max(0, min(639, round(fx * 640)))  # 640×480 grid → valid indices 0..639 / 0..479
        y = max(0, min(479, round(fy * 480)))
        self.controller.set_focus_magnifier_pos(x, y)
        self._magnifier_on = self._magnifier_seen = True

    def _on_magnifier_off(self) -> None:
        """Back to the full frame."""
        if self._magnifier_on:
            self.controller.set_focus_magnifier(False)
            self._magnifier_on = self._magnifier_seen = False
            self._set_status("Full frame — click the image to magnify")

    # ── live camera settings (ISO / shutter / aperture) ──────────

    def _on_camera_setting(self, which: str, combo) -> None:
        raw = combo.currentData()
        if raw is not None:
            # Buffer the latest value and write once the user pauses, so rapid stepping does not
            # queue a slow verified write per intermediate step.
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
        magnifier = data.get("magnifier")
        if isinstance(magnifier, dict):
            self._track_magnifier(bool(magnifier.get("on")))
        steppers = {
            "iso": [self.lv_window.iso_stepper],
            "shutter": [self.lv_window.shutter_stepper],
            "aperture": [self.lv_window.aperture_stepper],
        }
        # The calibration pop-up's ISO/aperture are frozen while it meters at them
        # (set_inputs_locked), so mirror the body onto them only when no calibration is running.
        # Otherwise the refresh re-enables the disabled steppers mid-run.
        if not self._calibrating_preset:
            steppers["iso"].append(self.calib_window.iso_stepper)
            steppers["aperture"].append(self.calib_window.aperture_stepper)
        # In manual mode the sidebar steppers are *controllers*, not mirrors: the body follows
        # them. Seed them from the body's choices once, or this periodic refresh snaps the user's
        # picks back whenever a write has not landed.
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
            self.status_strip.set_progress("Calibrating… %p%", frac)
            self._set_status(msg)

    @pyqtSlot(object)
    def _on_calibration_finished(self, result) -> None:
        self.status_strip.stop_progress()
        self._manual_mode = False  # a calibrated preset is read-only, never left in manual-edit mode
        levels, shutters = result.levels, result.shutters
        self.r_slider.setValue(int(levels[0]))
        self.g_slider.setValue(int(levels[1]))
        self.b_slider.setValue(int(levels[2]))
        self._light_debounce.start()  # setValue does not emit, and the new levels must reach the light
        # An RGB preset means the white LED is off. Without this the W slider keeps a prior white
        # preset's 255, which _update_settings_from_ui bakes into the saved preset.
        self._set_slider(self.w_slider, 0)
        shutter = shutters[0]  # one shared shutter (all three are equal)
        self._show_lone(self.shutter_stepper, shutter)
        self._set_capture_mode(result.single_capture)
        self._settings = replace(
            self._settings, white_mode=False, shutter_r=shutter, shutter_g=shutter, shutter_b=shutter, shutter_w=shutter
        )
        # The body sits at the metered ISO/aperture, so capture them: the preset bakes them and
        # later forces them, and the fields show them read-only.
        self._apply_preset_exposure(self._current_setting_label("iso"), self._current_setting_label("aperture", require_writable=True))
        self._update_settings_from_ui()
        self._save_settings()
        # A finished run is on target on every channel, since anything else aborts via
        # _on_calibration_exposure before a result exists, and it always carries its preset name:
        # the flow refuses to start without one. The guard remains because saving under an empty
        # name would write a nameless preset into the store.
        if self._calibrating_preset:
            name = self._calibrating_preset
            self._calibrating_preset = ""
            profile_note = ""
            sensor_profile = ""
            if self.calib_window.wants_sensor_profile():
                if result.sensor_matrix is not None:
                    SensorProfiles.save(name, list(result.sensor_matrix))
                    sensor_profile = name
                    profile_note = " and its sensor profile"
                else:
                    profile_note = ", with no sensor profile: the probe exposures were too dim to measure one"
            # persist + reload + select + re-gate (bakes settings.iso/aperture)
            self._save_current_as_preset(name, sensor_profile)
            self._lv_target = self.lv_image
            self.calib_window.hide()
            # Pinned: the slider writes above armed the light debounce, whose light_set echo lands
            # right after this line. Without the pin it replaced this outcome before anyone could
            # read it.
            self._set_status(f"Saved preset “{name}”{profile_note}", pinned=True)
        self._stop_calibration_live_view()  # calibration ran inside live view → tear it down

    @pyqtSlot(str)
    def _on_calibration_exposure(self, status: str) -> None:
        """The solver proved the exposure target unreachable ("over"/"under") and aborted — no
        preset exists. The calibration window stays a LIVE retry surface: unlike the cancel/error
        terminal, the live view keeps streaming into it (steppers keep mirroring the body, the ROI
        sits on a live frame) and the framing light is re-lit — the service switched the Scanlight
        off on its way out, and a black window here read as a crash on the rig. The user adjusts
        the aperture and recalibrates right there; the same advice lands in a pop-up."""
        name = self._calibrating_preset  # read before clearing (the pop-up names the preset)
        self._calibrating_preset = ""
        label, _cause, fix = _EXPOSURE_WARNINGS.get(status, _EXPOSURE_WARNINGS["over"])
        self.calib_window.set_inputs_locked(False)  # re-enable name / ROI / ISO / aperture for the retry
        self.calib_window.set_status(f"⚠ {label} — {fix}")
        self.calib_window.progress.setVisible(False)
        self._apply_gating()  # re-enable Scan — the capture thread is free again
        self.controller.set_scanlight_color(*REFERENCE_LEVELS, 0, self._settings.port)  # re-light for framing
        self._show_exposure_warning(name, status)

    def _show_exposure_warning(self, name: str, status: str) -> None:
        """The abort reason as a pop-up. The calibration window shows the same advice, but a run
        that ends without a preset deserves an unmissable surface. Non-blocking (show, not exec):
        the teardown behind it continues, and the pop-up is replaced on the next calibration."""
        label, cause, fix = _EXPOSURE_WARNINGS.get(status, _EXPOSURE_WARNINGS["over"])
        if self._exposure_popup is not None:
            self._exposure_popup.close()
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle("Calibration Stopped")
        box.setText(f"“{name}” was not saved — the film base is {label}.")
        box.setInformativeText(f"{cause} {fix[0].upper()}{fix[1:]}.")
        box.show()
        self._exposure_popup = box

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

    def _start_capture(self, retake: bool) -> None:
        if self._calibrating_preset:
            # Both ride one worker thread, so this would only queue, then fire with the exposure
            # the calibration is replacing.
            self._set_status("A calibration is running — wait for it to finish")
            return
        if self._scanning:
            return  # already capturing; a second click must not queue another frame
        output_folder = self.output.folder()
        if not output_folder:
            self.output.browse()
            output_folder = self.output.folder()
            if not output_folder:
                return
        if not (os.path.isabs(output_folder) and os.path.isdir(output_folder)):
            # Belt for a folder that vanished after gating last ran, or that was typed invalid.
            # Without this, makedirs resolves it against the app's working directory and the scan
            # lands somewhere the operator will never look.
            self._set_status(f"Output folder does not exist: {output_folder}")
            self._apply_gating()  # greys Scan and repeats the reason in the gate hint
            return

        roll = self.output.roll_name()
        if roll is None:
            self._set_status('Roll name must be a single safe name (not "." or "..", and no path separators).')
            return
        if not self.output.folder_is_roll():
            self.output.set_roll_text(roll)

        # Capture happens *inside* the live-view session, because the body grants one PTP claim.
        # The preview pauses for the shot and resumes, with no teardown and no reconnect.
        self._update_settings_from_ui()
        self._save_settings()

        from negpy.desktop.workers.capture_worker import CaptureRequest

        s = self._settings
        roll_folder = self.output.target_folder() or output_folder
        # Frame numbers come from the roll's folder, with no manual field: a fresh scan takes the
        # next free number and a retake overwrites the last one. The service creates the subfolder
        # before writing.
        last = self._last_frame_number(roll_folder, roll)
        frame_number = max(1, last if retake else last + 1)
        rgb = self._rgb_mode
        single = rgb and not s.white_mode and s.single_capture
        preset = (
            self._presets.get(self.preset_combo.currentData()) if single and self._preset_selected() and not self._manual_mode else None
        )
        req = CaptureRequest(
            roll_name=roll,
            frame_number=frame_number,
            output_folder=roll_folder,
            levels=(s.r_level, s.g_level, s.b_level),
            settle_s=_LED_SETTLE_S,
            inter_exposure_delay_s=s.inter_exposure_delay_ms / 1000.0,
            port=s.port,
            # Normal mode has no calibrated shutter or white level: the operator sets the exposure
            # with the live-view steppers, so leave the shutter blank and the camera keeps its own.
            shutters=(s.shutter_r, s.shutter_g, s.shutter_b) if rgb else ("", "", ""),
            white_mode=s.white_mode if rgb else False,
            w_level=s.w_level,
            shutter_w=s.shutter_w,
            white_process_mode=s.white_process_mode,
            is_retake=retake,
            rgb_mode=rgb,
            # Only the RGB triplet forces the preset's ISO/aperture. White-light and normal
            # scanning leave the body free, since the operator sets those in the live view.
            iso=s.iso if rgb and not s.white_mode else "",
            aperture=s.aperture if rgb and not s.white_mode else "",
            as_roll=self.output.as_roll(),
            single_capture=single,
            sensor_profile=preset.sensor_profile if preset is not None else "",
        )
        self.set_scanning(True)
        if rgb and not req.white_mode and not single:
            # Triplet progress arrives only after each channel, so show the bar at 0% right away
            # and the click has immediate feedback. Single-exposure captures emit no progress
            # events, so a bar there would sit at 0%. Skip it.
            self.lv_window.set_progress(0.0)
        self.controller.start_capture(req)

    @pyqtSlot(float)
    def _on_progress(self, progress: float) -> None:
        self.status_strip.set_progress("Capturing… %p%", progress)
        self.lv_window.set_progress(progress)

    @pyqtSlot(str)
    def _on_channel(self, letter: str) -> None:
        self.lv_window.set_channel(letter)

    @pyqtSlot(list)
    def _on_finished(self, paths: list) -> None:
        self.set_scanning(False)
        frame = paths[0].split("_Frame")[-1][:3] if paths else ""
        self.lv_window.flash_captured(frame)
        self._set_status(f"Captured frame {frame} — inverting in NegPy…")
        self._after_capture_live_view()  # re-light the still-running preview

    @pyqtSlot()
    def _on_cancelled(self) -> None:
        self.set_scanning(False)
        self.lv_window.clear_progress()
        if self._calibrating_preset:
            self._finish_calibration_terminal("Calibration canceled")
            self._set_status("Calibration canceled")
            return
        self._set_status("Capture canceled")
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
        self.lv_window.clear_progress()
        if self._calibrating_preset:
            # New-preset calibration failed: report in the pop-up, drop back to the scan target.
            self._finish_calibration_terminal(f"Calibration failed: {msg}")
        else:
            if self.lv_btn.isChecked():
                # CaptureWorker discards its camera session on errors, so close the frozen
                # preview. The operator can reopen it to establish a fresh session.
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
        self.status_strip.set_message(text)
        self.lv_window.set_status(text)

    def _set_conn_status(self, label, state, short: str, detail: str = "") -> None:
        label.setText(f"● {short}")
        set_hint_kind(label, "success" if state else ("error" if state is False else "muted"))
        label.setToolTip(detail or short)

    def _poll_connection_tick(self) -> None:
        """Timer tick: while the panel is visible, refresh the camera + light status in the
        background (the auto-connect that replaced 'Check'). Enumerating the USB bus does
        not claim the camera, so this keeps running through live view — that is the only
        way an unplug is noticed while the preview is up."""
        if not self.camera_body.isVisible():
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
        self._camera_has_preview = bool(status.get("camera_preview", True))
        self._camera_has_config = bool(status.get("camera_config", True))
        # An RGB triplet must write shutter/ISO/aperture onto the body. A camera whose driver
        # entry has no CONFIG cannot be told any of it, so only plain white-light scanning is
        # honest there, even with a Scanlight attached (issue #621).
        self._set_rgb_mode(status["light_ok"] and self._camera_has_config)
        self._refresh_light_channels()  # show/hide the W slider + white preset for the connected model
        claimed = bool(status.get("usb_claimed_elsewhere"))
        # A body another app holds the claim on is present but not usable. Gate Scan and
        # Calibrate as if it were absent, and let the dot say why instead of showing green.
        self._camera_verified = bool(status["usb_ok"]) and not claimed
        self._set_cam_status(bool(status["usb_ok"]), status["usb_model"], claimed_elsewhere=claimed)
        if self._camera_verified and not was_verified:
            self._set_status("")  # just connected → drop any stale failure line
        elif was_verified and not self._camera_verified and self.lv_btn.isChecked():
            # The body went away mid-stream, so close the preview instead of leaving the last
            # frame on screen looking live.
            self.lv_btn.setChecked(False)  # → _on_live_view_toggled(False) tears it down
            self._set_status("Camera disconnected")
        self._apply_gating()

    @pyqtSlot(object)
    def _on_light_temp(self, temp) -> None:
        """Show the live Scanlight LED temperature next to the Light status (amber when warm).
        RGB-only bodies (v1-v3) have no temperature sensor and report a bogus 0 °C, so hide it there
        (no white channel is our proxy for those models)."""
        if isinstance(temp, (int, float)) and self._light_has_white:
            set_hint_kind(self.light_temp, "warning" if temp >= 55 else "muted")
            self.light_temp.setText(f"{temp:.0f} °C")
            self.light_temp.show()
        else:
            self.light_temp.setText("")  # no light / no telemetry yet / RGB-only (no sensor)
            self.light_temp.hide()  # hide the widget entirely so no dark placeholder box lingers

    def _set_cam_status(self, ok: bool, model: str, claimed_elsewhere: bool = False) -> None:
        """Camera dot: '● Camera (USB)' when a body answered, '● Camera' when none did — and
        '● Camera (in use)' when it sits on the bus but another program holds its USB claim
        (on macOS the system camera daemon, woken by any app that watches for cameras). The
        enumeration behind the dot cannot see a claim, so without this state the dot showed a
        healthy green while every live-view/scan attempt failed."""
        if claimed_elsewhere:
            self._set_conn_status(self.cam_status, False, "Camera (in use)", "Another app is using the camera")
            # The advice must be readable in the tab, not buried in a tooltip. The connection
            # hint line turns amber and says what to do, instead of the "plug it in" nudge, which
            # is wrong advice for a present body.
            self._conn_hint.setText(
                "⚠ Another program is using the camera. On macOS a background app that watches for cameras, "
                "a cloud sync client for example, holds it through the system camera daemon. Quit that app or "
                "unplug the cable. NegPy reconnects by itself."
            )
            set_hint_kind(self._conn_hint, "warning")
            self._conn_hint.setVisible(True)
            return
        self._conn_hint.setText("Connect the camera by USB, in PC Remote mode — it's detected automatically.")
        set_hint_kind(self._conn_hint, "muted")
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
        # The worker runs one job at a time, so a scan clicked mid-calibration would only queue,
        # then fire with the exposure the calibration was about to replace.
        if self._calibrating_preset:
            m.append("wait for the calibration to finish")
        if self._pending_exposure_writes:
            m.append("wait for the preset exposure to reach the camera")
        if self._metering:
            m.append("wait for the meter")
        if not self._camera_verified:
            m.append("connect the camera")
        if self._rgb_mode:
            if not self._light_verified:
                m.append("connect the Scanlight")
            if not self._preset_selected():
                m.append("select or create a preset")
        folder = self.output.folder()
        if not folder:
            m.append("choose an output folder")
        elif not (os.path.isabs(folder) and os.path.isdir(folder)):
            # A typed or stale path (an ejected drive, a literal "~", relative text) must not gate
            # through. The capture's makedirs would create it relative to the app's working
            # directory and the scans would vanish with no error shown.
            m.append("choose a valid output folder (the current one does not exist)")
        return m

    def _apply_gating(self) -> None:
        """“Live View & Scan” needs camera+light+folder+preset; the new-preset (+)
        button only needs camera+light. When scanning is blocked, say why (task 5)."""
        missing = self._missing_requirements()
        can_scan = not missing
        # Live view needs only the camera; it stays enabled while open so it can be toggled off.
        self.lv_btn.setEnabled((self._camera_verified and not self._calibrating_preset) or self.lv_btn.isChecked())
        self.scan_btn.setEnabled(can_scan or self._scanning)
        self.retake_btn.setEnabled(can_scan and not self._scanning)
        # Calibration needs the camera, the light, an idle capture worker and live view, whose
        # stream is how the film base is aimed at: the crosshair sits on a live frame. A body
        # without preview can still scan, but it cannot be calibrated this way (issue #621).
        can_calibrate = (
            self._camera_verified
            and self._light_verified
            and self._camera_has_preview
            and not self._scanning
            and not self._calibrating_preset
        )
        self.preset_new_btn.setEnabled(can_calibrate)
        self.calib_window.calibrate_btn.setEnabled(can_calibrate)  # the pop-up may already be open
        self.preset_new_btn.setToolTip(
            "This camera has no live view, which calibration needs to aim at the film base"
            if not self._camera_has_preview and self._camera_verified
            else "Create a preset by calibrating on the film base (auto-meters the exposure)"
        )
        for btn in (self.lv_window.scan_btn, self.lv_window.retake_btn):
            btn.setEnabled(can_scan)
        # Meter needs the camera and an idle capture worker, not a folder or a preset.
        self.lv_window.meter_btn.setEnabled(
            self._camera_verified and not self._scanning and not self._calibrating_preset and not self._metering
        )
        if missing:
            self.scan_btn.setToolTip(wrap_tooltip("Can't scan yet — " + "; ".join(missing)))
            self.gate_hint.setText("⚠ To scan: " + ", ".join(missing) + ".")
            self.gate_hint.setVisible(True)
        else:
            self.scan_btn.setToolTip(wrap_tooltip("Capture the next frame into the roll, or stop the capture"))
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
        self.lv_window.meter_btn.setVisible(not locked)
        if not hasattr(self, "inter_exposure_delay_slider"):
            return  # first call lands during __init__, before the RGB section is built
        self._exposure_widget.setVisible(not self._settings.white_mode)
        editable = self._manual_mode
        self.capture_btn.setEnabled(editable)
        self.inter_exposure_delay_slider.setEnabled(not self._settings.single_capture)
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
        # A manual preset steps through the *camera's* own ISO/shutter/aperture choices, so grey
        # the option out with no camera: the values differ per model.
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
        self.w_slider.setVisible(has_white)
        model = self.preset_combo.model()
        for white_name in _BUILTIN_WHITE_PRESETS:
            white_idx = self.preset_combo.findData(white_name)
            if white_idx >= 0 and isinstance(model, QStandardItemModel):
                item = model.item(white_idx)
                if item is not None:
                    item.setEnabled(has_white)
        # A white-light preset selected on an RGB-only body cannot run, so drop it.
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
        self.light_body.setVisible(on)
        self._rgb_hint.setVisible(not on)
        self.cards_changed.emit()
        if not on:
            self._set_status("")  # drop a lingering "Light: R… G… B…" — there's no Scanlight now
        self._apply_gating()

    # ── state helpers ─────────────────────────────────────────────────

    def set_scanning(self, active: bool) -> None:
        self._scanning = active
        if active:
            self.status_strip.start_progress("Capturing… %p%")
        else:
            self.status_strip.stop_progress()
        self.scan_btn.setText(" Stop" if active else " Scan")
        self.scan_btn.setIcon(
            qta.icon("fa5s.stop" if active else "fa5s.camera", color=THEME.accent_secondary if active else THEME.text_on_accent)
        )
        # Qt re-reads a QSS property selector only on a repolish.
        self.scan_btn.setProperty("scanning", "true" if active else "false")
        style = self.scan_btn.style()
        style.unpolish(self.scan_btn)
        style.polish(self.scan_btn)
        self.lv_window.set_scanning(active)
        self._apply_gating()  # a running scan locks the "+" calibration button

    def _update_settings_from_ui(self) -> None:
        # white_mode / white_process_mode are set by preset selection, not by widgets. Exposure
        # comes from the steppers only while building a manual preset. Otherwise it is the preset
        # or calibration value already in settings, not what the disabled steppers show.
        if self._manual_mode:
            shutter = self._stepper_label(self.shutter_stepper)
            iso = self._stepper_label(self.iso_stepper)
            aperture = self._stepper_label(self.aperture_stepper)
        else:
            shutter, iso, aperture = self._settings.shutter_r, self._settings.iso, self._settings.aperture
        updated = replace(
            self._settings,
            r_level=int(self.r_slider.value()),
            g_level=int(self.g_slider.value()),
            b_level=int(self.b_slider.value()),
            w_level=int(self.w_slider.value()),
            inter_exposure_delay_ms=int(self.inter_exposure_delay_slider.value()),
            single_capture=self.capture_btn.currentIndex() == 1,
            shutter_r=shutter,
            shutter_g=shutter,
            shutter_b=shutter,
            # A white-light frame is not calibrated, so there is no baked exposure to force: its
            # live-view steppers are the exposure control (see _refresh_preset_ui). Copying the RGB
            # shutter here overwrote the operator's choice with a narrowband value, too long under
            # white light and refused outright by a dial-locked body (issue #746).
            shutter_w="" if self._settings.white_mode else shutter,
            iso=iso,
            aperture=aperture,
        )
        if updated == self._settings:
            return  # nothing changed → skip the disk write + re-gate (the 3 s poll calls this each tick)
        self._settings = updated
        self._save_settings()
        self._apply_gating()
