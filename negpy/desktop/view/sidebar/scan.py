from negpy.kernel.system.i18n import tr
import os
from enum import StrEnum

import qtawesome as qta
from PyQt6.QtCore import QTimer, QUrl, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QDesktopServices, QIntValidator
from PyQt6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QInputDialog,
    QLineEdit,
    QMenu,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from negpy.kernel.system.text import count_of, human_bytes
from negpy.desktop.view.sidebar.scan_output import ScanOutputPanel
from negpy.desktop.view.styles.templates import (
    default_button_height,
    field_row,
    hint_label,
    set_hint_kind,
    icon_button as _icon_button,
    icon_text,
    labeled_action,
    SCAN_BUTTON_HEIGHT,
    StatusStrip,
    tool_toggle,
    wrap_tooltip,
)
from negpy.desktop.view.styles.theme import THEME
from negpy.desktop.view.widgets.segmented_choice import SegmentedChoice
from negpy.desktop.view.widgets.sliders import CompactSlider, SliderGroup
from negpy.desktop.view.widgets.step_combo import StepCombo
from negpy.infrastructure.scanners.base import ScannerCapabilities, ScannerDevice
from negpy.infrastructure.scanners.params import (
    DEFAULT_N_PASSES,
    FILM_TYPES,
    FilmType,
    MAX_N_PASSES,
    MIN_N_PASSES,
    MultiExposureMode,
    film_passes_infrared,
)
from negpy.infrastructure.scanners import nkscan_log
from negpy.infrastructure.scanners.registry import DEFAULT_BACKEND_ID, backend_choices
from negpy.infrastructure.scanners.settings import OUTPUT_FORMATS, ScannerSettings


class ScanCaptureMode(StrEnum):
    """The 4 scan modes this app exposes — a UI-only presentation of pyopticfilm's two real,
    orthogonal axes (``multi_exposure`` and ``n_passes``). Never persisted or sent to the
    backend directly; ``_capture_mode_from_params``/``_params_from_capture_mode`` translate to
    and from the real ``ScanParams``/``ScannerSettings`` fields. pyopticfilm's manual exposure
    overrides are lab/debug-only (see Scan Lab) and have no equivalent here."""

    SINGLE_PASS = "single_pass"
    MULTI_PASS = "multi_pass"
    ADAPTIVE_ME = "adaptive_me"
    ADAPTIVE_MULTI_PASS = "adaptive_multi_pass"


#: Label + explanatory tooltip for each capture mode, in display order.
_CAPTURE_MODE_LABELS: tuple[tuple[ScanCaptureMode, str, str], ...] = (
    (ScanCaptureMode.SINGLE_PASS, tr("Single-Pass"), tr("One exposure per scan, the fastest at standard quality.")),
    (
        ScanCaptureMode.MULTI_PASS,
        tr("Multi-Pass"),
        tr("Repeats the same exposure and stacks the results to reduce noise, slower and best for a well-exposed frame."),
    ),
    (
        ScanCaptureMode.ADAPTIVE_ME,
        tr("Adaptive Multi-Exposure"),
        tr("Captures a short and a long exposure and fuses them for more dynamic range, without stacking."),
    ),
    (
        ScanCaptureMode.ADAPTIVE_MULTI_PASS,
        tr("Adaptive Multi-Pass"),
        tr("Combines the short and long exposure fusion with multi-pass stacking for the most range and least noise, the slowest option."),
    ),
)

_ME_CAPTURE_MODES = (ScanCaptureMode.ADAPTIVE_ME, ScanCaptureMode.ADAPTIVE_MULTI_PASS)
_STACKING_CAPTURE_MODES = (ScanCaptureMode.MULTI_PASS, ScanCaptureMode.ADAPTIVE_MULTI_PASS)

#: Passes slider floor. UI-only — 1 pass is "not stacking", represented by mode choice, so the
#: slider (shown only for a stacking mode) never needs to reach it. The ceiling is
#: pyopticfilm's own ``MAX_N_PASSES``, imported directly since this file already reaches into
#: params.py for other names.
MIN_PASSES_UI = 2


def _capture_mode_from_params(mode: MultiExposureMode, n_passes: int) -> ScanCaptureMode:
    me = mode == MultiExposureMode.ADAPTIVE
    stacking = n_passes > 1
    if me:
        return ScanCaptureMode.ADAPTIVE_MULTI_PASS if stacking else ScanCaptureMode.ADAPTIVE_ME
    return ScanCaptureMode.MULTI_PASS if stacking else ScanCaptureMode.SINGLE_PASS


def _params_from_capture_mode(capture_mode: ScanCaptureMode, slider_value: int) -> tuple[MultiExposureMode, int]:
    mode = MultiExposureMode.ADAPTIVE if capture_mode in _ME_CAPTURE_MODES else MultiExposureMode.OFF
    n_passes = slider_value if capture_mode in _STACKING_CAPTURE_MODES else 1
    return mode, n_passes


def _valid_capture_mode(desired: ScanCaptureMode, *, has_me: bool, has_stack: bool) -> ScanCaptureMode:
    """``desired`` masked against what the device actually supports — drops just the axis
    (ME or stacking) the device lacks, rather than falling all the way back to Single-Pass
    unless neither axis is available."""
    want_me = desired in _ME_CAPTURE_MODES and has_me
    want_stack = desired in _STACKING_CAPTURE_MODES and has_stack
    if want_me and want_stack:
        return ScanCaptureMode.ADAPTIVE_MULTI_PASS
    if want_me:
        return ScanCaptureMode.ADAPTIVE_ME
    if want_stack:
        return ScanCaptureMode.MULTI_PASS
    return ScanCaptureMode.SINGLE_PASS


_SAMPLE_COUNTS = (1, 2, 4, 8, 16)


def _reaches_a_strip(caps: ScannerCapabilities) -> bool:
    """Whether a device holds several frames: a feeder with a capacity, or a measured strip."""
    return caps.adapter_frame_capacity is not None or caps.roll_discovery


def estimated_frame_bytes(
    caps: ScannerCapabilities,
    dpi: int,
    depth: int,
    *,
    capture_ir: bool = False,
    window: tuple[float, float, float, float] | None = None,
) -> int:
    """Uncompressed size of one scanned frame, for the summary the operator reads before
    committing a batch. The frame area comes from the adapter, the window shrinks it, and
    an IR pass adds a fourth plane."""
    width_mm, height_mm = caps.max_area_mm if caps.max_area_mm else (36.0, 24.0)
    pixels = (width_mm / 25.4 * dpi) * (height_mm / 25.4 * dpi)
    if window and len(window) == 4:
        x1, y1, x2, y2 = window
        pixels *= max(0.0, x2 - x1) * max(0.0, y2 - y1)
    planes = 4 if capture_ir else 3
    return int(pixels * planes * (2 if depth > 8 else 1))


class ScanSidebar(QWidget):
    """Scanner control panel — replaces the originally planned modal ScanDialog."""

    _INDETERMINATE_SCAN_PHASES = frozenset({tr("Preparing long exposure"), "Merging exposures"})
    # How often the panel asks whether film is loaded. A unit that ejects without a command needs it.
    _MEDIA_POLL_MS = 1500

    cards_changed = pyqtSignal()

    def __init__(self, controller, output: ScanOutputPanel | None = None) -> None:
        super().__init__()
        self.controller = controller
        self.output = output if output is not None else ScanOutputPanel(controller.session.repo)
        self._settings: ScannerSettings = self._load_settings()
        self._devices: list[ScannerDevice] = []
        self._scanning = False
        self._devices_loaded = False
        self._media_loaded: bool | None = None  # the last poll for the selected device, None if unknown
        self._media_poll_pending = False
        self._caps_autofocus = False
        self._caps_auto_exposure = False
        self._caps_clean = False
        self._caps_superfine = False
        self._caps_max_samples = 1
        self._caps_film_formats: tuple[str, ...] = ()
        self._caps_film_types: tuple[str, ...] = ()
        self._device_ir = False
        self._init_ui()
        self._connect_signals()
        self._sync_debug_log()

    def _load_settings(self) -> ScannerSettings:
        from dataclasses import replace

        data = self.controller.session.repo.get_global_setting("scanner_settings", default={})
        if isinstance(data, dict) and data:
            try:
                settings = ScannerSettings.from_dict(data)
            except Exception:
                settings = ScannerSettings.defaults()
        else:
            settings = ScannerSettings.defaults()
        # Drop backends that no longer ship on this platform (e.g. saved "sane" on Windows).
        if settings.backend not in {bid for bid, _ in backend_choices()}:
            settings = replace(settings, backend=DEFAULT_BACKEND_ID)
        # Frame picks belong to the loaded strip; the unit can return it while NegPy is closed.
        return replace(settings, selected_frames=(), frame_windows={}, frame_offsets={})

    def _save_settings(self) -> None:
        from dataclasses import asdict

        self.controller.session.repo.save_global_setting("scanner_settings", asdict(self._settings))

    @property
    def settings(self) -> ScannerSettings:
        return self._settings

    @settings.setter
    def settings(self, value: ScannerSettings) -> None:
        self._settings = value
        self._save_settings()
        # Every writer routes through here, so the frame box follows the selection wherever it
        # was set — the strip dialog, a clear, an eject. A stale box would wipe it on the next
        # edit of any other control.
        self._sync_frame_spec()

    @staticmethod
    def _body() -> tuple[QWidget, QVBoxLayout]:
        body = QWidget()
        body.setObjectName("collapsible_content_body")
        col = QVBoxLayout(body)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(THEME.space_md)
        return body, col

    @staticmethod
    def _row_widget(row: QHBoxLayout) -> QWidget:
        row.setContentsMargins(0, 0, 0, 0)
        widget = QWidget()
        widget.setObjectName("collapsible_content_body")
        widget.setLayout(row)
        return widget

    def _init_ui(self) -> None:
        """Five bodies that the Scan tab puts in cards; a standalone panel stacks them."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(THEME.space_lg)

        self.device_body, device = self._body()
        backends = backend_choices()
        self.backend_btn = StepCombo.of(
            [label for _id, label in backends], tr("Scanner transport backend"), [bid for bid, _label in backends]
        )
        self.backend_btn.setCurrentIndex(max(self.backend_btn.findData(self._settings.backend), 0))
        device.addLayout(field_row(tr("Backend"), self.backend_btn))

        self.device_combo = StepCombo()
        self.device_combo.setToolTip(wrap_tooltip(tr("Select scanner")))
        self.device_combo.addItem(tr("Detecting scanners…"), None)
        self.refresh_btn = _icon_button("fa5s.redo", tr("Refresh device list"))
        self.eject_btn = _icon_button("fa5s.eject", tr("Eject now, or eject the strip when a batch is done"))
        self.eject_btn.setObjectName("icon_menu_btn")
        eject_menu = QMenu(self.eject_btn)
        eject_menu.setToolTipsVisible(True)
        self.eject_now_act = eject_menu.addAction(tr("Eject Now"))
        self.eject_after_act = eject_menu.addAction(tr("Eject When Done"))
        self.eject_after_act.setCheckable(True)
        self.eject_after_act.setChecked(self._settings.eject_after_batch)
        self.eject_after_act.setToolTip(tr("Return the strip after a batch, or keep it and its previews loaded when this is off."))
        self.eject_after_act.setVisible(False)
        self.eject_btn.setMenu(eject_menu)
        self.eject_btn.setVisible(False)
        device.addLayout(field_row(tr("Device"), self.device_combo, self.refresh_btn, self.eject_btn))

        self.debug_log_btn = StepCombo.of(
            [level.title() for level in nkscan_log.LEVELS],
            tr("Write a diagnostics log (nkscan.log) in the NegPy folder, with Trace for a bug report."),
        )
        level = self._settings.nkscan_log_level
        self.debug_log_btn.setCurrentIndex(nkscan_log.LEVELS.index(level) if level in nkscan_log.LEVELS else 0)
        self.debug_log_folder_btn = _icon_button("fa5s.folder-open", tr("Show nkscan.log in its folder"))
        self.debug_log_widget = self._row_widget(field_row(tr("Debug log"), self.debug_log_btn, self.debug_log_folder_btn))
        device.addWidget(self.debug_log_widget)

        # The crop label shows only when there is a crop. This crop is the Prescan crop.
        self.crop_label = hint_label("")
        self.crop_label.setVisible(False)
        device.addWidget(self.crop_label)
        layout.addWidget(self.device_body)

        self.quality_body, quality = self._body()

        # What is on the film: it decides which way the frame boundaries read on a strip, and
        # whether an IR pass has anything to see.
        self.film_type_btn = StepCombo.of([], tr("What is on the film, because silver and Kodachrome block infrared"))
        self.film_type_widget = self._row_widget(field_row(tr("Film"), self.film_type_btn))
        quality.addWidget(self.film_type_widget)

        # Frame length, for a transport that measures the strip and cannot infer it.
        self.format_combo = StepCombo()
        self.format_combo.setToolTip(wrap_tooltip(tr("Frame length on the loaded film; Auto where the holder fixes it")))
        # Not "Format": the output file format owns that label in the Output card.
        self.film_format_widget = self._row_widget(field_row(tr("Film format"), self.format_combo))
        quality.addWidget(self.film_format_widget)

        self.dpi_combo = QComboBox()
        self.dpi_combo.setToolTip(wrap_tooltip(tr("Scan resolution in dots per inch; type a value between the listed stops")))
        self.dpi_combo.setEditable(True)
        self.dpi_combo.setFixedHeight(default_button_height())
        quality.addLayout(field_row(tr("Resolution"), self.dpi_combo))

        self.depth_btn = StepCombo.of([], tr("Bits per channel in the scanned file"))
        self.depth_widget = self._row_widget(field_row(tr("Bit depth"), self.depth_btn))
        quality.addWidget(self.depth_widget)

        self.mode_btn = StepCombo.of(
            [label for _mode, label, _tip in _CAPTURE_MODE_LABELS],
            tr("How the scan is exposed: one pass, several passes for less noise, a short and a long exposure for more range, or both."),
            [mode.value for mode, _label, _tip in _CAPTURE_MODE_LABELS],
        )
        for i, (_mode, _label, tip) in enumerate(_CAPTURE_MODE_LABELS):
            self.mode_btn.set_item_tooltip(i, tip)
        self.mode_widget = self._row_widget(field_row(tr("Scan mode"), self.mode_btn))
        quality.addWidget(self.mode_widget)
        # The mode the user picked; the combo shows it masked by what the device does at this resolution.
        self._wanted_capture_mode = ScanCaptureMode.SINGLE_PASS
        self.mode_hint = hint_label("")
        self.mode_hint.setVisible(False)
        quality.addWidget(self.mode_hint)
        self.passes_slider = CompactSlider(tr("Passes"), MIN_PASSES_UI, MAX_N_PASSES, DEFAULT_N_PASSES, step=1, precision=1)
        self.passes_slider.setToolTip(tr("Number of exposures to stack, where each extra pass adds about one scan pass per exposure."))
        self.passes_rail = SliderGroup(self.passes_slider)
        self.passes_rail.setVisible(False)
        quality.addWidget(self.passes_rail)

        # Multi-sample: repeated reads of one line the scanner averages, for shadow noise.
        self.samples_btn = StepCombo.of([], tr("Reads per line the scanner averages"))
        self.samples_widget = self._row_widget(field_row(tr("Samples"), self.samples_btn))
        quality.addWidget(self.samples_widget)

        self.ir_btn = tool_toggle("fa5s.layer-group", tr("IR"), tr("Scan a separate infrared channel for dust detection"))
        # IR and ICE read the same pass, and ICE bakes its repair into the file, so a raw IR
        # plane beside it would only have cleaned pixels to detect dust in: one choice of three.
        self.dust_btn = SegmentedChoice(
            (("", tr("Off")), ("fa5s.layer-group", tr("IR")), ("fa5s.broom", tr("ICE"))),
            tr("How the scan uses the infrared pass against dust: not at all, as a separate channel, or as ICE in the scanner."),
            data=("off", "ir", "ice"),
        )
        self.dust_btn.set_choice_tooltip(1, tr("Scan a separate infrared channel for dust detection"))
        self.dust_btn.set_choice_tooltip(2, tr("Remove dust and scratches with the infrared channel while scanning, baked into the file"))
        self.dust_widget = self._row_widget(field_row(tr("Infrared"), self.dust_btn))
        self.dust_widget.setVisible(False)
        quality.addWidget(self.dust_widget)
        self.superfine_btn = tool_toggle(
            "fa5s.align-justify", tr("Superfine"), tr("Read one line per pass: slower, and free of line registration")
        )
        self.autofocus_btn = tool_toggle("fa5s.bullseye", tr("Autofocus"), tr("Autofocus before scanning (film is rarely perfectly flat)"))
        self.autofocus_btn.setChecked(True)
        self.ae_btn = tool_toggle("fa5s.sun", tr("Auto-exposure"), tr("Meter exposure in hardware before the scan"))
        for row_btns in ((self.ir_btn, self.superfine_btn), (self.autofocus_btn, self.ae_btn)):
            row = QHBoxLayout()
            for btn in row_btns:
                btn.setVisible(False)
                row.addWidget(btn, 1)
            quality.addLayout(row)

        # Scan exposure time (SANE `scan-exposure-time`), shown only when the device reports a
        # usable range.
        self.exposure_slider = CompactSlider(tr("Exposure"), 0, 1, 0, step=1, precision=1, unit=" µs")
        self.exposure_slider.setToolTip(tr("Scan exposure time, for a scanner that exposes it"))
        self.exposure_slider.setVisible(False)
        quality.addWidget(self.exposure_slider)
        layout.addWidget(self.quality_body)

        self.framing_body, framing = self._body()

        # Which frames the batch scans, for roll and strip feeders only.
        self.frame_spec_edit = QLineEdit()
        self.frame_spec_edit.setPlaceholderText(tr("All Frames"))
        self.frame_spec_edit.setToolTip(wrap_tooltip(tr("Frames to scan, such as 1-6 or 1,2,5, or empty for every frame.")))
        self.frame_spec_widget = self._row_widget(field_row(tr("Frames"), self.frame_spec_edit))
        framing.addWidget(self.frame_spec_widget)

        # Scan window (strip/roll feeders): set once from a preview, reused per frame.
        self.scan_window_btn = labeled_action(
            "fa5s.eye", tr("Preview…"), tr("Preview a frame and set the scan window reused for every frame")
        )
        self.scan_window_clear_btn = labeled_action("fa5s.times", tr("Clear"), tr("Scan the whole default frame instead"))
        scan_window_row = field_row(tr("Batch"), self.scan_window_btn, self.scan_window_clear_btn)
        self.scan_window_row_label = scan_window_row.itemAt(0).widget()
        self.scan_window_widget = self._row_widget(scan_window_row)
        framing.addWidget(self.scan_window_widget)
        self.scan_window_status = hint_label("")
        framing.addWidget(self.scan_window_status)
        self.scan_window_widget.setVisible(False)
        self.scan_window_status.setVisible(False)

        # Exposure lock: meter one frame, then every scan of the roll reuses its exposure.
        self.exposure_meter_btn = labeled_action(
            "fa5s.lock",
            tr("Meter Frame…"),
            tr("Measure one frame inside the strip, not at its end, and use its exposure for every scan until you unlock it."),
        )
        self.exposure_unlock_btn = labeled_action("fa5s.lock-open", tr("Unlock"), tr("Meter every frame on its own again"))
        self.exposure_lock_widget = self._row_widget(field_row(tr("Exposure lock"), self.exposure_meter_btn, self.exposure_unlock_btn))
        framing.addWidget(self.exposure_lock_widget)
        self.exposure_lock_status = hint_label("")
        framing.addWidget(self.exposure_lock_status)
        self._set_exposure_lock_visible(False)

        # Prescan + crop (Plustek SE): low-DPI full window → interactive crop → scan_window.
        self.prescan_btn = labeled_action("fa5s.eye", tr("Prescan…"), tr("Scan a low-DPI preview and set the crop for the next scan"))
        self.prescan_clear_btn = labeled_action("fa5s.times", tr("Clear"), tr("Scan the full window instead of a crop"))
        self.prescan_widget = self._row_widget(field_row(tr("Crop"), self.prescan_btn, self.prescan_clear_btn))
        self.prescan_widget.setVisible(False)
        framing.addWidget(self.prescan_widget)
        layout.addWidget(self.framing_body)

        self.output_body, out = self._body()
        self.fmt_btn = StepCombo.of(
            list(OUTPUT_FORMATS),
            tr("Output file format, where Mono writes one gray plane, for film with a single record."),
            list(OUTPUT_FORMATS),
        )
        out.addLayout(field_row(tr("Format"), self.fmt_btn))
        self.pattern_edit = QLineEdit()
        self.pattern_edit.setToolTip(
            wrap_tooltip(tr('Jinja2 template with the variables {{ date }} and {{ seq }}, for example {{ date }}_{{ "%03d" % seq }}'))
        )
        out.addLayout(field_row(tr("Filename"), self.pattern_edit))
        layout.addWidget(self.output_body)

        # One reserved row for all three: the pass that is running, the message it left, and
        # the resting summary of what Scan will cost. Three rows that come and go move the
        # button under them, which is where the cursor already is.
        self.footer, foot = self._body()
        self.status_strip = StatusStrip()
        foot.addWidget(self.status_strip)
        self.scan_btn = QPushButton(icon_text(tr("Scan")))
        self.scan_btn.setObjectName("scan_btn")
        self.scan_btn.setFixedHeight(SCAN_BUTTON_HEIGHT)
        self.scan_btn.setProperty("scanning", "false")
        self.scan_btn.setIcon(qta.icon("fa5s.camera-retro", color=THEME.text_on_accent))
        self.scan_btn.setToolTip(wrap_tooltip(tr("Scan the selected frames into the output folder, or stop the scan")))
        foot.addWidget(self.scan_btn)
        layout.addWidget(self.footer)

        layout.addStretch()

        # Pre-fill from persisted settings
        self.fmt_btn.setCurrentIndex(max(self.fmt_btn.findData(self._settings.output_format), 0))
        self.pattern_edit.setText(self._settings.filename_pattern)
        self.autofocus_btn.setChecked(self._settings.autofocus)
        self.ae_btn.setChecked(self._settings.auto_exposure)
        self.exposure_slider.setEnabled(not self._settings.auto_exposure)

    def _connect_signals(self) -> None:
        self.refresh_btn.clicked.connect(self._on_refresh)
        self.eject_now_act.triggered.connect(self._on_eject)
        self.debug_log_btn.currentIndexChanged.connect(self._on_debug_log_changed)
        self.debug_log_folder_btn.clicked.connect(self._on_show_debug_log)
        self.backend_btn.currentIndexChanged.connect(self._on_backend_changed)
        self.device_combo.currentIndexChanged.connect(self._on_device_changed)
        self.scan_btn.clicked.connect(self._on_scan)
        self.pattern_edit.textChanged.connect(lambda: self._update_settings_from_ui())
        self.fmt_btn.currentIndexChanged.connect(lambda: self._update_settings_from_ui())
        self.dpi_combo.currentTextChanged.connect(self._on_dpi_changed)
        self.depth_btn.currentIndexChanged.connect(lambda: self._update_settings_from_ui())
        self.ir_btn.toggled.connect(self._on_ir_toggled)
        self.mode_btn.currentIndexChanged.connect(self._on_mode_changed)
        self.passes_slider.valueChanged.connect(lambda _v: self._update_settings_from_ui())
        self.autofocus_btn.toggled.connect(lambda: self._update_settings_from_ui())
        self.eject_after_act.toggled.connect(lambda: self._update_settings_from_ui())
        self.ae_btn.toggled.connect(lambda: self._on_ae_toggled())
        self.dust_btn.currentChanged.connect(lambda _i: self._on_ir_toggled(self.dust_btn.currentData() == "ir"))
        self.superfine_btn.toggled.connect(lambda: self._update_settings_from_ui())
        self.samples_btn.currentIndexChanged.connect(lambda: self._update_settings_from_ui())
        self.format_combo.currentIndexChanged.connect(lambda: self._update_settings_from_ui())
        self.film_type_btn.currentIndexChanged.connect(lambda: self._on_film_type_changed())
        self.exposure_slider.valueChanged.connect(lambda _v: self._update_settings_from_ui())
        self.frame_spec_edit.textChanged.connect(self._on_frame_spec_typed)
        self.frame_spec_edit.editingFinished.connect(self._update_settings_from_ui)
        self.scan_window_btn.clicked.connect(self._on_set_scan_window)
        self.scan_window_clear_btn.clicked.connect(self._on_clear_scan_window)
        self.prescan_btn.clicked.connect(self._on_prescan)
        self.prescan_clear_btn.clicked.connect(self._on_clear_prescan_crop)
        self.exposure_meter_btn.clicked.connect(self._on_meter_frame)
        self.exposure_unlock_btn.clicked.connect(self._on_unlock_exposure)

        # Controller signals
        self.controller.scan_devices_ready.connect(self._on_devices_ready)
        self.controller.scan_progress.connect(self._on_scan_progress)
        self.controller.scan_finished.connect(self._on_scan_finished)
        self.controller.scan_error.connect(self._on_scan_error)
        self.controller.scan_cancelled.connect(self._on_scan_cancelled)
        self.controller.scan_frame_done.connect(self._on_scan_frame_done)
        self.controller.scan_batch_finished.connect(self._on_scan_batch_finished)
        self.controller.scan_ejected.connect(self._on_ejected)
        self.controller.scan_eject_error.connect(self._on_eject_error)
        self.controller.scan_strip_returned.connect(self._on_strip_returned)
        self.controller.scan_media_polled.connect(self._on_media_polled)
        self._media_timer = QTimer(self)
        self._media_timer.setInterval(self._MEDIA_POLL_MS)
        self._media_timer.timeout.connect(self._poll_media)
        self._media_timer.start()
        self.controller.scan_exposure_metered.connect(self._on_exposure_metered)
        self.controller.scan_meter_error.connect(self._on_meter_error)

    def on_activated(self) -> None:
        """Called when the Scan tab is switched to."""
        if not self._devices_loaded:
            self._request_devices()

    def _request_devices(self) -> None:
        """Request device list from the scan worker thread."""
        self._sync_debug_log()
        self.controller.set_scan_backend(self._current_backend_id())
        self.device_combo.clear()
        self.device_combo.addItem(tr("Detecting scanners…"), None)
        self.device_combo.setEnabled(False)
        self.status_strip.set_message(tr("Detecting scanners…"))
        self.controller.request_scan_devices()

    def _on_refresh(self) -> None:
        self._request_devices()

    def _current_backend_id(self) -> str:
        return self.backend_btn.currentData() or DEFAULT_BACKEND_ID

    def _on_backend_changed(self, _index: int) -> None:
        # Device lists are backend-specific, so persist the choice and then re-enumerate.
        # Per-backend UI tweaks that capabilities cannot express branch here on
        # _current_backend_id(). None are needed today.
        self._update_settings_from_ui()
        self._request_devices()

    def _sync_debug_log(self) -> None:
        # Before the device request, so the log holds the probe that opens the unit.
        is_nkscan = self._current_backend_id() == "nkscan"
        self.debug_log_widget.setVisible(is_nkscan)
        if is_nkscan:
            self._apply_debug_log(self._settings.nkscan_log_level)

    def _on_debug_log_changed(self, index: int) -> None:
        self._apply_debug_log(nkscan_log.LEVELS[index])

    def _apply_debug_log(self, level: str) -> None:
        """Start `level` and save it; a level that cannot start shows and saves Off."""
        from dataclasses import replace

        if not nkscan_log.set_level(level):
            level = "off"
            self.debug_log_btn.blockSignals(True)
            self.debug_log_btn.setCurrentIndex(0)
            self.debug_log_btn.blockSignals(False)
            self.status_strip.set_message(tr("nkscan is not installed, so there is no debug log to write"))
        if level != self._settings.nkscan_log_level:
            self.settings = replace(self._settings, nkscan_log_level=level)

    def _on_show_debug_log(self) -> None:
        path = nkscan_log.log_path()
        folder = os.path.dirname(path)
        os.makedirs(folder, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(folder))

    def _on_eject(self) -> None:
        device = self._current_device()
        if device is None:
            return
        self.eject_btn.setEnabled(False)
        self.status_strip.set_message(tr("Ejecting film…"))
        self.controller.eject_scanner(device.id)

    @pyqtSlot(list)
    def _on_devices_ready(self, devices: list) -> None:
        self._devices = devices
        self._devices_loaded = True
        self.device_combo.clear()
        self.device_combo.setEnabled(True)

        if not devices:
            self.device_combo.addItem(tr("No scanners detected"), None)
            self.device_combo.setEnabled(False)
            self.status_strip.set_message(tr("No scanners detected. Plug in your scanner and click Refresh."))
            self.scan_btn.setEnabled(False)
            return

        for d in devices:
            label_text = f"{d.vendor} {d.model}" if d.vendor else d.model
            self.device_combo.addItem(label_text, d.id)

        # Restore last-used device if present
        if self._settings.last_device_id:
            for i in range(self.device_combo.count()):
                if self.device_combo.itemData(i) == self._settings.last_device_id:
                    self.device_combo.setCurrentIndex(i)
                    break

        self._update_device_caps()

    def _on_device_changed(self, _index: int) -> None:
        self._media_loaded = None
        self._update_device_caps()

    def _poll_media(self) -> None:
        """Ask whether film is loaded, while the panel shows and the unit is idle.

        The right panel shows the parts of this widget, not the widget, thus the footer tells if the panel shows.
        """
        device = self._current_device()
        if device is None or self._scanning or self._media_poll_pending or not self.footer.isVisible():
            return
        self._media_poll_pending = True
        self.controller.poll_scanner_media(device.id)

    @pyqtSlot(str, object)
    def _on_media_polled(self, device_id: str, loaded: bool | None) -> None:
        self._media_poll_pending = False
        device = self._current_device()
        if loaded is None or device is None or device.id != device_id:
            return
        before, self._media_loaded = self._media_loaded, loaded
        if before is None:
            if not loaded:
                self.status_strip.set_message(tr("No film loaded"))
            return
        if loaded == before:
            return
        if loaded:
            self.status_strip.set_message(tr("Film loaded"))
            return
        stale = self._drop_strip_state()
        self.status_strip.set_message(tr("Film removed. Frame selection cleared.") if stale else tr("Film removed"))

    def _current_device(self) -> ScannerDevice | None:
        device_id = self.device_combo.currentData()
        if not device_id:
            return None
        for d in self._devices:
            if d.id == device_id:
                return d
        return None

    def _update_device_caps(self) -> None:
        device = self._current_device()
        if device is None:
            self.scan_btn.setEnabled(False)
            self.dpi_combo.setEnabled(False)
            self.depth_btn.setEnabled(False)
            self.depth_widget.setVisible(False)
            self.ir_btn.setVisible(False)
            self.mode_widget.setVisible(False)
            self.mode_hint.setVisible(False)
            self.passes_rail.setVisible(False)
            self.ir_btn.setEnabled(False)
            self.mode_btn.setEnabled(False)
            self.eject_btn.setVisible(False)
            self.frame_spec_widget.setVisible(False)
            self.eject_after_act.setVisible(False)
            self.scan_window_widget.setVisible(False)
            self.scan_window_status.setVisible(False)
            self._set_exposure_lock_visible(False)
            self.exposure_slider.setVisible(False)
            self.autofocus_btn.setVisible(False)
            self.ae_btn.setVisible(False)
            self.prescan_widget.setVisible(False)
            self.crop_label.setVisible(False)
            self.dust_widget.setVisible(False)
            self.superfine_btn.setVisible(False)
            self.samples_widget.setVisible(False)
            self.film_format_widget.setVisible(False)
            self.film_type_widget.setVisible(False)
            self._caps_autofocus = False
            self._caps_auto_exposure = False
            self._caps_clean = False
            self._caps_superfine = False
            self._caps_max_samples = 1
            self._caps_film_formats = ()
            self._caps_film_types = ()
            self._device_ir = False
            self._sync_bodies()
            self._update_summary()
            return

        caps = device.capabilities
        self.dpi_combo.setEnabled(True)
        self.depth_btn.setEnabled(True)
        self.ir_btn.setEnabled(True)
        self.mode_btn.setEnabled(True)
        self.eject_btn.setVisible(caps.can_eject)
        self.eject_btn.setEnabled(caps.can_eject and not self._scanning)

        # If no film sources, show banner
        if not caps.sources:
            self.status_strip.set_message(tr("This scanner reports no film/transparency sources. NegPy v1 supports film scanning only."))
            self.scan_btn.setEnabled(False)
        else:
            self.status_strip.set_message("")
            self.scan_btn.setEnabled(True)

        self._populate_form(caps)
        self._sync_bodies()
        self._update_summary()

    def _populate_form(self, caps: ScannerCapabilities) -> None:
        self.dpi_combo.blockSignals(True)
        self.depth_btn.blockSignals(True)
        self.ir_btn.blockSignals(True)
        self.mode_btn.blockSignals(True)
        self.passes_slider.blockSignals(True)
        self.ae_btn.blockSignals(True)
        self.frame_spec_edit.blockSignals(True)

        # DPI
        self.dpi_combo.clear()
        if caps.supported_dpi:
            for d in caps.supported_dpi:
                self.dpi_combo.addItem(str(d), d)
        if self._settings.dpi and caps.supported_dpi:
            idx = self.dpi_combo.findData(self._settings.dpi)
            if idx < 0:
                # A saved resolution this device does not offer: show the nearest one it does,
                # rather than a number the scan silently replaces.
                nearest = min(caps.supported_dpi, key=lambda d: abs(d - self._settings.dpi))
                idx = self.dpi_combo.findData(nearest)
            if idx >= 0:
                self.dpi_combo.setCurrentIndex(idx)
        # The combo is editable because the stops are a ladder, not the device's limits. A
        # validator keeps a typo off the scan: unparsable text silently fell back to a default.
        editor = self.dpi_combo.lineEdit()
        if editor is not None and caps.supported_dpi:
            editor.setValidator(QIntValidator(min(caps.supported_dpi), max(caps.supported_dpi), self.dpi_combo))

        # Depth, shown only when the device offers more than one bit depth. Default to the
        # deepest supported when the saved value is absent: a saved 16 does not exist on a
        # 14-bit LS-50, and findData returning -1 must not leave the combo on index 0 (8-bit).
        self.depth_btn.set_choices([f"{d}-bit" for d in caps.supported_depths], list(caps.supported_depths))
        if caps.supported_depths:
            idx = self.depth_btn.findData(self._settings.depth) if self._settings.depth else -1
            if idx < 0:
                idx = self.depth_btn.findData(max(caps.supported_depths))
            if idx >= 0:
                self.depth_btn.setCurrentIndex(idx)
        self.depth_widget.setVisible(len(caps.supported_depths) > 1)

        # IR
        self._device_ir = bool(caps.ir_channel)
        # A transport with ICE offers IR inside the Off / IR / ICE choice instead.
        self._caps_clean = bool(caps.hw_clean)
        self.ir_btn.setVisible(self._device_ir and not self._caps_clean)
        self.ir_btn.setEnabled(caps.ir_channel)
        if caps.ir_channel:
            self.ir_btn.setChecked(self._settings.capture_ir)
            self.ir_btn.setToolTip(tr("Scan a separate infrared channel for dust detection"))
        else:
            self.ir_btn.setChecked(False)
            self.ir_btn.setToolTip(tr("IR scanning not supported by this device"))

        # Scan mode (Plustek GL128 scan-ready models): one combo presenting pyopticfilm's two
        # real orthogonal axes (multi_exposure_mode, n_passes) as 4 named options. Per-item
        # capability gating below; the whole row hides when neither ME nor Multi-Pass applies —
        # a device with neither has nothing to choose, mode is implicitly Single-Pass.
        show_mode = bool(caps.multi_exposure) or caps.max_n_passes > 1
        self.mode_widget.setVisible(show_mode)
        self.mode_btn.setEnabled(show_mode)
        self._wanted_capture_mode = _capture_mode_from_params(
            MultiExposureMode(self._settings.multi_exposure_mode)
            if self._settings.multi_exposure_mode in set(MultiExposureMode)
            else MultiExposureMode.OFF,
            self._settings.n_passes,
        )
        self._sync_capture_modes()
        # IR and Multi-Pass stacking cannot combine (see _on_ir_toggled) — a settings blob
        # saved with both set (signals are blocked through this whole method, so the toggle
        # handlers that normally resolve this never fire) must self-heal here the same way,
        # rather than reaching Scan and failing there.
        if self._capture_mode() in _STACKING_CAPTURE_MODES:
            self._drop_ir()
        ceiling = max(MIN_PASSES_UI, caps.max_n_passes)
        starting_passes = DEFAULT_N_PASSES if self._settings.n_passes <= MIN_N_PASSES else self._settings.n_passes
        self.passes_slider.set_range(MIN_PASSES_UI, ceiling)
        self.passes_slider.setValue(min(max(starting_passes, MIN_PASSES_UI), ceiling))
        self._sync_passes_visibility()

        # Autofocus and auto-exposure, shown only when the device reports them.
        self._caps_autofocus = bool(caps.autofocus)
        self._caps_auto_exposure = bool(caps.auto_exposure)
        self.autofocus_btn.blockSignals(True)
        self.autofocus_btn.setVisible(self._caps_autofocus)
        if self._caps_autofocus:
            self.autofocus_btn.setChecked(self._settings.autofocus)
            self.autofocus_btn.setToolTip(tr("Autofocus before scanning (film is rarely perfectly flat)"))
        else:
            self.autofocus_btn.setChecked(False)
        self.autofocus_btn.blockSignals(False)

        self.ae_btn.setVisible(self._caps_auto_exposure)
        if self._caps_auto_exposure:
            self.ae_btn.setChecked(self._settings.auto_exposure)
            self.ae_btn.setToolTip(tr("Meter exposure in hardware before the scan"))
        else:
            self.ae_btn.setChecked(False)
            self.ae_btn.setToolTip(tr("Auto-exposure not supported by this device"))

        # Dust removal, multi-sample and superfine: only where the transport does them itself.
        self._caps_superfine = bool(caps.superfine)
        self.dust_btn.blockSignals(True)
        self.dust_widget.setVisible(self._caps_clean)
        saved = "ice" if self._settings.clean else "ir" if self._settings.capture_ir and self._device_ir else "off"
        self.dust_btn.setCurrentIndex(self.dust_btn.findData(saved if self._caps_clean else "off"))
        self.dust_btn.blockSignals(False)

        self.superfine_btn.blockSignals(True)
        self.superfine_btn.setVisible(self._caps_superfine)
        self.superfine_btn.setChecked(self._caps_superfine and self._settings.superfine)
        self.superfine_btn.blockSignals(False)

        self._caps_max_samples = int(caps.max_samples)
        self.samples_btn.blockSignals(True)
        show_samples = caps.max_samples > 1
        counts = tuple(count for count in _SAMPLE_COUNTS if show_samples and count <= caps.max_samples)
        self.samples_btn.set_choices([str(count) for count in counts], list(counts))
        self.samples_btn.setCurrentIndex(max(self.samples_btn.findData(self._settings.samples), 0))
        self.samples_widget.setVisible(show_samples)
        self.samples_btn.blockSignals(False)

        self._caps_film_types = tuple(caps.film_types)
        self.film_type_btn.blockSignals(True)
        self.film_type_btn.set_choices([FILM_TYPES[t][0] for t in self._caps_film_types], list(self._caps_film_types))
        self.film_type_btn.setCurrentIndex(max(self.film_type_btn.findData(self._settings.film_type), 0))
        self.film_type_widget.setVisible(bool(self._caps_film_types))
        self.film_type_btn.blockSignals(False)

        # Last: it gates IR and ICE on the film, so it needs both the ICE capability and the
        # film list this device offers, which are read further up.
        self._apply_film_type_to_ir()

        self._caps_film_formats = tuple(caps.film_formats)
        self.format_combo.blockSignals(True)
        self.format_combo.clear()
        show_formats = bool(caps.film_formats)
        if show_formats:
            self.format_combo.addItem(tr("Auto"), None)
            for film_format in caps.film_formats:
                self.format_combo.addItem(film_format, film_format)
            idx = self.format_combo.findData(self._settings.film_format)
            self.format_combo.setCurrentIndex(max(idx, 0))
        self.film_format_widget.setVisible(show_formats)
        self.format_combo.blockSignals(False)

        # Scan exposure time, shown only when the device reports a usable range.
        self.exposure_slider.blockSignals(True)
        et_range = caps.exposure_time_us
        if et_range is not None:
            lo_us, hi_us = et_range
            self.exposure_slider.set_range(int(lo_us), int(hi_us))
            current = self._settings.exposure_time_us
            if current is None or current < lo_us or current > hi_us:
                current = lo_us
            self.exposure_slider.setValue(int(current))
            self.exposure_slider.setVisible(True)
        else:
            self.exposure_slider.set_range(0, 1)
            self.exposure_slider.setValue(0)
            self.exposure_slider.setVisible(False)
        self.exposure_slider.blockSignals(False)

        # Which frames to scan: every transport that reaches more than one frame, whether it
        # counts slots or measures them off the film.
        is_strip = _reaches_a_strip(caps)
        self.frame_spec_widget.setVisible(is_strip)
        self.eject_after_act.setVisible(is_strip and caps.can_eject)
        if is_strip:
            self._sync_frame_spec()

        # Scan window: crop UI for every backend but pyOpticfilm, which uses Prescan instead;
        # both wrote the same scan_window setting.
        use_window = self._current_backend_id() != "plustek"
        self.scan_window_widget.setVisible(use_window)
        self.scan_window_status.setVisible(use_window)
        if use_window:
            self.scan_window_row_label.setText(tr("Batch") if is_strip else tr("Window"))
            if is_strip:
                self.scan_window_btn.setText(icon_text(tr("Preview Strip…")))
                self.scan_window_btn.setToolTip(tr("Preview each frame, set a window per frame, and pick which frames to scan"))
            else:
                self.scan_window_btn.setText(icon_text(tr("Preview…")))
                self.scan_window_btn.setToolTip(tr("Preview the current holder position and set a crop window for the scan"))
            self._update_scan_window_status()

        self._set_exposure_lock_visible(caps.exposure_lock)
        if caps.exposure_lock:
            self._update_exposure_lock_status()

        show_prescan = bool(caps.prescan)
        self.prescan_widget.setVisible(show_prescan)
        if show_prescan:
            self._update_crop_label()
        else:
            self.crop_label.setVisible(False)

        self.dpi_combo.blockSignals(False)
        self.depth_btn.blockSignals(False)
        self.ir_btn.blockSignals(False)
        self.mode_btn.blockSignals(False)
        self.passes_slider.blockSignals(False)
        self.ae_btn.blockSignals(False)
        self.frame_spec_edit.blockSignals(False)

    def _film_type(self) -> str:
        default = FilmType.NEGATIVE.value
        return str(self.film_type_btn.currentData() or default) if self._caps_film_types else default

    def _on_film_type_changed(self) -> None:
        self._apply_film_type_to_ir()
        self._update_settings_from_ui()

    def _apply_film_type_to_ir(self) -> None:
        """Silver grain and Kodachrome's dyes stop infrared, so its mask comes back as the
        picture rather than the dust on it. Both controls go with the film, not the scanner."""
        passes = film_passes_infrared(self._film_type())
        self.ir_btn.setEnabled(self._device_ir and passes)
        self.dust_btn.setEnabled(self._caps_clean and passes)
        if not passes:
            self._drop_ir()
            self.dust_btn.blockSignals(True)
            self.dust_btn.setCurrentIndex(self.dust_btn.findData("off"))
            self.dust_btn.blockSignals(False)
        if not passes and (self._device_ir or self._caps_clean):
            reason = tr("{value} blocks infrared", value=FILM_TYPES[self._film_type()][0])
            self.ir_btn.setToolTip(reason)
            self.dust_btn.setToolTip(reason)

    def _ir_on(self) -> bool:
        if self._caps_clean:
            return self.dust_btn.isEnabled() and self.dust_btn.currentData() == "ir"
        return self.ir_btn.isEnabled() and self.ir_btn.isChecked()

    def _ice_on(self) -> bool:
        return self._caps_clean and self.dust_btn.currentData() == "ice"

    def _drop_ir(self) -> None:
        self.ir_btn.blockSignals(True)
        self.ir_btn.setChecked(False)
        self.ir_btn.blockSignals(False)
        if self.dust_btn.currentData() == "ir":
            self.dust_btn.blockSignals(True)
            self.dust_btn.setCurrentIndex(self.dust_btn.findData("off"))
            self.dust_btn.blockSignals(False)

    def _on_ir_toggled(self, checked: bool) -> None:
        self._update_settings_from_ui()
        if not checked:
            return
        # IR and Multi-Pass stacking cannot combine yet — pyopticfilm rejects the combination
        # outright (each repeat is its own motor cycle; IR stacking is unvalidated). Drop the
        # mode to its non-stacking equivalent rather than silently discarding a hidden slider
        # value, so the visible mode reflects what actually happens.
        self._wanted_capture_mode = _valid_capture_mode(self._wanted_capture_mode, has_me=True, has_stack=False)
        mode = self._capture_mode()
        if mode == ScanCaptureMode.MULTI_PASS:
            self._set_capture_mode(ScanCaptureMode.SINGLE_PASS)
        elif mode == ScanCaptureMode.ADAPTIVE_MULTI_PASS:
            self._set_capture_mode(ScanCaptureMode.ADAPTIVE_ME)

    def _on_dpi_changed(self) -> None:
        self._sync_capture_modes()
        self._update_settings_from_ui()

    def _sync_capture_modes(self) -> None:
        """Enable the modes the device runs at the current resolution and show the wanted mode,
        masked to those."""
        device = self._current_device()
        if device is None:
            return
        caps = device.capabilities
        dpi = self._dpi()
        has_me = caps.multi_exposure_at(dpi)
        has_stack = caps.multi_pass_at(dpi)
        enabled = {
            ScanCaptureMode.SINGLE_PASS: True,
            ScanCaptureMode.MULTI_PASS: has_stack,
            ScanCaptureMode.ADAPTIVE_ME: has_me,
            ScanCaptureMode.ADAPTIVE_MULTI_PASS: has_me and has_stack,
        }
        for capture_mode, _label, _tooltip in _CAPTURE_MODE_LABELS:
            self.mode_btn.set_item_enabled(self.mode_btn.findData(capture_mode.value), enabled[capture_mode])
        blocked = self.mode_btn.blockSignals(True)
        self._set_capture_mode(_valid_capture_mode(self._wanted_capture_mode, has_me=has_me, has_stack=has_stack))
        self.mode_btn.blockSignals(blocked)
        limited = (caps.multi_exposure and not has_me) or (caps.max_n_passes > 1 and not has_stack)
        self.mode_hint.setText(tr("Multi-Exposure and Multi-Pass are not available at {dpi} dpi.", dpi=dpi))
        self.mode_hint.setVisible(limited)
        self._sync_passes_visibility()

    def _on_mode_changed(self) -> None:
        self._wanted_capture_mode = self._capture_mode()
        self._sync_passes_visibility()
        if self._capture_mode() in _STACKING_CAPTURE_MODES:
            self._drop_ir()
        self._update_settings_from_ui()

    def _sync_passes_visibility(self) -> None:
        self.passes_rail.setVisible(self.mode_btn.isEnabled() and self._capture_mode() in _STACKING_CAPTURE_MODES)

    def _set_capture_mode(self, mode: ScanCaptureMode) -> None:
        idx = self.mode_btn.findData(mode.value)
        if idx >= 0:
            self.mode_btn.setCurrentIndex(idx)

    def _capture_mode(self) -> ScanCaptureMode:
        if not self.mode_btn.isEnabled():
            return ScanCaptureMode.SINGLE_PASS
        return ScanCaptureMode(self.mode_btn.currentData() or ScanCaptureMode.SINGLE_PASS.value)

    def _passes_slider_value(self) -> int:
        return int(self.passes_slider.value())

    def _samples(self) -> int:
        if self._caps_max_samples <= 1:
            return 1
        return int(self.samples_btn.currentData() or 1)

    def _film_format(self) -> str | None:
        return self.format_combo.currentData() if self._caps_film_formats else None

    def _on_ae_toggled(self) -> None:
        self.exposure_slider.setEnabled(not self.ae_btn.isChecked())
        self._update_settings_from_ui()

    def _frame_spec(self) -> tuple[int, ...] | None:
        """The typed frame selection, or None where the text cannot be read or names a
        frame past the holder's last slot."""
        from negpy.infrastructure.scanners.settings import parse_frame_spec

        try:
            spec = parse_frame_spec(self.frame_spec_edit.text())
        except ValueError:
            return None
        capacity = self._slot_capacity()
        if capacity is not None and any(f > capacity for f in spec):
            return None
        return spec

    def _slot_capacity(self) -> int | None:
        """The holder's slot count, where frame numbers past it cannot be reached."""
        device = self._current_device()
        if device is None or device.capabilities.roll_discovery:
            return None
        return device.capabilities.adapter_frame_capacity

    def _sync_frame_spec(self) -> None:
        """Write the stored selection into the box, which the strip dialog also sets."""
        from negpy.infrastructure.scanners.settings import format_frame_spec

        text = format_frame_spec(self._settings.selected_frames)
        if text != self.frame_spec_edit.text():
            self.frame_spec_edit.setText(text)
        self._update_summary()

    def _on_frame_spec_typed(self, _text: str) -> None:
        # Only the summary follows every keystroke; the selection is stored on editingFinished.
        self._update_summary()

    def _on_set_scan_window(self) -> None:
        from dataclasses import replace

        device = self._current_device()
        if device is None:
            return

        if _reaches_a_strip(device.capabilities):
            from negpy.desktop.view.widgets.strip_preview_dialog import StripPreviewDialog

            dialog = StripPreviewDialog(
                self.controller,
                device,
                initial_windows=self._settings.frame_windows,
                initial_selected=self._settings.selected_frames,
                initial_offset=self._settings.frame_offset_mm,
                initial_offset_modifier=self._settings.frame_offset_modifier_mm,
                initial_frame_offsets=self._settings.frame_offsets,
                initial_tile_height=self._settings.strip_tile_height,
                film_format=self._film_format(),
                film_type=self._film_type(),
                parent=self,
                repo=self.controller.session.repo,
            )
            if dialog.exec():
                self.settings = replace(
                    self._settings,
                    frame_windows=dialog.frame_windows(),
                    selected_frames=dialog.selected_frames(),
                    frame_offset_mm=dialog.frame_offset(),
                    frame_offset_modifier_mm=dialog.frame_offset_modifier(),
                    frame_offsets=dialog.frame_offsets(),
                    strip_tile_height=dialog.tile_height(),
                )
                self._update_scan_window_status()
                if dialog.scan_requested():
                    self._on_scan()
            return

        from negpy.desktop.view.widgets.quick_scan_preview_dialog import QuickScanPreviewDialog

        dialog = QuickScanPreviewDialog(
            self.controller,
            device,
            initial_window=self._settings.scan_window,
            film_type=self._film_type(),
            parent=self,
            repo=self.controller.session.repo,
        )
        if dialog.exec():
            self.settings = replace(self._settings, scan_window=dialog.window())
            self._update_scan_window_status()
            if dialog.scan_requested():
                self._on_scan()

    def _on_clear_scan_window(self) -> None:
        from dataclasses import replace

        self.settings = replace(self._settings, scan_window=None, frame_windows={}, selected_frames=())
        self._update_scan_window_status()

    def _on_prescan(self) -> None:
        from dataclasses import replace

        from negpy.desktop.view.widgets.prescan_dialog import PrescanCropDialog

        device = self._current_device()
        if device is None or not device.capabilities.prescan:
            return
        dialog = PrescanCropDialog(
            self.controller,
            device,
            initial_window=self._settings.scan_window,
            parent=self,
            repo=self.controller.session.repo,
        )
        if dialog.exec():
            self.settings = replace(self._settings, scan_window=dialog.scan_window())
            self._update_crop_label()
            self._save_settings()
            if dialog.scan_requested():
                self._on_scan()

    def _on_clear_prescan_crop(self) -> None:
        from dataclasses import replace

        self.settings = replace(self._settings, scan_window=None)
        self._update_crop_label()
        self._save_settings()

    def _update_crop_label(self) -> None:
        """Next to Frame info, and only shown when there is an actual crop to report —
        a full-window scan says nothing here rather than stating the obvious."""
        from negpy.infrastructure.scanners.params import scan_window_to_area

        device = self._current_device()
        area = (
            scan_window_to_area(self._settings.scan_window, device.capabilities.max_area_mm)
            if device and self._settings.scan_window
            else None
        )
        if area is None:
            self.crop_label.setVisible(False)
        else:
            tl_x, tl_y, br_x, br_y = area
            self.crop_label.setText(tr("Crop: {value:.1f} × {value1:.1f} mm", value=br_x - tl_x, value1=br_y - tl_y))
            self.crop_label.setVisible(True)

    def _set_exposure_lock_visible(self, visible: bool) -> None:
        self.exposure_lock_widget.setVisible(visible)
        self.exposure_lock_status.setVisible(visible)

    def _locked_exposures(self, device: ScannerDevice | None) -> dict[str, int] | None:
        """The lock, where this device offers one and metered it: exposures belong to one unit."""
        if device is None or not device.capabilities.exposure_lock or self._settings.exposure_lock_device != device.id:
            return None
        return self._settings.exposure_lock

    def _update_exposure_lock_status(self) -> None:
        from datetime import datetime

        locked = self._locked_exposures(self._current_device()) is not None
        self.exposure_unlock_btn.setEnabled(locked and not self._scanning)
        set_hint_kind(self.exposure_lock_status, "warning" if locked else "muted")
        self._update_summary()
        if not locked:
            self.exposure_lock_status.setText(tr("Each frame is metered on its own. Meter a mid-strip frame to lock the roll."))
            return
        try:
            when = f", {datetime.fromisoformat(self._settings.exposure_lock_at):%d %b %H:%M}"
        except ValueError:
            when = ""
        self.exposure_lock_status.setText(
            tr(
                "Locked from frame {exposure_lock_frame}{when}. Every scan reuses it.",
                exposure_lock_frame=self._settings.exposure_lock_frame,
                when=when,
            )
        )

    def _on_meter_frame(self) -> None:
        from negpy.desktop.workers.scan_worker import MeterRequest
        from negpy.infrastructure.scanners.params import ScanParams

        device = self._current_device()
        if device is None or not device.capabilities.exposure_lock:
            return
        frame, ok = QInputDialog.getInt(self, tr("Exposure Lock"), tr("Meter frame:"), self._settings.exposure_lock_frame or 2, 1, 99)
        if not ok:
            return
        params = ScanParams(
            dpi=self._dpi(),
            depth=16,
            capture_ir=False,
            frame=frame,
            frame_offset_mm=self._settings.frame_offset_mm,
            film_format=self._film_format(),
            film_type=self._film_type(),
        )
        try:
            self.controller.start_meter(
                MeterRequest(
                    device_id=device.id,
                    params=params,
                    frame_offset_modifier_mm=self._settings.frame_offset_modifier_mm,
                    frame_offsets=self._settings.frame_offsets,
                )
            )
        except RuntimeError as e:
            self.status_strip.set_message(tr("Scanner busy: {e}", e=e))
            return
        self.set_scanning(True)

    @pyqtSlot(object, int)
    def _on_exposure_metered(self, exposures: dict, frame: int) -> None:
        from dataclasses import replace
        from datetime import datetime

        device = self._current_device()
        self.settings = replace(
            self._settings,
            exposure_lock=dict(exposures),
            exposure_lock_device=device.id if device else "",
            exposure_lock_frame=frame,
            exposure_lock_at=datetime.now().isoformat(timespec="seconds"),
        )
        self._save_settings()
        self.set_scanning(False)
        self.status_strip.set_message(tr("Exposure locked from frame {frame}", frame=frame))

    @pyqtSlot(str)
    def _on_meter_error(self, msg: str) -> None:
        self.set_scanning(False)
        self.status_strip.set_message(tr("Metering failed: {value}", value=msg or "unknown error"))

    def _on_unlock_exposure(self) -> None:
        from dataclasses import replace

        self.settings = replace(self._settings, exposure_lock=None, exposure_lock_device="", exposure_lock_frame=0, exposure_lock_at="")
        self._save_settings()
        self._update_exposure_lock_status()

    def _update_scan_window_status(self) -> None:
        from negpy.infrastructure.scanners.params import scan_window_to_area

        offset = self._settings.frame_offset_mm
        offset_txt = f"  ·  offset {offset:.1f} mm" if offset else ""
        drift = self._settings.frame_offset_modifier_mm
        offset_txt += f"  ·  drift {drift:+.2f} mm/frame" if drift else ""
        device = self._current_device()
        if device is not None and _reaches_a_strip(device.capabilities):
            n_windows = len(self._settings.frame_windows)
            win_txt = count_of(n_windows, "window") if n_windows else tr("Full frame")
            self.scan_window_status.setText(tr("{value}{value1}", value=win_txt, value1=offset_txt))
            return
        area = scan_window_to_area(self._settings.scan_window, device.capabilities.max_area_mm) if device else None
        if area is None:
            self.scan_window_status.setText(tr("Full frame{offset_txt}", offset_txt=offset_txt))
        else:
            tl_x, tl_y, br_x, br_y = area
            self.scan_window_status.setText(
                tr("{value:.1f} × {value1:.1f} mm{offset_txt}", value=br_x - tl_x, value1=br_y - tl_y, offset_txt=offset_txt)
            )

    def _sync_bodies(self) -> None:
        self.framing_body.setVisible(
            any(not w.isHidden() for w in (self.frame_spec_widget, self.scan_window_widget, self.exposure_lock_widget, self.prescan_widget))
        )
        self.cards_changed.emit()

    def _dpi(self) -> int:
        """The resolution the next scan runs at: the value in the box, typed or picked,
        else the finest the device offers."""
        device = self._current_device()
        supported = device.capabilities.supported_dpi if device else ()
        fallback = max(supported) if supported else 3600
        try:
            return int(self.dpi_combo.currentText() or self.dpi_combo.currentData())
        except (ValueError, TypeError):
            return fallback

    def _update_summary(self) -> None:
        """One line saying what pressing Scan will do: how much film, at what resolution,
        with which passes, and how much disk it takes."""
        from dataclasses import replace

        from negpy.infrastructure.scanners.settings import resolve_batch_selection

        device = self._current_device()
        if device is None:
            self.status_strip.set_summary("")
            return
        caps = device.capabilities
        spec = self._frame_spec()
        if spec is None:
            capacity = self._slot_capacity()
            self.status_strip.set_summary(
                tr("Frames: the holder has {value}", value=count_of(capacity, "slot")) if capacity else tr("Frames: cannot read that")
            )
            self.scan_btn.setEnabled(False)
            return
        if not self._scanning:
            self.scan_btn.setEnabled(True)
        frames, windows, base_window = resolve_batch_selection(
            replace(self._settings, selected_frames=spec),
            capacity=caps.adapter_frame_capacity,
            whole_strip=caps.roll_discovery,
        )
        if not _reaches_a_strip(caps):
            frames = (1,)
            base_window = self._settings.scan_window
        dpi = self._dpi()
        depth = int(self.depth_btn.currentData() or 16)
        capture_ir = self._ir_on()
        # An unmeasured strip has no frame count yet, so the size is quoted per frame.
        per_frame = [
            estimated_frame_bytes(caps, dpi, depth, capture_ir=capture_ir, window=windows.get(f, base_window)) for f in frames
        ] or [estimated_frame_bytes(caps, dpi, depth, capture_ir=capture_ir, window=base_window)]
        size = f"~{human_bytes(sum(per_frame))}" if frames else f"~{human_bytes(per_frame[0])}/frame"
        passes = [name for name, on in ((tr("IR"), capture_ir), (tr("ICE"), self._ice_on())) if on]
        if self._caps_superfine and self.superfine_btn.isChecked():
            passes.append(tr("Superfine"))
        if self._samples() > 1:
            passes.append(tr("{count}× sampled", count=self._samples()))
        capture_mode = self._capture_mode()
        if capture_mode != ScanCaptureMode.SINGLE_PASS:
            mode_label = next(label for mode, label, _ in _CAPTURE_MODE_LABELS if mode == capture_mode)
            if capture_mode in _STACKING_CAPTURE_MODES:
                mode_label = f"{mode_label} ({self._passes_slider_value()} passes)"
            passes.append(mode_label)
        # The count and the size are what the operator checks before committing, so they carry
        # primary weight; the rest of the line stays secondary.
        strong = f'<span style="color: {THEME.text_primary}">{{}}</span>'
        parts = [
            strong.format(count_of(len(frames), "frame") if frames else tr("Whole strip")),
            f"{dpi} dpi",
            *passes,
            strong.format(size),
        ]
        if self._locked_exposures(device) is not None:
            parts.append(f'<span style="color: {THEME.warn_amber}">Exposure locked (frame {self._settings.exposure_lock_frame})</span>')
        self.status_strip.set_summary("  ·  ".join(parts))

    def _on_scan(self) -> None:
        if self._scanning:
            # Cancel
            self.controller.cancel_scan()
            return

        # Validate
        device = self._current_device()
        if device is None:
            return

        if not self.output.folder():
            self.output.browse()
            if not self.output.folder():
                return
        folder = self.output.folder()
        if not (os.path.isabs(folder) and os.path.isdir(folder)):
            # The writer's makedirs would resolve it against the working directory.
            self.status_strip.set_message(tr("Output folder does not exist: {folder}", folder=folder))
            return
        output_folder = self.output.target_folder()
        if output_folder is None:
            self.status_strip.set_message(tr('Roll name must be a single safe name (not "." or "..", and no path separators).'))
            return
        as_roll = self.output.as_roll()

        from negpy.desktop.workers.scan_worker import BatchRequest, ScanRequest
        from negpy.infrastructure.scanners.params import ScanParams
        from negpy.infrastructure.scanners.settings import resolve_batch_selection

        dpi = self._dpi()
        depth = int(self.depth_btn.currentData() or 16)
        capture_ir = self._ir_on()
        me_mode, n_passes = _params_from_capture_mode(self._capture_mode(), self._passes_slider_value())
        autofocus = self._caps_autofocus and self.autofocus_btn.isChecked()
        auto_exposure = self._caps_auto_exposure and self.ae_btn.isChecked()
        pattern = self.pattern_edit.text().strip() or '{{ date }}_{{ "%03d" % seq }}'
        fmt = str(self.fmt_btn.currentData())
        # Before the scan, or a finished scan is discarded when its file cannot be named.
        try:
            from negpy.services.scanning.templating import require_sequence_varying_scan_filename

            require_sequence_varying_scan_filename(pattern, "20000101")
        except ValueError:
            self.status_strip.set_message(tr("Filename pattern must include the sequence number ({{ seq }})"))
            return

        if self._frame_spec() is None:
            return
        self._update_settings_from_ui()
        frames, frame_windows, base_window = resolve_batch_selection(
            self._settings,
            capacity=device.capabilities.adapter_frame_capacity,
            whole_strip=device.capabilities.roll_discovery,
        )
        exposure_time_us = (
            self._settings.exposure_time_us if self._settings.exposure_time_us is not None and not self.exposure_slider.isHidden() else None
        )
        base_params = ScanParams(
            dpi=dpi,
            depth=depth,
            capture_ir=capture_ir,
            multi_exposure_mode=me_mode,
            n_passes=n_passes,
            autofocus=autofocus,
            auto_exposure=auto_exposure,
            exposure_time_us=exposure_time_us,
            window=base_window,
            frame_offset_mm=self._settings.frame_offset_mm,
            clean=self._ice_on(),
            samples=self._samples(),
            superfine=self._caps_superfine and self.superfine_btn.isChecked(),
            film_format=self._film_format(),
            film_type=self._film_type(),
            exposures=self._locked_exposures(device),
        )

        self._update_settings_from_ui()
        self._save_settings()
        self.set_scanning(True)

        try:
            if _reaches_a_strip(device.capabilities):
                self.controller.start_batch(
                    BatchRequest(
                        device_id=device.id,
                        params=base_params,
                        output_folder=output_folder,
                        filename_pattern=pattern,
                        output_format=fmt,
                        frames=frames,
                        frame_windows=frame_windows,
                        frame_offset_modifier_mm=self._settings.frame_offset_modifier_mm,
                        frame_offsets=self._settings.frame_offsets,
                        eject_when_done=self._settings.eject_after_batch,
                        as_roll=as_roll,
                    )
                )
            else:
                self.controller.start_scan(
                    ScanRequest(
                        device_id=device.id,
                        params=base_params,
                        output_folder=output_folder,
                        filename_pattern=pattern,
                        output_format=fmt,
                        as_roll=as_roll,
                    )
                )
        except RuntimeError as e:
            self.set_scanning(False)
            self.status_strip.set_message(tr("Scanner busy: {e}", e=e))

    @pyqtSlot(float, str)
    def _on_scan_progress(self, progress: float, phase_name: str = "Scanning") -> None:
        if phase_name in self._INDETERMINATE_SCAN_PHASES:
            self.status_strip.set_progress_indeterminate(f"{phase_name}…")
        else:
            self.status_strip.set_progress(f"{phase_name}… %p%", progress)

    @pyqtSlot(str)
    def _on_scan_finished(self, path: str) -> None:
        self.set_scanning(False)
        self.status_strip.stop_progress()
        self.status_strip.set_message(tr("Scanned: {path}", path=path))

    @pyqtSlot(int, str)
    def _on_scan_frame_done(self, frame: int, path: str) -> None:
        self.status_strip.set_message(tr("Scanned frame {frame}: {path}", frame=frame, path=path))

    @pyqtSlot(list)
    def _on_scan_batch_finished(self, paths: list) -> None:
        self.set_scanning(False)
        self.status_strip.stop_progress()
        if paths:
            self.status_strip.set_message(tr("Batch complete: {value}", value=count_of(len(paths), "frame")))

    @pyqtSlot()
    def _on_scan_cancelled(self) -> None:
        self.set_scanning(False)
        self.status_strip.stop_progress()
        self.status_strip.set_message(tr("Scan stopped"))

    @pyqtSlot(str)
    def _on_scan_error(self, msg: str) -> None:
        self.set_scanning(False)
        self.status_strip.stop_progress()
        text = msg or tr("Unknown scan error")
        self.status_strip.set_message(tr("Error: {text}", text=text))
        # Unsupported pyOpticfilm models: status alone is easy to miss.
        if "cannot scan with pyOpticfilm" in text:
            QMessageBox.warning(self, tr("Scan"), text)

    @pyqtSlot(bool)
    def _on_ejected(self, triggered: bool) -> None:
        device = self._current_device()
        self.eject_btn.setEnabled(bool(device and device.capabilities.can_eject) and not self._scanning)
        if not triggered:
            self.status_strip.set_message(tr("This device has no eject control"))
            return
        self._media_loaded = False
        stale = self._drop_strip_state()
        self.status_strip.set_message(tr("Film ejected. Frame selection cleared.") if stale else tr("Film ejected"))

    @pyqtSlot(bool)
    def _on_strip_returned(self, loaded: bool) -> None:
        self._media_loaded = loaded
        stale = self._drop_strip_state()
        if not loaded:
            message = tr("The scanner returned the strip while idle. Insert it again.")
        elif stale:
            message = tr("The scanner sat idle long enough to return the strip. Frame selection cleared.")
        else:
            message = tr("The scanner sat idle long enough to return the strip")
        self.status_strip.set_message(message)

    def _drop_strip_state(self) -> bool:
        """Forget the frame picks, crops and per-frame offsets; True when there were any.

        Offset and Drift belong to the scanner, not the strip, and stay.
        """
        from dataclasses import replace

        stale = bool(self._settings.selected_frames or self._settings.frame_windows or self._settings.frame_offsets)
        if stale:
            self.settings = replace(self._settings, selected_frames=(), frame_windows={}, frame_offsets={})
            self._update_scan_window_status()
            self._update_summary()
        return stale

    @pyqtSlot(str)
    def _on_eject_error(self, msg: str) -> None:
        device = self._current_device()
        self.eject_btn.setEnabled(bool(device and device.capabilities.can_eject) and not self._scanning)
        self.status_strip.set_message(tr("Eject failed: {msg}", msg=msg))

    def set_scanning(self, active: bool) -> None:
        self._scanning = active
        device = self._current_device()
        for body in (self.device_body, self.quality_body, self.framing_body, self.output_body):
            body.setEnabled(not active)
        self.backend_btn.setEnabled(not active)
        self.eject_btn.setEnabled(bool(device and device.capabilities.can_eject) and not active)
        if active:
            self.scan_btn.setText(icon_text(tr("Stop")))
            self.scan_btn.setIcon(qta.icon("fa5s.stop", color=THEME.text_primary))
            self.status_strip.start_progress(tr("Scanning… %p%"))
            self.prescan_btn.setEnabled(False)
            self.exposure_meter_btn.setEnabled(False)
            self.exposure_unlock_btn.setEnabled(False)
        else:
            self.scan_btn.setText(icon_text(tr("Scan")))
            self.scan_btn.setIcon(qta.icon("fa5s.camera-retro", color=THEME.text_on_accent))
            self.prescan_btn.setEnabled(True)
            self.exposure_meter_btn.setEnabled(True)
            self._update_exposure_lock_status()
            self.status_strip.stop_progress()
        # The filled/hollow swap is a QSS property selector, and Qt only re-reads those on a
        # repolish.
        self.scan_btn.setProperty("scanning", "true" if active else "false")
        style = self.scan_btn.style()
        style.unpolish(self.scan_btn)
        style.polish(self.scan_btn)

    def _update_settings_from_ui(self) -> None:
        # Editing anything means the last pass's message has been read: let the summary back.
        self.status_strip.set_message("")
        dpi = self._dpi()
        try:
            depth = int(self.depth_btn.currentData() or 16)
        except (ValueError, TypeError):
            depth = 16

        from dataclasses import replace

        device = self._current_device()
        me_mode, n_passes = _params_from_capture_mode(self._capture_mode(), self._passes_slider_value())
        # replace(), never a fresh ScannerSettings: fields with no sidebar control must survive
        # UI edits, and reconstruction silently resets any field missing from this list.
        self.settings = replace(
            self._settings,
            last_device_id=device.id if device else self._settings.last_device_id,
            backend=self._current_backend_id(),
            dpi=dpi,
            depth=depth,
            capture_ir=self._ir_on(),
            multi_exposure_mode=me_mode.value,
            n_passes=n_passes,
            autofocus=self._caps_autofocus and self.autofocus_btn.isChecked(),
            auto_exposure=self._caps_auto_exposure and self.ae_btn.isChecked(),
            exposure_time_us=(int(self.exposure_slider.value()) if not self.exposure_slider.isHidden() else None),
            clean=self._ice_on(),
            samples=self._samples(),
            superfine=self._caps_superfine and self.superfine_btn.isChecked(),
            film_format=self._film_format(),
            film_type=self._film_type(),
            selected_frames=(spec if (spec := self._frame_spec()) is not None else self._settings.selected_frames),
            output_format=str(self.fmt_btn.currentData()),
            filename_pattern=self.pattern_edit.text().strip() or '{{ date }}_{{ "%03d" % seq }}',
            eject_after_batch=self.eject_after_act.isChecked(),
        )
        self._update_summary()
