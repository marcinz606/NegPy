import qtawesome as qta
from PyQt6.QtCore import Qt, pyqtSlot
from PyQt6.QtGui import QIntValidator
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from negpy.kernel.system.text import count_of, human_bytes
from negpy.desktop.view.sidebar.base import install_wheel_guards
from negpy.desktop.view.styles.templates import hint_label, section_subheader
from negpy.desktop.view.styles.theme import THEME
from negpy.infrastructure.scanners.base import ScannerCapabilities, ScannerDevice
from negpy.infrastructure.scanners.params import FILM_TYPES, FilmType, film_passes_infrared
from negpy.infrastructure.scanners.registry import DEFAULT_BACKEND_ID, backend_choices
from negpy.infrastructure.scanners.settings import ScannerSettings


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

    def __init__(self, controller) -> None:
        super().__init__()
        self.controller = controller
        self._settings: ScannerSettings = self._load_settings()
        self._devices: list[ScannerDevice] = []
        self._scanning = False
        self._devices_loaded = False
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
        install_wheel_guards(self)

    # ── settings persistence ──────────────────────────────────────────

    def _load_settings(self) -> ScannerSettings:
        from dataclasses import replace

        data = self.controller.session.repo.get_global_setting("scanner_settings", default={})
        if isinstance(data, dict) and data:
            try:
                settings = ScannerSettings(**data)
            except Exception:
                settings = ScannerSettings.defaults()
        else:
            settings = ScannerSettings.defaults()
        # Drop backends that no longer ship on this platform (e.g. saved "sane" on Windows).
        if settings.backend not in {bid for bid, _ in backend_choices()}:
            settings = replace(settings, backend=DEFAULT_BACKEND_ID)
        return settings

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

    # ── UI construction ───────────────────────────────────────────────

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(THEME.space_xl, 0, THEME.space_xl, 5)
        layout.setSpacing(THEME.space_lg)

        # ── DEVICE ───────────────────────────────────────────
        device_form = QFormLayout()
        device_form.setSpacing(6)

        self.backend_combo = QComboBox()
        self.backend_combo.setToolTip("Scanner transport backend")
        for backend_id, backend_label in backend_choices():
            self.backend_combo.addItem(backend_label, backend_id)
        idx = self.backend_combo.findData(self._settings.backend)
        self.backend_combo.setCurrentIndex(idx if idx >= 0 else 0)
        device_form.addRow("Backend", self.backend_combo)

        device_row = QHBoxLayout()
        device_row.setContentsMargins(0, 0, 0, 0)
        self.device_combo = QComboBox()
        self.device_combo.setToolTip("Select scanner")
        self.device_combo.addItem("Detecting scanners…", None)

        self.refresh_btn = QPushButton()
        self.refresh_btn.setIcon(qta.icon("fa5s.redo", color=THEME.text_secondary))
        self.refresh_btn.setToolTip("Refresh device list")
        self.refresh_btn.setFixedWidth(32)

        self.eject_btn = QPushButton()
        self.eject_btn.setIcon(qta.icon("fa5s.eject", color=THEME.text_secondary))
        self.eject_btn.setToolTip("Eject film")
        self.eject_btn.setFixedWidth(32)
        self.eject_btn.setVisible(False)

        device_row.addWidget(self.device_combo, 1)
        device_row.addWidget(self.refresh_btn)
        device_row.addWidget(self.eject_btn)
        device_row_widget = QWidget()
        device_row_widget.setLayout(device_row)
        device_form.addRow("Device", device_row_widget)
        layout.addLayout(device_form)

        # ── CAPS INFO ───────────────────────────────────────
        self.frame_label = hint_label("")
        layout.addWidget(self.frame_label)

        # ── SETTINGS ────────────────────────────────────────
        # Four labelled groups in one form, in the order the operator decides them: what is
        # on the film, how to read it, which frames, where the files go. Every row keeps its
        # own visibility gate; a header hides when its whole group is gated away.
        self.form = QFormLayout()
        self.form.setSpacing(6)

        self.film_header = section_subheader("Film")
        self.form.addRow(self.film_header)

        # What is on the film: it decides which way the frame boundaries read on a strip, and
        # whether an IR pass has anything to see.
        self.film_type_combo = QComboBox()
        self.film_type_combo.setToolTip("What is on the film. Silver and Kodachrome block infrared")
        self.film_type_label = QLabel("Film")
        self.form.addRow(self.film_type_label, self.film_type_combo)
        self.film_type_label.setVisible(False)
        self.film_type_combo.setVisible(False)

        # Frame length, for a transport that measures the strip and cannot infer it.
        self.format_combo = QComboBox()
        self.format_combo.setToolTip("Frame length on the loaded film; Auto where the holder fixes it")
        # Not "Format": the output file format owns that label further down the panel.
        self.film_format_label = QLabel("Film format")
        self.form.addRow(self.film_format_label, self.format_combo)
        self.film_format_label.setVisible(False)
        self.format_combo.setVisible(False)

        self.quality_header = section_subheader("Quality")
        self.form.addRow(self.quality_header)

        self.dpi_combo = QComboBox()
        self.dpi_combo.setToolTip("Resolution (DPI)")
        self.dpi_combo.setEditable(True)
        self.form.addRow("DPI", self.dpi_combo)

        self.depth_combo = QComboBox()
        self.depth_combo.setToolTip("Bit depth")
        self.depth_label = QLabel("Depth")
        self.form.addRow(self.depth_label, self.depth_combo)
        self.depth_combo.setVisible(False)
        self.depth_label.setVisible(False)

        # Spanning rows, like every other toggle: sharing the Depth row left these indented into
        # the field column on a single-depth device, where the combo beside them is hidden.
        self.ir_check = QCheckBox("IR")
        self.ir_check.setToolTip("Scan a separate infrared channel for dust detection")
        self.form.addRow(self.ir_check)

        self.clean_check = QCheckBox("ICE")
        self.clean_check.setToolTip(
            "Remove dust and scratches with the infrared channel while scanning.\nBaked into the file — color film only."
        )
        self.form.addRow(self.clean_check)
        self.clean_check.setVisible(False)

        self.me_check = QCheckBox("Multi-exposure")
        self.me_check.setToolTip("Merge short and long color passes for more highlight and shadow detail. Takes longer.")
        self.form.addRow(self.me_check)

        self.superfine_check = QCheckBox("Superfine")
        self.superfine_check.setToolTip("Read one line per pass: slower, and free of line registration")
        self.form.addRow(self.superfine_check)
        self.superfine_check.setVisible(False)

        # Multi-sample: repeated reads of one line the scanner averages, for shadow noise.
        self.samples_combo = QComboBox()
        self.samples_combo.setToolTip("Reads per line the scanner averages")
        self.samples_label = QLabel("Samples")
        self.form.addRow(self.samples_label, self.samples_combo)
        self.samples_label.setVisible(False)
        self.samples_combo.setVisible(False)

        self.autofocus_check = QCheckBox("Autofocus")
        self.autofocus_check.setChecked(True)
        self.autofocus_check.setToolTip("Autofocus before scanning (film is rarely perfectly flat)")
        self.form.addRow(self.autofocus_check)
        self.autofocus_check.setVisible(False)

        self.ae_check = QCheckBox("Auto-exposure")
        self.ae_check.setToolTip("Meter exposure in hardware before the scan")
        self.form.addRow(self.ae_check)
        self.ae_check.setVisible(False)

        # Scan exposure time (SANE `scan-exposure-time`), shown only when the device reports a
        # usable range. The slider is in microseconds and the label shows a readable value.
        self.exposure_row_widget = QWidget()
        exposure_layout = QHBoxLayout(self.exposure_row_widget)
        exposure_layout.setContentsMargins(0, 0, 0, 0)
        exposure_layout.setSpacing(6)
        self.exposure_slider = QSlider(Qt.Orientation.Horizontal)
        self.exposure_slider.setSingleStep(1)
        self.exposure_slider.setToolTip("Scan exposure time (microseconds)")
        self.exposure_value_label = QLabel()
        self.exposure_value_label.setMinimumWidth(64)
        exposure_layout.addWidget(self.exposure_slider, 1)
        exposure_layout.addWidget(self.exposure_value_label)
        self.exposure_label = QLabel("Exposure")
        self.form.addRow(self.exposure_label, self.exposure_row_widget)
        self.exposure_label.setVisible(False)
        self.exposure_row_widget.setVisible(False)

        self.framing_header = section_subheader("Framing")
        self.form.addRow(self.framing_header)

        # Frame range, for roll and strip feeders only. Shown when a live capacity is known.
        self.frame_range_widget = QWidget()
        frame_row = QHBoxLayout(self.frame_range_widget)
        frame_row.setContentsMargins(0, 0, 0, 0)
        self.frame_from_spin = QSpinBox()
        self.frame_from_spin.setMinimum(1)
        self.frame_from_spin.setToolTip("First frame to scan")
        self.frame_to_spin = QSpinBox()
        self.frame_to_spin.setMinimum(1)
        self.frame_to_spin.setToolTip("Last frame to scan")
        frame_row.addWidget(self.frame_from_spin)
        frame_row.addWidget(QLabel("–"))
        frame_row.addWidget(self.frame_to_spin)
        frame_row.addStretch()
        self.frame_range_label = QLabel("Frames")
        self.form.addRow(self.frame_range_label, self.frame_range_widget)
        self.frame_range_label.setVisible(False)
        self.frame_range_widget.setVisible(False)

        # Scan window (strip/roll feeders): set once from a preview, reused per frame.
        self.scan_window_widget = QWidget()
        scan_window_row = QHBoxLayout(self.scan_window_widget)
        scan_window_row.setContentsMargins(0, 0, 0, 0)
        self.scan_window_btn = QPushButton("Set scan window…")
        self.scan_window_btn.setToolTip("Preview a frame and set the scan window reused for every frame")
        self.scan_window_clear_btn = QPushButton("Clear")
        self.scan_window_clear_btn.setFixedWidth(56)
        self.scan_window_clear_btn.setToolTip("Scan the whole default frame instead")
        scan_window_row.addWidget(self.scan_window_btn, 1)
        scan_window_row.addWidget(self.scan_window_clear_btn)
        self.scan_window_row_label = QLabel("Batch")
        self.form.addRow(self.scan_window_row_label, self.scan_window_widget)
        self.scan_window_status = hint_label("")
        self.form.addRow("", self.scan_window_status)
        self.scan_window_row_label.setVisible(False)
        self.scan_window_widget.setVisible(False)
        self.scan_window_status.setVisible(False)

        # Prescan + crop (Plustek SE): low-DPI full window → interactive crop → scan_window.
        self.prescan_widget = QWidget()
        prescan_row = QHBoxLayout(self.prescan_widget)
        prescan_row.setContentsMargins(0, 0, 0, 0)
        self.prescan_btn = QPushButton("Prescan…")
        self.prescan_btn.setToolTip("Scan a low-DPI preview and set the crop for the next scan")
        self.prescan_clear_btn = QPushButton("Clear")
        self.prescan_clear_btn.setFixedWidth(56)
        self.prescan_clear_btn.setToolTip("Scan the full window instead of a crop")
        prescan_row.addWidget(self.prescan_btn, 1)
        prescan_row.addWidget(self.prescan_clear_btn)
        self.prescan_label = QLabel("Prescan")
        self.form.addRow(self.prescan_label, self.prescan_widget)
        self.prescan_status = hint_label("")
        self.form.addRow("", self.prescan_status)
        self.prescan_label.setVisible(False)
        self.prescan_widget.setVisible(False)
        self.prescan_status.setVisible(False)

        self.output_header = section_subheader("Output")
        self.form.addRow(self.output_header)

        self.fmt_combo = QComboBox()
        self.fmt_combo.addItems(["TIFF", "DNG"])
        self.fmt_combo.setToolTip("Output file format")
        self.form.addRow("Format", self.fmt_combo)

        folder_row = QHBoxLayout()
        self.folder_edit = QLineEdit()
        self.folder_edit.setPlaceholderText("Output folder…")
        self.folder_edit.setToolTip("Directory for scanned files")
        self.browse_btn = QPushButton("…")
        self.browse_btn.setFixedWidth(32)
        self.browse_btn.setToolTip("Browse for output folder")
        folder_row.addWidget(self.folder_edit)
        folder_row.addWidget(self.browse_btn)
        self.form.addRow("Folder", folder_row)

        self.pattern_edit = QLineEdit()
        self.pattern_edit.setToolTip('Jinja2 template. Variables: {{ date }}, {{ seq }}.\nExample: {{ date }}_{{ "%03d" % seq }}')
        self.form.addRow("Filename", self.pattern_edit)

        layout.addLayout(self.form)

        # ── PROGRESS ────────────────────────────────────────
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Scanning… %p%")
        layout.addWidget(self.progress_bar)

        # ── STATUS ──────────────────────────────────────────
        self.status_label = hint_label("")
        layout.addWidget(self.status_label)

        # ── SUMMARY + SCAN BUTTON ───────────────────────────
        self.summary_label = hint_label("")
        layout.addWidget(self.summary_label)

        self.scan_btn = QPushButton(" Scan")
        self.scan_btn.setObjectName("scan_btn")
        self.scan_btn.setFixedHeight(40)
        self.scan_btn.setIcon(qta.icon("fa5s.camera-retro", color=THEME.text_primary))
        layout.addWidget(self.scan_btn)

        layout.addStretch()

        # Pre-fill from persisted settings
        self.fmt_combo.setCurrentText(self._settings.output_format)
        self.folder_edit.setText(self._settings.output_folder)
        self.pattern_edit.setText(self._settings.filename_pattern)
        self.autofocus_check.setChecked(self._settings.autofocus)
        self.ae_check.setChecked(self._settings.auto_exposure)
        self.exposure_slider.setEnabled(not self._settings.auto_exposure)

    def _connect_signals(self) -> None:
        self.refresh_btn.clicked.connect(self._on_refresh)
        self.eject_btn.clicked.connect(self._on_eject)
        self.backend_combo.currentIndexChanged.connect(self._on_backend_changed)
        self.device_combo.currentIndexChanged.connect(self._on_device_changed)
        self.browse_btn.clicked.connect(self._on_browse)
        self.scan_btn.clicked.connect(self._on_scan)
        self.folder_edit.textChanged.connect(lambda: self._update_settings_from_ui())
        self.pattern_edit.textChanged.connect(lambda: self._update_settings_from_ui())
        self.fmt_combo.currentTextChanged.connect(lambda: self._update_settings_from_ui())
        self.dpi_combo.currentTextChanged.connect(lambda: self._update_settings_from_ui())
        self.depth_combo.currentTextChanged.connect(lambda: self._update_settings_from_ui())
        self.ir_check.toggled.connect(lambda: self._update_settings_from_ui())
        self.me_check.toggled.connect(lambda: self._update_settings_from_ui())
        self.autofocus_check.toggled.connect(lambda: self._update_settings_from_ui())
        self.ae_check.toggled.connect(lambda: self._on_ae_toggled())
        self.clean_check.toggled.connect(lambda: self._update_settings_from_ui())
        self.superfine_check.toggled.connect(lambda: self._update_settings_from_ui())
        self.samples_combo.currentIndexChanged.connect(lambda: self._update_settings_from_ui())
        self.format_combo.currentIndexChanged.connect(lambda: self._update_settings_from_ui())
        self.film_type_combo.currentIndexChanged.connect(lambda: self._on_film_type_changed())
        self.exposure_slider.valueChanged.connect(self._on_exposure_changed)
        self.frame_from_spin.valueChanged.connect(self._on_frame_from_changed)
        self.frame_to_spin.valueChanged.connect(self._on_frame_to_changed)
        self.scan_window_btn.clicked.connect(self._on_set_scan_window)
        self.scan_window_clear_btn.clicked.connect(self._on_clear_scan_window)
        self.prescan_btn.clicked.connect(self._on_prescan)
        self.prescan_clear_btn.clicked.connect(self._on_clear_prescan_crop)

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

    # ── activation hook ───────────────────────────────────────────────

    def on_activated(self) -> None:
        """Called when the Scan tab is switched to."""
        if not self._devices_loaded:
            self._request_devices()

    # ── slots ─────────────────────────────────────────────────────────

    def _request_devices(self) -> None:
        """Request device list from the scan worker thread."""
        self.controller.set_scan_backend(self._current_backend_id())
        self.device_combo.clear()
        self.device_combo.addItem("Detecting scanners…", None)
        self.device_combo.setEnabled(False)
        self.status_label.setText("Detecting scanners…")
        self.controller.request_scan_devices()

    def _on_refresh(self) -> None:
        self._request_devices()

    def _current_backend_id(self) -> str:
        return self.backend_combo.currentData() or DEFAULT_BACKEND_ID

    def _on_backend_changed(self, _index: int) -> None:
        # Device lists are backend-specific, so persist the choice and then re-enumerate.
        # Per-backend UI tweaks that capabilities cannot express branch here on
        # _current_backend_id(). None are needed today.
        self._update_settings_from_ui()
        self._request_devices()

    def _on_eject(self) -> None:
        device = self._current_device()
        if device is None:
            return
        self.eject_btn.setEnabled(False)
        self.status_label.setText("Ejecting film…")
        self.controller.eject_scanner(device.id)

    @pyqtSlot(list)
    def _on_devices_ready(self, devices: list) -> None:
        self._devices = devices
        self._devices_loaded = True
        self.device_combo.clear()
        self.device_combo.setEnabled(True)

        if not devices:
            self.device_combo.addItem("No scanners detected", None)
            self.device_combo.setEnabled(False)
            self.status_label.setText("No scanners detected. Plug in your scanner and click Refresh.")
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
        self._update_device_caps()

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
            self.frame_label.setText("")
            self.dpi_combo.setEnabled(False)
            self.depth_combo.setEnabled(False)
            self.depth_combo.setVisible(False)
            self.depth_label.setVisible(False)
            self.ir_check.setVisible(False)
            self.me_check.setVisible(False)
            self.ir_check.setEnabled(False)
            self.me_check.setEnabled(False)
            self.eject_btn.setVisible(False)
            self.frame_range_label.setVisible(False)
            self.frame_range_widget.setVisible(False)
            self.scan_window_row_label.setVisible(False)
            self.scan_window_widget.setVisible(False)
            self.scan_window_status.setVisible(False)
            self.exposure_label.setVisible(False)
            self.exposure_row_widget.setVisible(False)
            self.autofocus_check.setVisible(False)
            self.ae_check.setVisible(False)
            self.prescan_label.setVisible(False)
            self.prescan_widget.setVisible(False)
            self.prescan_status.setVisible(False)
            self.clean_check.setVisible(False)
            self.superfine_check.setVisible(False)
            self.samples_label.setVisible(False)
            self.samples_combo.setVisible(False)
            self.film_format_label.setVisible(False)
            self.format_combo.setVisible(False)
            self.film_type_label.setVisible(False)
            self.film_type_combo.setVisible(False)
            self._caps_autofocus = False
            self._caps_auto_exposure = False
            self._caps_clean = False
            self._caps_superfine = False
            self._caps_max_samples = 1
            self._caps_film_formats = ()
            self._caps_film_types = ()
            self._device_ir = False
            self._sync_group_headers()
            self._update_summary()
            return

        caps = device.capabilities
        self.dpi_combo.setEnabled(True)
        self.depth_combo.setEnabled(True)
        self.ir_check.setEnabled(True)
        self.me_check.setEnabled(True)
        self.eject_btn.setVisible(caps.can_eject)
        self.eject_btn.setEnabled(caps.can_eject and not self._scanning)
        self.frame_label.setText(f"Frame: {caps.max_area_mm[0]:.0f} × {caps.max_area_mm[1]:.0f} mm")

        # If no film sources, show banner
        if not caps.sources:
            self.status_label.setText("This scanner reports no film/transparency sources. NegPy v1 supports film scanning only.")
            self.scan_btn.setEnabled(False)
        else:
            self.status_label.setText("")
            self.scan_btn.setEnabled(True)

        self._populate_form(caps)
        self._sync_group_headers()
        self._update_summary()

    def _populate_form(self, caps: ScannerCapabilities) -> None:
        self.dpi_combo.blockSignals(True)
        self.depth_combo.blockSignals(True)
        self.ir_check.blockSignals(True)
        self.me_check.blockSignals(True)
        self.ae_check.blockSignals(True)
        self.frame_from_spin.blockSignals(True)
        self.frame_to_spin.blockSignals(True)

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
        self.depth_combo.clear()
        if caps.supported_depths:
            for d in caps.supported_depths:
                self.depth_combo.addItem(f"{d}-bit", d)
            idx = self.depth_combo.findData(self._settings.depth) if self._settings.depth else -1
            if idx < 0:
                idx = self.depth_combo.findData(max(caps.supported_depths))
            if idx >= 0:
                self.depth_combo.setCurrentIndex(idx)
        show_depth = len(caps.supported_depths) > 1
        self.depth_combo.setVisible(show_depth)
        self.depth_label.setVisible(show_depth)

        # IR
        self._device_ir = bool(caps.ir_channel)
        self.ir_check.setVisible(self._device_ir)
        self.ir_check.setEnabled(caps.ir_channel)
        if caps.ir_channel:
            self.ir_check.setChecked(self._settings.capture_ir)
            self.ir_check.setToolTip("Scan a separate infrared channel for dust detection")
        else:
            self.ir_check.setChecked(False)
            self.ir_check.setToolTip("IR scanning not supported by this device")
        self._apply_film_type_to_ir()

        # Multi-exposure (Plustek SE only today)
        self.me_check.setVisible(bool(caps.multi_exposure))
        self.me_check.setEnabled(caps.multi_exposure)
        if caps.multi_exposure:
            self.me_check.setChecked(self._settings.multi_exposure)
            self.me_check.setToolTip("Merge short and long color passes for more highlight and shadow detail. Takes longer.")
        else:
            self.me_check.setChecked(False)
            self.me_check.setToolTip("Multi-exposure not supported by this device")

        # Autofocus and auto-exposure, shown only when the device reports them.
        self._caps_autofocus = bool(caps.autofocus)
        self._caps_auto_exposure = bool(caps.auto_exposure)
        self.autofocus_check.blockSignals(True)
        self.autofocus_check.setVisible(self._caps_autofocus)
        if self._caps_autofocus:
            self.autofocus_check.setChecked(self._settings.autofocus)
            self.autofocus_check.setToolTip("Autofocus before scanning (film is rarely perfectly flat)")
        else:
            self.autofocus_check.setChecked(False)
        self.autofocus_check.blockSignals(False)

        self.ae_check.setVisible(self._caps_auto_exposure)
        if self._caps_auto_exposure:
            self.ae_check.setChecked(self._settings.auto_exposure)
            self.ae_check.setToolTip("Meter exposure in hardware before the scan")
        else:
            self.ae_check.setChecked(False)
            self.ae_check.setToolTip("Auto-exposure not supported by this device")

        # Dust removal, multi-sample and superfine: only where the transport does them itself.
        self._caps_clean = bool(caps.hw_clean)
        self._caps_superfine = bool(caps.superfine)
        self.clean_check.blockSignals(True)
        self.clean_check.setVisible(self._caps_clean)
        self.clean_check.setChecked(self._caps_clean and self._settings.clean)
        self.clean_check.blockSignals(False)

        self.superfine_check.blockSignals(True)
        self.superfine_check.setVisible(self._caps_superfine)
        self.superfine_check.setChecked(self._caps_superfine and self._settings.superfine)
        self.superfine_check.blockSignals(False)

        self._caps_max_samples = int(caps.max_samples)
        self.samples_combo.blockSignals(True)
        self.samples_combo.clear()
        show_samples = caps.max_samples > 1
        if show_samples:
            for count in _SAMPLE_COUNTS:
                if count <= caps.max_samples:
                    self.samples_combo.addItem(str(count), count)
            idx = self.samples_combo.findData(self._settings.samples)
            self.samples_combo.setCurrentIndex(max(idx, 0))
        self.samples_label.setVisible(show_samples)
        self.samples_combo.setVisible(show_samples)
        self.samples_combo.blockSignals(False)

        self._caps_film_types = tuple(caps.film_types)
        self.film_type_combo.blockSignals(True)
        self.film_type_combo.clear()
        for film_type in self._caps_film_types:
            self.film_type_combo.addItem(FILM_TYPES[film_type][0], film_type)
        idx = self.film_type_combo.findData(self._settings.film_type)
        self.film_type_combo.setCurrentIndex(max(idx, 0))
        self.film_type_label.setVisible(bool(self._caps_film_types))
        self.film_type_combo.setVisible(bool(self._caps_film_types))
        self.film_type_combo.blockSignals(False)

        self._caps_film_formats = tuple(caps.film_formats)
        self.format_combo.blockSignals(True)
        self.format_combo.clear()
        show_formats = bool(caps.film_formats)
        if show_formats:
            self.format_combo.addItem("Auto", None)
            for film_format in caps.film_formats:
                self.format_combo.addItem(film_format, film_format)
            idx = self.format_combo.findData(self._settings.film_format)
            self.format_combo.setCurrentIndex(max(idx, 0))
        self.film_format_label.setVisible(show_formats)
        self.format_combo.setVisible(show_formats)
        self.format_combo.blockSignals(False)

        # Scan exposure time, shown only when the device reports a usable range.
        self.exposure_slider.blockSignals(True)
        et_range = caps.exposure_time_us
        if et_range is not None:
            lo_us, hi_us = et_range
            self.exposure_slider.setRange(int(lo_us), int(hi_us))
            current = self._settings.exposure_time_us
            if current is None or current < lo_us or current > hi_us:
                current = lo_us
            self.exposure_slider.setValue(int(current))
            self.exposure_label.setVisible(True)
            self.exposure_row_widget.setVisible(True)
        else:
            self.exposure_slider.setRange(0, 1)
            self.exposure_slider.setValue(0)
            self.exposure_label.setVisible(False)
            self.exposure_row_widget.setVisible(False)
        self.exposure_slider.blockSignals(False)
        self._update_exposure_value_label()

        # Frame range, only for a roll or strip feeder reporting a live capacity. A transport
        # that measures the strip has no capacity to range over: its frames come from the
        # strip dialog.
        capacity = caps.adapter_frame_capacity
        has_frames = capacity is not None
        is_strip = _reaches_a_strip(caps)
        self.frame_range_label.setVisible(has_frames)
        self.frame_range_widget.setVisible(has_frames)
        if has_frames:
            self.frame_from_spin.setMaximum(capacity)
            self.frame_to_spin.setMaximum(capacity)
            frm = min(max(self._settings.frame_from, 1), capacity)
            to = min(max(self._settings.frame_to, frm), capacity)
            # A stored (1, 1) is the unset default → offer the whole strip.
            if self._settings.frame_from == 1 and self._settings.frame_to == 1:
                to = capacity
            self.frame_from_spin.setValue(frm)
            self.frame_to_spin.setValue(to)

        # Scan window: crop UI for every backend but pyOpticfilm, which uses Prescan instead;
        # both wrote the same scan_window setting.
        use_window = self._current_backend_id() != "plustek"
        self.scan_window_row_label.setVisible(use_window)
        self.scan_window_widget.setVisible(use_window)
        self.scan_window_status.setVisible(use_window)
        if use_window:
            self.scan_window_row_label.setText("Batch" if is_strip else "Window")
            if is_strip:
                self.scan_window_btn.setText("Preview strip…")
                self.scan_window_btn.setToolTip("Preview each frame, set a window per frame, and pick which frames to scan")
            else:
                self.scan_window_btn.setText("Preview…")
                self.scan_window_btn.setToolTip("Preview the current holder position and set a crop window for the scan")
            self._update_scan_window_status()

        show_prescan = bool(caps.prescan)
        self.prescan_label.setVisible(show_prescan)
        self.prescan_widget.setVisible(show_prescan)
        self.prescan_status.setVisible(show_prescan)
        if show_prescan:
            self._update_prescan_status()

        self.dpi_combo.blockSignals(False)
        self.depth_combo.blockSignals(False)
        self.ir_check.blockSignals(False)
        self.me_check.blockSignals(False)
        self.ae_check.blockSignals(False)
        self.frame_from_spin.blockSignals(False)
        self.frame_to_spin.blockSignals(False)

    def _film_type(self) -> str:
        default = FilmType.NEGATIVE.value
        return str(self.film_type_combo.currentData() or default) if self._caps_film_types else default

    def _on_film_type_changed(self) -> None:
        self._apply_film_type_to_ir()
        self._update_settings_from_ui()

    def _apply_film_type_to_ir(self) -> None:
        """Silver grain and Kodachrome's dyes stop infrared, so its mask comes back as the
        picture rather than the dust on it. Both controls go with the film, not the scanner."""
        passes = film_passes_infrared(self._film_type())
        for control, supported in ((self.ir_check, self._device_ir), (self.clean_check, self._caps_clean)):
            control.blockSignals(True)
            control.setEnabled(supported and passes)
            if not passes:
                control.setChecked(False)
            control.blockSignals(False)
        if not passes and (self._device_ir or self._caps_clean):
            reason = f"{FILM_TYPES[self._film_type()][0]} blocks infrared"
            self.ir_check.setToolTip(reason)
            self.clean_check.setToolTip(reason)

    def _samples(self) -> int:
        if self._caps_max_samples <= 1:
            return 1
        return int(self.samples_combo.currentData() or 1)

    def _film_format(self) -> str | None:
        return self.format_combo.currentData() if self._caps_film_formats else None

    def _on_ae_toggled(self) -> None:
        self.exposure_slider.setEnabled(not self.ae_check.isChecked())
        self._update_settings_from_ui()

    def _on_exposure_changed(self, _value: int) -> None:
        self._update_exposure_value_label()
        self._update_settings_from_ui()

    def _update_exposure_value_label(self) -> None:
        us = self.exposure_slider.value()
        if us >= 1_000_000:
            self.exposure_value_label.setText(f"{us / 1_000_000:.2f} s")
        elif us >= 1_000:
            self.exposure_value_label.setText(f"{us / 1_000:.1f} ms")
        else:
            self.exposure_value_label.setText(f"{us} us")

    def _on_frame_from_changed(self, _value: int) -> None:
        if self.frame_to_spin.value() < self.frame_from_spin.value():
            self.frame_to_spin.setValue(self.frame_from_spin.value())
        self._update_settings_from_ui()

    def _on_frame_to_changed(self, _value: int) -> None:
        if self.frame_from_spin.value() > self.frame_to_spin.value():
            self.frame_from_spin.setValue(self.frame_to_spin.value())
        self._update_settings_from_ui()

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
                film_format=self._film_format(),
                film_type=self._film_type(),
                parent=self,
            )
            if dialog.exec():
                self.settings = replace(
                    self._settings,
                    frame_windows=dialog.frame_windows(),
                    selected_frames=dialog.selected_frames(),
                    frame_offset_mm=dialog.frame_offset(),
                    frame_offset_modifier_mm=dialog.frame_offset_modifier(),
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
        )
        if dialog.exec():
            self.settings = replace(self._settings, scan_window=dialog.scan_window())
            self._update_prescan_status()
            self._save_settings()

    def _on_clear_prescan_crop(self) -> None:
        from dataclasses import replace

        self.settings = replace(self._settings, scan_window=None)
        self._update_prescan_status()
        self._save_settings()

    def _update_prescan_status(self) -> None:
        from negpy.infrastructure.scanners.params import scan_window_to_area

        device = self._current_device()
        area = (
            scan_window_to_area(self._settings.scan_window, device.capabilities.max_area_mm)
            if device and self._settings.scan_window
            else None
        )
        if area is None:
            self.prescan_status.setText("Full window")
        else:
            tl_x, tl_y, br_x, br_y = area
            self.prescan_status.setText(f"Crop {br_x - tl_x:.1f} × {br_y - tl_y:.1f} mm")

    def _update_scan_window_status(self) -> None:
        from negpy.infrastructure.scanners.params import scan_window_to_area

        offset = self._settings.frame_offset_mm
        offset_txt = f"  ·  offset {offset:.1f} mm" if offset else ""
        drift = self._settings.frame_offset_modifier_mm
        offset_txt += f"  ·  drift {drift:+.2f} mm/frame" if drift else ""
        selected = self._settings.selected_frames
        if selected:
            frames_txt = ", ".join(str(f) for f in sorted(selected))
            n_windows = len(self._settings.frame_windows)
            win_txt = f" · {count_of(n_windows, 'window')}" if n_windows else ""
            self.scan_window_status.setText(f"Frames {frames_txt}{win_txt}{offset_txt}")
            return
        device = self._current_device()
        if device is not None and device.capabilities.roll_discovery:
            # Nothing picked yet, and a measured strip has no frame range to fall back on.
            self.scan_window_status.setText("Whole strip — pick frames in the preview")
            return
        area = scan_window_to_area(self._settings.scan_window, device.capabilities.max_area_mm) if device else None
        if area is None:
            self.scan_window_status.setText(f"Full frame{offset_txt}")
        else:
            tl_x, tl_y, br_x, br_y = area
            self.scan_window_status.setText(f"{br_x - tl_x:.1f} × {br_y - tl_y:.1f} mm{offset_txt}")

    def _sync_group_headers(self) -> None:
        """A group header only earns its space when the group has a visible row."""
        self.film_header.setVisible(self.film_type_combo.isVisibleTo(self) or self.format_combo.isVisibleTo(self))
        self.framing_header.setVisible(
            self.frame_range_widget.isVisibleTo(self) or self.scan_window_widget.isVisibleTo(self) or self.prescan_widget.isVisibleTo(self)
        )

    def _dpi(self) -> int:
        """The resolution the next scan runs at: the picked stop, else the typed value,
        else the finest the device offers."""
        device = self._current_device()
        supported = device.capabilities.supported_dpi if device else ()
        fallback = max(supported) if supported else 3600
        try:
            return int(self.dpi_combo.currentData() or self.dpi_combo.currentText())
        except (ValueError, TypeError):
            return fallback

    def _update_summary(self) -> None:
        """One line saying what pressing Scan will do: how much film, at what resolution,
        with which passes, and how much disk it takes."""
        from negpy.infrastructure.scanners.settings import resolve_batch_selection

        device = self._current_device()
        if device is None:
            self.summary_label.setText("")
            return
        caps = device.capabilities
        frames, windows, base_window = resolve_batch_selection(
            self._settings,
            self.frame_from_spin.value(),
            self.frame_to_spin.value(),
            whole_strip=caps.roll_discovery,
        )
        if not _reaches_a_strip(caps):
            frames = (1,)
            base_window = self._settings.scan_window
        dpi = self._dpi()
        depth = int(self.depth_combo.currentData() or 16)
        capture_ir = self.ir_check.isEnabled() and self.ir_check.isChecked()
        # An unmeasured strip has no frame count yet, so the size is quoted per frame.
        per_frame = [
            estimated_frame_bytes(caps, dpi, depth, capture_ir=capture_ir, window=windows.get(f, base_window)) for f in frames
        ] or [estimated_frame_bytes(caps, dpi, depth, capture_ir=capture_ir, window=base_window)]
        size = f"~{human_bytes(sum(per_frame))}" if frames else f"~{human_bytes(per_frame[0])}/frame"
        passes = [name for name, on in (("IR", capture_ir), ("ICE", self._caps_clean and self.clean_check.isChecked())) if on]
        if self._caps_superfine and self.superfine_check.isChecked():
            passes.append("Superfine")
        if self._samples() > 1:
            passes.append(f"{self._samples()}× sampled")
        if self.me_check.isEnabled() and self.me_check.isChecked():
            passes.append("Multi-exposure")
        parts = [count_of(len(frames), "frame") if frames else "Whole strip", f"{dpi} dpi", *passes, size]
        self.summary_label.setText("  ·  ".join(parts))

    def _on_browse(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if folder:
            self.folder_edit.setText(folder)
            self._update_settings_from_ui()

    def _on_scan(self) -> None:
        if self._scanning:
            # Cancel
            self.controller.cancel_scan()
            return

        # Validate
        device = self._current_device()
        if device is None:
            return

        output_folder = self.folder_edit.text().strip()
        if not output_folder:
            self._on_browse()
            output_folder = self.folder_edit.text().strip()
            if not output_folder:
                return

        from negpy.desktop.workers.scan_worker import BatchRequest, ScanRequest
        from negpy.infrastructure.scanners.params import ScanParams
        from negpy.infrastructure.scanners.settings import resolve_batch_selection

        dpi = self._dpi()
        depth = int(self.depth_combo.currentData() or 16)
        capture_ir = self.ir_check.isEnabled() and self.ir_check.isChecked()
        multi_exposure = self.me_check.isEnabled() and self.me_check.isChecked()
        autofocus = self._caps_autofocus and self.autofocus_check.isChecked()
        auto_exposure = self._caps_auto_exposure and self.ae_check.isChecked()
        pattern = self.pattern_edit.text().strip() or '{{ date }}_{{ "%03d" % seq }}'
        fmt = self.fmt_combo.currentText()

        frames, frame_windows, base_window = resolve_batch_selection(
            self._settings,
            self.frame_from_spin.value(),
            self.frame_to_spin.value(),
            whole_strip=device.capabilities.roll_discovery,
        )
        exposure_time_us = (
            self._settings.exposure_time_us
            if self._settings.exposure_time_us is not None and self.exposure_row_widget.isVisible()
            else None
        )
        base_params = ScanParams(
            dpi=dpi,
            depth=depth,
            capture_ir=capture_ir,
            multi_exposure=multi_exposure,
            autofocus=autofocus,
            auto_exposure=auto_exposure,
            exposure_time_us=exposure_time_us,
            window=base_window,
            frame_offset_mm=self._settings.frame_offset_mm,
            clean=self._caps_clean and self.clean_check.isChecked(),
            samples=self._samples(),
            superfine=self._caps_superfine and self.superfine_check.isChecked(),
            film_format=self._film_format(),
            film_type=self._film_type(),
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
                    )
                )
        except RuntimeError as e:
            self.set_scanning(False)
            self.status_label.setText(f"Scanner busy: {e}")

    @pyqtSlot(float, str)
    def _on_scan_progress(self, progress: float, phase_name: str = "Scanning") -> None:
        self.progress_bar.setVisible(True)
        self.progress_bar.setFormat(f"{phase_name}… %p%")
        self.progress_bar.setValue(int(progress * 100))

    @pyqtSlot(str)
    def _on_scan_finished(self, path: str) -> None:
        self.set_scanning(False)
        self.progress_bar.setVisible(False)
        self.status_label.setText(f"Scanned: {path}")

    @pyqtSlot(int, str)
    def _on_scan_frame_done(self, frame: int, path: str) -> None:
        self.status_label.setText(f"Scanned frame {frame}: {path}")

    @pyqtSlot(list)
    def _on_scan_batch_finished(self, paths: list) -> None:
        self.set_scanning(False)
        self.progress_bar.setVisible(False)
        if paths:
            self.status_label.setText(f"Batch complete: {count_of(len(paths), 'frame')}")

    @pyqtSlot()
    def _on_scan_cancelled(self) -> None:
        self.set_scanning(False)
        self.progress_bar.setVisible(False)
        self.status_label.setText("Scan stopped")

    @pyqtSlot(str)
    def _on_scan_error(self, msg: str) -> None:
        self.set_scanning(False)
        self.progress_bar.setVisible(False)
        text = msg or "Unknown scan error"
        self.status_label.setText(f"Error: {text}")
        # Unsupported pyOpticfilm models: status alone is easy to miss.
        if "cannot scan with pyOpticfilm" in text:
            QMessageBox.warning(self, "Scan failed", text)

    @pyqtSlot(bool)
    def _on_ejected(self, triggered: bool) -> None:
        from dataclasses import replace

        device = self._current_device()
        self.eject_btn.setEnabled(bool(device and device.capabilities.can_eject) and not self._scanning)
        if not triggered:
            self.status_label.setText("This device has no eject control")
            return
        # Frames and their crops describe the piece of film that just came out; the next strip
        # is a different one, and silently reusing them scans the wrong frames.
        stale = bool(self._settings.selected_frames or self._settings.frame_windows)
        if stale:
            self.settings = replace(self._settings, selected_frames=(), frame_windows={})
            self._update_scan_window_status()
            self._update_summary()
        self.status_label.setText("Film ejected — frame selection cleared" if stale else "Film ejected")

    @pyqtSlot(str)
    def _on_eject_error(self, msg: str) -> None:
        device = self._current_device()
        self.eject_btn.setEnabled(bool(device and device.capabilities.can_eject) and not self._scanning)
        self.status_label.setText(f"Eject failed: {msg}")

    # ── state helpers ─────────────────────────────────────────────────

    def set_scanning(self, active: bool) -> None:
        self._scanning = active
        device = self._current_device()
        self.backend_combo.setEnabled(not active)
        self.eject_btn.setEnabled(bool(device and device.capabilities.can_eject) and not active)
        if active:
            self.scan_btn.setText(" Stop")
            self.scan_btn.setIcon(qta.icon("fa5s.stop", color=THEME.text_primary))
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            self.prescan_btn.setEnabled(False)
        else:
            self.scan_btn.setText(" Scan")
            self.scan_btn.setIcon(qta.icon("fa5s.camera-retro", color=THEME.text_primary))
            self.prescan_btn.setEnabled(True)

    def _update_settings_from_ui(self) -> None:
        dpi = self._dpi()
        try:
            depth = int(self.depth_combo.currentData() or 16)
        except (ValueError, TypeError):
            depth = 16

        from dataclasses import replace

        device = self._current_device()
        # replace(), never a fresh ScannerSettings: fields with no sidebar control must survive
        # UI edits, and reconstruction silently resets any field missing from this list.
        self.settings = replace(
            self._settings,
            last_device_id=device.id if device else self._settings.last_device_id,
            backend=self._current_backend_id(),
            dpi=dpi,
            depth=depth,
            capture_ir=self.ir_check.isChecked() and self.ir_check.isEnabled(),
            multi_exposure=self.me_check.isChecked() and self.me_check.isEnabled(),
            autofocus=self._caps_autofocus and self.autofocus_check.isChecked(),
            auto_exposure=self._caps_auto_exposure and self.ae_check.isChecked(),
            exposure_time_us=(self.exposure_slider.value() if self.exposure_row_widget.isVisible() else None),
            clean=self._caps_clean and self.clean_check.isChecked(),
            samples=self._samples(),
            superfine=self._caps_superfine and self.superfine_check.isChecked(),
            film_format=self._film_format(),
            film_type=self._film_type(),
            frame_from=self.frame_from_spin.value(),
            frame_to=self.frame_to_spin.value(),
            output_folder=self.folder_edit.text().strip(),
            output_format=self.fmt_combo.currentText(),
            filename_pattern=self.pattern_edit.text().strip() or '{{ date }}_{{ "%03d" % seq }}',
        )
        self._update_summary()
