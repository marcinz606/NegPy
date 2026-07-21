import qtawesome as qta
from PyQt6.QtCore import Qt, pyqtSlot
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from negpy.desktop.view.styles.templates import hint_label, section_subheader
from negpy.desktop.view.styles.theme import THEME
from negpy.infrastructure.scanners.base import ScannerCapabilities, ScannerDevice
from negpy.infrastructure.scanners.settings import ScannerSettings


class ScanSidebar(QWidget):
    """Scanner control panel — replaces the originally planned modal ScanDialog."""

    def __init__(self, controller) -> None:
        super().__init__()
        self.controller = controller
        self._settings: ScannerSettings = self._load_settings()
        self._devices: list[ScannerDevice] = []
        self._scanning = False
        self._devices_loaded = False
        self._init_ui()
        self._connect_signals()

    # ── settings persistence ──────────────────────────────────────────

    def _load_settings(self) -> ScannerSettings:
        data = self.controller.session.repo.get_global_setting("scanner_settings", default={})
        if isinstance(data, dict) and data:
            try:
                return ScannerSettings(**data)
            except Exception:
                pass
        return ScannerSettings.defaults()

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
        layout.addWidget(section_subheader("DEVICE"))

        device_row = QHBoxLayout()
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
        layout.addLayout(device_row)

        # ── CAPS INFO ───────────────────────────────────────
        self.frame_label = hint_label("")
        layout.addWidget(self.frame_label)

        # ── SETTINGS ────────────────────────────────────────
        self.form = QFormLayout()
        self.form.setSpacing(6)

        self.dpi_combo = QComboBox()
        self.dpi_combo.setToolTip("Resolution (DPI)")
        self.dpi_combo.setEditable(True)
        self.form.addRow("DPI", self.dpi_combo)

        self.ir_check = QCheckBox("IR")
        self.ir_check.setToolTip("Scan a separate infrared channel for dust detection")

        depth_row = QHBoxLayout()
        depth_row.setContentsMargins(0, 0, 0, 0)
        self.depth_combo = QComboBox()
        self.depth_combo.setToolTip("Bit depth")
        depth_row.addWidget(self.depth_combo, 1)
        depth_row.addWidget(self.ir_check)
        self.form.addRow("Depth", depth_row)

        # Spanning rows (no label column) so the checkboxes sit at the left edge.
        self.autofocus_check = QCheckBox("Autofocus")
        self.autofocus_check.setChecked(True)
        self.autofocus_check.setToolTip("Autofocus before scanning (film is rarely perfectly flat)")
        self.form.addRow(self.autofocus_check)

        self.ae_check = QCheckBox("Auto-exposure")
        self.ae_check.setToolTip("Meter exposure in hardware before the scan")
        self.form.addRow(self.ae_check)

        # Frame range (roll/strip feeders only — shown when a live capacity is known).
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

        # ── SCAN BUTTON ─────────────────────────────────────
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

    def _connect_signals(self) -> None:
        self.refresh_btn.clicked.connect(self._on_refresh)
        self.eject_btn.clicked.connect(self._on_eject)
        self.device_combo.currentIndexChanged.connect(self._on_device_changed)
        self.browse_btn.clicked.connect(self._on_browse)
        self.scan_btn.clicked.connect(self._on_scan)
        self.folder_edit.textChanged.connect(lambda: self._update_settings_from_ui())
        self.pattern_edit.textChanged.connect(lambda: self._update_settings_from_ui())
        self.fmt_combo.currentTextChanged.connect(lambda: self._update_settings_from_ui())
        self.dpi_combo.currentTextChanged.connect(lambda: self._update_settings_from_ui())
        self.depth_combo.currentTextChanged.connect(lambda: self._update_settings_from_ui())
        self.ir_check.toggled.connect(lambda: self._update_settings_from_ui())
        self.autofocus_check.toggled.connect(lambda: self._update_settings_from_ui())
        self.ae_check.toggled.connect(lambda: self._update_settings_from_ui())
        self.frame_from_spin.valueChanged.connect(self._on_frame_from_changed)
        self.frame_to_spin.valueChanged.connect(self._on_frame_to_changed)
        self.scan_window_btn.clicked.connect(self._on_set_scan_window)
        self.scan_window_clear_btn.clicked.connect(self._on_clear_scan_window)

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
        self.device_combo.clear()
        self.device_combo.addItem("Detecting scanners…", None)
        self.device_combo.setEnabled(False)
        self.status_label.setText("Detecting scanners…")
        self.controller.request_scan_devices()

    def _on_refresh(self) -> None:
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
            self.ir_check.setEnabled(False)
            self.eject_btn.setVisible(False)
            self.frame_range_label.setVisible(False)
            self.frame_range_widget.setVisible(False)
            return

        caps = device.capabilities
        self.dpi_combo.setEnabled(True)
        self.depth_combo.setEnabled(True)
        self.ir_check.setEnabled(True)
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

    def _populate_form(self, caps: ScannerCapabilities) -> None:
        self.dpi_combo.blockSignals(True)
        self.depth_combo.blockSignals(True)
        self.ir_check.blockSignals(True)
        self.ae_check.blockSignals(True)
        self.frame_from_spin.blockSignals(True)
        self.frame_to_spin.blockSignals(True)

        # DPI
        self.dpi_combo.clear()
        if caps.supported_dpi:
            for d in caps.supported_dpi:
                self.dpi_combo.addItem(str(d), d)
        if self._settings.dpi:
            idx = self.dpi_combo.findData(self._settings.dpi)
            if idx >= 0:
                self.dpi_combo.setCurrentIndex(idx)
            else:
                self.dpi_combo.setCurrentText(str(self._settings.dpi))

        # Depth — default to the deepest supported when the saved value is absent
        # (a saved 16 does not exist on a 14-bit LS-50; findData → -1 must not
        # leave the combo silently on index 0 = 8-bit).
        self.depth_combo.clear()
        if caps.supported_depths:
            for d in caps.supported_depths:
                self.depth_combo.addItem(f"{d}-bit", d)
            idx = self.depth_combo.findData(self._settings.depth) if self._settings.depth else -1
            if idx < 0:
                idx = self.depth_combo.findData(max(caps.supported_depths))
            if idx >= 0:
                self.depth_combo.setCurrentIndex(idx)

        # IR
        self.ir_check.setEnabled(caps.ir_channel)
        if caps.ir_channel:
            self.ir_check.setChecked(self._settings.capture_ir)
            self.ir_check.setToolTip("Scan a separate infrared channel for dust detection")
        else:
            self.ir_check.setChecked(False)
            self.ir_check.setToolTip("IR scanning not supported by this device")

        # Auto-exposure
        self.ae_check.setEnabled(caps.auto_exposure)
        if caps.auto_exposure:
            self.ae_check.setChecked(self._settings.auto_exposure)
            self.ae_check.setToolTip("Meter exposure in hardware before the scan")
        else:
            self.ae_check.setChecked(False)
            self.ae_check.setToolTip("Auto-exposure not supported by this device")

        # Frame range — only a roll/strip feeder reporting a live capacity
        capacity = caps.adapter_frame_capacity
        has_frames = capacity is not None
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

        self.scan_window_row_label.setVisible(has_frames)
        self.scan_window_widget.setVisible(has_frames)
        self.scan_window_status.setVisible(has_frames)
        if has_frames:
            self.scan_window_btn.setText("Preview strip…")
            self.scan_window_btn.setToolTip("Preview each frame, set a window per frame, and pick which frames to scan")
            self._update_scan_window_status()

        self.dpi_combo.blockSignals(False)
        self.depth_combo.blockSignals(False)
        self.ir_check.blockSignals(False)
        self.ae_check.blockSignals(False)
        self.frame_from_spin.blockSignals(False)
        self.frame_to_spin.blockSignals(False)

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

        from negpy.desktop.view.widgets.strip_preview_dialog import StripPreviewDialog

        device = self._current_device()
        if device is None:
            return
        dialog = StripPreviewDialog(
            self.controller,
            device,
            initial_windows=self._settings.frame_windows,
            initial_selected=self._settings.selected_frames,
            initial_offset=self._settings.frame_offset_mm,
            initial_offset_modifier=self._settings.frame_offset_modifier_mm,
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

    def _on_clear_scan_window(self) -> None:
        from dataclasses import replace

        self.settings = replace(self._settings, scan_window=None, frame_windows={}, selected_frames=())
        self._update_scan_window_status()

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
            win_txt = f" · {n_windows} window(s)" if n_windows else ""
            self.scan_window_status.setText(f"Frames {frames_txt}{win_txt}{offset_txt}")
            return
        device = self._current_device()
        area = scan_window_to_area(self._settings.scan_window, device.capabilities.max_area_mm) if device else None
        if area is None:
            self.scan_window_status.setText(f"Full frame{offset_txt}")
        else:
            tl_x, tl_y, br_x, br_y = area
            self.scan_window_status.setText(f"{br_x - tl_x:.1f} × {br_y - tl_y:.1f} mm{offset_txt}")

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

        dpi = int(self.dpi_combo.currentData() or self.dpi_combo.currentText() or 3600)
        depth = int(self.depth_combo.currentData() or 16)
        capture_ir = self.ir_check.isEnabled() and self.ir_check.isChecked()
        autofocus = self.autofocus_check.isChecked()
        auto_exposure = self.ae_check.isEnabled() and self.ae_check.isChecked()
        pattern = self.pattern_edit.text().strip() or '{{ date }}_{{ "%03d" % seq }}'
        fmt = self.fmt_combo.currentText()

        frames, frame_windows, base_window = resolve_batch_selection(
            self._settings, self.frame_from_spin.value(), self.frame_to_spin.value()
        )
        base_params = ScanParams(
            dpi=dpi,
            depth=depth,
            capture_ir=capture_ir,
            autofocus=autofocus,
            auto_exposure=auto_exposure,
            window=base_window,
            frame_offset_mm=self._settings.frame_offset_mm,
        )

        self._update_settings_from_ui()
        self._save_settings()
        self.set_scanning(True)

        try:
            if device.capabilities.adapter_frame_capacity is not None:
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

    @pyqtSlot(float)
    def _on_scan_progress(self, progress: float) -> None:
        self.progress_bar.setVisible(True)
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
            self.status_label.setText(f"Batch complete: {len(paths)} frame(s)")

    @pyqtSlot()
    def _on_scan_cancelled(self) -> None:
        self.set_scanning(False)
        self.progress_bar.setVisible(False)
        self.status_label.setText("Scan stopped")

    @pyqtSlot(str)
    def _on_scan_error(self, msg: str) -> None:
        self.set_scanning(False)
        self.progress_bar.setVisible(False)
        self.status_label.setText(f"Error: {msg}")

    @pyqtSlot(bool)
    def _on_ejected(self, triggered: bool) -> None:
        device = self._current_device()
        self.eject_btn.setEnabled(bool(device and device.capabilities.can_eject) and not self._scanning)
        self.status_label.setText("Film ejected" if triggered else "This device has no eject control")

    @pyqtSlot(str)
    def _on_eject_error(self, msg: str) -> None:
        device = self._current_device()
        self.eject_btn.setEnabled(bool(device and device.capabilities.can_eject) and not self._scanning)
        self.status_label.setText(f"Eject failed: {msg}")

    # ── state helpers ─────────────────────────────────────────────────

    def set_scanning(self, active: bool) -> None:
        self._scanning = active
        device = self._current_device()
        self.eject_btn.setEnabled(bool(device and device.capabilities.can_eject) and not active)
        if active:
            self.scan_btn.setText(" Stop")
            self.scan_btn.setIcon(qta.icon("fa5s.stop", color=THEME.text_primary))
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
        else:
            self.scan_btn.setText(" Scan")
            self.scan_btn.setIcon(qta.icon("fa5s.camera-retro", color=THEME.text_primary))

    def _update_settings_from_ui(self) -> None:
        dpi_text = self.dpi_combo.currentData() or self.dpi_combo.currentText()
        depth_text = self.depth_combo.currentData() or 16
        try:
            dpi = int(dpi_text)
        except (ValueError, TypeError):
            dpi = 3600
        try:
            depth = int(depth_text)
        except (ValueError, TypeError):
            depth = 16

        from dataclasses import replace

        device = self._current_device()
        # replace(), never a fresh ScannerSettings: fields with no sidebar
        # control must survive UI edits — reconstruction silently resets any
        # field missing from this list.
        self.settings = replace(
            self._settings,
            last_device_id=device.id if device else self._settings.last_device_id,
            dpi=dpi,
            depth=depth,
            capture_ir=self.ir_check.isChecked() and self.ir_check.isEnabled(),
            autofocus=self.autofocus_check.isChecked(),
            auto_exposure=self.ae_check.isChecked() and self.ae_check.isEnabled(),
            frame_from=self.frame_from_spin.value(),
            frame_to=self.frame_to_spin.value(),
            output_folder=self.folder_edit.text().strip(),
            output_format=self.fmt_combo.currentText(),
            filename_pattern=self.pattern_edit.text().strip() or '{{ date }}_{{ "%03d" % seq }}',
        )


class _ScanUnsupportedPlaceholder(QWidget):
    """Shown on Windows where SANE is not available."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        # No layout alignment and no QSS padding: either one breaks the wrapped
        # QLabel's height-for-width negotiation and clips the text — the label
        # must be stretched to full width so it can report its wrapped height.
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)

        label = QLabel("Scanner support not yet available on Windows.")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setWordWrap(True)
        label.setStyleSheet(f"color: {THEME.text_muted}; font-size: {THEME.font_size_base}px;")
        layout.addWidget(label)
