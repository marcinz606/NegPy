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

        device_row.addWidget(self.device_combo, 1)
        device_row.addWidget(self.refresh_btn)
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

    def _connect_signals(self) -> None:
        self.refresh_btn.clicked.connect(self._on_refresh)
        self.device_combo.currentIndexChanged.connect(self._on_device_changed)
        self.browse_btn.clicked.connect(self._on_browse)
        self.scan_btn.clicked.connect(self._on_scan)
        self.folder_edit.textChanged.connect(lambda: self._update_settings_from_ui())
        self.pattern_edit.textChanged.connect(lambda: self._update_settings_from_ui())
        self.fmt_combo.currentTextChanged.connect(lambda: self._update_settings_from_ui())
        self.dpi_combo.currentTextChanged.connect(lambda: self._update_settings_from_ui())
        self.depth_combo.currentTextChanged.connect(lambda: self._update_settings_from_ui())
        self.ir_check.toggled.connect(lambda: self._update_settings_from_ui())

        # Controller signals
        self.controller.scan_devices_ready.connect(self._on_devices_ready)
        self.controller.scan_progress.connect(self._on_scan_progress)
        self.controller.scan_finished.connect(self._on_scan_finished)
        self.controller.scan_error.connect(self._on_scan_error)

    # ── activation hook ───────────────────────────────────────────────

    def on_activated(self) -> None:
        """Called when the Scan tab is switched to."""
        if not self._devices_loaded:
            self._request_devices()

    # ── slots ─────────────────────────────────────────────────────────

    def _request_devices(self) -> None:
        """Request device list from the scan worker thread."""
        if not self._sane_available():
            self._show_sane_missing()
            return
        self.device_combo.clear()
        self.device_combo.addItem("Detecting scanners…", None)
        self.device_combo.setEnabled(False)
        self.status_label.setText("Detecting scanners…")
        self.controller.request_scan_devices()

    @staticmethod
    def _sane_available() -> bool:
        try:
            import sane  # noqa: F401

            return True
        except Exception:
            return False

    def _show_sane_missing(self) -> None:
        import sys

        if sys.platform == "darwin":
            hint = "brew install sane-backends"
        else:
            hint = "sudo apt install libsane  # Debian/Ubuntu\nsudo pacman -S sane  # Arch\nor your distro's sane equivalent"
        self.device_combo.clear()
        self.device_combo.addItem("SANE not available", None)
        self.device_combo.setEnabled(False)
        self.scan_btn.setEnabled(False)
        self.status_label.setText(f"Scanner support requires SANE (libsane).\n\nTo enable:\n{hint}")

    def _on_refresh(self) -> None:
        self._request_devices()

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
            return

        caps = device.capabilities
        self.dpi_combo.setEnabled(True)
        self.depth_combo.setEnabled(True)
        self.ir_check.setEnabled(True)
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

        # Depth
        self.depth_combo.clear()
        if caps.supported_depths:
            for d in caps.supported_depths:
                self.depth_combo.addItem(f"{d}-bit", d)
        if self._settings.depth:
            idx = self.depth_combo.findData(self._settings.depth)
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

        self.dpi_combo.blockSignals(False)
        self.depth_combo.blockSignals(False)
        self.ir_check.blockSignals(False)

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

        # Build ScanRequest
        from negpy.desktop.workers.scan_worker import ScanRequest
        from negpy.infrastructure.scanners.params import ScanParams

        dpi = int(self.dpi_combo.currentData() or self.dpi_combo.currentText() or 3600)
        depth = int(self.depth_combo.currentData() or 16)
        capture_ir = self.ir_check.isEnabled() and self.ir_check.isChecked()

        params = ScanParams(
            dpi=dpi,
            depth=depth,
            capture_ir=capture_ir,
            area=None,
        )
        req = ScanRequest(
            device_id=device.id,
            params=params,
            output_folder=output_folder,
            filename_pattern=self.pattern_edit.text().strip() or '{{ date }}_{{ "%03d" % seq }}',
            output_format=self.fmt_combo.currentText(),
        )

        self._update_settings_from_ui()
        self._save_settings()

        self.set_scanning(True)
        self.controller.start_scan(req)

    @pyqtSlot(float)
    def _on_scan_progress(self, progress: float) -> None:
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(int(progress * 100))

    @pyqtSlot(str)
    def _on_scan_finished(self, path: str) -> None:
        self.set_scanning(False)
        self.progress_bar.setVisible(False)
        self.status_label.setText(f"Scanned: {path}")

    @pyqtSlot(str)
    def _on_scan_error(self, msg: str) -> None:
        self.set_scanning(False)
        self.progress_bar.setVisible(False)
        self.status_label.setText(f"Error: {msg}")

    # ── state helpers ─────────────────────────────────────────────────

    def set_scanning(self, active: bool) -> None:
        self._scanning = active
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

        device = self._current_device()
        self.settings = ScannerSettings(
            last_device_id=device.id if device else self._settings.last_device_id,
            dpi=dpi,
            depth=depth,
            capture_ir=self.ir_check.isChecked() and self.ir_check.isEnabled(),
            output_folder=self.folder_edit.text().strip(),
            output_format=self.fmt_combo.currentText(),
            filename_pattern=self.pattern_edit.text().strip() or '{{ date }}_{{ "%03d" % seq }}',
        )


class _ScanUnsupportedPlaceholder(QWidget):
    """Shown on Windows where SANE is not available."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        label = QLabel("Scanner support not yet available on Windows.")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setWordWrap(True)
        label.setStyleSheet(f"color: {THEME.text_muted}; font-size: {THEME.font_size_base}px; padding: 20px;")
        layout.addWidget(label)
