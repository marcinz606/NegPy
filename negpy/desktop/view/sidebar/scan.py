import json
import os
from pathlib import Path

import qtawesome as qta
from PyQt6.QtCore import Qt, pyqtSlot
from PyQt6.QtGui import QImage
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
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
from negpy.desktop.view.widgets.roll_slot_model import RollPreviewSlot
from negpy.desktop.view.widgets.roll_slot_selector import RollSlotSelector
from negpy.desktop.view.widgets.roll_thumbnail_renderer import (
    render_roll_thumbnail_rgb8,
)
from negpy.desktop.workers.ls5000_roll_worker import (
    RollPreviewRequest,
    RollScanCompletion,
    RollScanRequest,
    RollWorkerFailure,
    frame_choices,
)
from negpy.infrastructure.scanners.base import ScannerCapabilities, ScannerDevice
from negpy.infrastructure.scanners.settings import ScannerSettings
from negpy.kernel.system.config import APP_CONFIG


class ScanSidebar(QWidget):
    """Scanner control panel — replaces the originally planned modal ScanDialog."""

    def __init__(self, controller) -> None:
        super().__init__()
        self.controller = controller
        self._settings: ScannerSettings = self._load_settings()
        self._devices: list[ScannerDevice] = []
        self._scanning = False
        self._roll_scanning = False
        self._ejecting = False
        self._roll_preview_token: str | None = None
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
        self.refresh_btn.setFixedSize(40, 40)

        self.eject_btn = QPushButton()
        self.eject_btn.setIcon(qta.icon("fa5s.eject", color=THEME.text_secondary))
        self.eject_btn.setToolTip("Eject the loaded film")
        self.eject_btn.setFixedSize(40, 40)
        self.eject_btn.setVisible(False)

        device_row.addWidget(self.device_combo, 1)
        device_row.addWidget(self.refresh_btn)
        device_row.addWidget(self.eject_btn)
        layout.addLayout(device_row)

        # ── CAPS INFO ───────────────────────────────────────
        self.frame_label = hint_label("")
        layout.addWidget(self.frame_label)

        # ── LS-5000 ROLL WORKFLOW ──────────────────────────
        self.roll_preview_btn = QPushButton(" Load Roll Thumbnails")
        self.roll_preview_btn.setIcon(qta.icon("fa5s.images", color=THEME.text_primary))
        self.roll_preview_btn.setToolTip("Read the complete low-resolution roll index before choosing full-quality scans")
        self.roll_preview_btn.setVisible(False)
        layout.addWidget(self.roll_preview_btn)

        self.roll_slot_selector = RollSlotSelector()
        self.roll_slot_selector.setVisible(False)
        layout.addWidget(self.roll_slot_selector)

        self.roll_stop_btn = QPushButton(" Stop after current frame")
        self.roll_stop_btn.setIcon(qta.icon("fa5s.stop", color=THEME.text_primary))
        self.roll_stop_btn.setToolTip("Let the active scanner transaction finish safely, then stop before the next selected frame")
        self.roll_stop_btn.setVisible(False)
        layout.addWidget(self.roll_stop_btn)

        self.roll_status_label = hint_label("")
        self.roll_status_label.setVisible(False)
        layout.addWidget(self.roll_status_label)

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

        self.frame_spin = QSpinBox()
        self.frame_spin.setRange(0, 0)
        self.frame_spin.setSpecialValueText("Current")
        self.frame_spin.setToolTip("Frame selection not supported by this device")
        self.form.addRow("Frame #", self.frame_spin)

        self.autofocus_check = QCheckBox("Autofocus")
        self.autofocus_check.setChecked(True)
        self.autofocus_check.setToolTip("Autofocus before scanning (recommended because film is rarely perfectly flat)")
        self.form.addRow("Autofocus", self.autofocus_check)

        self.ae_check = QCheckBox("Auto-Exposure")
        self.ae_check.setToolTip("Hardware auto-exposure not supported by this device")
        self.form.addRow("Auto-Exposure", self.ae_check)

        self.samples_combo = QComboBox()
        self.samples_combo.setToolTip("Hardware multi-sampling passes per line (higher = less noise, slower)")
        for n in (1, 2, 4, 8, 16):
            self.samples_combo.addItem(f"{n}x", n)
        self.form.addRow("Samples/pass", self.samples_combo)

        self.archival_split_check = QCheckBox("Archival Split (RGB4x + IR1x)")
        self.archival_split_check.setToolTip("Requires both IR and hardware multi-sampling support")
        self.form.addRow("Archival", self.archival_split_check)

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

        # ── REGISTRATION ──────────────────────────────────────
        layout.addWidget(section_subheader("REGISTRATION"))

        self.registered_geometry_check = QCheckBox("Use Registered Geometry")
        self.registered_geometry_check.setToolTip("Registered geometry not supported by this device")
        layout.addWidget(self.registered_geometry_check)

        self.registration_form = QFormLayout()
        self.registration_form.setSpacing(6)

        self.subframe_spin = QDoubleSpinBox()
        self.subframe_spin.setDecimals(2)
        # Upper bound mirrors RollRegistrationConfig.maximum_subframe_mm (the
        # validated LS-5000 frame-pitch ceiling). This is a UI convenience bound
        # only; SaneBackend.scan() is the real enforcement point.
        self.subframe_spin.setRange(0.0, 37.82)
        self.subframe_spin.setSuffix(" mm")
        self.subframe_spin.setEnabled(False)
        self.subframe_spin.setToolTip("Fine transport shift for this frame (RegisteredScanGeometry.subframe_mm)")
        self.registration_form.addRow("Subframe", self.subframe_spin)

        self.br_y_spin = QSpinBox()
        # Upper bound mirrors RollRegistrationConfig.maximum_br_y_device_px
        # (the validated LS-5000 device-pixel ceiling). This is a UI convenience
        # bound only; SaneBackend.scan() is the real enforcement point.
        self.br_y_spin.setRange(0, 5958)
        self.br_y_spin.setEnabled(False)
        self.br_y_spin.setToolTip(
            "Inclusive bottom coordinate of the shortened scan window, in device pixels (RegisteredScanGeometry.br_y_device_px)"
        )
        self.registration_form.addRow("BR-Y", self.br_y_spin)

        layout.addLayout(self.registration_form)

        self.load_registration_btn = QPushButton(" Load Registration JSON…")
        self.load_registration_btn.setIcon(qta.icon("fa5s.folder-open", color=THEME.text_primary))
        self.load_registration_btn.setEnabled(False)
        self.load_registration_btn.setToolTip(
            "Load Subframe/BR-Y for the current Frame # from a registration manifest "
            '({"frames": [{"frame": N, "subframe_mm": ..., "br_y": ...}]})'
        )
        layout.addWidget(self.load_registration_btn)

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
        self.archival_split_check.setChecked(self._settings.archival_split_capture)

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
        self.samples_combo.currentTextChanged.connect(lambda: self._update_settings_from_ui())
        self.archival_split_check.toggled.connect(self._on_archival_split_toggled)
        self.registered_geometry_check.toggled.connect(self._on_registered_geometry_toggled)
        self.load_registration_btn.clicked.connect(self._on_load_registration_json)
        self.roll_preview_btn.clicked.connect(self._on_roll_preview)
        self.roll_stop_btn.clicked.connect(self._on_roll_stop)
        self.roll_slot_selector.scan_requested.connect(self._on_roll_scan_selected)
        self.roll_slot_selector.thumbnail_reload_requested.connect(self.controller.reload_ls5000_roll_thumbnail)

        # Controller signals
        self.controller.scan_devices_ready.connect(self._on_devices_ready)
        self.controller.scan_progress.connect(self._on_scan_progress)
        self.controller.scan_finished.connect(self._on_scan_finished)
        self.controller.scan_error.connect(self._on_scan_error)
        self.controller.scan_ejected.connect(self._on_ejected)
        self.controller.scan_eject_error.connect(self._on_eject_error)
        self.controller.ls5000_roll_preview_ready.connect(self._on_roll_preview_ready)
        self.controller.ls5000_roll_preview_invalidated.connect(
            self._on_roll_preview_invalidated
        )
        self.controller.ls5000_roll_thumbnail_ready.connect(self._on_roll_thumbnail_ready)
        self.controller.ls5000_roll_progress.connect(self._on_roll_progress)
        self.controller.ls5000_roll_finished.connect(self._on_roll_finished)
        self.controller.ls5000_roll_error.connect(self._on_roll_error)

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
        self._invalidate_roll_preview()
        self._request_devices()

    def _on_eject(self) -> None:
        device = self._current_device()
        if device is None or self._scanning or self._roll_scanning or self._ejecting:
            return
        self._ejecting = True
        self.device_combo.setEnabled(False)
        self.refresh_btn.setEnabled(False)
        self.eject_btn.setEnabled(False)
        self.status_label.setText("Ejecting film…")
        self.controller.eject_scanner(device.id)

    @pyqtSlot(bool)
    def _on_ejected(self, ejected: bool) -> None:
        self._ejecting = False
        self.device_combo.setEnabled(bool(self._devices))
        self.refresh_btn.setEnabled(True)
        device = self._current_device()
        self.eject_btn.setEnabled(bool(device and device.capabilities.can_eject))
        if ejected:
            self._invalidate_roll_preview()
            self.status_label.setText("Film ejected.")
        else:
            self.status_label.setText("This scanner does not expose a film-eject control.")

    @pyqtSlot(str)
    def _on_eject_error(self, message: str) -> None:
        self._ejecting = False
        self.device_combo.setEnabled(bool(self._devices))
        self.refresh_btn.setEnabled(True)
        device = self._current_device()
        self.eject_btn.setEnabled(bool(device and device.capabilities.can_eject))
        self.status_label.setText(f"Could not eject film: {message}")

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
        self._invalidate_roll_preview()
        self._update_device_caps()

    def _current_device(self) -> ScannerDevice | None:
        device_id = self.device_combo.currentData()
        if not device_id:
            return None
        for d in self._devices:
            if d.id == device_id:
                return d
        return None

    @staticmethod
    def _supports_ls5000_roll_workflow(device: ScannerDevice | None) -> bool:
        if device is None:
            return False
        normalized_model = device.model.upper().replace(" ", "-")
        return (
            "LS-5000" in normalized_model
            and device.capabilities.adapter_frame_capacity == 40
        )

    def _clear_roll_preview(self) -> None:
        """Clear stale thumbnails without sending another worker command."""

        self._roll_preview_token = None
        self.roll_slot_selector.set_slots([])
        self.roll_slot_selector.setVisible(False)
        self.roll_preview_btn.setText(" Load Roll Thumbnails")

    def _invalidate_roll_preview(self) -> None:
        """Clear the UI and invalidate the worker-owned coordinate binding."""

        self._clear_roll_preview()
        self.controller.invalidate_ls5000_roll_preview()

    def _update_device_caps(self) -> None:
        device = self._current_device()
        if device is None:
            self.scan_btn.setEnabled(False)
            self.frame_label.setText("")
            self.dpi_combo.setEnabled(False)
            self.depth_combo.setEnabled(False)
            self.ir_check.setEnabled(False)
            self.samples_combo.setEnabled(False)
            self.frame_spin.setEnabled(False)
            self.ae_check.setEnabled(False)
            self.archival_split_check.setEnabled(False)
            self.registered_geometry_check.setEnabled(False)
            self.roll_preview_btn.setVisible(False)
            self.roll_slot_selector.setVisible(False)
            self.roll_status_label.setVisible(False)
            self.eject_btn.setVisible(False)
            self.eject_btn.setEnabled(False)
            self._update_registration_fields_enabled()
            return

        caps = device.capabilities
        self.dpi_combo.setEnabled(True)
        self.depth_combo.setEnabled(True)
        self.ir_check.setEnabled(True)
        self.samples_combo.setEnabled(True)
        self.frame_label.setText(f"Frame: {caps.max_area_mm[0]:.0f} × {caps.max_area_mm[1]:.0f} mm")
        self.eject_btn.setVisible(caps.can_eject)
        self.eject_btn.setEnabled(
            caps.can_eject
            and not self._scanning
            and not self._roll_scanning
            and not self._ejecting
        )
        roll_supported = self._supports_ls5000_roll_workflow(device)
        self.roll_preview_btn.setVisible(roll_supported)
        self.roll_preview_btn.setEnabled(roll_supported and not self._roll_scanning)
        if not roll_supported:
            self.roll_slot_selector.setVisible(False)
            self.roll_status_label.setVisible(False)

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
        self.samples_combo.blockSignals(True)
        self.frame_spin.blockSignals(True)
        self.ae_check.blockSignals(True)
        self.archival_split_check.blockSignals(True)
        self.registered_geometry_check.blockSignals(True)

        # DPI
        self.dpi_combo.clear()
        if caps.supported_dpi:
            best_dpi = max(caps.supported_dpi)
            for d in caps.supported_dpi:
                label = f"{d} dpi"
                if d == best_dpi:
                    label += " (Best quality)"
                self.dpi_combo.addItem(label, d)
        if caps.supported_dpi:
            self.dpi_combo.setCurrentIndex(self.dpi_combo.findData(max(caps.supported_dpi)))
        elif self._settings.dpi:
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

        # Samples/pass (hardware multi-sampling)
        self.samples_combo.setEnabled(caps.multi_sample)
        if caps.multi_sample:
            idx = self.samples_combo.findData(self._settings.samples_per_scan)
            self.samples_combo.setCurrentIndex(idx if idx >= 0 else 0)
            self.samples_combo.setToolTip("Hardware multi-sampling passes per line (higher = less noise, slower)")
        else:
            self.samples_combo.setCurrentIndex(0)
            self.samples_combo.setToolTip("Multi-sampling not supported by this device")

        # Frame selection (roll-adapter transport position)
        if caps.adapter_frame_capacity is not None:
            self.frame_spin.setEnabled(True)
            self.frame_spin.setRange(0, caps.adapter_frame_capacity)
            self.frame_spin.setValue(min(self.frame_spin.value(), caps.adapter_frame_capacity))
            self.frame_spin.setToolTip(
                f"Select a specific frame (1 to {caps.adapter_frame_capacity}) on the roll adapter before scanning "
                "(“Current” leaves the transport position unchanged)"
            )
        else:
            self.frame_spin.setEnabled(False)
            self.frame_spin.setRange(0, 0)
            self.frame_spin.setValue(0)
            self.frame_spin.setToolTip("Frame selection not supported by this device")

        # Hardware auto-exposure
        self.ae_check.setEnabled(caps.auto_exposure)
        if caps.auto_exposure:
            self.ae_check.setChecked(self._settings.auto_exposure)
            self.ae_check.setToolTip("Meter this positioned frame in hardware before capture")
        else:
            self.ae_check.setChecked(False)
            self.ae_check.setToolTip("Hardware auto-exposure not supported by this device")

        # Archival split-capture (validated RGB4x + IR1x practical-parity recipe)
        archival_supported = caps.ir_channel and caps.multi_sample
        self.archival_split_check.setEnabled(archival_supported)
        if archival_supported:
            self.archival_split_check.setChecked(self._settings.archival_split_capture)
            self.archival_split_check.setToolTip(
                "Validated archival recipe: 4× hardware-multisampled RGB plus a registered single-pass IR channel"
            )
        else:
            self.archival_split_check.setChecked(False)
            self.archival_split_check.setToolTip("Requires both IR and hardware multi-sampling support")

        # Registered geometry (fine transport shift + shortened scan window).
        # Always resets on a capability refresh (device switch or Refresh
        # click); frame-specific geometry from a previous device/session
        # must never silently carry over to a new one.
        self.registered_geometry_check.setEnabled(caps.registered_geometry)
        self.registered_geometry_check.setChecked(False)
        self.subframe_spin.setValue(0.0)
        self.br_y_spin.setValue(0)
        if caps.registered_geometry:
            self.registered_geometry_check.setToolTip("Position a fine transport shift and shortened scan window for this frame")
        else:
            self.registered_geometry_check.setToolTip("Registered geometry not supported by this device")

        self.dpi_combo.blockSignals(False)
        self.depth_combo.blockSignals(False)
        self.ir_check.blockSignals(False)
        self.samples_combo.blockSignals(False)
        self.frame_spin.blockSignals(False)
        self.ae_check.blockSignals(False)
        self.archival_split_check.blockSignals(False)
        self.registered_geometry_check.blockSignals(False)

        # Cross-widget effects that must run after signals are unblocked
        # (archival split may need to re-lock IR/samples; registration fields
        # follow the "Use Registered Geometry" checkbox's restored state).
        self._apply_archival_split_interlock()
        self._update_registration_fields_enabled()

    # ── new-control interlocks ───────────────────────────────────────────

    def _apply_archival_split_interlock(self) -> None:
        """Force + lock IR/samples to the validated RGB4x+IR1x recipe while
        archival split-capture is active; leaves them alone otherwise."""
        if not (self.archival_split_check.isChecked() and self.archival_split_check.isEnabled()):
            return
        self.ir_check.blockSignals(True)
        self.ir_check.setChecked(True)
        self.ir_check.blockSignals(False)
        idx = self.samples_combo.findData(4)
        if idx >= 0:
            self.samples_combo.blockSignals(True)
            self.samples_combo.setCurrentIndex(idx)
            self.samples_combo.blockSignals(False)
        self.ir_check.setEnabled(False)
        self.samples_combo.setEnabled(False)

    def _on_archival_split_toggled(self, _checked: bool) -> None:
        if not self.archival_split_check.isChecked():
            # Restore IR/samples to whatever the device's own capabilities allow.
            device = self._current_device()
            caps = device.capabilities if device is not None else None
            self.ir_check.setEnabled(caps.ir_channel if caps is not None else False)
            self.samples_combo.setEnabled(caps.multi_sample if caps is not None else False)
        self._apply_archival_split_interlock()
        self._update_settings_from_ui()

    def _on_registered_geometry_toggled(self, _checked: bool) -> None:
        self._update_registration_fields_enabled()

    def _update_registration_fields_enabled(self) -> None:
        active = self.registered_geometry_check.isChecked() and self.registered_geometry_check.isEnabled()
        self.subframe_spin.setEnabled(active)
        self.br_y_spin.setEnabled(active)
        self.load_registration_btn.setEnabled(self.registered_geometry_check.isEnabled())

    def _on_load_registration_json(self) -> None:
        frame_value = self.frame_spin.value()
        if frame_value <= 0:
            self.status_label.setText("Set a Frame # before loading a registration manifest.")
            return

        path, _ = QFileDialog.getOpenFileName(self, "Load Registration JSON", "", "JSON Files (*.json)")
        if not path:
            return

        from negpy.infrastructure.scanners.params import parse_registration_manifest

        try:
            with open(path, encoding="utf-8") as stream:
                data = json.load(stream)
            geometry = parse_registration_manifest(data, frame=frame_value)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            self.status_label.setText(f"Could not load registration manifest: {exc}")
            return

        self.subframe_spin.setValue(geometry.subframe_mm)
        self.br_y_spin.setValue(geometry.br_y_device_px)
        self.registered_geometry_check.setChecked(True)
        self._update_registration_fields_enabled()
        self.status_label.setText(f"Loaded registration for frame {frame_value} from {os.path.basename(path)}")

    def _on_browse(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if folder:
            self.folder_edit.setText(folder)
            self._update_settings_from_ui()

    def _on_roll_preview(self) -> None:
        if self._roll_scanning:
            return
        device = self._current_device()
        if not self._supports_ls5000_roll_workflow(device):
            self.roll_status_label.setText("Whole-roll thumbnails require a Nikon LS-5000 roll feeder.")
            self.roll_status_label.setVisible(True)
            return
        attempts_root = Path(APP_CONFIG.cache_dir) / "ls5000-roll" / "preview-attempts"
        self._invalidate_roll_preview()
        self._set_roll_scanning(True)
        self.roll_status_label.setText("Reading the whole roll for thumbnails…")
        self.roll_status_label.setVisible(True)
        self.controller.start_ls5000_roll_preview(
            RollPreviewRequest(
                attempts_root=str(attempts_root),
                device_id=device.id,
                adapter_frame_capacity=device.capabilities.adapter_frame_capacity,
            )
        )

    def _on_roll_stop(self) -> None:
        self.controller.cancel_scan()
        self.roll_stop_btn.setEnabled(False)
        self.roll_status_label.setText("Stopping safely after the active scanner transaction…")

    def _on_roll_scan_selected(self, slot_ids: list[int]) -> None:
        if self._roll_scanning or not slot_ids:
            return
        device = self._current_device()
        if not self._supports_ls5000_roll_workflow(device) or device is None:
            return
        if self._roll_preview_token is None:
            self.roll_status_label.setText(
                "Load a new whole-roll preview before scanning selected frames."
            )
            self.roll_status_label.setVisible(True)
            return

        unconfirmed_offsets = [slot_id for slot_id in slot_ids if not self.roll_slot_selector.model.boundary_offset_is_confirmed(slot_id)]
        if unconfirmed_offsets:
            formatted = ", ".join(f"{slot_id:02d}" for slot_id in unconfirmed_offsets)
            self.roll_status_label.setText(f"Reload the edited thumbnail(s) for slot {formatted} before scanning.")
            self.roll_status_label.setVisible(True)
            return

        output_folder = self.folder_edit.text().strip()
        if not output_folder:
            self._on_browse()
            output_folder = self.folder_edit.text().strip()
            if not output_folder:
                return

        material = self.roll_slot_selector.scan_material()
        if not material.captures_infrared:
            caps = device.capabilities
            if not (caps.multi_sample and caps.registered_geometry and caps.auto_exposure):
                self.roll_status_label.setText(
                    "B&W RGB-only scanning requires the patched LS-5000 SANE "
                    "driver with 4× sampling, registered geometry, and auto-exposure."
                )
                self.roll_status_label.setVisible(True)
                return

        offsets = {slot_id: self.roll_slot_selector.model.boundary_offset_for_slot_id(slot_id) for slot_id in slot_ids}
        choices = frame_choices(slot_ids, offsets)
        attempts_root = Path(output_folder) / ".negpy-ls5000" / "attempts"
        request = RollScanRequest(
            device_id=device.id,
            adapter_frame_capacity=device.capabilities.adapter_frame_capacity,
            preview_token=self._roll_preview_token,
            attempts_root=str(attempts_root),
            output_folder=output_folder,
            filename_pattern=(self.pattern_edit.text().strip() or '{{ date }}_{{ "%03d" % seq }}'),
            material=material,
            frames=choices,
        )
        self._update_settings_from_ui()
        self._save_settings()
        self._set_roll_scanning(True)
        self.controller.start_ls5000_roll_scan(request)

    @staticmethod
    def _roll_thumbnail_qimage(thumbnail: object) -> QImage:
        """Make a small positive display image from scanner-linear negative RGB."""

        display = render_roll_thumbnail_rgb8(thumbnail)
        image = QImage(
            display.data,
            display.shape[1],
            display.shape[0],
            display.strides[0],
            QImage.Format.Format_RGB888,
        )
        return image.copy()

    @pyqtSlot(str, object)
    def _on_roll_preview_ready(self, preview_token: str, session: object) -> None:
        try:
            slots = [
                RollPreviewSlot(
                    slot_id=slot.slot_id,
                    thumbnail=self._roll_thumbnail_qimage(slot.thumbnail),
                    warnings=tuple(slot.warnings),
                    boundary_offset=slot.boundary_offset_rows,
                )
                for slot in session.slots
            ]
        except Exception as error:
            self._on_roll_error(
                RollWorkerFailure(
                    f"Could not display roll thumbnails: {error}",
                    False,
                )
            )
            return
        self._roll_preview_token = preview_token
        self.roll_slot_selector.set_slots(slots)
        self.roll_slot_selector.setVisible(True)
        self.roll_preview_btn.setText(" Reload Roll Thumbnails")
        self.roll_status_label.setText(
            f"Loaded {len(slots)} scanner slots. Adjust Film Spacing Offset and "
            "reload any thumbnail that does not show the complete negative. "
            "If the film is ejected or reinserted, reload the whole roll first."
        )
        self._set_roll_scanning(False)

    @pyqtSlot()
    def _on_roll_preview_invalidated(self) -> None:
        self._clear_roll_preview()

    @pyqtSlot(int, int, object)
    def _on_roll_thumbnail_ready(
        self,
        slot_id: int,
        boundary_offset: int,
        thumbnail: object,
    ) -> None:
        try:
            self.roll_slot_selector.confirm_slot_thumbnail(
                slot_id,
                boundary_offset,
                self._roll_thumbnail_qimage(thumbnail),
            )
            self.roll_status_label.setText(f"Reloaded slot {slot_id:02d} at Film Spacing Offset {boundary_offset:+d}.")
        except Exception as error:
            self.roll_status_label.setText(f"Could not display reloaded slot {slot_id:02d}: {error}")

    @pyqtSlot(object)
    def _on_roll_progress(self, progress: object) -> None:
        total = max(1, int(progress.total))
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(int(100 * int(progress.completed) / total))
        self.progress_bar.setFormat(str(progress.message))
        self.roll_status_label.setText(str(progress.message))
        self.roll_status_label.setVisible(True)

    @pyqtSlot(object)
    def _on_roll_finished(self, completion: RollScanCompletion) -> None:
        self._set_roll_scanning(False)
        count = len(completion.rgb_paths)
        if completion.stopped:
            self.roll_status_label.setText(f"Stopped safely after completing {count} frame(s).")
        else:
            self.roll_status_label.setText(f"Finished {count} full-quality frame(s); added them to NegPy.")

    @pyqtSlot(object)
    def _on_roll_error(self, failure: RollWorkerFailure) -> None:
        self._set_roll_scanning(False)
        if failure.recovery_required:
            self._clear_roll_preview()
        suffix = " Power-cycle the scanner before trying again." if failure.recovery_required else ""
        self.roll_status_label.setText(f"{failure.message}{suffix}")
        self.roll_status_label.setVisible(True)

    def _set_roll_scanning(self, active: bool) -> None:
        self._roll_scanning = active
        self.device_combo.setEnabled(not active)
        self.roll_preview_btn.setEnabled(not active)
        self.roll_slot_selector.setEnabled(not active)
        self.scan_btn.setEnabled(not active)
        self.refresh_btn.setEnabled(not active)
        device = self._current_device()
        self.eject_btn.setEnabled(
            not active
            and not self._ejecting
            and bool(device and device.capabilities.can_eject)
        )
        self.roll_stop_btn.setVisible(active)
        self.roll_stop_btn.setEnabled(active)
        self.progress_bar.setVisible(active)
        if active:
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(0)
        else:
            self.progress_bar.setVisible(False)
            self.progress_bar.setFormat("Scanning… %p%")

    def _on_scan(self) -> None:
        if self._roll_scanning:
            return
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

        dpi = int(self.dpi_combo.currentData() or self.dpi_combo.currentText() or 3600)
        depth = int(self.depth_combo.currentData() or 16)
        # isChecked() alone is authoritative here (no isEnabled() guard): a
        # device is guaranteed selected at this point (checked above), and
        # ir_check.isChecked() is always kept truthful for it: forced False
        # by _populate_form when the device lacks IR, forced True by the
        # archival split-capture interlock, which also *disables* the box
        # while leaving it checked=True to lock out manual edits. Gating on
        # isEnabled() too would silently drop IR during archival capture.
        capture_ir = self.ir_check.isChecked()
        frame = self.frame_spin.value() or None
        use_registered_geometry = self.registered_geometry_check.isEnabled() and self.registered_geometry_check.isChecked()

        try:
            params = self.controller.build_scan_params(
                dpi=dpi,
                depth=depth,
                capture_ir=capture_ir,
                autofocus=self.autofocus_check.isChecked(),
                samples_per_scan=int(self.samples_combo.currentData() or 1),
                frame=frame,
                auto_exposure=self.ae_check.isEnabled() and self.ae_check.isChecked(),
                subframe_mm=self.subframe_spin.value() if use_registered_geometry else None,
                br_y_device_px=self.br_y_spin.value() if use_registered_geometry else None,
            )
        except ValueError as exc:
            self.status_label.setText(f"Cannot start scan: {exc}")
            return

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
        device = self._current_device()
        self.eject_btn.setEnabled(
            not active
            and not self._ejecting
            and bool(device and device.capabilities.can_eject)
        )
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
            # See the matching comment in _on_scan(): isChecked() alone is
            # authoritative; an isEnabled() guard would misread the archival
            # split-capture interlock's locked-but-checked state as "off".
            capture_ir=self.ir_check.isChecked(),
            autofocus=self.autofocus_check.isChecked(),
            samples_per_scan=int(self.samples_combo.currentData() or 1),
            auto_exposure=self.ae_check.isChecked() and self.ae_check.isEnabled(),
            archival_split_capture=self.archival_split_check.isChecked() and self.archival_split_check.isEnabled(),
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
