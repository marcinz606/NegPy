import json
import os
import shutil
from dataclasses import replace
from pathlib import Path

import qtawesome as qta
from PyQt6.QtCore import Qt, pyqtSignal, pyqtSlot
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
    RollOperation,
    RollPreviewRequest,
    RollScanCompletion,
    RollScanRequest,
    RollThumbnailReloadFailure,
    RollWorkerFailure,
    frame_choices,
)
from negpy.infrastructure.scanners.ls5000_single_pass.capture_process import (
    ReviewedRollFingerprint,
    compare_reviewed_roll_fingerprints,
)
from negpy.infrastructure.scanners.base import ScannerCapabilities, ScannerDevice
from negpy.infrastructure.scanners.settings import (
    MAX_PREVIEW_METER_INSET_PERCENT,
    ScannerSettings,
)
from negpy.kernel.system.config import APP_CONFIG
from negpy.services.scanning.roll_preview_controls import ScanMaterial


_SCANNER_STATUS_COLORS = {
    "muted": "#888780",
    "ready": "#1D9E75",
    "active": "#4A8FE8",
    "warning": "#C8922E",
    "error": "#E24B4A",
}

_ROLL_OPERATION_PREVIEW = "preview"
_ROLL_OPERATION_SELECTED_SCAN = "selected-scan"
_ROLL_DEFAULT_FILENAME_PATTERN = (
    '{{ date }}_slot{{ "%02d" % slot }}_{{ "%03d" % seq }}'
)


class ScanSidebar(QWidget):
    """Scanner control panel — replaces the originally planned modal ScanDialog."""

    roll_retry_selection_restored = pyqtSignal()

    def __init__(self, controller) -> None:
        super().__init__()
        self.controller = controller
        self._settings: ScannerSettings = self._load_settings()
        self._devices: list[ScannerDevice] = []
        self._scanning = False
        self._scan_cancel_pending = False
        self._active_scan_device_id: str | None = None
        self._active_scan_frame: int | None = None
        self._roll_scanning = False
        self._roll_operation: str | None = None
        self._ejecting = False
        self._roll_preview_token: str | None = None
        self._roll_preview_session: object | None = None
        self._roll_preview_thumbnails: dict[int, object] = {}
        self._roll_thumbnail_reload_pending: tuple[int, int] | None = None
        self._roll_batch_selected_slots: tuple[int, ...] = ()
        self._roll_completed_slots: set[int] = set()
        self._roll_batch_fingerprint: ReviewedRollFingerprint | None = None
        self._roll_batch_material: ScanMaterial | None = None
        self._roll_retry_pending_slots: tuple[int, ...] = ()
        self._roll_retry_fingerprint: ReviewedRollFingerprint | None = None
        self._roll_retry_material: ScanMaterial | None = None
        self._roll_batch_in_progress = False
        self._devices_loaded = False
        self._init_ui()
        self._connect_signals()
        self._update_roll_storage_estimate()

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

        self.device_status_label = QLabel()
        self.device_status_label.setObjectName("scanner_status")
        self.device_status_label.setWordWrap(True)
        layout.addWidget(self.device_status_label)
        self._set_scanner_status("Detecting scanners…", "active")

        # ── CAPS INFO ───────────────────────────────────────
        self.frame_label = hint_label("")
        layout.addWidget(self.frame_label)

        # ── LS-5000 ROLL WORKFLOW ──────────────────────────
        self.sa30_compatible_check = QCheckBox("SA-30 / converted SA-21 (40 slots)")
        self.sa30_compatible_check.setAccessibleName("Use 40-slot SA-30-compatible roll feeder")
        self.sa30_compatible_check.setToolTip(
            "Enable only for a genuine SA-30 or an SA-21 converted to the "
            "same 40-slot transport. A parked feeder cannot report its capacity."
        )
        self.sa30_compatible_check.setVisible(False)
        layout.addWidget(self.sa30_compatible_check)

        self.roll_preview_btn = QPushButton(" Load Roll Thumbnails")
        self.roll_preview_btn.setIcon(qta.icon("fa5s.images", color=THEME.text_primary))
        self.roll_preview_btn.setToolTip("Read the complete low-resolution roll index before choosing full-quality scans")
        self.roll_preview_btn.setVisible(False)
        layout.addWidget(self.roll_preview_btn)

        self.roll_quality_widget = QWidget()
        roll_quality_form = QFormLayout(self.roll_quality_widget)
        roll_quality_form.setContentsMargins(0, 0, 0, 0)
        roll_quality_form.setSpacing(6)

        self.roll_preview_resolution_box = QLineEdit("97 dpi (thumbnails only)")
        self.roll_preview_resolution_box.setReadOnly(True)
        self.roll_preview_resolution_box.setAccessibleName("Roll preview resolution")
        self.roll_preview_resolution_box.setToolTip("Low-resolution roll thumbnails used only to choose and align frames")
        roll_quality_form.addRow("Roll preview", self.roll_preview_resolution_box)

        self.preview_meter_inset_spin = QSpinBox()
        self.preview_meter_inset_spin.setObjectName("roll_preview_meter_inset")
        self.preview_meter_inset_spin.setAccessibleName("Metering crop percentage per edge")
        self.preview_meter_inset_spin.setRange(0, MAX_PREVIEW_METER_INSET_PERCENT)
        self.preview_meter_inset_spin.setValue(self._settings.preview_meter_inset_percent)
        self.preview_meter_inset_spin.setSuffix("% per edge")
        self.preview_meter_inset_spin.setToolTip(
            "Crop this percentage from every edge only when calculating thumbnail brightness. "
            "Changing it recalculates all loaded previews from saved thumbnail data; it does not scan the film again. "
            "The outlined box shows the metered area. The full negative and rebate stay visible, and the saved negative is unchanged. "
            "10% matches full-scan metering."
        )
        roll_quality_form.addRow("Metering crop", self.preview_meter_inset_spin)

        self.preview_non_inverted_negative_check = QCheckBox("Show non-inverted negative")
        self.preview_non_inverted_negative_check.setObjectName("roll_preview_non_inverted_negative")
        self.preview_non_inverted_negative_check.setAccessibleName("Show non-inverted roll previews")
        self.preview_non_inverted_negative_check.setChecked(self._settings.preview_show_non_inverted_negative)
        self.preview_non_inverted_negative_check.setToolTip(
            "Display only. Off shows a readable inverted positive; on shows a "
            "display-leveled negative without inversion while retaining its channel "
            "balance. The saved scan is unchanged."
        )
        roll_quality_form.addRow("Preview display", self.preview_non_inverted_negative_check)

        self.roll_scan_resolution_box = QLineEdit("4000 dpi (Best quality)")
        self.roll_scan_resolution_box.setReadOnly(True)
        self.roll_scan_resolution_box.setAccessibleName("Full roll scan resolution")
        self.roll_scan_resolution_box.setToolTip("Native-resolution capture used for every selected full-quality frame")
        roll_quality_form.addRow("Final scan", self.roll_scan_resolution_box)

        self.roll_master_format_box = QLineEdit("16-bit TIFF (linear negative)")
        self.roll_master_format_box.setReadOnly(True)
        self.roll_master_format_box.setAccessibleName("Roll master format")
        self.roll_master_format_box.setToolTip("The roll workflow saves scanner-linear negative data; inversion happens in Process")
        roll_quality_form.addRow("Saved master", self.roll_master_format_box)

        self.roll_quality_widget.setVisible(False)
        layout.addWidget(self.roll_quality_widget)

        self.roll_slot_selector = RollSlotSelector()
        self.roll_slot_selector.set_meter_inset_percent(self._settings.preview_meter_inset_percent)
        self.roll_slot_selector.setVisible(False)
        layout.addWidget(self.roll_slot_selector)

        self.roll_stop_btn = QPushButton(" Stop after current frame")
        self.roll_stop_btn.setIcon(qta.icon("fa5s.stop", color=THEME.text_primary))
        self.roll_stop_btn.setToolTip("Let the active scanner transaction finish safely, then stop before the next selected frame")
        self.roll_stop_btn.setVisible(False)
        layout.addWidget(self.roll_stop_btn)

        self.retry_remaining_btn = QPushButton("Select remaining")
        self.retry_remaining_btn.setObjectName("roll_retry_remaining")
        self.retry_remaining_btn.setAccessibleName("Select unfinished roll slots")
        self.retry_remaining_btn.setToolTip(
            "Reselect only slots that did not finish in the last roll scan. "
            "Review their current thumbnails before scanning again."
        )
        self.retry_remaining_btn.setVisible(False)
        layout.addWidget(self.retry_remaining_btn)

        self.roll_status_label = hint_label("")
        self.roll_status_label.setVisible(False)
        layout.addWidget(self.roll_status_label)

        # ── SETTINGS ────────────────────────────────────────
        self.form = QFormLayout()
        self.form.setSpacing(6)

        self.dpi_combo = QComboBox()
        self.dpi_combo.setToolTip(
            "Resolution for a conventional single-frame scan. Roll thumbnails use 97 dpi; selected roll frames use 4000 dpi."
        )
        self.dpi_combo.setEditable(True)
        self.form.addRow("Single-frame DPI", self.dpi_combo)

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

        self.archival_split_check = QCheckBox("Archival RGB 4x + IR")
        self.archival_split_check.setToolTip("Requires both IR and hardware multi-sampling support")
        self.form.addRow("Archival", self.archival_split_check)

        self.fmt_combo = QComboBox()
        self.fmt_combo.addItems(["TIFF", "DNG"])
        self.fmt_combo.setToolTip(
            "TIFF and DNG are containers for the same scanner-linear capture. "
            "The file format does not control inversion; use Process for that."
        )
        self.form.addRow("Single-frame format", self.fmt_combo)

        self.format_hint = hint_label(
            "TIFF and DNG contain the same scanner-linear capture. File format does not control inversion; use Process for that."
        )
        self.format_hint.setWordWrap(True)
        self.form.addRow("", self.format_hint)

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
        self.pattern_edit.setToolTip(
            'Jinja2 template. Variables: {{ date }}, {{ seq }}, and {{ slot }} for roll scans.\n'
            'The default roll name includes its physical slot automatically.'
        )
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
        self.sa30_compatible_check.setChecked(self._settings.sa30_compatible_roll_feeder)

    def _connect_signals(self) -> None:
        self.refresh_btn.clicked.connect(self._on_refresh)
        self.eject_btn.clicked.connect(self._on_eject)
        self.device_combo.currentIndexChanged.connect(self._on_device_changed)
        self.browse_btn.clicked.connect(self._on_browse)
        self.scan_btn.clicked.connect(self._on_scan)
        self.folder_edit.textChanged.connect(self._on_output_folder_changed)
        self.pattern_edit.textChanged.connect(lambda: self._update_settings_from_ui())
        self.fmt_combo.currentTextChanged.connect(lambda: self._update_settings_from_ui())
        self.dpi_combo.currentTextChanged.connect(lambda: self._update_settings_from_ui())
        self.depth_combo.currentTextChanged.connect(self._on_depth_changed)
        self.ir_check.toggled.connect(lambda: self._update_settings_from_ui())
        self.autofocus_check.toggled.connect(lambda: self._update_settings_from_ui())
        self.ae_check.toggled.connect(lambda: self._update_settings_from_ui())
        self.samples_combo.currentTextChanged.connect(lambda: self._update_settings_from_ui())
        self.archival_split_check.toggled.connect(self._on_archival_split_toggled)
        self.registered_geometry_check.toggled.connect(self._on_registered_geometry_toggled)
        self.load_registration_btn.clicked.connect(self._on_load_registration_json)
        self.sa30_compatible_check.toggled.connect(self._on_sa30_compatible_toggled)
        self.preview_meter_inset_spin.valueChanged.connect(self._on_preview_meter_inset_changed)
        self.preview_non_inverted_negative_check.toggled.connect(self._on_preview_polarity_changed)
        self.roll_preview_btn.clicked.connect(self._on_roll_preview)
        self.roll_stop_btn.clicked.connect(self._on_roll_stop)
        self.retry_remaining_btn.clicked.connect(self._select_roll_retry_remaining)
        self.roll_slot_selector.scan_requested.connect(self._on_roll_scan_selected)
        self.roll_slot_selector.thumbnail_reload_requested.connect(
            self._on_roll_thumbnail_reload_requested
        )

        # Controller signals
        self.controller.scan_devices_ready.connect(self._on_devices_ready)
        self.controller.scan_progress.connect(self._on_scan_progress)
        self.controller.scan_finished.connect(self._on_scan_finished)
        self.controller.scan_cancelled.connect(self._on_scan_cancelled)
        self.controller.scan_error.connect(self._on_scan_error)
        self.controller.scan_ejected.connect(self._on_ejected)
        self.controller.scan_eject_error.connect(self._on_eject_error)
        self.controller.ls5000_roll_preview_ready.connect(self._on_roll_preview_ready)
        self.controller.ls5000_roll_preview_invalidated.connect(self._on_roll_preview_invalidated)
        self.controller.ls5000_roll_thumbnail_ready.connect(self._on_roll_thumbnail_ready)
        self.controller.ls5000_roll_thumbnail_error.connect(
            self._on_roll_thumbnail_error
        )
        self.controller.ls5000_roll_progress.connect(self._on_roll_progress)
        self.controller.ls5000_roll_frame_finished.connect(self._on_roll_frame_finished)
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
        self._set_scanner_status("Detecting scanners…", "active")
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
        self._set_scanner_status("Unavailable · SANE is not installed", "error")

    def _on_refresh(self) -> None:
        if self._scanning or self._roll_scanning or self._roll_thumbnail_reload_pending:
            return
        self._invalidate_roll_preview()
        self._request_devices()

    def _on_eject(self) -> None:
        device = self._current_device()
        if (
            device is None
            or self._scanning
            or self._roll_scanning
            or self._roll_thumbnail_reload_pending
            or self._ejecting
        ):
            return
        self._ejecting = True
        self.device_combo.setEnabled(False)
        self.refresh_btn.setEnabled(False)
        self.eject_btn.setEnabled(False)
        self.status_label.setText("Ejecting film…")
        self._set_scanner_status("Ejecting film…", "active")
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
            self._set_scanner_status("Connected · Film ejected", "ready")
        else:
            self.status_label.setText("This scanner does not expose a film-eject control.")
            self._set_scanner_status("Connected · Eject unavailable", "warning")

    @pyqtSlot(str)
    def _on_eject_error(self, message: str) -> None:
        self._ejecting = False
        self.device_combo.setEnabled(bool(self._devices))
        self.refresh_btn.setEnabled(True)
        device = self._current_device()
        self.eject_btn.setEnabled(bool(device and device.capabilities.can_eject))
        self.status_label.setText(f"Could not eject film: {message}")
        self._set_scanner_status("Error · Eject failed", "error")

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
            self._set_scanner_status("Disconnected · No scanner detected", "error")
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
    def _is_ls5000(device: ScannerDevice | None) -> bool:
        if device is None:
            return False
        normalized_model = device.model.upper().replace(" ", "-")
        return "LS-5000" in normalized_model

    def _can_configure_sa30_profile(
        self,
        device: ScannerDevice | None,
    ) -> bool:
        if not self._is_ls5000(device) or device is None:
            return False
        caps = device.capabilities
        return caps.adapter_frame_control and caps.adapter_frame_capacity is None

    def _effective_roll_capacity(
        self,
        device: ScannerDevice | None,
    ) -> int | None:
        if not self._is_ls5000(device) or device is None:
            return None
        caps = device.capabilities
        if caps.adapter_frame_capacity is not None:
            return 40 if caps.adapter_frame_capacity == 40 else None
        profile_enabled = type(self._settings.sa30_compatible_roll_feeder) is bool and self._settings.sa30_compatible_roll_feeder
        if caps.adapter_frame_control and profile_enabled:
            return 40
        return None

    def _supports_ls5000_roll_workflow(
        self,
        device: ScannerDevice | None,
    ) -> bool:
        return self._effective_roll_capacity(device) == 40

    def _clear_roll_preview(self) -> None:
        """Clear stale thumbnails without sending another worker command."""

        self._roll_preview_token = None
        self._roll_preview_session = None
        self._roll_preview_thumbnails.clear()
        self.roll_slot_selector.set_slots([])
        self.roll_slot_selector.setVisible(False)
        self.roll_preview_btn.setText(" Load Roll Thumbnails")

    def _invalidate_roll_preview(self, *, clear_retry: bool = True) -> None:
        """Clear the UI and invalidate the worker-owned coordinate binding."""

        if clear_retry:
            self._clear_roll_retry_state()
        self._clear_roll_preview()
        self.controller.invalidate_ls5000_roll_preview()

    def _update_device_caps(self) -> None:
        device = self._current_device()
        if device is None:
            if self._devices_loaded:
                self._set_scanner_status("Disconnected · No scanner selected", "muted")
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
            self.sa30_compatible_check.setVisible(False)
            self.sa30_compatible_check.setEnabled(False)
            self.roll_preview_btn.setVisible(False)
            self.roll_quality_widget.setVisible(False)
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
        reload_idle = self._roll_thumbnail_reload_pending is None
        self.eject_btn.setEnabled(caps.can_eject and not self._scanning and not self._roll_scanning and reload_idle and not self._ejecting)
        can_configure_sa30 = self._can_configure_sa30_profile(device)
        self.sa30_compatible_check.blockSignals(True)
        self.sa30_compatible_check.setChecked(self._settings.sa30_compatible_roll_feeder)
        self.sa30_compatible_check.blockSignals(False)
        self.sa30_compatible_check.setVisible(can_configure_sa30)
        self.sa30_compatible_check.setEnabled(can_configure_sa30 and not self._scanning and not self._roll_scanning and reload_idle)
        roll_supported = self._supports_ls5000_roll_workflow(device)
        self.roll_preview_btn.setVisible(roll_supported)
        self.roll_preview_btn.setEnabled(roll_supported and not self._scanning and not self._roll_scanning and reload_idle)
        self.roll_quality_widget.setVisible(roll_supported)
        if not roll_supported:
            self.roll_slot_selector.setVisible(False)
            self.roll_status_label.setVisible(False)

        # If no film sources, show banner
        if not caps.sources:
            self.status_label.setText("This scanner reports no film/transparency sources. NegPy v1 supports film scanning only.")
            self._set_scanner_status("Connected · Film source unavailable", "warning")
            self.scan_btn.setEnabled(False)
        else:
            self.status_label.setText("")
            if can_configure_sa30 and not roll_supported:
                self._set_scanner_status(
                    "Connected · Feeder parked or empty · Confirm 40-slot adapter",
                    "warning",
                )
            elif can_configure_sa30:
                self._set_scanner_status(
                    "Connected · Feeder parked or empty · Preview can try to re-arm",
                    "warning",
                )
            elif self._is_ls5000(device) and caps.adapter_frame_capacity == 40:
                self._set_scanner_status(
                    "Connected · Roll feeder ready · 40 slots",
                    "ready",
                )
            elif self._is_ls5000(device) and caps.adapter_frame_capacity is not None:
                self._set_scanner_status(
                    f"Connected · Feeder ready · {caps.adapter_frame_capacity} slots",
                    "ready",
                )
            else:
                self._set_scanner_status("Connected · Ready", "ready")
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
            best_depth = max(caps.supported_depths)
            for d in caps.supported_depths:
                label = f"{d}-bit"
                if d == best_depth:
                    label += " (Best quality)"
                self.depth_combo.addItem(label, d)
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

        # Archival RGB plus IR recipe. The public label deliberately does not
        # claim an IR sample count: the packed LS-5000 stream transfers one
        # aligned IR plane, but does not reveal how firmware produced it.
        archival_supported = caps.ir_channel and caps.multi_sample and 16 in caps.supported_depths
        self.archival_split_check.setEnabled(archival_supported)
        if archival_supported:
            self.archival_split_check.setChecked(self._settings.archival_split_capture)
            self.archival_split_check.setToolTip(
                "4× hardware-multisampled RGB plus one aligned IR plane; "
                "the scanner's internal IR sampling is not known"
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
        """Lock the controls to the archival RGB 4x plus IR recipe."""
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
        depth_index = self.depth_combo.findData(16)
        if depth_index >= 0:
            self.depth_combo.blockSignals(True)
            self.depth_combo.setCurrentIndex(depth_index)
            self.depth_combo.blockSignals(False)
        self.ir_check.setEnabled(False)
        self.samples_combo.setEnabled(False)
        self.depth_combo.setEnabled(False)

    def _on_archival_split_toggled(self, _checked: bool) -> None:
        if not self.archival_split_check.isChecked():
            # Restore IR/samples to whatever the device's own capabilities allow.
            device = self._current_device()
            caps = device.capabilities if device is not None else None
            self.ir_check.setEnabled(caps.ir_channel if caps is not None else False)
            self.samples_combo.setEnabled(caps.multi_sample if caps is not None else False)
            self.depth_combo.setEnabled(device is not None)
        self._apply_archival_split_interlock()
        self._update_settings_from_ui()

    def _on_depth_changed(self, _text: str) -> None:
        if self.archival_split_check.isChecked() and self.archival_split_check.isEnabled() and self.depth_combo.currentData() != 16:
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

    def _on_output_folder_changed(self, _text: str) -> None:
        self._update_settings_from_ui()
        self._update_roll_storage_estimate()

    def _update_roll_storage_estimate(self) -> None:
        """Read free space from the closest existing output-folder ancestor."""

        raw_folder = self.folder_edit.text().strip()
        if not raw_folder:
            self.roll_slot_selector.set_available_storage_bytes(None)
            return
        candidate = Path(raw_folder).expanduser()
        while not candidate.exists() and candidate.parent != candidate:
            candidate = candidate.parent
        if candidate.is_file():
            candidate = candidate.parent
        try:
            available_bytes = shutil.disk_usage(candidate).free
        except OSError:
            available_bytes = None
        self.roll_slot_selector.set_available_storage_bytes(available_bytes)

    def _on_sa30_compatible_toggled(self, _checked: bool) -> None:
        device = self._current_device()
        if not self._can_configure_sa30_profile(device):
            self.sa30_compatible_check.blockSignals(True)
            self.sa30_compatible_check.setChecked(self._settings.sa30_compatible_roll_feeder)
            self.sa30_compatible_check.blockSignals(False)
            return
        self._update_settings_from_ui()
        self._invalidate_roll_preview()
        self._update_device_caps()

    def _on_roll_preview(self) -> None:
        if (
            self._scanning
            or self._roll_scanning
            or self._roll_thumbnail_reload_pending is not None
        ):
            return
        device = self._current_device()
        adapter_frame_capacity = self._effective_roll_capacity(device)
        if adapter_frame_capacity is None or device is None:
            self.roll_status_label.setText("Whole-roll thumbnails require a Nikon LS-5000 roll feeder.")
            self.roll_status_label.setVisible(True)
            return
        attempts_root = Path(APP_CONFIG.cache_dir) / "ls5000-roll" / "preview-attempts"
        self._invalidate_roll_preview(clear_retry=False)
        self._set_roll_scanning(True, operation=_ROLL_OPERATION_PREVIEW)
        self.roll_status_label.setText("Reading the whole roll for thumbnails…")
        self.roll_status_label.setVisible(True)
        self._set_scanner_status("Making previews · Reading the whole roll", "active")
        self.controller.start_ls5000_roll_preview(
            RollPreviewRequest(
                attempts_root=str(attempts_root),
                device_id=device.id,
                adapter_frame_capacity=adapter_frame_capacity,
            )
        )

    def _on_roll_stop(self) -> None:
        if not self._roll_scanning or self._roll_operation != _ROLL_OPERATION_SELECTED_SCAN:
            return
        self.controller.cancel_scan()
        self.roll_stop_btn.setEnabled(False)
        self.roll_status_label.setText("Stopping safely after the active scanner transaction…")
        self._set_scanner_status("Stopping safely after the current frame…", "warning")

    def _on_roll_scan_selected(self, slot_ids: list[int]) -> None:
        if (
            self._scanning
            or self._roll_scanning
            or self._roll_thumbnail_reload_pending is not None
            or not slot_ids
        ):
            return
        device = self._current_device()
        adapter_frame_capacity = self._effective_roll_capacity(device)
        if adapter_frame_capacity is None or device is None:
            return
        if self._roll_preview_token is None:
            self.roll_status_label.setText("Load a new whole-roll preview before scanning selected frames.")
            self.roll_status_label.setVisible(True)
            return
        if self._roll_preview_session is None:
            self.roll_status_label.setText(
                "The reviewed roll evidence is unavailable. Load new roll thumbnails before scanning."
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
        approved_slots = set(
            self.roll_slot_selector.approved_blocking_slot_ids()
        ).intersection(slot_ids)
        try:
            reviewed_fingerprint = self._roll_preview_session.reviewed_fingerprint()
            approvals = {
                slot_id: self._roll_preview_session.approve_manual_origin(
                    slot_id,
                    offsets[slot_id],
                )
                for slot_id in approved_slots
            }
            choices = frame_choices(slot_ids, offsets, approvals=approvals)
        except Exception as error:
            self.roll_status_label.setText(
                f"Could not bind reviewed thumbnails to this scan: {error}"
            )
            self.roll_status_label.setVisible(True)
            return
        attempts_root = Path(output_folder) / ".negpy-ls5000" / "attempts"
        filename_pattern = self.pattern_edit.text().strip()
        if not filename_pattern or filename_pattern == ScannerSettings.defaults().filename_pattern:
            filename_pattern = _ROLL_DEFAULT_FILENAME_PATTERN
        request = RollScanRequest(
            device_id=device.id,
            adapter_frame_capacity=adapter_frame_capacity,
            preview_token=self._roll_preview_token,
            attempts_root=str(attempts_root),
            output_folder=output_folder,
            filename_pattern=filename_pattern,
            material=material,
            frames=choices,
            reviewed_fingerprint=reviewed_fingerprint,
        )
        self._roll_batch_selected_slots = tuple(slot_ids)
        self._roll_completed_slots.clear()
        self._roll_batch_fingerprint = reviewed_fingerprint
        self._roll_batch_material = material
        self._roll_retry_pending_slots = ()
        self._roll_retry_fingerprint = None
        self._roll_retry_material = None
        self._roll_batch_in_progress = True
        self._sync_roll_retry_ui()
        self._update_settings_from_ui()
        self._save_settings()
        self._set_roll_scanning(True, operation=_ROLL_OPERATION_SELECTED_SCAN)
        self.controller.start_ls5000_roll_scan(request)

    def _roll_thumbnail_qimage(self, thumbnail: object) -> QImage:
        """Make a small display image from scanner-linear negative RGB."""

        display = render_roll_thumbnail_rgb8(
            thumbnail,
            meter_inset_percent=self.preview_meter_inset_spin.value(),
            inverted=not self.preview_non_inverted_negative_check.isChecked(),
        )
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
            raw_thumbnails = {slot.slot_id: slot.thumbnail for slot in session.slots}
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
        self._roll_preview_session = session
        self._roll_preview_thumbnails = raw_thumbnails
        self.roll_slot_selector.set_slots(slots)
        self.roll_slot_selector.setVisible(True)
        self.roll_preview_btn.setText(" Reload Roll Thumbnails")
        status = (
            f"Loaded {len(slots)} scanner slots. Adjust Film Spacing Offset and "
            "reload any thumbnail that does not show the complete negative. "
            "If the film is ejected or reinserted, reload the whole roll first."
        )
        if self._roll_retry_pending_slots:
            if self._roll_retry_matches_current_preview():
                if self._roll_retry_material is not None:
                    self.roll_slot_selector.set_scan_material(
                        self._roll_retry_material
                    )
                status += " The unfinished selection is ready to restore."
            else:
                self._clear_roll_retry_state()
                status += (
                    " The previous unfinished selection was cleared because "
                    "this preview appears to be a different roll."
                )
        self.roll_status_label.setText(status)
        self._sync_roll_retry_ui()
        self._set_roll_scanning(False)
        self._set_scanner_status(f"Connected · {len(slots)} previews ready", "ready")

    @pyqtSlot()
    def _on_roll_preview_invalidated(self) -> None:
        if self._roll_batch_in_progress:
            self._remember_remaining_roll_slots()
        self._clear_roll_preview()
        self._sync_roll_retry_ui()

    @pyqtSlot(int, int)
    def _on_roll_thumbnail_reload_requested(
        self,
        slot_id: int,
        boundary_offset: int,
    ) -> None:
        """Start one identified offline reload while locking scanner controls."""

        if (
            self._scanning
            or self._roll_scanning
            or self._ejecting
            or self._roll_thumbnail_reload_pending is not None
        ):
            return
        self._roll_thumbnail_reload_pending = (slot_id, boundary_offset)
        self._sync_roll_thumbnail_reload_controls()
        self.roll_status_label.setText(
            f"Reloading slot {slot_id:02d} at Film Spacing Offset "
            f"{boundary_offset:+d}…"
        )
        self.roll_status_label.setVisible(True)
        try:
            self.controller.reload_ls5000_roll_thumbnail(
                slot_id,
                boundary_offset,
            )
        except BaseException:
            self._roll_thumbnail_reload_pending = None
            self._sync_roll_thumbnail_reload_controls()
            raise

    @pyqtSlot(int, int, object)
    def _on_roll_thumbnail_ready(
        self,
        slot_id: int,
        boundary_offset: int,
        thumbnail: object,
    ) -> None:
        if self._roll_thumbnail_reload_pending != (slot_id, boundary_offset):
            return
        try:
            display = self._roll_thumbnail_qimage(thumbnail)
            self.roll_slot_selector.confirm_slot_thumbnail(
                slot_id,
                boundary_offset,
                display,
            )
            self._roll_preview_thumbnails[slot_id] = thumbnail
            self.roll_status_label.setText(f"Reloaded slot {slot_id:02d} at Film Spacing Offset {boundary_offset:+d}.")
            count = self.roll_slot_selector.model.rowCount()
            self._set_scanner_status(f"Connected · {count} previews ready", "ready")
        except Exception as error:
            self.roll_status_label.setText(f"Could not display reloaded slot {slot_id:02d}: {error}")
            self._set_scanner_status("Error · Preview adjustment failed", "error")
        finally:
            self._roll_thumbnail_reload_pending = None
            self._sync_roll_thumbnail_reload_controls()

    @pyqtSlot(object)
    def _on_roll_thumbnail_error(
        self,
        failure: RollThumbnailReloadFailure,
    ) -> None:
        if not isinstance(failure, RollThumbnailReloadFailure):
            return
        identity = (failure.slot_id, failure.boundary_offset_rows)
        if self._roll_thumbnail_reload_pending != identity:
            return
        self._roll_thumbnail_reload_pending = None
        self._sync_roll_thumbnail_reload_controls()
        self.roll_status_label.setText(failure.message)
        self.roll_status_label.setVisible(True)
        self._set_scanner_status("Error · Preview adjustment failed", "error")

    @pyqtSlot(int)
    def _on_preview_meter_inset_changed(self, _value: int) -> None:
        """Persist and rerender the retained contact sheet without scanner I/O."""

        self.settings = replace(
            self._settings,
            preview_meter_inset_percent=self.preview_meter_inset_spin.value(),
        )
        self.roll_slot_selector.set_meter_inset_percent(self.preview_meter_inset_spin.value())
        self._rerender_roll_thumbnails()

    @pyqtSlot(bool)
    def _on_preview_polarity_changed(self, _checked: bool) -> None:
        """Switch contact-sheet polarity offline without changing scan data."""

        self.settings = replace(
            self._settings,
            preview_show_non_inverted_negative=self.preview_non_inverted_negative_check.isChecked(),
        )
        self._rerender_roll_thumbnails()

    def _rerender_roll_thumbnails(self) -> None:
        """Rebuild every retained thumbnail from the saved scanner preview."""

        try:
            rendered = {slot_id: self._roll_thumbnail_qimage(thumbnail) for slot_id, thumbnail in self._roll_preview_thumbnails.items()}
            for slot_id, image in rendered.items():
                self.roll_slot_selector.update_slot_thumbnail(slot_id, image)
        except Exception as error:
            self.roll_status_label.setText(f"Could not update preview display: {error}")
            self.roll_status_label.setVisible(True)
            self._set_scanner_status("Error · Preview display update failed", "error")

    @pyqtSlot(object)
    def _on_roll_progress(self, progress: object) -> None:
        total = max(1, int(progress.total))
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(int(100 * int(progress.completed) / total))
        self.progress_bar.setFormat(str(progress.message))
        self.roll_status_label.setText(str(progress.message))
        self.roll_status_label.setVisible(True)
        message = str(progress.message)
        operation = getattr(progress, "operation", None)
        preview_complete = operation is RollOperation.PREVIEW and int(progress.completed) >= total
        state = "ready" if preview_complete else "active"
        self._set_scanner_status(message, state)

    @pyqtSlot(int, str)
    def _on_roll_frame_finished(self, slot_id: int, _rgb_path: str) -> None:
        if self._roll_batch_in_progress and slot_id in self._roll_batch_selected_slots:
            self._roll_completed_slots.add(slot_id)
        self._update_roll_storage_estimate()

    @pyqtSlot(object)
    def _on_roll_finished(self, completion: RollScanCompletion) -> None:
        operation = getattr(completion, "operation", RollOperation.FULL_SCAN)
        if operation is RollOperation.FULL_SCAN and self._roll_batch_in_progress:
            if completion.stopped:
                self._remember_remaining_roll_slots()
            else:
                self._clear_roll_retry_state()
        self._set_roll_scanning(False)
        count = len(completion.rgb_paths)
        if completion.stopped and operation is RollOperation.PREVIEW:
            self.roll_status_label.setText("Whole-roll preview stopped before completion. No thumbnails were loaded.")
            self._set_scanner_status(
                "Preview stopped · No thumbnails loaded",
                "warning",
            )
        elif completion.stopped:
            self.roll_status_label.setText(f"Stopped safely after completing {count} frame(s).")
            self._set_scanner_status(f"Stopped safely · {count} frames ready", "warning")
        else:
            self.roll_status_label.setText(f"Finished {count} full-quality frame(s); added them to NegPy.")
            noun = "frame" if count == 1 else "frames"
            self._set_scanner_status(f"Complete · {count} {noun} ready", "ready")

    @pyqtSlot(object)
    def _on_roll_error(self, failure: RollWorkerFailure) -> None:
        if self._roll_batch_in_progress:
            self._remember_remaining_roll_slots()
        self._set_roll_scanning(False)
        if failure.recovery_required:
            self._clear_roll_preview()
        suffix = " Power-cycle the scanner before trying again." if failure.recovery_required else ""
        self.roll_status_label.setText(f"{failure.message}{suffix}")
        self.roll_status_label.setVisible(True)
        if failure.recovery_required:
            self._set_scanner_status("Error · Power-cycle required", "error")
        else:
            self._set_scanner_status("Error · Check scan details", "error")

    def _remember_remaining_roll_slots(self) -> None:
        """Retain unfinished physical slots from the active batch only."""

        self._roll_retry_pending_slots = tuple(
            slot_id
            for slot_id in self._roll_batch_selected_slots
            if slot_id not in self._roll_completed_slots
        )
        if self._roll_retry_pending_slots:
            self._roll_retry_fingerprint = self._roll_batch_fingerprint
            self._roll_retry_material = self._roll_batch_material
        else:
            self._roll_retry_fingerprint = None
            self._roll_retry_material = None
        self._roll_batch_in_progress = False
        self._sync_roll_retry_ui()

    def _clear_roll_retry_state(self) -> None:
        self._roll_batch_selected_slots = ()
        self._roll_completed_slots.clear()
        self._roll_batch_fingerprint = None
        self._roll_batch_material = None
        self._roll_retry_pending_slots = ()
        self._roll_retry_fingerprint = None
        self._roll_retry_material = None
        self._roll_batch_in_progress = False
        self._sync_roll_retry_ui()

    def _roll_retry_matches_current_preview(self) -> bool:
        """Return whether the loaded preview is the physical roll being retried."""

        if (
            self._roll_preview_session is None
            or self._roll_retry_fingerprint is None
            or self._roll_retry_material is None
        ):
            return False
        try:
            fresh_fingerprint = self._roll_preview_session.reviewed_fingerprint()
            comparison = compare_reviewed_roll_fingerprints(
                self._roll_retry_fingerprint,
                fresh_fingerprint,
            )
        except (AttributeError, TypeError, ValueError):
            return False
        return comparison.matches

    def _sync_roll_retry_ui(self) -> None:
        if not hasattr(self, "retry_remaining_btn"):
            return
        count = len(self._roll_retry_pending_slots)
        self.retry_remaining_btn.setVisible(count > 0)
        if count == 0:
            self.retry_remaining_btn.setEnabled(False)
            self.retry_remaining_btn.setText("Select remaining")
            return
        preview_ready = (
            self._roll_preview_token is not None
            and self._roll_retry_matches_current_preview()
        )
        self.retry_remaining_btn.setEnabled(
            preview_ready
            and not self._scanning
            and not self._roll_scanning
            and self._roll_thumbnail_reload_pending is None
        )
        if preview_ready:
            self.retry_remaining_btn.setText(f"Select remaining ({count})")
        else:
            self.retry_remaining_btn.setText(
                f"Reload thumbnails to select remaining ({count})"
            )

    def _select_roll_retry_remaining(self) -> None:
        if self._roll_preview_token is None or self._roll_scanning or self._scanning:
            return
        if not self._roll_retry_matches_current_preview():
            self._clear_roll_retry_state()
            self.roll_status_label.setText(
                "Could not restore unfinished slots because this is not the reviewed roll."
            )
            self.roll_status_label.setVisible(True)
            return
        if self._roll_retry_material is not None:
            self.roll_slot_selector.set_scan_material(self._roll_retry_material)
        available = set(range(1, self.roll_slot_selector.model.rowCount() + 1))
        remaining = [
            slot_id
            for slot_id in self._roll_retry_pending_slots
            if slot_id in available
        ]
        if not remaining:
            return
        self.roll_slot_selector.set_selected_slot_ids(remaining)
        self.roll_status_label.setText(
            f"Selected {len(remaining)} unfinished slots. Review their alignment, then scan selected."
        )
        self.roll_status_label.setVisible(True)
        self.roll_retry_selection_restored.emit()

    def _set_roll_scanning(
        self,
        active: bool,
        *,
        operation: str | None = None,
    ) -> None:
        if active:
            if operation not in {
                _ROLL_OPERATION_PREVIEW,
                _ROLL_OPERATION_SELECTED_SCAN,
            }:
                raise ValueError("an active roll operation must identify preview or selected-scan")
            self._roll_operation = operation
        else:
            self._roll_operation = None
        self._roll_scanning = active
        self._sync_roll_thumbnail_reload_controls()
        can_stop_after_frame = active and self._roll_operation == _ROLL_OPERATION_SELECTED_SCAN
        self.roll_stop_btn.setVisible(can_stop_after_frame)
        self.roll_stop_btn.setEnabled(can_stop_after_frame)
        self.progress_bar.setVisible(active)
        if active:
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(0)
        else:
            self.progress_bar.setVisible(False)
            self.progress_bar.setFormat("Scanning… %p%")
        self._sync_roll_retry_ui()

    def _sync_roll_thumbnail_reload_controls(self) -> None:
        """Serialize an offline thumbnail job against every scanner command."""

        device = self._current_device()
        blocked = (
            self._roll_scanning
            or self._roll_thumbnail_reload_pending is not None
        )
        controls_available = not self._scanning and not blocked
        self.device_combo.setEnabled(controls_available and bool(self._devices))
        self.refresh_btn.setEnabled(controls_available)
        self.sa30_compatible_check.setEnabled(
            controls_available and self._can_configure_sa30_profile(device)
        )
        self.roll_preview_btn.setEnabled(
            controls_available and self._supports_ls5000_roll_workflow(device)
        )
        self.roll_slot_selector.setEnabled(controls_available)
        if not self._scanning:
            self.scan_btn.setEnabled(controls_available and device is not None)
        self.eject_btn.setEnabled(
            controls_available
            and not self._ejecting
            and bool(device and device.capabilities.can_eject)
        )
        self._sync_roll_retry_ui()

    def _on_scan(self) -> None:
        if self._roll_scanning or self._roll_thumbnail_reload_pending is not None:
            return
        if self._scanning:
            # Cancel
            if self._scan_cancel_pending:
                return
            self._scan_cancel_pending = True
            self.scan_btn.setEnabled(False)
            self.status_label.setText("Stopping the current frame…")
            self._set_scanner_status("Stopping the current frame…", "warning")
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
            self._set_scanner_status("Error · Scan could not start", "error")
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
        if not self._scanning:
            return
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(int(progress * 100))
        if self._scan_cancel_pending:
            return
        self._set_scanner_status(self._single_frame_progress_text(progress), "active")

    @pyqtSlot(str)
    def _on_scan_finished(self, path: str) -> None:
        self.set_scanning(False)
        self.status_label.setText(f"Scanned: {path}")
        self._set_scanner_status("Complete · 1 frame ready", "ready")

    @pyqtSlot()
    def _on_scan_cancelled(self) -> None:
        """Restore conventional scan controls without disturbing a roll job."""

        if not self._scanning:
            return
        self.set_scanning(False)
        self.status_label.setText("Scan stopped.")
        self._set_scanner_status("Connected · Scan stopped", "ready")

    @pyqtSlot(str)
    def _on_scan_error(self, msg: str) -> None:
        self.set_scanning(False)
        self.status_label.setText(f"Error: {msg}")
        self._set_scanner_status("Error · Scan failed", "error")

    # ── state helpers ─────────────────────────────────────────────────

    def set_scanning(self, active: bool) -> None:
        was_scanning = self._scanning
        if active and not was_scanning:
            device = self._current_device()
            frame = self.frame_spin.value()
            self._active_scan_device_id = device.id if device is not None else None
            self._active_scan_frame = frame if frame > 0 else None

        self._scanning = active
        self._scan_cancel_pending = False
        device = self._current_device()
        reload_idle = self._roll_thumbnail_reload_pending is None
        self._set_conventional_request_controls_enabled(not active and reload_idle, device)
        self.eject_btn.setEnabled(not active and reload_idle and not self._ejecting and bool(device and device.capabilities.can_eject))
        roll_supported = self._supports_ls5000_roll_workflow(device)
        self.sa30_compatible_check.setEnabled(not active and reload_idle and not self._roll_scanning and self._can_configure_sa30_profile(device))
        self.roll_preview_btn.setEnabled(not active and reload_idle and not self._roll_scanning and roll_supported)
        self.roll_slot_selector.setEnabled(not active and reload_idle and not self._roll_scanning)
        if active:
            self.scan_btn.setEnabled(True)
            self.scan_btn.setText(" Stop")
            self.scan_btn.setIcon(qta.icon("fa5s.stop", color=THEME.text_primary))
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            self._set_scanner_status(self._single_frame_progress_text(), "active")
        else:
            self.scan_btn.setEnabled(reload_idle and not self._roll_scanning and device is not None)
            self.scan_btn.setText(" Scan")
            self.scan_btn.setIcon(qta.icon("fa5s.camera-retro", color=THEME.text_primary))
            self.progress_bar.setVisible(False)
            self.progress_bar.setFormat("Scanning… %p%")

        if not active:
            self._active_scan_device_id = None
            self._active_scan_frame = None

    def _set_conventional_request_controls_enabled(
        self,
        enabled: bool,
        device: ScannerDevice | None,
    ) -> None:
        """Lock a conventional request while its scanner transaction is active."""

        self.device_combo.setEnabled(enabled and bool(self._devices))
        self.refresh_btn.setEnabled(enabled)

        request_widgets = (
            self.dpi_combo,
            self.depth_combo,
            self.ir_check,
            self.frame_spin,
            self.autofocus_check,
            self.ae_check,
            self.samples_combo,
            self.archival_split_check,
            self.fmt_combo,
            self.folder_edit,
            self.browse_btn,
            self.pattern_edit,
            self.registered_geometry_check,
            self.subframe_spin,
            self.br_y_spin,
            self.load_registration_btn,
        )
        if not enabled:
            for widget in request_widgets:
                widget.setEnabled(False)
            return

        caps = device.capabilities if device is not None else None
        self.dpi_combo.setEnabled(device is not None)
        self.depth_combo.setEnabled(device is not None)
        self.frame_spin.setEnabled(caps is not None and caps.adapter_frame_capacity is not None)
        self.autofocus_check.setEnabled(device is not None)
        self.ae_check.setEnabled(caps is not None and caps.auto_exposure)
        self.archival_split_check.setEnabled(caps is not None and caps.ir_channel and caps.multi_sample and 16 in caps.supported_depths)
        self.ir_check.setEnabled(caps is not None and caps.ir_channel)
        self.samples_combo.setEnabled(caps is not None and caps.multi_sample)
        self.registered_geometry_check.setEnabled(caps is not None and caps.registered_geometry)

        self.fmt_combo.setEnabled(True)
        self.folder_edit.setEnabled(True)
        self.browse_btn.setEnabled(True)
        self.pattern_edit.setEnabled(True)

        self._apply_archival_split_interlock()
        self._update_registration_fields_enabled()

    def _single_frame_progress_text(self, progress: float | None = None) -> str:
        frame = self._active_scan_frame if self._scanning else self.frame_spin.value()
        frame = frame or 0
        location = f"Slot {frame:02d}" if frame > 0 else "Current position"
        message = f"Scanning frame 1 of 1 · {location}"
        if progress is not None:
            percent = max(0, min(100, round(float(progress) * 100)))
            message += f" · {percent}%"
        return message

    def _set_scanner_status(self, message: str, state: str) -> None:
        """Keep scanner connection and operation state visible near its name."""

        color = _SCANNER_STATUS_COLORS.get(state, _SCANNER_STATUS_COLORS["muted"])
        self.device_status_label.setProperty("scannerState", state)
        self.device_status_label.setText(f"● {message}")
        self.device_status_label.setAccessibleName(f"Scanner status: {message}")
        self.device_status_label.setAccessibleDescription(message)
        self.device_status_label.setStyleSheet(f"color: {color}; font-size: {THEME.font_size_small}px;")
        self.device_status_label.setToolTip(
            f"{message}. Connection is updated when scanners are detected, refreshed, or an operation reports a failure."
        )

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
            sa30_compatible_roll_feeder=(self.sa30_compatible_check.isChecked()),
            preview_meter_inset_percent=self.preview_meter_inset_spin.value(),
            preview_show_non_inverted_negative=self.preview_non_inverted_negative_check.isChecked(),
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
