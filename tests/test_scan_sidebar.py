"""Offline tests for the Scan sidebar's archival-recipe controls.

Constructs the real ScanSidebar + AppController under QT_QPA_PLATFORM=offscreen
(set globally by tests/conftest.py) with a mocked DesktopSessionManager and a
fabricated ScannerDevice/ScannerCapabilities -- no live SANE device is ever
opened. This proves the new controls (frame selection, hardware AE, the
RGB4x+IR1x archival split-capture toggle, registered geometry) instantiate,
capability-gate, and wire their Qt signals correctly.

It does NOT and cannot verify that the rendered UI looks/behaves right on a
real device -- that remains an unverified gap requiring a live app + hardware.
"""

import gc
import hashlib
import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
from PyQt6.QtCore import QObject, Qt, pyqtSignal
from PyQt6.QtWidgets import QApplication, QFormLayout

from negpy.desktop.controller import AppController
from negpy.desktop.session import DesktopSessionManager, AppState
from negpy.desktop.view.sidebar.scan import ScanSidebar
from negpy.desktop.view.widgets.roll_slot_model import RollSlotModel
from negpy.desktop.workers import ls5000_roll_worker as roll_worker_module
from negpy.desktop.workers.ls5000_roll_worker import (
    RollOperation,
    RollPreviewRequest,
    RollScanRequest,
)
from negpy.infrastructure.scanners.ls5000_single_pass.capture_process import (
    ManualFrameApproval,
    ReviewedRollFingerprint,
)
from negpy.infrastructure.scanners.base import ScannerCapabilities, ScannerDevice
from negpy.infrastructure.scanners.params import ScanMode
from negpy.services.rendering.preview_manager import PreviewManager
from negpy.services.scanning.roll_preview_controls import ScanMaterial

if not QApplication.instance():
    _app = QApplication(sys.argv)


# A device exposing every capability the new controls gate on: frame
# selection, hardware auto-exposure, registered geometry, and both IR +
# multi-sampling (needed together for the archival split-capture toggle).
FULL_CAPS = ScannerCapabilities(
    ir_channel=True,
    supported_dpi=(1000, 4000),
    supported_depths=(8, 16),
    sources=(ScanMode.NEGATIVE,),
    max_area_mm=(36.0, 24.0),
    multi_sample=True,
    adapter_frame_capacity=40,
    auto_exposure=True,
    registered_geometry=True,
    can_eject=True,
    adapter_frame_control=True,
)
FULL_DEVICE = ScannerDevice(id="coolscan3:usb:libusb:001:007", vendor="Nikon", model="LS-5000", capabilities=FULL_CAPS)

# A device with none of the archival extras (only the baseline fields every
# existing control already handled).
MINIMAL_CAPS = ScannerCapabilities(
    ir_channel=False,
    supported_dpi=(1200, 2400),
    supported_depths=(8, 16),
    sources=(ScanMode.NEGATIVE,),
    max_area_mm=(36.0, 24.0),
)
MINIMAL_DEVICE = ScannerDevice(id="plustek:libusb:001:008", vendor="Plustek", model="OpticFilm", capabilities=MINIMAL_CAPS)

# Still an LS-5000 roll feeder, but missing registered geometry. Color scans
# use the packaged single-pass worker and remain available; conventional B&W
# must fail closed because its RGB-only route depends on patched SANE options.
LS5000_WITHOUT_BW_ROUTE_CAPS = ScannerCapabilities(
    ir_channel=True,
    supported_dpi=(1000, 4000),
    supported_depths=(16,),
    sources=(ScanMode.NEGATIVE,),
    max_area_mm=(36.0, 24.0),
    multi_sample=True,
    adapter_frame_capacity=40,
    auto_exposure=True,
    registered_geometry=False,
)
LS5000_WITHOUT_BW_ROUTE = ScannerDevice(
    id="coolscan3:usb:libusb:001:009",
    vendor="Nikon",
    model="LS-5000",
    capabilities=LS5000_WITHOUT_BW_ROUTE_CAPS,
)

STOCK_SA21_CAPS = ScannerCapabilities(
    ir_channel=True,
    supported_dpi=(1000, 4000),
    supported_depths=(16,),
    sources=(ScanMode.NEGATIVE,),
    max_area_mm=(36.0, 24.0),
    multi_sample=True,
    adapter_frame_capacity=6,
    auto_exposure=True,
    registered_geometry=True,
    adapter_frame_control=True,
)
STOCK_SA21_DEVICE = ScannerDevice(
    id="coolscan3:usb:libusb:001:011",
    vendor="Nikon",
    model="LS-5000",
    capabilities=STOCK_SA21_CAPS,
)

PARKED_SA30_CAPS = replace(FULL_CAPS, adapter_frame_capacity=None)
PARKED_SA30_DEVICE = ScannerDevice(
    id="coolscan3:usb:libusb:001:012",
    vendor="Nikon",
    model="LS-5000",
    capabilities=PARKED_SA30_CAPS,
)


def _roll_preview_session(
    count: int = 40,
    *,
    inferred_slots: frozenset[int] = frozenset(),
    source_marker: str = "a",
    visual_xor: int = 0,
):
    """Build a tiny decoded roll index with no scanner or filesystem I/O."""

    thumbnail = np.arange(6 * 8 * 3, dtype=np.uint16).reshape(6, 8, 3)
    fingerprint = ReviewedRollFingerprint(
        source_preview_sha256=source_marker * 64,
        source_table_sha256=("b" if source_marker == "a" else "d") * 64,
        preview_shape=(count * 6, 8, 3),
        frame_start_rows=tuple(range(0, count * 6, 6)),
        frame_native_origins=tuple(range(1_000, 1_000 + count)),
        frame_visual_hashes=tuple(
            f"{int(hashlib.sha256(f'roll-slot-{slot_id}'.encode()).hexdigest(), 16) ^ visual_xor:064x}"
            for slot_id in range(1, count + 1)
        ),
        frame_visual_log_spans=tuple(2.0 for _slot_id in range(1, count + 1)),
    )

    def approve_manual_origin(
        slot_id: int,
        boundary_offset_rows: int = 0,
    ) -> ManualFrameApproval:
        return ManualFrameApproval(
            reviewed_fingerprint_sha256=fingerprint.binding_sha256,
            slot=slot_id,
            boundary_offset_rows=boundary_offset_rows,
            thumbnail_sha256=f"{slot_id:064x}",
            reviewed_lookup_row=slot_id,
            reviewed_native_origin=1_000 + slot_id,
            review_reasons=("transport-origin-inferred",),
        )

    return SimpleNamespace(
        slots=tuple(
            SimpleNamespace(
                slot_id=slot_id,
                thumbnail=thumbnail,
                warnings=(
                    ("transport-origin-inferred",)
                    if slot_id in inferred_slots
                    else ("Likely blank tail slot",)
                    if slot_id == count
                    else ()
                ),
                boundary_offset_rows=0,
            )
            for slot_id in range(1, count + 1)
        ),
        reviewed_fingerprint=lambda: fingerprint,
        approve_manual_origin=approve_manual_origin,
    )


def _build_controller() -> AppController:
    mock_session_manager = MagicMock(spec=DesktopSessionManager)
    mock_session_manager.state = AppState()
    mock_session_manager.repo = MagicMock()

    with (
        patch("negpy.desktop.controller.RenderWorker") as mock_rw_class,
        patch("negpy.desktop.controller.PreviewManager") as mock_pm_class,
    ):
        mock_rw_class.return_value = MagicMock()
        mock_pm_class.return_value = MagicMock(spec=PreviewManager)
        mock_pm_class.return_value.load_linear_preview.return_value = (None, (0, 0), {})
        controller = AppController(mock_session_manager)
    return controller


def _stop_threads(controller: AppController) -> None:
    for thread in [
        controller.render_thread,
        controller.export_thread,
        controller.thumb_thread,
        controller.norm_thread,
        controller.discovery_thread,
        controller.preview_load_thread,
        controller.scan_thread,
    ]:
        if thread is not None and thread.isRunning():
            thread.quit()
            thread.wait()
    del controller
    gc.collect()


def _select_device(sidebar: ScanSidebar, device: ScannerDevice) -> None:
    """Populate the sidebar as if devices_ready delivered exactly `device`,
    without going through the (mocked-away) scan worker thread."""
    sidebar._devices = [device]
    sidebar.device_combo.blockSignals(True)
    sidebar.device_combo.clear()
    sidebar.device_combo.addItem(device.model, device.id)
    sidebar.device_combo.setCurrentIndex(0)
    sidebar.device_combo.blockSignals(False)
    sidebar._update_device_caps()


class _LightweightScanController(QObject):
    """Signal-compatible controller for pure ScanSidebar state tests.

    Unlike AppController, this fixture owns no worker threads or GPU-backed
    services. Tests that exercise controller orchestration keep using the real
    integration fixture below.
    """

    scan_devices_ready = pyqtSignal(list)
    scan_progress = pyqtSignal(float)
    scan_finished = pyqtSignal(str)
    scan_cancelled = pyqtSignal()
    scan_error = pyqtSignal(str)
    scan_ejected = pyqtSignal(bool)
    scan_eject_error = pyqtSignal(str)
    ls5000_roll_preview_ready = pyqtSignal(str, object)
    ls5000_roll_preview_invalidated = pyqtSignal()
    ls5000_roll_thumbnail_ready = pyqtSignal(int, int, object)
    ls5000_roll_thumbnail_error = pyqtSignal(object)
    ls5000_roll_progress = pyqtSignal(object)
    ls5000_roll_frame_finished = pyqtSignal(int, str)
    ls5000_roll_finished = pyqtSignal(object)
    ls5000_roll_error = pyqtSignal(object)

    def __init__(self) -> None:
        super().__init__()
        repo = MagicMock()
        repo.get_global_setting.return_value = {}
        self.session = SimpleNamespace(repo=repo)
        self.preview_requests: list[RollPreviewRequest] = []
        self.scan_requests: list[RollScanRequest] = []
        self.preview_invalidations = 0
        self.reload_requests: list[tuple[int, int]] = []
        self.cancel_calls = 0

    def reload_ls5000_roll_thumbnail(self, slot_id: int, offset: int) -> None:
        self.reload_requests.append((slot_id, offset))

    def invalidate_ls5000_roll_preview(self) -> None:
        self.preview_invalidations += 1

    def start_ls5000_roll_preview(self, request: RollPreviewRequest) -> None:
        self.preview_requests.append(request)

    def start_ls5000_roll_scan(self, request: RollScanRequest) -> None:
        self.scan_requests.append(request)

    def cancel_scan(self) -> None:
        self.cancel_calls += 1


class LightweightScanSidebarTestCase(unittest.TestCase):
    """Real sidebar with signals, but no AppController threads or GPU state."""

    def setUp(self):
        self.controller = _LightweightScanController()
        self.sidebar = ScanSidebar(self.controller)

    def tearDown(self):
        self.sidebar.close()
        self.sidebar.deleteLater()
        self.controller.deleteLater()
        app = QApplication.instance()
        if app is not None:
            app.processEvents()


class TestLightweightParkedRollFeeder(LightweightScanSidebarTestCase):
    def test_unknown_capacity_is_blocked_until_user_confirms_40_slot_adapter(self):
        _select_device(self.sidebar, PARKED_SA30_DEVICE)

        self.assertFalse(self.sidebar.sa30_compatible_check.isHidden())
        self.assertFalse(self.sidebar.sa30_compatible_check.isChecked())
        self.assertTrue(self.sidebar.roll_preview_btn.isHidden())
        self.assertTrue(self.sidebar.roll_quality_widget.isHidden())
        self.assertEqual(
            self.sidebar.device_status_label.text(),
            "● Connected · Feeder parked or empty · Confirm 40-slot adapter",
        )
        self.assertEqual(
            self.sidebar.device_status_label.property("scannerState"),
            "warning",
        )

    def test_opt_in_persists_and_preview_request_uses_effective_40_slots(self):
        _select_device(self.sidebar, PARKED_SA30_DEVICE)
        repo = self.controller.session.repo
        repo.save_global_setting.reset_mock()

        self.sidebar.sa30_compatible_check.setChecked(True)

        self.assertTrue(self.sidebar.settings.sa30_compatible_roll_feeder)
        repo.save_global_setting.assert_called()
        persisted = repo.save_global_setting.call_args.args[1]
        self.assertTrue(persisted["sa30_compatible_roll_feeder"])
        self.assertFalse(self.sidebar.roll_preview_btn.isHidden())
        self.assertEqual(
            self.sidebar.device_status_label.text(),
            "● Connected · Feeder parked or empty · Preview can try to re-arm",
        )

        self.sidebar._on_roll_preview()

        self.assertEqual(len(self.controller.preview_requests), 1)
        self.assertEqual(
            self.controller.preview_requests[0].adapter_frame_capacity,
            40,
        )
        self.assertFalse(self.sidebar.sa30_compatible_check.isEnabled())

    def test_malformed_persisted_profile_values_do_not_enable_40_slot_mode(self):
        for malformed in (1, "true"):
            with self.subTest(malformed=malformed):
                controller = _LightweightScanController()
                controller.session.repo.get_global_setting.return_value = {
                    "dpi": 1_000,
                    "output_folder": "/tmp/kept-setting",
                    "sa30_compatible_roll_feeder": malformed,
                }
                sidebar = ScanSidebar(controller)
                try:
                    _select_device(sidebar, PARKED_SA30_DEVICE)

                    self.assertFalse(sidebar.settings.sa30_compatible_roll_feeder)
                    self.assertEqual(sidebar.settings.dpi, 1_000)
                    self.assertEqual(
                        sidebar.settings.output_folder,
                        "/tmp/kept-setting",
                    )
                    self.assertIsNone(sidebar._effective_roll_capacity(PARKED_SA30_DEVICE))
                    self.assertTrue(sidebar.roll_preview_btn.isHidden())
                finally:
                    sidebar.close()
                    sidebar.deleteLater()
                    controller.deleteLater()

    def test_live_40_slot_capacity_is_authoritative_without_profile_prompt(self):
        _select_device(self.sidebar, FULL_DEVICE)

        self.assertTrue(self.sidebar.sa30_compatible_check.isHidden())
        self.assertFalse(self.sidebar.roll_preview_btn.isHidden())
        self.assertEqual(
            self.sidebar._effective_roll_capacity(FULL_DEVICE),
            40,
        )
        self.assertEqual(
            self.sidebar.device_status_label.text(),
            "● Connected · Roll feeder ready · 40 slots",
        )
        self.assertEqual(
            self.sidebar.device_status_label.property("scannerState"),
            "ready",
        )

    def test_live_6_slot_capacity_blocks_roll_even_with_saved_40_slot_profile(self):
        self.sidebar.settings = replace(
            self.sidebar.settings,
            sa30_compatible_roll_feeder=True,
        )

        _select_device(self.sidebar, STOCK_SA21_DEVICE)

        self.assertTrue(self.sidebar.sa30_compatible_check.isHidden())
        self.assertTrue(self.sidebar.sa30_compatible_check.isChecked())
        self.assertTrue(self.sidebar.roll_preview_btn.isHidden())
        self.assertIsNone(self.sidebar._effective_roll_capacity(STOCK_SA21_DEVICE))
        self.assertEqual(
            self.sidebar.device_status_label.text(),
            "● Connected · Feeder ready · 6 slots",
        )

        self.sidebar._on_roll_preview()

        self.assertEqual(self.controller.preview_requests, [])

    def test_non_ls5000_never_uses_saved_40_slot_profile(self):
        self.sidebar.settings = replace(
            self.sidebar.settings,
            sa30_compatible_roll_feeder=True,
        )
        non_ls5000_with_unknown_frame_control = ScannerDevice(
            id=MINIMAL_DEVICE.id,
            vendor=MINIMAL_DEVICE.vendor,
            model=MINIMAL_DEVICE.model,
            capabilities=replace(
                MINIMAL_CAPS,
                adapter_frame_control=True,
            ),
        )

        _select_device(self.sidebar, non_ls5000_with_unknown_frame_control)

        self.assertTrue(self.sidebar.sa30_compatible_check.isHidden())
        self.assertTrue(self.sidebar.roll_preview_btn.isHidden())
        self.assertIsNone(self.sidebar._effective_roll_capacity(non_ls5000_with_unknown_frame_control))
        self.assertEqual(
            self.sidebar.device_status_label.text(),
            "● Connected · Ready",
        )

    def test_opted_in_selected_scan_request_uses_effective_40_slots(self):
        _select_device(self.sidebar, PARKED_SA30_DEVICE)
        self.sidebar.sa30_compatible_check.setChecked(True)
        self.sidebar._on_roll_preview_ready(
            "parked-preview-token",
            _roll_preview_session(3),
        )
        self.sidebar.folder_edit.setText("/tmp/negpy-parked-roll")

        self.sidebar._on_roll_scan_selected([2])

        self.assertEqual(len(self.controller.scan_requests), 1)
        request = self.controller.scan_requests[0]
        self.assertEqual(request.preview_token, "parked-preview-token")
        self.assertEqual(request.adapter_frame_capacity, 40)


class TestLightweightScannerStatus(LightweightScanSidebarTestCase):
    def test_single_frame_progress_is_visible_beside_device(self):
        _select_device(self.sidebar, FULL_DEVICE)
        self.sidebar.frame_spin.setValue(17)

        self.sidebar.set_scanning(True)
        self.assertEqual(self.sidebar._active_scan_device_id, FULL_DEVICE.id)
        self.assertEqual(self.sidebar._active_scan_frame, 17)
        self.assertFalse(self.sidebar.device_combo.isEnabled())
        self.assertFalse(self.sidebar.refresh_btn.isEnabled())
        for widget in (
            self.sidebar.dpi_combo,
            self.sidebar.depth_combo,
            self.sidebar.ir_check,
            self.sidebar.frame_spin,
            self.sidebar.autofocus_check,
            self.sidebar.ae_check,
            self.sidebar.samples_combo,
            self.sidebar.archival_split_check,
            self.sidebar.fmt_combo,
            self.sidebar.folder_edit,
            self.sidebar.browse_btn,
            self.sidebar.pattern_edit,
            self.sidebar.registered_geometry_check,
            self.sidebar.subframe_spin,
            self.sidebar.br_y_spin,
            self.sidebar.load_registration_btn,
        ):
            self.assertFalse(widget.isEnabled())

        # Programmatic widget changes must not rename the request already in
        # flight. Disabled controls also prevent this through the UI.
        self.sidebar.frame_spin.setValue(18)
        self.sidebar._on_scan_progress(0.42)

        self.assertEqual(
            self.sidebar.device_status_label.text(),
            "● Scanning frame 1 of 1 · Slot 17 · 42%",
        )
        self.assertEqual(
            self.sidebar.device_status_label.property("scannerState"),
            "active",
        )
        self.assertEqual(
            self.sidebar.device_status_label.accessibleName(),
            "Scanner status: Scanning frame 1 of 1 · Slot 17 · 42%",
        )
        self.assertEqual(
            self.sidebar.device_status_label.accessibleDescription(),
            "Scanning frame 1 of 1 · Slot 17 · 42%",
        )

    def test_single_frame_cancel_stays_visible_until_terminal_signal(self):
        _select_device(self.sidebar, FULL_DEVICE)
        self.sidebar.set_scanning(True)

        self.sidebar._on_scan()

        self.assertTrue(self.sidebar._scanning)
        self.assertTrue(self.sidebar._scan_cancel_pending)
        self.assertEqual(self.controller.cancel_calls, 1)
        self.assertFalse(self.sidebar.scan_btn.isEnabled())
        self.assertEqual(
            self.sidebar.device_status_label.text(),
            "● Stopping the current frame…",
        )

        # SANE can only observe Stop after its active blocking read returns.
        # Progress from that read may still arrive, but must not replace the
        # user's visible stopping state.
        self.controller.scan_progress.emit(0.51)

        self.assertEqual(self.sidebar.progress_bar.value(), 51)
        self.assertEqual(
            self.sidebar.device_status_label.text(),
            "● Stopping the current frame…",
        )
        self.sidebar._on_scan()
        self.assertEqual(self.controller.cancel_calls, 1)

    def test_single_frame_cancelled_restores_controls_and_ignores_stale_progress(self):
        _select_device(self.sidebar, FULL_DEVICE)
        self.sidebar.set_scanning(True)
        self.sidebar._on_scan()

        self.controller.scan_cancelled.emit()

        self.assertFalse(self.sidebar._scanning)
        self.assertFalse(self.sidebar._scan_cancel_pending)
        self.assertEqual(self.sidebar.scan_btn.text(), " Scan")
        self.assertTrue(self.sidebar.scan_btn.isEnabled())
        self.assertTrue(self.sidebar.progress_bar.isHidden())
        self.assertTrue(self.sidebar.eject_btn.isEnabled())
        self.assertTrue(self.sidebar.roll_preview_btn.isEnabled())
        self.assertTrue(self.sidebar.roll_slot_selector.isEnabled())
        self.assertTrue(self.sidebar.device_combo.isEnabled())
        self.assertTrue(self.sidebar.refresh_btn.isEnabled())
        self.assertTrue(self.sidebar.dpi_combo.isEnabled())
        self.assertTrue(self.sidebar.depth_combo.isEnabled())
        self.assertTrue(self.sidebar.frame_spin.isEnabled())
        self.assertTrue(self.sidebar.autofocus_check.isEnabled())
        self.assertTrue(self.sidebar.ae_check.isEnabled())
        self.assertTrue(self.sidebar.archival_split_check.isEnabled())
        self.assertTrue(self.sidebar.ir_check.isEnabled())
        self.assertTrue(self.sidebar.samples_combo.isEnabled())
        self.assertTrue(self.sidebar.fmt_combo.isEnabled())
        self.assertTrue(self.sidebar.folder_edit.isEnabled())
        self.assertTrue(self.sidebar.browse_btn.isEnabled())
        self.assertTrue(self.sidebar.pattern_edit.isEnabled())
        self.assertTrue(self.sidebar.registered_geometry_check.isEnabled())
        self.assertFalse(self.sidebar.subframe_spin.isEnabled())
        self.assertFalse(self.sidebar.br_y_spin.isEnabled())
        self.assertTrue(self.sidebar.load_registration_btn.isEnabled())
        self.assertIsNone(self.sidebar._active_scan_device_id)
        self.assertIsNone(self.sidebar._active_scan_frame)
        self.assertEqual(self.sidebar.status_label.text(), "Scan stopped.")
        self.assertEqual(
            self.sidebar.device_status_label.text(),
            "● Connected · Scan stopped",
        )
        self.assertEqual(
            self.sidebar.device_status_label.property("scannerState"),
            "ready",
        )

        self.controller.scan_progress.emit(0.75)

        self.assertTrue(self.sidebar.progress_bar.isHidden())
        self.assertEqual(
            self.sidebar.device_status_label.text(),
            "● Connected · Scan stopped",
        )

    def test_single_frame_scan_blocks_both_roll_launch_paths(self):
        _select_device(self.sidebar, FULL_DEVICE)
        self.sidebar.set_scanning(True)

        self.assertFalse(self.sidebar.roll_preview_btn.isEnabled())
        self.assertFalse(self.sidebar.roll_slot_selector.isEnabled())

        self.sidebar._on_roll_preview()
        self.sidebar._on_roll_scan_selected([2])

        self.assertEqual(self.controller.preview_requests, [])
        self.assertEqual(self.controller.scan_requests, [])
        self.assertFalse(self.sidebar._roll_scanning)

    def test_conventional_controls_restore_minimal_device_capabilities(self):
        _select_device(self.sidebar, MINIMAL_DEVICE)

        self.sidebar.set_scanning(True)
        self.sidebar.set_scanning(False)

        self.assertTrue(self.sidebar.device_combo.isEnabled())
        self.assertTrue(self.sidebar.refresh_btn.isEnabled())
        self.assertTrue(self.sidebar.dpi_combo.isEnabled())
        self.assertTrue(self.sidebar.depth_combo.isEnabled())
        self.assertTrue(self.sidebar.autofocus_check.isEnabled())
        self.assertTrue(self.sidebar.fmt_combo.isEnabled())
        self.assertFalse(self.sidebar.ir_check.isEnabled())
        self.assertFalse(self.sidebar.frame_spin.isEnabled())
        self.assertFalse(self.sidebar.ae_check.isEnabled())
        self.assertFalse(self.sidebar.samples_combo.isEnabled())
        self.assertFalse(self.sidebar.archival_split_check.isEnabled())
        self.assertFalse(self.sidebar.registered_geometry_check.isEnabled())

    def test_conventional_cancel_signal_does_not_end_active_roll_preview(self):
        _select_device(self.sidebar, FULL_DEVICE)
        self.sidebar._on_roll_preview()
        self.assertTrue(self.sidebar._roll_scanning)

        self.controller.scan_cancelled.emit()

        self.assertTrue(self.sidebar._roll_scanning)
        self.assertFalse(self.sidebar._scanning)
        self.assertEqual(
            self.sidebar.device_status_label.text(),
            "● Making previews · Reading the whole roll",
        )

    def test_scanner_status_sits_below_device_and_reports_ready(self):
        _select_device(self.sidebar, FULL_DEVICE)

        layout = self.sidebar.layout()
        self.assertLess(
            layout.indexOf(self.sidebar.device_status_label),
            layout.indexOf(self.sidebar.frame_label),
        )
        self.assertEqual(
            self.sidebar.device_status_label.text(),
            "● Connected · Roll feeder ready · 40 slots",
        )
        self.assertEqual(
            self.sidebar.device_status_label.accessibleName(),
            "Scanner status: Connected · Roll feeder ready · 40 slots",
        )
        self.assertEqual(
            self.sidebar.device_status_label.accessibleDescription(),
            "Connected · Roll feeder ready · 40 slots",
        )
        self.assertEqual(
            self.sidebar.device_status_label.property("scannerState"),
            "ready",
        )

    def test_scanner_status_reports_no_detected_device(self):
        self.sidebar._on_devices_ready([])

        self.assertEqual(
            self.sidebar.device_status_label.text(),
            "● Disconnected · No scanner detected",
        )
        self.assertEqual(
            self.sidebar.device_status_label.accessibleName(),
            "Scanner status: Disconnected · No scanner detected",
        )
        self.assertEqual(
            self.sidebar.device_status_label.accessibleDescription(),
            "Disconnected · No scanner detected",
        )
        self.assertEqual(
            self.sidebar.device_status_label.property("scannerState"),
            "error",
        )

    def test_roll_workflow_separates_thumbnail_and_final_scan_quality(self):
        _select_device(self.sidebar, FULL_DEVICE)

        self.assertFalse(self.sidebar.roll_quality_widget.isHidden())
        self.assertEqual(
            self.sidebar.roll_preview_resolution_box.text(),
            "97 dpi (thumbnails only)",
        )
        self.assertEqual(
            self.sidebar.roll_scan_resolution_box.text(),
            "4000 dpi (Best quality)",
        )
        self.assertEqual(
            self.sidebar.roll_master_format_box.text(),
            "16-bit TIFF (linear negative)",
        )

    def test_roll_storage_estimate_uses_the_selected_output_filesystem(self):
        _select_device(self.sidebar, FULL_DEVICE)
        self.sidebar._on_roll_preview_ready(
            "preview-token",
            _roll_preview_session(3),
        )

        with tempfile.TemporaryDirectory() as output_folder:
            self.sidebar.folder_edit.setText(output_folder)
            self.sidebar.roll_slot_selector.set_selected_slot_ids([1, 2])

            estimate = self.sidebar.roll_slot_selector.storage_estimate_label.text()
            self.assertIn("2 frames", estimate)
            self.assertIn("GiB free", estimate)
            self.assertNotIn("free space unknown", estimate)

    def test_roll_preview_meter_inset_is_clear_bounded_and_persisted(self):
        _select_device(self.sidebar, FULL_DEVICE)
        repo = self.controller.session.repo
        repo.save_global_setting.reset_mock()

        field = self.sidebar.preview_meter_inset_spin
        form = self.sidebar.roll_quality_widget.layout()
        self.assertIsInstance(form, QFormLayout)
        self.assertEqual(form.labelForField(field).text(), "Preview meter inset")
        self.assertEqual((field.minimum(), field.maximum(), field.value()), (0, 30, 10))
        self.assertEqual(field.suffix(), "% per edge")
        self.assertIn("brightness", field.toolTip())
        self.assertIn("whole negative", field.toolTip())
        self.assertIn("Final scans are unaffected", field.toolTip())
        self.assertIn("10% matches full-scan metering", field.toolTip())

        field.setValue(17)

        self.assertEqual(self.sidebar.settings.preview_meter_inset_percent, 17)
        repo.save_global_setting.assert_called()
        persisted = repo.save_global_setting.call_args.args[1]
        self.assertEqual(persisted["preview_meter_inset_percent"], 17)

    def test_roll_preview_non_inverted_negative_toggle_is_clear_and_persisted(self):
        _select_device(self.sidebar, FULL_DEVICE)
        repo = self.controller.session.repo
        repo.save_global_setting.reset_mock()

        toggle = self.sidebar.preview_non_inverted_negative_check
        form = self.sidebar.roll_quality_widget.layout()
        self.assertIsInstance(form, QFormLayout)
        self.assertEqual(form.labelForField(toggle).text(), "Preview display")
        self.assertFalse(toggle.isChecked())
        self.assertIn("display only", toggle.toolTip().lower())
        self.assertIn("saved scan", toggle.toolTip().lower())

        toggle.setChecked(True)

        self.assertTrue(self.sidebar.settings.preview_show_non_inverted_negative)
        repo.save_global_setting.assert_called()
        persisted = repo.save_global_setting.call_args.args[1]
        self.assertIs(persisted["preview_show_non_inverted_negative"], True)

    def test_preview_display_changes_preserve_disabled_scan_preferences(self):
        _select_device(self.sidebar, FULL_DEVICE)
        self.sidebar.ae_check.setChecked(True)
        self.sidebar.archival_split_check.setChecked(True)
        self.assertTrue(self.sidebar.settings.auto_exposure)
        self.assertTrue(self.sidebar.settings.archival_split_capture)

        self.sidebar.set_scanning(True)
        self.assertFalse(self.sidebar.ae_check.isEnabled())
        self.assertFalse(self.sidebar.archival_split_check.isEnabled())
        self.assertTrue(self.sidebar.ae_check.isChecked())
        self.assertTrue(self.sidebar.archival_split_check.isChecked())

        repo = self.controller.session.repo
        repo.save_global_setting.reset_mock()
        self.sidebar.preview_meter_inset_spin.setValue(17)
        self.sidebar.preview_non_inverted_negative_check.setChecked(True)

        self.assertTrue(self.sidebar.settings.auto_exposure)
        self.assertTrue(self.sidebar.settings.archival_split_capture)
        persisted = repo.save_global_setting.call_args.args[1]
        self.assertTrue(persisted["auto_exposure"])
        self.assertTrue(persisted["archival_split_capture"])
        self.assertEqual(persisted["preview_meter_inset_percent"], 17)
        self.assertTrue(persisted["preview_show_non_inverted_negative"])

    def test_changing_preview_polarity_rerenders_every_loaded_slot_offline(self):
        _select_device(self.sidebar, FULL_DEVICE)
        raw = np.arange(100 * 100 * 3, dtype=np.uint16).reshape(100, 100, 3)
        session = SimpleNamespace(
            slots=tuple(
                SimpleNamespace(
                    slot_id=slot_id,
                    thumbnail=raw,
                    warnings=(),
                    boundary_offset_rows=0,
                )
                for slot_id in (1, 2)
            )
        )
        self.sidebar._on_roll_preview_ready("preview-token", session)
        self.sidebar.roll_slot_selector.set_selected_slot_ids([2])
        self.controller.reload_ls5000_roll_thumbnail = MagicMock()

        before = [
            self.sidebar.roll_slot_selector.model.data(
                self.sidebar.roll_slot_selector.model.index(row, 0),
                Qt.ItemDataRole.DecorationRole,
            ).cacheKey()
            for row in range(2)
        ]

        self.sidebar.preview_non_inverted_negative_check.setChecked(True)
        QApplication.processEvents()

        after = [
            self.sidebar.roll_slot_selector.model.data(
                self.sidebar.roll_slot_selector.model.index(row, 0),
                Qt.ItemDataRole.DecorationRole,
            ).cacheKey()
            for row in range(2)
        ]
        self.assertTrue(all(new != old for old, new in zip(before, after, strict=True)))
        self.assertEqual(self.sidebar.roll_slot_selector.selected_slot_ids(), [2])
        self.controller.reload_ls5000_roll_thumbnail.assert_not_called()

    def test_changing_preview_meter_inset_rerenders_every_loaded_slot_offline(self):
        _select_device(self.sidebar, FULL_DEVICE)
        interior_ramp = np.linspace(20_000, 40_000, 80, dtype=np.uint16)
        raw = np.full((100, 100, 3), 65_535, dtype=np.uint16)
        raw[10:90, 10:90, :] = interior_ramp[:, None, None]
        session = SimpleNamespace(
            slots=tuple(
                SimpleNamespace(
                    slot_id=slot_id,
                    thumbnail=raw,
                    warnings=(),
                    boundary_offset_rows=0,
                )
                for slot_id in (1, 2)
            )
        )
        self.sidebar._on_roll_preview_ready("preview-token", session)
        self.sidebar.roll_slot_selector.set_selected_slot_ids([2])
        self.controller.reload_ls5000_roll_thumbnail = MagicMock()

        def center_red(row: int) -> int:
            index = self.sidebar.roll_slot_selector.model.index(row, 0)
            icon = self.sidebar.roll_slot_selector.model.data(
                index,
                Qt.ItemDataRole.DecorationRole,
            )
            image = icon.pixmap(100, 100).toImage()
            return image.pixelColor(image.width() // 2, image.height() // 2).red()

        before = [center_red(row) for row in range(2)]

        self.sidebar.preview_meter_inset_spin.setValue(0)
        QApplication.processEvents()

        after = [center_red(row) for row in range(2)]
        self.assertTrue(all(new > old + 40 for old, new in zip(before, after, strict=True)))
        self.assertTrue(
            all(
                self.sidebar.roll_slot_selector.model.data(
                    self.sidebar.roll_slot_selector.model.index(row, 0),
                    RollSlotModel.METER_INSET_PERCENT_ROLE,
                )
                == 0
                for row in range(2)
            )
        )
        self.assertEqual(self.sidebar.roll_slot_selector.selected_slot_ids(), [2])
        self.controller.reload_ls5000_roll_thumbnail.assert_not_called()

    def test_single_frame_format_does_not_claim_to_control_inversion(self):
        dpi_label = self.sidebar.form.labelForField(self.sidebar.dpi_combo)
        format_label = self.sidebar.form.labelForField(self.sidebar.fmt_combo)

        self.assertEqual(dpi_label.text(), "Single-frame DPI")
        self.assertEqual(format_label.text(), "Single-frame format")
        self.assertIn("same scanner-linear capture", self.sidebar.format_hint.text())
        self.assertIn("does not control inversion", self.sidebar.format_hint.text())
        self.assertNotIn("negative data", self.sidebar.format_hint.text())

    def test_roll_progress_reports_selected_frame_and_slot_at_device_name(self):
        _select_device(self.sidebar, FULL_DEVICE)

        self.sidebar._on_roll_progress(
            SimpleNamespace(
                completed=1,
                total=5,
                message="Scanning frame 2 of 5 · Slot 17",
                operation=RollOperation.FULL_SCAN,
            )
        )

        self.assertEqual(
            self.sidebar.device_status_label.text(),
            "● Scanning frame 2 of 5 · Slot 17",
        )
        self.assertEqual(
            self.sidebar.device_status_label.property("scannerState"),
            "active",
        )

    def test_roll_bw_progress_includes_frame_slot_and_percent(self):
        _select_device(self.sidebar, FULL_DEVICE)

        self.sidebar._on_roll_progress(
            SimpleNamespace(
                completed=2,
                total=4,
                message="Scanning frame 3 of 4 · Slot 07 · 63%",
                operation=RollOperation.FULL_SCAN,
            )
        )

        self.assertEqual(
            self.sidebar.device_status_label.text(),
            "● Scanning frame 3 of 4 · Slot 07 · 63%",
        )

    def test_preview_completion_state_uses_typed_operation_not_message_text(self):
        _select_device(self.sidebar, FULL_DEVICE)

        self.sidebar._on_roll_progress(
            SimpleNamespace(
                completed=1,
                total=1,
                message="All thumbnail slots are ready",
                operation=RollOperation.PREVIEW,
            )
        )

        self.assertEqual(
            self.sidebar.device_status_label.text(),
            "● All thumbnail slots are ready",
        )
        self.assertEqual(
            self.sidebar.device_status_label.property("scannerState"),
            "ready",
        )

    def test_roll_completion_and_error_are_visible_at_device_name(self):
        _select_device(self.sidebar, FULL_DEVICE)

        self.sidebar._on_roll_finished(
            SimpleNamespace(
                rgb_paths=("frame-01.tif", "frame-02.tif"),
                stopped=False,
                operation=RollOperation.FULL_SCAN,
            )
        )
        self.assertEqual(
            self.sidebar.device_status_label.text(),
            "● Complete · 2 frames ready",
        )
        self.assertEqual(
            self.sidebar.device_status_label.property("scannerState"),
            "ready",
        )

        self.sidebar._on_roll_error(
            SimpleNamespace(
                message="USB endpoint stalled",
                recovery_required=True,
            )
        )
        self.assertEqual(
            self.sidebar.device_status_label.text(),
            "● Error · Power-cycle required",
        )
        self.assertEqual(
            self.sidebar.device_status_label.property("scannerState"),
            "error",
        )

    def test_preview_cannot_be_stopped_mid_transaction_and_reports_preview_stop(self):
        _select_device(self.sidebar, FULL_DEVICE)

        self.sidebar._on_roll_preview()

        self.assertTrue(self.sidebar._roll_scanning)
        self.assertTrue(self.sidebar.roll_stop_btn.isHidden())
        self.assertFalse(self.sidebar.roll_stop_btn.isEnabled())
        self.sidebar._on_roll_stop()
        self.assertEqual(self.controller.cancel_calls, 0)

        self.sidebar._on_roll_finished(
            SimpleNamespace(
                rgb_paths=(),
                stopped=True,
                operation=RollOperation.PREVIEW,
            )
        )

        self.assertFalse(self.sidebar._roll_scanning)
        self.assertIsNone(self.sidebar._roll_operation)
        self.assertEqual(
            self.sidebar.device_status_label.text(),
            "● Preview stopped · No thumbnails loaded",
        )
        self.assertIn("preview stopped", self.sidebar.roll_status_label.text().lower())

    def test_stopped_roll_can_reselect_only_unfinished_slots(self):
        _select_device(self.sidebar, FULL_DEVICE)
        self.sidebar._on_roll_preview_ready(
            "preview-token",
            _roll_preview_session(5),
        )
        self.sidebar.folder_edit.setText("/tmp/negpy-roll-retry")
        self.sidebar.roll_slot_selector.set_selected_slot_ids([1, 3, 5])

        self.sidebar._on_roll_scan_selected([1, 3, 5])
        self.controller.ls5000_roll_frame_finished.emit(1, "frame-01.tif")
        self.sidebar._on_roll_finished(
            SimpleNamespace(
                rgb_paths=("frame-01.tif",),
                stopped=True,
                operation=RollOperation.FULL_SCAN,
            )
        )

        self.assertFalse(self.sidebar.retry_remaining_btn.isHidden())
        self.assertTrue(self.sidebar.retry_remaining_btn.isEnabled())
        self.assertEqual(self.sidebar.retry_remaining_btn.text(), "Select remaining (2)")

        self.sidebar.retry_remaining_btn.click()

        self.assertEqual(
            self.sidebar.roll_slot_selector.selected_slot_ids(),
            [3, 5],
        )
        self.assertIn("Selected 2 unfinished slots", self.sidebar.roll_status_label.text())

    def test_failed_roll_keeps_remaining_slots_through_preview_reload(self):
        _select_device(self.sidebar, FULL_DEVICE)
        self.sidebar._on_roll_preview_ready(
            "preview-token",
            _roll_preview_session(4),
        )
        self.sidebar.folder_edit.setText("/tmp/negpy-roll-retry")
        self.sidebar._on_roll_scan_selected([1, 2, 4])
        self.controller.ls5000_roll_frame_finished.emit(1, "frame-01.tif")

        self.controller.ls5000_roll_preview_invalidated.emit()
        self.sidebar._on_roll_error(
            SimpleNamespace(
                message="USB endpoint stalled",
                recovery_required=True,
            )
        )

        self.assertFalse(self.sidebar.retry_remaining_btn.isHidden())
        self.assertFalse(self.sidebar.retry_remaining_btn.isEnabled())
        self.assertIn("Reload thumbnails", self.sidebar.retry_remaining_btn.text())

        self.sidebar._on_roll_preview_ready(
            "replacement-token",
            _roll_preview_session(4, source_marker="c"),
        )

        self.assertTrue(self.sidebar.retry_remaining_btn.isEnabled())
        self.assertEqual(self.sidebar.retry_remaining_btn.text(), "Select remaining (2)")
        self.sidebar.retry_remaining_btn.click()
        self.assertEqual(
            self.sidebar.roll_slot_selector.selected_slot_ids(),
            [2, 4],
        )

    def test_failed_roll_retry_is_cleared_when_preview_is_a_different_roll(self):
        _select_device(self.sidebar, FULL_DEVICE)
        self.sidebar._on_roll_preview_ready(
            "preview-token",
            _roll_preview_session(4),
        )
        self.sidebar.folder_edit.setText("/tmp/negpy-roll-retry")
        self.sidebar._on_roll_scan_selected([1, 2, 4])
        self.controller.ls5000_roll_frame_finished.emit(1, "frame-01.tif")
        self.controller.ls5000_roll_preview_invalidated.emit()

        self.sidebar._on_roll_preview_ready(
            "different-roll-token",
            _roll_preview_session(4, source_marker="c", visual_xor=(1 << 256) - 1),
        )

        self.assertTrue(self.sidebar.retry_remaining_btn.isHidden())
        self.assertEqual(self.sidebar._roll_retry_pending_slots, ())
        self.assertIn("different roll", self.sidebar.roll_status_label.text().lower())

    def test_equivalent_retry_preview_restores_original_scan_material(self):
        _select_device(self.sidebar, FULL_DEVICE)
        self.sidebar._on_roll_preview_ready(
            "preview-token",
            _roll_preview_session(4),
        )
        self.sidebar.folder_edit.setText("/tmp/negpy-roll-retry")
        self.sidebar.roll_slot_selector.set_scan_material(
            ScanMaterial.BLACK_AND_WHITE_NEGATIVE
        )
        self.sidebar._on_roll_scan_selected([1, 2])
        self.controller.ls5000_roll_preview_invalidated.emit()
        self.sidebar.roll_slot_selector.set_scan_material(
            ScanMaterial.COLOR_NEGATIVE
        )

        self.sidebar._on_roll_preview_ready(
            "replacement-token",
            _roll_preview_session(4, source_marker="c"),
        )

        self.assertEqual(
            self.sidebar.roll_slot_selector.scan_material(),
            ScanMaterial.BLACK_AND_WHITE_NEGATIVE,
        )
        self.assertTrue(self.sidebar.retry_remaining_btn.isEnabled())


class ScanSidebarTestCase(unittest.TestCase):
    """Base class: real AppController + real ScanSidebar, offscreen, no device I/O."""

    def setUp(self):
        self.controller = _build_controller()
        self.sidebar = ScanSidebar(self.controller)

    def tearDown(self):
        del self.sidebar
        _stop_threads(self.controller)


class TestNewControlsInstantiate(ScanSidebarTestCase):
    def test_new_widgets_exist(self):
        for name in (
            "frame_spin",
            "ae_check",
            "archival_split_check",
            "registered_geometry_check",
            "subframe_spin",
            "br_y_spin",
            "load_registration_btn",
        ):
            self.assertTrue(hasattr(self.sidebar, name), f"missing widget: {name}")

    def test_new_controls_disabled_with_no_device_selected(self):
        # Fresh widgets default to Qt's enabled=True until something actually
        # runs the no-device gating -- exactly like the pre-existing
        # dpi/depth/ir/samples controls, none of which are disabled by
        # __init__ alone either. _update_device_caps() is what a real
        # "no device" state (e.g. _on_device_changed on the placeholder item)
        # actually triggers.
        self.sidebar._update_device_caps()
        self.assertFalse(self.sidebar.frame_spin.isEnabled())
        self.assertFalse(self.sidebar.ae_check.isEnabled())
        self.assertFalse(self.sidebar.archival_split_check.isEnabled())
        self.assertFalse(self.sidebar.registered_geometry_check.isEnabled())
        self.assertFalse(self.sidebar.subframe_spin.isEnabled())
        self.assertFalse(self.sidebar.br_y_spin.isEnabled())
        self.assertFalse(self.sidebar.load_registration_btn.isEnabled())

    def test_controller_signals_connected_without_error(self):
        # _connect_signals() already ran in setUp via __init__; a bad signal/slot
        # signature would have raised there. Emitting confirms the real
        # pyqtSignal -> slot binding is live (not a MagicMock no-op).
        self.sidebar.set_scanning(True)
        self.sidebar.controller.scan_progress.emit(0.5)
        self.assertEqual(self.sidebar.progress_bar.value(), 50)


class TestCapabilityGating(ScanSidebarTestCase):
    def test_full_capability_device_enables_new_controls(self):
        _select_device(self.sidebar, FULL_DEVICE)
        self.assertTrue(self.sidebar.frame_spin.isEnabled())
        self.assertEqual(self.sidebar.frame_spin.maximum(), 40)
        self.assertTrue(self.sidebar.ae_check.isEnabled())
        self.assertTrue(self.sidebar.archival_split_check.isEnabled())
        self.assertTrue(self.sidebar.registered_geometry_check.isEnabled())
        self.assertTrue(self.sidebar.load_registration_btn.isEnabled())
        # Subframe/BR-Y stay disabled until "Use Registered Geometry" is checked.
        self.assertFalse(self.sidebar.subframe_spin.isEnabled())
        self.assertFalse(self.sidebar.br_y_spin.isEnabled())
        self.assertFalse(self.sidebar.eject_btn.isHidden())
        self.assertTrue(self.sidebar.eject_btn.isEnabled())

    def test_device_defaults_to_highest_supported_dpi_with_best_quality_label(self):
        _select_device(self.sidebar, FULL_DEVICE)

        self.assertEqual(self.sidebar.dpi_combo.currentData(), 4000)
        self.assertEqual(
            self.sidebar.dpi_combo.currentText(),
            "4000 dpi (Best quality)",
        )

    def test_device_defaults_to_16_bit_with_best_quality_label(self):
        _select_device(self.sidebar, MINIMAL_DEVICE)

        self.assertEqual(self.sidebar.depth_combo.currentData(), 16)
        self.assertEqual(
            self.sidebar.depth_combo.currentText(),
            "16-bit (Best quality)",
        )

    def test_device_best_quality_overrides_saved_lower_dpi(self):
        self.sidebar._settings = replace(self.sidebar._settings, dpi=1000)

        _select_device(self.sidebar, FULL_DEVICE)

        self.assertEqual(self.sidebar.dpi_combo.currentData(), 4000)
        self.assertEqual(
            self.sidebar.dpi_combo.currentText(),
            "4000 dpi (Best quality)",
        )

    def test_minimal_capability_device_disables_new_controls(self):
        _select_device(self.sidebar, MINIMAL_DEVICE)
        self.assertFalse(self.sidebar.frame_spin.isEnabled())
        self.assertFalse(self.sidebar.ae_check.isEnabled())
        self.assertFalse(self.sidebar.archival_split_check.isEnabled())
        self.assertFalse(self.sidebar.registered_geometry_check.isEnabled())
        self.assertFalse(self.sidebar.load_registration_btn.isEnabled())
        self.assertTrue(self.sidebar.eject_btn.isHidden())
        # Pre-existing controls unaffected by the new gating.
        self.assertFalse(self.sidebar.ir_check.isEnabled())
        self.assertTrue(self.sidebar.dpi_combo.isEnabled())
        self.assertTrue(self.sidebar.roll_quality_widget.isHidden())

    def test_switching_to_a_minimal_device_clears_stale_registration(self):
        """Registered geometry is frame/device-specific; it must never
        silently carry over onto a different device."""
        _select_device(self.sidebar, FULL_DEVICE)
        self.sidebar.registered_geometry_check.setChecked(True)
        self.sidebar.subframe_spin.setValue(6.35)
        self.sidebar.br_y_spin.setValue(5003)

        _select_device(self.sidebar, MINIMAL_DEVICE)

        self.assertFalse(self.sidebar.registered_geometry_check.isChecked())
        self.assertEqual(self.sidebar.subframe_spin.value(), 0.0)
        self.assertEqual(self.sidebar.br_y_spin.value(), 0)

    def test_eject_runs_on_controller_and_invalidates_loaded_preview(self):
        _select_device(self.sidebar, FULL_DEVICE)
        self.sidebar._on_roll_preview_ready(
            "preview-token",
            _roll_preview_session(5),
        )
        self.controller.eject_scanner = MagicMock()
        self.controller.invalidate_ls5000_roll_preview = MagicMock()

        self.sidebar.eject_btn.click()

        self.controller.eject_scanner.assert_called_once_with(FULL_DEVICE.id)
        self.assertFalse(self.sidebar.eject_btn.isEnabled())
        self.sidebar._on_ejected(True)
        self.assertIsNone(self.sidebar._roll_preview_token)
        self.assertEqual(self.sidebar.roll_slot_selector.model.rowCount(), 0)
        self.controller.invalidate_ls5000_roll_preview.assert_called_once()
        self.assertEqual(self.sidebar.status_label.text(), "Film ejected.")


class TestArchivalSplitInterlock(ScanSidebarTestCase):
    def test_checking_archival_forces_and_locks_ir_samples_and_16_bit_depth(self):
        _select_device(self.sidebar, FULL_DEVICE)
        self.sidebar.ir_check.setChecked(False)
        self.sidebar.depth_combo.setCurrentIndex(self.sidebar.depth_combo.findData(8))

        self.sidebar.archival_split_check.setChecked(True)

        self.assertTrue(self.sidebar.ir_check.isChecked())
        self.assertEqual(self.sidebar.samples_combo.currentData(), 4)
        self.assertEqual(self.sidebar.depth_combo.currentData(), 16)
        self.assertEqual(
            self.sidebar.depth_combo.currentText(),
            "16-bit (Best quality)",
        )
        self.assertFalse(self.sidebar.ir_check.isEnabled())
        self.assertFalse(self.sidebar.samples_combo.isEnabled())
        self.assertFalse(self.sidebar.depth_combo.isEnabled())
        self.assertEqual(self.sidebar.settings.depth, 16)
        persisted = self.controller.session.repo.save_global_setting.call_args.args[1]
        self.assertEqual(persisted["depth"], 16)

        self.sidebar.depth_combo.setCurrentIndex(self.sidebar.depth_combo.findData(8))

        self.assertEqual(self.sidebar.depth_combo.currentData(), 16)
        self.assertEqual(self.sidebar.settings.depth, 16)

    def test_unchecking_archival_restores_capability_derived_enabled_state(self):
        _select_device(self.sidebar, FULL_DEVICE)
        self.sidebar.archival_split_check.setChecked(True)

        self.sidebar.archival_split_check.setChecked(False)

        self.assertTrue(self.sidebar.ir_check.isEnabled())
        self.assertTrue(self.sidebar.samples_combo.isEnabled())
        self.assertTrue(self.sidebar.depth_combo.isEnabled())

    def test_archival_unavailable_on_minimal_device(self):
        _select_device(self.sidebar, MINIMAL_DEVICE)
        self.assertFalse(self.sidebar.archival_split_check.isChecked())
        self.assertFalse(self.sidebar.archival_split_check.isEnabled())


class TestRegisteredGeometryToggle(ScanSidebarTestCase):
    def test_checking_enables_subframe_and_br_y_fields(self):
        _select_device(self.sidebar, FULL_DEVICE)

        self.sidebar.registered_geometry_check.setChecked(True)

        self.assertTrue(self.sidebar.subframe_spin.isEnabled())
        self.assertTrue(self.sidebar.br_y_spin.isEnabled())

    def test_unchecking_disables_subframe_and_br_y_fields(self):
        _select_device(self.sidebar, FULL_DEVICE)
        self.sidebar.registered_geometry_check.setChecked(True)

        self.sidebar.registered_geometry_check.setChecked(False)

        self.assertFalse(self.sidebar.subframe_spin.isEnabled())
        self.assertFalse(self.sidebar.br_y_spin.isEnabled())

    def test_load_registration_json_populates_fields(self):
        import tempfile
        import os

        _select_device(self.sidebar, FULL_DEVICE)
        self.sidebar.frame_spin.setValue(3)

        with tempfile.TemporaryDirectory() as tmp_dir:
            manifest_path = os.path.join(tmp_dir, "registration.json")
            with open(manifest_path, "w", encoding="utf-8") as fh:
                json.dump({"frames": [{"frame": 3, "subframe_mm": 6.35, "br_y": 5003}]}, fh)

            with patch(
                "negpy.desktop.view.sidebar.scan.QFileDialog.getOpenFileName",
                return_value=(manifest_path, "JSON Files (*.json)"),
            ):
                self.sidebar._on_load_registration_json()

        self.assertTrue(self.sidebar.registered_geometry_check.isChecked())
        self.assertAlmostEqual(self.sidebar.subframe_spin.value(), 6.35, places=2)
        self.assertEqual(self.sidebar.br_y_spin.value(), 5003)
        self.assertIn("Loaded registration for frame 3", self.sidebar.status_label.text())

    def test_load_registration_json_requires_frame_first(self):
        _select_device(self.sidebar, FULL_DEVICE)
        self.sidebar.frame_spin.setValue(0)  # "Current" sentinel -> no frame chosen

        with patch("negpy.desktop.view.sidebar.scan.QFileDialog.getOpenFileName") as mock_dialog:
            self.sidebar._on_load_registration_json()
            mock_dialog.assert_not_called()

        self.assertIn("Frame #", self.sidebar.status_label.text())

    def test_load_registration_json_no_matching_frame_shows_error(self):
        import tempfile
        import os

        _select_device(self.sidebar, FULL_DEVICE)
        self.sidebar.frame_spin.setValue(9)

        with tempfile.TemporaryDirectory() as tmp_dir:
            manifest_path = os.path.join(tmp_dir, "registration.json")
            with open(manifest_path, "w", encoding="utf-8") as fh:
                json.dump({"frames": [{"frame": 3, "subframe_mm": 6.35, "br_y": 5003}]}, fh)

            with patch(
                "negpy.desktop.view.sidebar.scan.QFileDialog.getOpenFileName",
                return_value=(manifest_path, "JSON Files (*.json)"),
            ):
                self.sidebar._on_load_registration_json()

        self.assertFalse(self.sidebar.registered_geometry_check.isChecked())
        self.assertIn("Could not load registration manifest", self.sidebar.status_label.text())


class TestScanParamsAssembly(ScanSidebarTestCase):
    """_on_scan() gathers widget values and hands them to
    controller.build_scan_params(); this checks that hand-off end to end
    against the real (non-mocked) controller method, without ever calling
    controller.start_scan / touching the scan worker thread."""

    def test_archival_recipe_end_to_end_through_build_scan_params(self):
        _select_device(self.sidebar, FULL_DEVICE)
        self.sidebar.folder_edit.setText("/tmp/negpy-scan-test")
        self.sidebar.archival_split_check.setChecked(True)
        self.sidebar.ae_check.setChecked(True)
        self.sidebar.frame_spin.setValue(3)
        self.sidebar.registered_geometry_check.setChecked(True)
        self.sidebar.subframe_spin.setValue(6.35)
        self.sidebar.br_y_spin.setValue(5003)

        captured = {}
        self.controller.start_scan = MagicMock(side_effect=lambda req: captured.update(req=req))

        self.sidebar._on_scan()

        params = captured["req"].params
        self.assertEqual(params.depth, 16)
        self.assertTrue(params.capture_ir)
        self.assertEqual(params.samples_per_scan, 4)
        self.assertTrue(params.auto_exposure)
        self.assertIsNone(params.frame)  # rides inside registered_geometry instead
        self.assertEqual(params.registered_geometry.frame, 3)
        self.assertEqual(params.registered_geometry.subframe_mm, 6.35)
        self.assertEqual(params.registered_geometry.br_y_device_px, 5003)


class TestLS5000RollWorkflow(ScanSidebarTestCase):
    """Nikon-style roll controls, stopped at the controller request boundary."""

    def test_roll_controls_are_gated_by_ls5000_roll_feeder_capability(self):
        _select_device(self.sidebar, FULL_DEVICE)
        self.assertFalse(self.sidebar.roll_preview_btn.isHidden())
        self.assertTrue(self.sidebar.roll_preview_btn.isEnabled())
        self.assertTrue(self.sidebar.roll_slot_selector.isHidden())

        _select_device(self.sidebar, MINIMAL_DEVICE)
        self.assertTrue(self.sidebar.roll_preview_btn.isHidden())
        self.assertTrue(self.sidebar.roll_slot_selector.isHidden())

        ls5000_without_roll_capacity = ScannerDevice(
            id="coolscan3:usb:libusb:001:010",
            vendor="Nikon",
            model="LS-5000",
            capabilities=MINIMAL_CAPS,
        )
        _select_device(self.sidebar, ls5000_without_roll_capacity)
        self.assertTrue(self.sidebar.roll_preview_btn.isHidden())

        _select_device(self.sidebar, STOCK_SA21_DEVICE)
        self.assertTrue(self.sidebar.roll_preview_btn.isHidden())

    def test_load_roll_thumbnails_sends_preview_request_and_locks_controls(self):
        _select_device(self.sidebar, FULL_DEVICE)
        self.controller.invalidate_ls5000_roll_preview = MagicMock()
        self.controller.start_ls5000_roll_preview = MagicMock()

        self.sidebar._on_roll_preview()

        self.controller.invalidate_ls5000_roll_preview.assert_called_once()
        self.controller.start_ls5000_roll_preview.assert_called_once()
        request = self.controller.start_ls5000_roll_preview.call_args.args[0]
        self.assertIsInstance(request, RollPreviewRequest)
        self.assertEqual(request.device_id, FULL_DEVICE.id)
        self.assertEqual(request.adapter_frame_capacity, 40)
        self.assertEqual(
            Path(request.attempts_root).parts[-2:],
            ("ls5000-roll", "preview-attempts"),
        )
        self.assertTrue(self.sidebar._roll_scanning)
        self.assertFalse(self.sidebar.roll_preview_btn.isEnabled())
        self.assertTrue(self.sidebar.roll_stop_btn.isHidden())
        self.assertFalse(self.sidebar.roll_stop_btn.isEnabled())
        self.assertIn("whole roll", self.sidebar.roll_status_label.text())
        self.assertEqual(
            self.sidebar.device_status_label.text(),
            "● Making previews · Reading the whole roll",
        )
        self.assertEqual(
            self.sidebar.device_status_label.property("scannerState"),
            "active",
        )

    def test_decoded_preview_displays_every_fixed_scanner_slot(self):
        _select_device(self.sidebar, FULL_DEVICE)

        self.sidebar._on_roll_preview_ready("preview-token", _roll_preview_session())

        model = self.sidebar.roll_slot_selector.model
        self.assertEqual(model.rowCount(), 40)
        self.assertEqual(
            [model.data(model.index(row, 0), RollSlotModel.SLOT_ID_ROLE) for row in range(model.rowCount())],
            list(range(1, 41)),
        )
        self.assertEqual(
            model.data(model.index(39, 0), Qt.ItemDataRole.DisplayRole),
            "40",
        )
        self.assertTrue(model.data(model.index(39, 0), RollSlotModel.HAS_WARNINGS_ROLE))
        self.assertEqual(self.sidebar.roll_slot_selector.selected_slot_ids(), [])
        self.assertFalse(self.sidebar.roll_slot_selector.isHidden())
        self.assertEqual(
            self.sidebar.roll_preview_btn.text(),
            " Reload Roll Thumbnails",
        )
        self.assertIn("Loaded 40 scanner slots", self.sidebar.roll_status_label.text())
        self.assertFalse(self.sidebar._roll_scanning)
        self.assertEqual(
            self.sidebar.device_status_label.text(),
            "● Connected · 40 previews ready",
        )

    def test_refresh_clears_bound_thumbnails_and_invalidates_worker(self):
        _select_device(self.sidebar, FULL_DEVICE)
        self.sidebar._on_roll_preview_ready(
            "preview-token",
            _roll_preview_session(5),
        )
        self.controller.invalidate_ls5000_roll_preview = MagicMock()
        self.sidebar._request_devices = MagicMock()

        self.sidebar._on_refresh()

        self.assertIsNone(self.sidebar._roll_preview_token)
        self.assertEqual(self.sidebar.roll_slot_selector.model.rowCount(), 0)
        self.assertTrue(self.sidebar.roll_slot_selector.isHidden())
        self.assertEqual(
            self.sidebar.roll_preview_btn.text(),
            " Load Roll Thumbnails",
        )
        self.controller.invalidate_ls5000_roll_preview.assert_called_once()
        self.sidebar._request_devices.assert_called_once()

    def test_device_change_clears_bound_thumbnails_and_invalidates_worker(self):
        _select_device(self.sidebar, FULL_DEVICE)
        self.sidebar._on_roll_preview_ready(
            "preview-token",
            _roll_preview_session(5),
        )
        self.controller.invalidate_ls5000_roll_preview = MagicMock()

        self.sidebar._on_device_changed(0)

        self.assertIsNone(self.sidebar._roll_preview_token)
        self.assertEqual(self.sidebar.roll_slot_selector.model.rowCount(), 0)
        self.assertTrue(self.sidebar.roll_slot_selector.isHidden())
        self.controller.invalidate_ls5000_roll_preview.assert_called_once()

    def test_worker_invalidation_signal_clears_bound_thumbnails(self):
        _select_device(self.sidebar, FULL_DEVICE)
        self.sidebar._on_roll_preview_ready(
            "preview-token",
            _roll_preview_session(5),
        )

        self.controller.ls5000_roll_preview_invalidated.emit()

        self.assertIsNone(self.sidebar._roll_preview_token)
        self.assertEqual(self.sidebar.roll_slot_selector.model.rowCount(), 0)
        self.assertTrue(self.sidebar.roll_slot_selector.isHidden())

    def test_recovery_error_clears_thumbnails_and_keeps_reload_warning(self):
        _select_device(self.sidebar, FULL_DEVICE)
        self.sidebar._on_roll_preview_ready(
            "preview-token",
            _roll_preview_session(5),
        )

        self.sidebar._on_roll_error(
            SimpleNamespace(
                message="Roll preview failed: USB endpoint stalled",
                recovery_required=True,
            )
        )

        self.assertIsNone(self.sidebar._roll_preview_token)
        self.assertEqual(self.sidebar.roll_slot_selector.model.rowCount(), 0)
        self.assertIn("Power-cycle", self.sidebar.roll_status_label.text())

    def test_scan_selected_refuses_without_a_bound_preview_token(self):
        _select_device(self.sidebar, FULL_DEVICE)
        self.sidebar.folder_edit.setText("/tmp/negpy-roll-color")
        self.controller.start_ls5000_roll_scan = MagicMock()

        self.sidebar._on_roll_scan_selected([2])

        self.controller.start_ls5000_roll_scan.assert_not_called()
        self.assertIn(
            "Load a new whole-roll preview",
            self.sidebar.roll_status_label.text(),
        )

    def test_boundary_offset_reload_reaches_controller_signal_with_exact_slot(self):
        _select_device(self.sidebar, FULL_DEVICE)
        self.sidebar._on_roll_preview_ready("preview-token", _roll_preview_session(4))
        self.sidebar.roll_slot_selector.set_selected_slot_ids([3])

        # Keep this an offline UI test: retain the selector -> controller
        # connection, but disconnect the controller -> worker-thread receiver.
        self.controller.ls5000_roll_thumbnail_reload_requested.disconnect()
        reloads: list[tuple[int, int]] = []
        self.controller.ls5000_roll_thumbnail_reload_requested.connect(lambda slot_id, offset: reloads.append((slot_id, offset)))

        self.sidebar.roll_slot_selector.boundary_offset_spin.setValue(-23)
        self.assertFalse(self.sidebar.roll_slot_selector.scan_button.isEnabled())
        self.sidebar.roll_slot_selector.reload_thumbnail_button.click()

        self.assertEqual(
            self.sidebar.roll_slot_selector.model.boundary_offset_for_slot_id(3),
            -23,
        )
        self.assertEqual(reloads, [(3, -23)])
        replacement = np.full((100, 100, 3), 65_535, dtype=np.uint16)
        replacement[10:90, 10:90, :] = np.linspace(
            20_000,
            40_000,
            80,
            dtype=np.uint16,
        )[:, None, None]
        self.sidebar._on_roll_thumbnail_ready(3, -23, replacement)
        self.assertTrue(self.sidebar.roll_slot_selector.scan_button.isEnabled())

        index = self.sidebar.roll_slot_selector.model.index(2, 0)
        icon_before = self.sidebar.roll_slot_selector.model.data(
            index,
            Qt.ItemDataRole.DecorationRole,
        )
        before = icon_before.pixmap(100, 100).toImage().pixelColor(50, 50).red()

        self.sidebar.preview_meter_inset_spin.setValue(0)

        icon_after = self.sidebar.roll_slot_selector.model.data(
            index,
            Qt.ItemDataRole.DecorationRole,
        )
        after = icon_after.pixmap(100, 100).toImage().pixelColor(50, 50).red()
        self.assertGreater(after, before + 40)

    def test_thumbnail_reload_serializes_controls_and_failure_is_nonterminal(self):
        _select_device(self.sidebar, FULL_DEVICE)
        self.sidebar._on_roll_preview_ready("preview-token", _roll_preview_session(4))
        self.sidebar.roll_slot_selector.set_selected_slot_ids([2])
        self.sidebar.roll_slot_selector.boundary_offset_spin.setValue(-17)
        self.controller.reload_ls5000_roll_thumbnail = MagicMock()
        self.controller.start_ls5000_roll_preview = MagicMock()

        self.sidebar.roll_slot_selector.reload_thumbnail_button.click()

        self.controller.reload_ls5000_roll_thumbnail.assert_called_once_with(2, -17)
        self.assertEqual(self.sidebar._roll_thumbnail_reload_pending, (2, -17))
        self.assertFalse(self.sidebar.roll_preview_btn.isEnabled())
        self.assertFalse(self.sidebar.roll_slot_selector.isEnabled())
        self.sidebar._on_roll_preview()
        self.controller.start_ls5000_roll_preview.assert_not_called()

        self.controller.ls5000_roll_thumbnail_error.emit(
            roll_worker_module.RollThumbnailReloadFailure(
                slot_id=2,
                boundary_offset_rows=-17,
                message="Could not reload thumbnail: invalid offset",
            )
        )

        self.assertIsNone(self.sidebar._roll_thumbnail_reload_pending)
        self.assertFalse(self.sidebar._roll_scanning)
        self.assertTrue(self.sidebar.roll_preview_btn.isEnabled())
        self.assertTrue(self.sidebar.roll_slot_selector.isEnabled())
        self.assertIn("invalid offset", self.sidebar.roll_status_label.text())

    def test_color_negative_request_keeps_offsets_and_selects_rgbi_route(self):
        _select_device(self.sidebar, FULL_DEVICE)
        self.sidebar._on_roll_preview_ready("preview-token", _roll_preview_session(8))
        self.sidebar.folder_edit.setText("/tmp/negpy-roll-color")
        self.sidebar.pattern_edit.setText("roll_{{ '%03d' % seq }}")
        self.sidebar.roll_slot_selector.set_scan_material(ScanMaterial.COLOR_NEGATIVE)
        self.sidebar.roll_slot_selector.set_selected_slot_ids([2])
        self.sidebar.roll_slot_selector.boundary_offset_spin.setValue(-17)
        self.sidebar.roll_slot_selector.confirm_slot_thumbnail(2, -17, None)
        self.controller.start_ls5000_roll_scan = MagicMock()

        self.sidebar._on_roll_scan_selected([2, 7])

        self.controller.start_ls5000_roll_scan.assert_called_once()
        request = self.controller.start_ls5000_roll_scan.call_args.args[0]
        self.assertIsInstance(request, RollScanRequest)
        self.assertEqual(request.preview_token, "preview-token")
        self.assertEqual(request.device_id, FULL_DEVICE.id)
        self.assertEqual(request.adapter_frame_capacity, 40)
        self.assertIs(request.material, ScanMaterial.COLOR_NEGATIVE)
        self.assertTrue(request.material.captures_infrared)
        self.assertEqual(
            [(frame.slot_id, frame.boundary_offset_rows) for frame in request.frames],
            [(2, -17), (7, 0)],
        )
        self.assertEqual(
            Path(request.attempts_root),
            Path(request.output_folder) / ".negpy-ls5000" / "attempts",
        )
        self.assertFalse(self.sidebar.roll_stop_btn.isHidden())
        self.assertTrue(self.sidebar.roll_stop_btn.isEnabled())
        recipe = self.sidebar.roll_slot_selector.scan_material_status_label.text()
        self.assertIn("4000 dpi", recipe)
        self.assertIn("16-bit", recipe)
        self.assertIn("RGB 4× + IR", recipe)
        self.assertIn(
            "IR repair: On after import",
            self.sidebar.roll_slot_selector.ir_repair_status_label.text(),
        )

    def test_default_roll_filename_keeps_physical_slot_identity(self):
        _select_device(self.sidebar, FULL_DEVICE)
        self.sidebar._on_roll_preview_ready(
            "preview-token",
            _roll_preview_session(8),
        )
        self.sidebar.folder_edit.setText("/tmp/negpy-roll-slot-name")
        self.controller.start_ls5000_roll_scan = MagicMock()

        self.sidebar._on_roll_scan_selected([7])

        request = self.controller.start_ls5000_roll_scan.call_args.args[0]
        self.assertEqual(
            request.filename_pattern,
            '{{ date }}_slot{{ "%02d" % slot }}_{{ "%03d" % seq }}',
        )

    def test_scan_request_binds_explicit_inferred_origin_review_to_preview(self):
        _select_device(self.sidebar, FULL_DEVICE)
        session = _roll_preview_session(
            4,
            inferred_slots=frozenset({2}),
        )
        self.sidebar._on_roll_preview_ready("preview-token", session)
        self.sidebar.folder_edit.setText("/tmp/negpy-roll-reviewed-origin")
        self.sidebar.roll_slot_selector.set_selected_slot_ids([2])
        self.sidebar.roll_slot_selector.boundary_offset_spin.setValue(-17)
        self.sidebar.roll_slot_selector.confirm_slot_thumbnail(2, -17, None)
        self.sidebar.roll_slot_selector.review_approval_check.setChecked(True)
        self.sidebar.roll_slot_selector.set_selected_slot_ids([2, 4])
        self.controller.start_ls5000_roll_scan = MagicMock()

        self.sidebar._on_roll_scan_selected([2, 4])

        request = self.controller.start_ls5000_roll_scan.call_args.args[0]
        self.assertEqual(request.reviewed_fingerprint, session.reviewed_fingerprint())
        choices = {choice.slot_id: choice for choice in request.frames}
        self.assertEqual(
            choices[2].manual_review_approval,
            session.approve_manual_origin(2, -17),
        )
        self.assertIsNone(choices[4].manual_review_approval)

    def test_black_and_white_request_selects_rgb_only_route(self):
        _select_device(self.sidebar, FULL_DEVICE)
        self.sidebar._on_roll_preview_ready("preview-token", _roll_preview_session(6))
        self.sidebar.folder_edit.setText("/tmp/negpy-roll-bw")
        self.sidebar.roll_slot_selector.set_scan_material(ScanMaterial.BLACK_AND_WHITE_NEGATIVE)
        self.controller.start_ls5000_roll_scan = MagicMock()

        self.sidebar._on_roll_scan_selected([1, 6])

        self.controller.start_ls5000_roll_scan.assert_called_once()
        request = self.controller.start_ls5000_roll_scan.call_args.args[0]
        self.assertIsInstance(request, RollScanRequest)
        self.assertIs(request.material, ScanMaterial.BLACK_AND_WHITE_NEGATIVE)
        self.assertFalse(request.material.captures_infrared)
        self.assertEqual(
            [(frame.slot_id, frame.boundary_offset_rows) for frame in request.frames],
            [(1, 0), (6, 0)],
        )
        recipe = self.sidebar.roll_slot_selector.scan_material_status_label.text()
        self.assertIn("4000 dpi", recipe)
        self.assertIn("16-bit", recipe)
        self.assertIn("RGB only, 4×", recipe)
        self.assertIn(
            "IR repair: Off",
            self.sidebar.roll_slot_selector.ir_repair_status_label.text(),
        )

    def test_black_and_white_refuses_missing_sane_capability_but_color_remains_available(
        self,
    ):
        _select_device(self.sidebar, LS5000_WITHOUT_BW_ROUTE)
        self.sidebar._on_roll_preview_ready("preview-token", _roll_preview_session(3))
        self.sidebar.folder_edit.setText("/tmp/negpy-roll-route-gating")
        self.controller.start_ls5000_roll_scan = MagicMock()
        self.sidebar.roll_slot_selector.set_scan_material(ScanMaterial.BLACK_AND_WHITE_NEGATIVE)

        self.sidebar._on_roll_scan_selected([2])

        self.controller.start_ls5000_roll_scan.assert_not_called()
        self.assertFalse(self.sidebar._roll_scanning)
        self.assertIn(
            "requires the patched LS-5000 SANE driver",
            self.sidebar.roll_status_label.text(),
        )

        self.sidebar.roll_slot_selector.set_scan_material(ScanMaterial.COLOR_NEGATIVE)
        self.sidebar._on_roll_scan_selected([2])

        self.controller.start_ls5000_roll_scan.assert_called_once()
        request = self.controller.start_ls5000_roll_scan.call_args.args[0]
        self.assertIs(request.material, ScanMaterial.COLOR_NEGATIVE)


if __name__ == "__main__":
    unittest.main()
