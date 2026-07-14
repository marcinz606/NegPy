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
import json
import sys
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication

from negpy.desktop.controller import AppController
from negpy.desktop.session import DesktopSessionManager, AppState
from negpy.desktop.view.sidebar.scan import ScanSidebar
from negpy.desktop.view.widgets.roll_slot_model import RollSlotModel
from negpy.desktop.workers.ls5000_roll_worker import (
    RollPreviewRequest,
    RollScanRequest,
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
    supported_depths=(16,),
    sources=(ScanMode.NEGATIVE,),
    max_area_mm=(36.0, 24.0),
    multi_sample=True,
    adapter_frame_capacity=40,
    auto_exposure=True,
    registered_geometry=True,
    can_eject=True,
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
)
STOCK_SA21_DEVICE = ScannerDevice(
    id="coolscan3:usb:libusb:001:011",
    vendor="Nikon",
    model="LS-5000",
    capabilities=STOCK_SA21_CAPS,
)


def _roll_preview_session(count: int = 40):
    """Build a tiny decoded roll index with no scanner or filesystem I/O."""

    thumbnail = np.arange(6 * 8 * 3, dtype=np.uint16).reshape(6, 8, 3)
    return SimpleNamespace(
        slots=tuple(
            SimpleNamespace(
                slot_id=slot_id,
                thumbnail=thumbnail,
                warnings=("Likely blank tail slot",) if slot_id == count else (),
                boundary_offset_rows=0,
            )
            for slot_id in range(1, count + 1)
        )
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
    def test_checking_archival_forces_and_locks_ir_and_samples(self):
        _select_device(self.sidebar, FULL_DEVICE)
        self.sidebar.ir_check.setChecked(False)

        self.sidebar.archival_split_check.setChecked(True)

        self.assertTrue(self.sidebar.ir_check.isChecked())
        self.assertEqual(self.sidebar.samples_combo.currentData(), 4)
        self.assertFalse(self.sidebar.ir_check.isEnabled())
        self.assertFalse(self.sidebar.samples_combo.isEnabled())

    def test_unchecking_archival_restores_capability_derived_enabled_state(self):
        _select_device(self.sidebar, FULL_DEVICE)
        self.sidebar.archival_split_check.setChecked(True)

        self.sidebar.archival_split_check.setChecked(False)

        self.assertTrue(self.sidebar.ir_check.isEnabled())
        self.assertTrue(self.sidebar.samples_combo.isEnabled())

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
        self.assertIn("whole roll", self.sidebar.roll_status_label.text())

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
        replacement = np.full((4, 96, 3), 12_345, dtype=np.uint16)
        self.sidebar._on_roll_thumbnail_ready(3, -23, replacement)
        self.assertTrue(self.sidebar.roll_slot_selector.scan_button.isEnabled())

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
        recipe = self.sidebar.roll_slot_selector.scan_material_status_label.text()
        self.assertIn("4000 dpi", recipe)
        self.assertIn("16-bit", recipe)
        self.assertIn("RGB 4× + IR", recipe)
        self.assertIn("IR dust repair available", recipe)

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
        self.assertIn("IR/dust repair off", recipe)

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
