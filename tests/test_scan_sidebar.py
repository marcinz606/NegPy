"""Offline tests for the generic Scan sidebar.

Constructs the real ScanSidebar against a light fake controller (no AppController,
no GPU, no live SANE) under an offscreen Qt platform, with fabricated
ScannerCapabilities. Proves the generic controls capability-gate, the depth combo
defaults correctly on a 14-bit scanner, the frame-range batch routes correctly,
and a non-Coolscan device hides every Coolscan-only control (the multi-backend
invariant).
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import sys
from dataclasses import replace
from types import SimpleNamespace

import pytest
from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtWidgets import QApplication

from negpy.desktop.view.sidebar.scan import ScanSidebar
from negpy.infrastructure.scanners.base import ScannerCapabilities, ScannerDevice
from negpy.infrastructure.scanners.params import ScanMode

if not QApplication.instance():
    _app = QApplication(sys.argv)


FULL_CAPS = ScannerCapabilities(
    ir_channel=True,
    supported_dpi=(1000, 4000),
    supported_depths=(8, 16),
    sources=(ScanMode.NEGATIVE,),
    max_area_mm=(36.0, 24.0),
    auto_exposure=True,
    autofocus=True,
    adapter_frame_capacity=40,
    adapter_frame_control=True,
    can_eject=True,
)
FULL_DEVICE = ScannerDevice(id="coolscan3:usb:libusb:001:007", vendor="Nikon", model="LS-5000", capabilities=FULL_CAPS)

# A real LS-50 ED / Coolscan V: 14-bit (not 16), no IR on stock SANE, 6-frame SA-21.
LS50_CAPS = ScannerCapabilities(
    ir_channel=False,
    supported_dpi=(1000, 4000),
    supported_depths=(8, 14),
    sources=(ScanMode.NEGATIVE,),
    max_area_mm=(25.0571, 37.83965),
    auto_exposure=True,
    autofocus=True,
    adapter_frame_capacity=6,
    adapter_frame_control=True,
    can_eject=True,
)
LS50_DEVICE = ScannerDevice(id="coolscan3:usb:libusb:001:050", vendor="Nikon", model="LS-50 ED", capabilities=LS50_CAPS)

# A plain Plustek film scanner without Prescan (locked-out / non-SE model).
MINIMAL_CAPS = ScannerCapabilities(
    ir_channel=False,
    supported_dpi=(1200, 2400),
    supported_depths=(16,),
    sources=(ScanMode.NEGATIVE,),
    max_area_mm=(36.0, 24.0),
)
MINIMAL_DEVICE = ScannerDevice(id="plustek:libusb:001:008", vendor="Plustek", model="OpticFilm", capabilities=MINIMAL_CAPS)

# OpticFilm 8200i SE: Prescan + IR, single depth, no roll feeder.
SE_CAPS = ScannerCapabilities(
    ir_channel=True,
    supported_dpi=(1200, 1800, 3600),
    supported_depths=(16,),
    sources=(ScanMode.TRANSPARENCY,),
    max_area_mm=(36.33, 25.0),
    prescan=True,
    prescan_dpi=1200,
    prescan_mirror_x=True,
    prescan_default_crop=(0.0, 0.35, 1.0, 0.65),
)
SE_DEVICE = ScannerDevice(
    id="plustek:usb:07b3:1825:002:006",
    vendor="PLUSTEK",
    model="OpticFilm 8200i SE",
    capabilities=SE_CAPS,
)


class _FakeRepo:
    def __init__(self, settings: dict | None = None) -> None:
        self._store: dict = {"scanner_settings": settings} if settings else {}

    def get_global_setting(self, key: str, default=None):
        return self._store.get(key, default)

    def save_global_setting(self, key: str, value) -> None:
        self._store[key] = value


class _FakeController(QObject):
    scan_devices_ready = pyqtSignal(list)
    scan_progress = pyqtSignal(float, str)  # progress, phase name
    scan_finished = pyqtSignal(str)
    scan_error = pyqtSignal(str)
    scan_cancelled = pyqtSignal()
    scan_frame_done = pyqtSignal(int, str)
    scan_batch_finished = pyqtSignal(list)
    scan_ejected = pyqtSignal(bool)
    scan_eject_error = pyqtSignal(str)

    def __init__(self, settings: dict | None = None) -> None:
        super().__init__()
        self.session = SimpleNamespace(repo=_FakeRepo(settings))
        self.started: list[tuple[str, object]] = []
        self.ejected_ids: list[str] = []
        self.device_requests = 0
        self.backend_requests: list[str] = []
        self.cancels = 0

    def request_scan_devices(self) -> None:
        self.device_requests += 1

    def set_scan_backend(self, backend_id: str) -> None:
        self.backend_requests.append(backend_id)

    def start_scan(self, req) -> None:
        self.started.append(("scan", req))

    def start_batch(self, req) -> None:
        self.started.append(("batch", req))

    def cancel_scan(self) -> None:
        self.cancels += 1

    def eject_scanner(self, device_id: str) -> None:
        self.ejected_ids.append(device_id)


def _sidebar(device: ScannerDevice | None = None, settings: dict | None = None) -> tuple[ScanSidebar, _FakeController]:
    controller = _FakeController(settings)
    sidebar = ScanSidebar(controller)
    if device is not None:
        sidebar._on_devices_ready([device])
    return sidebar, controller


def test_controller_signals_connect_without_error() -> None:
    # Construction wires every scan_* signal; a missing one would raise here.
    sidebar, _ = _sidebar()
    assert sidebar is not None


def test_no_device_disables_controls() -> None:
    sidebar, _ = _sidebar()
    sidebar._update_device_caps()  # no device selected
    assert sidebar.scan_btn.isEnabled() is False
    assert sidebar.eject_btn.isVisibleTo(sidebar) is False
    assert sidebar.frame_range_widget.isVisibleTo(sidebar) is False
    assert sidebar.ae_check.isVisibleTo(sidebar) is False
    assert sidebar.autofocus_check.isVisibleTo(sidebar) is False
    assert sidebar.depth_combo.isVisibleTo(sidebar) is False
    assert sidebar.depth_label.isVisibleTo(sidebar) is False
    assert sidebar.scan_window_widget.isVisibleTo(sidebar) is False


def test_full_capability_device_enables_coolscan_controls() -> None:
    sidebar, _ = _sidebar(FULL_DEVICE)
    assert sidebar.ir_check.isEnabled() is True
    assert sidebar.ae_check.isVisibleTo(sidebar) is True
    assert sidebar.autofocus_check.isVisibleTo(sidebar) is True
    assert sidebar.eject_btn.isVisibleTo(sidebar) is True
    assert sidebar.frame_range_widget.isVisibleTo(sidebar) is True
    assert sidebar.frame_from_spin.maximum() == 40
    assert sidebar.frame_to_spin.maximum() == 40
    assert sidebar.depth_combo.isVisibleTo(sidebar) is True
    assert sidebar.depth_label.isVisibleTo(sidebar) is True


def test_minimal_device_hides_coolscan_controls() -> None:
    # The multi-backend invariant: a plain Plustek shows none of the Coolscan
    # controls and still scans.
    sidebar, _ = _sidebar(MINIMAL_DEVICE)
    assert sidebar.ir_check.isEnabled() is False
    assert sidebar.ae_check.isVisibleTo(sidebar) is False
    assert sidebar.autofocus_check.isVisibleTo(sidebar) is False
    assert sidebar.eject_btn.isVisibleTo(sidebar) is False
    assert sidebar.frame_range_widget.isVisibleTo(sidebar) is False
    assert sidebar.depth_combo.isVisibleTo(sidebar) is False
    assert sidebar.depth_label.isVisibleTo(sidebar) is False
    assert sidebar.depth_combo.currentData() == 16
    assert sidebar.prescan_widget.isVisibleTo(sidebar) is False
    assert sidebar.scan_btn.isEnabled() is True


def test_se_device_shows_prescan() -> None:
    sidebar, _ = _sidebar(SE_DEVICE)
    assert sidebar.prescan_widget.isVisibleTo(sidebar) is True
    assert sidebar.prescan_label.isVisibleTo(sidebar) is True
    assert sidebar.ir_check.isEnabled() is True
    assert sidebar.frame_range_widget.isVisibleTo(sidebar) is False


def test_scan_params_include_prescan_crop() -> None:
    sidebar, controller = _sidebar(SE_DEVICE, settings={"scan_window": (0.1, 0.2, 0.9, 0.8)})
    sidebar.folder_edit.setText("/tmp/negpy-scan-out")
    sidebar._on_scan()
    kind, req = controller.started[0]
    assert kind == "scan"
    assert req.params.window == (0.1, 0.2, 0.9, 0.8)


def test_minimal_device_still_gets_a_single_shot_preview_window_control() -> None:
    # No frame adapter to page through, but a single manual holder still gets a
    # quick low-res preview to set one crop window before the real scan.
    sidebar, _ = _sidebar(MINIMAL_DEVICE)
    assert sidebar.scan_window_widget.isVisibleTo(sidebar) is True
    assert sidebar.scan_window_btn.text() == "Preview…"
    assert sidebar.scan_window_row_label.text() == "Window"


def test_full_capability_device_gets_the_strip_preview_window_control() -> None:
    sidebar, _ = _sidebar(FULL_DEVICE)
    assert sidebar.scan_window_widget.isVisibleTo(sidebar) is True
    assert sidebar.scan_window_btn.text() == "Preview strip…"
    assert sidebar.scan_window_row_label.text() == "Batch"


def test_minimal_device_scan_window_opens_the_quick_preview_dialog(monkeypatch) -> None:
    sidebar, _ = _sidebar(MINIMAL_DEVICE)
    rect = (0.2, 0.2, 0.8, 0.8)

    class _FakeDialog:
        def __init__(self, controller, device, initial_window=None, parent=None) -> None:
            self.seen = (controller, device, initial_window)

        def exec(self) -> bool:
            return True

        def window(self):
            return rect

        def scan_requested(self) -> bool:
            return False

    monkeypatch.setattr("negpy.desktop.view.widgets.quick_scan_preview_dialog.QuickScanPreviewDialog", _FakeDialog)

    sidebar._on_set_scan_window()

    assert sidebar._settings.scan_window == rect


def test_full_capability_device_scan_window_still_opens_the_strip_dialog(monkeypatch) -> None:
    sidebar, _ = _sidebar(FULL_DEVICE)
    opened: list = []

    class _FakeDialog:
        def __init__(self, controller, device, **kwargs) -> None:
            opened.append(device)

        def exec(self) -> bool:
            return False  # cancelled — settings must stay untouched

    monkeypatch.setattr("negpy.desktop.view.widgets.strip_preview_dialog.StripPreviewDialog", _FakeDialog)

    sidebar._on_set_scan_window()

    assert opened == [FULL_DEVICE]


def test_14_bit_device_defaults_to_14_not_8() -> None:
    sidebar, _ = _sidebar(LS50_DEVICE)
    # Saved default depth 16 is not offered on an (8, 14) scanner; the combo must
    # land on the deepest supported, never silently on index 0 = 8-bit.
    assert sidebar.depth_combo.isVisibleTo(sidebar) is True
    assert sidebar.depth_combo.currentData() == 14


def test_saved_depth_wins_when_the_device_offers_it() -> None:
    sidebar, _ = _sidebar(LS50_DEVICE, settings={"depth": 8})
    assert sidebar.depth_combo.currentData() == 8


def test_frame_range_keeps_from_not_after_to() -> None:
    sidebar, _ = _sidebar(FULL_DEVICE)
    sidebar.frame_from_spin.setValue(5)
    sidebar.frame_to_spin.setValue(3)
    assert sidebar.frame_from_spin.value() == 3
    assert sidebar.frame_to_spin.value() == 3


def test_scan_on_capacity_device_routes_to_batch() -> None:
    sidebar, controller = _sidebar(FULL_DEVICE)
    sidebar.folder_edit.setText("/tmp/negpy-scan-out")
    sidebar.frame_from_spin.setValue(2)
    sidebar.frame_to_spin.setValue(4)

    sidebar._on_scan()

    assert len(controller.started) == 1
    kind, req = controller.started[0]
    assert kind == "batch"
    assert req.frames == (2, 3, 4)
    assert req.frame_windows == {}
    assert req.device_id == FULL_DEVICE.id


def test_scan_on_plain_device_routes_to_single() -> None:
    sidebar, controller = _sidebar(MINIMAL_DEVICE)
    sidebar.folder_edit.setText("/tmp/negpy-scan-out")

    sidebar._on_scan()

    assert len(controller.started) == 1
    kind, req = controller.started[0]
    assert kind == "scan"
    assert req.params.frame is None


def test_scan_uses_dialog_selection_and_per_frame_windows() -> None:
    sidebar, controller = _sidebar(LS50_DEVICE)
    sidebar.folder_edit.setText("/tmp/negpy-scan-out")
    rect = (0.1, 0.1, 0.5, 0.5)
    sidebar.settings = replace(sidebar._settings, selected_frames=(1, 2, 4), frame_windows={4: rect})

    sidebar._on_scan()

    kind, req = controller.started[0]
    assert kind == "batch"
    assert req.frames == (1, 2, 4)
    assert req.frame_windows == {4: rect}


def test_clear_scan_window_reverts_to_spinbox_mode() -> None:
    sidebar, _ = _sidebar(LS50_DEVICE)
    sidebar.settings = replace(sidebar._settings, selected_frames=(1, 3), frame_windows={1: (0.0, 0.0, 1.0, 1.0)})

    sidebar._on_clear_scan_window()

    assert sidebar._settings.selected_frames == ()
    assert sidebar._settings.frame_windows == {}


def test_ui_edit_preserves_dialog_selection() -> None:
    sidebar, _ = _sidebar(LS50_DEVICE)
    rect = (0.1, 0.1, 0.5, 0.5)
    sidebar.settings = replace(sidebar._settings, selected_frames=(1, 2, 4), frame_windows={4: rect})

    sidebar.folder_edit.setText("/tmp/somewhere-else")  # fires _update_settings_from_ui

    assert sidebar._settings.selected_frames == (1, 2, 4)
    assert sidebar._settings.frame_windows == {4: rect}


def test_backend_combo_lists_registry_default() -> None:
    from negpy.infrastructure.scanners.registry import DEFAULT_BACKEND_ID

    sidebar, _ = _sidebar()
    assert sidebar.backend_combo.findData("plustek") >= 0
    assert sidebar._current_backend_id() == DEFAULT_BACKEND_ID


def test_request_devices_routes_the_backend() -> None:
    from negpy.infrastructure.scanners.registry import DEFAULT_BACKEND_ID

    sidebar, controller = _sidebar()
    controller.device_requests = 0
    sidebar._request_devices()
    assert controller.backend_requests[-1] == DEFAULT_BACKEND_ID
    assert controller.device_requests == 1


def test_load_settings_coerces_unavailable_backend(monkeypatch) -> None:
    """A saved Unix-only backend must not stick on Windows."""
    import sys

    if sys.platform != "win32":
        pytest.skip("Windows-only coercion")

    from negpy.infrastructure.scanners.registry import DEFAULT_BACKEND_ID
    from negpy.infrastructure.scanners.settings import ScannerSettings

    sidebar, controller = _sidebar()
    controller.session.repo.save_global_setting(
        "scanner_settings",
        {"backend": "sane", "dpi": 1800, "depth": 16},
    )
    loaded = sidebar._load_settings()
    assert loaded.backend == DEFAULT_BACKEND_ID
    assert isinstance(loaded, ScannerSettings)


def test_backend_change_persists_and_re_enumerates(monkeypatch) -> None:
    from negpy.desktop.view.sidebar import scan as scan_mod

    monkeypatch.setattr(scan_mod, "backend_choices", lambda: [("sane", "SANE"), ("mock", "Mock")])
    sidebar, controller = _sidebar()
    before = controller.device_requests

    sidebar.backend_combo.setCurrentIndex(1)  # fires _on_backend_changed

    assert sidebar._current_backend_id() == "mock"
    assert controller.backend_requests[-1] == "mock"
    assert controller.device_requests == before + 1
    assert controller.session.repo.get_global_setting("scanner_settings")["backend"] == "mock"


def test_ui_edit_preserves_offset_and_drift() -> None:
    sidebar, _ = _sidebar(LS50_DEVICE)
    sidebar.settings = replace(sidebar._settings, frame_offset_mm=1.5, frame_offset_modifier_mm=-0.1)

    sidebar.folder_edit.setText("/tmp/somewhere-else")  # fires _update_settings_from_ui

    assert sidebar._settings.frame_offset_mm == 1.5
    assert sidebar._settings.frame_offset_modifier_mm == -0.1


def test_scan_carries_offset_and_drift_into_the_batch_request() -> None:
    # _on_scan() re-reads settings from the UI right before building the
    # request — the rebuild must not wipe dialog-owned fields.
    sidebar, controller = _sidebar(LS50_DEVICE)
    sidebar.folder_edit.setText("/tmp/negpy-scan-out")
    sidebar.settings = replace(sidebar._settings, frame_offset_mm=1.5, frame_offset_modifier_mm=0.2)

    sidebar._on_scan()

    kind, req = controller.started[0]
    assert kind == "batch"
    assert req.params.frame_offset_mm == 1.5
    assert req.frame_offset_modifier_mm == 0.2


def test_eject_button_calls_controller() -> None:
    sidebar, controller = _sidebar(FULL_DEVICE)
    sidebar._on_eject()
    assert controller.ejected_ids == [FULL_DEVICE.id]


def test_ae_flag_flows_into_scan_params() -> None:
    sidebar, controller = _sidebar(FULL_DEVICE)
    sidebar.folder_edit.setText("/tmp/negpy-scan-out")
    sidebar.ae_check.setChecked(True)
    sidebar.autofocus_check.setChecked(True)

    sidebar._on_scan()

    _kind, req = controller.started[0]
    assert req.params.auto_exposure is True
    assert req.params.autofocus is True


def test_unsupported_ae_af_forced_off_in_scan_params() -> None:
    sidebar, controller = _sidebar(MINIMAL_DEVICE)
    sidebar.folder_edit.setText("/tmp/negpy-scan-out")
    sidebar.ae_check.setChecked(True)
    sidebar.autofocus_check.setChecked(True)

    sidebar._on_scan()

    _kind, req = controller.started[0]
    assert req.params.auto_exposure is False
    assert req.params.autofocus is False


def test_scan_error_resets_ui_and_shows_status(monkeypatch) -> None:
    sidebar, _ = _sidebar(SE_DEVICE)
    sidebar.set_scanning(True)
    sidebar.progress_bar.setVisible(True)
    popped: list[tuple[str, str]] = []

    def _fake_warning(parent, title, text, *args, **kwargs):
        del parent, args, kwargs
        popped.append((title, text))
        return 0

    monkeypatch.setattr("negpy.desktop.view.sidebar.scan.QMessageBox.warning", _fake_warning)
    sidebar._on_scan_error("USB I/O failed")

    assert sidebar._scanning is False
    assert sidebar.progress_bar.isVisible() is False
    assert "Error: USB I/O failed" in sidebar.status_label.text()
    assert popped == []


def test_lockout_scan_error_shows_message_box(monkeypatch) -> None:
    sidebar, _ = _sidebar(SE_DEVICE)
    popped: list[tuple[str, str]] = []

    def _fake_warning(parent, title, text, *args, **kwargs):
        del parent, args, kwargs
        popped.append((title, text))
        return 0

    monkeypatch.setattr("negpy.desktop.view.sidebar.scan.QMessageBox.warning", _fake_warning)
    msg = "OpticFilm 8100 (GL845) cannot scan with pyOpticfilm in this release — only OpticFilm 8200i SE is validated."
    sidebar._on_scan_error(msg)

    assert f"Error: {msg}" in sidebar.status_label.text()
    assert popped == [("Scan failed", msg)]
