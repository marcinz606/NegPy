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
import re

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import sys
from dataclasses import replace
from types import SimpleNamespace

import pytest
from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtGui import QValidator
from PyQt6.QtWidgets import QApplication

from negpy.desktop.view.sidebar.scan import ScanSidebar, estimated_frame_bytes
from negpy.desktop.view.styles.theme import THEME
from negpy.infrastructure.scanners.base import ScannerCapabilities, ScannerDevice
from negpy.infrastructure.scanners.params import FILM_TYPES, ScanMode

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
    multi_exposure=True,
    prescan_default_crop=(0.0, 0.35, 1.0, 0.65),
)
SE_DEVICE = ScannerDevice(
    id="plustek:usb:07b3:1825:002:006",
    vendor="PLUSTEK",
    model="OpticFilm 8200i SE",
    capabilities=SE_CAPS,
)


# A Coolscan over nkscan: it measures the strip, cleans dust itself and multi-samples.
NKSCAN_CAPS = ScannerCapabilities(
    ir_channel=True,
    supported_dpi=(1000, 4000),
    supported_depths=(16,),
    sources=(ScanMode.NEGATIVE, ScanMode.POSITIVE),
    max_area_mm=(24.0, 36.0),
    can_eject=True,
    hw_clean=True,
    roll_discovery=True,
    film_formats=("135", "66"),
    film_types=tuple(FILM_TYPES),
    max_samples=16,
    superfine=True,
)
NKSCAN_DEVICE = ScannerDevice(id="usb:1-3.2", vendor="Nikon", model="LS-50", capabilities=NKSCAN_CAPS)


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


def _summary(sidebar: ScanSidebar) -> str:
    """The summary line as the eye reads it: the count and size carry markup for weight."""
    return re.sub(r"<[^>]+>", "", sidebar.status_strip._summary.text())


def test_controller_signals_connect_without_error() -> None:
    # Construction wires every scan_* signal; a missing one would raise here.
    sidebar, _ = _sidebar()
    assert sidebar is not None


def test_no_device_disables_controls() -> None:
    sidebar, _ = _sidebar()
    sidebar._update_device_caps()  # no device selected
    assert sidebar.scan_btn.isEnabled() is False
    assert sidebar.eject_btn.isVisibleTo(sidebar) is False
    assert sidebar.frame_spec_edit.isVisibleTo(sidebar) is False
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
    assert sidebar.frame_spec_edit.isVisibleTo(sidebar) is True
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
    assert sidebar.frame_spec_edit.isVisibleTo(sidebar) is False
    assert sidebar.depth_combo.isVisibleTo(sidebar) is False
    assert sidebar.depth_label.isVisibleTo(sidebar) is False
    assert sidebar.depth_combo.currentData() == 16
    assert sidebar.prescan_widget.isVisibleTo(sidebar) is False
    assert sidebar.scan_btn.isEnabled() is True


def test_se_device_shows_prescan() -> None:
    sidebar, _ = _sidebar(SE_DEVICE, settings={"backend": "plustek"})
    assert sidebar.prescan_widget.isVisibleTo(sidebar) is True
    assert sidebar.prescan_label.isVisibleTo(sidebar) is True
    assert sidebar.scan_window_widget.isVisibleTo(sidebar) is False
    assert sidebar.ir_check.isEnabled() is True
    assert sidebar.me_check.isEnabled() is True
    assert sidebar.frame_spec_edit.isVisibleTo(sidebar) is False


def test_plustek_backend_hides_sane_window_control() -> None:
    sidebar, _ = _sidebar(MINIMAL_DEVICE, settings={"backend": "plustek"})
    assert sidebar.scan_window_widget.isVisibleTo(sidebar) is False
    assert sidebar.scan_window_status.isVisibleTo(sidebar) is False


def test_sane_backend_keeps_single_holder_window_control(monkeypatch) -> None:
    sidebar, _ = _sidebar(MINIMAL_DEVICE, settings={"backend": "plustek"})
    monkeypatch.setattr(sidebar, "_current_backend_id", lambda: "sane")
    sidebar._update_device_caps()
    assert sidebar.scan_window_widget.isVisibleTo(sidebar) is True
    assert sidebar.scan_window_btn.text() == "Preview…"
    assert sidebar.scan_window_row_label.text() == "Window"


def test_minimal_device_disables_multi_exposure() -> None:
    sidebar, _ = _sidebar(MINIMAL_DEVICE)
    assert sidebar.me_check.isEnabled() is False
    assert sidebar.me_check.isChecked() is False


def test_scan_params_include_prescan_crop() -> None:
    sidebar, controller = _sidebar(
        SE_DEVICE,
        settings={"backend": "plustek", "scan_window": (0.1, 0.2, 0.9, 0.8)},
    )
    sidebar.folder_edit.setText("/tmp/negpy-scan-out")
    sidebar._on_scan()
    kind, req = controller.started[0]
    assert kind == "scan"
    assert req.params.window == (0.1, 0.2, 0.9, 0.8)


def test_full_capability_device_gets_the_strip_preview_window_control(monkeypatch) -> None:
    sidebar, _ = _sidebar(FULL_DEVICE, settings={"backend": "plustek"})
    monkeypatch.setattr(sidebar, "_current_backend_id", lambda: "sane")
    sidebar._update_device_caps()
    assert sidebar.scan_window_widget.isVisibleTo(sidebar) is True
    assert sidebar.scan_window_btn.text() == "Preview strip…"
    assert sidebar.scan_window_row_label.text() == "Batch"


def test_sane_backend_scan_window_opens_the_quick_preview_dialog(monkeypatch) -> None:
    sidebar, _ = _sidebar(MINIMAL_DEVICE, settings={"backend": "plustek"})
    monkeypatch.setattr(sidebar, "_current_backend_id", lambda: "sane")
    sidebar._update_device_caps()
    rect = (0.2, 0.2, 0.8, 0.8)

    class _FakeDialog:
        def __init__(self, controller, device, initial_window=None, film_type="negative", parent=None) -> None:
            self.seen = (controller, device, initial_window, film_type)

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
    sidebar, _ = _sidebar(FULL_DEVICE, settings={"backend": "plustek"})
    monkeypatch.setattr(sidebar, "_current_backend_id", lambda: "sane")
    sidebar._update_device_caps()
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


def test_an_unreadable_frame_list_refuses_the_scan() -> None:
    sidebar, controller = _sidebar(FULL_DEVICE)
    sidebar.folder_edit.setText("/tmp/negpy-scan-out")
    sidebar.frame_spec_edit.setText("2-")

    assert sidebar.scan_btn.isEnabled() is False
    sidebar._on_scan()
    assert controller.started == []

    sidebar.frame_spec_edit.setText("2-4")
    assert sidebar.scan_btn.isEnabled() is True


def test_scan_on_capacity_device_routes_to_batch() -> None:
    sidebar, controller = _sidebar(FULL_DEVICE)
    sidebar.folder_edit.setText("/tmp/negpy-scan-out")
    sidebar.frame_spec_edit.setText("2-4")

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
    popped: list[tuple[str, str]] = []

    def _fake_warning(parent, title, text, *args, **kwargs):
        del parent, args, kwargs
        popped.append((title, text))
        return 0

    monkeypatch.setattr("negpy.desktop.view.sidebar.scan.QMessageBox.warning", _fake_warning)
    sidebar._on_scan_error("USB I/O failed")

    assert sidebar._scanning is False
    assert sidebar.status_strip.showing() == "message"
    assert "Error: USB I/O failed" in sidebar.status_strip.message()
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

    assert f"Error: {msg}" in sidebar.status_strip.message()
    assert popped == [("Scan failed", msg)]


# ── nkscan-only controls ──────────────────────────────────────────────────


def test_a_measured_strip_device_shows_the_nkscan_controls() -> None:
    sidebar, _ = _sidebar(NKSCAN_DEVICE)
    assert sidebar.clean_check.isVisibleTo(sidebar) is True
    assert sidebar.superfine_check.isVisibleTo(sidebar) is True
    assert sidebar.samples_combo.isVisibleTo(sidebar) is True
    assert sidebar.format_combo.isVisibleTo(sidebar) is True
    assert [sidebar.samples_combo.itemData(i) for i in range(sidebar.samples_combo.count())] == [1, 2, 4, 8, 16]
    assert [sidebar.format_combo.itemData(i) for i in range(sidebar.format_combo.count())] == [None, "135", "66"]


def test_a_sane_device_shows_none_of_them() -> None:
    sidebar, _ = _sidebar(FULL_DEVICE)
    assert sidebar.clean_check.isVisibleTo(sidebar) is False
    assert sidebar.superfine_check.isVisibleTo(sidebar) is False
    assert sidebar.samples_combo.isVisibleTo(sidebar) is False
    assert sidebar.format_combo.isVisibleTo(sidebar) is False


def test_a_measured_strip_offers_a_frame_list_and_the_strip_dialog() -> None:
    sidebar, _ = _sidebar(NKSCAN_DEVICE)
    assert sidebar.frame_spec_edit.isVisibleTo(sidebar) is True
    assert sidebar.scan_window_btn.text() == "Preview strip…"


def test_a_measured_strip_scans_as_a_batch() -> None:
    sidebar, controller = _sidebar(NKSCAN_DEVICE)
    sidebar.folder_edit.setText("/tmp/negpy-test")
    sidebar._on_scan()
    assert [kind for kind, _req in controller.started] == ["batch"]


def test_the_nkscan_options_reach_the_request() -> None:
    sidebar, controller = _sidebar(NKSCAN_DEVICE)
    sidebar.folder_edit.setText("/tmp/negpy-test")
    sidebar.clean_check.setChecked(True)
    sidebar.superfine_check.setChecked(True)
    sidebar.samples_combo.setCurrentIndex(sidebar.samples_combo.findData(4))
    sidebar.format_combo.setCurrentIndex(sidebar.format_combo.findData("66"))

    sidebar._on_scan()
    params = controller.started[-1][1].params
    assert (params.clean, params.superfine, params.samples, params.film_format) == (True, True, 4, "66")


def test_a_saved_clean_never_reaches_a_device_that_cannot_do_it() -> None:
    """The same guard as autofocus: a stale saved value must not become a refused option."""
    sidebar, controller = _sidebar(FULL_DEVICE, settings={"clean": True, "superfine": True, "samples": 8})
    sidebar.folder_edit.setText("/tmp/negpy-test")
    sidebar._on_scan()

    params = controller.started[-1][1].params
    assert (params.clean, params.superfine, params.samples, params.film_format) == (False, False, 1, None)


def test_the_nkscan_options_persist() -> None:
    sidebar, _ = _sidebar(NKSCAN_DEVICE)
    sidebar.clean_check.setChecked(True)
    sidebar.samples_combo.setCurrentIndex(sidebar.samples_combo.findData(2))
    assert (sidebar.settings.clean, sidebar.settings.samples) == (True, 2)


def test_a_measured_strip_says_what_scan_would_do_before_a_preview() -> None:
    """Nothing is cropped yet, and the summary quotes the size of one frame."""
    sidebar, _ = _sidebar(NKSCAN_DEVICE)
    assert sidebar.scan_window_status.text().startswith("Full frame")
    assert _summary(sidebar).startswith("Whole strip")


def test_a_typed_frame_list_reaches_the_batch_without_a_preview() -> None:
    sidebar, controller = _sidebar(NKSCAN_DEVICE)
    sidebar.folder_edit.setText("/tmp/negpy-test")
    sidebar.frame_spec_edit.setText("1,3-5")

    sidebar._on_scan()

    assert controller.started[-1][1].frames == (1, 3, 4, 5)
    assert sidebar.settings.selected_frames == (1, 3, 4, 5)


def test_a_selection_from_the_strip_dialog_shows_in_the_frame_box() -> None:
    sidebar, _ = _sidebar(NKSCAN_DEVICE, settings={"selected_frames": [1, 2, 3, 6]})
    assert sidebar.frame_spec_edit.text() == "1-3,6"


def test_a_measured_strip_scans_the_frames_the_strip_dialog_picked() -> None:
    sidebar, controller = _sidebar(NKSCAN_DEVICE, settings={"selected_frames": [2, 4]})
    assert sidebar.frame_spec_edit.text() == "2,4"
    sidebar.folder_edit.setText("/tmp/negpy-test")
    sidebar._on_scan()
    assert controller.started[-1][1].frames == (2, 4)


def test_a_measured_strip_with_nothing_picked_scans_the_whole_strip() -> None:
    """Its frame count is unknown until the film is measured, so the batch names no frames."""
    sidebar, controller = _sidebar(NKSCAN_DEVICE)
    sidebar.folder_edit.setText("/tmp/negpy-test")
    sidebar._on_scan()

    kind, req = controller.started[-1]
    assert kind == "batch" and req.frames == ()


def test_a_feeder_with_nothing_picked_still_uses_its_frame_range() -> None:
    sidebar, controller = _sidebar(LS50_DEVICE)
    sidebar.folder_edit.setText("/tmp/negpy-test")
    sidebar._on_scan()

    assert controller.started[-1][1].frames == tuple(range(1, 7))


def test_selecting_a_device_keeps_the_saved_capability_gated_settings() -> None:
    """Populating the form must not persist a value read before the caps flags catch up."""
    saved = {"backend": "nkscan", "clean": True, "superfine": True, "samples": 8}
    sidebar, _ = _sidebar(NKSCAN_DEVICE, settings=saved)

    assert (sidebar.settings.clean, sidebar.settings.superfine, sidebar.settings.samples) == (True, True, 8)
    assert sidebar.clean_check.isChecked() and sidebar.superfine_check.isChecked()
    assert sidebar.samples_combo.currentData() == 8


def test_selecting_a_device_keeps_a_saved_auto_exposure() -> None:
    sidebar, _ = _sidebar(FULL_DEVICE, settings={"auto_exposure": True, "autofocus": False})
    assert sidebar.settings.auto_exposure is True
    assert sidebar.settings.autofocus is False


# ── what is on the film ───────────────────────────────────────────────────


def test_the_film_types_the_transport_takes_are_offered() -> None:
    sidebar, _ = _sidebar(NKSCAN_DEVICE)
    labels = [sidebar.film_type_combo.itemText(i) for i in range(sidebar.film_type_combo.count())]
    assert labels == ["Color negative", "B&W negative", "Slide", "Kodachrome"]
    assert sidebar.film_type_combo.currentData() == "negative"


def test_a_device_that_is_told_nothing_about_the_film_hides_the_control() -> None:
    sidebar, _ = _sidebar(FULL_DEVICE)
    assert sidebar.film_type_combo.isVisibleTo(sidebar) is False


def test_a_film_that_blocks_infrared_disables_ir_and_ice() -> None:
    """Silver grain stops infrared as it stops light, so the mask is the picture again."""
    sidebar, _ = _sidebar(NKSCAN_DEVICE)
    sidebar.ir_check.setChecked(True)
    sidebar.clean_check.setChecked(True)

    sidebar.film_type_combo.setCurrentIndex(sidebar.film_type_combo.findData("mono"))

    assert sidebar.ir_check.isEnabled() is False and sidebar.ir_check.isChecked() is False
    assert sidebar.clean_check.isEnabled() is False and sidebar.clean_check.isChecked() is False
    assert "blocks infrared" in sidebar.ir_check.toolTip()


def test_ir_and_ice_untick_each_other() -> None:
    """Both read the same pass, and ICE bakes its repair in, so only one can be asked for."""
    sidebar, _ = _sidebar(NKSCAN_DEVICE)

    sidebar.ir_check.setChecked(True)
    sidebar.clean_check.setChecked(True)
    assert (sidebar.ir_check.isChecked(), sidebar.clean_check.isChecked()) == (False, True)
    assert (sidebar.settings.capture_ir, sidebar.settings.clean) == (False, True)

    sidebar.ir_check.setChecked(True)
    assert (sidebar.ir_check.isChecked(), sidebar.clean_check.isChecked()) == (True, False)
    assert (sidebar.settings.capture_ir, sidebar.settings.clean) == (True, False)


def test_a_saved_ir_and_ice_pair_comes_back_as_ice_alone() -> None:
    saved = {"backend": "nkscan", "capture_ir": True, "clean": True}
    sidebar, _ = _sidebar(NKSCAN_DEVICE, settings=saved)
    assert (sidebar.ir_check.isChecked(), sidebar.clean_check.isChecked()) == (False, True)


def test_kodachrome_blocks_infrared_too() -> None:
    sidebar, _ = _sidebar(NKSCAN_DEVICE)
    sidebar.film_type_combo.setCurrentIndex(sidebar.film_type_combo.findData("kodachrome"))
    assert sidebar.clean_check.isEnabled() is False


def test_a_saved_bw_film_greys_ir_and_ice_out_on_the_device_it_is_switched_to() -> None:
    """The film gate reads the ICE capability and the film list, so it runs after both."""
    sidebar, _ = _sidebar(settings={"backend": "nkscan", "film_type": "mono"})
    sidebar._on_devices_ready([FULL_DEVICE, NKSCAN_DEVICE])
    sidebar.device_combo.setCurrentIndex(1)

    assert sidebar.film_type_combo.currentData() == "mono"
    assert sidebar.ir_check.isEnabled() is False and sidebar.ir_check.isChecked() is False
    assert sidebar.clean_check.isEnabled() is False and sidebar.clean_check.isChecked() is False


def test_colour_negative_leaves_ice_usable_on_the_device_it_is_switched_to() -> None:
    sidebar, _ = _sidebar(settings={"backend": "nkscan", "film_type": "negative"})
    sidebar._on_devices_ready([SE_DEVICE, NKSCAN_DEVICE])
    sidebar.device_combo.setCurrentIndex(1)

    assert sidebar.ir_check.isEnabled() is True
    assert sidebar.clean_check.isEnabled() is True


def test_going_back_to_colour_negative_restores_them() -> None:
    sidebar, _ = _sidebar(NKSCAN_DEVICE)
    sidebar.film_type_combo.setCurrentIndex(sidebar.film_type_combo.findData("mono"))
    sidebar.film_type_combo.setCurrentIndex(sidebar.film_type_combo.findData("negative"))

    assert sidebar.ir_check.isEnabled() is True and sidebar.clean_check.isEnabled() is True


def test_the_film_type_reaches_the_request_and_the_settings() -> None:
    sidebar, controller = _sidebar(NKSCAN_DEVICE)
    sidebar.folder_edit.setText("/tmp/negpy-test")
    sidebar.film_type_combo.setCurrentIndex(sidebar.film_type_combo.findData("positive"))
    sidebar._on_scan()

    assert controller.started[-1][1].params.film_type == "positive"
    assert sidebar.settings.film_type == "positive"


# ── group headers ─────────────────────────────────────────────────────


def test_a_group_header_hides_with_its_whole_group() -> None:
    sidebar, _ = _sidebar(MINIMAL_DEVICE)
    # Nothing to say about the film, and one manual holder to frame.
    assert sidebar.film_header.isVisibleTo(sidebar) is False
    assert sidebar.quality_header.isVisibleTo(sidebar) is True
    assert sidebar.output_header.isVisibleTo(sidebar) is True


def test_the_film_group_appears_where_the_transport_asks_about_the_film() -> None:
    sidebar, _ = _sidebar(NKSCAN_DEVICE)
    assert sidebar.film_header.isVisibleTo(sidebar) is True
    assert sidebar.framing_header.isVisibleTo(sidebar) is True


def test_no_device_hides_the_optional_headers() -> None:
    sidebar, _ = _sidebar()
    sidebar._update_device_caps()
    assert sidebar.film_header.isVisibleTo(sidebar) is False
    assert sidebar.framing_header.isVisibleTo(sidebar) is False


# ── the status strip ──────────────────────────────────────────────────


def test_the_status_row_keeps_its_height_through_every_role() -> None:
    """Three rows that come and go moved the Scan button under the cursor."""
    sidebar, _ = _sidebar(FULL_DEVICE)
    sidebar.show()
    sidebar.layout().activate()
    reserved = sidebar.status_strip.height()
    top_of_button = sidebar.scan_btn.pos().y()

    for step in (
        lambda: sidebar.set_scanning(True),
        lambda: sidebar._on_scan_progress(0.4, "Scanning"),
        lambda: sidebar._on_scan_error("USB I/O failed on a long path that would wrap a label"),
    ):
        step()
        sidebar.layout().activate()
        assert sidebar.status_strip.height() == reserved
        assert sidebar.scan_btn.pos().y() == top_of_button


def test_the_row_prefers_the_pass_then_the_message_then_the_summary() -> None:
    sidebar, _ = _sidebar(FULL_DEVICE)
    assert sidebar.status_strip.showing() == "summary"

    sidebar.status_strip.set_message("Scanned: /tmp/a.tiff")
    assert sidebar.status_strip.showing() == "message"

    sidebar.status_strip.set_progress("Scanning… %p%", 0.5)
    assert sidebar.status_strip.showing() == "progress"  # a running pass outranks the message

    sidebar.status_strip.stop_progress()
    assert sidebar.status_strip.showing() == "message"

    sidebar._update_settings_from_ui()  # the operator edits: the summary is due again
    assert sidebar.status_strip.showing() == "summary"


# ── the commit button ─────────────────────────────────────────────────


def test_the_scan_button_goes_hollow_while_a_scan_runs() -> None:
    """The fill says "this starts a scan"; Stop must not wear it. QSS reads the property."""
    sidebar, _ = _sidebar(FULL_DEVICE)
    assert sidebar.scan_btn.property("scanning") == "false"

    sidebar.set_scanning(True)
    assert sidebar.scan_btn.property("scanning") == "true"
    assert sidebar.scan_btn.text().strip() == "Stop"

    sidebar.set_scanning(False)
    assert sidebar.scan_btn.property("scanning") == "false"
    assert sidebar.scan_btn.text().strip() == "Scan"


def test_the_scan_button_has_a_rule_to_fill_it() -> None:
    """objectName without a matching rule is how it stayed transparent."""
    from negpy.desktop.view.styles.templates import load_stylesheet

    assert "QPushButton#scan_btn" in load_stylesheet()


# ── the summary above the Scan button ─────────────────────────────────


def test_the_summary_counts_the_frames_the_batch_will_scan() -> None:
    sidebar, _ = _sidebar(FULL_DEVICE, settings={"selected_frames": [1, 3, 5], "dpi": 4000})
    text = _summary(sidebar)
    assert text.startswith("3 frames  ·  4000 dpi")
    assert "GB" in text or "MB" in text


def test_the_summary_quotes_a_per_frame_size_for_an_unmeasured_strip() -> None:
    sidebar, _ = _sidebar(NKSCAN_DEVICE)
    text = _summary(sidebar)
    assert text.startswith("Whole strip")
    assert "/frame" in text


def test_the_summary_names_the_extra_passes() -> None:
    sidebar, _ = _sidebar(NKSCAN_DEVICE)
    sidebar.ir_check.setChecked(True)
    sidebar.samples_combo.setCurrentIndex(sidebar.samples_combo.findData(4))

    assert "IR" in _summary(sidebar) and "4× sampled" in _summary(sidebar)

    sidebar.clean_check.setChecked(True)

    assert "ICE" in _summary(sidebar) and "IR" not in _summary(sidebar)


def test_the_count_and_the_size_carry_the_weight_in_the_summary() -> None:
    """The two numbers the operator checks before committing are the two that stand out."""
    sidebar, _ = _sidebar(FULL_DEVICE, settings={"selected_frames": [1, 3, 5], "dpi": 4000})
    markup = sidebar.status_strip._summary.text()

    assert f'<span style="color: {THEME.text_primary}">3 frames</span>' in markup
    assert markup.count("<span") == 2  # the count and the size, nothing else
    assert "4000 dpi" in _summary(sidebar)


def test_a_single_holder_device_summarizes_one_frame() -> None:
    # The saved 3600 is not offered here, so the nearest stop the device has is shown and used.
    sidebar, _ = _sidebar(MINIMAL_DEVICE)
    assert _summary(sidebar).startswith("1 frame  ·  2400 dpi")
    assert sidebar._dpi() == 2400


def test_the_summary_shrinks_with_the_crop() -> None:
    full, _ = _sidebar(MINIMAL_DEVICE)
    cropped, _ = _sidebar(MINIMAL_DEVICE, settings={"scan_window": [0.0, 0.0, 0.5, 0.5]})
    assert _summary(full) != _summary(cropped)


def test_an_ir_pass_costs_a_fourth_plane() -> None:
    caps = MINIMAL_CAPS
    plain = estimated_frame_bytes(caps, 2400, 16)
    with_ir = estimated_frame_bytes(caps, 2400, 16, capture_ir=True)
    assert with_ir == pytest.approx(plain * 4 / 3, rel=1e-3)


def test_a_window_scales_the_estimate_by_its_area() -> None:
    caps = MINIMAL_CAPS
    assert estimated_frame_bytes(caps, 2400, 16, window=(0.0, 0.0, 0.5, 0.5)) == pytest.approx(
        estimated_frame_bytes(caps, 2400, 16) / 4, rel=1e-3
    )


# ── the film that came out ────────────────────────────────────────────


def test_ejecting_drops_the_frame_selection_of_the_film_that_left() -> None:
    sidebar, _ = _sidebar(FULL_DEVICE, settings={"selected_frames": [1, 3], "frame_windows": {"1": [0.1, 0.1, 0.9, 0.9]}})
    assert sidebar.settings.selected_frames == (1, 3)

    sidebar._on_ejected(True)

    assert sidebar.settings.selected_frames == ()
    assert sidebar.settings.frame_windows == {}
    assert "frame selection cleared" in sidebar.status_strip.message()


def test_ejecting_keeps_the_registration_offsets() -> None:
    # Offset and drift belong to the transport's own registration, not to one strip.
    sidebar, _ = _sidebar(FULL_DEVICE, settings={"selected_frames": [1], "frame_offset_mm": 1.5, "frame_offset_modifier_mm": 0.2})

    sidebar._on_ejected(True)

    assert sidebar.settings.frame_offset_mm == 1.5
    assert sidebar.settings.frame_offset_modifier_mm == 0.2


def test_ejecting_with_nothing_picked_says_only_that() -> None:
    sidebar, _ = _sidebar(FULL_DEVICE)
    sidebar._on_ejected(True)
    assert sidebar.status_strip.message() == "Film ejected"


# ── typed DPI ─────────────────────────────────────────────────────────


def test_a_typed_dpi_outside_the_device_range_cannot_be_entered() -> None:
    sidebar, _ = _sidebar(MINIMAL_DEVICE)  # supported_dpi=(1200, 2400)
    editor = sidebar.dpi_combo.lineEdit()
    assert editor is not None
    assert editor.validator() is not None
    assert editor.validator().validate("99999", 5)[0] != QValidator.State.Acceptable


def test_an_empty_dpi_box_falls_back_to_the_finest_the_device_offers() -> None:
    sidebar, _ = _sidebar(MINIMAL_DEVICE)
    sidebar.dpi_combo.clear()  # nothing selected and nothing typed
    assert sidebar._dpi() == 2400


def test_a_pass_the_device_cannot_run_is_not_shown_at_all() -> None:
    # Disabled-with-a-reason is for a pass the film blocks; one the transport lacks goes away.
    sidebar, _ = _sidebar(MINIMAL_DEVICE)
    assert sidebar.ir_check.isVisibleTo(sidebar) is False
    assert sidebar.me_check.isVisibleTo(sidebar) is False

    sidebar, _ = _sidebar(SE_DEVICE, settings={"backend": "plustek"})
    assert sidebar.ir_check.isVisibleTo(sidebar) is True
    assert sidebar.me_check.isVisibleTo(sidebar) is True


def test_a_film_that_blocks_infrared_leaves_the_control_visible_to_explain_itself() -> None:
    sidebar, _ = _sidebar(NKSCAN_DEVICE)
    sidebar.film_type_combo.setCurrentIndex(sidebar.film_type_combo.findData("mono"))

    assert sidebar.ir_check.isVisibleTo(sidebar) is True
    assert sidebar.ir_check.isEnabled() is False
