# SPDX-License-Identifier: GPL-3.0-or-later
"""In-tree PlustekBackend contract tests (no hardware required)."""

from __future__ import annotations

import ast
import sys
import threading
import types
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from negpy.infrastructure.scanners.base import ScannerUnavailable, TransientScanError
from negpy.infrastructure.scanners.params import ScanParams
from pyopticfilm.exceptions import ScanCancelled, UsbError
from pyopticfilm.image import ScanImage
from pyopticfilm.usb.device import (
    PID_OPTICFILM_8200I_SE,
    VID_PLUSTEK,
    UsbDeviceInfo,
)
from negpy.infrastructure.scanners.plustek_backend import PlustekBackend
from negpy.infrastructure.scanners.result import ScanResult

_NEGPY_ROOT = Path(__file__).resolve().parents[2] / "negpy"
_ADAPTER = _NEGPY_ROOT / "infrastructure" / "scanners" / "plustek_backend.py"
_DEVICE_ID = "plustek:usb:07b3:1825:002:006"
_BACKEND = "negpy.infrastructure.scanners.plustek_backend"


def _is_driver_module(mod: str) -> bool:
    if mod == "negpy.infrastructure.scanners.plustek_backend":
        return False
    return mod.startswith("pyopticfilm")


def _collect_driver_imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    hits: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                mod = alias.name
                if _is_driver_module(mod):
                    hits.append(f"{path.relative_to(_NEGPY_ROOT.parent)}:{node.lineno}:{mod}")
        elif isinstance(node, ast.ImportFrom) and node.module:
            mod = node.module
            if _is_driver_module(mod):
                hits.append(f"{path.relative_to(_NEGPY_ROOT.parent)}:{node.lineno}:{mod}")
    return hits


def test_only_adapter_imports_plustek_driver() -> None:
    """After extraction, only plustek_backend.py may import pyopticfilm."""
    offenders: list[str] = []
    for path in _NEGPY_ROOT.rglob("*.py"):
        rel = path.relative_to(_NEGPY_ROOT)
        if rel == Path("infrastructure/scanners/plustek_backend.py"):
            continue
        offenders.extend(_collect_driver_imports(path))
    assert offenders == [], "driver imports outside adapter:\n" + "\n".join(offenders)


def _params(**kwargs) -> ScanParams:
    base = dict(dpi=1800, depth=16, capture_ir=False, autofocus=False)
    base.update(kwargs)
    return ScanParams(**base)


def _info() -> UsbDeviceInfo:
    return UsbDeviceInfo(
        vendor_id=VID_PLUSTEK,
        product_id=PID_OPTICFILM_8200I_SE,
        bus=2,
        address=6,
    )


def _stub_pyusb(monkeypatch) -> None:
    """Satisfy PlustekBackend.__init__ when usb.core is unavailable (defense-in-depth)."""
    usb = types.ModuleType("usb")
    usb.core = types.ModuleType("usb.core")
    monkeypatch.setitem(sys.modules, "usb", usb)
    monkeypatch.setitem(sys.modules, "usb.core", usb.core)


def _patch_enum(monkeypatch, devices: list[UsbDeviceInfo] | None = None) -> None:
    _stub_pyusb(monkeypatch)
    devices = devices if devices is not None else [_info()]
    monkeypatch.setattr(f"{_BACKEND}.find_devices", lambda supported_only=True: list(devices))
    monkeypatch.setattr(f"{_BACKEND}.list_devices", lambda: list(devices))


def _fake_scanner(*, progress_steps: int = 0, scan_error: Exception | None = None):
    from pyopticfilm.device.model_8200i_se import MODEL_8200I_SE

    rgb = np.zeros((8, 8, 3), dtype=np.uint16)
    image = ScanImage(rgb=rgb, dpi=1800, device_model="PLUSTEK OpticFilm 8200i SE")

    scanner = MagicMock()
    scanner.model = MODEL_8200I_SE
    scanner.asic._initialized = True
    scanner.asic.is_at_home.return_value = True

    def scan(**kwargs):
        cancel = kwargs.get("cancel")
        if cancel is not None and cancel.is_set():
            raise ScanCancelled("cancelled")
        progress = kwargs.get("progress")
        if progress is not None:
            for i in range(1, progress_steps + 1):
                progress(i / progress_steps)
        if scan_error is not None:
            raise scan_error
        mode = kwargs.get("mode", "color")
        if mode == "infrared":
            return ScanImage(
                rgb=rgb,
                dpi=kwargs.get("resolution", 1800),
                device_model=image.device_model,
                ir=rgb[:, :, 1].copy(),
            )
        return ScanImage(
            rgb=rgb,
            dpi=kwargs.get("resolution", 1800),
            device_model=image.device_model,
        )

    scanner.scan.side_effect = scan
    scanner.calibrate = MagicMock(return_value=MagicMock(asic_shading=True, has_asic_blob=True))
    scanner.calibrator = MagicMock()
    scanner.calibrator.find_for_scan.return_value = None
    scanner.calibrator.ensure_colour_asic_shading = MagicMock(return_value=MagicMock(asic_shading=True, has_asic_blob=True))
    scanner._bringup_motor_armed = True
    scanner.disarm_bringup_motor = MagicMock()
    scanner.arm_bringup_motor = MagicMock()
    scanner.close = MagicMock()
    return scanner


class _FakeOpen:
    def __init__(self, scanner):
        self._scanner = scanner

    def __call__(self, *args, **kwargs):
        return self

    def __enter__(self):
        return self._scanner

    def __exit__(self, *exc):
        self._scanner.close()
        return False


def test_backend_list_devices_empty(monkeypatch):
    _patch_enum(monkeypatch, [])
    assert PlustekBackend().list_devices() == []


def test_backend_list_devices_maps_caps(monkeypatch):
    _patch_enum(monkeypatch)
    devices = PlustekBackend().list_devices()
    assert len(devices) == 1
    dev = devices[0]
    assert dev.id == _info().device_id
    assert "Transparency" in [str(s) for s in dev.capabilities.sources]
    assert 3600 in dev.capabilities.supported_dpi
    assert dev.capabilities.can_eject is False
    assert dev.capabilities.exposure_time_us is None
    assert dev.capabilities.ir_channel is True
    assert dev.capabilities.auto_exposure is False
    assert dev.capabilities.autofocus is False
    assert dev.capabilities.prescan is True
    assert dev.capabilities.prescan_dpi == 1200
    assert dev.capabilities.prescan_mirror_x is True
    assert dev.capabilities.prescan_default_crop is not None


def test_refresh_devices_re_enumerates(monkeypatch):
    _patch_enum(monkeypatch)
    backend = PlustekBackend()
    assert backend.refresh_devices() == backend.list_devices()


def test_eject_returns_false(monkeypatch):
    _stub_pyusb(monkeypatch)
    assert PlustekBackend().eject(_DEVICE_ID) is False


def test_unavailable_without_pyusb(monkeypatch):
    import builtins

    monkeypatch.delitem(sys.modules, "usb", raising=False)
    monkeypatch.delitem(sys.modules, "usb.core", raising=False)

    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "usb.core" or name == "usb" or (name == "usb" and fromlist):
            raise ImportError("simulated missing pyusb")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ScannerUnavailable, match="pyopticfilm"):
        PlustekBackend()


def test_scan_returns_well_formed_result(monkeypatch):
    _patch_enum(monkeypatch)
    scanner = _fake_scanner(progress_steps=2)
    monkeypatch.setattr(f"{_BACKEND}.Scanner.open", _FakeOpen(scanner))
    result = PlustekBackend().scan(_DEVICE_ID, _params(), lambda *_: None, threading.Event())
    assert isinstance(result, ScanResult)
    assert result.rgb.ndim == 3 and result.rgb.shape[2] == 3
    assert result.dpi == 1800
    assert result.device_model


def test_scan_honours_pre_set_cancel(monkeypatch):
    _patch_enum(monkeypatch)
    scanner = _fake_scanner()
    monkeypatch.setattr(f"{_BACKEND}.Scanner.open", _FakeOpen(scanner))
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(Exception, match="[Cc]ancel"):
        PlustekBackend().scan(_DEVICE_ID, _params(), lambda *_: None, cancel)


def test_progress_stays_within_unit_range(monkeypatch):
    _patch_enum(monkeypatch)
    scanner = _fake_scanner(progress_steps=4)
    monkeypatch.setattr(f"{_BACKEND}.Scanner.open", _FakeOpen(scanner))
    seen: list[float] = []
    PlustekBackend().scan(
        _DEVICE_ID,
        _params(),
        lambda fraction, phase="Scanning": seen.append(fraction),
        threading.Event(),
    )
    assert seen
    assert all(0.0 <= v <= 1.0 for v in seen)


def test_transport_glitches_are_typed_transient(monkeypatch):
    _patch_enum(monkeypatch)
    scanner = _fake_scanner(scan_error=UsbError("Error during device I/O"))
    monkeypatch.setattr(f"{_BACKEND}.Scanner.open", _FakeOpen(scanner))
    with pytest.raises(TransientScanError):
        PlustekBackend().scan(_DEVICE_ID, _params(), lambda *_: None, threading.Event())


def test_real_errors_are_not_transient(monkeypatch):
    _patch_enum(monkeypatch)
    scanner = _fake_scanner()
    monkeypatch.setattr(f"{_BACKEND}.Scanner.open", _FakeOpen(scanner))
    with pytest.raises(Exception) as excinfo:
        PlustekBackend().scan(
            _DEVICE_ID,
            _params(frame=3),
            lambda *_: None,
            threading.Event(),
        )
    assert not isinstance(excinfo.value, TransientScanError)
    assert "frame" in str(excinfo.value).lower()


def test_capture_ir_returns_ir_plane(monkeypatch):
    _patch_enum(monkeypatch)
    scanner = _fake_scanner()
    monkeypatch.setattr(f"{_BACKEND}.Scanner.open", _FakeOpen(scanner))
    result = PlustekBackend().scan(
        _DEVICE_ID,
        _params(capture_ir=True),
        lambda *_: None,
        threading.Event(),
    )
    assert result.ir is not None
    assert result.ir.ndim == 2
    assert scanner.scan.call_count == 2


def test_scan_ensures_colour_calib_before_scan(monkeypatch):
    _patch_enum(monkeypatch)
    scanner = _fake_scanner()
    order: list[str] = []
    scanner.calibrator.ensure_colour_asic_shading.side_effect = lambda *_a, **_k: (
        order.append("ensure") or MagicMock(asic_shading=True, has_asic_blob=True)
    )
    inner_scan = scanner.scan.side_effect

    def scan_tracked(**kwargs):
        order.append("scan")
        return inner_scan(**kwargs)

    scanner.scan.side_effect = scan_tracked
    monkeypatch.setattr(f"{_BACKEND}.Scanner.open", _FakeOpen(scanner))
    PlustekBackend().scan(_DEVICE_ID, _params(), lambda *_: None, threading.Event())
    assert order[0] == "ensure"
    assert "scan" in order


def test_scan_skips_calib_when_asic_already_ready(monkeypatch):
    _patch_enum(monkeypatch)
    scanner = _fake_scanner()
    entry = MagicMock(asic_shading=True, has_asic_blob=True)
    scanner.calibrator.find_for_scan.return_value = entry
    scanner.asic.asic_shading_ready = True
    monkeypatch.setattr(f"{_BACKEND}.Scanner.open", _FakeOpen(scanner))
    PlustekBackend().scan(_DEVICE_ID, _params(), lambda *_: None, threading.Event())
    scanner.calibrator.ensure_colour_asic_shading.assert_not_called()


def test_default_se_scan_passes_preview_safe_geometry(monkeypatch):
    """window=None must not use bare area=None (feed2=13704 motor-gate failure)."""
    _patch_enum(monkeypatch)
    scanner = _fake_scanner()
    monkeypatch.setattr(f"{_BACKEND}.Scanner.open", _FakeOpen(scanner))
    PlustekBackend().scan(_DEVICE_ID, _params(dpi=1200), lambda *_: None, threading.Event())
    assert scanner.scan.call_count >= 1
    kwargs = scanner.scan.call_args.kwargs
    assert kwargs.get("geometry") is not None
    assert kwargs.get("area") is None
    assert kwargs["geometry"].lincnt_register == 4836
    feed2 = scanner.model.feed_to_scan_steps_for_area(kwargs["geometry"].area)
    assert feed2 == 13128


def test_explicit_window_uses_crop_geometry(monkeypatch):
    """SE crop must pass forced geometry (LINCNT-clamped), not bare area=window."""
    _patch_enum(monkeypatch)
    scanner = _fake_scanner()
    monkeypatch.setattr(f"{_BACKEND}.Scanner.open", _FakeOpen(scanner))
    window = (0.1, 0.1, 0.9, 0.5)
    PlustekBackend().scan(
        _DEVICE_ID,
        _params(dpi=1200, window=window),
        lambda *_: None,
        threading.Event(),
    )
    kwargs = scanner.scan.call_args.kwargs
    assert kwargs.get("geometry") is not None
    assert kwargs.get("area") is None
    geo = kwargs["geometry"]
    assert geo.pixels % 2 == 0
    feed2 = scanner.model.feed_to_scan_steps_for_area(geo.area)
    max_lc = scanner.model.max_lincnt_for(feed2, 1200)
    assert geo.lincnt_register <= max_lc


def test_open_session_shape(monkeypatch):
    _patch_enum(monkeypatch)
    scanner = _fake_scanner()
    monkeypatch.setattr(f"{_BACKEND}.Scanner.open", lambda *a, **k: scanner)
    backend = PlustekBackend()
    session = backend.open_session(_DEVICE_ID)
    try:
        assert session.device_id == _DEVICE_ID
        for method in ("scan", "eject", "close", "__enter__", "__exit__"):
            assert callable(getattr(session, method, None))
    finally:
        session.close()


def test_session_scans_on_held_handle(monkeypatch):
    _patch_enum(monkeypatch)
    scanner = _fake_scanner()
    monkeypatch.setattr(f"{_BACKEND}.Scanner.open", lambda *a, **k: scanner)
    backend = PlustekBackend()
    with backend.open_session(_DEVICE_ID) as session:
        result = session.scan(_params(), lambda *_: None, threading.Event())
    assert isinstance(result, ScanResult)
    scanner.close.assert_called()


def test_session_close_is_idempotent(monkeypatch):
    _patch_enum(monkeypatch)
    scanner = _fake_scanner()
    monkeypatch.setattr(f"{_BACKEND}.Scanner.open", lambda *a, **k: scanner)
    backend = PlustekBackend()
    session = backend.open_session(_DEVICE_ID)
    session.close()
    session.close()


def test_backend_scan_refuses_while_session_held(monkeypatch):
    _patch_enum(monkeypatch)
    scanner = _fake_scanner()
    monkeypatch.setattr(f"{_BACKEND}.Scanner.open", lambda *a, **k: scanner)
    backend = PlustekBackend()
    session = backend.open_session(_DEVICE_ID)
    try:
        with pytest.raises(RuntimeError, match="held"):
            backend.scan(_DEVICE_ID, _params(), lambda *_: None, threading.Event())
    finally:
        session.close()
