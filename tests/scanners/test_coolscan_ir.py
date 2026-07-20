"""Tests for coolscan3-style inline IR: option strategy, net IDs, channel repair.

The tested Nikon Coolscan LS-5000 exposes IR as a boolean `infrared`
option; the frame then carries 4 samples/pixel while reporting
SANE_FRAME_RGB (pieusb convention). python-sane's C reader hardcodes
3 samples/pixel for non-gray frames, so the array arrives byte-intact but
misshaped — `_reinterpret_channels` recovers it from the frame geometry.
"""

import threading
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest

from negpy.infrastructure.scanners.params import ScanParams
from negpy.infrastructure.scanners.sane_backend import (
    SaneBackend,
    _caps_from_options,
    _detect_ir,
    _find_coolscan3_ir_option,
    _find_ir_option,
    _reinterpret_channels,
    _strip_net_prefix,
)


@dataclass
class FakeOption:
    """Stand-in for python-sane's Option (only the fields the module reads)."""

    constraint: Any = None
    desc: str = ""
    active: bool = True
    settable: bool = True

    def is_active(self) -> bool:
        return self.active

    def is_settable(self) -> bool:
        return self.settable


class SettableOnlyOption:
    """Option shim for bindings that expose writability but not activity."""

    constraint = None
    desc = ""

    def is_settable(self) -> bool:
        return False


COOLSCAN3_OPT = {
    "infrared": FakeOption(),
    "depth": FakeOption(constraint=[8, 16]),
    "resolution": FakeOption(constraint=[4000, 2000, 1000]),
    "frame": FakeOption(constraint=(1, 40, 1)),
    # NB: no "mode", no "source" — like the real backend.
}


class TestNetPrefix:
    def test_strips_saned_prefix(self) -> None:
        assert _strip_net_prefix("net:scanner.example:coolscan3:usb:libusb:001:007") == "coolscan3:usb:libusb:001:007"

    def test_plain_id_unchanged(self) -> None:
        assert _strip_net_prefix("coolscan3:usb:libusb:001:007") == "coolscan3:usb:libusb:001:007"

    def test_pieusb_over_net(self) -> None:
        assert _strip_net_prefix("net:host:pieusb:libusb:001:004") == "pieusb:libusb:001:004"

    def test_bracketed_ipv6_host(self) -> None:
        assert _strip_net_prefix("net:[2001:db8::1]:coolscan3:usb:test") == "coolscan3:usb:test"


class TestIrOptionDetection:
    def test_finds_coolscan3_infrared(self) -> None:
        assert _find_coolscan3_ir_option(COOLSCAN3_OPT) == "infrared"

    def test_legacy_finder_does_not_claim_coolscan3_infrared(self) -> None:
        assert _find_ir_option(COOLSCAN3_OPT) is None

    def test_detect_ir_true(self) -> None:
        assert _detect_ir(COOLSCAN3_OPT, "coolscan3:usb:test") is True

    def test_inactive_ir_option_is_not_advertised(self) -> None:
        assert _detect_ir({"infrared": FakeOption(active=False)}, "coolscan3:usb:test") is False

    def test_read_only_ir_option_is_not_advertised(self) -> None:
        assert _detect_ir({"infrared": FakeOption(settable=False)}, "coolscan3:usb:test") is False

    def test_coolscan2_infrared_option_is_not_advertised(self) -> None:
        assert _detect_ir({"infrared": FakeOption()}, "coolscan2:usb:test") is False

    def test_legacy_ir_option_keeps_presence_based_capability(self) -> None:
        assert _detect_ir({"ir": FakeOption(active=False)}, "legacy:usb:test") is True

    def test_no_ir_on_flatbed(self) -> None:
        assert _find_ir_option({"mode": FakeOption(constraint=["Color", "Gray"])}) is None


class TestCoolscanCapabilities:
    def test_film_scanner_inferred_without_source(self) -> None:
        caps = _caps_from_options(COOLSCAN3_OPT, "coolscan3:usb:libusb:001:007")
        assert caps.ir_channel is True
        assert caps.sources  # inferred, not skipped
        assert caps.supported_depths == (8, 16)

    def test_film_scanner_inferred_over_net(self) -> None:
        caps = _caps_from_options(COOLSCAN3_OPT, "net:scanner.example:coolscan3:usb:libusb:001:007")
        assert caps.sources

    def test_unknown_backend_ir_option_does_not_imply_film_scanner(self) -> None:
        caps = _caps_from_options({"infrared": FakeOption()}, "mystery:001")
        assert caps.sources == ()
        assert caps.ir_channel is False

    def test_inactive_ir_still_identifies_film_scanner_without_advertising_ir(self) -> None:
        caps = _caps_from_options({"infrared": FakeOption(active=False)}, "coolscan3:usb:test")
        assert caps.sources
        assert caps.ir_channel is False

    def test_coolscan2_does_not_advertise_unsupported_inline_ir(self) -> None:
        caps = _caps_from_options({"infrared": FakeOption()}, "coolscan2:usb:test")
        assert caps.ir_channel is False


def _emulate_python_sane_read(true_frame: np.ndarray) -> np.ndarray:
    """Reproduce python-sane's C reader on a 4-sample frame.

    It assumes 3 samples/pixel, reads `3 * width`-sample chunks, and
    DISCARDS a partial final chunk at EOF (_sane.c snap loop) — so when
    `4 * lines` is not a multiple of 3, trailing samples are lost.
    """
    h, w, c = true_frame.shape
    flat = true_frame.reshape(-1)
    chunk = 3 * w
    n_full = flat.size // chunk
    return flat[: n_full * chunk].reshape(n_full, w, 3)


class TestReinterpretChannels:
    def _rgbi(self, h: int, w: int) -> np.ndarray:
        rng = np.random.default_rng(42)
        return rng.integers(0, 65535, size=(h, w, 4), dtype=np.uint16)

    def test_recovers_misread_rgbi_divisible(self) -> None:
        h, w = 6, 5  # 4h % 3 == 0: no truncation, full recovery
        true = self._rgbi(h, w)
        fixed = _reinterpret_channels(_emulate_python_sane_read(true), w, h)
        assert fixed.shape == (h, w, 4)
        assert np.array_equal(fixed, true)

    def test_recovers_truncated_rgbi_mod1(self) -> None:
        h, w = 7, 5  # 4h % 3 == 1 — the real LS-5000 case (1489, 5959 lines)
        true = self._rgbi(h, w)
        encoded = _emulate_python_sane_read(true)
        fixed = _reinterpret_channels(encoded, w, h)
        assert fixed.shape == (h - 1, w, 4)  # padded edge row dropped
        assert np.array_equal(fixed, true[: h - 1])
        assert np.shares_memory(fixed, encoded)

    def test_recovers_truncated_rgbi_mod2(self) -> None:
        h, w = 5, 4  # 4h % 3 == 2
        true = self._rgbi(h, w)
        fixed = _reinterpret_channels(_emulate_python_sane_read(true), w, h)
        assert fixed.shape == (h - 1, w, 4)
        assert np.array_equal(fixed, true[: h - 1])

    def test_correct_rgb_untouched(self) -> None:
        arr = np.zeros((10, 20, 3), dtype=np.uint16)
        assert _reinterpret_channels(arr, 20, 10) is arr

    def test_unknown_geometry_untouched(self) -> None:
        arr = np.zeros((8, 5, 3), dtype=np.uint16)
        assert _reinterpret_channels(arr, -1, -1) is arr

    def test_indivisible_untouched(self) -> None:
        arr = np.zeros((7, 5, 3), dtype=np.uint16)
        assert _reinterpret_channels(arr, 5, 6) is arr


class FakeSaneDev:
    """Mimics python-sane SaneDev for a coolscan3-like device.

    Setting an attribute that is not an internal field and not a known SANE
    option raises AttributeError, like python-sane does. arr_snap() returns
    the misshaped array the real C module produces for 4-sample frames.
    """

    _INTERNAL = (
        "recorded",
        "true_frame",
        "cancelled",
        "closed",
        "opt_map",
        "started",
        "parameter_format",
        "parameter_last",
        "parameter_depth",
        "parameter_bpl",
    )

    def __init__(
        self,
        true_frame: np.ndarray,
        opt_map: dict | None = None,
        *,
        parameter_format: str = "color",
        parameter_last: bool = True,
        parameter_depth: int = 16,
        parameter_bpl: int | None = None,
    ) -> None:
        object.__setattr__(self, "opt_map", COOLSCAN3_OPT if opt_map is None else opt_map)
        object.__setattr__(self, "recorded", {})
        object.__setattr__(self, "true_frame", true_frame)
        object.__setattr__(self, "cancelled", False)
        object.__setattr__(self, "closed", False)
        object.__setattr__(self, "started", False)
        object.__setattr__(self, "parameter_format", parameter_format)
        object.__setattr__(self, "parameter_last", parameter_last)
        object.__setattr__(self, "parameter_depth", parameter_depth)
        object.__setattr__(self, "parameter_bpl", parameter_bpl)

    @property
    def opt(self):
        return self.opt_map

    def __setattr__(self, name: str, value: Any) -> None:
        if name in self._INTERNAL:
            object.__setattr__(self, name, value)
            return
        if name not in self.opt_map:
            raise AttributeError(f"No such SANE option: {name}")
        self.recorded[name] = value

    def start(self) -> None:
        object.__setattr__(self, "started", True)

    def get_parameters(self):
        h, w, _ = self.true_frame.shape
        bytes_per_line = self.parameter_bpl
        if bytes_per_line is None:
            bytes_per_line = w * 4 * ((self.parameter_depth + 7) // 8)
        return (self.parameter_format, self.parameter_last, (w, h), self.parameter_depth, bytes_per_line)

    def arr_snap(self) -> np.ndarray:
        if self.true_frame.shape[2] == 3:
            return self.true_frame  # 3-channel frames come back correctly
        return _emulate_python_sane_read(self.true_frame)

    def cancel(self) -> None:
        object.__setattr__(self, "cancelled", True)

    def close(self) -> None:
        object.__setattr__(self, "closed", True)


@dataclass
class FakeSaneModule:
    dev: FakeSaneDev
    opened: list = field(default_factory=list)

    def open(self, device_id: str) -> FakeSaneDev:
        self.opened.append(device_id)
        return self.dev


def _make_backend(dev: FakeSaneDev) -> SaneBackend:
    backend = SaneBackend.__new__(SaneBackend)
    backend._sane = FakeSaneModule(dev)
    backend._sane_initialized = True
    backend._devices_cache = None
    return backend


class TestScanWithOptionStrategy:
    def _run(self, device_id: str, h: int = 6) -> tuple:
        rng = np.random.default_rng(7)
        true = rng.integers(0, 65535, size=(h, 5, 4), dtype=np.uint16)
        dev = FakeSaneDev(true)
        backend = _make_backend(dev)
        params = ScanParams(dpi=1000, depth=16, capture_ir=True)
        result = backend.scan(device_id, params, None, threading.Event())
        return true, dev, result

    def test_ir_split_and_shape_repair(self) -> None:
        true, dev, result = self._run("coolscan3:usb:libusb:001:007")
        assert result.rgb.shape == (6, 5, 3)
        assert result.ir is not None and result.ir.shape == (6, 5)
        assert np.array_equal(result.rgb, true[:, :, :3])
        assert np.array_equal(result.ir, true[:, :, 3])

    def test_ir_split_with_python_sane_truncation(self) -> None:
        # 4h % 3 == 1: python-sane drops the stream tail (real LS-5000 case);
        # scan() must still deliver IR, minus the one padded edge row.
        true, dev, result = self._run("coolscan3:usb:libusb:001:007", h=7)
        assert result.rgb.shape == (6, 5, 3)
        assert result.ir is not None and result.ir.shape == (6, 5)
        assert np.array_equal(result.rgb, true[:6, :, :3])
        assert np.array_equal(result.ir, true[:6, :, 3])

    def test_infrared_option_enabled_and_mode_untouched(self) -> None:
        _, dev, _ = self._run("coolscan3:usb:libusb:001:007")
        assert dev.recorded.get("infrared") is True
        assert "mode" not in dev.recorded  # device has no mode option; must not be set

    def test_works_over_net_device_id(self) -> None:
        _, dev, result = self._run("net:scanner.example:coolscan3:usb:libusb:001:007")
        assert result.ir is not None
        assert dev.recorded.get("infrared") is True

    def test_no_ir_requested_scans_plain(self) -> None:
        rng = np.random.default_rng(9)
        true = rng.integers(0, 65535, size=(6, 5, 3), dtype=np.uint16)
        dev = FakeSaneDev(true)
        backend = _make_backend(dev)
        params = ScanParams(dpi=1000, depth=16, capture_ir=False)
        result = backend.scan("coolscan3:usb:libusb:001:007", params, None, threading.Event())
        assert "infrared" not in dev.recorded
        assert result.ir is None
        assert result.rgb.shape == (6, 5, 3)
        assert np.array_equal(result.rgb, true)

    def test_inactive_infrared_option_fails_before_scan_start(self) -> None:
        rng = np.random.default_rng(10)
        opt = dict(COOLSCAN3_OPT)
        opt["infrared"] = FakeOption(active=False)
        dev = FakeSaneDev(rng.integers(0, 65535, size=(6, 5, 4), dtype=np.uint16), opt_map=opt)

        with pytest.raises(RuntimeError, match="infrared.*inactive"):
            self._scan_device(dev, ScanParams(dpi=1000, depth=16, capture_ir=True))

        assert dev.started is False
        assert "infrared" not in dev.recorded

    def test_read_only_infrared_option_fails_before_scanner_configuration(self) -> None:
        rng = np.random.default_rng(100)
        opt = dict(COOLSCAN3_OPT)
        opt["infrared"] = FakeOption(settable=False)
        dev = FakeSaneDev(rng.integers(0, 65535, size=(6, 5, 4), dtype=np.uint16), opt_map=opt)

        with pytest.raises(RuntimeError, match="infrared.*not settable"):
            self._scan_device(dev, ScanParams(dpi=1000, depth=16, capture_ir=True))

        assert dev.started is False
        assert dev.recorded == {}

    def test_settable_only_read_only_option_still_fails_preflight(self) -> None:
        rng = np.random.default_rng(1001)
        opt = dict(COOLSCAN3_OPT)
        opt["infrared"] = SettableOnlyOption()
        dev = FakeSaneDev(rng.integers(0, 65535, size=(6, 5, 4), dtype=np.uint16), opt_map=opt)

        with pytest.raises(RuntimeError, match="infrared.*not settable"):
            self._scan_device(dev, ScanParams(dpi=1000, depth=16, capture_ir=True))

        assert dev.started is False
        assert dev.recorded == {}

    @pytest.mark.parametrize(
        ("device_kwargs", "message"),
        (
            ({"parameter_last": False}, "last frame"),
            ({"parameter_format": "gray"}, "frame format"),
            ({"parameter_depth": 8}, "depth"),
            ({"parameter_bpl": 5 * 4 * 2 + 2}, "bytes per line"),
        ),
    )
    def test_invalid_inline_rgbi_parameters_fail_loud(self, device_kwargs: dict, message: str) -> None:
        rng = np.random.default_rng(1002)
        dev = FakeSaneDev(
            rng.integers(0, 65535, size=(6, 5, 4), dtype=np.uint16),
            **device_kwargs,
        )

        with pytest.raises(RuntimeError, match=message):
            self._scan_device(dev, ScanParams(dpi=1000, depth=16, capture_ir=True))

        assert dev.cancelled

    def test_requested_ir_without_a_usable_mechanism_fails_before_start(self) -> None:
        rng = np.random.default_rng(101)
        opt = {name: value for name, value in COOLSCAN3_OPT.items() if name != "infrared"}
        dev = FakeSaneDev(rng.integers(0, 65535, size=(6, 5, 3), dtype=np.uint16), opt_map=opt)

        with pytest.raises(RuntimeError, match="no usable infrared mechanism"):
            self._scan_device(dev, ScanParams(dpi=1000, depth=16, capture_ir=True))

        assert dev.started is False

    @staticmethod
    def _scan_device(dev: FakeSaneDev, params: ScanParams):
        backend = _make_backend(dev)
        return backend.scan("coolscan3:usb:libusb:001:007", params, None, threading.Event())

    def test_option_strategy_not_claimed_for_coolscan2(self) -> None:
        # coolscan2 exposes the same option name but delivers IR as a separate
        # later frame — the inline-option contract must not be applied to it.
        rng = np.random.default_rng(11)
        dev = FakeSaneDev(rng.integers(0, 65535, size=(6, 5, 4), dtype=np.uint16))
        assert SaneBackend._ir_strategy(dev, "coolscan3:usb:libusb:001:007") == "option"
        assert SaneBackend._ir_strategy(dev, "coolscan2:usb:libusb:001:007") is None
        assert SaneBackend._ir_strategy(dev, "net:scanner.example:coolscan2:usb:x") is None

    def test_net_pieusb_keeps_preexisting_rgbi_strategy(self) -> None:
        rng = np.random.default_rng(12)
        opt = {"mode": FakeOption(constraint=["Color", "RGBI"])}
        dev = FakeSaneDev(rng.integers(0, 65535, size=(6, 5, 4), dtype=np.uint16), opt_map=opt)
        assert SaneBackend._ir_strategy(dev, "pieusb:libusb:001:004") == "internal"
        assert SaneBackend._ir_strategy(dev, "net:scanner.example:pieusb:libusb:001:004") == "rgbi"

    def test_generic_rgbi_mode_keeps_preexisting_rgb_fallback(self) -> None:
        rng = np.random.default_rng(13)
        true = rng.integers(0, 65535, size=(6, 5, 3), dtype=np.uint16)
        opt = {
            "mode": FakeOption(constraint=["Color", "RGBI"]),
            "depth": FakeOption(constraint=[8, 16]),
            "resolution": FakeOption(constraint=[1000]),
        }
        dev = FakeSaneDev(true, opt_map=opt)
        backend = _make_backend(dev)

        result = backend.scan(
            "generic:usb:test",
            ScanParams(dpi=1000, depth=16, capture_ir=True),
            None,
            threading.Event(),
        )

        assert np.array_equal(result.rgb, true)
        assert result.ir is None

    @pytest.mark.parametrize(
        ("device_id", "opt"),
        (
            ("legacy:usb:test", {"ir": FakeOption()}),
            ("coolscan2:usb:test", {"infrared": FakeOption()}),
        ),
    )
    def test_non_coolscan3_option_keeps_preexisting_rgb_fallback(self, device_id: str, opt: dict) -> None:
        rng = np.random.default_rng(14)
        true = rng.integers(0, 65535, size=(6, 5, 3), dtype=np.uint16)
        opt.update(
            {
                "depth": FakeOption(constraint=[8, 16]),
                "resolution": FakeOption(constraint=[1000]),
            }
        )
        dev = FakeSaneDev(true, opt_map=opt)
        backend = _make_backend(dev)

        result = backend.scan(
            device_id,
            ScanParams(dpi=1000, depth=16, capture_ir=True),
            None,
            threading.Event(),
        )

        assert np.array_equal(result.rgb, true)
        assert result.ir is None


def test_requested_ir_without_channel_raises() -> None:
    rng = np.random.default_rng(17)
    # Device frame is plain 3-channel: inline-IR strategy cannot deliver
    # a 4th channel and the scan must fail loud, not return ir=None.
    dev = FakeSaneDev(rng.integers(0, 65535, size=(6, 5, 3), dtype=np.uint16))
    backend = _make_backend(dev)
    with pytest.raises(RuntimeError, match="no 4th channel"):
        backend.scan(
            "coolscan3:usb:libusb:001:007",
            ScanParams(dpi=1000, depth=16, capture_ir=True),
            None,
            threading.Event(),
        )
    assert dev.cancelled
