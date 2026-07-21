"""Tests for coolscan3-style inline IR: option strategy, net IDs, channel repair.

coolscan3 (Nikon Coolscan LS-50/5000) exposes IR as a boolean `infrared`
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

from negpy.infrastructure.scanners.params import ScanMode, ScanParams
from negpy.infrastructure.scanners.sane_backend import (
    SaneBackend,
    _caps_from_options,
    _detect_ir,
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


COOLSCAN3_OPT = {
    "infrared": FakeOption(),
    "depth": FakeOption(constraint=[8, 16]),
    "resolution": FakeOption(constraint=[4000, 2000, 1000]),
    "frame": FakeOption(constraint=(1, 40, 1)),
    "frame_count": FakeOption(constraint=(1, 40, 1)),
    "subframe": FakeOption(constraint=(0.0, 37.8333, 0.0)),
    "br_y": FakeOption(constraint=(0, 5958, 1)),
    "autofocus": FakeOption(),
    "ae": FakeOption(),
    "samples_per_scan": FakeOption(constraint=(1, 16, 1)),
    # NB: no "mode", no "source" — like the real backend.
}


class TestNetPrefix:
    def test_strips_saned_prefix(self) -> None:
        assert _strip_net_prefix("net:10.0.0.100:coolscan3:usb:libusb:001:007") == "coolscan3:usb:libusb:001:007"

    def test_plain_id_unchanged(self) -> None:
        assert _strip_net_prefix("coolscan3:usb:libusb:001:007") == "coolscan3:usb:libusb:001:007"

    def test_pieusb_over_net(self) -> None:
        assert _strip_net_prefix("net:host:pieusb:libusb:001:004") == "pieusb:libusb:001:004"


class TestIrOptionDetection:
    def test_exact_infrared_is_not_a_legacy_dedicated_ir_option(self) -> None:
        assert _find_ir_option(COOLSCAN3_OPT) is None

    @pytest.mark.parametrize(
        "device_id",
        (
            "coolscan3:usb:libusb:001:007",
            "net:scanner:coolscan3:usb:libusb:001:007",
        ),
    )
    def test_detects_exact_infrared_only_for_coolscan3(self, device_id: str) -> None:
        assert _detect_ir(COOLSCAN3_OPT, device_id) is True

    def test_no_ir_on_flatbed(self) -> None:
        assert _find_ir_option({"mode": FakeOption(constraint=["Color", "Gray"])}) is None


class TestCoolscanCapabilities:
    def test_film_scanner_inferred_without_source(self) -> None:
        caps = _caps_from_options(COOLSCAN3_OPT, "coolscan3:usb:libusb:001:007")
        assert caps.ir_channel is True
        assert caps.sources  # inferred, not skipped
        assert caps.supported_depths == (8, 16)

    def test_film_scanner_inferred_over_net(self) -> None:
        caps = _caps_from_options(COOLSCAN3_OPT, "net:10.0.0.100:coolscan3:usb:libusb:001:007")
        assert caps.sources

    def test_coolscan2_exact_infrared_option_is_not_advertised(self) -> None:
        opt = {
            "infrared": FakeOption(),
            "source": FakeOption(constraint=["Transparency"]),
        }

        caps = _caps_from_options(opt, "coolscan2:usb:libusb:001:007")

        assert caps.sources == (ScanMode.TRANSPARENCY,)
        assert caps.ir_channel is False

    @pytest.mark.parametrize(
        "infrared_option",
        (
            FakeOption(active=False),
            FakeOption(settable=False),
        ),
    )
    def test_unusable_coolscan3_exact_infrared_is_not_advertised(self, infrared_option: FakeOption) -> None:
        opt = dict(COOLSCAN3_OPT)
        opt["infrared"] = infrared_option

        caps = _caps_from_options(opt, "coolscan3:usb:libusb:001:007")

        assert caps.sources
        assert caps.ir_channel is False

    @pytest.mark.parametrize("option_name", ("ir", "preview_ir"))
    @pytest.mark.parametrize(
        "legacy_option",
        (
            FakeOption(active=False),
            FakeOption(settable=False),
        ),
    )
    def test_legacy_ir_capability_remains_presence_only(self, option_name: str, legacy_option: FakeOption) -> None:
        opt = {
            option_name: legacy_option,
            "source": FakeOption(constraint=["Transparency"]),
        }

        caps = _caps_from_options(opt, "plustek:libusb:001:008")

        assert caps.ir_channel is True

    def test_mystery_exact_infrared_does_not_imply_film_scanner(self) -> None:
        caps = _caps_from_options({"infrared": FakeOption()}, "mystery:001")

        assert caps.sources == ()
        assert caps.ir_channel is False

    def test_mystery_legacy_ir_capability_does_not_imply_film_scanner(self) -> None:
        caps = _caps_from_options({"ir": FakeOption()}, "mystery:001")

        assert caps.ir_channel is True
        assert caps.sources == ()


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

    _INTERNAL = ("recorded", "events", "true_frame", "cancelled", "closed", "opt_map")

    def __init__(self, true_frame: np.ndarray, opt_map: dict | None = None) -> None:
        object.__setattr__(self, "opt_map", COOLSCAN3_OPT if opt_map is None else opt_map)
        object.__setattr__(self, "recorded", {})
        object.__setattr__(self, "events", [])
        object.__setattr__(self, "true_frame", true_frame)
        object.__setattr__(self, "cancelled", False)
        object.__setattr__(self, "closed", False)

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
        self.events.append(("set", name, value))

    def __getattr__(self, name: str) -> Any:
        # python-sane exposes every known option as a readable attribute; the
        # scan path hasattr-guards geometry writes, so reads must not raise.
        if name in self.opt_map:
            return self.recorded.get(name)
        raise AttributeError(f"No readable SANE option: {name}")

    def start(self) -> None:
        self.events.append(("start",))

    def get_parameters(self):
        h, w, _ = self.true_frame.shape
        return ("color", 1, (w, h), 16, w * 4 * 2)

    def arr_snap(self, progress=None) -> np.ndarray:
        if progress is not None:
            # Mimic _sane.c's SaneDev_snap: invoke the callback once per
            # output line with (current_line, total_lines), 1-based.
            h = self.true_frame.shape[0]
            for i in range(1, h + 1):
                progress(i, h)
        if self.true_frame.shape[2] == 3:
            return self.true_frame  # 3-channel frames come back correctly
        return _emulate_python_sane_read(self.true_frame)

    def cancel(self) -> None:
        object.__setattr__(self, "cancelled", True)

    def close(self) -> None:
        object.__setattr__(self, "closed", True)


class ParameterOverrideFakeSaneDev(FakeSaneDev):
    """Inline-IR fake with an explicit get_parameters() result."""

    _INTERNAL = FakeSaneDev._INTERNAL + ("parameters",)

    def __init__(self, true_frame: np.ndarray, parameters: tuple, opt_map: dict | None = None) -> None:
        super().__init__(true_frame, opt_map=opt_map)
        object.__setattr__(self, "parameters", parameters)

    def get_parameters(self):
        return self.parameters


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
    backend._id_remap = {}
    backend._active_sessions = {}
    backend._session_lock = threading.Lock()
    return backend


def test_scan_initializes_sane_before_opening_a_fresh_backend() -> None:
    true = np.zeros((6, 5, 3), dtype=np.uint16)
    dev = FakeSaneDev(true)

    class InitRequiredSaneModule:
        def __init__(self) -> None:
            self.initialized = False
            self.init_calls = 0

        def init(self) -> None:
            self.initialized = True
            self.init_calls += 1

        def open(self, _device_id: str) -> FakeSaneDev:
            assert self.initialized
            return dev

    module = InitRequiredSaneModule()
    backend = SaneBackend.__new__(SaneBackend)
    backend._sane = module
    backend._sane_initialized = False
    backend._devices_cache = None
    backend._id_remap = {}
    backend._active_sessions = {}
    backend._session_lock = threading.Lock()

    result = backend.scan(
        "coolscan3:usb:test",
        ScanParams(dpi=1000, depth=16, capture_ir=False),
        None,
        threading.Event(),
    )

    assert module.init_calls == 1
    assert result.rgb.shape == (6, 5, 3)


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
        _, dev, result = self._run("net:10.0.0.100:coolscan3:usb:libusb:001:007")
        assert result.ir is not None
        assert dev.recorded.get("infrared") is True

    def test_inline_ir_rejects_non_color_frame_metadata_before_split(self) -> None:
        rng = np.random.default_rng(701)
        true = rng.integers(0, 65535, size=(6, 5, 4), dtype=np.uint16)
        dev = ParameterOverrideFakeSaneDev(true, ("gray", 1, (5, 6), 16, 40))
        backend = _make_backend(dev)

        with pytest.raises(RuntimeError, match="format.*gray.*color"):
            backend.scan(
                "coolscan3:usb:libusb:001:007",
                ScanParams(dpi=1000, depth=16, capture_ir=True),
                None,
                threading.Event(),
            )

        assert dev.cancelled

    def test_inline_ir_rejects_nonterminal_frame_before_arr_snap(self) -> None:
        class ReadTrackingDev(ParameterOverrideFakeSaneDev):
            _INTERNAL = ParameterOverrideFakeSaneDev._INTERNAL + ("snap_called",)

            def __init__(self, *args, **kwargs) -> None:
                super().__init__(*args, **kwargs)
                object.__setattr__(self, "snap_called", False)

            def arr_snap(self, progress=None) -> np.ndarray:
                object.__setattr__(self, "snap_called", True)
                return super().arr_snap(progress=progress)

        rng = np.random.default_rng(707)
        true = rng.integers(0, 65535, size=(6, 5, 4), dtype=np.uint16)
        dev = ReadTrackingDev(true, ("color", False, (5, 6), 16, 40))
        backend = _make_backend(dev)

        with pytest.raises(RuntimeError, match="last_frame.*true"):
            backend.scan(
                "coolscan3:usb:libusb:001:007",
                ScanParams(dpi=1000, depth=16, capture_ir=True),
                None,
                threading.Event(),
            )

        assert dev.snap_called is False
        assert dev.cancelled

    def test_inline_ir_rejects_returned_depth_that_differs_from_request(self) -> None:
        rng = np.random.default_rng(702)
        true = rng.integers(0, 65535, size=(6, 5, 4), dtype=np.uint16)
        dev = ParameterOverrideFakeSaneDev(true, ("color", 1, (5, 6), 8, 20))
        backend = _make_backend(dev)

        with pytest.raises(RuntimeError, match="depth 8.*requested 16"):
            backend.scan(
                "coolscan3:usb:libusb:001:007",
                ScanParams(dpi=1000, depth=16, capture_ir=True),
                None,
                threading.Event(),
            )

        assert dev.cancelled

    def test_inline_ir_rejects_returned_depth_outside_supported_sample_sizes(self) -> None:
        rng = np.random.default_rng(704)
        true = rng.integers(0, 65535, size=(6, 5, 4), dtype=np.uint16)
        dev = ParameterOverrideFakeSaneDev(true, ("color", 1, (5, 6), 12, 40))
        backend = _make_backend(dev)

        with pytest.raises(RuntimeError, match="unusable sample depth 12"):
            backend.scan(
                "coolscan3:usb:libusb:001:007",
                ScanParams(dpi=1000, depth=12, capture_ir=True),
                None,
                threading.Event(),
            )

        assert dev.cancelled

    def test_inline_ir_rejects_non_rgbi_bytes_per_line_before_split(self) -> None:
        rng = np.random.default_rng(703)
        true = rng.integers(0, 65535, size=(6, 5, 4), dtype=np.uint16)
        dev = ParameterOverrideFakeSaneDev(true, ("color", 1, (5, 6), 16, 42))
        backend = _make_backend(dev)

        with pytest.raises(RuntimeError, match="bytes_per_line 42.*expected 40"):
            backend.scan(
                "coolscan3:usb:libusb:001:007",
                ScanParams(dpi=1000, depth=16, capture_ir=True),
                None,
                threading.Event(),
            )

        assert dev.cancelled

    def test_ignored_ir_rgb_payload_cannot_fit_by_accident(self) -> None:
        # Eight returned RGB rows contain the same number of samples as six
        # nominal RGBI rows. Payload-size-only repair used to reshape this into
        # a plausible 6x5x4 frame even though SANE declared three-channel BPL.
        rng = np.random.default_rng(705)
        ignored_ir_rgb = rng.integers(0, 65535, size=(8, 5, 3), dtype=np.uint16)
        dev = ParameterOverrideFakeSaneDev(ignored_ir_rgb, ("color", 1, (5, 6), 16, 30))
        backend = _make_backend(dev)

        with pytest.raises(RuntimeError, match="bytes_per_line 30.*expected 40"):
            backend.scan(
                "coolscan3:usb:libusb:001:007",
                ScanParams(dpi=1000, depth=16, capture_ir=True),
                None,
                threading.Event(),
            )

        assert dev.cancelled

    def test_no_ir_requested_scans_plain(self) -> None:
        rng = np.random.default_rng(9)
        true = rng.integers(0, 65535, size=(6, 5, 3), dtype=np.uint16)
        opt = {name: option for name, option in COOLSCAN3_OPT.items() if name != "infrared"}
        dev = FakeSaneDev(true, opt_map=opt)
        backend = _make_backend(dev)
        params = ScanParams(dpi=1000, depth=16, capture_ir=False)
        result = backend.scan("coolscan3:usb:libusb:001:007", params, None, threading.Event())
        assert "infrared" not in dev.recorded
        assert result.ir is None
        assert result.rgb.shape == (6, 5, 3)
        assert np.array_equal(result.rgb, true)

    def test_ir_requested_without_usable_mechanism_fails_before_start(self) -> None:
        rng = np.random.default_rng(10)
        true = rng.integers(0, 65535, size=(6, 5, 3), dtype=np.uint16)
        opt = {name: option for name, option in COOLSCAN3_OPT.items() if name != "infrared"}
        dev = FakeSaneDev(true, opt_map=opt)
        backend = _make_backend(dev)

        with pytest.raises(RuntimeError, match="IR.*no usable.*mechanism"):
            backend.scan(
                "coolscan3:usb:libusb:001:007",
                ScanParams(dpi=1000, depth=16, capture_ir=True),
                None,
                threading.Event(),
            )

        assert ("start",) not in dev.events

    def test_option_strategy_not_claimed_for_coolscan2(self) -> None:
        # coolscan2 exposes the same option name but delivers IR as a separate
        # later frame — the inline-option contract must not be applied to it.
        rng = np.random.default_rng(11)
        dev = FakeSaneDev(rng.integers(0, 65535, size=(6, 5, 4), dtype=np.uint16))
        assert SaneBackend._ir_strategy(dev, "coolscan3:usb:libusb:001:007") == "option"
        assert SaneBackend._ir_strategy(dev, "coolscan2:usb:libusb:001:007") is None
        assert SaneBackend._ir_strategy(dev, "net:10.0.0.100:coolscan2:usb:x") is None


class TestPieusbCompatibility:
    @staticmethod
    def _options() -> dict[str, FakeOption]:
        return {
            "mode": FakeOption(constraint=["Color", "RGBI"]),
            "depth": FakeOption(constraint=[8, 16]),
            "resolution": FakeOption(constraint=[1000]),
            "sharpen": FakeOption(),
            "shading_analysis": FakeOption(),
            "advance": FakeOption(),
            "calibration": FakeOption(),
            "correct_shading": FakeOption(),
            "clean_image": FakeOption(),
            "correct_infrared": FakeOption(),
        }

    def test_net_pieusb_keeps_legacy_rgbi_strategy_without_internal_flags(self) -> None:
        class NativePieusbDev(FakeSaneDev):
            def arr_snap(self, progress=None) -> np.ndarray:
                return self.true_frame

        rng = np.random.default_rng(709)
        true = rng.integers(0, 65535, size=(6, 5, 4), dtype=np.uint16)
        dev = NativePieusbDev(true, opt_map=self._options())
        backend = _make_backend(dev)

        result = backend.scan(
            "net:scanner:pieusb:libusb:001:004",
            ScanParams(dpi=1000, depth=16, capture_ir=True),
            None,
            threading.Event(),
        )

        assert dev.recorded["mode"] == "RGBI"
        assert "clean_image" not in dev.recorded
        assert "correct_infrared" not in dev.recorded
        assert result.ir is not None

    def test_direct_pieusb_keeps_internal_strategy_and_flags(self) -> None:
        rng = np.random.default_rng(710)
        true = rng.integers(0, 65535, size=(6, 5, 4), dtype=np.uint16)
        dev = FakeSaneDev(true, opt_map=self._options())
        backend = _make_backend(dev)

        result = backend.scan(
            "pieusb:libusb:001:004",
            ScanParams(dpi=1000, depth=16, capture_ir=True),
            None,
            threading.Event(),
        )

        assert dev.recorded["mode"] == "RGBI"
        assert dev.recorded["clean_image"] is False
        assert dev.recorded["correct_infrared"] is True
        assert result.ir is not None

    def test_net_pieusb_prefix_alone_does_not_add_film_sources(self) -> None:
        caps = _caps_from_options({}, "net:scanner:pieusb:libusb:001:004")

        assert caps.sources == ()


class TestGenericRgbiCompatibility:
    @staticmethod
    def _options() -> dict[str, FakeOption]:
        return {
            "mode": FakeOption(constraint=["Color", "RGBI"]),
            "depth": FakeOption(constraint=[8, 16]),
            "resolution": FakeOption(constraint=[1000]),
        }

    def test_native_four_channel_rgbi_ignores_coolscan_stream_metadata(self) -> None:
        class NativeRgbiDev(ParameterOverrideFakeSaneDev):
            def arr_snap(self, progress=None) -> np.ndarray:
                return self.true_frame

        rng = np.random.default_rng(711)
        true = rng.integers(0, 65535, size=(6, 5, 4), dtype=np.uint16)
        # Generic RGBI historically trusted the native four-channel ndarray;
        # these parameters deliberately do not satisfy coolscan3's raw-stream
        # contract and must not trigger its reinterpretation or hard gates.
        dev = NativeRgbiDev(true, ("gray", False, (5, 8), 8, 15), opt_map=self._options())
        backend = _make_backend(dev)

        result = backend.scan(
            "vendorfilm:libusb:001:009",
            ScanParams(dpi=1000, depth=16, capture_ir=True),
            None,
            threading.Event(),
        )

        assert dev.recorded["mode"] == "RGBI"
        assert np.array_equal(result.rgb, true[:, :, :3])
        assert result.ir is not None and np.array_equal(result.ir, true[:, :, 3])

    def test_missing_native_fourth_channel_keeps_legacy_soft_behavior(self) -> None:
        rng = np.random.default_rng(712)
        true = rng.integers(0, 65535, size=(6, 5, 3), dtype=np.uint16)
        dev = FakeSaneDev(true, opt_map=self._options())
        backend = _make_backend(dev)

        result = backend.scan(
            "vendorfilm:libusb:001:009",
            ScanParams(dpi=1000, depth=16, capture_ir=True),
            None,
            threading.Event(),
        )

        assert np.array_equal(result.rgb, true)
        assert result.ir is None


class TestAutofocusControls:
    """Autofocus plumbing and archival fail-loud semantics."""

    def _scan(self, params: ScanParams, dev: FakeSaneDev):
        backend = _make_backend(dev)
        return backend.scan("coolscan3:usb:libusb:001:007", params, None, threading.Event())

    def test_autofocus_enabled_by_default(self) -> None:
        rng = np.random.default_rng(13)
        dev = FakeSaneDev(rng.integers(0, 65535, size=(6, 5, 4), dtype=np.uint16))
        self._scan(ScanParams(dpi=1000, depth=16, capture_ir=True), dev)
        assert dev.recorded.get("autofocus") is True

    def test_autofocus_can_be_disabled(self) -> None:
        rng = np.random.default_rng(14)
        dev = FakeSaneDev(rng.integers(0, 65535, size=(6, 5, 4), dtype=np.uint16))
        self._scan(ScanParams(dpi=1000, depth=16, capture_ir=True, autofocus=False), dev)
        assert "autofocus" not in dev.recorded

    def test_requested_ir_without_channel_raises(self) -> None:
        import pytest

        rng = np.random.default_rng(17)
        # Device frame is plain 3-channel: inline-IR strategy cannot deliver
        # a 4th channel and the scan must fail loud, not return ir=None.
        dev = FakeSaneDev(rng.integers(0, 65535, size=(6, 5, 3), dtype=np.uint16))
        with pytest.raises(RuntimeError, match="no 4th channel"):
            self._scan(ScanParams(dpi=1000, depth=16, capture_ir=True), dev)
        assert dev.cancelled


class TestFrameSelection:
    """Roll-scanner frame selection (params.frame) plumbing."""

    def _scan(self, params: ScanParams, dev: FakeSaneDev):
        backend = _make_backend(dev)
        return backend.scan("coolscan3:usb:libusb:001:007", params, None, threading.Event())

    def test_frame_recorded_when_set(self) -> None:
        rng = np.random.default_rng(18)
        dev = FakeSaneDev(rng.integers(0, 65535, size=(6, 5, 4), dtype=np.uint16))
        self._scan(ScanParams(dpi=1000, depth=16, capture_ir=True, frame=12), dev)
        assert dev.recorded.get("frame") == 12

    def test_frame_untouched_when_none(self) -> None:
        rng = np.random.default_rng(19)
        dev = FakeSaneDev(rng.integers(0, 65535, size=(6, 5, 4), dtype=np.uint16))
        self._scan(ScanParams(dpi=1000, depth=16, capture_ir=True, frame=None), dev)
        assert "frame" not in dev.recorded

    def test_any_offset_scan_shortens_the_feed_extent(self) -> None:
        # The scan blacks out one pitch past the frame start (any frame) —
        # the offset must shorten the extent, not deliver a black tail.
        rng = np.random.default_rng(26)
        dev = FakeSaneDev(rng.integers(0, 65535, size=(6, 5, 4), dtype=np.uint16))
        self._scan(ScanParams(dpi=1000, depth=16, capture_ir=True, frame=12, frame_offset_mm=5.5), dev)
        assert dev.recorded.get("subframe") == 5.5
        assert dev.recorded.get("br_y") == int(round((1.0 - 5.5 / 37.8333) * 5958))

    def test_zero_offset_keeps_the_full_extent(self) -> None:
        rng = np.random.default_rng(27)
        dev = FakeSaneDev(rng.integers(0, 65535, size=(6, 5, 4), dtype=np.uint16))
        self._scan(ScanParams(dpi=1000, depth=16, capture_ir=True, frame=12, frame_offset_mm=0.0), dev)
        assert "br_y" not in dev.recorded  # no window, no cap → geometry untouched

    def test_negative_offset_is_clamped_to_zero(self) -> None:
        # Below 0 is unreachable — the scan blacks out at the frame boundary.
        rng = np.random.default_rng(25)
        dev = FakeSaneDev(rng.integers(0, 65535, size=(6, 5, 4), dtype=np.uint16))
        self._scan(ScanParams(dpi=1000, depth=16, capture_ir=True, frame=12, frame_offset_mm=-1.0), dev)
        assert dev.recorded.get("frame") == 12
        assert dev.recorded.get("subframe") == 0.0

    def test_frame_without_option_raises(self) -> None:
        import pytest

        rng = np.random.default_rng(20)
        opt = {k: v for k, v in COOLSCAN3_OPT.items() if k != "frame"}
        dev = FakeSaneDev(rng.integers(0, 65535, size=(6, 5, 4), dtype=np.uint16), opt_map=opt)
        with pytest.raises(RuntimeError, match="frame-selection option"):
            self._scan(ScanParams(dpi=1000, depth=16, capture_ir=True, frame=5), dev)
        # The missing-option check must fire before any attempt to set dev.frame.
        assert "frame" not in dev.recorded


class TestSnapProgressForwarding:
    """Cancellation responsiveness: progress forwarding during arr_snap()'s
    blocking read. See _snap_progress_callback's docstring for why the
    callback forwards progress but does NOT also raise to cancel the read
    early — python-sane 2.9.2's C reader Py_DECREFs the callback's return
    value before checking whether it raised, so an exception here would
    Py_DECREF(NULL) (undefined behaviour) against the real `_sane` extension.
    """

    def _scan(self, params: ScanParams, dev: FakeSaneDev, progress=None, cancel=None):
        backend = _make_backend(dev)
        return backend.scan("coolscan3:usb:libusb:001:007", params, progress, cancel or threading.Event())

    def test_progress_forwarded_during_read(self) -> None:
        rng = np.random.default_rng(22)
        dev = FakeSaneDev(rng.integers(0, 65535, size=(6, 5, 4), dtype=np.uint16))
        seen: list = []
        self._scan(ScanParams(dpi=1000, depth=16, capture_ir=True), dev, progress=seen.append)
        assert seen[0] == 0.0
        assert seen[-1] == 1.0
        # Per-line updates land strictly between the 0%/100% bookends —
        # this is the actual "progress bar moves during the read" behavior.
        assert any(0 < f < 1 for f in seen)
        assert seen == sorted(seen)  # monotonic, never regresses

    def test_no_progress_callable_is_a_no_op(self) -> None:
        # progress=None (as used by callers that don't want updates) must
        # not blow up the per-line forwarding.
        rng = np.random.default_rng(24)
        true = rng.integers(0, 65535, size=(6, 5, 4), dtype=np.uint16)
        dev = FakeSaneDev(true)
        result = self._scan(ScanParams(dpi=1000, depth=16, capture_ir=True), dev, progress=None)
        assert np.array_equal(result.rgb, true[:, :, :3])

    def test_bad_progress_callback_does_not_abort_scan(self) -> None:
        rng = np.random.default_rng(23)
        true = rng.integers(0, 65535, size=(6, 5, 4), dtype=np.uint16)
        dev = FakeSaneDev(true)

        def bad_progress(_frac: float) -> None:
            raise ValueError("boom")

        result = self._scan(ScanParams(dpi=1000, depth=16, capture_ir=True), dev, progress=bad_progress)
        assert result.rgb.shape == (6, 5, 3)
        assert np.array_equal(result.rgb, true[:, :, :3])

    def test_cancel_during_read_honored_after_read_completes(self) -> None:
        """Cancelling mid-read isn't instantaneous (see class docstring), but
        the read still completes and the existing post-read check honors the
        cancellation once arr_snap() returns."""
        import pytest

        rng = np.random.default_rng(21)
        dev = FakeSaneDev(rng.integers(0, 65535, size=(6, 5, 4), dtype=np.uint16))
        cancel = threading.Event()
        seen: list = []

        def progress(frac: float) -> None:
            seen.append(frac)
            if frac >= 0.5:
                cancel.set()

        with pytest.raises(RuntimeError, match="Scan cancelled"):
            self._scan(ScanParams(dpi=1000, depth=16, capture_ir=True), dev, progress=progress, cancel=cancel)
        assert dev.cancelled
        # The read really was in progress (not just the pre-start check) —
        # we saw live fractional updates before cancellation was honored.
        assert any(0 < f < 1 for f in seen)
