"""Tests for coolscan3-style inline IR: option strategy, net IDs, channel repair.

coolscan3 (Nikon Coolscan LS-50/5000) exposes IR as a boolean `infrared`
option; the frame then carries 4 samples/pixel while reporting
SANE_FRAME_RGB (pieusb convention). python-sane's C reader hardcodes
3 samples/pixel for non-gray frames, so the array arrives byte-intact but
misshaped — `_reinterpret_channels` recovers it from the frame geometry.
"""

import math
import threading
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
import pytest

from negpy.infrastructure.scanners.params import RegisteredScanGeometry, ScanMode, ScanParams
from negpy.infrastructure.scanners.sane_backend import (
    SaneBackend,
    _align_split_ir,
    _caps_from_options,
    _detect_ir,
    _estimate_split_shift,
    _find_ir_option,
    _persist_split_diagnostics,
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


def test_split_diagnostics_preserve_both_rgb_views_and_ir_after_refusal(tmp_path) -> None:
    reference = np.arange(6 * 5 * 3, dtype=np.uint16).reshape(6, 5, 3)
    proxy = reference + 1
    ir = np.arange(6 * 5, dtype=np.uint16).reshape(6, 5)

    _persist_split_diagnostics(
        str(tmp_path),
        reference_rgb=reference,
        proxy_rgb=proxy,
        ir=ir,
        error=RuntimeError("alignment refused"),
    )

    assert np.array_equal(np.load(tmp_path / "multisample-rgb.npy", allow_pickle=False), reference)
    assert np.array_equal(np.load(tmp_path / "single-pass-rgb-proxy.npy", allow_pickle=False), proxy)
    assert np.array_equal(np.load(tmp_path / "single-pass-ir.npy", allow_pickle=False), ir)
    assert "alignment refused" in (tmp_path / "manifest.json").read_text(encoding="utf-8")


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
        if name in self.opt_map and name in self.recorded:
            return self.recorded[name]
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


class SplitCaptureFakeSaneDev(FakeSaneDev):
    """Coolscan fake that rejects multisampled IR like the real LS-5000.

    It accepts an N-pass RGB capture and a later 1-pass RGBI capture on the
    same open handle.  Asking for N-pass RGBI reproduces the user-visible
    failure that the split-capture path must avoid.

    The exposure counter is enforced only while the `frame_count` option is
    active, mirroring the real backend: a single-frame holder marks it
    inactive and never decrements or checks it.
    """

    _INTERNAL = FakeSaneDev._INTERNAL + (
        "multi_rgb",
        "single_rgbi",
        "active_frame",
        "start_settings",
        "remaining_frame_count",
        "on_set",
    )

    def __init__(self, multi_rgb: np.ndarray, single_rgbi: np.ndarray, opt_map: dict | None = None) -> None:
        super().__init__(single_rgbi, opt_map=opt_map)
        object.__setattr__(self, "multi_rgb", multi_rgb)
        object.__setattr__(self, "single_rgbi", single_rgbi)
        object.__setattr__(self, "active_frame", single_rgbi)
        object.__setattr__(self, "start_settings", [])
        object.__setattr__(self, "remaining_frame_count", 1)
        object.__setattr__(self, "on_set", None)

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "frame_count":
            object.__setattr__(self, "remaining_frame_count", int(value))
        super().__setattr__(name, value)
        hook = self.on_set
        if hook is not None and name not in self._INTERNAL:
            hook(name, value)

    def _counter_enforced(self) -> bool:
        option = self.opt_map.get("frame_count")
        return option is not None and option.is_active()

    def start(self) -> None:
        if self._counter_enforced() and self.remaining_frame_count == 0:
            raise OSError("no more frames")
        settings = dict(self.recorded)
        self.start_settings.append(settings)
        self.events.append(("start",))
        infrared = bool(settings.get("infrared", False))
        samples = int(settings.get("samples_per_scan", 1))
        if infrared and samples > 1:
            object.__setattr__(self, "active_frame", None)
        elif infrared:
            object.__setattr__(self, "active_frame", self.single_rgbi)
        else:
            object.__setattr__(self, "active_frame", self.multi_rgb)

    def get_parameters(self):
        frame = self.active_frame if self.active_frame is not None else self.single_rgbi
        h, w, channels = frame.shape
        return ("color", 1, (w, h), 16, w * channels * 2)

    def arr_snap(self, progress=None) -> np.ndarray:
        if self.active_frame is None:
            raise OSError("LS-5000 wedged by multisampled infrared")
        frame = self.active_frame
        if progress is not None:
            for i in range(1, frame.shape[0] + 1):
                progress(i, frame.shape[0])
        object.__setattr__(self, "remaining_frame_count", 0)
        if frame.shape[2] == 3:
            return frame
        return _emulate_python_sane_read(frame)


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
        assert result.ir_alignment is None  # single-capture path measures nothing

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
        true = rng.integers(0, 65535, size=(6, 5, 3), dtype=np.uint16)
        dev = FakeSaneDev(true, opt_map=self._options())
        backend = _make_backend(dev)

        result = backend.scan(
            "pieusb:libusb:001:004",
            ScanParams(dpi=1000, depth=16, capture_ir=True),
            None,
            threading.Event(),
        )

        assert dev.recorded["mode"] == "Color"
        assert dev.recorded["clean_image"] is True
        assert dev.recorded["correct_infrared"] is True
        assert result.ir is None

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


class TestArchivalControls:
    """Autofocus + multi-sampling plumbing and archival fail-loud semantics."""

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

    @pytest.mark.parametrize("constraint", ((1, 16, 1), [1, 2, 4, 16]))
    def test_supported_samples_per_scan_recorded(self, constraint: tuple[int, ...] | list[int]) -> None:
        rng = np.random.default_rng(15)
        opt = dict(COOLSCAN3_OPT)
        opt["samples_per_scan"] = FakeOption(constraint=constraint)
        dev = FakeSaneDev(rng.integers(0, 65535, size=(6, 5, 3), dtype=np.uint16), opt_map=opt)
        self._scan(ScanParams(dpi=1000, depth=16, capture_ir=False, samples_per_scan=16), dev)
        assert dev.recorded.get("samples_per_scan") == 16

    def test_coolscan_multisample_ir_uses_two_captures_without_repositioning(self) -> None:
        rng = np.random.default_rng(151)
        multi_rgb = rng.integers(0, 65535, size=(6, 5, 3), dtype=np.uint16)
        ir = rng.integers(0, 65535, size=(6, 5), dtype=np.uint16)
        single_rgbi = np.dstack((multi_rgb, ir))
        dev = SplitCaptureFakeSaneDev(multi_rgb, single_rgbi)
        backend = _make_backend(dev)

        result = backend.scan(
            "coolscan3:usb:libusb:001:007",
            ScanParams(
                dpi=4000,
                depth=16,
                capture_ir=True,
                samples_per_scan=4,
                frame=3,
                auto_exposure=True,
            ),
            None,
            threading.Event(),
        )

        assert np.array_equal(result.rgb, multi_rgb)
        assert result.ir is not None and np.array_equal(result.ir, ir)
        # One reservation is a design requirement: both captures must ride a
        # single sane.open() so the feeder can never be repositioned between.
        assert backend._sane.opened == ["coolscan3:usb:libusb:001:007"]
        assert result.ir_alignment is not None
        assert result.ir_alignment.mode == "identity"
        assert result.ir_alignment.dx_px == 0.0 and result.ir_alignment.dy_px == 0.0
        assert len(dev.start_settings) == 2
        assert dev.start_settings[0]["samples_per_scan"] == 4
        assert dev.start_settings[0].get("infrared", False) is False
        assert dev.start_settings[0]["autofocus"] is True
        assert dev.start_settings[0]["ae"] is True
        assert dev.start_settings[1]["samples_per_scan"] == 1
        assert dev.start_settings[1]["infrared"] is True
        assert dev.start_settings[1]["autofocus"] is False
        assert dev.start_settings[1]["ae"] is False

        start_indices = [i for i, event in enumerate(dev.events) if event == ("start",)]
        frame_sets = [i for i, event in enumerate(dev.events) if event[:2] == ("set", "frame")]
        frame_count_sets = [i for i, event in enumerate(dev.events) if event[:2] == ("set", "frame_count")]
        assert len(start_indices) == 2
        assert len(frame_sets) == 1 and frame_sets[0] < start_indices[0]
        assert frame_count_sets and start_indices[0] < frame_count_sets[-1] < start_indices[1]

    def test_coolscan_split_capture_validates_second_pass_rgbi_metadata(self) -> None:
        class ThreeChannelSecondPassMetadata(SplitCaptureFakeSaneDev):
            def get_parameters(self):
                frame_format, last, (width, lines), depth, bytes_per_line = super().get_parameters()
                if len(self.start_settings) == 2:
                    bytes_per_line = width * 3 * math.ceil(depth / 8)
                return frame_format, last, (width, lines), depth, bytes_per_line

        rng = np.random.default_rng(706)
        multi_rgb = rng.integers(0, 65535, size=(6, 5, 3), dtype=np.uint16)
        ir = rng.integers(0, 65535, size=(6, 5), dtype=np.uint16)
        dev = ThreeChannelSecondPassMetadata(multi_rgb, np.dstack((multi_rgb, ir)))
        backend = _make_backend(dev)

        with pytest.raises(RuntimeError, match="Single-pass Coolscan.*bytes_per_line 30.*expected 40"):
            backend.scan(
                "coolscan3:usb:libusb:001:007",
                ScanParams(dpi=1000, depth=16, capture_ir=True, samples_per_scan=4),
                None,
                threading.Event(),
            )

        assert len(dev.start_settings) == 2
        assert dev.cancelled

    def test_coolscan_split_capture_rejects_nonterminal_second_pass_before_read(self) -> None:
        class NonterminalSecondPass(SplitCaptureFakeSaneDev):
            _INTERNAL = SplitCaptureFakeSaneDev._INTERNAL + ("read_count",)

            def __init__(self, *args, **kwargs) -> None:
                super().__init__(*args, **kwargs)
                object.__setattr__(self, "read_count", 0)

            def get_parameters(self):
                frame_format, last, dimensions, depth, bytes_per_line = super().get_parameters()
                if len(self.start_settings) == 2:
                    last = False
                return frame_format, last, dimensions, depth, bytes_per_line

            def arr_snap(self, progress=None) -> np.ndarray:
                object.__setattr__(self, "read_count", self.read_count + 1)
                return super().arr_snap(progress=progress)

        rng = np.random.default_rng(708)
        multi_rgb = rng.integers(0, 65535, size=(6, 5, 3), dtype=np.uint16)
        ir = rng.integers(0, 65535, size=(6, 5), dtype=np.uint16)
        dev = NonterminalSecondPass(multi_rgb, np.dstack((multi_rgb, ir)))
        backend = _make_backend(dev)

        with pytest.raises(RuntimeError, match="Single-pass Coolscan.*last_frame.*true"):
            backend.scan(
                "coolscan3:usb:libusb:001:007",
                ScanParams(dpi=1000, depth=16, capture_ir=True, samples_per_scan=4),
                None,
                threading.Event(),
            )

        assert len(dev.start_settings) == 2
        assert dev.read_count == 1
        assert dev.cancelled

    def test_coolscan_split_capture_aligns_ir_to_multisampled_rgb(self) -> None:
        rng = np.random.default_rng(152)
        height, width = 120, 160
        dx, dy = 2, 8
        base = rng.integers(2000, 62000, size=(height, width), dtype=np.uint16)
        multi_rgb = np.repeat(base[:, :, None], 3, axis=2)
        target_ir = rng.integers(1000, 64000, size=(height, width), dtype=np.uint16)

        def shifted(array: np.ndarray) -> np.ndarray:
            out = np.zeros_like(array)
            out[dy:, dx:] = array[: height - dy, : width - dx]
            return out

        proxy_rgb = shifted(multi_rgb)
        moving_ir = shifted(target_ir)
        dev = SplitCaptureFakeSaneDev(
            multi_rgb,
            np.dstack((proxy_rgb, moving_ir)),
        )

        result = self._scan(
            ScanParams(
                dpi=4000,
                depth=16,
                capture_ir=True,
                samples_per_scan=4,
                frame=3,
            ),
            dev,
        )

        assert result.rgb.shape == (height - dy, width - dx, 3)
        assert result.ir is not None and result.ir.shape == (height - dy, width - dx)
        assert np.array_equal(result.rgb, multi_rgb[: height - dy, : width - dx])
        error = np.abs(result.ir.astype(np.int32) - target_ir[: height - dy, : width - dx].astype(np.int32))
        assert float(np.percentile(error, 99)) <= 2.0
        alignment = result.ir_alignment
        assert alignment is not None and alignment.mode == "phase-ecc"
        assert abs(alignment.dx_px - dx) <= 0.5 and abs(alignment.dy_px - dy) <= 0.5
        assert min(alignment.phase_responses) >= 0.10
        assert alignment.ecc_coefficient is not None and alignment.ecc_coefficient >= 0.65

    def test_coolscan_split_capture_progress_spans_both_reads(self) -> None:
        rng = np.random.default_rng(153)
        multi_rgb = rng.integers(0, 65535, size=(12, 10, 3), dtype=np.uint16)
        ir = rng.integers(0, 65535, size=(12, 10), dtype=np.uint16)
        dev = SplitCaptureFakeSaneDev(multi_rgb, np.dstack((multi_rgb, ir)))
        backend = _make_backend(dev)
        seen: list[float] = []

        backend.scan(
            "coolscan3:usb:libusb:001:007",
            ScanParams(dpi=4000, depth=16, capture_ir=True, samples_per_scan=4),
            seen.append,
            threading.Event(),
        )

        assert seen[0] == 0.0 and seen[-1] == 1.0
        assert seen == sorted(seen)
        assert any(0.0 < value < 0.75 for value in seen)
        assert any(0.75 < value < 1.0 for value in seen)

    def test_coolscan_split_capture_cancelled_between_reads_never_starts_ir(self) -> None:
        rng = np.random.default_rng(154)
        multi_rgb = rng.integers(0, 65535, size=(12, 10, 3), dtype=np.uint16)
        ir = rng.integers(0, 65535, size=(12, 10), dtype=np.uint16)
        dev = SplitCaptureFakeSaneDev(multi_rgb, np.dstack((multi_rgb, ir)))
        backend = _make_backend(dev)
        cancel = threading.Event()

        def stop_after_rgb(fraction: float) -> None:
            if fraction >= 0.75:
                cancel.set()

        with pytest.raises(RuntimeError, match="cancelled"):
            backend.scan(
                "coolscan3:usb:libusb:001:007",
                ScanParams(dpi=4000, depth=16, capture_ir=True, samples_per_scan=4),
                stop_after_rgb,
                cancel,
            )

        assert len(dev.start_settings) == 1

    def test_samples_without_option_raises(self) -> None:
        import pytest

        rng = np.random.default_rng(16)
        opt = {k: v for k, v in COOLSCAN3_OPT.items() if k != "samples_per_scan"}
        dev = FakeSaneDev(rng.integers(0, 65535, size=(6, 5, 4), dtype=np.uint16), opt_map=opt)
        with pytest.raises(RuntimeError, match="multi-sampling"):
            self._scan(ScanParams(dpi=1000, depth=16, capture_ir=True, samples_per_scan=4), dev)

    def test_requested_ir_without_channel_raises(self) -> None:
        import pytest

        rng = np.random.default_rng(17)
        # Device frame is plain 3-channel: inline-IR strategy cannot deliver
        # a 4th channel and the scan must fail loud, not return ir=None.
        dev = FakeSaneDev(rng.integers(0, 65535, size=(6, 5, 3), dtype=np.uint16))
        with pytest.raises(RuntimeError, match="no 4th channel"):
            self._scan(ScanParams(dpi=1000, depth=16, capture_ir=True), dev)
        assert dev.cancelled


def _shifted(array: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """Displace content by (dx, dy) with zero fill; supports negative shifts."""
    out = np.zeros_like(array)
    h, w = array.shape[:2]
    out[max(0, dy) : h + min(0, dy), max(0, dx) : w + min(0, dx)] = array[max(0, -dy) : h + min(0, -dy), max(0, -dx) : w + min(0, -dx)]
    return out


class TestSplitCaptureHardening:
    """Fail-closed pre-hardware checklist: option preflight, second-capture
    boundary failures, cancellation windows, and single-frame holders."""

    DEVICE = "coolscan3:usb:libusb:001:007"

    def _split_dev(self, seed: int = 200, h: int = 12, w: int = 10, opt_map: dict | None = None) -> SplitCaptureFakeSaneDev:
        rng = np.random.default_rng(seed)
        multi_rgb = rng.integers(0, 65535, size=(h, w, 3), dtype=np.uint16)
        ir = rng.integers(0, 65535, size=(h, w), dtype=np.uint16)
        return SplitCaptureFakeSaneDev(multi_rgb, np.dstack((multi_rgb, ir)), opt_map=opt_map)

    def _params(self, **overrides) -> ScanParams:
        base: dict[str, Any] = dict(dpi=1000, depth=16, capture_ir=True, samples_per_scan=4)
        base.update(overrides)
        return ScanParams(**base)

    def test_single_frame_holder_with_inactive_frame_count_completes(self) -> None:
        # A true single-frame holder marks frame_count inactive and coolscan3
        # never decrements/checks it — the split capture must neither require
        # nor write the option, and both starts must still complete.
        opt = dict(COOLSCAN3_OPT)
        opt["frame"] = FakeOption(constraint=(1, 0, 1), active=False)
        opt["frame_count"] = FakeOption(constraint=(1, 0, 1), active=False)
        dev = self._split_dev(opt_map=opt)
        backend = _make_backend(dev)

        result = backend.scan(self.DEVICE, self._params(), None, threading.Event())

        assert result.ir is not None
        assert len(dev.start_settings) == 2
        assert not any(event[:2] == ("set", "frame_count") for event in dev.events)

    def test_roll_frame_with_inactive_frame_count_fails_before_positioning(self) -> None:
        opt = dict(COOLSCAN3_OPT)
        opt["frame_count"] = FakeOption(constraint=(1, 0, 1), active=False)
        dev = self._split_dev(opt_map=opt)
        backend = _make_backend(dev)

        with pytest.raises(RuntimeError, match="frame_count.*inactive"):
            backend.scan(self.DEVICE, self._params(frame=3), None, threading.Event())

        assert not any(event[0] == "set" and event[1] in {"frame", "subframe", "br_y"} for event in dev.events)
        assert ("start",) not in dev.events

    def test_inactive_boolean_ir_fails_before_transport_or_geometry_writes(self) -> None:
        opt = dict(COOLSCAN3_OPT)
        opt["infrared"] = FakeOption(active=False)
        dev = self._split_dev(opt_map=opt)
        backend = _make_backend(dev)
        geometry = RegisteredScanGeometry(frame=3, subframe_mm=6.35, br_y_device_px=800)

        with pytest.raises(RuntimeError, match="infrared.*inactive"):
            backend.scan(
                self.DEVICE,
                self._params(registered_geometry=geometry),
                None,
                threading.Event(),
            )

        assert not any(event[0] == "set" and event[1] in {"frame", "subframe", "br_y"} for event in dev.events)
        assert ("start",) not in dev.events

    def test_read_only_boolean_ir_fails_before_transport_or_geometry_writes(self) -> None:
        opt = dict(COOLSCAN3_OPT)
        opt["infrared"] = FakeOption(active=True, settable=False)
        dev = self._split_dev(opt_map=opt)
        backend = _make_backend(dev)
        geometry = RegisteredScanGeometry(frame=3, subframe_mm=6.35, br_y_device_px=800)

        with pytest.raises(RuntimeError, match="infrared.*not settable"):
            backend.scan(
                self.DEVICE,
                self._params(registered_geometry=geometry),
                None,
                threading.Event(),
            )

        assert not any(event[0] == "set" and event[1] in {"frame", "subframe", "br_y"} for event in dev.events)
        assert ("start",) not in dev.events

    def test_indeterminate_ir_settable_state_fails_closed_before_transport(self) -> None:
        class IndeterminateSettableOption(FakeOption):
            def is_settable(self) -> bool:
                raise OSError("descriptor query failed")

        opt = dict(COOLSCAN3_OPT)
        opt["infrared"] = IndeterminateSettableOption()
        dev = self._split_dev(opt_map=opt)
        backend = _make_backend(dev)
        geometry = RegisteredScanGeometry(frame=3, subframe_mm=6.35, br_y_device_px=800)

        with pytest.raises(RuntimeError, match="Could not determine whether requested SANE option 'infrared' is settable"):
            backend.scan(
                self.DEVICE,
                self._params(registered_geometry=geometry),
                None,
                threading.Event(),
            )

        assert not any(event[0] == "set" and event[1] in {"frame", "subframe", "br_y"} for event in dev.events)
        assert ("start",) not in dev.events

    def test_read_only_ir_without_is_active_still_fails_before_transport(self) -> None:
        class SettableOnlyOption:
            constraint = None

            def is_settable(self) -> bool:
                return False

        opt = dict(COOLSCAN3_OPT)
        opt["infrared"] = SettableOnlyOption()
        dev = self._split_dev(opt_map=opt)
        backend = _make_backend(dev)
        geometry = RegisteredScanGeometry(frame=3, subframe_mm=6.35, br_y_device_px=800)

        with pytest.raises(RuntimeError, match="infrared.*not settable"):
            backend.scan(
                self.DEVICE,
                self._params(registered_geometry=geometry),
                None,
                threading.Event(),
            )

        assert not any(event[0] == "set" and event[1] in {"frame", "subframe", "br_y"} for event in dev.events)
        assert ("start",) not in dev.events

    @pytest.mark.parametrize("break_option", ("missing", "inactive"))
    def test_multisample_option_validated_before_any_positioning(self, break_option: str) -> None:
        if break_option == "missing":
            opt = {k: v for k, v in COOLSCAN3_OPT.items() if k != "samples_per_scan"}
            expected = "multi-sampling"
        else:
            opt = dict(COOLSCAN3_OPT)
            opt["samples_per_scan"] = FakeOption(constraint=(1, 16, 1), active=False)
            expected = "samples_per_scan.*inactive"
        dev = self._split_dev(opt_map=opt)
        backend = _make_backend(dev)
        geometry = RegisteredScanGeometry(frame=3, subframe_mm=6.35, br_y_device_px=800)

        with pytest.raises(RuntimeError, match=expected):
            backend.scan(
                self.DEVICE,
                self._params(registered_geometry=geometry, auto_exposure=True),
                None,
                threading.Event(),
            )

        assert not any(event[0] == "set" and event[1] in {"frame", "subframe", "br_y"} for event in dev.events)
        assert ("start",) not in dev.events

    def test_out_of_range_multisample_value_fails_before_positioning(self) -> None:
        dev = self._split_dev()
        backend = _make_backend(dev)
        geometry = RegisteredScanGeometry(frame=3, subframe_mm=6.35, br_y_device_px=800)

        with pytest.raises(RuntimeError, match="samples_per_scan=17.*constraint"):
            backend.scan(
                self.DEVICE,
                self._params(samples_per_scan=17, registered_geometry=geometry),
                None,
                threading.Event(),
            )

        assert not any(event[0] == "set" and event[1] in {"frame", "subframe", "br_y"} for event in dev.events)
        assert ("start",) not in dev.events

    def test_unlisted_multisample_value_fails_before_positioning(self) -> None:
        opt = dict(COOLSCAN3_OPT)
        opt["samples_per_scan"] = FakeOption(constraint=[1, 2, 4, 16])
        dev = self._split_dev(opt_map=opt)
        backend = _make_backend(dev)
        geometry = RegisteredScanGeometry(frame=3, subframe_mm=6.35, br_y_device_px=800)

        with pytest.raises(RuntimeError, match="samples_per_scan=8.*constraint"):
            backend.scan(
                self.DEVICE,
                self._params(samples_per_scan=8, registered_geometry=geometry),
                None,
                threading.Event(),
            )

        assert not any(event[0] == "set" and event[1] in {"frame", "subframe", "br_y"} for event in dev.events)
        assert ("start",) not in dev.events

    def test_second_start_failure_is_contextualized_and_cleaned_up(self) -> None:
        class FailingSecondStart(SplitCaptureFakeSaneDev):
            def start(self) -> None:
                if len(self.start_settings) == 1:
                    raise OSError("SCAN rejected")
                super().start()

        rng = np.random.default_rng(201)
        multi_rgb = rng.integers(0, 65535, size=(12, 10, 3), dtype=np.uint16)
        ir = rng.integers(0, 65535, size=(12, 10), dtype=np.uint16)
        dev = FailingSecondStart(multi_rgb, np.dstack((multi_rgb, ir)))
        backend = _make_backend(dev)

        with pytest.raises(RuntimeError, match="Infrared scan failed.*SCAN rejected"):
            backend.scan(self.DEVICE, self._params(), None, threading.Event())

        assert len(dev.start_settings) == 1
        assert dev.cancelled and dev.closed

    def test_second_read_failure_is_contextualized_and_cleaned_up(self) -> None:
        class FailingSecondRead(SplitCaptureFakeSaneDev):
            _INTERNAL = SplitCaptureFakeSaneDev._INTERNAL + ("reads",)

            def __init__(self, *args, **kwargs) -> None:
                super().__init__(*args, **kwargs)
                object.__setattr__(self, "reads", 0)

            def arr_snap(self, progress=None) -> np.ndarray:
                object.__setattr__(self, "reads", self.reads + 1)
                if self.reads == 2:
                    raise OSError("USB stall mid-frame")
                return super().arr_snap(progress)

        rng = np.random.default_rng(205)
        multi_rgb = rng.integers(0, 65535, size=(12, 10, 3), dtype=np.uint16)
        ir = rng.integers(0, 65535, size=(12, 10), dtype=np.uint16)
        dev = FailingSecondRead(multi_rgb, np.dstack((multi_rgb, ir)))
        backend = _make_backend(dev)

        with pytest.raises(RuntimeError, match="Infrared scan failed.*USB stall"):
            backend.scan(self.DEVICE, self._params(), None, threading.Event())

        assert len(dev.start_settings) == 2
        assert dev.cancelled and dev.closed

    def test_cancel_immediately_before_second_start_prevents_it(self) -> None:
        dev = self._split_dev()
        backend = _make_backend(dev)
        cancel = threading.Event()

        def cancel_on_ir_enable(name: str, value: Any) -> None:
            if name == "infrared" and value is True:
                cancel.set()

        dev.on_set = cancel_on_ir_enable

        with pytest.raises(RuntimeError, match="Scan cancelled"):
            backend.scan(self.DEVICE, self._params(), None, cancel)

        assert len(dev.start_settings) == 1
        assert dev.cancelled

    def test_cancel_after_second_read_skips_registration_and_returns_nothing(self) -> None:
        # Proxy is unrelated to the reference: if registration ran, it would
        # fail loudly with an alignment error. Seeing "Scan cancelled" instead
        # proves the post-read cancel check fires before any CV work.
        rng = np.random.default_rng(202)
        h, w = 24, 20
        multi_rgb = rng.integers(0, 65535, size=(h, w, 3), dtype=np.uint16)
        proxy = np.random.default_rng(203).integers(0, 65535, size=(h, w, 3), dtype=np.uint16)
        ir = rng.integers(0, 65535, size=(h, w), dtype=np.uint16)
        dev = SplitCaptureFakeSaneDev(multi_rgb, np.dstack((proxy, ir)))
        backend = _make_backend(dev)
        cancel = threading.Event()

        def cancel_during_ir_read(fraction: float) -> None:
            if 0.75 < fraction < 1.0:
                cancel.set()

        with pytest.raises(RuntimeError, match="Scan cancelled"):
            backend.scan(self.DEVICE, self._params(), cancel_during_ir_read, cancel)

        assert len(dev.start_settings) == 2
        assert dev.cancelled

    def test_negative_shift_alignment_and_crop(self) -> None:
        rng = np.random.default_rng(204)
        height, width = 120, 160
        dx, dy = -2, -8
        base = rng.integers(2000, 62000, size=(height, width), dtype=np.uint16)
        multi_rgb = np.repeat(base[:, :, None], 3, axis=2)
        target_ir = rng.integers(1000, 64000, size=(height, width), dtype=np.uint16)
        dev = SplitCaptureFakeSaneDev(
            multi_rgb,
            np.dstack((_shifted(multi_rgb, dx, dy), _shifted(target_ir, dx, dy))),
        )
        backend = _make_backend(dev)

        result = backend.scan(self.DEVICE, self._params(dpi=4000, frame=3), None, threading.Event())

        assert result.rgb.shape == (height + dy, width + dx, 3)
        assert result.ir is not None and result.ir.shape == (height + dy, width + dx)
        assert np.array_equal(result.rgb, multi_rgb[-dy:, -dx:])
        error = np.abs(result.ir.astype(np.int32) - target_ir[-dy:, -dx:].astype(np.int32))
        assert float(np.percentile(error, 99)) <= 2.0
        alignment = result.ir_alignment
        assert alignment is not None and alignment.mode == "phase-ecc"
        assert abs(alignment.dx_px - dx) <= 0.5 and abs(alignment.dy_px - dy) <= 0.5


class TestAlignSplitIr:
    """Direct geometry/confidence gates of _align_split_ir — all fail-closed."""

    def _trio(self, h: int = 64, w: int = 96, seed: int = 51) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        reference = rng.integers(0, 65535, size=(h, w, 3), dtype=np.uint16)
        ir = rng.integers(0, 65535, size=(h, w), dtype=np.uint16)
        return reference, reference.copy(), ir

    def test_rejects_unequal_widths(self) -> None:
        reference, proxy, ir = self._trio()
        with pytest.raises(RuntimeError, match="different widths"):
            _align_split_ir(reference, proxy[:, :-1], ir[:, :-1])

    def test_rejects_height_loss_beyond_known_tail_row(self) -> None:
        reference, proxy, ir = self._trio()
        with pytest.raises(RuntimeError, match="irreconcilable heights"):
            _align_split_ir(reference, proxy[:-2], ir[:-2])

    def test_rejects_proxy_taller_than_reference(self) -> None:
        reference, proxy, ir = self._trio()
        with pytest.raises(RuntimeError, match="irreconcilable heights"):
            _align_split_ir(reference[:-1], proxy, ir)

    def test_accepts_the_known_one_row_tail_loss(self) -> None:
        reference, proxy, ir = self._trio()
        rgb, aligned, alignment = _align_split_ir(reference, proxy[:-1], ir[:-1])
        assert rgb.shape == (reference.shape[0] - 1, reference.shape[1], 3)
        assert aligned.shape == rgb.shape[:2]
        assert alignment.mode == "identity"

    def test_low_texture_fails_closed(self) -> None:
        h, w = 64, 96
        reference = np.full((h, w, 3), 30000, dtype=np.uint16)
        proxy = reference.copy()
        proxy[0, 0, 0] += 1  # defeat the byte-identical fast path
        ir = np.full((h, w), 15000, dtype=np.uint16)
        with pytest.raises(RuntimeError, match="texture"):
            _align_split_ir(reference, proxy, ir)

    def test_byte_identical_uniform_pair_fails_closed(self) -> None:
        h, w = 64, 96
        reference = np.full((h, w, 3), 30000, dtype=np.uint16)
        ir = np.full((h, w), 15000, dtype=np.uint16)

        with pytest.raises(RuntimeError, match="texture"):
            _align_split_ir(reference, reference.copy(), ir)

    def test_unrelated_images_fail_closed(self) -> None:
        h, w = 96, 128
        reference = np.random.default_rng(52).integers(0, 65535, size=(h, w, 3), dtype=np.uint16)
        proxy = np.random.default_rng(53).integers(0, 65535, size=(h, w, 3), dtype=np.uint16)
        ir = np.random.default_rng(54).integers(0, 65535, size=(h, w), dtype=np.uint16)
        with pytest.raises(RuntimeError, match="Could not safely align"):
            _align_split_ir(reference, proxy, ir)

    def test_rejects_corroborated_horizontal_displacement_over_two_pixels(self) -> None:
        reference, _proxy, ir = self._trio(h=120, w=160, seed=8154)
        proxy = _shifted(reference, 3, 0)

        with pytest.raises(RuntimeError, match="horizontal.*2.0 px"):
            _align_split_ir(reference, proxy, ir)

    def test_rejects_corroborated_vertical_displacement_over_eight_pixels(self) -> None:
        reference, _proxy, ir = self._trio(h=120, w=160, seed=8155)
        proxy = _shifted(reference, 0, 9)

        with pytest.raises(RuntimeError, match="vertical.*8.0 px"):
            _align_split_ir(reference, proxy, ir)

    def test_full_frame_phase_inputs_are_preserved_for_ecc_at_optimal_dft_size(self, monkeypatch) -> None:
        rng = np.random.default_rng(8153)
        height, width = 512, 768
        assert cv2.getOptimalDFTSize(height) == height
        assert cv2.getOptimalDFTSize(width) == width
        expected_dx, expected_dy = 1.25, -0.75
        reference = rng.integers(1000, 64000, size=(height, width, 3), dtype=np.uint16)
        proxy = cv2.warpAffine(
            reference,
            np.asarray([[1.0, 0.0, expected_dx], [0.0, 1.0, expected_dy]], dtype=np.float32),
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

        real_phase_correlate = cv2.phaseCorrelate
        real_find_transform_ecc = cv2.findTransformECC
        phase_sources: list[tuple[np.ndarray, np.ndarray]] = []
        phase_shifts: list[tuple[float, float]] = []

        def observe_real_phase(reference_plane, moving_plane, window):
            reference_before = reference_plane.copy()
            moving_before = moving_plane.copy()
            shift, response = real_phase_correlate(reference_plane, moving_plane, window)
            if reference_plane.shape == (height, width):
                phase_sources.append((reference_before, moving_before))
                phase_shifts.append((float(shift[0]), float(shift[1])))
            return shift, response

        def assert_preserved_before_real_ecc(template, moving, warp, *args):
            assert len(phase_sources) == 3
            phase_dx = sum(shift[0] for shift in phase_shifts) / 3.0
            phase_dy = sum(shift[1] for shift in phase_shifts) / 3.0
            seed_dx, seed_dy = int(round(phase_dx)), int(round(phase_dy))
            ref_top, ref_bottom = max(0, -seed_dy), height + min(0, -seed_dy)
            ref_left, ref_right = max(0, -seed_dx), width + min(0, -seed_dx)
            expected_reference = (phase_sources[0][0] + phase_sources[1][0] + phase_sources[2][0]) / 3.0
            expected_moving = (phase_sources[0][1] + phase_sources[1][1] + phase_sources[2][1]) / 3.0
            np.testing.assert_array_equal(
                template,
                expected_reference[ref_top:ref_bottom, ref_left:ref_right],
            )
            np.testing.assert_array_equal(
                moving,
                expected_moving[
                    ref_top + seed_dy : ref_bottom + seed_dy,
                    ref_left + seed_dx : ref_right + seed_dx,
                ],
            )
            return real_find_transform_ecc(template, moving, warp, *args)

        monkeypatch.setattr(cv2, "phaseCorrelate", observe_real_phase)
        monkeypatch.setattr(cv2, "findTransformECC", assert_preserved_before_real_ecc)

        dx, dy, metrics = _estimate_split_shift(reference, proxy)

        assert metrics["mode"] == "phase-ecc"
        assert dx == pytest.approx(expected_dx, abs=0.05)
        assert dy == pytest.approx(expected_dy, abs=0.05)

    def test_corroborating_ecc_drift_beyond_subpixel_yields_the_phase_estimate(self, monkeypatch) -> None:
        # ECC that agrees within the divergence gate but disagrees by more
        # than a sub-pixel refinement must not relocate the estimate: its
        # optimum tracks photometric agreement and slid whole pixels on the
        # 2026-07-12 live captures while three-channel phase held the truth.
        rng = np.random.default_rng(8157)
        height, width = 512, 768
        reference = rng.integers(1000, 64000, size=(height, width, 3), dtype=np.uint16)
        proxy = reference.copy()
        drift_px = 1.5

        def drifted_ecc(template, moving, warp, *args):
            drifted = warp.copy()
            drifted[0, 2] += np.float32(drift_px)
            return 0.93, drifted

        monkeypatch.setattr(cv2, "findTransformECC", drifted_ecc)

        dx, dy, metrics = _estimate_split_shift(reference, proxy)

        assert metrics["mode"] == "phase-ecc"
        assert metrics["ecc_coefficient"] == pytest.approx(0.93)
        assert dx == pytest.approx(0.0, abs=0.1)
        assert dy == pytest.approx(0.0, abs=0.1)

    def test_subpixel_corroborating_ecc_still_refines_the_phase_estimate(self, monkeypatch) -> None:
        rng = np.random.default_rng(8158)
        height, width = 512, 768
        reference = rng.integers(1000, 64000, size=(height, width, 3), dtype=np.uint16)
        proxy = reference.copy()
        refinement_px = 0.3

        def refined_ecc(template, moving, warp, *args):
            refined = warp.copy()
            refined[0, 2] += np.float32(refinement_px)
            return 0.97, refined

        monkeypatch.setattr(cv2, "findTransformECC", refined_ecc)

        dx, dy, metrics = _estimate_split_shift(reference, proxy)

        assert metrics["mode"] == "phase-ecc"
        assert dx == pytest.approx(refinement_px, abs=0.1)
        assert dy == pytest.approx(0.0, abs=0.1)

    def test_real_periodic_short_wide_capture_uses_cross_scale_global_consensus(self) -> None:
        rng = np.random.default_rng(8156)
        height, width, period = 256, 1024, 64
        y, x = np.mgrid[:height, :width]
        periodic = 26000.0 + 9000.0 * np.sin(2.0 * np.pi * x / period)
        periodic += 4000.0 * np.sin(4.0 * np.pi * x / period)
        sprockets = np.zeros((height, width), dtype=np.float32)
        for left in range(0, width, period):
            sprockets[10:55, left + 8 : left + 40] += 16000.0
            sprockets[height - 55 : height - 10, left + 8 : left + 40] += 16000.0
        unique_texture = cv2.GaussianBlur(
            rng.normal(0.0, 2500.0, size=(height, width)).astype(np.float32),
            (0, 0),
            1.0,
        )
        plane = np.clip(periodic + sprockets + unique_texture, 1000.0, 64000.0)
        reference = np.stack(
            (
                plane,
                np.clip(plane * 0.97 + 700.0, 0.0, 65535.0),
                np.clip(plane * 1.02 - 500.0, 0.0, 65535.0),
            ),
            axis=2,
        ).astype(np.uint16)
        expected_dx, expected_dy = 0.20, -0.10
        proxy = cv2.warpAffine(
            reference,
            np.asarray([[1.0, 0.0, expected_dx], [0.0, 1.0, expected_dy]], dtype=np.float32),
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )
        proxy = np.clip(proxy.astype(np.float32) * 0.92 + 1200.0, 0.0, 65535.0).astype(np.uint16)

        dx, dy, metrics = _estimate_split_shift(reference, proxy)

        assert metrics["mode"] == "multiscale-global"
        assert dx == pytest.approx(expected_dx, abs=0.10)
        assert dy == pytest.approx(expected_dy, abs=0.10)
        assert metrics["estimator_version"] == 2
        assert metrics["multiscale_max_dimensions"] == (512, 1024)
        assert len(metrics["multiscale_channel_shifts_px"]) == 2
        assert len(metrics["multiscale_responses"]) == 2

    def test_periodic_global_alias_requires_stable_tiled_consensus_at_two_scales(self, monkeypatch) -> None:
        reference, _proxy, _ir = self._trio(h=256, w=1024, seed=8157)
        proxy = np.clip(reference.astype(np.float32) * 0.92 + 1200.0, 0.0, 65535.0).astype(np.uint16)

        def periodic_phase(image, _other, _window):
            if image.shape == (128, 512):
                return (-64.0, 0.0), 0.65
            if image.shape == (256, 1024):
                return (-128.0, 0.0), 0.70
            if image.shape == (64, 128):
                return (0.10, -0.05), 0.60
            if image.shape == (128, 256):
                return (0.20, -0.10), 0.62
            raise AssertionError(f"unexpected phase-correlation shape {image.shape}")

        monkeypatch.setattr(cv2, "phaseCorrelate", periodic_phase)
        monkeypatch.setattr(
            cv2,
            "findTransformECC",
            lambda *_args: pytest.fail("short-wide periodic aliases must not reach ECC"),
        )

        dx, dy, metrics = _estimate_split_shift(reference, proxy)

        assert metrics["mode"] == "multiscale-tiled"
        assert dx == pytest.approx(0.20, abs=0.01)
        assert dy == pytest.approx(-0.10, abs=0.01)
        assert metrics["multiscale_max_dimensions"] == (512, 1024)
        assert metrics["multiscale_tile_support_counts"] == ((8, 8, 8), (8, 8, 8))
        assert len(metrics["multiscale_global_alias_shifts_px"]) == 2

    @pytest.mark.xfail(
        strict=True,
        reason="an exactly repeated texture translated by exactly one period is mathematically indistinguishable",
    )
    def test_exact_period_translation_is_an_explicit_unobservable_limit(self) -> None:
        rng = np.random.default_rng(8158)
        height, width, period = 256, 1024, 64
        cell = cv2.GaussianBlur(
            rng.normal(0.0, 1.0, size=(height, period)).astype(np.float32),
            (0, 0),
            1.0,
        )
        plane = np.tile(cell, (1, width // period))
        plane = np.asarray(2000.0 + 60000.0 * (plane - plane.min()) / np.ptp(plane), dtype=np.uint16)
        reference = np.repeat(plane[:, :, None], 3, axis=2)
        proxy = np.roll(reference, period, axis=1)

        with pytest.raises(RuntimeError, match="ambiguous periodic translation"):
            _estimate_split_shift(reference, proxy)

    def test_strong_near_zero_phase_can_reject_a_false_ecc_local_optimum(self, monkeypatch) -> None:
        reference, _proxy, ir = self._trio(h=160, w=200, seed=8150)
        # Preserve geometry while changing the capture's tone enough to defeat
        # the byte-identical fast path, as RGB4x versus RGB1x does in hardware.
        proxy = np.clip(reference.astype(np.float32) * 0.92 + 1200.0, 0, 65535).astype(np.uint16)

        def divergent_ecc(_template, _input, warp, *_args):
            wrong = warp.copy()
            wrong[0, 2] = -2.1
            wrong[1, 2] = 0.6
            return 0.90, wrong

        monkeypatch.setattr(cv2, "findTransformECC", divergent_ecc)

        _rgb, _aligned, alignment = _align_split_ir(reference, proxy, ir)

        assert alignment.mode == "phase-only"
        assert max(abs(alignment.dx_px), abs(alignment.dy_px)) <= 2.0
        assert min(alignment.phase_responses) >= 0.50
        assert alignment.channel_spread_px is not None and alignment.channel_spread_px <= 0.10

    def test_ecc_divergence_is_measured_in_native_output_pixels(self, monkeypatch) -> None:
        reference, _proxy, ir = self._trio(h=2048, w=512, seed=8159)
        proxy = np.clip(reference.astype(np.float32) * 0.92 + 1200.0, 0, 65535).astype(np.uint16)

        monkeypatch.setattr(cv2, "phaseCorrelate", lambda *_args: ((0.0, 0.0), 0.55))

        def false_ecc_local_optimum(_template, _input, warp, *_args):
            wrong = warp.copy()
            # This is only 1.5 px from phase at the 1024 px estimation scale,
            # but 3 px on the native output where the alignment is applied.
            wrong[0, 2] = -1.5
            wrong[1, 2] = 0.25
            return 0.90, wrong

        monkeypatch.setattr(cv2, "findTransformECC", false_ecc_local_optimum)

        _rgb, _aligned, alignment = _align_split_ir(reference, proxy, ir)

        assert alignment.mode == "phase-only"
        assert alignment.dx_px == 0.0
        assert alignment.dy_px == 0.0

    def test_phase_only_fallback_still_rejects_weak_phase_evidence(self, monkeypatch) -> None:
        reference, _proxy, ir = self._trio(h=160, w=200, seed=8151)
        proxy = np.clip(reference.astype(np.float32) * 0.92 + 1200.0, 0, 65535).astype(np.uint16)

        monkeypatch.setattr(cv2, "phaseCorrelate", lambda *_args: ((0.0, 0.0), 0.49))

        def divergent_ecc(_template, _input, warp, *_args):
            wrong = warp.copy()
            wrong[0, 2] = -2.1
            return 0.90, wrong

        monkeypatch.setattr(cv2, "findTransformECC", divergent_ecc)

        with pytest.raises(RuntimeError, match="diverges from the phase estimate"):
            _align_split_ir(reference, proxy, ir)

    def test_multiscale_tiled_consensus_rejects_a_periodic_full_width_alias(self, monkeypatch) -> None:
        reference, _proxy, ir = self._trio(h=96, w=512, seed=8152)
        proxy = np.clip(reference.astype(np.float32) * 0.92 + 1200.0, 0, 65535).astype(np.uint16)

        def phase(image, _other, _window):
            if image.shape == (48, 256):
                return (-37.5, 0.0), 0.35
            if image.shape == (96, 512):
                return (-75.0, 0.0), 0.45
            if image.shape == (24, 64):
                return (0.025, -0.015), 0.45
            if image.shape == (48, 128):
                return (0.05, -0.03), 0.45
            raise AssertionError(f"unexpected phase-correlation shape {image.shape}")

        monkeypatch.setattr(cv2, "phaseCorrelate", phase)
        monkeypatch.setattr(
            cv2,
            "findTransformECC",
            lambda *_args: pytest.fail("confident tiled phase must bypass periodic ECC"),
        )

        _rgb, _aligned, alignment = _align_split_ir(reference, proxy, ir)

        assert alignment.mode == "multiscale-tiled"
        assert alignment.estimator_version == 2
        assert alignment.multiscale_max_dimensions == (256, 512)
        assert len(alignment.multiscale_channel_shifts_px) == 2
        assert alignment.multiscale_tile_support_counts == ((8, 8, 8), (8, 8, 8))
        assert len(alignment.multiscale_global_alias_shifts_px) == 2
        assert alignment.tile_support_counts == (8, 8, 8)
        assert alignment.tile_shift_spread_px is not None and alignment.tile_shift_spread_px <= 0.35
        assert max(abs(alignment.dx_px), abs(alignment.dy_px)) <= 2.0


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

    def test_frame_without_option_raises(self) -> None:
        import pytest

        rng = np.random.default_rng(20)
        opt = {k: v for k, v in COOLSCAN3_OPT.items() if k != "frame"}
        dev = FakeSaneDev(rng.integers(0, 65535, size=(6, 5, 4), dtype=np.uint16), opt_map=opt)
        with pytest.raises(RuntimeError, match="frame-selection option"):
            self._scan(ScanParams(dpi=1000, depth=16, capture_ir=True, frame=5), dev)
        # The missing-option check must fire before any attempt to set dev.frame.
        assert "frame" not in dev.recorded


class TestRegisteredGeometry:
    """Registered positioning is applied as one unit before AF, AE, and scan."""

    def test_sets_coupled_geometry_and_auto_exposure_before_start(self) -> None:
        rng = np.random.default_rng(25)
        dev = FakeSaneDev(rng.integers(0, 65535, size=(6, 5, 4), dtype=np.uint16))
        backend = _make_backend(dev)
        geometry = RegisteredScanGeometry(frame=3, subframe_mm=6.35, br_y_device_px=5003)

        backend.scan(
            "coolscan3:usb:libusb:001:007",
            ScanParams(
                dpi=4000,
                depth=16,
                capture_ir=True,
                registered_geometry=geometry,
                auto_exposure=True,
            ),
            None,
            threading.Event(),
        )

        assert dev.recorded["frame"] == 3
        assert dev.recorded["subframe"] == 6.35
        assert dev.recorded["br_y"] == 5003
        assert dev.recorded["ae"] is True
        ordered = [event[1] if event[0] == "set" else "start" for event in dev.events]
        for positioning_option in ("frame", "subframe", "br_y"):
            assert ordered.index(positioning_option) < ordered.index("autofocus")
            assert ordered.index(positioning_option) < ordered.index("ae")
            assert ordered.index(positioning_option) < ordered.index("start")

    def test_readback_disagreement_refuses_before_autofocus_or_start(self) -> None:
        rng = np.random.default_rng(250)

        class QuantizingDevice(FakeSaneDev):
            def __getattr__(self, name: str) -> Any:
                value = super().__getattr__(name)
                return value - 1 if name == "br_y" else value

        dev = QuantizingDevice(rng.integers(0, 65535, size=(6, 5, 3), dtype=np.uint16))
        backend = _make_backend(dev)
        geometry = RegisteredScanGeometry(
            frame=3,
            subframe_mm=6.35,
            br_y_device_px=5003,
        )

        with pytest.raises(RuntimeError, match="readback disagrees.*br_y"):
            backend.scan(
                "coolscan3:usb:libusb:001:007",
                ScanParams(
                    dpi=4000,
                    depth=16,
                    capture_ir=False,
                    registered_geometry=geometry,
                ),
                None,
                threading.Event(),
            )

        assert not any(event[0] == "set" and event[1] in {"autofocus", "ae"} for event in dev.events)
        assert ("start",) not in dev.events

    def test_ls5000_bw_recipe_requires_exact_full_frame_metadata_and_payload(
        self,
    ) -> None:
        row_values = np.arange(5_959 * 3, dtype=np.uint16).reshape(5_959, 1, 3)
        rgb = np.broadcast_to(row_values, (5_959, 3_946, 3))
        dev = ParameterOverrideFakeSaneDev(
            rgb,
            ("color", 1, (3_946, 5_959), 16, 23_676),
        )
        backend = _make_backend(dev)

        result = backend.scan(
            "coolscan3:usb:libusb:001:007",
            ScanParams(
                dpi=4_000,
                depth=16,
                capture_ir=False,
                samples_per_scan=4,
                registered_geometry=RegisteredScanGeometry(
                    frame=2,
                    subframe_mm=0.25 * (25.4 / 4_000),
                    br_y_device_px=5_958,
                ),
                auto_exposure=True,
            ),
            None,
            threading.Event(),
        )

        assert result.rgb.shape == (5_959, 3_946, 3)
        assert result.rgb.dtype == np.uint16
        assert result.ir is None
        assert dev.recorded["resolution"] == 4_000
        assert dev.recorded["depth"] == 16
        assert dev.recorded["samples_per_scan"] == 4
        assert dev.recorded["infrared"] is False
        assert dev.recorded["autofocus"] is True
        assert dev.recorded["ae"] is True
        option_order = [event[1] for event in dev.events if event[0] == "set"]
        assert option_order.index("infrared") < option_order.index("autofocus")
        assert option_order.index("infrared") < option_order.index("ae")

    @pytest.mark.parametrize(
        ("option_name", "bad_value"),
        (
            ("resolution", 2_000),
            ("depth", 8),
            ("samples_per_scan", 1),
            ("infrared", True),
            ("autofocus", False),
            ("ae", False),
        ),
    )
    def test_ls5000_bw_recipe_refuses_silently_ignored_options_before_start(
        self,
        option_name: str,
        bad_value: Any,
    ) -> None:
        class IgnoredOptionDevice(ParameterOverrideFakeSaneDev):
            def __getattr__(self, name: str) -> Any:
                if name == option_name:
                    return bad_value
                return super().__getattr__(name)

        dev = IgnoredOptionDevice(
            np.zeros((6, 5, 3), dtype=np.uint16),
            ("color", 1, (3_946, 5_959), 16, 23_676),
        )
        backend = _make_backend(dev)

        with pytest.raises(RuntimeError, match=rf"readback disagrees for {option_name}"):
            backend.scan(
                "coolscan3:usb:libusb:001:007",
                ScanParams(
                    dpi=4_000,
                    depth=16,
                    capture_ir=False,
                    samples_per_scan=4,
                    registered_geometry=RegisteredScanGeometry(
                        frame=2,
                        subframe_mm=0.25 * (25.4 / 4_000),
                        br_y_device_px=5_958,
                    ),
                    auto_exposure=True,
                ),
                None,
                threading.Event(),
            )

        assert ("start",) not in dev.events

    @pytest.mark.parametrize(
        "disabled_parameter",
        ("autofocus", "auto_exposure"),
    )
    def test_ls5000_bw_recipe_requires_focus_and_metering_parameters(
        self,
        disabled_parameter: str,
    ) -> None:
        dev = ParameterOverrideFakeSaneDev(
            np.zeros((6, 5, 3), dtype=np.uint16),
            ("color", 1, (3_946, 5_959), 16, 23_676),
        )
        backend = _make_backend(dev)
        overrides = {
            "autofocus": True,
            "auto_exposure": True,
        }
        overrides[disabled_parameter] = False

        with pytest.raises(
            RuntimeError,
            match="requires autofocus=True and auto_exposure=True",
        ):
            backend.scan(
                "coolscan3:usb:libusb:001:007",
                ScanParams(
                    dpi=4_000,
                    depth=16,
                    capture_ir=False,
                    samples_per_scan=4,
                    registered_geometry=RegisteredScanGeometry(
                        frame=2,
                        subframe_mm=0.25 * (25.4 / 4_000),
                        br_y_device_px=5_958,
                    ),
                    **overrides,
                ),
                None,
                threading.Event(),
            )

        assert ("start",) not in dev.events

    def test_ls5000_bw_recipe_rechecks_geometry_after_auto_exposure(self) -> None:
        class AutoExposureMovesWindow(ParameterOverrideFakeSaneDev):
            def __setattr__(self, name: str, value: Any) -> None:
                super().__setattr__(name, value)
                if name == "ae":
                    self.recorded["br_y"] = 5_957

        dev = AutoExposureMovesWindow(
            np.zeros((6, 5, 3), dtype=np.uint16),
            ("color", 1, (3_946, 5_959), 16, 23_676),
        )
        backend = _make_backend(dev)

        with pytest.raises(RuntimeError, match="readback disagrees.*br_y"):
            backend.scan(
                "coolscan3:usb:libusb:001:007",
                ScanParams(
                    dpi=4_000,
                    depth=16,
                    capture_ir=False,
                    samples_per_scan=4,
                    registered_geometry=RegisteredScanGeometry(
                        frame=2,
                        subframe_mm=0.25 * (25.4 / 4_000),
                        br_y_device_px=5_958,
                    ),
                    auto_exposure=True,
                ),
                None,
                threading.Event(),
            )

        assert ("start",) not in dev.events

    def test_ls5000_bw_recipe_requires_writable_infrared_option_before_positioning(
        self,
    ) -> None:
        opt = {name: value for name, value in COOLSCAN3_OPT.items() if name != "infrared"}
        dev = ParameterOverrideFakeSaneDev(
            np.zeros((6, 5, 3), dtype=np.uint16),
            ("color", 1, (3_946, 5_959), 16, 23_676),
            opt_map=opt,
        )
        backend = _make_backend(dev)

        with pytest.raises(RuntimeError, match="writable infrared control so infrared can be proven off"):
            backend.scan(
                "coolscan3:usb:libusb:001:007",
                ScanParams(
                    dpi=4_000,
                    depth=16,
                    capture_ir=False,
                    samples_per_scan=4,
                    registered_geometry=RegisteredScanGeometry(
                        frame=2,
                        subframe_mm=0.25 * (25.4 / 4_000),
                        br_y_device_px=5_958,
                    ),
                    auto_exposure=True,
                ),
                None,
                threading.Event(),
            )

        assert not any(
            event[0] == "set" and event[1] in {"frame", "subframe", "br_y"}
            for event in dev.events
        )
        assert ("start",) not in dev.events

    def test_ls5000_bw_recipe_refuses_wrong_metadata_before_payload_read(
        self,
    ) -> None:
        rgb = np.zeros((6, 5, 3), dtype=np.uint16)

        class ReadTrackingDevice(ParameterOverrideFakeSaneDev):
            _INTERNAL = ParameterOverrideFakeSaneDev._INTERNAL + ("read_called",)

            def __init__(self) -> None:
                super().__init__(rgb, ("color", 1, (3_945, 5_959), 16, 23_670))
                object.__setattr__(self, "read_called", False)

            def arr_snap(self, progress=None) -> np.ndarray:
                object.__setattr__(self, "read_called", True)
                return super().arr_snap(progress=progress)

        dev = ReadTrackingDevice()
        backend = _make_backend(dev)

        with pytest.raises(RuntimeError, match="expected 3946x5959"):
            backend.scan(
                "coolscan3:usb:libusb:001:007",
                ScanParams(
                    dpi=4_000,
                    depth=16,
                    capture_ir=False,
                    samples_per_scan=4,
                    registered_geometry=RegisteredScanGeometry(
                        frame=2,
                        subframe_mm=0.25 * (25.4 / 4_000),
                        br_y_device_px=5_958,
                    ),
                    auto_exposure=True,
                ),
                None,
                threading.Event(),
            )

        assert dev.read_called is False

    def test_generic_area_cannot_overwrite_a_registered_scan_window(self) -> None:
        rng = np.random.default_rng(28)
        dev = FakeSaneDev(rng.integers(0, 65535, size=(6, 5, 4), dtype=np.uint16))
        backend = _make_backend(dev)
        geometry = RegisteredScanGeometry(frame=3, subframe_mm=6.35, br_y_device_px=5003)

        with pytest.raises(RuntimeError, match="area.*registered geometry"):
            backend.scan(
                "coolscan3:usb:libusb:001:007",
                ScanParams(
                    dpi=4000,
                    depth=16,
                    capture_ir=True,
                    area=(0, 0, 3945, 5958),
                    registered_geometry=geometry,
                ),
                None,
                threading.Event(),
            )

        assert not any(event[0] == "set" and event[1] in {"frame", "subframe", "br_y", "autofocus", "ae"} for event in dev.events)
        assert ("start",) not in dev.events

    @pytest.mark.parametrize("missing_option", ("br_y", "ae"))
    def test_missing_requested_option_fails_before_any_positioning(self, missing_option: str) -> None:
        rng = np.random.default_rng(26)
        opt = {key: value for key, value in COOLSCAN3_OPT.items() if key != missing_option}
        dev = FakeSaneDev(rng.integers(0, 65535, size=(6, 5, 4), dtype=np.uint16), opt_map=opt)
        backend = _make_backend(dev)
        geometry = RegisteredScanGeometry(frame=3, subframe_mm=6.35, br_y_device_px=5003)

        with pytest.raises(RuntimeError, match=rf"{missing_option}.*option"):
            backend.scan(
                "coolscan3:usb:libusb:001:007",
                ScanParams(
                    dpi=4000,
                    depth=16,
                    capture_ir=True,
                    registered_geometry=geometry,
                    auto_exposure=True,
                ),
                None,
                threading.Event(),
            )

        assert not any(event[0] == "set" and event[1] in {"frame", "subframe", "br_y", "autofocus", "ae"} for event in dev.events)
        assert ("start",) not in dev.events

    @pytest.mark.parametrize("inactive_option", ("subframe", "ae"))
    def test_inactive_requested_option_fails_before_any_positioning(self, inactive_option: str) -> None:
        rng = np.random.default_rng(27)
        opt = dict(COOLSCAN3_OPT)
        opt[inactive_option] = FakeOption(active=False)
        dev = FakeSaneDev(rng.integers(0, 65535, size=(6, 5, 4), dtype=np.uint16), opt_map=opt)
        backend = _make_backend(dev)
        geometry = RegisteredScanGeometry(frame=3, subframe_mm=6.35, br_y_device_px=5003)

        with pytest.raises(RuntimeError, match=rf"{inactive_option}.*inactive"):
            backend.scan(
                "coolscan3:usb:libusb:001:007",
                ScanParams(
                    dpi=4000,
                    depth=16,
                    capture_ir=True,
                    registered_geometry=geometry,
                    auto_exposure=True,
                ),
                None,
                threading.Event(),
            )

        assert not any(event[0] == "set" and event[1] in {"frame", "subframe", "br_y", "autofocus", "ae"} for event in dev.events)
        assert ("start",) not in dev.events


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
