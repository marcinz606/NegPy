"""Fail-closed tests for the practical-parity scanner runner."""

import argparse
import builtins
import json
import struct
import sys
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pytest
import tifffile

from negpy.infrastructure.scanners import practical_parity_runner as runner
from negpy.infrastructure.scanners.result import ScanResult, SplitIrAlignment
from negpy.infrastructure.scanners.sane_backend import (
    _SPLIT_MAX_CHANNEL_SPREAD_PX,
    _SPLIT_MIN_ECC,
    _SPLIT_MIN_PHASE_RESPONSE,
)


@dataclass
class FakeOption:
    constraint: Any = None
    active: bool = True
    settable: bool = True

    def is_active(self) -> bool:
        return self.active

    def is_settable(self) -> bool:
        return self.settable


def _film_loaded_opt() -> dict:
    return {
        "frame": FakeOption(constraint=(1, 6, 1)),
        "frame_count": FakeOption(constraint=(1, 6, 1)),
        "subframe": FakeOption(constraint=(0.0, 37.8333, 0.0)),
        "br_y": FakeOption(constraint=(0, 5958, 1)),
        "autofocus": FakeOption(),
        "ae": FakeOption(),
        "samples_per_scan": FakeOption(constraint=(1, 16, 1)),
        "infrared": FakeOption(),
        "depth": FakeOption(constraint=[8, 16]),
        "resolution": FakeOption(constraint=[4000, 2000, 1000]),
    }


def _no_film_opt() -> dict:
    # The state observed on 2026-07-11: device idle, no selectable frames.
    opt = _film_loaded_opt()
    opt["frame"] = FakeOption(constraint=(1, 0, 1), active=False)
    opt["frame_count"] = FakeOption(constraint=(1, 0, 1), active=False)
    return opt


class FakeDev:
    """Options-only device: any scan attempt is a hard test failure."""

    def __init__(self, opt_map: dict) -> None:
        self.opt = opt_map
        self.closed = False
        self.started = False

    def start(self) -> None:
        self.started = True
        raise AssertionError("runner attempted to start a scan")

    def close(self) -> None:
        self.closed = True


class FakeSaneModule:
    def __init__(self, dev: FakeDev, device_id: str) -> None:
        self._dev = dev
        self._device_id = device_id
        self.opened: list[str] = []

    def init(self) -> None:
        pass

    def get_devices(self):
        return [(self._device_id, "Nikon", "LS-5000 ED", "film scanner")]

    def open(self, device_id: str) -> FakeDev:
        self.opened.append(device_id)
        return self._dev


class SaneMustNotInitialize:
    def init(self) -> None:
        raise AssertionError("invalid registration reached sane.init()")


def _write_registration_manifest(tmp_path, frames: list[dict]) -> str:
    path = tmp_path / "registration.json"
    path.write_text(json.dumps({"frames": frames}))
    return str(path)


def _args(**overrides) -> argparse.Namespace:
    base = dict(profile="small", live=False, device=None, out_dir="/tmp/unused", json=None)
    base.update(overrides)
    return argparse.Namespace(**base)


def _textured_rgb_with_repeated_suffix(*, height: int = 96, width: int = 80, repeated_rows: int = 21) -> np.ndarray:
    rng = np.random.default_rng(20260712)
    rgb = rng.integers(1000, 64000, size=(height, width, 3), dtype=np.uint16)
    anchor = rng.integers(4000, 60000, size=(width, 3), dtype=np.uint16)
    rgb[-repeated_rows:] = anchor
    return rgb


class TestCli:
    def test_live_nondefault_frame_requires_registration_before_sane_import(
        self,
        monkeypatch,
        capsys,
    ) -> None:
        real_import = builtins.__import__
        sane_import_attempted = False

        def guarded_import(name, *args, **kwargs):
            nonlocal sane_import_attempted
            if name == "sane":
                sane_import_attempted = True
                raise AssertionError("unsafe frame selection reached the SANE import")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", guarded_import)

        with pytest.raises(SystemExit) as error:
            runner.main(["--live", "--frame", "7"])

        assert error.value.code == 2
        assert sane_import_attempted is False
        assert "live scans for non-default --frame require --registration-json" in capsys.readouterr().err

    def test_live_default_frame_keeps_profile_geometry_and_reaches_preflight(self, monkeypatch, tmp_path) -> None:
        dev = FakeDev(_no_film_opt())
        module = FakeSaneModule(dev, "net:scanner:coolscan3:usb:libusb:001:017")
        monkeypatch.setitem(sys.modules, "sane", module)

        code = runner.main(["--live", "--out-dir", str(tmp_path)])

        assert code == 2
        [report_path] = list(tmp_path.glob("*.json"))
        with report_path.open() as stream:
            report = json.load(stream)
        assert report["profile"]["frame"] == 3
        assert report["profile"]["subframe_mm"] == 6.35
        assert report["profile"]["br_y"] == 800
        assert report["verdict"] == "preconditions-failed"
        assert dev.started is False

    def test_registration_manifest_overrides_both_geometry_fields(self, monkeypatch, tmp_path) -> None:
        registration_path = _write_registration_manifest(
            tmp_path,
            [{"frame": 7, "subframe_mm": 5.84, "br_y": 5134}],
        )
        opt = _film_loaded_opt()
        opt["frame"] = FakeOption(constraint=(1, 24, 1))
        opt["frame_count"] = FakeOption(constraint=(1, 24, 1))
        module = FakeSaneModule(
            FakeDev(opt),
            "net:10.0.0.100:coolscan3:usb:libusb:001:017",
        )
        monkeypatch.setitem(sys.modules, "sane", module)

        code = runner.main(
            [
                "--profile",
                "small",
                "--frame",
                "7",
                "--registration-json",
                registration_path,
                "--out-dir",
                str(tmp_path / "reports"),
            ]
        )

        assert code == 0
        [report_path] = list((tmp_path / "reports").glob("*.json"))
        with report_path.open() as stream:
            report = json.load(stream)
        assert report["profile"]["dpi"] == 1000
        assert report["profile"]["frame"] == 7
        assert report["profile"]["subframe_mm"] == 5.84
        assert report["profile"]["br_y"] == 5134

    def test_registration_manifest_missing_requested_frame_is_a_cli_error_before_sane_init(
        self,
        monkeypatch,
        tmp_path,
        capsys,
    ) -> None:
        registration_path = _write_registration_manifest(
            tmp_path,
            [{"frame": 8, "subframe_mm": 5.84, "br_y": 5134}],
        )
        monkeypatch.setitem(sys.modules, "sane", SaneMustNotInitialize())

        with pytest.raises(SystemExit) as error:
            runner.main(["--frame", "7", "--registration-json", registration_path])

        assert error.value.code == 2
        assert "registration manifest has no entry for frame 7" in capsys.readouterr().err

    def test_registration_manifest_duplicate_requested_frame_is_a_cli_error_before_sane_init(
        self,
        monkeypatch,
        tmp_path,
        capsys,
    ) -> None:
        registration_path = _write_registration_manifest(
            tmp_path,
            [
                {"frame": 7, "subframe_mm": 5.84, "br_y": 5134},
                {"frame": 7, "subframe_mm": 5.59, "br_y": 5023},
            ],
        )
        monkeypatch.setitem(sys.modules, "sane", SaneMustNotInitialize())

        with pytest.raises(SystemExit) as error:
            runner.main(["--frame", "7", "--registration-json", registration_path])

        assert error.value.code == 2
        assert "registration manifest has duplicate entries for frame 7" in capsys.readouterr().err

    def test_registration_manifest_nonpositive_subframe_is_a_cli_error_before_sane_init(
        self,
        monkeypatch,
        tmp_path,
        capsys,
    ) -> None:
        registration_path = _write_registration_manifest(
            tmp_path,
            [{"frame": 7, "subframe_mm": 0.0, "br_y": 5134}],
        )
        monkeypatch.setitem(sys.modules, "sane", SaneMustNotInitialize())

        with pytest.raises(SystemExit) as error:
            runner.main(["--frame", "7", "--registration-json", registration_path])

        assert error.value.code == 2
        assert "frame 7 has invalid subframe_mm; expected a finite number > 0" in capsys.readouterr().err

    def test_registration_manifest_nonfinite_subframe_is_a_cli_error_before_sane_init(
        self,
        monkeypatch,
        tmp_path,
        capsys,
    ) -> None:
        registration_path = _write_registration_manifest(
            tmp_path,
            [{"frame": 7, "subframe_mm": float("inf"), "br_y": 5134}],
        )
        monkeypatch.setitem(sys.modules, "sane", SaneMustNotInitialize())

        with pytest.raises(SystemExit) as error:
            runner.main(["--frame", "7", "--registration-json", registration_path])

        assert error.value.code == 2
        assert "frame 7 has invalid subframe_mm; expected a finite number > 0" in capsys.readouterr().err

    def test_registration_manifest_noninteger_scan_height_is_a_cli_error_before_sane_init(
        self,
        monkeypatch,
        tmp_path,
        capsys,
    ) -> None:
        registration_path = _write_registration_manifest(
            tmp_path,
            [{"frame": 7, "subframe_mm": 5.84, "br_y": 5134.0}],
        )
        monkeypatch.setitem(sys.modules, "sane", SaneMustNotInitialize())

        with pytest.raises(SystemExit) as error:
            runner.main(["--frame", "7", "--registration-json", registration_path])

        assert error.value.code == 2
        assert "frame 7 has invalid br_y; expected an integer in [0, 5958]" in capsys.readouterr().err

    def test_registration_manifest_out_of_range_scan_height_is_a_cli_error_before_sane_init(
        self,
        monkeypatch,
        tmp_path,
        capsys,
    ) -> None:
        registration_path = _write_registration_manifest(
            tmp_path,
            [{"frame": 7, "subframe_mm": 5.84, "br_y": 5959}],
        )
        monkeypatch.setitem(sys.modules, "sane", SaneMustNotInitialize())

        with pytest.raises(SystemExit) as error:
            runner.main(["--frame", "7", "--registration-json", registration_path])

        assert error.value.code == 2
        assert "frame 7 has invalid br_y; expected an integer in [0, 5958]" in capsys.readouterr().err

    def test_registration_manifest_requires_explicit_frame_before_sane_init(
        self,
        monkeypatch,
        tmp_path,
        capsys,
    ) -> None:
        registration_path = _write_registration_manifest(
            tmp_path,
            [{"frame": 3, "subframe_mm": 6.35, "br_y": 5003}],
        )
        monkeypatch.setitem(sys.modules, "sane", SaneMustNotInitialize())

        with pytest.raises(SystemExit) as error:
            runner.main(["--registration-json", registration_path])

        assert error.value.code == 2
        assert "--registration-json requires --frame" in capsys.readouterr().err

    def test_missing_registration_manifest_is_a_cli_error_before_sane_init(
        self,
        monkeypatch,
        tmp_path,
        capsys,
    ) -> None:
        registration_path = str(tmp_path / "missing-registration.json")
        monkeypatch.setitem(sys.modules, "sane", SaneMustNotInitialize())

        with pytest.raises(SystemExit) as error:
            runner.main(["--frame", "7", "--registration-json", registration_path])

        assert error.value.code == 2
        assert f"cannot read registration manifest {registration_path}" in capsys.readouterr().err

    def test_malformed_registration_manifest_is_a_cli_error_before_sane_init(
        self,
        monkeypatch,
        tmp_path,
        capsys,
    ) -> None:
        registration_path = tmp_path / "registration.json"
        registration_path.write_text("not JSON")
        monkeypatch.setitem(sys.modules, "sane", SaneMustNotInitialize())

        with pytest.raises(SystemExit) as error:
            runner.main(["--frame", "7", "--registration-json", str(registration_path)])

        assert error.value.code == 2
        assert f"registration manifest {registration_path} is not valid JSON" in capsys.readouterr().err

    def test_registration_manifest_requires_frames_array_before_sane_init(
        self,
        monkeypatch,
        tmp_path,
        capsys,
    ) -> None:
        registration_path = tmp_path / "registration.json"
        registration_path.write_text("{}")
        monkeypatch.setitem(sys.modules, "sane", SaneMustNotInitialize())

        with pytest.raises(SystemExit) as error:
            runner.main(["--frame", "7", "--registration-json", str(registration_path)])

        assert error.value.code == 2
        assert "registration manifest must contain a 'frames' array" in capsys.readouterr().err

    def test_registration_manifest_frame_record_requires_integer_frame_before_sane_init(
        self,
        monkeypatch,
        tmp_path,
        capsys,
    ) -> None:
        registration_path = _write_registration_manifest(
            tmp_path,
            [{"subframe_mm": 5.84, "br_y": 5134}],
        )
        monkeypatch.setitem(sys.modules, "sane", SaneMustNotInitialize())

        with pytest.raises(SystemExit) as error:
            runner.main(["--frame", "7", "--registration-json", registration_path])

        assert error.value.code == 2
        assert "registration manifest frame entry 1 must contain an integer 'frame'" in capsys.readouterr().err

    def test_registration_manifest_rejects_duplicate_unrequested_frame_before_sane_init(
        self,
        monkeypatch,
        tmp_path,
        capsys,
    ) -> None:
        registration_path = _write_registration_manifest(
            tmp_path,
            [
                {"frame": 7, "subframe_mm": 5.84, "br_y": 5134},
                {"frame": 8, "subframe_mm": 5.59, "br_y": 5023},
                {"frame": 8, "subframe_mm": 5.40, "br_y": 5028},
            ],
        )
        monkeypatch.setitem(sys.modules, "sane", SaneMustNotInitialize())

        with pytest.raises(SystemExit) as error:
            runner.main(["--frame", "7", "--registration-json", registration_path])

        assert error.value.code == 2
        assert "registration manifest has duplicate entries for frame 8" in capsys.readouterr().err

    def test_registration_manifest_selected_frame_requires_geometry_fields_before_sane_init(
        self,
        monkeypatch,
        tmp_path,
        capsys,
    ) -> None:
        registration_path = _write_registration_manifest(
            tmp_path,
            [{"frame": 7, "br_y": 5134}],
        )
        monkeypatch.setitem(sys.modules, "sane", SaneMustNotInitialize())

        with pytest.raises(SystemExit) as error:
            runner.main(["--frame", "7", "--registration-json", registration_path])

        assert error.value.code == 2
        assert "registration entry for frame 7 must contain subframe_mm and br_y" in capsys.readouterr().err

    @pytest.mark.parametrize(
        ("frame_args", "expected_frame"),
        [
            (["--frame", "7"], 7),
            (["--frame", "14"], 14),
            (["--frame", "16"], 16),
            ([], 3),
        ],
    )
    def test_frame_override_flows_to_profile_and_output_stem(
        self,
        monkeypatch,
        tmp_path,
        frame_args: list[str],
        expected_frame: int,
    ) -> None:
        opt = _film_loaded_opt()
        opt["frame"] = FakeOption(constraint=(1, 24, 1))
        opt["frame_count"] = FakeOption(constraint=(1, 24, 1))
        module = FakeSaneModule(
            FakeDev(opt),
            "net:10.0.0.100:coolscan3:usb:libusb:001:017",
        )
        monkeypatch.setitem(sys.modules, "sane", module)

        code = runner.main(
            [
                "--profile",
                "small",
                *frame_args,
                "--out-dir",
                str(tmp_path),
            ]
        )

        assert code == 0
        [report_path] = list(tmp_path.glob("*.json"))
        with report_path.open() as stream:
            report = json.load(stream)
        assert report["profile"]["frame"] == expected_frame
        assert report["profile"]["subframe_mm"] == 6.35
        assert report["profile"]["br_y"] == 800
        assert f"-frame{expected_frame:02d}-" in report_path.name


class TestDeviceDiscovery:
    def test_discovers_coolscan_over_net(self) -> None:
        module = FakeSaneModule(FakeDev(_film_loaded_opt()), "net:10.0.0.100:coolscan3:usb:libusb:001:017")
        assert runner.discover_coolscan_device(module) == "net:10.0.0.100:coolscan3:usb:libusb:001:017"

    def test_no_device_is_an_error(self) -> None:
        module = FakeSaneModule(FakeDev(_film_loaded_opt()), "plustek:libusb:001:003")
        with pytest.raises(RuntimeError, match="no coolscan3 device"):
            runner.discover_coolscan_device(module)

    def test_run_reports_zero_discovered_coolscans_as_failed_preconditions(self) -> None:
        class NoDevicesModule:
            def get_devices(self):
                return []

            def open(self, device_id: str):
                raise AssertionError(f"runner unexpectedly opened {device_id}")

        report, code = runner.run(
            NoDevicesModule(),
            _args(),
            runner.PROFILES["small"],
        )

        assert code == 2
        assert report["verdict"] == "preconditions-failed"
        assert report["option_snapshot"] == {}
        assert report["precondition_failures"] == ["device discovery failed: RuntimeError: no coolscan3 device found (saw: none)"]

    def test_run_reports_multiple_discovered_coolscans_as_failed_preconditions(self) -> None:
        device_ids = [
            "coolscan3:usb:libusb:001:017",
            "net:scanner:coolscan3:usb:libusb:001:018",
        ]

        class MultipleDevicesModule:
            def get_devices(self):
                return [(device_id, "Nikon", "LS-5000 ED", "film scanner") for device_id in device_ids]

            def open(self, device_id: str):
                raise AssertionError(f"runner unexpectedly opened {device_id}")

        report, code = runner.run(
            MultipleDevicesModule(),
            _args(),
            runner.PROFILES["small"],
        )

        assert code == 2
        assert report["verdict"] == "preconditions-failed"
        assert report["option_snapshot"] == {}
        assert report["precondition_failures"] == [
            f"device discovery failed: RuntimeError: multiple coolscan3 devices found, pass --device explicitly: {device_ids}"
        ]

    def test_explicit_non_coolscan_device_is_refused_before_open(self) -> None:
        module = FakeSaneModule(FakeDev(_film_loaded_opt()), "plustek:libusb:001:003")

        report, code = runner.run(
            module,
            _args(device="plustek:libusb:001:003"),
            runner.PROFILES["small"],
        )

        assert code == 2
        assert report["verdict"] == "preconditions-failed"
        assert report["precondition_failures"] == ["explicit device is not a coolscan3 backend: plustek:libusb:001:003"]
        assert module.opened == []


class TestPreconditions:
    def test_film_loaded_state_passes(self) -> None:
        snapshot = runner.snapshot_options(FakeDev(_film_loaded_opt()))
        assert runner.evaluate_preconditions(snapshot, runner.PROFILES["small"]) == []

    def test_inactive_zero_frame_constraint_refuses(self) -> None:
        snapshot = runner.snapshot_options(FakeDev(_no_film_opt()))
        failures = runner.evaluate_preconditions(snapshot, runner.PROFILES["small"])
        assert any("'frame' is inactive" in failure for failure in failures)
        assert any("'frame_count' is inactive" in failure for failure in failures)

    def test_active_but_zero_max_frame_refuses(self) -> None:
        opt = _film_loaded_opt()
        opt["frame"] = FakeOption(constraint=(1, 0, 1), active=True)
        snapshot = runner.snapshot_options(FakeDev(opt))
        failures = runner.evaluate_preconditions(snapshot, runner.PROFILES["small"])
        assert any("no selectable frame" in failure for failure in failures)

    def test_frame_max_below_target_refuses(self) -> None:
        opt = _film_loaded_opt()
        opt["frame"] = FakeOption(constraint=(1, 2, 1), active=True)
        snapshot = runner.snapshot_options(FakeDev(opt))
        failures = runner.evaluate_preconditions(snapshot, runner.PROFILES["small"])
        assert any("no selectable frame >= 3" in failure for failure in failures)

    def test_frame_range_that_starts_after_requested_frame_refuses(self) -> None:
        opt = _film_loaded_opt()
        opt["frame"] = FakeOption(constraint=(4, 6, 1))

        snapshot = runner.snapshot_options(FakeDev(opt))
        failures = runner.evaluate_preconditions(snapshot, runner.PROFILES["small"])

        assert any("frame constraint [4, 6, 1] cannot select frame 3" in failure for failure in failures)

    def test_frame_count_constraint_must_allow_single_exposure_reset(self) -> None:
        opt = _film_loaded_opt()
        opt["frame_count"] = FakeOption(constraint=(2, 6, 1))

        snapshot = runner.snapshot_options(FakeDev(opt))
        failures = runner.evaluate_preconditions(snapshot, runner.PROFILES["small"])

        assert any("frame_count constraint [2, 6, 1] cannot reset to 1" in failure for failure in failures)

    def test_missing_multisample_refuses(self) -> None:
        opt = _film_loaded_opt()
        del opt["samples_per_scan"]
        snapshot = runner.snapshot_options(FakeDev(opt))
        failures = runner.evaluate_preconditions(snapshot, runner.PROFILES["small"])
        assert any("'samples_per_scan' is missing" in failure for failure in failures)

    def test_multisample_range_quantum_must_allow_requested_value(self) -> None:
        opt = _film_loaded_opt()
        opt["samples_per_scan"] = FakeOption(constraint=(1, 8, 2))

        snapshot = runner.snapshot_options(FakeDev(opt))
        failures = runner.evaluate_preconditions(snapshot, runner.PROFILES["small"])

        assert any("cannot honor 4x" in failure for failure in failures)

    def test_active_but_read_only_capture_option_refuses(self) -> None:
        opt = _film_loaded_opt()
        opt["infrared"] = FakeOption(settable=False)

        snapshot = runner.snapshot_options(FakeDev(opt))
        failures = runner.evaluate_preconditions(snapshot, runner.PROFILES["small"])

        assert any("option 'infrared' is not settable" in failure for failure in failures)

    def test_subframe_constraint_must_allow_profile_value(self) -> None:
        opt = _film_loaded_opt()
        opt["subframe"] = FakeOption(constraint=(0.0, 5.0, 0.0))

        snapshot = runner.snapshot_options(FakeDev(opt))
        failures = runner.evaluate_preconditions(snapshot, runner.PROFILES["small"])

        assert any("subframe constraint [0.0, 5.0, 0.0] cannot honor 6.35" in failure for failure in failures)

    def test_subframe_numeric_enum_can_allow_profile_value(self) -> None:
        opt = _film_loaded_opt()
        opt["subframe"] = FakeOption(constraint=[0.0, 6.35, 12.7])

        snapshot = runner.snapshot_options(FakeDev(opt))
        failures = runner.evaluate_preconditions(snapshot, runner.PROFILES["small"])

        assert not any("subframe constraint" in failure for failure in failures)

    def test_scan_height_constraint_must_allow_profile_value(self) -> None:
        opt = _film_loaded_opt()
        opt["br_y"] = FakeOption(constraint=(0, 799, 1))

        snapshot = runner.snapshot_options(FakeDev(opt))
        failures = runner.evaluate_preconditions(snapshot, runner.PROFILES["small"])

        assert any("br_y constraint [0, 799, 1] cannot honor 800" in failure for failure in failures)


class TestDryRun:
    def test_dry_run_with_no_film_refuses_and_never_starts(self) -> None:
        dev = FakeDev(_no_film_opt())
        module = FakeSaneModule(dev, "net:10.0.0.100:coolscan3:usb:libusb:001:017")

        report, code = runner.run(module, _args(), runner.PROFILES["small"])

        assert code == 2
        assert report["verdict"] == "preconditions-failed"
        assert report["precondition_failures"]
        assert dev.started is False
        assert dev.closed is True  # probe handle released

    def test_dry_run_with_film_passes_and_never_starts(self) -> None:
        dev = FakeDev(_film_loaded_opt())
        module = FakeSaneModule(dev, "net:10.0.0.100:coolscan3:usb:libusb:001:017")

        report, code = runner.run(module, _args(), runner.PROFILES["small"])

        assert code == 0
        assert report["verdict"] == "dry-run-ok"
        assert dev.started is False

    def test_live_mode_still_refuses_on_failed_preconditions(self) -> None:
        dev = FakeDev(_no_film_opt())
        module = FakeSaneModule(dev, "net:10.0.0.100:coolscan3:usb:libusb:001:017")

        report, code = runner.run(module, _args(live=True), runner.PROFILES["small"])

        assert code == 2
        assert report["verdict"] == "preconditions-failed"
        assert dev.started is False

    def test_live_mode_refuses_when_preflight_handle_does_not_close(self) -> None:
        class CloseFailingDev(FakeDev):
            def close(self) -> None:
                self.closed = True
                raise RuntimeError("USB close failed")

        dev = CloseFailingDev(_film_loaded_opt())
        module = FakeSaneModule(dev, "net:10.0.0.100:coolscan3:usb:libusb:001:017")

        report, code = runner.run(module, _args(live=True), runner.PROFILES["small"])

        assert code == 2
        assert report["verdict"] == "preconditions-failed"
        assert any("preflight probe failed: RuntimeError: USB close failed" in failure for failure in report["precondition_failures"])
        assert dev.started is False
        assert len(module.opened) == 1


class TestResponsivenessProbe:
    def test_close_failure_means_scanner_is_not_responsive(self) -> None:
        class CloseFailingDev(FakeDev):
            def close(self) -> None:
                raise RuntimeError("USB close failed")

        module = FakeSaneModule(
            CloseFailingDev(_film_loaded_opt()),
            "net:10.0.0.100:coolscan3:usb:libusb:001:017",
        )

        responsive, snapshot, error = runner.probe_responsiveness(
            module,
            "net:10.0.0.100:coolscan3:usb:libusb:001:017",
            runner.PROFILES["small"],
        )

        assert responsive is False
        assert snapshot is None
        assert error == "RuntimeError: USB close failed"

    def test_unrecognizable_option_map_is_not_responsive(self) -> None:
        module = FakeSaneModule(
            FakeDev({}),
            "net:10.0.0.100:coolscan3:usb:libusb:001:017",
        )

        responsive, snapshot, error = runner.probe_responsiveness(
            module,
            "net:10.0.0.100:coolscan3:usb:libusb:001:017",
            runner.PROFILES["small"],
        )

        assert responsive is False
        assert snapshot is not None
        assert error == "post-scan probe did not expose depth, resolution, and infrared options"


class TestRecordingDev:
    def test_records_option_write_after_the_device_accepts_it(self) -> None:
        events: list[dict] = []

        class RealDev:
            def __setattr__(self, name, value) -> None:
                events.append({"event": "underlying_set", "option": name, "value": value})
                object.__setattr__(self, name, value)

        dev = runner.RecordingDev(RealDev(), events)
        dev.infrared = True

        assert [event["event"] for event in events] == ["underlying_set", "set"]

    def test_records_rejected_option_write_as_a_failure(self) -> None:
        events: list[dict] = []

        class RefusingDev:
            def __setattr__(self, name, value) -> None:
                raise RuntimeError("device rejected option")

        dev = runner.RecordingDev(RefusingDev(), events)

        with pytest.raises(RuntimeError, match="device rejected option"):
            dev.infrared = True

        assert [event["event"] for event in events] == ["set_failed"]
        assert events[0]["option"] == "infrared"
        assert events[0]["value"] is True
        assert events[0]["error"] == "RuntimeError: device rejected option"


class TestStoppedTransportSmearAssessment:
    def test_detects_a_textured_repeated_suffix_after_an_abrupt_boundary(self) -> None:
        rgb = _textured_rgb_with_repeated_suffix()

        assessment = runner.assess_stopped_transport_smear(rgb, dpi=1000)

        assert assessment.verdict == "smear"
        assert assessment.minimum_matches == 16
        assert assessment.suffix_rows == 20
        assert assessment.start_row == 76

    def test_random_eof_is_clean(self) -> None:
        rng = np.random.default_rng(91)
        rgb = rng.integers(0, 65536, size=(64, 80, 3), dtype=np.uint16)

        assessment = runner.assess_stopped_transport_smear(rgb, dpi=1000)

        assert assessment.verdict == "clean"
        assert assessment.suffix_rows == 0
        assert assessment.start_row is None

    def test_repeated_block_that_does_not_reach_eof_is_clean(self) -> None:
        rng = np.random.default_rng(92)
        rgb = rng.integers(0, 65536, size=(96, 80, 3), dtype=np.uint16)
        rgb[30:60] = rgb[29]

        assessment = runner.assess_stopped_transport_smear(rgb, dpi=1000)

        assert assessment.verdict == "clean"

    def test_partial_eof_signature_is_indeterminate(self) -> None:
        rgb = _textured_rgb_with_repeated_suffix(repeated_rows=16)

        assessment = runner.assess_stopped_transport_smear(rgb, dpi=1000)

        assert assessment.verdict == "indeterminate"
        assert assessment.minimum_matches == 16
        assert assessment.suffix_rows == 15

    def test_exact_minimum_signature_is_smear(self) -> None:
        rgb = _textured_rgb_with_repeated_suffix(repeated_rows=17)

        assessment = runner.assess_stopped_transport_smear(rgb, dpi=1000)

        assert assessment.verdict == "smear"
        assert assessment.suffix_rows == 16

    def test_4000_dpi_requires_64_matching_transitions(self) -> None:
        rgb = _textured_rgb_with_repeated_suffix(height=160, repeated_rows=65)

        assessment = runner.assess_stopped_transport_smear(rgb, dpi=4000)

        assert assessment.verdict == "smear"
        assert assessment.minimum_matches == 64
        assert assessment.suffix_rows == 64

    def test_whole_repeated_image_is_indeterminate_without_a_reference_band(self) -> None:
        rng = np.random.default_rng(93)
        anchor = rng.integers(4000, 60000, size=(80, 3), dtype=np.uint16)
        rgb = np.repeat(anchor[np.newaxis, :, :], 96, axis=0)

        assessment = runner.assess_stopped_transport_smear(rgb, dpi=1000)

        assert assessment.verdict == "indeterminate"
        assert "reference band" in assessment.reason

    def test_low_texture_repeated_tail_is_indeterminate(self) -> None:
        rng = np.random.default_rng(94)
        rgb = rng.integers(1000, 64000, size=(96, 80, 3), dtype=np.uint16)
        anchor = np.linspace(10000, 11000, 80 * 3, dtype=np.uint16).reshape(80, 3)
        rgb[-21:] = anchor

        assessment = runner.assess_stopped_transport_smear(rgb, dpi=1000)

        assert assessment.verdict == "indeterminate"
        assert assessment.texture_span is not None
        assert assessment.texture_span < 2048
        assert "texture" in assessment.reason

    def test_gradual_tail_without_an_abrupt_boundary_is_indeterminate(self) -> None:
        anchor = np.linspace(10000, 50000, 80 * 3, dtype=np.uint16).reshape(80, 3)
        rgb = np.empty((96, 80, 3), dtype=np.uint16)
        for row in range(75):
            rgb[row] = anchor + (200 if row % 2 else 0)
        for offset, row in enumerate(range(75, 96)):
            rgb[row] = anchor + offset * 100

        assessment = runner.assess_stopped_transport_smear(rgb, dpi=1000)

        assert assessment.verdict == "indeterminate"
        assert assessment.pre_tail_median_rms is not None
        assert assessment.tail_median_rms is not None
        assert assessment.pre_tail_median_rms < 4 * assessment.tail_median_rms
        assert "abrupt" in assessment.reason

    @pytest.mark.parametrize(
        ("rgb", "dpi"),
        [
            (np.zeros((40, 40, 3), dtype=np.float32), 1000),
            (np.zeros((40, 40), dtype=np.uint16), 1000),
            (np.zeros((40, 40, 3), dtype=np.uint16), 0),
        ],
    )
    def test_invalid_or_unmeasurable_input_is_indeterminate(self, rgb: np.ndarray, dpi: int) -> None:
        assessment = runner.assess_stopped_transport_smear(rgb, dpi=dpi)

        assert assessment.verdict == "indeterminate"

    def test_only_the_central_ninety_percent_drives_detection(self) -> None:
        rng = np.random.default_rng(95)
        rgb = rng.integers(1000, 64000, size=(96, 100, 3), dtype=np.uint16)
        rgb[-21:, :5] = rgb[-22, :5]
        rgb[-21:, 95:] = rgb[-22, 95:]

        assessment = runner.assess_stopped_transport_smear(rgb, dpi=1000)

        assert assessment.verdict == "clean"


class TestArtifactAcceptance:
    def test_artifact_report_exposes_file_backed_tiff_inspection(self, tmp_path) -> None:
        rgb = np.arange(4 * 5 * 3, dtype=np.uint16).reshape(4, 5, 3)
        result = ScanResult(
            rgb=rgb,
            ir=np.zeros(rgb.shape[:2], dtype=np.uint16),
            dpi=1000,
            device_model="LS-5000 ED",
        )
        rgb_path = tmp_path / "frame.tif"
        ir_path = tmp_path / "frame_IR.tif"
        tifffile.imwrite(
            rgb_path,
            result.rgb,
            photometric="rgb",
            resolution=(1000, 1000),
            resolutionunit="INCH",
        )
        tifffile.imwrite(
            ir_path,
            result.ir,
            photometric="minisblack",
            resolution=(1000, 1000),
            resolutionunit="INCH",
        )

        artifacts, _ = runner.inspect_artifacts(result, str(rgb_path), str(ir_path), expected_dpi=1000)

        assert artifacts["rgb"]["page_count"] == 1
        assert artifacts["rgb"]["payload_within_file"] is True
        assert artifacts["rgb"]["x_resolution"] == [1000, 1]
        assert artifacts["rgb"]["y_resolution"] == [1000, 1]
        assert artifacts["rgb"]["resolution_unit"] == "INCH"

    def test_multipage_tiff_fails_artifact_acceptance(self, tmp_path) -> None:
        rgb = np.arange(4 * 5 * 3, dtype=np.uint16).reshape(4, 5, 3)
        result = ScanResult(
            rgb=rgb,
            ir=np.zeros(rgb.shape[:2], dtype=np.uint16),
            dpi=1000,
            device_model="LS-5000 ED",
        )
        rgb_path = tmp_path / "frame.tif"
        ir_path = tmp_path / "frame_IR.tif"
        with tifffile.TiffWriter(rgb_path) as document:
            document.write(
                result.rgb,
                photometric="rgb",
                resolution=(1000, 1000),
                resolutionunit="INCH",
            )
            document.write(
                result.rgb,
                photometric="rgb",
                resolution=(1000, 1000),
                resolutionunit="INCH",
            )
        tifffile.imwrite(
            ir_path,
            result.ir,
            photometric="minisblack",
            resolution=(1000, 1000),
            resolutionunit="INCH",
        )

        artifacts, checks = runner.inspect_artifacts(result, str(rgb_path), str(ir_path), expected_dpi=1000)

        assert artifacts["rgb"]["page_count"] == 2
        assert checks["tiff_pair_reloads_16bit_at_capture_dpi"] is False

    def test_mismatched_y_resolution_fails_artifact_acceptance(self, tmp_path) -> None:
        rgb = np.arange(4 * 5 * 3, dtype=np.uint16).reshape(4, 5, 3)
        result = ScanResult(
            rgb=rgb,
            ir=np.zeros(rgb.shape[:2], dtype=np.uint16),
            dpi=1000,
            device_model="LS-5000 ED",
        )
        rgb_path = tmp_path / "frame.tif"
        ir_path = tmp_path / "frame_IR.tif"
        tifffile.imwrite(
            rgb_path,
            result.rgb,
            photometric="rgb",
            resolution=(1000, 999),
            resolutionunit="INCH",
        )
        tifffile.imwrite(
            ir_path,
            result.ir,
            photometric="minisblack",
            resolution=(1000, 1000),
            resolutionunit="INCH",
        )

        artifacts, checks = runner.inspect_artifacts(result, str(rgb_path), str(ir_path), expected_dpi=1000)

        assert artifacts["rgb"]["y_resolution"] == [999, 1]
        assert checks["tiff_pair_reloads_16bit_at_capture_dpi"] is False

    def test_missing_y_resolution_fails_artifact_acceptance(self, tmp_path) -> None:
        rgb = np.arange(4 * 5 * 3, dtype=np.uint16).reshape(4, 5, 3)
        result = ScanResult(
            rgb=rgb,
            ir=np.zeros(rgb.shape[:2], dtype=np.uint16),
            dpi=1000,
            device_model="LS-5000 ED",
        )
        rgb_path = tmp_path / "frame.tif"
        ir_path = tmp_path / "frame_IR.tif"
        tifffile.imwrite(
            rgb_path,
            result.rgb,
            photometric="rgb",
            resolution=(1000, 1000),
            resolutionunit="INCH",
        )
        tifffile.imwrite(
            ir_path,
            result.ir,
            photometric="minisblack",
            resolution=(1000, 1000),
            resolutionunit="INCH",
        )
        with tifffile.TiffFile(rgb_path) as document:
            byteorder = document.byteorder
            page = cast(tifffile.TiffPage, document.pages[0])
            tag_offset = page.tags["YResolution"].offset
        with rgb_path.open("r+b") as stream:
            stream.seek(tag_offset)
            stream.write(struct.pack(f"{byteorder}H", 65000))

        artifacts, checks = runner.inspect_artifacts(result, str(rgb_path), str(ir_path), expected_dpi=1000)

        assert artifacts["rgb"]["y_resolution"] is None
        assert checks["tiff_pair_reloads_16bit_at_capture_dpi"] is False

    def test_non_inch_resolution_unit_fails_artifact_acceptance(self, tmp_path) -> None:
        rgb = np.arange(4 * 5 * 3, dtype=np.uint16).reshape(4, 5, 3)
        result = ScanResult(
            rgb=rgb,
            ir=np.zeros(rgb.shape[:2], dtype=np.uint16),
            dpi=1000,
            device_model="LS-5000 ED",
        )
        rgb_path = tmp_path / "frame.tif"
        ir_path = tmp_path / "frame_IR.tif"
        tifffile.imwrite(
            rgb_path,
            result.rgb,
            photometric="rgb",
            resolution=(1000, 1000),
            resolutionunit="CENTIMETER",
        )
        tifffile.imwrite(
            ir_path,
            result.ir,
            photometric="minisblack",
            resolution=(1000, 1000),
            resolutionunit="INCH",
        )

        artifacts, checks = runner.inspect_artifacts(result, str(rgb_path), str(ir_path), expected_dpi=1000)

        assert artifacts["rgb"]["resolution_unit"] == "CENTIMETER"
        assert checks["tiff_pair_reloads_16bit_at_capture_dpi"] is False

    def test_rational_dpi_requires_exact_equality_but_accepts_equivalent_fractions(self, tmp_path) -> None:
        rgb = np.arange(4 * 5 * 3, dtype=np.uint16).reshape(4, 5, 3)
        result = ScanResult(
            rgb=rgb,
            ir=np.zeros(rgb.shape[:2], dtype=np.uint16),
            dpi=1000,
            device_model="LS-5000 ED",
        )
        ir_path = tmp_path / "frame_IR.tif"
        tifffile.imwrite(
            ir_path,
            result.ir,
            photometric="minisblack",
            resolution=(1000, 1000),
            resolutionunit="INCH",
        )

        def _write_rgb_with_rational(path, numerator: int, denominator: int) -> None:
            tifffile.imwrite(
                path,
                result.rgb,
                photometric="rgb",
                resolution=(1000, 1000),
                resolutionunit="INCH",
            )
            with tifffile.TiffFile(path) as document:
                byteorder = document.byteorder
                page = cast(tifffile.TiffPage, document.pages[0])
                value_offsets = [page.tags[name].valueoffset for name in ("XResolution", "YResolution")]
            with path.open("r+b") as stream:
                for value_offset in value_offsets:
                    stream.seek(value_offset)
                    stream.write(struct.pack(f"{byteorder}II", numerator, denominator))

        equivalent_path = tmp_path / "equivalent.tif"
        _write_rgb_with_rational(equivalent_path, 100000, 100)
        equivalent_artifacts, equivalent_checks = runner.inspect_artifacts(
            result,
            str(equivalent_path),
            str(ir_path),
            expected_dpi=1000,
        )

        unequal_path = tmp_path / "unequal.tif"
        _write_rgb_with_rational(unequal_path, 100001, 100)
        unequal_artifacts, unequal_checks = runner.inspect_artifacts(
            result,
            str(unequal_path),
            str(ir_path),
            expected_dpi=1000,
        )

        assert equivalent_artifacts["rgb"]["x_resolution"] == [100000, 100]
        assert equivalent_checks["tiff_pair_reloads_16bit_at_capture_dpi"] is True
        assert unequal_artifacts["rgb"]["x_resolution"] == [100001, 100]
        assert unequal_checks["tiff_pair_reloads_16bit_at_capture_dpi"] is False

    def test_out_of_bounds_tiff_payload_fails_artifact_acceptance(self, tmp_path) -> None:
        rgb = np.arange(4 * 5 * 3, dtype=np.uint16).reshape(4, 5, 3)
        result = ScanResult(
            rgb=rgb,
            ir=np.zeros(rgb.shape[:2], dtype=np.uint16),
            dpi=1000,
            device_model="LS-5000 ED",
        )
        rgb_path = tmp_path / "frame.tif"
        ir_path = tmp_path / "frame_IR.tif"
        tifffile.imwrite(
            rgb_path,
            result.rgb,
            photometric="rgb",
            resolution=(1000, 1000),
            resolutionunit="INCH",
        )
        tifffile.imwrite(
            ir_path,
            result.ir,
            photometric="minisblack",
            resolution=(1000, 1000),
            resolutionunit="INCH",
        )
        with tifffile.TiffFile(rgb_path) as document:
            byteorder = document.byteorder
            page = cast(tifffile.TiffPage, document.pages[0])
            value_offset = page.tags["StripOffsets"].valueoffset
        with rgb_path.open("r+b") as stream:
            stream.seek(value_offset)
            stream.write(struct.pack(f"{byteorder}I", rgb_path.stat().st_size + 100))

        artifacts, checks = runner.inspect_artifacts(result, str(rgb_path), str(ir_path), expected_dpi=1000)

        assert artifacts["rgb"]["payload_within_file"] is False
        assert checks["tiff_pair_reloads_16bit_at_capture_dpi"] is False

    def test_reloaded_clean_rgb_passes_smear_acceptance(self, tmp_path) -> None:
        rng = np.random.default_rng(96)
        rgb = rng.integers(0, 65536, size=(64, 80, 3), dtype=np.uint16)
        result = ScanResult(
            rgb=rgb,
            ir=np.zeros(rgb.shape[:2], dtype=np.uint16),
            dpi=1000,
            device_model="LS-5000 ED",
        )
        rgb_path = tmp_path / "frame.tif"
        ir_path = tmp_path / "frame_IR.tif"
        tifffile.imwrite(rgb_path, rgb, photometric="rgb", resolution=(1000, 1000))
        tifffile.imwrite(ir_path, result.ir, photometric="minisblack", resolution=(1000, 1000))

        artifacts, checks = runner.inspect_artifacts(result, str(rgb_path), str(ir_path), expected_dpi=1000)

        assert artifacts["rgb"]["stopped_transport_smear"]["verdict"] == "clean"
        assert checks["no_stopped_transport_smear_suffix"] is True

    def test_reloaded_rgb_smear_is_recorded_and_fails_artifact_acceptance(self, tmp_path) -> None:
        rgb = _textured_rgb_with_repeated_suffix()
        result = ScanResult(
            rgb=rgb,
            ir=np.zeros(rgb.shape[:2], dtype=np.uint16),
            dpi=1000,
            device_model="LS-5000 ED",
        )
        rgb_path = tmp_path / "frame.tif"
        ir_path = tmp_path / "frame_IR.tif"
        tifffile.imwrite(rgb_path, rgb, photometric="rgb", resolution=(1000, 1000))
        tifffile.imwrite(ir_path, result.ir, photometric="minisblack", resolution=(1000, 1000))

        artifacts, checks = runner.inspect_artifacts(result, str(rgb_path), str(ir_path), expected_dpi=1000)

        assert artifacts["rgb"]["stopped_transport_smear"]["verdict"] == "smear"
        assert artifacts["rgb"]["stopped_transport_smear"]["suffix_rows"] == 20
        assert checks["no_stopped_transport_smear_suffix"] is False

    def test_smear_fails_final_verdict_but_responsiveness_is_still_probed(self, tmp_path) -> None:
        rgb = _textured_rgb_with_repeated_suffix()
        result = ScanResult(
            rgb=rgb,
            ir=np.zeros(rgb.shape[:2], dtype=np.uint16),
            dpi=1000,
            device_model="LS-5000 ED",
        )
        module = FakeSaneModule(
            FakeDev(_film_loaded_opt()),
            "net:10.0.0.100:coolscan3:usb:libusb:001:017",
        )

        def write_pair(result, path):
            tifffile.imwrite(path, result.rgb, photometric="rgb", resolution=(1000, 1000))
            tifffile.imwrite(path.replace(".tif", "_IR.tif"), result.ir, photometric="minisblack", resolution=(1000, 1000))
            return path

        report, code = runner.finalize_live_capture(
            report={"mode": "live"},
            checks={"capture_succeeded": True},
            result=result,
            scan_error=None,
            sane_module=module,
            device_id="net:10.0.0.100:coolscan3:usb:libusb:001:017",
            args=_args(live=True, out_dir=str(tmp_path)),
            profile=runner.PROFILES["small"],
            write_tiff_fn=write_pair,
        )

        assert code == 1
        assert report["verdict"] == "live-failed"
        assert report["assertions"]["no_stopped_transport_smear_suffix"] is False
        assert report["assertions"]["artifact_pipeline_succeeded"] is False
        assert report["assertions"]["all_passed"] is False
        assert report["assertions"]["scanner_responsive_after_scan"] is True
        assert report["artifacts"]["rgb"]["stopped_transport_smear"]["verdict"] == "smear"
        assert len(module.opened) == 1

    def test_reloaded_tiff_shapes_must_match_in_memory_result(self, tmp_path) -> None:
        result = ScanResult(
            rgb=np.zeros((4, 5, 3), dtype=np.uint16),
            ir=np.zeros((4, 5), dtype=np.uint16),
            dpi=1000,
            device_model="LS-5000 ED",
        )
        rgb_path = tmp_path / "frame.tif"
        ir_path = tmp_path / "frame_IR.tif"
        tifffile.imwrite(rgb_path, np.zeros((3, 5, 3), dtype=np.uint16), photometric="rgb", resolution=(1000, 1000))
        tifffile.imwrite(ir_path, result.ir, photometric="minisblack", resolution=(1000, 1000))

        _, checks = runner.inspect_artifacts(result, str(rgb_path), str(ir_path), expected_dpi=1000)

        assert checks["tiff_shapes_match_in_memory"] is False
        assert checks["no_stopped_transport_smear_suffix"] is False

    def test_artifact_failure_is_reported_and_responsiveness_is_still_probed(self, tmp_path) -> None:
        result = ScanResult(
            rgb=np.zeros((4, 5, 3), dtype=np.uint16),
            ir=np.zeros((4, 5), dtype=np.uint16),
            dpi=1000,
            device_model="LS-5000 ED",
        )
        module = FakeSaneModule(
            FakeDev(_film_loaded_opt()),
            "net:10.0.0.100:coolscan3:usb:libusb:001:017",
        )

        def fail_write(result, path):
            raise OSError("disk full")

        report, code = runner.finalize_live_capture(
            report={"mode": "live"},
            checks={"capture_succeeded": True},
            result=result,
            scan_error=None,
            sane_module=module,
            device_id="net:10.0.0.100:coolscan3:usb:libusb:001:017",
            args=_args(live=True, out_dir=str(tmp_path)),
            profile=runner.PROFILES["small"],
            write_tiff_fn=fail_write,
        )

        assert code == 1
        assert report["verdict"] == "live-failed"
        assert report["artifact_error"] == "OSError: disk full"
        assert report["assertions"]["artifact_pipeline_succeeded"] is False
        assert report["assertions"]["no_stopped_transport_smear_suffix"] is False
        assert report["assertions"]["scanner_responsive_after_scan"] is True
        assert len(module.opened) == 1
        json.dumps(report)

    def test_failure_report_falls_back_to_a_timestamped_json_path(self, tmp_path) -> None:
        blocked_out_dir = tmp_path / "not-a-directory"
        blocked_out_dir.write_text("occupied")
        report = {"mode": "live", "verdict": "live-failed", "artifact_error": "OSError: disk full"}

        report_path = runner.write_report(
            report,
            _args(live=True, out_dir=str(blocked_out_dir)),
            fallback_dir=str(tmp_path),
        )

        assert report_path.startswith(str(tmp_path))
        assert "report-live-" in report_path
        assert report_path.endswith(".json")
        with open(report_path) as stream:
            saved = json.load(stream)
        assert saved["artifact_error"] == "OSError: disk full"
        assert "report_write_error" in saved


class TestDeriveAssertions:
    def _events(self) -> list[dict]:
        return [
            {"ts": "t", "event": "open", "device": "d"},
            {"ts": "t", "event": "set", "option": "depth", "value": 16},
            {"ts": "t", "event": "set", "option": "resolution", "value": 1000},
            {"ts": "t", "event": "set", "option": "frame", "value": 3},
            {"ts": "t", "event": "set", "option": "subframe", "value": 6.35},
            {"ts": "t", "event": "set", "option": "br_y", "value": 800},
            {"ts": "t", "event": "set", "option": "autofocus", "value": True},
            {"ts": "t", "event": "set", "option": "ae", "value": True},
            {"ts": "t", "event": "set", "option": "samples_per_scan", "value": 4},
            {"ts": "t", "event": "set", "option": "infrared", "value": False},
            {"ts": "t", "event": "start"},
            {"ts": "t", "event": "read_end"},
            {"ts": "t", "event": "set", "option": "frame_count", "value": 1},
            {"ts": "t", "event": "set", "option": "samples_per_scan", "value": 1},
            {"ts": "t", "event": "set", "option": "infrared", "value": True},
            {"ts": "t", "event": "set", "option": "autofocus", "value": False},
            {"ts": "t", "event": "set", "option": "ae", "value": False},
            {"ts": "t", "event": "start"},
            {"ts": "t", "event": "read_end"},
        ]

    def _result(self, alignment: SplitIrAlignment) -> ScanResult:
        return ScanResult(
            rgb=np.zeros((4, 5, 3), dtype=np.uint16),
            ir=np.zeros((4, 5), dtype=np.uint16),
            dpi=1000,
            device_model="LS-5000 ED",
            ir_alignment=alignment,
        )

    def test_clean_split_capture_log_passes_event_checks(self) -> None:
        checks = runner.derive_assertions(self._events(), open_count=1, result=None, profile=runner.PROFILES["small"])
        for name in (
            "one_device_open",
            "exactly_two_starts",
            "exactly_two_completed_reads_in_order",
            "frame_written_once_before_first_start",
            "subframe_written_once_before_first_start",
            "br_y_written_once_before_first_start",
            "depth_16_written_once_before_first_start",
            "resolution_written_once_before_first_start",
            "frame_count_reset_once_between_captures",
            "no_geometry_write_after_first_start",
            "samples_n_then_1",
            "infrared_false_then_true",
            "autofocus_true_then_false",
            "ae_true_then_false",
        ):
            assert checks[name], name

    def test_wrong_registered_frame_write_is_flagged(self) -> None:
        events = self._events()
        next(event for event in events if event.get("option") == "frame")["value"] = 2

        checks = runner.derive_assertions(
            events,
            open_count=1,
            result=None,
            profile=runner.PROFILES["small"],
        )

        assert checks["frame_written_once_before_first_start"] is False

    def test_wrong_registered_subframe_write_is_flagged(self) -> None:
        events = self._events()
        next(event for event in events if event.get("option") == "subframe")["value"] = 6.0

        checks = runner.derive_assertions(
            events,
            open_count=1,
            result=None,
            profile=runner.PROFILES["small"],
        )

        assert checks["subframe_written_once_before_first_start"] is False

    def test_wrong_registered_scan_height_write_is_flagged(self) -> None:
        events = self._events()
        next(event for event in events if event.get("option") == "br_y")["value"] = 799

        checks = runner.derive_assertions(
            events,
            open_count=1,
            result=None,
            profile=runner.PROFILES["small"],
        )

        assert checks["br_y_written_once_before_first_start"] is False

    def test_wrong_bit_depth_write_is_flagged(self) -> None:
        events = self._events()
        next(event for event in events if event.get("option") == "depth")["value"] = 8

        checks = runner.derive_assertions(
            events,
            open_count=1,
            result=None,
            profile=runner.PROFILES["small"],
        )

        assert checks["depth_16_written_once_before_first_start"] is False

    def test_wrong_resolution_write_is_flagged(self) -> None:
        events = self._events()
        next(event for event in events if event.get("option") == "resolution")["value"] = 2000

        checks = runner.derive_assertions(
            events,
            open_count=1,
            result=None,
            profile=runner.PROFILES["small"],
        )

        assert checks["resolution_written_once_before_first_start"] is False

    def test_frame_count_reset_must_be_between_first_read_and_second_start(self) -> None:
        events = self._events()
        reset_index = next(i for i, event in enumerate(events) if event.get("option") == "frame_count")
        reset = events.pop(reset_index)
        first_start = next(i for i, event in enumerate(events) if event["event"] == "start")
        events.insert(first_start, reset)

        checks = runner.derive_assertions(
            events,
            open_count=1,
            result=None,
            profile=runner.PROFILES["small"],
        )

        assert checks["frame_count_reset_once_between_captures"] is False

    def test_single_sample_switch_must_be_between_first_read_and_second_start(self) -> None:
        events = self._events()
        switch_index = next(i for i, event in enumerate(events) if event.get("option") == "samples_per_scan" and event["value"] == 1)
        switch = events.pop(switch_index)
        second_start = [i for i, event in enumerate(events) if event["event"] == "start"][1]
        events.insert(second_start + 1, switch)

        checks = runner.derive_assertions(
            events,
            open_count=1,
            result=None,
            profile=runner.PROFILES["small"],
        )

        assert checks["samples_n_then_1"] is False

    def test_infrared_enable_must_be_between_first_read_and_second_start(self) -> None:
        events = self._events()
        switch_index = next(i for i, event in enumerate(events) if event.get("option") == "infrared" and event["value"] is True)
        switch = events.pop(switch_index)
        second_start = [i for i, event in enumerate(events) if event["event"] == "start"][1]
        events.insert(second_start + 1, switch)

        checks = runner.derive_assertions(
            events,
            open_count=1,
            result=None,
            profile=runner.PROFILES["small"],
        )

        assert checks["infrared_false_then_true"] is False

    def test_autofocus_disable_must_be_between_first_read_and_second_start(self) -> None:
        events = self._events()
        switch_index = next(i for i, event in enumerate(events) if event.get("option") == "autofocus" and event["value"] is False)
        switch = events.pop(switch_index)
        second_start = [i for i, event in enumerate(events) if event["event"] == "start"][1]
        events.insert(second_start + 1, switch)

        checks = runner.derive_assertions(
            events,
            open_count=1,
            result=None,
            profile=runner.PROFILES["small"],
        )

        assert checks["autofocus_true_then_false"] is False

    def test_auto_exposure_disable_must_be_between_first_read_and_second_start(self) -> None:
        events = self._events()
        switch_index = next(i for i, event in enumerate(events) if event.get("option") == "ae" and event["value"] is False)
        switch = events.pop(switch_index)
        second_start = [i for i, event in enumerate(events) if event["event"] == "start"][1]
        events.insert(second_start + 1, switch)

        checks = runner.derive_assertions(
            events,
            open_count=1,
            result=None,
            profile=runner.PROFILES["small"],
        )

        assert checks["ae_true_then_false"] is False

    def test_geometry_write_after_first_start_is_flagged(self) -> None:
        events = self._events()
        events.append({"ts": "t", "event": "set", "option": "frame", "value": 4})
        checks = runner.derive_assertions(events, open_count=1, result=None, profile=runner.PROFILES["small"])
        assert checks["no_geometry_write_after_first_start"] is False

    def test_double_open_is_flagged(self) -> None:
        checks = runner.derive_assertions(self._events(), open_count=2, result=None, profile=runner.PROFILES["small"])
        assert checks["one_device_open"] is False

    def test_missing_second_completed_read_is_flagged(self) -> None:
        events = self._events()
        events.pop()

        checks = runner.derive_assertions(events, open_count=1, result=None, profile=runner.PROFILES["small"])

        assert checks["exactly_two_completed_reads_in_order"] is False

    def test_identity_alignment_requires_exactly_zero_finite_shift(self) -> None:
        result = self._result(SplitIrAlignment(mode="identity", dx_px=0.25, dy_px=0.0))

        checks = runner.derive_assertions(self._events(), open_count=1, result=result, profile=runner.PROFILES["small"])

        assert checks["alignment_confident"] is False

    def test_phase_ecc_alignment_accepts_all_production_confidence_metrics(self) -> None:
        result = self._result(
            SplitIrAlignment(
                mode="phase-ecc",
                dx_px=1.25,
                dy_px=-0.5,
                phase_responses=(0.20, 0.25, 0.30),
                channel_spread_px=0.5,
                ecc_coefficient=0.9,
            )
        )

        checks = runner.derive_assertions(self._events(), open_count=1, result=result, profile=runner.PROFILES["small"])

        assert checks["alignment_confident"] is True

    def test_phase_ecc_alignment_revalidates_shift_against_result_width(self) -> None:
        result = self._result(
            SplitIrAlignment(
                mode="phase-ecc",
                dx_px=17.0,
                dy_px=0.0,
                phase_responses=(0.20, 0.25, 0.30),
                channel_spread_px=0.5,
                ecc_coefficient=0.9,
            )
        )

        checks = runner.derive_assertions(
            self._events(),
            open_count=1,
            result=result,
            profile=runner.PROFILES["small"],
        )

        assert result.rgb.shape[1] == 5
        assert checks["alignment_confident"] is False

    def test_phase_ecc_alignment_rejects_one_phase_response_below_production_gate(self) -> None:
        result = self._result(
            SplitIrAlignment(
                mode="phase-ecc",
                dx_px=1.25,
                dy_px=-0.5,
                phase_responses=(0.20, _SPLIT_MIN_PHASE_RESPONSE - 0.01, 0.30),
                channel_spread_px=0.5,
                ecc_coefficient=0.9,
            )
        )

        checks = runner.derive_assertions(self._events(), open_count=1, result=result, profile=runner.PROFILES["small"])

        assert checks["alignment_confident"] is False

    def test_phase_ecc_alignment_requires_one_response_per_rgb_channel(self) -> None:
        result = self._result(
            SplitIrAlignment(
                mode="phase-ecc",
                dx_px=1.25,
                dy_px=-0.5,
                phase_responses=(0.20, 0.25),
                channel_spread_px=0.5,
                ecc_coefficient=0.9,
            )
        )

        checks = runner.derive_assertions(
            self._events(),
            open_count=1,
            result=result,
            profile=runner.PROFILES["small"],
        )

        assert checks["alignment_confident"] is False

    def test_phase_ecc_alignment_rejects_channel_spread_above_production_gate(self) -> None:
        result = self._result(
            SplitIrAlignment(
                mode="phase-ecc",
                dx_px=1.25,
                dy_px=-0.5,
                phase_responses=(0.20, 0.25, 0.30),
                channel_spread_px=_SPLIT_MAX_CHANNEL_SPREAD_PX + 0.01,
                ecc_coefficient=0.9,
            )
        )

        checks = runner.derive_assertions(self._events(), open_count=1, result=result, profile=runner.PROFILES["small"])

        assert checks["alignment_confident"] is False

    def test_phase_ecc_alignment_rejects_ecc_below_production_gate(self) -> None:
        result = self._result(
            SplitIrAlignment(
                mode="phase-ecc",
                dx_px=1.25,
                dy_px=-0.5,
                phase_responses=(0.20, 0.25, 0.30),
                channel_spread_px=0.5,
                ecc_coefficient=_SPLIT_MIN_ECC - 0.01,
            )
        )

        checks = runner.derive_assertions(self._events(), open_count=1, result=result, profile=runner.PROFILES["small"])

        assert checks["alignment_confident"] is False
