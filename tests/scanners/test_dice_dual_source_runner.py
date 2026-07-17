from __future__ import annotations

import json
import hashlib
import threading
from dataclasses import dataclass, field

import numpy as np
import pytest

from negpy.infrastructure.scanners import dice_dual_source_runner as runner
from negpy.infrastructure.scanners.dice_dual_source_runner import (
    BundleVerificationError,
    CaptureCancelled,
    DiceDualSourcePlan,
    OptionInfo,
    PixelWindow,
    RawFrame,
    ScannerIdentity,
    _decode_rgbi16,
    acquire_dual_sources,
    exact_next_command,
    main,
    verify_capture_bundle,
    write_capture_bundle,
)


def _option(
    name: str,
    *,
    constraint: tuple[object, ...] | None = None,
    active: bool = True,
    settable: bool = True,
    maximum: float = 4000.0,
) -> OptionInfo:
    return OptionInfo(
        name=name,
        value_type=1,
        active=active,
        settable=settable,
        constraint=constraint,
        range_constraint=None if constraint is not None else (0.0, maximum, 1.0),
    )


@dataclass
class FakeRawDevice:
    prepass: np.ndarray
    main: np.ndarray
    device_id: str = "coolscan3:usb:test"
    values: dict[str, object] = field(default_factory=dict)
    writes: list[tuple[str, object]] = field(default_factory=list)
    reads: list[str] = field(default_factory=list)
    cancelled: bool = False
    bad_readback: str | None = None
    identity: ScannerIdentity = field(
        default_factory=lambda: ScannerIdentity(
            device_id="coolscan3:usb:test",
            vendor="Nikon",
            model="LS-5000 ED",
            kind="film scanner",
        )
    )

    def __post_init__(self) -> None:
        self.values.update(
            {
                "focus": 1,
                "exposure": 1.0,
                "red_exposure": 1200.0,
                "green_exposure": 1200.0,
                "blue_exposure": 1000.0,
                "frame_count": 1,
            }
        )
        names = (
            "depth",
            "resolution",
            "preview",
            "negative",
            "samples_per_scan",
            "infrared",
            "autofocus",
            "ae",
            "focus",
            "exposure",
            "red_exposure",
            "green_exposure",
            "blue_exposure",
            "tl_x",
            "tl_y",
            "br_x",
            "br_y",
            "frame_count",
        )
        self.option_map = {name: _option(name) for name in names}
        self.option_map["depth"] = _option("depth", constraint=(8, 16))
        self.option_map["resolution"] = _option(
            "resolution", constraint=(285, 4000)
        )
        self.option_map["samples_per_scan"] = _option(
            "samples_per_scan", constraint=(1, 2, 4, 8, 16)
        )

    def options(self) -> dict[str, OptionInfo]:
        return dict(self.option_map)

    def set_option(self, name: str, value: object) -> None:
        self.values[name] = value
        self.writes.append((name, value))

    def get_option(self, name: str) -> object:
        value = self.values[name]
        if self.bad_readback == name:
            if isinstance(value, bool):
                return not value
            if isinstance(value, (int, float)):
                return value + 1
        return value

    def read_rgbi(
        self,
        *,
        expected_shape: tuple[int, int],
        label: str,
        progress=None,
        cancel: threading.Event | None = None,
    ) -> RawFrame:
        if cancel is not None and cancel.is_set():
            raise CaptureCancelled(f"{label} cancelled")
        self.reads.append(label)
        if label == "prepass":
            self.values.update(
                {
                    "focus": 216,
                    "exposure": 1.0,
                    "red_exposure": 1370.0,
                    "green_exposure": 1290.0,
                    "blue_exposure": 1120.0,
                }
            )
            array = self.prepass
        else:
            array = self.main
        assert array.shape[:2] == expected_shape
        if progress is not None:
            progress(0.5)
            progress(1.0)
        bytes_per_line = array.shape[1] * 4 * 2
        return RawFrame(
            rgbi=array,
            bytes_per_line=bytes_per_line,
            bytes_read=bytes_per_line * array.shape[0],
        )

    def cancel(self) -> None:
        self.cancelled = True

    def close(self) -> None:
        pass


def _small_plan() -> DiceDualSourcePlan:
    # Twenty-eight native pixels give a non-empty two-pixel 285 dpi prepass
    # while keeping the acquisition tests tiny.
    return DiceDualSourcePlan(
        window=PixelWindow(0, 0, 27, 27),
        transport="mounted",
    )


def _arrays(plan: DiceDualSourcePlan) -> tuple[np.ndarray, np.ndarray]:
    prepass = np.arange(
        plan.prepass_full_shape[0] * plan.prepass_full_shape[1] * 4,
        dtype=np.uint16,
    ).reshape(*plan.prepass_full_shape, 4)
    main = np.arange(
        plan.main_full_shape[0] * plan.main_full_shape[1] * 4,
        dtype=np.uint16,
    ).reshape(*plan.main_full_shape, 4)
    return prepass, main


def test_exact_plans_are_285_then_4000_single_sample_rgbi() -> None:
    roll = DiceDualSourcePlan.for_transport("roll")
    mounted = DiceDualSourcePlan.for_transport("mounted")

    assert roll.prepass_full_shape == (425, 281)
    assert roll.main_full_shape == (5959, 3946)
    assert mounted.prepass_full_shape == (413, 281)
    assert mounted.main_full_shape == (5782, 3946)
    assert mounted.semantic_dict()["samples_per_scan"] == 1
    assert mounted.semantic_dict()["orientation"] == "scanner_native_portrait"


def test_plan_rejects_non_exact_resolution_and_mounted_transport_position() -> None:
    with pytest.raises(ValueError, match="285 dpi prepass"):
        DiceDualSourcePlan(prepass_dpi=500)
    with pytest.raises(ValueError, match="4000 dpi main"):
        DiceDualSourcePlan(main_dpi=500)
    with pytest.raises(ValueError, match="mounted transport"):
        DiceDualSourcePlan.for_transport("mounted", frame=1)


def test_raw_decoder_preserves_the_final_rgbi_row() -> None:
    expected = np.arange(17 * 3 * 4, dtype=np.uint16).reshape(17, 3, 4)

    decoded = _decode_rgbi16(
        expected.tobytes(),
        width=3,
        height=17,
        bytes_per_line=3 * 4 * 2,
    )

    assert np.array_equal(decoded, expected)
    assert np.array_equal(decoded[-1], expected[-1])


def test_acquisition_is_one_handle_prepass_then_locked_main() -> None:
    plan = _small_plan()
    prepass, main_array = _arrays(plan)
    device = FakeRawDevice(prepass, main_array)
    progress: list[float] = []

    capture = acquire_dual_sources(device, plan, progress=progress.append)

    assert device.reads == ["prepass", "main"]
    assert capture.assertions["all_passed"] is True
    assert capture.capture_state.focus_position == 216
    assert capture.prepass_rgbi.flags.writeable is False
    assert capture.main_rgbi.flags.writeable is False
    assert np.array_equal(capture.prepass_rgbi, prepass)
    assert np.array_equal(capture.main_rgbi, main_array)
    assert progress[0] == 0.0
    assert progress[-1] == 1.0
    assert progress == sorted(progress)
    assert [value for name, value in device.writes if name == "resolution"] == [
        285,
        4000,
    ]
    assert [value for name, value in device.writes if name == "samples_per_scan"] == [
        1
    ]
    assert ("autofocus", False) in device.writes
    assert ("ae", False) in device.writes
    assert ("focus", 216) in device.writes


def test_preflight_refuses_before_scanner_mutation() -> None:
    plan = _small_plan()
    prepass, main_array = _arrays(plan)
    device = FakeRawDevice(prepass, main_array)
    del device.option_map["infrared"]

    with pytest.raises(RuntimeError, match="preflight failed"):
        acquire_dual_sources(device, plan)

    assert device.writes == []
    assert device.reads == []


def test_preflight_refuses_a_different_coolscan_model_before_mutation() -> None:
    plan = _small_plan()
    prepass, main_array = _arrays(plan)
    device = FakeRawDevice(
        prepass,
        main_array,
        identity=ScannerIdentity(
            device_id="coolscan3:usb:test",
            vendor="Nikon",
            model="LS-4000 ED",
            kind="film scanner",
        ),
    )

    with pytest.raises(RuntimeError, match="Super Coolscan 5000 ED"):
        acquire_dual_sources(device, plan)

    assert device.writes == []
    assert device.reads == []


def test_cancellation_and_readback_mismatch_fail_closed() -> None:
    plan = _small_plan()
    prepass, main_array = _arrays(plan)
    cancelled = FakeRawDevice(prepass, main_array)
    event = threading.Event()
    event.set()

    with pytest.raises(CaptureCancelled):
        acquire_dual_sources(cancelled, plan, cancel=event)
    assert cancelled.cancelled is True
    assert cancelled.writes == []

    mismatch = FakeRawDevice(prepass, main_array, bad_readback="negative")
    with pytest.raises(RuntimeError, match="read back"):
        acquire_dual_sources(mismatch, plan)
    assert mismatch.cancelled is True
    assert mismatch.reads == []


def test_capture_bundle_is_transactional_and_self_verifying(tmp_path) -> None:
    plan = _small_plan()
    prepass, main_array = _arrays(plan)
    capture = acquire_dual_sources(FakeRawDevice(prepass, main_array), plan)

    bundle = write_capture_bundle(
        tmp_path,
        device_id="coolscan3:usb:test",
        plan=plan,
        capture=capture,
        run_id="pair-001",
    )

    assert bundle == tmp_path / "pair-001"
    assert not any(path.name.endswith(".partial") for path in tmp_path.iterdir())
    manifest = verify_capture_bundle(bundle)
    assert manifest["same_frame_id"] == capture.same_frame_id
    assert set(manifest["artifacts"]) == {"prepass_rgbi", "main_rgbi"}
    assert manifest["scanner_handle_lifecycle"].startswith("caller must release")

    main_path = bundle / "main_rgbi.npy"
    payload = bytearray(main_path.read_bytes())
    payload[-1] ^= 1
    main_path.write_bytes(payload)
    with pytest.raises(BundleVerificationError, match="file SHA-256"):
        verify_capture_bundle(bundle)


def test_bundle_verifier_rejects_plan_shape_tampering(tmp_path) -> None:
    plan = _small_plan()
    prepass, main_array = _arrays(plan)
    capture = acquire_dual_sources(FakeRawDevice(prepass, main_array), plan)
    bundle = write_capture_bundle(
        tmp_path,
        device_id=capture.scanner_identity.device_id,
        plan=plan,
        capture=capture,
        run_id="pair-shape",
    )
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["plan"]["main"]["full_shape_hw"] = [1, 1]
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    receipt_path = bundle / "receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["manifest_sha256"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(BundleVerificationError, match="inconsistent with the pixel window"):
        verify_capture_bundle(bundle)


def test_libsane_open_closes_handle_if_wrapper_initialization_fails(monkeypatch) -> None:
    closed: list[object] = []

    class FakeLibrary:
        @staticmethod
        def sane_open(_device_id, _handle) -> int:
            return 0

        @staticmethod
        def sane_close(handle) -> None:
            closed.append(handle)

    sane = object.__new__(runner.Libsane)
    sane._lib = FakeLibrary()
    sane._check = lambda status, _action: None if status == 0 else pytest.fail()
    identity = ScannerIdentity(
        device_id="coolscan3:usb:test",
        vendor="Nikon",
        model="LS-5000 ED",
        kind="film scanner",
    )

    def fail_wrapper(*_args, **_kwargs):
        raise RuntimeError("option discovery failed")

    monkeypatch.setattr(runner, "LibsaneRawDevice", fail_wrapper)

    with pytest.raises(RuntimeError, match="option discovery failed"):
        sane.open(identity.device_id, identity=identity)

    assert len(closed) == 1


def test_command_and_dry_run_make_the_live_boundary_explicit(capsys, tmp_path) -> None:
    command = exact_next_command(
        output_dir=tmp_path,
        device_id="coolscan3:usb:test",
        frame=3,
        subframe_mm=0.25,
    )

    assert "--live" in command
    assert "--confirm-film-stationary" in command
    assert "--frame 3" in command
    assert "--subframe-mm 0.25" in command

    assert main(["--out-dir", str(tmp_path), "--transport", "mounted"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["plan"]["prepass"]["dpi"] == 285
    assert payload["plan"]["main"]["dpi"] == 4000
    assert payload["plan"]["samples_per_scan"] == 1


def test_live_mode_requires_physical_confirmation(tmp_path) -> None:
    with pytest.raises(SystemExit):
        main(["--live", "--out-dir", str(tmp_path)])
