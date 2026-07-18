from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from negpy.services.scanning import ls5000_ice
from negpy.services.scanning.ls5000_ice import (
    ICE_FRAME_RECEIPT_KIND,
    IceRollError,
    ProcessedIceFrame,
    acquire_ice_bundle,
    process_ice_bundle,
    publish_ice_frame,
)
from negpy.services.scanning.ls5000_sane_rgb import (
    LS5000_FINE_WIDTH,
    LS5000_FULL_WINDOW_ROWS,
)
from negpy.infrastructure.scanners.dice_dual_source_runner import DiceDualSourcePlan


class _FakeDevice:
    def __init__(self, log: list[str]) -> None:
        self._log = log

    def close(self) -> None:
        self._log.append("device.close")


class _FakeLibsane:
    instances: list["_FakeLibsane"] = []

    def __init__(self) -> None:
        self.log: list[str] = []
        _FakeLibsane.instances.append(self)

    def require_ls5000(self, device_id: str):
        self.log.append(f"require:{device_id}")
        return SimpleNamespace(device_id=device_id)

    def open(self, device_id: str, *, identity=None):
        self.log.append(f"open:{device_id}")
        return _FakeDevice(self.log)

    def close(self) -> None:
        self.log.append("sane.close")


def test_acquire_closes_every_handle_before_the_bundle_is_written(monkeypatch, tmp_path) -> None:
    _FakeLibsane.instances = []
    plan = DiceDualSourcePlan.for_transport("roll", frame=3, subframe_mm=1.25)
    capture = SimpleNamespace(scanner_identity=SimpleNamespace(device_id="coolscan3:usb:test"))
    order: list[str] = []

    def fake_acquire(device, acquired_plan, *, progress=None):
        assert acquired_plan is plan
        _FakeLibsane.instances[0].log.append("acquire")
        return capture

    def fake_write(bundle_root, *, device_id, plan, capture, run_id):
        order.extend(_FakeLibsane.instances[0].log)
        order.append("write_bundle")
        return Path(bundle_root) / run_id

    monkeypatch.setattr(ls5000_ice, "Libsane", _FakeLibsane)
    monkeypatch.setattr(ls5000_ice, "acquire_dual_sources", fake_acquire)
    monkeypatch.setattr(ls5000_ice, "write_capture_bundle", fake_write)

    bundle = acquire_ice_bundle(
        device_id="coolscan3:usb:test",
        plan=plan,
        bundle_root=tmp_path,
        run_id="slot03-test",
    )

    assert bundle == tmp_path / "slot03-test"
    assert order == [
        "require:coolscan3:usb:test",
        "open:coolscan3:usb:test",
        "acquire",
        "device.close",
        "sane.close",
        "write_bundle",
    ]


def test_acquire_closes_handles_even_when_acquisition_fails(monkeypatch, tmp_path) -> None:
    _FakeLibsane.instances = []

    def failing_acquire(device, plan, *, progress=None):
        raise RuntimeError("transport jammed")

    monkeypatch.setattr(ls5000_ice, "Libsane", _FakeLibsane)
    monkeypatch.setattr(ls5000_ice, "acquire_dual_sources", failing_acquire)

    with pytest.raises(RuntimeError, match="transport jammed"):
        acquire_ice_bundle(
            device_id="coolscan3:usb:test",
            plan=DiceDualSourcePlan.for_transport("roll", frame=1),
            bundle_root=tmp_path,
            run_id="slot01-fail",
        )

    assert _FakeLibsane.instances[0].log[-2:] == ["device.close", "sane.close"]


def test_process_reloads_through_the_gate_and_binds_identity(monkeypatch, tmp_path) -> None:
    bundle_dir = tmp_path / "pair-a"
    bundle_dir.mkdir()
    (bundle_dir / "receipt.json").write_text(
        json.dumps({"manifest": "manifest.json", "manifest_sha256": "f" * 64}),
        encoding="utf-8",
    )
    plan = DiceDualSourcePlan.for_transport("roll", frame=2)
    capture = SimpleNamespace(
        scanner_identity=SimpleNamespace(vendor="Nikon", model="LS-5000 ED"),
    )
    ice_result = SimpleNamespace(marker="engine-output")
    loads: list[Path] = []

    def fake_load(path):
        loads.append(Path(path))
        return capture, plan

    def fake_apply(loaded_capture, *, plan, backend, progress=None):
        assert loaded_capture is capture
        assert backend == "cpu-fast"
        return ice_result

    monkeypatch.setattr(ls5000_ice, "load_capture_bundle", fake_load)
    monkeypatch.setattr(ls5000_ice, "apply_portable_digital_ice", fake_apply)

    processed = process_ice_bundle(bundle_dir, backend="cpu-fast")

    assert loads == [bundle_dir]
    assert processed.ice is ice_result
    assert processed.plan is plan
    assert processed.bundle_manifest_sha256 == "f" * 64
    assert processed.device_model == "Nikon LS-5000 ED"


class _WritingService:
    """Minimal ScannerService stand-in that actually writes the RGB file."""

    def __init__(self) -> None:
        self.write_calls: list[dict[str, Any]] = []

    def write_result(self, *, result, output_folder, filename_pattern, output_format, slot):
        self.write_calls.append(
            {
                "result": result,
                "output_folder": output_folder,
                "filename_pattern": filename_pattern,
                "output_format": output_format,
                "slot": slot,
            }
        )
        path = Path(output_folder) / f"ice_slot{slot:02d}.tif"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"cleaned-master")
        return str(path)


def _full_cleaned() -> np.ndarray:
    rows = np.arange(LS5000_FULL_WINDOW_ROWS, dtype=np.uint16).reshape(-1, 1, 1)
    return np.broadcast_to(rows, (LS5000_FULL_WINDOW_ROWS, LS5000_FINE_WIDTH, 3))


def _processed(cleaned: np.ndarray) -> ProcessedIceFrame:
    receipt = SimpleNamespace(
        status="processed",
        same_frame_id="roll-slot-05",
    )
    ice = SimpleNamespace(
        cleaned_rgb16=cleaned,
        requested_backend=SimpleNamespace(value="cpu-fast"),
        used_backend=SimpleNamespace(value="cpu-fast"),
        selection_reason="explicit CPU-FAST request",
        receipt=receipt,
    )
    return ProcessedIceFrame(
        ice=ice,
        plan=DiceDualSourcePlan.for_transport("roll", frame=5),
        bundle_manifest_sha256="e" * 64,
        device_model="Nikon LS-5000 ED",
    )


def test_publish_writes_master_and_receipt_with_bound_evidence(monkeypatch, tmp_path) -> None:
    service = _WritingService()
    processed = _processed(_full_cleaned())
    # The engine receipt is a frozen dataclass in production; the fake uses a
    # namespace, so serialize it the same way.
    monkeypatch.setattr(ls5000_ice, "asdict", lambda value: dict(vars(value)))

    rgb_path = publish_ice_frame(
        processed,
        service=service,
        output_folder=str(tmp_path / "scans"),
        filename_pattern='roll_{{ "%03d" % seq }}',
        roll_slot=5,
        boundary_offset_rows=-7,
    )

    [call] = service.write_calls
    assert call["slot"] == 5
    assert call["output_format"] == "TIFF"
    # Stored orientation: rot90 of scanner-native portrait.
    assert call["result"].rgb.shape == (LS5000_FINE_WIDTH, LS5000_FULL_WINDOW_ROWS, 3)
    assert call["result"].ir is None
    assert call["result"].dpi == 4_000

    receipt_path = Path(rgb_path).with_name("ice_slot05_SCAN.json")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["kind"] == ICE_FRAME_RECEIPT_KIND
    assert receipt["roll_slot"] == 5
    assert receipt["boundary_offset_rows"] == -7
    assert receipt["bundle_manifest_sha256"] == "e" * 64
    assert receipt["backend"]["used"] == "cpu-fast"
    assert receipt["engine_receipt"]["same_frame_id"] == "roll-slot-05"
    assert receipt["plan"]["transport"] == "roll"
    assert receipt["output"]["rgb"]["path"] == "ice_slot05.tif"
    assert receipt["output"]["rgb"]["bytes"] == len(b"cleaned-master")


def test_publish_refuses_a_wrong_shape_master_before_writing(tmp_path) -> None:
    service = _WritingService()
    processed = _processed(np.zeros((10, 12, 3), dtype=np.uint16))

    with pytest.raises(IceRollError, match="uint16 RGB"):
        publish_ice_frame(
            processed,
            service=service,
            output_folder=str(tmp_path / "scans"),
            filename_pattern='roll_{{ "%03d" % seq }}',
            roll_slot=5,
            boundary_offset_rows=0,
        )

    assert service.write_calls == []


def test_publish_removes_the_master_when_the_receipt_cannot_be_written(monkeypatch, tmp_path) -> None:
    service = _WritingService()
    processed = _processed(_full_cleaned())
    # A receipt that cannot serialize must never leave a master without its
    # evidence beside it.
    monkeypatch.setattr(ls5000_ice, "asdict", lambda value: {"unserializable": object()})
    scans = tmp_path / "scans"

    with pytest.raises(TypeError):
        publish_ice_frame(
            processed,
            service=service,
            output_folder=str(scans),
            filename_pattern='roll_{{ "%03d" % seq }}',
            roll_slot=5,
            boundary_offset_rows=0,
        )

    assert service.write_calls  # the master was written first...
    assert not (scans / "ice_slot05.tif").exists()  # ...then removed
    assert not (scans / "ice_slot05_SCAN.json").exists()


# --- hybrid repair ------------------------------------------------------------


_HYBRID_ENV_VALUES = {
    "NEGPY_HYBRID_CLI": "cli",
    "NEGPY_HYBRID_IOPAINT_PYTHON": "iopaint-python",
    "NEGPY_HYBRID_IOPAINT_EXECUTABLE": "iopaint-exe",
    "NEGPY_HYBRID_IOPAINT_SOURCE_SHA256": "1" * 64,
    "NEGPY_HYBRID_MODEL_DIR": "model-dir",
    "NEGPY_HYBRID_MODEL_WEIGHTS": "model-weights",
    "NEGPY_HYBRID_MODEL_WEIGHTS_SHA256": "2" * 64,
}


def _set_hybrid_env(monkeypatch, tmp_path) -> dict[str, str]:
    values: dict[str, str] = {}
    for variable, name in _HYBRID_ENV_VALUES.items():
        if variable.endswith("SHA256"):
            values[variable] = name
        else:
            target = tmp_path / name
            if name == "model-dir":
                target.mkdir()
            else:
                target.write_text("x", encoding="utf-8")
            values[variable] = str(target)
        monkeypatch.setenv(variable, values[variable])
    return values


def test_hybrid_config_reads_a_complete_environment(monkeypatch, tmp_path) -> None:
    _set_hybrid_env(monkeypatch, tmp_path)
    monkeypatch.setenv("NEGPY_HYBRID_INPAINT_DEVICE", "mps")

    config = ls5000_ice.HybridRepairConfig.from_env()

    assert config.cli == tmp_path / "cli"
    assert config.model_dir == tmp_path / "model-dir"
    assert config.iopaint_source_sha256 == "1" * 64
    assert config.inpaint_device == "mps"

    ok, reason = ls5000_ice.hybrid_availability()
    assert ok is True
    assert "mps" in reason


def test_hybrid_config_refuses_missing_variables_and_paths(monkeypatch, tmp_path) -> None:
    for variable in _HYBRID_ENV_VALUES:
        monkeypatch.delenv(variable, raising=False)

    with pytest.raises(ls5000_ice.IceRollError, match="NEGPY_HYBRID_CLI"):
        ls5000_ice.HybridRepairConfig.from_env()

    _set_hybrid_env(monkeypatch, tmp_path)
    (tmp_path / "model-weights").unlink()
    with pytest.raises(ls5000_ice.IceRollError, match="NEGPY_HYBRID_MODEL_WEIGHTS"):
        ls5000_ice.HybridRepairConfig.from_env()

    ok, reason = ls5000_ice.hybrid_availability()
    assert ok is False
    assert "NEGPY_HYBRID_MODEL_WEIGHTS" in reason


def test_hybrid_config_refuses_an_unknown_inpaint_device(monkeypatch, tmp_path) -> None:
    _set_hybrid_env(monkeypatch, tmp_path)
    monkeypatch.setenv("NEGPY_HYBRID_INPAINT_DEVICE", "tpu")

    with pytest.raises(ls5000_ice.IceRollError, match="cpu, mps, or cuda"):
        ls5000_ice.HybridRepairConfig.from_env()


def _hybrid_config(tmp_path) -> "ls5000_ice.HybridRepairConfig":
    (tmp_path / "model-dir").mkdir(exist_ok=True)
    for name in ("cli", "iopaint-python", "iopaint-exe", "model-weights"):
        (tmp_path / name).write_text("x", encoding="utf-8")
    return ls5000_ice.HybridRepairConfig(
        cli=tmp_path / "cli",
        iopaint_python=tmp_path / "iopaint-python",
        iopaint_executable=tmp_path / "iopaint-exe",
        iopaint_source_sha256="1" * 64,
        model_dir=tmp_path / "model-dir",
        model_weights=tmp_path / "model-weights",
        model_weights_sha256="2" * 64,
        inpaint_device="cpu",
    )


def _fake_bundle(monkeypatch, tmp_path) -> Path:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "receipt.json").write_text(
        json.dumps({"manifest": "manifest.json", "manifest_sha256": "d" * 64}),
        encoding="utf-8",
    )
    manifest = {
        "same_frame_id": "roll-slot-07",
        "plan": {"transport": "roll", "frame": 7},
        "scanner_identity": {"vendor": "Nikon", "model": "LS-5000 ED"},
    }
    monkeypatch.setattr(ls5000_ice, "verify_capture_bundle", lambda root: manifest)
    return bundle


def test_run_hybrid_repair_builds_the_full_argv_and_parses_outputs(monkeypatch, tmp_path) -> None:
    config = _hybrid_config(tmp_path)
    bundle = _fake_bundle(monkeypatch, tmp_path)
    out_dir = tmp_path / "hybrid-out"
    seen: dict[str, Any] = {}

    def fake_runner(argv, *, capture_output, text, timeout):
        seen["argv"] = list(argv)
        out = Path(argv[argv.index("--out") + 1])
        out.mkdir(parents=True)
        rows = np.arange(4, dtype=np.uint16).reshape(1, 1, 4)
        np.save(out / "output-hybrid.rgb16.npy", rows[:, :, :3])
        np.save(out / "output.rgb16.npy", rows[:, :, :3])
        (out / "synth-mask.png").write_bytes(b"png")
        (out / "hybrid-receipt.json").write_text(
            json.dumps({"synthesis": {"pixel_count": 3}}), encoding="utf-8"
        )
        (out / "routing.json").write_text(json.dumps({"regions": []}), encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    outputs = ls5000_ice.run_hybrid_repair(
        bundle,
        out_dir,
        config=config,
        backend="cpu-fast",
        runner=fake_runner,
    )

    argv = seen["argv"]
    assert argv[0] == str(config.cli)
    assert argv[argv.index("--prepass") + 1] == str(bundle / "prepass_rgbi.npy")
    assert argv[argv.index("--main") + 1] == str(bundle / "main_rgbi.npy")
    assert argv[argv.index("--same-frame-id") + 1] == "roll-slot-07"
    assert "--assert-focus-exposure-locked" in argv
    assert argv[argv.index("--backend") + 1] == "cpu-fast"
    assert argv[argv.index("--inpaint-device") + 1] == "cpu"
    assert argv[argv.index("--model-weights-sha256") + 1] == "2" * 64
    assert outputs.bundle_manifest_sha256 == "d" * 64
    assert outputs.device_model == "Nikon LS-5000 ED"
    assert outputs.plan_semantic == {"transport": "roll", "frame": 7}
    assert outputs.hybrid_receipt == {"synthesis": {"pixel_count": 3}}


def test_run_hybrid_repair_fails_loud_on_exit_code_and_missing_outputs(monkeypatch, tmp_path) -> None:
    config = _hybrid_config(tmp_path)
    bundle = _fake_bundle(monkeypatch, tmp_path)

    def failing_runner(argv, *, capture_output, text, timeout):
        return SimpleNamespace(returncode=3, stdout="", stderr="engine exploded")

    with pytest.raises(ls5000_ice.IceRollError, match="exit 3.*engine exploded"):
        ls5000_ice.run_hybrid_repair(
            bundle, tmp_path / "out-a", config=config, backend="cpu", runner=failing_runner
        )

    def incomplete_runner(argv, *, capture_output, text, timeout):
        out = Path(argv[argv.index("--out") + 1])
        out.mkdir(parents=True)
        (out / "output.rgb16.npy").write_bytes(b"")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    with pytest.raises(ls5000_ice.IceRollError, match="output-hybrid.rgb16.npy is missing"):
        ls5000_ice.run_hybrid_repair(
            bundle, tmp_path / "out-b", config=config, backend="cpu", runner=incomplete_runner
        )

    with pytest.raises(ls5000_ice.IceRollError, match="already exists"):
        existing = tmp_path / "out-c"
        existing.mkdir()
        ls5000_ice.run_hybrid_repair(
            bundle, existing, config=config, backend="cpu", runner=failing_runner
        )


def _hybrid_outputs(tmp_path) -> "ls5000_ice.HybridRunOutputs":
    out_dir = tmp_path / "hybrid-run"
    out_dir.mkdir()
    mask = out_dir / "synth-mask.png"
    mask.write_bytes(b"mask-bytes")
    return ls5000_ice.HybridRunOutputs(
        out_dir=out_dir,
        hybrid_rgb16=_full_cleaned(),
        pure_rgb16=_full_cleaned(),
        synth_mask_path=mask,
        hybrid_receipt={"synthesis": {"pixel_count": 9, "fraction": 0.0001}},
        routing={"regions": []},
        bundle_manifest_sha256="d" * 64,
        plan_semantic={"transport": "roll", "frame": 7},
        device_model="Nikon LS-5000 ED",
    )


def test_publish_hybrid_frame_writes_both_masters_mask_and_receipt(monkeypatch, tmp_path) -> None:
    service = _WritingService()
    outputs = _hybrid_outputs(tmp_path)
    written_pure: list[str] = []

    def fake_write_tiff(result, path):
        written_pure.append(path)
        final = path + ".tif" if not path.endswith(".tif") else path
        Path(final).write_bytes(b"pure-master")
        return final

    monkeypatch.setattr("negpy.services.scanning.writer.write_tiff_16bit", fake_write_tiff)

    rgb_path = ls5000_ice.publish_hybrid_frame(
        outputs,
        service=service,
        output_folder=str(tmp_path / "scans"),
        filename_pattern='roll_{{ "%03d" % seq }}',
        roll_slot=7,
        boundary_offset_rows=3,
    )

    base = rgb_path.removesuffix(".tif")
    assert written_pure == [base + "_ICE"]
    assert Path(base + "_ICE.tif").read_bytes() == b"pure-master"
    assert Path(base + "_SYNTH.png").read_bytes() == b"mask-bytes"
    receipt = json.loads(Path(base + "_SCAN.json").read_text(encoding="utf-8"))
    assert receipt["kind"] == ls5000_ice.ICE_HYBRID_RECEIPT_KIND
    assert receipt["roll_slot"] == 7
    assert receipt["boundary_offset_rows"] == 3
    assert receipt["bundle_manifest_sha256"] == "d" * 64
    assert receipt["synthesis"] == {"pixel_count": 9, "fraction": 0.0001}
    assert receipt["hybrid_receipt"]["synthesis"]["pixel_count"] == 9
    assert set(receipt["outputs"]) == {"hybrid_rgb", "ice_rgb", "synth_mask"}
    assert receipt["outputs"]["synth_mask"]["bytes"] == len(b"mask-bytes")


def test_publish_hybrid_frame_cleans_everything_on_receipt_failure(monkeypatch, tmp_path) -> None:
    service = _WritingService()
    outputs = _hybrid_outputs(tmp_path)
    object.__setattr__(outputs, "hybrid_receipt", {"synthesis": object()})

    def fake_write_tiff(result, path):
        final = path + ".tif"
        Path(final).write_bytes(b"pure-master")
        return final

    monkeypatch.setattr("negpy.services.scanning.writer.write_tiff_16bit", fake_write_tiff)

    with pytest.raises(TypeError):
        ls5000_ice.publish_hybrid_frame(
            outputs,
            service=service,
            output_folder=str(tmp_path / "scans"),
            filename_pattern='roll_{{ "%03d" % seq }}',
            roll_slot=7,
            boundary_offset_rows=0,
        )

    scans = tmp_path / "scans"
    leftovers = sorted(p.name for p in scans.iterdir()) if scans.exists() else []
    assert leftovers == []
