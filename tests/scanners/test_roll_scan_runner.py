"""Command-level safety tests for the registered roll-scan runner."""

from __future__ import annotations

import json
import argparse
import hashlib
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import cast

import pytest
import numpy as np
import tifffile

from negpy.infrastructure.scanners import roll_scan_runner as runner
from negpy.services.scanning.roll_service import (
    PreviewScanRecipe,
    RollFrameRecord,
    RollScanPlan,
    RollStage,
    RollScanService,
)
from negpy.services.scanning.provenance import PlanIdentity
from negpy.services.scanning.quality import inspect_tiff_payload


TEST_IDENTITY = PlanIdentity(
    "11111111-1111-4111-8111-111111111111",
    "22222222-2222-4222-8222-222222222222",
    "33333333-3333-4333-8333-333333333333",
)


def _plan(stage: RollStage = RollStage.REVIEW_REQUIRED):
    return SimpleNamespace(
        device_id="coolscan3:usb:test",
        stage=stage,
        approved=stage in (RollStage.APPROVED, RollStage.COMPLETE),
        frames=(),
    )


@dataclass
class FakeService:
    calls: list[tuple[str, dict]] = field(default_factory=list)

    def prepare_roll(self, **kwargs):
        self.calls.append(("prepare", kwargs))
        return _plan()

    def approve_roll(self, plan_path, **kwargs):
        self.calls.append(("approve", {"plan_path": plan_path, **kwargs}))
        return _plan(RollStage.APPROVED)

    def scan_fine(self, **kwargs):
        self.calls.append(("fine", kwargs))
        return _plan(RollStage.APPROVED)

    def scan_full_negative(self, **kwargs):
        self.calls.append(("full-negative", kwargs))
        return {}


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("2-4", (2, 3, 4)),
        ("2-4,7,10-12", (2, 3, 4, 7, 10, 11, 12)),
    ],
)
def test_parse_frames(text: str, expected: tuple[int, ...]) -> None:
    assert runner.parse_frames(text) == expected


@pytest.mark.parametrize("text", ["", "0", "4-2", "2,2", "3,2", "2--4"])
def test_parse_frames_rejects_unsafe_lists(text: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        runner.parse_frames(text)


def test_prepare_refuses_before_service_construction_without_double_live_gate(monkeypatch, tmp_path, capsys) -> None:
    constructed = False

    def forbidden_factory():
        nonlocal constructed
        constructed = True
        raise AssertionError("service must not be constructed")

    monkeypatch.delenv(runner.LIVE_ENV, raising=False)
    code = runner.main(
        ["prepare", "--frames", "2-4", "--output", str(tmp_path), "--device", "coolscan3:usb:test"],
        service_factory=forbidden_factory,
    )

    assert code == 2
    assert constructed is False
    assert runner.LIVE_ENV in capsys.readouterr().err


def test_prepare_uses_safe_preview_recipe_and_explicit_frames(monkeypatch, tmp_path) -> None:
    service = FakeService()
    monkeypatch.setenv(runner.LIVE_ENV, "YES")

    code = runner.main(
        [
            "prepare",
            "--live",
            "--frames",
            "2-4",
            "--output",
            str(tmp_path),
            "--device",
            "coolscan3:usb:test",
        ],
        service_factory=lambda: cast(RollScanService, service),
    )

    assert code == 0
    [(command, call)] = service.calls
    assert command == "prepare"
    assert call["device_id"] == "coolscan3:usb:test"
    assert call["frames"] == (2, 3, 4)
    params = call["preview_params"]
    assert (params.dpi, params.depth) == (400, 16)
    assert params.capture_ir is False
    assert params.autofocus is False
    assert params.samples_per_scan == 1
    assert params.auto_exposure is False
    assert params.area is None


def test_prepare_loads_roll_wide_context_from_prior_chunk_plans(monkeypatch, tmp_path) -> None:
    service = FakeService()
    monkeypatch.setenv(runner.LIVE_ENV, "YES")
    context_root = tmp_path / "chunk-002-004"
    wide_path = context_root / "wide" / "frame002.tif"
    wide_path.parent.mkdir(parents=True)
    wide = np.full((12, 16, 3), 1234, dtype=np.uint16)
    tifffile.imwrite(wide_path, wide, photometric="rgb", resolution=(400, 400), resolutionunit="INCH")
    wide_inspection = inspect_tiff_payload(wide_path)
    legacy_signature = runner._registration_signature()
    legacy_signature["version"] = 2
    legacy_config = legacy_signature["config"]
    assert isinstance(legacy_config, dict)
    for config_field in ("frame_pitch_mm", "maximum_subframe_mm", "maximum_br_y_device_px"):
        legacy_config.pop(config_field)
    context_plan = RollScanPlan(
        identity=TEST_IDENTITY,
        device_id="coolscan3:usb:test",
        stage=RollStage.WIDE_PREVIEW,
        approved=False,
        preview_dpi=400,
        preview_depth=16,
        preview_recipe=PreviewScanRecipe(
            dpi=400,
            depth=16,
            capture_ir=False,
            autofocus=False,
            samples_per_scan=1,
            auto_exposure=False,
        ),
        registration_signature=legacy_signature,
        visual_override_frames=(),
        calibration_context=(),
        frames=(
            RollFrameRecord(
                frame=2,
                wide_path="wide/frame002.tif",
                wide_rgb_shape=wide.shape,
                wide_dtype="uint16",
                wide_sha256=wide_inspection.sha256,
                wide_dpi=400,
            ),
        ),
    )
    plan_path = context_root / "roll-scan.json"
    plan_path.write_text(json.dumps(context_plan.to_dict()), encoding="utf-8")

    code = runner.main(
        [
            "prepare",
            "--live",
            "--frames",
            "5-7",
            "--output",
            str(tmp_path / "chunk-005-007"),
            "--device",
            "coolscan3:usb:test",
            "--context-plan",
            str(plan_path),
        ],
        service_factory=lambda: cast(RollScanService, service),
    )

    assert code == 0
    [(command, call)] = service.calls
    assert command == "prepare"
    assert tuple(call["calibration_previews"]) == (2,)
    np.testing.assert_array_equal(call["calibration_previews"][2], wide)


def test_approve_requires_visual_confirmation_and_passes_frame_overrides(tmp_path) -> None:
    service = FakeService()
    plan_path = tmp_path / "roll-scan.json"

    refused = runner.main(
        ["approve", "--plan", str(plan_path)],
        service_factory=lambda: cast(RollScanService, service),
    )
    accepted = runner.main(
        ["approve", "--plan", str(plan_path), "--yes", "--allow-unverified", "2,4"],
        service_factory=lambda: cast(RollScanService, service),
    )

    assert refused == 2
    assert accepted == 0
    [(command, call)] = service.calls
    assert command == "approve"
    assert call["allow_unverified_frames"] == (2, 4)


def test_fine_uses_practical_parity_recipe_and_explicit_selection(monkeypatch, tmp_path) -> None:
    service = FakeService()
    monkeypatch.setenv(runner.LIVE_ENV, "YES")

    code = runner.main(
        [
            "fine",
            "--live",
            "--plan",
            str(tmp_path / "roll-scan.json"),
            "--frames",
            "3",
        ],
        service_factory=lambda: cast(RollScanService, service),
    )

    assert code == 0
    [(command, call)] = service.calls
    assert command == "fine"
    assert call["frames"] == (3,)
    params = call["fine_params"]
    assert (params.dpi, params.depth) == (4000, 16)
    assert params.capture_ir is True
    assert params.autofocus is True
    assert params.samples_per_scan == 4
    assert params.auto_exposure is True
    assert params.area is None


def test_fine_supports_a_1000_dpi_split_capture_discriminator(monkeypatch, tmp_path) -> None:
    service = FakeService()
    monkeypatch.setenv(runner.LIVE_ENV, "YES")

    code = runner.main(
        [
            "fine",
            "--live",
            "--plan",
            str(tmp_path / "roll-scan.json"),
            "--frames",
            "3",
            "--dpi",
            "1000",
        ],
        service_factory=lambda: cast(RollScanService, service),
    )

    assert code == 0
    [(command, call)] = service.calls
    assert command == "fine"
    assert call["fine_params"].dpi == 1000
    assert call["fine_params"].samples_per_scan == 4
    assert call["fine_params"].capture_ir is True


def test_full_negative_uses_approved_plan_recipe_and_persisted_insertion_identity(monkeypatch, tmp_path) -> None:
    service = FakeService()
    monkeypatch.setenv(runner.LIVE_ENV, "YES")
    output = tmp_path / "full"
    plan = tmp_path / "roll-scan.json"
    monkeypatch.setattr(
        runner,
        "_read_plan",
        lambda _path: SimpleNamespace(
            identity=TEST_IDENTITY,
            frames=(SimpleNamespace(frame=3), SimpleNamespace(frame=7)),
            device_id="coolscan3:usb:test",
        ),
    )
    monkeypatch.setattr(runner, "approved_plan_binding_sha256", lambda _plan: "a" * 64)

    code = runner.main(
        [
            "full-negative",
            "--live",
            "--plan",
            str(plan),
            "--output",
            str(output),
            "--frames",
            "3,7",
        ],
        service_factory=lambda: cast(RollScanService, service),
    )

    assert code == 0
    [(command, call)] = service.calls
    assert command == "full-negative"
    assert call["frames"] == (3, 7)
    assert call["fine_params"] == runner._fine_params(4000)
    identity_path = output / runner.FULL_NEGATIVE_IDENTITY_FILENAME
    identity_payload = json.loads(identity_path.read_text())
    identity = runner.PlanIdentity.from_dict(identity_payload["identity"])
    assert call["identity"] == identity
    assert identity == TEST_IDENTITY
    assert identity_payload["approved_plan_binding_sha256"] == "a" * 64

    refused = runner.main(
        [
            "full-negative",
            "--live",
            "--plan",
            str(plan),
            "--output",
            str(output),
            "--frames",
            "3,7",
        ],
        service_factory=lambda: cast(RollScanService, service),
    )
    assert refused == 2


def test_status_validates_plan_without_constructing_scanner(tmp_path, capsys) -> None:
    plan = RollScanPlan(
        identity=TEST_IDENTITY,
        device_id="coolscan3:usb:test",
        stage=RollStage.WIDE_PREVIEW,
        approved=False,
        preview_dpi=400,
        preview_depth=16,
        preview_recipe=PreviewScanRecipe(
            dpi=400,
            depth=16,
            capture_ir=False,
            autofocus=False,
            samples_per_scan=1,
            auto_exposure=False,
        ),
        registration_signature={"algorithm": "test", "version": 1},
        visual_override_frames=(),
        calibration_context=(),
        frames=(RollFrameRecord(frame=2),),
    )
    plan_path = tmp_path / "roll-scan.json"
    plan_path.write_text(json.dumps(plan.to_dict()), encoding="utf-8")

    def forbidden_factory():
        raise AssertionError("status must not construct scanner service")

    code = runner.main(["status", "--plan", str(plan_path)], service_factory=forbidden_factory)

    assert code == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["stage"] == "wide_preview"
    assert summary["frames"] == [
        {
            "frame": 2,
            "wide_path": None,
            "aligned_path": None,
            "geometry": None,
            "registration_confidence": None,
            "verification": None,
            "verification_passed": None,
            "fine_path": None,
        }
    ]


@dataclass
class FakeForwardOperations:
    calls: list[tuple[str, tuple[int, ...]]] = field(default_factory=list)

    def prepare_chunk(self, *, frames, plan_dir, identity, prior_plan_paths, stop, progress):
        selected = tuple(frames)
        self.calls.append(("prepare", selected))
        plan_dir.mkdir(parents=True, exist_ok=True)
        return plan_dir / "roll-scan.json"

    def approve_chunk(self, *, plan_path, allow_unverified_frames):
        self.calls.append(("approve", tuple(allow_unverified_frames)))

    def capture_chunk(self, *, plan_path, archive_dir, identity, frames, stop, progress):
        selected = tuple(frames)
        self.calls.append(("capture", selected))
        manifests = {}
        for frame in selected:
            path = archive_dir / "manifests" / f"frame{frame:03d}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(str(frame), encoding="utf-8")
            manifests[frame] = path
        return manifests

    def validate_manifest(self, *, manifest_path, identity, frame):
        assert manifest_path.read_text(encoding="utf-8") == str(frame)
        return hashlib.sha256(manifest_path.read_bytes()).hexdigest()


@dataclass
class FakeForwardOperationsWithEject(FakeForwardOperations):
    """A FakeForwardOperations that also exposes the optional eject hook."""

    eject_calls: int = 0
    eject_error: Exception | None = None
    eject_result: bool = True

    def eject(self) -> bool:
        self.eject_calls += 1
        if self.eject_error is not None:
            raise self.eject_error
        return self.eject_result


@dataclass
class FailingCaptureForwardOperations(FakeForwardOperationsWithEject):
    """Aborts mid-chunk so the run never reaches successful completion."""

    def capture_chunk(self, *, plan_path, archive_dir, identity, frames, stop, progress):
        self.calls.append(("capture", tuple(frames)))
        raise RuntimeError("synthetic capture failure")


def test_forward_roll_cli_requires_confirmation_and_runs_sequential_chunks(monkeypatch, tmp_path, capsys) -> None:
    operations = FakeForwardOperations()
    monkeypatch.setenv(runner.LIVE_ENV, "YES")
    output = tmp_path / "roll"

    refused = runner.main(
        [
            "forward-roll",
            "--live",
            "--frames",
            "1-5",
            "--output",
            str(output),
            "--device",
            "coolscan3:usb:test",
        ],
        forward_operations_factory=lambda _service, _device: operations,
    )
    accepted = runner.main(
        [
            "forward-roll",
            "--live",
            "--yes",
            "--frames",
            "1-5",
            "--chunk-size",
            "3",
            "--max-chunks",
            "1",
            "--allow-unverified",
            "1",
            "--exclude-calibration",
            "1",
            "--output",
            str(output),
            "--device",
            "coolscan3:usb:test",
        ],
        forward_operations_factory=lambda service, _device: (
            operations
            if service._current_registration_signature()["config"]["excluded_frames"] == [1]
            else (_ for _ in ()).throw(AssertionError("frame 1 must be excluded only from calibration"))
        ),
    )

    assert refused == 2
    assert accepted == 0
    first_summary = json.loads(capsys.readouterr().out)
    assert first_summary["complete"] is False
    assert first_summary["completed_frames"] == [1, 2, 3]

    resumed = runner.main(
        [
            "forward-roll",
            "--live",
            "--yes",
            "--resume",
            "--frames",
            "1-5",
            "--chunk-size",
            "3",
            "--allow-unverified",
            "1",
            "--exclude-calibration",
            "1",
            "--output",
            str(output),
            "--device",
            "coolscan3:usb:test",
        ],
        forward_operations_factory=lambda service, _device: (
            operations
            if service._current_registration_signature()["config"]["excluded_frames"] == [1]
            else (_ for _ in ()).throw(AssertionError("frame 1 must be excluded only from calibration"))
        ),
    )

    assert resumed == 0
    assert operations.calls == [
        ("prepare", (1, 2, 3)),
        ("approve", (1,)),
        ("capture", (1, 2, 3)),
        ("prepare", (4, 5)),
        ("approve", ()),
        ("capture", (4, 5)),
    ]
    summary = json.loads(capsys.readouterr().out)
    assert summary["complete"] is True
    assert summary["completed_frames"] == [1, 2, 3, 4, 5]


def _forward_roll_argv(output, *, frames: str = "1-3", extra: list[str] | None = None) -> list[str]:
    return [
        "forward-roll",
        "--live",
        "--yes",
        "--frames",
        frames,
        "--chunk-size",
        "3",
        "--output",
        str(output),
        "--device",
        "coolscan3:usb:test",
        *(extra or []),
    ]


def test_forward_roll_ejects_after_full_completion_by_default(monkeypatch, tmp_path, capsys) -> None:
    """(a) eject fires exactly once when the declared roll fully completes."""
    operations = FakeForwardOperationsWithEject()
    monkeypatch.setenv(runner.LIVE_ENV, "YES")

    code = runner.main(
        _forward_roll_argv(tmp_path / "roll"),
        forward_operations_factory=lambda _service, _device: operations,
    )

    assert code == 0
    assert operations.eject_calls == 1
    summary = json.loads(capsys.readouterr().out)
    assert summary["complete"] is True
    assert summary["ejected"] is True


def test_forward_roll_does_not_eject_when_a_max_chunks_pause_leaves_it_incomplete(monkeypatch, tmp_path, capsys) -> None:
    """(b) a deliberate --max-chunks review pause must not eject mid-roll."""
    operations = FakeForwardOperationsWithEject()
    monkeypatch.setenv(runner.LIVE_ENV, "YES")

    # Two chunks (1-3, 4-6) so --max-chunks 1 genuinely truncates the run
    # instead of coinciding with the roll's only/last chunk.
    code = runner.main(
        _forward_roll_argv(tmp_path / "roll", extra=["--max-chunks", "1"], frames="1-6"),
        forward_operations_factory=lambda _service, _device: operations,
    )

    assert code == 0
    assert operations.eject_calls == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["complete"] is False
    assert summary["ejected"] is False


def test_forward_roll_does_not_eject_when_the_run_aborts(monkeypatch, tmp_path) -> None:
    """(b) a hard failure mid-chunk must not eject; the run never reaches completion."""
    operations = FailingCaptureForwardOperations()
    monkeypatch.setenv(runner.LIVE_ENV, "YES")

    code = runner.main(
        _forward_roll_argv(tmp_path / "roll"),
        forward_operations_factory=lambda _service, _device: operations,
    )

    assert code == 2
    assert operations.eject_calls == 0


def test_forward_roll_swallows_an_eject_failure_and_still_reports_success(monkeypatch, tmp_path, capsys) -> None:
    """(c) a mocked eject failure must not fail an otherwise-complete run."""
    operations = FakeForwardOperationsWithEject(eject_error=RuntimeError("transport busy"))
    monkeypatch.setenv(runner.LIVE_ENV, "YES")

    code = runner.main(
        _forward_roll_argv(tmp_path / "roll"),
        forward_operations_factory=lambda _service, _device: operations,
    )

    assert code == 0
    assert operations.eject_calls == 1
    captured = capsys.readouterr()
    summary = json.loads(captured.out)
    assert summary["complete"] is True
    assert summary["ejected"] is False
    assert "transport busy" in captured.err


def test_forward_roll_skips_eject_cleanly_when_operations_exposes_no_eject_capability(monkeypatch, tmp_path, capsys) -> None:
    """(d) capability-gating: an operations boundary without eject() is a clean no-op."""
    operations = FakeForwardOperations()  # the plain fake predates the eject hook entirely
    monkeypatch.setenv(runner.LIVE_ENV, "YES")

    code = runner.main(
        _forward_roll_argv(tmp_path / "roll"),
        forward_operations_factory=lambda _service, _device: operations,
    )

    assert code == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["complete"] is True
    assert summary["ejected"] is False


def test_forward_roll_no_eject_flag_suppresses_eject_on_full_completion(monkeypatch, tmp_path, capsys) -> None:
    """--no-eject opts out even when the roll completes cleanly."""
    operations = FakeForwardOperationsWithEject()
    monkeypatch.setenv(runner.LIVE_ENV, "YES")

    code = runner.main(
        _forward_roll_argv(tmp_path / "roll", extra=["--no-eject"]),
        forward_operations_factory=lambda _service, _device: operations,
    )

    assert code == 0
    assert operations.eject_calls == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["complete"] is True
    assert summary["ejected"] is False


def test_runner_forward_operations_eject_delegates_to_the_roll_scan_service() -> None:
    """The production adapter's eject() is a thin passthrough to the service."""

    class RecordingService:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def eject(self, device_id: str) -> bool:
            self.calls.append(device_id)
            return True

    service = RecordingService()
    operations = runner._RunnerForwardOperations(
        cast(RollScanService, service),
        device_id="coolscan3:usb:test",
        registration_signature={"algorithm": "test", "version": 1},
    )

    assert operations.eject() is True
    assert service.calls == ["coolscan3:usb:test"]
