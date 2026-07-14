"""Hardware-free contracts for the process-isolated RGBI4x capture bridge."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from negpy.infrastructure.scanners.ls5000_single_pass import capture_process as capture
from negpy.infrastructure.scanners.ls5000_single_pass.bundle import (
    CAPTURE_BUNDLE_SHA256,
    CAPTURE_WORKER_SHA256,
)
from negpy.infrastructure.scanners.ls5000_single_pass.plan import (
    CANONICAL_FINE_READ_BYTES,
    CANONICAL_FINE_READ_COUNT,
    CANONICAL_PLAN_SHA256,
)


def _argument(argv: Sequence[str], name: str) -> str:
    index = argv.index(name)
    return argv[index + 1]


@dataclass
class FakeRunner:
    worker_sha256: str
    bundle_sha256: str | None = None
    status: str = "complete"
    recovery: str | None = None
    returncode: int = 0
    stdout: str = "worker stdout\n"
    stderr: str = "worker stderr\n"
    mutate_journal: Callable[[dict[str, object]], None] | None = None
    during_run: Callable[[], None] | None = None
    write_journal: bool = True
    calls: list[tuple[tuple[str, ...], Path]] = field(default_factory=list)

    def __call__(self, argv: Sequence[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
        command = tuple(argv)
        self.calls.append((command, cwd))
        if self.during_run is not None:
            self.during_run()

        output = Path(_argument(command, "--output"))
        journal_path = Path(_argument(command, "--journal"))
        selected = int(_argument(command, "--frame")) if "--frame" in command else None
        if "--preview-only" in command:
            mode = "preview-only"
            expected_reads = 0
            expected_bytes = 0
        elif "--meter-only" in command:
            mode = "meter-only"
            expected_reads = capture.METER_READ_COUNT
            expected_bytes = capture.METER_CAPTURE_BYTES
        else:
            mode = "full"
            expected_reads = CANONICAL_FINE_READ_COUNT
            expected_bytes = CANONICAL_FINE_READ_COUNT * CANONICAL_FINE_READ_BYTES

        if self.returncode == 0:
            with output.open("xb") as stream:
                stream.truncate(expected_bytes)
        payload: dict[str, object] = {
            "status": self.status,
            "plan_sha256": CANONICAL_PLAN_SHA256,
            "capture_engine_sha256": self.worker_sha256,
            "output": str(output.resolve()),
            "capture_mode": mode,
            "requested_frame": selected,
            "expected_frame_count": None,
            "expected_reads": expected_reads,
            "expected_bytes": expected_bytes,
            "requested_boundary_offset_rows": int(
                _argument(command, "--boundary-offset-rows")
            ),
            "completed_reads": expected_reads if self.returncode == 0 else 0,
            "completed_bytes": expected_bytes if self.returncode == 0 else 0,
        }
        if self.bundle_sha256 is not None:
            payload["capture_bundle_sha256"] = self.bundle_sha256
        if self.returncode == 0:
            payload.update(
                disk_bytes=expected_bytes,
                unit_released=True,
                output_sha256="a" * 64,
            )
            if mode != "preview-only":
                payload.update(
                    applied_boundary_offset_rows=payload[
                        "requested_boundary_offset_rows"
                    ],
                    resolved_lookup_row=2400,
                    resolved_native_origin=100_000,
                )
        else:
            payload["recovery_required"] = self.recovery
        if self.mutate_journal is not None:
            self.mutate_journal(payload)
        if self.write_journal:
            journal_path.write_text(json.dumps(payload), encoding="utf-8")
        return subprocess.CompletedProcess(command, self.returncode, self.stdout, self.stderr)


@dataclass(frozen=True)
class Binding:
    worker: Path
    manifest: Path
    worker_sha256: str


@pytest.fixture
def binding(tmp_path: Path) -> Binding:
    worker = tmp_path / "capture_rgbi4.py"
    worker.write_text("# fake external capture worker\n", encoding="utf-8")
    worker_sha256 = hashlib.sha256(worker.read_bytes()).hexdigest()
    manifest = tmp_path / "replay-first-rgbi4-manifest.json"
    manifest.write_text(json.dumps({"plan_sha256": CANONICAL_PLAN_SHA256}), encoding="utf-8")
    return Binding(worker, manifest, worker_sha256)


def _adapter(tmp_path: Path, binding: Binding, runner: FakeRunner) -> capture.CaptureProcessAdapter:
    return capture.CaptureProcessAdapter(
        worker_path=binding.worker,
        expected_worker_sha256=binding.worker_sha256,
        manifest_path=binding.manifest,
        attempts_root=tmp_path / "attempts",
        python_executable=sys.executable,
        runner=runner,
    )


def test_preview_uses_preview_only_without_slot_or_exposure_count(tmp_path: Path, binding: Binding) -> None:
    runner = FakeRunner(binding.worker_sha256)
    adapter = _adapter(tmp_path, binding, runner)

    result = adapter.run_attempt(capture.CaptureRequest(mode=capture.CaptureMode.PREVIEW))

    assert result.outcome is capture.CaptureOutcome.COMPLETE
    assert "--preview-only" in result.argv
    assert "--frame" not in result.argv
    assert "--meter-only" not in result.argv
    assert "--confirm-full-capture" not in result.argv
    assert "--expected-frame-count" not in result.argv
    assert result.paths.output.stat().st_size == 0


def test_meter_uses_one_explicit_slot_and_never_passes_expected_count(tmp_path: Path, binding: Binding) -> None:
    runner = FakeRunner(binding.worker_sha256)
    adapter = _adapter(tmp_path, binding, runner)

    result = adapter.run_attempt(capture.CaptureRequest(mode=capture.CaptureMode.METER_ONLY, selected_slot=18))

    assert result.outcome is capture.CaptureOutcome.COMPLETE
    assert _argument(result.argv, "--frame") == "18"
    assert "--meter-only" in result.argv
    assert "--preview-only" not in result.argv
    assert "--expected-frame-count" not in result.argv
    assert result.journal is not None
    assert result.journal["expected_frame_count"] is None


def test_full_capture_uses_complete_stream_confirmation(tmp_path: Path, binding: Binding) -> None:
    runner = FakeRunner(binding.worker_sha256)
    adapter = _adapter(tmp_path, binding, runner)

    result = adapter.run_attempt(capture.CaptureRequest(mode=capture.CaptureMode.FULL, selected_slot=7))

    assert result.outcome is capture.CaptureOutcome.COMPLETE
    assert _argument(result.argv, "--frame") == "7"
    assert _argument(result.argv, "--reads") == "2980"
    assert "--confirm-full-capture" in result.argv
    assert "--meter-only" not in result.argv
    assert "--preview-only" not in result.argv
    assert "--expected-frame-count" not in result.argv
    assert result.paths.output.stat().st_size == 619_458_560


@pytest.mark.parametrize(
    ("slot", "offset"),
    [(1, 0), (1, 144), (2, -144), (18, 73), (40, 144)],
)
def test_per_frame_boundary_offset_is_passed_to_the_isolated_worker(
    tmp_path: Path,
    binding: Binding,
    slot: int,
    offset: int,
) -> None:
    runner = FakeRunner(binding.worker_sha256)
    adapter = _adapter(tmp_path, binding, runner)

    result = adapter.run_attempt(
        capture.CaptureRequest(
            mode=capture.CaptureMode.FULL,
            selected_slot=slot,
            boundary_offset_rows=offset,
        )
    )

    assert _argument(result.argv, "--boundary-offset-rows") == str(offset)
    assert result.journal is not None
    assert result.journal["requested_boundary_offset_rows"] == offset


@pytest.mark.parametrize(
    ("mode", "slot", "offset"),
    [
        (capture.CaptureMode.PREVIEW, None, 1),
        (capture.CaptureMode.FULL, 1, -1),
        (capture.CaptureMode.FULL, 1, 145),
        (capture.CaptureMode.FULL, 2, -145),
        (capture.CaptureMode.FULL, 2, 145),
        (capture.CaptureMode.FULL, 2, True),
    ],
)
def test_boundary_offset_refuses_values_outside_nikon_ui_semantics(
    mode: capture.CaptureMode,
    slot: int | None,
    offset: int,
) -> None:
    with pytest.raises((TypeError, ValueError), match="boundary offset"):
        capture.CaptureRequest(
            mode=mode,
            selected_slot=slot,
            boundary_offset_rows=offset,
        )


def test_worker_and_materialized_package_plan_are_hash_pinned_before_launch(
    tmp_path: Path,
    binding: Binding,
) -> None:
    runner = FakeRunner(binding.worker_sha256)
    adapter = _adapter(tmp_path, binding, runner)
    result = adapter.run_attempt(capture.CaptureRequest(mode=capture.CaptureMode.PREVIEW))

    assert hashlib.sha256(result.paths.plan.read_bytes()).hexdigest() == CANONICAL_PLAN_SHA256
    assert _argument(result.argv, "--plan") == str(result.paths.plan)

    binding.worker.write_text("changed after binding\n", encoding="utf-8")
    with pytest.raises(capture.CaptureIntegrityError, match="worker SHA-256 mismatch"):
        adapter.run_attempt(capture.CaptureRequest(mode=capture.CaptureMode.PREVIEW))
    assert len(runner.calls) == 1


def test_manifest_must_bind_the_packaged_plan_before_launch(tmp_path: Path, binding: Binding) -> None:
    binding.manifest.write_text(json.dumps({"plan_sha256": "0" * 64}), encoding="utf-8")
    runner = FakeRunner(binding.worker_sha256)
    adapter = _adapter(tmp_path, binding, runner)

    with pytest.raises(capture.CaptureIntegrityError, match="not bound"):
        adapter.run_attempt(capture.CaptureRequest(mode=capture.CaptureMode.PREVIEW))
    assert runner.calls == []


def test_packaged_factory_uses_isolated_module_dispatch_and_internal_manifest(
    tmp_path: Path,
) -> None:
    runner = FakeRunner(CAPTURE_WORKER_SHA256, CAPTURE_BUNDLE_SHA256)
    adapter = capture.CaptureProcessAdapter.packaged(
        tmp_path / "attempts",
        runner=runner,
    )

    result = adapter.run_attempt(capture.CaptureRequest(capture.CaptureMode.PREVIEW))

    assert result.argv[:4] == (
        sys.executable,
        "-I",
        "-m",
        capture.PACKAGED_WORKER_MODULE,
    )
    assert result.paths.manifest.is_file()
    assert _argument(result.argv, "--manifest") == str(result.paths.manifest)
    assert result.journal is not None
    assert result.journal["capture_bundle_sha256"] == CAPTURE_BUNDLE_SHA256


def test_packaged_factory_uses_frozen_app_helper_dispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    runner = FakeRunner(CAPTURE_WORKER_SHA256, CAPTURE_BUNDLE_SHA256)
    adapter = capture.CaptureProcessAdapter.packaged(
        tmp_path / "attempts",
        runner=runner,
    )

    result = adapter.run_attempt(capture.CaptureRequest(capture.CaptureMode.PREVIEW))

    assert result.argv[:2] == (sys.executable, capture.CAPTURE_HELPER_FLAG)
    assert "-m" not in result.argv[:2]


def test_attempt_paths_never_overlap_and_capture_stdout_stderr(tmp_path: Path, binding: Binding) -> None:
    runner = FakeRunner(binding.worker_sha256)
    adapter = _adapter(tmp_path, binding, runner)

    first = adapter.run_attempt(capture.CaptureRequest(mode=capture.CaptureMode.PREVIEW))
    second = adapter.run_attempt(capture.CaptureRequest(mode=capture.CaptureMode.PREVIEW))

    assert first.paths.directory != second.paths.directory
    assert first.paths.stdout.read_text(encoding="utf-8") == runner.stdout
    assert first.paths.stderr.read_text(encoding="utf-8") == runner.stderr
    assert first.stdout == runner.stdout
    assert first.stderr == runner.stderr


def test_synchronized_refusal_is_safe_to_retry_without_power_cycle(tmp_path: Path, binding: Binding) -> None:
    runner = FakeRunner(
        binding.worker_sha256,
        status="failed",
        recovery="none",
        returncode=1,
    )
    result = _adapter(tmp_path, binding, runner).run_attempt(capture.CaptureRequest(mode=capture.CaptureMode.METER_ONLY, selected_slot=18))

    assert result.outcome is capture.CaptureOutcome.SYNCHRONIZED_REFUSAL
    assert result.recovery_required is False
    assert result.journal_error is None


def test_desynchronized_failure_requires_power_cycle(tmp_path: Path, binding: Binding) -> None:
    runner = FakeRunner(
        binding.worker_sha256,
        status="failed",
        recovery=capture.POWER_CYCLE_RECOVERY,
        returncode=1,
    )
    result = _adapter(tmp_path, binding, runner).run_attempt(capture.CaptureRequest(mode=capture.CaptureMode.FULL, selected_slot=18))

    assert result.outcome is capture.CaptureOutcome.RECOVERY_REQUIRED
    assert result.recovery_required is True
    assert result.journal is not None


def test_completed_frame_capture_requires_resolved_boundary_offset_evidence(
    tmp_path: Path,
    binding: Binding,
) -> None:
    runner = FakeRunner(
        binding.worker_sha256,
        mutate_journal=lambda journal: journal.update(
            applied_boundary_offset_rows=12,
            resolved_native_origin=None,
        ),
    )
    request = capture.CaptureRequest(
        mode=capture.CaptureMode.FULL,
        selected_slot=18,
        boundary_offset_rows=11,
    )

    result = _adapter(tmp_path, binding, runner).run_attempt(request)

    assert result.outcome is capture.CaptureOutcome.RECOVERY_REQUIRED
    assert result.journal is None
    assert "applied_boundary_offset_rows" in (result.journal_error or "")


@pytest.mark.parametrize(
    "runner_factory",
    [
        lambda digest: FakeRunner(digest, returncode=1, write_journal=False),
        lambda digest: FakeRunner(
            digest,
            returncode=1,
            status="failed",
            recovery="mystery",
        ),
        lambda digest: FakeRunner(
            digest,
            mutate_journal=lambda journal: journal.update(capture_engine_sha256="f" * 64),
        ),
    ],
)
def test_missing_or_untrustworthy_journal_fails_closed_to_recovery(
    tmp_path: Path,
    binding: Binding,
    runner_factory: Callable[[str], FakeRunner],
) -> None:
    runner = runner_factory(binding.worker_sha256)
    result = _adapter(tmp_path, binding, runner).run_attempt(capture.CaptureRequest(mode=capture.CaptureMode.PREVIEW))

    assert result.outcome is capture.CaptureOutcome.RECOVERY_REQUIRED
    assert result.recovery_required is True
    assert result.journal is None
    assert result.journal_error


def test_stop_requested_during_child_waits_for_that_attempt_then_blocks_next(
    tmp_path: Path,
    binding: Binding,
) -> None:
    runner = FakeRunner(binding.worker_sha256)
    adapter = _adapter(tmp_path, binding, runner)
    runner.during_run = adapter.request_stop

    active = adapter.run_attempt(capture.CaptureRequest(mode=capture.CaptureMode.PREVIEW))

    assert active.outcome is capture.CaptureOutcome.COMPLETE
    with pytest.raises(capture.CaptureStopped, match="between attempts"):
        adapter.run_attempt(capture.CaptureRequest(mode=capture.CaptureMode.PREVIEW))
    assert len(runner.calls) == 1

    adapter.clear_stop()
    runner.during_run = None
    resumed = adapter.run_attempt(capture.CaptureRequest(mode=capture.CaptureMode.PREVIEW))
    assert resumed.outcome is capture.CaptureOutcome.COMPLETE


@pytest.mark.parametrize(
    "request_factory",
    [
        lambda: capture.CaptureRequest(mode=capture.CaptureMode.PREVIEW, selected_slot=1),
        lambda: capture.CaptureRequest(mode=capture.CaptureMode.METER_ONLY),
        lambda: capture.CaptureRequest(mode=capture.CaptureMode.FULL, selected_slot=0),
        lambda: capture.CaptureRequest(mode=capture.CaptureMode.FULL, selected_slot=True),
        lambda: capture.CaptureRequest(mode=capture.CaptureMode.FULL, selected_slot=41),
    ],
)
def test_request_rejects_ambiguous_or_out_of_capacity_slots(request_factory: Callable[[], object]) -> None:
    with pytest.raises(ValueError):
        request_factory()


def test_default_runner_uses_argv_without_shell_and_isolates_child_signals(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    recorded: dict[str, object] = {}

    def fake_run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        recorded["argv"] = argv
        recorded.update(kwargs)
        return subprocess.CompletedProcess(argv, 0, "out", "err")

    monkeypatch.setattr(capture.subprocess, "run", fake_run)
    result = capture._run_subprocess(["python", "worker.py", "--live"], cwd=tmp_path)

    assert result.returncode == 0
    assert recorded["argv"] == ["python", "worker.py", "--live"]
    assert recorded["shell"] is False
    assert recorded["start_new_session"] is True
    assert recorded["capture_output"] is True
