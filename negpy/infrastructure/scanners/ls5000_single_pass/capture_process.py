"""Process-isolated bridge to NegPy's proven LS-5000 RGBI4x capture worker.

The package owns and integrity-pins the USB worker. This adapter gives the
application a narrow, fail-closed process boundary around it:

* every launch uses an argv list (never a shell command),
* both the worker and bundled replay plan are hash-pinned before launch,
* preview attempts expose the scanner's addressable candidates without a
  count hint, while meter/full attempts capture one explicit scanner slot, and
* a stop request is observed only between attempts.  An active child is never
  signalled or killed by this adapter.

The worker's ``--preview-only`` operation persists the roll preview and 0x8e
table before any frame binding.  Exposure-count labels are not part of this
interface: the adapter never emits ``--expected-frame-count``.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import threading
from dataclasses import dataclass
from enum import StrEnum
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Protocol, Sequence

from .bundle import (
    CANONICAL_MANIFEST_FILENAME,
    CAPTURE_BUNDLE_SHA256,
    CAPTURE_WORKER_SHA256,
    CaptureBundleIntegrityError,
    canonical_manifest_bytes,
    verify_capture_bundle,
)
from .plan import (
    CANONICAL_FINE_READ_BYTES,
    CANONICAL_FINE_READ_COUNT,
    CANONICAL_PLAN_FILENAME,
    CANONICAL_PLAN_SHA256,
    canonical_plan_bytes,
)


METER_READ_COUNT = 15
METER_CAPTURE_BYTES = 3_264_000
POWER_CYCLE_RECOVERY = "power-cycle scanner before another attempt"
CAPTURE_HELPER_FLAG = "--ls5000-capture-helper"
PACKAGED_WORKER_MODULE = (
    "negpy.infrastructure.scanners.ls5000_single_pass.worker"
)


class CaptureMode(StrEnum):
    """The safe operations currently exposed by the capture worker."""

    PREVIEW = "preview"
    METER_ONLY = "meter-only"
    FULL = "full"


class CaptureOutcome(StrEnum):
    """Application-level interpretation of a completed child process."""

    COMPLETE = "complete"
    SYNCHRONIZED_REFUSAL = "synchronized-refusal"
    RECOVERY_REQUIRED = "recovery-required"


class CaptureProcessError(RuntimeError):
    """The adapter refused before a trustworthy worker result was available."""


class CaptureIntegrityError(CaptureProcessError):
    """A pinned executable, plan, or manifest failed verification."""


class CaptureStopped(CaptureProcessError):
    """A stop request prevented the next attempt from launching."""


@dataclass(frozen=True)
class CaptureRequest:
    """One explicit scanner-addressable slot and capture mode.

    Roll exposure counts intentionally do not appear here.  A 36-exposure roll
    can contain a 37th image, and the preview UI may show blank or unusable tail
    slots up to the scanner's capacity.  Selection policy belongs above this
    process boundary.
    """

    mode: CaptureMode
    selected_slot: int | None = None
    boundary_offset_rows: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.mode, CaptureMode):
            raise TypeError("mode must be a CaptureMode")
        if isinstance(self.boundary_offset_rows, bool) or not isinstance(
            self.boundary_offset_rows, int
        ):
            raise TypeError("boundary offset must be an integer row count")
        if self.mode is CaptureMode.PREVIEW:
            if self.selected_slot is not None:
                raise ValueError("preview-only requests do not select a scanner slot")
            if self.boundary_offset_rows != 0:
                raise ValueError("preview boundary offset must be zero")
            return
        if isinstance(self.selected_slot, bool) or not isinstance(self.selected_slot, int) or not 1 <= self.selected_slot <= 40:
            raise ValueError("selected scanner slot must be an integer in 1..40")
        minimum_offset = 0 if self.selected_slot == 1 else -144
        if not minimum_offset <= self.boundary_offset_rows <= 144:
            raise ValueError(
                f"slot {self.selected_slot} boundary offset must be in "
                f"{minimum_offset}..144 rows"
            )


@dataclass(frozen=True)
class AttemptPaths:
    """All durable paths owned by one never-overwritten worker attempt."""

    directory: Path
    output: Path
    journal: Path
    plan: Path
    manifest: Path
    stdout: Path
    stderr: Path


@dataclass(frozen=True)
class CaptureAttemptResult:
    """Validated child result, including a conservative recovery decision."""

    outcome: CaptureOutcome
    request: CaptureRequest
    paths: AttemptPaths
    argv: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str
    journal: dict[str, Any] | None
    journal_error: str | None = None

    @property
    def recovery_required(self) -> bool:
        return self.outcome is CaptureOutcome.RECOVERY_REQUIRED


class ProcessRunner(Protocol):
    """Injectable child runner; tests can implement this without hardware."""

    def __call__(
        self,
        argv: Sequence[str],
        *,
        cwd: Path,
    ) -> subprocess.CompletedProcess[str]: ...


def _run_subprocess(
    argv: Sequence[str],
    *,
    cwd: Path,
) -> subprocess.CompletedProcess[str]:
    """Run the worker in its own process group and collect its text output.

    ``start_new_session`` keeps an application-level SIGINT/SIGTERM from being
    forwarded to the scanner child.  The parent may record a stop request, but
    this function waits for the current worker attempt to finish naturally.
    """

    return subprocess.run(
        list(argv),
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
        shell=False,
        start_new_session=True,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_exclusive(path: Path, payload: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


class CaptureProcessAdapter:
    """Launch one hash-pinned worker process per scanner attempt.

    Calls are serialized because there is one physical transport.  Calling
    :meth:`request_stop` while a child is active only marks the adapter; the
    active attempt finishes, and the next attempt raises :class:`CaptureStopped`.
    """

    def __init__(
        self,
        *,
        worker_path: Path,
        attempts_root: Path,
        expected_worker_sha256: str,
        manifest_path: Path | None = None,
        python_executable: str = sys.executable,
        runner: ProcessRunner = _run_subprocess,
        launcher: Sequence[str] | None = None,
        verify_worker_source: bool = True,
        expected_bundle_sha256: str | None = None,
        manifest_payload: bytes | None = None,
    ) -> None:
        if not _is_sha256(expected_worker_sha256):
            raise ValueError("expected worker SHA-256 is not a lowercase digest")
        self._worker_path = Path(worker_path).expanduser().resolve()
        self._attempts_root = Path(attempts_root).expanduser().resolve()
        self._manifest_path = (
            self._worker_path.with_name("replay-first-rgbi4-manifest.json")
            if manifest_path is None
            else Path(manifest_path).expanduser().resolve()
        )
        self._python_executable = str(python_executable)
        self._launcher = (
            (self._python_executable, str(self._worker_path))
            if launcher is None
            else tuple(str(item) for item in launcher)
        )
        if not self._launcher:
            raise ValueError("capture worker launcher cannot be empty")
        self._verify_worker_source = bool(verify_worker_source)
        self._expected_worker_sha256 = expected_worker_sha256
        if expected_bundle_sha256 is not None and not _is_sha256(
            expected_bundle_sha256
        ):
            raise ValueError("expected capture bundle SHA-256 is not a lowercase digest")
        self._expected_bundle_sha256 = expected_bundle_sha256
        self._manifest_payload = (
            None if manifest_payload is None else bytes(manifest_payload)
        )
        self._runner = runner
        self._stop_requested = threading.Event()
        self._attempt_lock = threading.Lock()

    @classmethod
    def packaged(
        cls,
        attempts_root: Path,
        *,
        runner: ProcessRunner = _run_subprocess,
    ) -> CaptureProcessAdapter:
        """Bind the adapter to NegPy's verified package-owned worker bundle.

        Source checkouts and installed wheels launch an isolated ``python -m``
        child and hash every scanner-facing source first.  Frozen builds
        relaunch the signed app executable with an internal helper flag; the
        helper dispatch occurs before desktop initialization.
        """

        frozen = bool(getattr(sys, "frozen", False))
        try:
            verify_capture_bundle(require_python_sources=not frozen)
            manifest_payload = canonical_manifest_bytes()
        except (CaptureBundleIntegrityError, OSError, ValueError) as error:
            raise CaptureIntegrityError(
                f"packaged capture bundle failed validation: {error}"
            ) from error
        spec = find_spec(PACKAGED_WORKER_MODULE)
        if spec is None or spec.origin is None:
            raise CaptureIntegrityError("packaged capture worker module is missing")
        worker_path = Path(spec.origin).resolve()
        launcher = (
            (sys.executable, CAPTURE_HELPER_FLAG)
            if frozen
            else (sys.executable, "-I", "-m", PACKAGED_WORKER_MODULE)
        )
        return cls(
            worker_path=worker_path,
            attempts_root=attempts_root,
            expected_worker_sha256=CAPTURE_WORKER_SHA256,
            manifest_path=worker_path.with_name(CANONICAL_MANIFEST_FILENAME),
            runner=runner,
            launcher=launcher,
            verify_worker_source=not frozen,
            expected_bundle_sha256=CAPTURE_BUNDLE_SHA256,
            manifest_payload=manifest_payload,
        )

    def request_stop(self) -> None:
        """Stop before the next attempt without touching an active child."""

        self._stop_requested.set()

    def clear_stop(self) -> None:
        """Allow launches again after the caller has acknowledged a stop."""

        self._stop_requested.clear()

    def run_attempt(self, request: CaptureRequest) -> CaptureAttemptResult:
        """Run and validate one worker attempt synchronously."""

        if not isinstance(request, CaptureRequest):
            raise TypeError("request must be a CaptureRequest")
        with self._attempt_lock:
            if self._stop_requested.is_set():
                raise CaptureStopped("capture stopped between attempts; no worker was launched")
            paths = self._prepare_attempt_paths(request)
            self._verify_worker()
            self._materialize_pinned_plan(paths.plan)
            self._materialize_pinned_manifest(paths.manifest)
            argv = self._build_argv(request, paths)
            # This is deliberately the final stop check.  Once runner() starts,
            # request_stop() cannot interrupt or signal the active worker.
            if self._stop_requested.is_set():
                raise CaptureStopped("capture stopped before worker launch")
            try:
                completed = self._runner(argv, cwd=paths.directory)
            except OSError as error:
                raise CaptureProcessError(f"could not launch capture worker: {error}") from error

            stdout = completed.stdout or ""
            stderr = completed.stderr or ""
            _write_exclusive(paths.stdout, stdout.encode("utf-8", errors="replace"))
            _write_exclusive(paths.stderr, stderr.encode("utf-8", errors="replace"))
            return self._interpret_result(
                request=request,
                paths=paths,
                argv=argv,
                returncode=completed.returncode,
                stdout=stdout,
                stderr=stderr,
            )

    def _prepare_attempt_paths(self, request: CaptureRequest) -> AttemptPaths:
        self._attempts_root.mkdir(parents=True, exist_ok=True)
        slot_suffix = "" if request.selected_slot is None else f"-slot{request.selected_slot:02d}"
        prefix = f"{request.mode.value}{slot_suffix}-"
        directory = Path(tempfile.mkdtemp(prefix=prefix, dir=self._attempts_root)).resolve()
        return AttemptPaths(
            directory=directory,
            output=directory / "capture.bin",
            journal=directory / "journal.json",
            plan=directory / CANONICAL_PLAN_FILENAME,
            manifest=directory / CANONICAL_MANIFEST_FILENAME,
            stdout=directory / "stdout.txt",
            stderr=directory / "stderr.txt",
        )

    def _verify_worker(self) -> None:
        if self._expected_bundle_sha256 is not None:
            try:
                actual_bundle = verify_capture_bundle(
                    require_python_sources=self._verify_worker_source
                )
            except (CaptureBundleIntegrityError, OSError, ValueError) as error:
                raise CaptureIntegrityError(
                    f"capture bundle failed validation before launch: {error}"
                ) from error
            if actual_bundle != self._expected_bundle_sha256:
                raise CaptureIntegrityError(
                    "capture bundle identity changed before launch"
                )
        if not self._verify_worker_source:
            return
        if not self._worker_path.is_file():
            raise CaptureIntegrityError(f"capture worker is not a regular file: {self._worker_path}")
        actual = _sha256_file(self._worker_path)
        if actual != self._expected_worker_sha256:
            raise CaptureIntegrityError(f"capture worker SHA-256 mismatch: expected {self._expected_worker_sha256}, got {actual}")

    def _materialize_pinned_plan(self, destination: Path) -> None:
        try:
            payload = canonical_plan_bytes()
        except (OSError, ValueError) as error:
            raise CaptureIntegrityError(f"bundled capture plan failed validation: {error}") from error
        if hashlib.sha256(payload).hexdigest() != CANONICAL_PLAN_SHA256:
            raise CaptureIntegrityError("bundled capture plan SHA-256 changed after validation")
        _write_exclusive(destination, payload)
        actual = _sha256_file(destination)
        if actual != CANONICAL_PLAN_SHA256:
            raise CaptureIntegrityError(f"materialized plan SHA-256 mismatch: {actual}")

    def _materialize_pinned_manifest(self, destination: Path) -> None:
        try:
            payload = (
                self._manifest_path.read_bytes()
                if self._manifest_payload is None
                else self._manifest_payload
            )
            manifest = json.loads(payload)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise CaptureIntegrityError(f"capture manifest could not be read: {error}") from error
        if not isinstance(manifest, dict):
            raise CaptureIntegrityError("capture manifest must be a JSON object")
        if manifest.get("plan_sha256") != CANONICAL_PLAN_SHA256:
            raise CaptureIntegrityError("capture manifest is not bound to the packaged canonical plan")
        _write_exclusive(destination, payload)

    def _build_argv(self, request: CaptureRequest, paths: AttemptPaths) -> tuple[str, ...]:
        argv = [
            *self._launcher,
            "--plan",
            str(paths.plan),
            "--manifest",
            str(paths.manifest),
            "--output",
            str(paths.output),
            "--journal",
            str(paths.journal),
            "--boundary-offset-rows",
            str(request.boundary_offset_rows),
            "--live",
        ]
        if request.mode is CaptureMode.PREVIEW:
            argv.append("--preview-only")
        elif request.mode is CaptureMode.METER_ONLY:
            argv.extend(("--frame", str(request.selected_slot), "--meter-only"))
        else:
            argv.extend(
                (
                    "--frame",
                    str(request.selected_slot),
                    "--reads",
                    str(CANONICAL_FINE_READ_COUNT),
                    "--confirm-full-capture",
                )
            )
        if "--expected-frame-count" in argv:
            raise AssertionError("exposure-count hints must never cross the capture boundary")
        return tuple(argv)

    def _interpret_result(
        self,
        *,
        request: CaptureRequest,
        paths: AttemptPaths,
        argv: tuple[str, ...],
        returncode: int,
        stdout: str,
        stderr: str,
    ) -> CaptureAttemptResult:
        try:
            journal = self._load_and_validate_journal(paths, request, returncode)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
            # A child was launched but did not leave trustworthy synchronization
            # evidence.  Conservatively require recovery instead of guessing.
            return CaptureAttemptResult(
                outcome=CaptureOutcome.RECOVERY_REQUIRED,
                request=request,
                paths=paths,
                argv=argv,
                returncode=returncode,
                stdout=stdout,
                stderr=stderr,
                journal=None,
                journal_error=str(error),
            )

        if returncode == 0:
            outcome = CaptureOutcome.COMPLETE
        elif journal["recovery_required"] == "none":
            outcome = CaptureOutcome.SYNCHRONIZED_REFUSAL
        else:
            outcome = CaptureOutcome.RECOVERY_REQUIRED
        return CaptureAttemptResult(
            outcome=outcome,
            request=request,
            paths=paths,
            argv=argv,
            returncode=returncode,
            stdout=stdout,
            stderr=stderr,
            journal=journal,
        )

    def _load_and_validate_journal(
        self,
        paths: AttemptPaths,
        request: CaptureRequest,
        returncode: int,
    ) -> dict[str, Any]:
        payload = json.loads(paths.journal.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("worker journal must be a JSON object")

        mode = {
            CaptureMode.PREVIEW: "preview-only",
            CaptureMode.METER_ONLY: "meter-only",
            CaptureMode.FULL: "full",
        }[request.mode]
        expected_reads = {
            CaptureMode.PREVIEW: 0,
            CaptureMode.METER_ONLY: METER_READ_COUNT,
            CaptureMode.FULL: CANONICAL_FINE_READ_COUNT,
        }[request.mode]
        expected_bytes = {
            CaptureMode.PREVIEW: 0,
            CaptureMode.METER_ONLY: METER_CAPTURE_BYTES,
            CaptureMode.FULL: CANONICAL_FINE_READ_COUNT * CANONICAL_FINE_READ_BYTES,
        }[request.mode]
        invariants: dict[str, object] = {
            "plan_sha256": CANONICAL_PLAN_SHA256,
            "capture_engine_sha256": self._expected_worker_sha256,
            "output": str(paths.output.resolve()),
            "capture_mode": mode,
            "requested_frame": request.selected_slot,
            "expected_frame_count": None,
            "expected_reads": expected_reads,
            "expected_bytes": expected_bytes,
            "requested_boundary_offset_rows": request.boundary_offset_rows,
        }
        if self._expected_bundle_sha256 is not None:
            invariants["capture_bundle_sha256"] = self._expected_bundle_sha256
        for key, expected in invariants.items():
            if payload.get(key) != expected:
                raise ValueError(f"worker journal {key}={payload.get(key)!r}, expected {expected!r}")

        status = payload.get("status")
        recovery = payload.get("recovery_required")
        if returncode == 0:
            if status != "complete":
                raise ValueError(f"worker exited zero with journal status {status!r}")
            if recovery not in (None, "none"):
                raise ValueError(f"completed worker requested recovery: {recovery!r}")
            if payload.get("completed_reads") != expected_reads or payload.get("completed_bytes") != expected_bytes:
                raise ValueError("completed worker journal has incomplete read or byte counts")
            if payload.get("disk_bytes") != expected_bytes:
                raise ValueError("completed worker journal has the wrong on-disk byte count")
            if payload.get("unit_released") is not True:
                raise ValueError("completed worker did not record unit release")
            if not _is_sha256(payload.get("output_sha256")):
                raise ValueError("completed worker output SHA-256 is missing or malformed")
            if not paths.output.is_file() or paths.output.stat().st_size != expected_bytes:
                raise ValueError("completed worker output file is missing or has the wrong size")
            if request.mode is not CaptureMode.PREVIEW:
                if payload.get("applied_boundary_offset_rows") != request.boundary_offset_rows:
                    raise ValueError(
                        "worker journal applied_boundary_offset_rows does not "
                        "match the requested boundary offset"
                    )
                resolved_row = payload.get("resolved_lookup_row")
                if isinstance(resolved_row, bool) or not isinstance(resolved_row, int) or resolved_row < 0:
                    raise ValueError("worker journal has no valid resolved_lookup_row")
                resolved_origin = payload.get("resolved_native_origin")
                if (
                    isinstance(resolved_origin, bool)
                    or not isinstance(resolved_origin, int)
                    or resolved_origin < 0
                ):
                    raise ValueError("worker journal has no valid resolved_native_origin")
        else:
            if status not in ("failed", "interrupted"):
                raise ValueError(f"failed worker has unexpected journal status {status!r}")
            if recovery not in ("none", POWER_CYCLE_RECOVERY):
                raise ValueError(f"failed worker has unknown recovery state {recovery!r}")
        return payload


__all__ = [
    "CaptureAttemptResult",
    "CAPTURE_HELPER_FLAG",
    "CaptureIntegrityError",
    "CaptureMode",
    "CaptureOutcome",
    "CaptureProcessAdapter",
    "CaptureProcessError",
    "CaptureRequest",
    "CaptureStopped",
]
