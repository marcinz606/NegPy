"""Process-isolated bridge to NegPy's proven LS-5000 RGBI4x capture worker.

The package owns and integrity-pins the USB worker. This adapter gives the
application a narrow, fail-closed process boundary around it:

* every launch uses an argv list (never a shell command),
* both the worker and bundled replay plan are hash-pinned before launch,
* preview attempts expose the scanner's addressable candidates without a
  count hint, while meter attempts capture one explicit scanner slot,
* selected full frames share one child/reservation and cross a durable
  frame-ready/parent-ACK boundary before the next frame starts, and
* a stop request is observed only at a safe frame boundary.  An active child
  is never signalled or killed by this adapter.

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
import time
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
from .continuation_plan import (
    CANONICAL_CONTINUATION_PLAN_FILENAME,
    CANONICAL_CONTINUATION_PLAN_SHA256,
    canonical_continuation_plan_bytes,
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


class BatchAckAction(StrEnum):
    """Parent decision written only after one frame is durably finalized."""

    CONTINUE = "continue"
    STOP = "stop"


class CaptureProcessError(RuntimeError):
    """The adapter refused before a trustworthy worker result was available."""


class CaptureBatchProcessError(CaptureProcessError):
    """A spawned batch failed after preserving its partial frame/release state."""

    def __init__(
        self,
        message: str,
        *,
        outcome: CaptureOutcome,
        paths: BatchSessionPaths,
        frames: Sequence[CaptureAttemptResult],
        returncode: int,
        session_journal: dict[str, Any] | None,
    ) -> None:
        super().__init__(message)
        self.outcome = outcome
        self.paths = paths
        self.frames = tuple(frames)
        self.returncode = returncode
        self.session_journal = session_journal

    @property
    def recovery_required(self) -> bool:
        return self.outcome is CaptureOutcome.RECOVERY_REQUIRED


class CaptureIntegrityError(CaptureProcessError):
    """A pinned executable, plan, or manifest failed verification."""


class CaptureStopped(CaptureProcessError):
    """A stop request prevented the next attempt from launching."""


class _BatchFrameRefused(CaptureProcessError):
    """A bad frame handoff was safely answered with a bound STOP ACK."""

    def __init__(self, message: str, *, slot: int) -> None:
        super().__init__(message)
        self.slot = slot


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
class CaptureBatchRequest:
    """One ordered set of full frames for a future shared USB reservation."""

    frames: tuple[CaptureRequest, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.frames, tuple):
            raise TypeError("batch frames must be an immutable tuple")
        if not self.frames:
            raise ValueError("batch must contain at least one frame")
        if not all(isinstance(frame, CaptureRequest) for frame in self.frames):
            raise TypeError("every batch frame must be a CaptureRequest")
        if any(frame.mode is not CaptureMode.FULL for frame in self.frames):
            raise ValueError("batch sessions currently support full captures only")
        slots = self.selected_slots
        if tuple(sorted(set(slots))) != slots:
            raise ValueError(
                "batch scanner slots must be unique and strictly increasing"
            )

    @property
    def selected_slots(self) -> tuple[int, ...]:
        """Return statically validated scanner slots without optional typing."""

        return tuple(
            frame.selected_slot
            for frame in self.frames
            if frame.selected_slot is not None
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
class BatchSessionPaths:
    """Durable, never-overwritten inputs for one batch child."""

    directory: Path
    job: Path
    first_plan: Path
    continuation_plan: Path
    manifest: Path
    session_journal: Path
    stdout: Path
    stderr: Path


@dataclass(frozen=True)
class PreparedCaptureBatch:
    """Hardware-free description of exactly one worker process."""

    request: CaptureBatchRequest
    paths: BatchSessionPaths
    argv: tuple[str, ...]
    job_sha256: str
    session_id: str


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
    batch_session_id: str | None = None
    batch_frame_index: int | None = None
    batch_frame_total: int | None = None
    batch_selected_slots: tuple[int, ...] = ()

    @property
    def recovery_required(self) -> bool:
        return self.outcome is CaptureOutcome.RECOVERY_REQUIRED


@dataclass(frozen=True)
class CaptureBatchResult:
    """One child process, its finalized frames, and final release receipt."""

    outcome: CaptureOutcome
    request: CaptureBatchRequest
    paths: BatchSessionPaths
    frames: tuple[CaptureAttemptResult, ...]
    returncode: int
    stopped: bool
    session_journal: dict[str, Any]
    stdout: str
    stderr: str


class ProcessRunner(Protocol):
    """Injectable child runner; tests can implement this without hardware."""

    def __call__(
        self,
        argv: Sequence[str],
        *,
        cwd: Path,
    ) -> subprocess.CompletedProcess[str]: ...


class RunningBatchProcess(Protocol):
    """The non-signalled subset of ``subprocess.Popen`` used by the adapter."""

    def poll(self) -> int | None: ...

    def wait(self, timeout: float | None = None) -> int: ...


class BatchProcessSpawner(Protocol):
    def __call__(
        self,
        argv: Sequence[str],
        *,
        cwd: Path,
        stdout: Any,
        stderr: Any,
    ) -> RunningBatchProcess: ...


class BatchFrameHandler(Protocol):
    def __call__(self, result: CaptureAttemptResult) -> BatchAckAction: ...


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


def _spawn_batch_subprocess(
    argv: Sequence[str],
    *,
    cwd: Path,
    stdout: Any,
    stderr: Any,
) -> RunningBatchProcess:
    """Start one isolated child without exposing any signal/kill seam."""

    return subprocess.Popen(
        list(argv),
        cwd=cwd,
        stdout=stdout,
        stderr=stderr,
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


def _publish_exclusive(path: Path, payload: bytes) -> None:
    """Atomically publish a complete never-overwritten handshake file."""

    temporary = path.with_suffix(path.suffix + ".tmp")
    _write_exclusive(temporary, payload)
    try:
        # Linking a fully flushed inode makes the final name visible in one
        # step and, unlike replace(), refuses a stale/replayed destination.
        os.link(temporary, path)
        try:
            directory = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        except OSError:
            # The ACK is already atomically visible and the live child may
            # consume it immediately.  Reporting failure after publication
            # could allow an unobserved next frame, so durability sync is
            # best-effort at this point.
            pass
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            if not path.exists():
                raise


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
        batch_spawner: BatchProcessSpawner = _spawn_batch_subprocess,
        batch_poll_seconds: float = 0.1,
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
        if batch_poll_seconds < 0:
            raise ValueError("batch_poll_seconds cannot be negative")
        self._batch_spawner = batch_spawner
        self._batch_poll_seconds = float(batch_poll_seconds)
        self._stop_requested = threading.Event()
        self._stop_gate = threading.Lock()
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

        with self._stop_gate:
            self._stop_requested.set()

    def clear_stop(self) -> None:
        """Allow launches again after the caller has acknowledged a stop."""

        with self._stop_gate:
            self._stop_requested.clear()

    def prepare_batch_session(
        self,
        request: CaptureBatchRequest,
    ) -> PreparedCaptureBatch:
        """Materialize one child/session contract without launching hardware."""

        if not isinstance(request, CaptureBatchRequest):
            raise TypeError("request must be a CaptureBatchRequest")
        with self._attempt_lock:
            if self._stop_requested.is_set():
                raise CaptureStopped(
                    "capture stopped between sessions; no batch was prepared"
                )
            return self._prepare_batch_session_locked(request)

    def _prepare_batch_session_locked(
        self,
        request: CaptureBatchRequest,
    ) -> PreparedCaptureBatch:
        paths = self._prepare_batch_session_paths(request)
        self._verify_worker()
        self._materialize_pinned_plan(paths.first_plan)
        self._materialize_pinned_continuation_plan(paths.continuation_plan)
        self._materialize_pinned_manifest(paths.manifest)
        session_id = paths.directory.name
        payload = self._batch_job_bytes(request, session_id=session_id)
        _write_exclusive(paths.job, payload)
        argv = self._build_batch_argv(paths)
        return PreparedCaptureBatch(
            request=request,
            paths=paths,
            argv=argv,
            job_sha256=hashlib.sha256(payload).hexdigest(),
            session_id=session_id,
        )

    def run_batch_session(
        self,
        request: CaptureBatchRequest,
        *,
        frame_handler: BatchFrameHandler,
    ) -> CaptureBatchResult:
        """Run one child and ACK only frames the parent finished consuming.

        ``frame_handler`` is the parent-owned finalization boundary.  It must
        validate, decode, promote, and optionally delete the scratch stream
        before returning.  Only then is an ACK written, which is the child's
        permission to begin the next frame.  A stop request changes that ACK
        to ``stop``; it never signals the active scanner child.
        """

        if not isinstance(request, CaptureBatchRequest):
            raise TypeError("request must be a CaptureBatchRequest")
        if not callable(frame_handler):
            raise TypeError("frame_handler must be callable")
        with self._attempt_lock:
            if self._stop_requested.is_set():
                raise CaptureStopped(
                    "capture stopped between sessions; no batch was launched"
                )
            prepared = self._prepare_batch_session_locked(request)
            if self._stop_requested.is_set():
                raise CaptureStopped("capture stopped before batch worker launch")
            return self._run_prepared_batch(prepared, frame_handler)

    def _run_prepared_batch(
        self,
        prepared: PreparedCaptureBatch,
        frame_handler: BatchFrameHandler,
    ) -> CaptureBatchResult:
        paths = prepared.paths
        handled: list[CaptureAttemptResult] = []
        stopped = False
        stopped_unhandled_slot: int | None = None
        handler_error: BaseException | None = None
        monitor_error: BaseException | None = None
        process: RunningBatchProcess | None = None
        returncode: int | None = None
        try:
            with paths.stdout.open("xb") as stdout_handle, paths.stderr.open(
                "xb"
            ) as stderr_handle:
                # Linearize the final stop check with process creation.  If a
                # stop wins this lock no child launches; if Popen wins, that
                # stop applies at the first durable frame boundary.
                with self._stop_gate:
                    if self._stop_requested.is_set():
                        raise CaptureStopped(
                            "capture stopped before batch worker launch"
                        )
                    process = self._batch_spawner(
                        prepared.argv,
                        cwd=paths.directory,
                        stdout=stdout_handle,
                        stderr=stderr_handle,
                    )
                try:
                    for frame_index, frame_request in enumerate(
                        prepared.request.frames,
                        start=1,
                    ):
                        frame_result = self._wait_for_batch_frame(
                            process,
                            prepared,
                            frame_request,
                            frame_index=frame_index,
                        )
                        try:
                            action = frame_handler(frame_result)
                            if not isinstance(action, BatchAckAction):
                                raise TypeError(
                                    "frame_handler must return BatchAckAction"
                                )
                        except BaseException as error:
                            handler_error = error
                            action = BatchAckAction.STOP
                        handled.append(frame_result)
                        # Linearize Stop against publishing CONTINUE.  A stop
                        # that wins this lock changes the current boundary to
                        # STOP; one arriving after publication applies to the
                        # newly active frame instead.
                        with self._stop_gate:
                            if self._stop_requested.is_set():
                                action = BatchAckAction.STOP
                            self._write_batch_ack(
                                frame_result,
                                prepared,
                                action=action,
                            )
                        if action is BatchAckAction.STOP:
                            stopped = True
                            break
                except BaseException as error:
                    monitor_error = error
                    if isinstance(error, _BatchFrameRefused):
                        stopped = True
                        stopped_unhandled_slot = error.slot
                finally:
                    # Never signal or abandon a child that may own the USB
                    # reservation.  With no valid ACK it will time out, clean
                    # up, and release; the parent must remain here to observe
                    # that receipt.
                    returncode, wait_error = self._wait_for_batch_exit(process)
                    if monitor_error is None:
                        monitor_error = wait_error
        except OSError as error:
            if process is None:
                raise CaptureProcessError(
                    f"could not launch batch capture worker: {error}"
                ) from error
            monitor_error = monitor_error or error

        if process is None or returncode is None:
            raise CaptureProcessError(
                "batch capture worker did not leave a completed child process"
            ) from monitor_error

        try:
            session_journal = self._load_and_validate_batch_session_journal(
                prepared,
                returncode=returncode,
                handled=handled,
                stopped=stopped,
                stopped_unhandled_slot=stopped_unhandled_slot,
            )
        except BaseException as error:
            cause = error if monitor_error is None else monitor_error
            raise CaptureBatchProcessError(
                "batch child finished without a trustworthy release receipt: "
                f"{error}",
                outcome=CaptureOutcome.RECOVERY_REQUIRED,
                paths=paths,
                frames=handled,
                returncode=returncode,
                session_journal=None,
            ) from cause
        receipt_outcome = (
            CaptureOutcome.SYNCHRONIZED_REFUSAL
            if session_journal.get("recovery_required") == "none"
            else CaptureOutcome.RECOVERY_REQUIRED
        )
        outcome = (
            CaptureOutcome.COMPLETE if returncode == 0 else receipt_outcome
        )
        try:
            stdout = paths.stdout.read_text(encoding="utf-8", errors="replace")
            stderr = paths.stderr.read_text(encoding="utf-8", errors="replace")
        except OSError as error:
            stdout = ""
            stderr = ""
            if monitor_error is None:
                monitor_error = error
        if monitor_error is not None:
            if isinstance(monitor_error, (KeyboardInterrupt, SystemExit)):
                raise monitor_error
            raise CaptureBatchProcessError(
                "batch adapter failed after launch; child cleanup and release "
                f"completed before returning: {monitor_error}",
                outcome=receipt_outcome,
                paths=paths,
                frames=handled,
                returncode=returncode,
                session_journal=session_journal,
            ) from monitor_error
        if handler_error is not None:
            raise CaptureBatchProcessError(
                f"batch frame finalization failed after safe stop: {handler_error}",
                outcome=receipt_outcome,
                paths=paths,
                frames=handled,
                returncode=returncode,
                session_journal=session_journal,
            ) from handler_error
        return CaptureBatchResult(
            outcome=outcome,
            request=prepared.request,
            paths=paths,
            frames=tuple(handled),
            returncode=returncode,
            stopped=stopped,
            session_journal=session_journal,
            stdout=stdout,
            stderr=stderr,
        )

    def _wait_for_batch_exit(
        self,
        process: RunningBatchProcess,
    ) -> tuple[int, BaseException | None]:
        """Defer interruption until the scanner child has cleaned up."""

        deferred: BaseException | None = None
        while True:
            try:
                return process.wait(), deferred
            except BaseException as error:
                if deferred is None:
                    deferred = error
                try:
                    returncode = process.poll()
                except BaseException:
                    returncode = None
                if returncode is not None:
                    return returncode, deferred
                if self._batch_poll_seconds:
                    time.sleep(self._batch_poll_seconds)

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

    def _prepare_batch_session_paths(
        self,
        request: CaptureBatchRequest,
    ) -> BatchSessionPaths:
        self._attempts_root.mkdir(parents=True, exist_ok=True)
        first, last = request.selected_slots[0], request.selected_slots[-1]
        prefix = f"batch-slot{first:02d}-slot{last:02d}-"
        directory = Path(
            tempfile.mkdtemp(prefix=prefix, dir=self._attempts_root)
        ).resolve()
        return BatchSessionPaths(
            directory=directory,
            job=directory / "batch-job.json",
            first_plan=directory / CANONICAL_PLAN_FILENAME,
            continuation_plan=(
                directory / CANONICAL_CONTINUATION_PLAN_FILENAME
            ),
            manifest=directory / CANONICAL_MANIFEST_FILENAME,
            session_journal=directory / "session-journal.json",
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

    def _materialize_pinned_continuation_plan(self, destination: Path) -> None:
        try:
            payload = canonical_continuation_plan_bytes()
        except (OSError, ValueError) as error:
            raise CaptureIntegrityError(
                f"bundled continuation plan failed validation: {error}"
            ) from error
        if hashlib.sha256(payload).hexdigest() != CANONICAL_CONTINUATION_PLAN_SHA256:
            raise CaptureIntegrityError(
                "bundled continuation plan SHA-256 changed after validation"
            )
        _write_exclusive(destination, payload)
        actual = _sha256_file(destination)
        if actual != CANONICAL_CONTINUATION_PLAN_SHA256:
            raise CaptureIntegrityError(
                f"materialized continuation plan SHA-256 mismatch: {actual}"
            )

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

    def _batch_job_bytes(
        self,
        request: CaptureBatchRequest,
        *,
        session_id: str,
    ) -> bytes:
        frames = [
            {
                "ack": f"frame-{frame.selected_slot:03d}/parent-ack.json",
                "boundary_offset_rows": frame.boundary_offset_rows,
                "journal": f"frame-{frame.selected_slot:03d}/journal.json",
                "output": f"frame-{frame.selected_slot:03d}/capture.bin",
                "slot": frame.selected_slot,
            }
            for frame in request.frames
        ]
        job = {
            "apply_all_boundary_offsets_before_first_frame": True,
            "capture_plan_sha256": CANONICAL_PLAN_SHA256,
            "continuation_plan_sha256": CANONICAL_CONTINUATION_PLAN_SHA256,
            "frames": frames,
            "parent_ack_required_after_every_frame": True,
            "release_once_after_last_frame": True,
            "schema_version": 1,
            "session_id": session_id,
            "session_contract": "one-process-one-reservation",
        }
        return (
            json.dumps(job, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode("utf-8")

    def _build_batch_argv(self, paths: BatchSessionPaths) -> tuple[str, ...]:
        return (
            *self._launcher,
            "--batch-job",
            str(paths.job),
            "--plan",
            str(paths.first_plan),
            "--continuation-plan",
            str(paths.continuation_plan),
            "--manifest",
            str(paths.manifest),
            "--session-journal",
            str(paths.session_journal),
            "--live",
        )

    def _batch_frame_paths(
        self,
        prepared: PreparedCaptureBatch,
        request: CaptureRequest,
    ) -> AttemptPaths:
        slot = request.selected_slot
        if slot is None:
            raise AssertionError("validated batch frame has no selected slot")
        directory = prepared.paths.directory / f"frame-{slot:03d}"
        return AttemptPaths(
            directory=directory,
            output=directory / "capture.bin",
            journal=directory / "journal.json",
            plan=prepared.paths.first_plan,
            manifest=prepared.paths.manifest,
            stdout=prepared.paths.stdout,
            stderr=prepared.paths.stderr,
        )

    def _wait_for_batch_frame(
        self,
        process: RunningBatchProcess,
        prepared: PreparedCaptureBatch,
        request: CaptureRequest,
        *,
        frame_index: int,
    ) -> CaptureAttemptResult:
        paths = self._batch_frame_paths(prepared, request)
        while True:
            if paths.journal.is_file():
                try:
                    payload = json.loads(paths.journal.read_text(encoding="utf-8"))
                except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                    payload = None
                if isinstance(payload, dict) and payload.get("status") == "frame-complete":
                    try:
                        return self._validate_batch_frame_result(
                            prepared,
                            request,
                            paths,
                            payload,
                            frame_index=frame_index,
                        )
                    except Exception as error:
                        self._write_identifiable_batch_stop(
                            prepared,
                            request,
                            paths,
                            payload,
                            frame_index=frame_index,
                        )
                        slot = request.selected_slot
                        if slot is None:
                            raise AssertionError(
                                "validated batch request lost its slot"
                            )
                        raise _BatchFrameRefused(
                            f"batch frame {frame_index} was refused and safely "
                            f"stopped: {error}",
                            slot=slot,
                        ) from error
            returncode = process.poll()
            if returncode is not None:
                raise CaptureProcessError(
                    f"batch worker exited {returncode} before frame "
                    f"{frame_index} became ready for parent finalization"
                )
            if self._batch_poll_seconds:
                time.sleep(self._batch_poll_seconds)

    def _validate_batch_frame_result(
        self,
        prepared: PreparedCaptureBatch,
        request: CaptureRequest,
        paths: AttemptPaths,
        payload: dict[str, Any],
        *,
        frame_index: int,
    ) -> CaptureAttemptResult:
        expected_bytes = CANONICAL_FINE_READ_COUNT * CANONICAL_FINE_READ_BYTES
        selected_slots = prepared.request.selected_slots
        expected_batch = {
            "frame_index": frame_index,
            "frame_total": len(selected_slots),
            "selected_slots": list(selected_slots),
            "session_id": prepared.session_id,
        }
        invariants: dict[str, object] = {
            "status": "frame-complete",
            "frame_complete": True,
            "session_reservation_retained": True,
            "unit_released": False,
            "batch_session": expected_batch,
            "capture_mode": "full",
            "requested_frame": request.selected_slot,
            "requested_boundary_offset_rows": request.boundary_offset_rows,
            "expected_reads": CANONICAL_FINE_READ_COUNT,
            "completed_reads": CANONICAL_FINE_READ_COUNT,
            "expected_bytes": expected_bytes,
            "completed_bytes": expected_bytes,
            "disk_bytes": expected_bytes,
            "output": str(paths.output.resolve()),
            "plan_sha256": CANONICAL_PLAN_SHA256,
            "continuation_plan_sha256": CANONICAL_CONTINUATION_PLAN_SHA256,
            "capture_engine_sha256": self._expected_worker_sha256,
        }
        if self._expected_bundle_sha256 is not None:
            invariants["capture_bundle_sha256"] = self._expected_bundle_sha256
        for key, expected in invariants.items():
            if payload.get(key) != expected:
                raise CaptureProcessError(
                    f"batch frame {frame_index} journal {key}="
                    f"{payload.get(key)!r}, expected {expected!r}"
                )
        if payload.get("recovery_required") not in (None, "none"):
            raise CaptureProcessError(
                f"batch frame {frame_index} requested scanner recovery"
            )
        nonce = payload.get("ack_nonce")
        if (
            not isinstance(nonce, str)
            or not 1 <= len(nonce) <= 128
            or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for character in nonce)
        ):
            raise CaptureProcessError(
                f"batch frame {frame_index} has no valid parent ACK nonce"
            )
        if not _is_sha256(payload.get("output_sha256")):
            raise CaptureProcessError(
                f"batch frame {frame_index} output SHA-256 is malformed"
            )
        if not paths.output.is_file() or paths.output.stat().st_size != expected_bytes:
            raise CaptureProcessError(
                f"batch frame {frame_index} output is missing or incomplete"
            )
        return CaptureAttemptResult(
            outcome=CaptureOutcome.COMPLETE,
            request=request,
            paths=paths,
            argv=prepared.argv,
            returncode=0,
            stdout="",
            stderr="",
            journal=payload,
            batch_session_id=prepared.session_id,
            batch_frame_index=frame_index,
            batch_frame_total=len(selected_slots),
            batch_selected_slots=selected_slots,
        )

    def _write_batch_ack(
        self,
        result: CaptureAttemptResult,
        prepared: PreparedCaptureBatch,
        *,
        action: BatchAckAction,
    ) -> None:
        if result.journal is None:
            raise CaptureProcessError("cannot ACK a batch frame without a journal")
        slot = result.request.selected_slot
        if slot is None or result.batch_frame_index is None:
            raise CaptureProcessError("cannot ACK a batch frame without identity")
        ack_path = result.paths.directory / "parent-ack.json"
        payload = {
            "ack_nonce": result.journal["ack_nonce"],
            "action": action.value,
            "frame_index": result.batch_frame_index,
            "schema_version": 1,
            "session_id": prepared.session_id,
            "slot": slot,
        }
        _publish_exclusive(
            ack_path,
            (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode(
                "utf-8"
            ),
        )

    def _write_identifiable_batch_stop(
        self,
        prepared: PreparedCaptureBatch,
        request: CaptureRequest,
        paths: AttemptPaths,
        payload: dict[str, Any],
        *,
        frame_index: int,
    ) -> None:
        """Stop a bad frame promptly when its handshake identity is exact."""

        slot = request.selected_slot
        if slot is None:
            raise CaptureProcessError("batch STOP has no selected slot")
        expected_batch = {
            "frame_index": frame_index,
            "frame_total": len(prepared.request.frames),
            "selected_slots": list(prepared.request.selected_slots),
            "session_id": prepared.session_id,
        }
        if payload.get("status") != "frame-complete":
            raise CaptureProcessError("batch STOP source is not frame-complete")
        if payload.get("batch_session") != expected_batch:
            raise CaptureProcessError(
                "bad batch frame has no trustworthy session identity for STOP"
            )
        if payload.get("requested_frame") != slot:
            raise CaptureProcessError(
                "bad batch frame has no trustworthy slot identity for STOP"
            )
        nonce = payload.get("ack_nonce")
        if (
            not isinstance(nonce, str)
            or not 1 <= len(nonce) <= 128
            or any(
                character
                not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
                for character in nonce
            )
        ):
            raise CaptureProcessError(
                "bad batch frame has no trustworthy nonce for STOP"
            )
        ack = {
            "ack_nonce": nonce,
            "action": BatchAckAction.STOP.value,
            "frame_index": frame_index,
            "schema_version": 1,
            "session_id": prepared.session_id,
            "slot": slot,
        }
        _publish_exclusive(
            paths.directory / "parent-ack.json",
            (json.dumps(ack, sort_keys=True, separators=(",", ":")) + "\n").encode(
                "utf-8"
            ),
        )

    def _load_and_validate_batch_session_journal(
        self,
        prepared: PreparedCaptureBatch,
        *,
        returncode: int,
        handled: Sequence[CaptureAttemptResult],
        stopped: bool,
        stopped_unhandled_slot: int | None = None,
    ) -> dict[str, Any]:
        try:
            payload = json.loads(
                prepared.paths.session_journal.read_text(encoding="utf-8")
            )
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise CaptureProcessError(
                f"batch worker left no trustworthy session journal: {error}"
            ) from error
        if not isinstance(payload, dict):
            raise CaptureProcessError("batch session journal must be an object")
        expected_status = "stopped" if stopped else "complete"
        expected_completed = [result.request.selected_slot for result in handled]
        if stopped_unhandled_slot is not None:
            expected_completed.append(stopped_unhandled_slot)
        invariants: dict[str, object] = {
            "session_id": prepared.session_id,
            "selected_slots": list(prepared.request.selected_slots),
            "batch_job_sha256": prepared.job_sha256,
            "capture_engine_sha256": self._expected_worker_sha256,
            "capture_bundle_sha256": (
                self._expected_bundle_sha256 or CAPTURE_BUNDLE_SHA256
            ),
            "plan_sha256": CANONICAL_PLAN_SHA256,
            "continuation_plan_sha256": CANONICAL_CONTINUATION_PLAN_SHA256,
        }
        for key, expected in invariants.items():
            if payload.get(key) != expected:
                raise CaptureProcessError(
                    f"batch session journal {key}={payload.get(key)!r}, "
                    f"expected {expected!r}"
                )
        completed = payload.get("completed_slots")
        selected = list(prepared.request.selected_slots)
        if (
            not isinstance(completed, list)
            or any(
                isinstance(slot, bool) or not isinstance(slot, int)
                for slot in completed
            )
            or completed != selected[: len(completed)]
            or completed != expected_completed
        ):
            raise CaptureProcessError(
                f"batch session journal completed_slots={completed!r} does not "
                f"match the observed frame prefix {expected_completed!r}"
            )
        status = payload.get("status")
        release_attempts = payload.get("unit_release_attempts")
        unit_released = payload.get("unit_released")
        reservation_acquired = payload.get("reservation_acquired")
        recovery = payload.get("recovery_required")
        if recovery not in ("none", POWER_CYCLE_RECOVERY):
            raise CaptureProcessError(
                f"batch session journal has unknown recovery state {recovery!r}"
            )
        if not isinstance(reservation_acquired, bool):
            raise CaptureProcessError(
                "batch session journal has no reservation-acquired receipt"
            )
        if returncode == 0:
            if status != expected_status or completed != expected_completed:
                raise CaptureProcessError(
                    "completed batch session has the wrong status or frame prefix"
            )
            if (
                reservation_acquired is not True
                or release_attempts != 1
                or unit_released is not True
                or recovery != "none"
            ):
                raise CaptureProcessError(
                    "completed batch session has no successful single-release receipt"
                )
        else:
            if status not in ("failed", "interrupted"):
                raise CaptureProcessError(
                    f"failed batch session has unexpected status {status!r}"
                )
            if (
                isinstance(release_attempts, bool)
                or release_attempts not in (0, 1)
            ):
                raise CaptureProcessError(
                    "failed batch session has an invalid release-attempt count"
                )
            if reservation_acquired is False:
                if (
                    completed
                    or release_attempts != 0
                    or unit_released is not False
                ):
                    raise CaptureProcessError(
                        "failed pre-reservation batch receipt is inconsistent"
                    )
                return payload
            if unit_released is True and (
                release_attempts != 1 or recovery != "none"
            ):
                raise CaptureProcessError(
                    "failed batch release receipt is internally inconsistent"
                )
            if unit_released is not True and recovery != POWER_CYCLE_RECOVERY:
                raise CaptureProcessError(
                    "failed batch without confirmed release must require recovery"
                )
        return payload

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
    "BatchAckAction",
    "BatchFrameHandler",
    "BatchProcessSpawner",
    "BatchSessionPaths",
    "CaptureAttemptResult",
    "CaptureBatchProcessError",
    "CaptureBatchResult",
    "CaptureBatchRequest",
    "CAPTURE_HELPER_FLAG",
    "CaptureIntegrityError",
    "CaptureMode",
    "CaptureOutcome",
    "CaptureProcessAdapter",
    "CaptureProcessError",
    "CaptureRequest",
    "CaptureStopped",
    "PreparedCaptureBatch",
]
