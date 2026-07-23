"""Roll-scanning service backed by the optional `coolscanpy` package.

Sibling of `negpy.services.capture.service.CaptureService`: one class that
orchestrates the hardware workflow and writes results to disk in NegPy's
conventional layout, on top of an infrastructure-layer adapter for an
optional dependency (`negpy.infrastructure.roll.coolscanpy_roll`, mirroring
`negpy.infrastructure.capture.gphoto`). It also mirrors the older, simpler
`negpy.services.scanning.service.ScannerService`: where that wraps a single
ad-hoc `Device.scan()`, this wraps coolscanpy's whole-roll workflow
(preview -> approve -> batch fine-scan).

Entirely inert if `coolscanpy` is not installed: `available()` is a cheap
presence check re-exported from that module, and every other method only
reaches coolscanpy (transitively, through `coolscanpy_roll`) once actually
called.

`write_frame` writes a captured frame across up to three output tiers --
unrepaired, repaired, positive -- each independently selectable and each
derived from the one before it. See its docstring for the tier definitions,
the write order, and how a tier that cannot be produced degrades instead of
losing the tiers that still can be. Tier 2 (repaired) is backed by
`negpy.infrastructure.roll.repair`; the bundled `fauxice_bridge` registers
the portable engine and reports it unavailable if that import fails. Tier 3
(positive) defaults to the receipt-bound portable Nikon builder/CMS path,
with `negpy.services.roll.positive` retained as an explicitly selected
approximate renderer.
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import io
import json
import os
import shutil
import stat
import tempfile
import threading
from contextvars import ContextVar
from datetime import date as _date
from typing import TYPE_CHECKING, Callable, Iterable, Iterator, cast

import numpy as np
import tifffile
from PIL import Image

try:
    import fcntl
except ImportError:  # pragma: no cover - exercised by Windows packaging smoke
    fcntl = None  # type: ignore[assignment]

from negpy.infrastructure.roll import coolscanpy_roll
from negpy.infrastructure.roll import fauxice_bridge as _fauxice_bridge  # noqa: F401 — registers engine on import
from negpy.infrastructure.roll import repair as roll_repair
from negpy.infrastructure.roll.repair import RepairMode
from negpy.kernel.system.logging import get_logger
from negpy.services.repair.fauxice_hybrid_runner import HybridRuntimeConfig
from negpy.services.rendering.image_processor import ImageProcessor
from negpy.services.roll import exact_color as roll_exact_color
from negpy.services.roll import native_builder as roll_native_builder
from negpy.services.roll import nikon_icc as roll_nikon_icc
from negpy.services.roll import positive as roll_positive
from negpy.services.scanning.templating import render_scan_filename

if TYPE_CHECKING:
    import coolscanpy
    from coolscanpy.protocol.ls5000_single_pass.capture_process import (
        ManualFrameApproval as CoolscanManualFrameApproval,
    )
    from coolscanpy.types import ProgressCallback as CoolscanProgressCallback

logger = get_logger(__name__)

_RETAINED_EVIDENCE_DIRECTORIES = frozenset(
    {
        ".negpy-dice-hybrid",
        ".negpy-dice-acquisition",
        ".negpy-native-builder",
        ".negpy-stage3-replay",
    }
)
_MAX_RECEIPT_BYTES = 16 * 1024 * 1024
_MAX_SHARED_EVIDENCE_BYTES = 64 * 1024 * 1024


def _strict_json_loads(payload: bytes | str) -> object:
    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON object key {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> object:
        raise ValueError(f"non-finite JSON constant {value!r}")

    return json.loads(
        payload,
        object_pairs_hook=unique_object,
        parse_constant=reject_constant,
    )


def _stable_regular_bytes(
    path: str,
    *,
    maximum_bytes: int,
    label: str,
) -> bytes:
    before = os.lstat(path)
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise OSError(f"{label} must be a regular non-symlink file")
    if before.st_size > maximum_bytes:
        raise OSError(f"{label} exceeds its safe size limit")
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    if hasattr(os, "O_NONBLOCK"):
        flags |= os.O_NONBLOCK
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)

        def identity(metadata: os.stat_result) -> tuple[int, int, int, int, int]:
            return (
                metadata.st_dev,
                metadata.st_ino,
                metadata.st_size,
                metadata.st_mtime_ns,
                metadata.st_ctime_ns,
            )

        if not stat.S_ISREG(opened.st_mode) or identity(opened) != identity(before):
            raise OSError(f"{label} changed while it was opened")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, maximum_bytes + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > maximum_bytes:
                raise OSError(f"{label} exceeds its safe size limit")
        after_read = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    after_path = os.lstat(path)
    if identity(after_read) != identity(opened) or identity(after_path) != identity(opened):
        raise OSError(f"{label} changed while it was read")
    return b"".join(chunks)


def _load_bounded_receipt(path: str) -> dict:
    payload = _stable_regular_bytes(
        path,
        maximum_bytes=_MAX_RECEIPT_BYTES,
        label="frame receipt",
    )
    try:
        document = _strict_json_loads(payload)
    except (UnicodeDecodeError, ValueError, TypeError) as error:
        raise OSError(f"frame receipt JSON is invalid: {error}") from error
    if not isinstance(document, dict):
        raise OSError("frame receipt must contain a JSON object")
    return document


def _fsync_directory(path: str) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    descriptor = os.open(path or ".", flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


# Re-exported so callers only need this module: `from negpy.services.roll
# .service import available`, matching how the plain Scan sidebar checks
# `_sane_available()` before showing its device combo.
available = coolscanpy_roll.available


class RollScanningError(RuntimeError):
    """Raised for a roll-scanning failure that originates in this service's
    own orchestration (lifecycle misuse), as opposed to one translated from
    coolscanpy itself. Mirrors `negpy.services.capture.service.CaptureError`."""


class OutputRollbackError(RollScanningError):
    """Publication failed and at least one prior file needs recovery."""

    def __init__(self, recovery_path: str, errors: list[str]) -> None:
        self.recovery_path = recovery_path
        self.errors = tuple(errors)
        super().__init__(
            f"frame publication failed and rollback was incomplete; prior files are retained at {recovery_path}: {'; '.join(errors)}"
        )


@dataclasses.dataclass(frozen=True)
class RollFrameOutput:
    """Where one scanned frame's files landed on disk, one pair of fields per
    tier. A tier that was not selected, or could not be produced, leaves its
    field(s) `None` -- the receipt at `receipt_path` records which of those
    two it was and, for a tier that did write, its provenance."""

    slot: int
    rgb_path: str | None  # Tier 1 (unrepaired): RGB plane
    ir_path: str | None  # Tier 1 (unrepaired): infrared plane
    repaired_rgb_path: str | None  # Tier 2 (repaired): RGB plane
    repaired_ir_path: str | None  # Tier 2 (repaired): infrared plane (Tier 1's own, retained unchanged)
    positive_path: str | None  # Tier 3 (positive)
    receipt_path: str
    synthesis_mask_path: str | None
    native_synthesis_mask_path: str | None
    hybrid_receipt_path: str | None


_ACTIVE_OUTPUT_TRANSACTION: ContextVar[_OutputTransaction | None] = ContextVar(
    "negpy_roll_output_transaction",
    default=None,
)


class _OutputTransaction:
    """Stage one frame's files, publish its receipt last, and roll back.

    The rollback guarantee covers process-level write/rename failures. File
    and directory fsync ordering narrows power-loss exposure, but a set of
    multiple POSIX pathnames is not one power-fail-atomic filesystem unit.
    """

    def __init__(self, output_folder: str) -> None:
        self.output_folder = os.path.realpath(output_folder)
        self._root = tempfile.mkdtemp(
            prefix=".negpy-frame-stage-",
            dir=self.output_folder,
        )
        self._staged: dict[str, str] = {}
        self._removals: set[str] = set()
        self._finished = False
        self._preserve_staging = False
        self._lock_descriptor: int | None = None

    def _validate_output_path(self, final_path: str) -> str:
        final = os.path.abspath(final_path)
        try:
            inside = os.path.commonpath((self.output_folder, final)) == self.output_folder
        except ValueError:
            inside = False
        if not inside:
            raise OSError("frame output escapes its selected output folder")

        parent = os.path.dirname(final)
        if os.path.commonpath((self.output_folder, os.path.realpath(parent))) != self.output_folder:
            raise OSError("frame output parent escapes through a symbolic link")
        relative_parent = os.path.relpath(parent, self.output_folder)
        current = self.output_folder
        if relative_parent != ".":
            for component in relative_parent.split(os.sep):
                current = os.path.join(current, component)
                try:
                    metadata = os.lstat(current)
                except FileNotFoundError:
                    break
                if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
                    raise OSError("frame output parent is not a real directory")
        return final

    def stage_path(self, final_path: str) -> str:
        final = self._validate_output_path(final_path)
        staged = self._staged.get(final)
        if staged is None:
            token = hashlib.sha256(final.encode("utf-8")).hexdigest()
            staged = os.path.join(self._root, f"{token}-{os.path.basename(final)}")
            self._staged[final] = staged
        return staged

    def staged_path(self, final_path: str) -> str:
        final = os.path.abspath(final_path)
        try:
            return self._staged[final]
        except KeyError as error:
            raise OSError(f"output was not staged: {final_path}") from error

    def discard(self, final_path: str) -> None:
        staged = self._staged.pop(os.path.abspath(final_path), None)
        if staged is not None:
            try:
                os.unlink(staged)
            except FileNotFoundError:
                pass

    def checkpoint(self) -> frozenset[str]:
        """Remember the outputs already staged by lower tiers."""

        return frozenset(self._staged)

    def discard_after(self, checkpoint: frozenset[str]) -> None:
        """Discard staged outputs added after ``checkpoint``."""

        for final_path in tuple(self._staged):
            if final_path not in checkpoint:
                self.discard(final_path)

    def schedule_remove(self, final_path: str) -> None:
        final = self._validate_output_path(final_path)
        self._removals.add(final)

    def _existing_receipt_paths(self, receipt_path: str) -> set[str]:
        try:
            os.lstat(receipt_path)
        except FileNotFoundError:
            return set()
        document = _load_bounded_receipt(receipt_path)
        return _receipt_output_paths(document, output_folder=self.output_folder)

    def _acquire_frame_lock(self, receipt_path: str) -> None:
        if self._lock_descriptor is not None:
            return
        if fcntl is None:
            raise OSError("safe cross-process frame locking is unavailable on this platform")
        lock_directory = os.path.join(self.output_folder, ".negpy-locks")
        try:
            metadata = os.lstat(lock_directory)
        except FileNotFoundError:
            try:
                os.mkdir(lock_directory, 0o700)
            except FileExistsError:
                pass
            metadata = os.lstat(lock_directory)
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise OSError("frame lock directory is not a real directory")
        lock_name = hashlib.sha256(os.path.abspath(receipt_path).encode("utf-8")).hexdigest() + ".lock"
        flags = os.O_CREAT | os.O_RDWR
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(
            os.path.join(lock_directory, lock_name),
            flags,
            0o600,
        )
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            os.close(descriptor)
            raise OSError("frame lock is not a regular file")
        try:
            # A crashed or wedged peer must not freeze the scanner worker/UI.
            # Publication fails closed immediately; the caller may retry once
            # the other process has completed or been recovered.
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            os.close(descriptor)
            raise OSError("frame output is busy in another NegPy process") from error
        except BaseException:
            os.close(descriptor)
            raise
        self._lock_descriptor = descriptor

    def _release_frame_lock(self) -> None:
        if self._lock_descriptor is None:
            return
        try:
            assert fcntl is not None
            fcntl.flock(self._lock_descriptor, fcntl.LOCK_UN)
        finally:
            os.close(self._lock_descriptor)
            self._lock_descriptor = None

    def schedule_stale_outputs(self, receipt_path: str) -> None:
        self._acquire_frame_lock(receipt_path)
        staged_receipt = self._staged.get(os.path.abspath(receipt_path))
        if staged_receipt is None:
            raise OSError("frame receipt was not staged")
        new_document = _load_bounded_receipt(staged_receipt)
        new_paths = _receipt_output_paths(
            new_document,
            output_folder=self.output_folder,
        )
        excluded_receipt = os.path.abspath(receipt_path)
        for final_path in tuple(self._staged):
            if final_path == excluded_receipt:
                continue
            if not _other_receipt_references(
                final_path,
                output_folder=self.output_folder,
                excluded_receipt=excluded_receipt,
            ):
                continue
            if _is_retained_evidence_path(final_path, self.output_folder):
                try:
                    staged_bytes = _stable_regular_bytes(
                        self._staged[final_path],
                        maximum_bytes=_MAX_SHARED_EVIDENCE_BYTES,
                        label="staged retained evidence",
                    )
                    existing_bytes = _stable_regular_bytes(
                        final_path,
                        maximum_bytes=_MAX_SHARED_EVIDENCE_BYTES,
                        label="shared retained evidence",
                    )
                except OSError:
                    pass
                else:
                    if staged_bytes == existing_bytes:
                        self.discard(final_path)
                        continue
            raise OSError(f"frame output is already owned by another receipt: {final_path}")
        staged_outputs = set(self._staged) - {os.path.abspath(receipt_path)}
        unbound = sorted(staged_outputs - new_paths)
        if unbound:
            raise OSError("frame transaction contains outputs absent from its receipt: " + ", ".join(unbound))
        base_path = receipt_path[: -len("_receipt.json")]
        for derived in (
            base_path + ".tif",
            base_path + "_IR.tif",
            base_path + "_repaired.tif",
            base_path + "_repaired_IR.tif",
            base_path + "_repaired_SYNTH.png",
            base_path + "_positive.tif",
        ):
            if os.path.abspath(derived) not in new_paths and not _other_receipt_references(
                derived,
                output_folder=self.output_folder,
                excluded_receipt=excluded_receipt,
            ):
                self.schedule_remove(derived)
        old_paths = self._existing_receipt_paths(receipt_path)
        for old_path in old_paths - new_paths:
            if not _is_retained_evidence_path(old_path, self.output_folder):
                continue
            if not _other_receipt_references(
                old_path,
                output_folder=self.output_folder,
                excluded_receipt=excluded_receipt,
            ):
                self.schedule_remove(old_path)

    def commit(self, *, receipt_path: str) -> None:
        receipt = os.path.abspath(receipt_path)
        if receipt not in self._staged:
            raise OSError("frame receipt was not staged")
        writes = sorted(path for path in self._staged if path != receipt)
        removals = sorted(self._removals - set(self._staged))
        backup_root = os.path.join(self._root, "backups")
        os.makedirs(backup_root, exist_ok=True)
        backups: dict[str, str] = {}
        installed: list[str] = []

        def backup(path: str) -> None:
            self._validate_output_path(path)
            if not os.path.lexists(path):
                return
            if os.path.islink(path) or not os.path.isfile(path):
                raise OSError(f"refusing to replace non-regular output {path}")
            backup_path = os.path.join(
                backup_root,
                hashlib.sha256(path.encode("utf-8")).hexdigest(),
            )
            os.replace(path, backup_path)
            backups[path] = backup_path

        try:
            for final in removals:
                backup(final)
            for final in writes:
                backup(final)
                self._validate_output_path(final)
                os.makedirs(os.path.dirname(final) or ".", exist_ok=True)
                self._validate_output_path(final)
                os.replace(self._staged[final], final)
                installed.append(final)
            for directory in sorted({os.path.dirname(path) or "." for path in (*writes, *removals)}):
                _fsync_directory(directory)
            # Keep the prior receipt in place until every referenced artifact
            # has landed.  The receipt rename is the transaction's commit mark.
            backup(receipt)
            self._validate_output_path(receipt)
            os.makedirs(os.path.dirname(receipt) or ".", exist_ok=True)
            self._validate_output_path(receipt)
            os.replace(self._staged[receipt], receipt)
            installed.append(receipt)
            _fsync_directory(os.path.dirname(receipt) or ".")
        except BaseException as publication_error:
            rollback_errors: list[str] = []
            for final in reversed(installed):
                try:
                    if os.path.isfile(final) and not os.path.islink(final):
                        os.unlink(final)
                    elif os.path.lexists(final):
                        rollback_errors.append(f"refused to remove replacement {final}")
                except OSError as error:
                    rollback_errors.append(f"could not remove replacement {final}: {error}")
            for final, backup_path in reversed(tuple(backups.items())):
                try:
                    os.makedirs(os.path.dirname(final) or ".", exist_ok=True)
                    os.replace(backup_path, final)
                except OSError as error:
                    rollback_errors.append(f"could not restore {final} from {backup_path}: {error}")
            if rollback_errors:
                recovery_path = self._retain_recovery(
                    publication_error=publication_error,
                    rollback_errors=rollback_errors,
                    backups=backups,
                )
                raise OutputRollbackError(recovery_path, rollback_errors) from publication_error
            raise
        else:
            self._finished = True
            self._cleanup_empty_evidence_directories(removals)
        finally:
            self._cleanup_staging()

    def _retain_recovery(
        self,
        *,
        publication_error: BaseException,
        rollback_errors: list[str],
        backups: dict[str, str],
    ) -> str:
        self._preserve_staging = True
        prior_root = self._root
        recovery_name = os.path.basename(self._root).replace(
            ".negpy-frame-stage-",
            ".negpy-recovery-",
            1,
        )
        recovery_path = os.path.join(os.path.dirname(self._root), recovery_name)
        try:
            os.rename(self._root, recovery_path)
        except OSError as error:
            recovery_path = self._root
            rollback_errors.append(f"could not rename recovery directory: {error}")
        else:
            self._root = recovery_path

        def relocated(path: str) -> str:
            return os.path.join(recovery_path, os.path.relpath(path, prior_root))

        manifest = {
            "publication_error": repr(publication_error),
            "recovery_path": recovery_path,
            "rollback_errors": rollback_errors,
            "unrestored_backups": {
                final: os.path.relpath(relocated(backup), recovery_path)
                for final, backup in backups.items()
                if os.path.exists(relocated(backup))
            },
        }
        try:
            with open(
                os.path.join(recovery_path, "RECOVERY.json"),
                "w",
                encoding="utf-8",
            ) as stream:
                json.dump(manifest, stream, indent=2, sort_keys=True)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
        except OSError as error:
            rollback_errors.append(f"could not write recovery manifest: {error}")
        return recovery_path

    def _cleanup_staging(self) -> None:
        if not self._preserve_staging and os.path.isdir(self._root):
            try:
                shutil.rmtree(self._root)
            except OSError as error:
                logger.warning("Could not remove frame staging directory %s: %s", self._root, error)
        self._release_frame_lock()

    def _cleanup_empty_evidence_directories(self, removals: list[str]) -> None:
        for removed in removals:
            evidence_root = _retained_evidence_root(
                removed,
                self.output_folder,
            )
            if evidence_root is None:
                continue
            parent = os.path.dirname(removed)
            while True:
                try:
                    inside = os.path.commonpath((evidence_root, parent)) == evidence_root
                except ValueError:
                    inside = False
                if not inside:
                    break
                try:
                    os.rmdir(parent)
                except OSError:
                    break
                if parent == evidence_root:
                    break
                parent = os.path.dirname(parent)

    def abort(self) -> None:
        if not self._finished:
            self._cleanup_staging()


def _transactional_frame_output(method):
    @functools.wraps(method)
    def wrapped(self, frame, output_folder, *args, **kwargs):
        os.makedirs(output_folder, exist_ok=True)
        transaction = _OutputTransaction(output_folder)
        token = _ACTIVE_OUTPUT_TRANSACTION.set(transaction)
        try:
            result = method(
                self,
                frame,
                transaction.output_folder,
                *args,
                **kwargs,
            )
            transaction.schedule_stale_outputs(result.receipt_path)
            transaction.commit(receipt_path=result.receipt_path)
            return result
        finally:
            _ACTIVE_OUTPUT_TRANSACTION.reset(token)
            transaction.abort()

    return wrapped


def _receipt_output_paths(
    document: object,
    *,
    output_folder: str,
) -> set[str]:
    if not isinstance(document, dict):
        return set()
    outputs = document.get("outputs")
    found: set[str] = set()

    def visit(value: object, key: str | None = None) -> None:
        if isinstance(value, dict):
            for child_key, child in value.items():
                visit(child, child_key)
        elif isinstance(value, list):
            for child in value:
                visit(child, key)
        elif isinstance(value, str) and key is not None and (key == "path" or key.endswith("_path")):
            candidate = os.path.abspath(value)
            try:
                inside = os.path.commonpath((output_folder, candidate)) == output_folder
            except ValueError:
                inside = False
            if inside:
                found.add(candidate)

    visit(outputs)
    return found


def _retained_evidence_root(path: str, output_folder: str) -> str | None:
    candidate = os.path.abspath(path)
    root = os.path.abspath(output_folder)
    try:
        if os.path.commonpath((root, candidate)) != root:
            return None
    except ValueError:
        return None
    relative_parts = os.path.relpath(candidate, root).split(os.sep)
    for index, component in enumerate(relative_parts):
        if component in _RETAINED_EVIDENCE_DIRECTORIES:
            return os.path.join(root, *relative_parts[: index + 1])
    return None


def _is_retained_evidence_path(path: str, output_folder: str) -> bool:
    evidence_root = _retained_evidence_root(path, output_folder)
    return evidence_root is not None and os.path.abspath(path) != evidence_root


def _other_receipt_references(
    path: str,
    *,
    output_folder: str,
    excluded_receipt: str,
) -> bool:
    try:

        def fail_walk(error: OSError) -> None:
            raise error

        walker = os.walk(
            output_folder,
            topdown=True,
            onerror=fail_walk,
            followlinks=False,
        )
        for directory, subdirectories, names in walker:
            retained_subdirectories: list[str] = []
            for name in subdirectories:
                child = os.path.join(directory, name)
                if (
                    name in _RETAINED_EVIDENCE_DIRECTORIES
                    or name.startswith(".negpy-frame-stage-")
                    or name.startswith(".negpy-recovery-")
                    or os.path.islink(child)
                ):
                    continue
                retained_subdirectories.append(name)
            subdirectories[:] = retained_subdirectories
            for name in names:
                if not name.endswith("_receipt.json"):
                    continue
                receipt = os.path.abspath(os.path.join(directory, name))
                if receipt == excluded_receipt:
                    continue
                try:
                    document = _load_bounded_receipt(receipt)
                except OSError:
                    # An unsafe/unreadable sibling may still own this path.
                    # Preserve rather than risk deleting or overwriting it.
                    return True
                if os.path.abspath(path) in _receipt_output_paths(
                    document,
                    output_folder=output_folder,
                ):
                    return True
    except OSError:
        return True
    return False


def _prepare_evidence_directory(path: str) -> None:
    if _ACTIVE_OUTPUT_TRANSACTION.get() is None:
        os.makedirs(path, exist_ok=True)


class RollScanningService:
    """Orchestrates coolscanpy device/roll lifecycle and output writing.

    One roll open at a time, matching `coolscanpy.Device.roll()`'s own
    single-reservation lock -- call `close()` (or use this as a context
    manager) before opening another.
    """

    def __init__(
        self,
        *,
        exact_color_builder: roll_exact_color.VerifiedStage1Builder | None = None,
        exact_color_evaluator: roll_exact_color.VerifiedPortableCMSEvaluator | None = None,
        hybrid_runtime: HybridRuntimeConfig | None = None,
    ) -> None:
        self._roll: coolscanpy_roll.RollHandle | None = None
        # Explicitly injected: NegPy never treats its generic renderer as a
        # Nikon-exact substitute when either verified stage is absent.
        self._exact_color_builder = exact_color_builder
        self._exact_color_evaluator = exact_color_evaluator
        self._hybrid_runtime = hybrid_runtime
        self._repair_cancel = threading.Event()
        # Built lazily, on the first frame that actually needs Tier 3 -- see
        # `_get_image_processor()`. Most batches never touch it.
        self._image_processor: ImageProcessor | None = None

    # -- device / roll lifecycle -----------------------------------------

    def list_devices(self) -> "list[coolscanpy.DeviceInfo]":
        return coolscanpy_roll.list_devices()

    def open_roll(
        self,
        device_id: str | None = None,
        *,
        material: "coolscanpy.Material | None" = None,
        attempts_root: str | os.PathLike[str] | None = None,
    ) -> None:
        """Open a device and its roll extension. Call `close()` when done."""
        if self._roll is not None:
            raise RollScanningError("a roll is already open on this service; call close() first")
        self._roll = coolscanpy_roll.open_roll(
            device_id,
            material=material,
            attempts_root=attempts_root,
        )

    def close(self) -> None:
        """Idempotent. Ends the roll reservation and releases the device."""
        if self._roll is not None:
            self._roll.close()
            self._roll = None

    def __enter__(self) -> "RollScanningService":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- preview / approval ------------------------------------------------

    def preview(
        self, slots: Iterable[int] | None = None, *, on_progress: "CoolscanProgressCallback | None" = None
    ) -> "list[coolscanpy.Thumbnail]":
        return self._require_roll().preview(slots, on_progress=on_progress)

    def restore_preview_session(
        self,
        payload: str,
        slots: Iterable[int] | None = None,
    ) -> "list[coolscanpy.Thumbnail]":
        """Restore a saved, content-verified review on the open roll."""

        return self._require_roll().restore_preview_session(payload, slots)

    def set_spacing_offset(self, slot: int, offset_rows: int) -> None:
        self._require_roll().set_spacing_offset(slot, offset_rows)

    def approve(self, slot: int) -> "CoolscanManualFrameApproval":
        """Approve one reviewed slot and return its content-bound receipt."""

        return self._require_roll().approve(slot)

    def needs_approval(self, slot: int) -> bool:
        return self._require_roll().needs_approval(slot)

    # -- scanning ----------------------------------------------------------

    def prepare_batch(self) -> None:
        """Arm a new batch before its queued worker request is exposed."""

        self._repair_cancel.clear()

    def scan_many(self, slots: Iterable[int], *, on_progress: "CoolscanProgressCallback | None" = None) -> Iterator["coolscanpy.Frame"]:
        if self._repair_cancel.is_set():
            raise roll_repair.RepairCancelled("roll batch cancelled before scanning started")
        yield from self._require_roll().scan_many(slots, on_progress=on_progress)

    def safe_stop(self) -> None:
        self._repair_cancel.set()
        if self._roll is not None:
            self._roll.safe_stop()

    # -- writing -------------------------------------------------------------

    @_transactional_frame_output
    def write_frame(
        self,
        frame: "coolscanpy.Frame",
        output_folder: str,
        filename_pattern: str,
        *,
        write_unrepaired: bool = True,
        write_repaired: bool = False,
        write_positive: bool = False,
        repair_mode: str = RepairMode.EXACT.value,
        positive_mode: str = roll_exact_color.PositiveColorMode.NIKON_EXACT.value,
        builder_receipt: roll_exact_color.BuilderReceipt | None = None,
        on_repair_progress: Callable[[float], None] | None = None,
    ) -> RollFrameOutput:
        """Write one scanned `Frame` to disk across up to three tiers, plus a
        `_receipt.json` sidecar. Each tier is independently selectable; any
        combination of the three flags is valid.

        Tier 1, unrepaired (`write_unrepaired`): the frame exactly as
        captured -- a 16-bit RGB TIFF plus an `_IR` sidecar when the frame
        carries an infrared plane, matching `writer.write_tiff_16bit`'s own
        `<basename>_IR.tif` convention. This is the archival master: the
        only tier the scanner itself can reproduce. A failure writing it is
        allowed to raise, unlike Tier 2/3 below.

        Tier 2, repaired (`write_repaired`): Tier 1 with infrared-guided
        dust/scratch repair applied through `negpy.infrastructure.roll
        .repair`, written as `<basename>_repaired.tif` plus a
        `<basename>_repaired_IR.tif` sidecar. That sidecar is Tier 1's own
        infrared plane, unchanged -- repair consumes infrared to find
        defects, it does not produce a repaired version of it, and keeping
        the original lets a later re-repair under a different mode start
        from the same evidence Tier 1 captured.

        Tier 3, positive (`write_positive`): Tier 2 inverted through either
        receipt-bound portable Nikon Stage-1 and CMS stages (`positive_mode=
        "nikon-exact", the fail-closed default) or NegPy's explicitly selected
        approximate renderer (`"negpy-approximate"`). Exact mode accepts an explicit Stage-3 replay
        receipt or automatically builds a native receipt when the frame carries
        proven per-frame density and analyzer evidence.
        Requested independently of `write_repaired`, but always derived from
        Tier 2's in-memory result -- repair runs whenever either flag needs
        it, so a positive is never silently missing the repair pass a caller
        asked for, and never from a Tier 1 or Tier 2 file already on disk
        (an upper tier is never derived from a lower tier's file, only from
        the in-memory data the lower tier is also optionally written from).

        Tier 2 and Tier 3 degrade instead of raising when they cannot be
        produced -- no repair engine registered, no infrared plane to guide
        one, the engine itself fails, or the render fails -- recording why in
        the receipt and leaving the corresponding `RollFrameOutput` path(s)
        `None`. Tier 1 always writes regardless (or Tier 2, for a Tier-3
        failure): losing the archival capture to a problem in a derived,
        frame-bound tier is exactly what this must not do. When Tier 1 has an
        infrared plane, its transaction also retains the scanner-native
        prepass, IR-validity map, and acquisition binding required to replay
        repair after the physical media has moved.

        When a frame carries the native per-acquisition Nikon builder
        evidence, that evidence is validated and retained independently of
        Tier 3 selection or success. The frame receipt binds the retained
        artifacts under ``outputs.native_color_evidence`` so a later exact
        render does not depend on the transient in-memory frame object.

        `filename_pattern` is the same Jinja2 template the plain scan path
        uses (variables: `date`, `seq`), but `seq` is seeded from the frame's
        physical slot number rather than probed for the next free name: a
        roll slot is already a stable identity, so re-scanning slot 5
        replaces slot 5's old files instead of piling up a `..._002` beside
        them, for every tier.

        `frame.ir_validity` is bound with the two scanner-native RGBI captures
        and consumed by repair. Invalid IR pixels retain their original RGB.
        """
        try:
            resolved_repair_mode = RepairMode(repair_mode)
        except ValueError as error:
            raise ValueError(f"unknown repair mode {repair_mode!r}") from error

        os.makedirs(output_folder, exist_ok=True)
        date_str = _date.today().strftime("%Y%m%d")
        basename = render_scan_filename(filename_pattern, date_str, frame.slot)
        base_path = os.path.join(output_folder, basename)

        outputs: dict = {}

        # -- Tier 1: unrepaired -----------------------------------------------
        rgb_path = ir_path = None
        if write_unrepaired:
            rgb_path = _atomic_write_tiff(base_path + ".tif", frame.rgb, photometric="rgb")
            if frame.ir is not None:
                ir_path = _atomic_write_tiff(base_path + "_IR.tif", frame.ir, photometric="minisblack")
            outputs["unrepaired"] = {"written": True, "rgb_path": rgb_path, "ir_path": ir_path}
        else:
            outputs["unrepaired"] = {"written": False, "status": "not selected"}

        # Tier-1 RGB/IR TIFFs are lossless, but by themselves omit the meter
        # prepass, per-pixel IR validity, and frame identities needed to
        # reconstruct the scanner-native RepairAcquisition after media moves.
        # Retain that small unique evidence independently of immediate Tier-2
        # availability or success. A retention failure is disclosed without
        # sacrificing the irreplaceable Tier-1 capture.
        prepared_repair_acquisition: roll_repair.RepairAcquisition | None = None
        if write_unrepaired and frame.ir is not None:
            transaction = _ACTIVE_OUTPUT_TRANSACTION.get()
            acquisition_checkpoint = transaction.checkpoint() if transaction is not None else frozenset()
            try:
                prepared_repair_acquisition = _repair_acquisition_from_frame(frame)
                retained_repair_acquisition = _retain_repair_acquisition_evidence(
                    base_path,
                    prepared_repair_acquisition,
                    storage_rgb=frame.rgb,
                    storage_ir=frame.ir,
                    rgb_path=rgb_path,
                    ir_path=ir_path,
                )
            except Exception as error:
                if transaction is not None:
                    transaction.discard_after(acquisition_checkpoint)
                outputs["repair_acquisition_evidence"] = {
                    "replayable": False,
                    "retained": False,
                    "status": f"unavailable: {error}",
                }
            else:
                outputs["repair_acquisition_evidence"] = {
                    **retained_repair_acquisition,
                    "retained": True,
                }

        # Preserve the scanner-native color inputs as an acquisition artifact,
        # not merely as a side effect of a successful Tier-3 render. A TIFF or
        # CMS failure must never discard the only evidence from which that
        # exact render can be retried later.
        native_receipt_from_frame: roll_exact_color.NativeValidatedBuilderReceipt | None = None
        retained_native_evidence: dict[str, object] | None = None
        native_receipt_error: Exception | None = None
        if getattr(frame, "nikon_exact_builder_evidence", None) is not None:
            transaction = _ACTIVE_OUTPUT_TRANSACTION.get()
            evidence_checkpoint = transaction.checkpoint() if transaction is not None else frozenset()
            try:
                native_receipt_from_frame = _build_native_receipt_from_frame(frame)
                retained_native_evidence = _retain_native_builder_evidence(
                    base_path,
                    native_receipt_from_frame,
                )
            except Exception as error:
                if transaction is not None:
                    transaction.discard_after(evidence_checkpoint)
                native_receipt_from_frame = None
                retained_native_evidence = None
                native_receipt_error = error
                outputs["native_color_evidence"] = {
                    "native_per_acquisition_builder": True,
                    "retained": False,
                    "scope": roll_exact_color.NATIVE_BUILDER_SCOPE,
                    "status": f"unavailable: {error}",
                }
            else:
                outputs["native_color_evidence"] = {
                    "builder_receipt": roll_exact_color.receipt_payload(native_receipt_from_frame),
                    "builder_receipt_sha256": native_receipt_from_frame.sha256,
                    "native_per_acquisition_builder": True,
                    "retained": True,
                    "retained_builder_evidence": retained_native_evidence,
                    "scope": roll_exact_color.NATIVE_BUILDER_SCOPE,
                }

        # -- Tier 2: repaired (also feeds Tier 3, even when not itself written) --
        outputs["repaired"] = {"written": False, "status": "not selected"}
        repair_result: roll_repair.RepairResult | None = None
        repaired_rgb_path = repaired_ir_path = None
        synthesis_mask_path = None
        native_synthesis_mask_path = None
        hybrid_receipt_path = None
        if write_repaired or write_positive:
            if frame.ir is None:
                outputs["repaired"] = {"written": False, "status": "unavailable: frame has no infrared plane to guide repair"}
            elif not roll_repair.available():
                outputs["repaired"] = {"written": False, "status": "unavailable: no dust-repair engine registered"}
            else:
                try:
                    acquisition = (
                        prepared_repair_acquisition if prepared_repair_acquisition is not None else _repair_acquisition_from_frame(frame)
                    )
                    repair_result = roll_repair.repair(
                        acquisition,
                        resolved_repair_mode,
                        hybrid_runtime=self._hybrid_runtime,
                        progress=on_repair_progress,
                        cancel=self._repair_cancel,
                    )
                except roll_repair.RepairCancelled:
                    raise
                except Exception as error:
                    outputs["repaired"] = {"written": False, "status": f"repair failed: {error}"}
                else:
                    transaction = _ACTIVE_OUTPUT_TRANSACTION.get()
                    repair_checkpoint = transaction.checkpoint() if transaction is not None else frozenset()
                    try:
                        source_rgb = np.asarray(repair_result.rgb)
                        rgb_snapshot = np.array(
                            source_rgb,
                            order="C",
                            copy=True,
                        )
                        rgb_snapshot.setflags(write=False)
                        routing_snapshot = (
                            dict(repair_result.routing_counts)
                            if isinstance(repair_result.routing_counts, dict)
                            else repair_result.routing_counts
                        )
                        repair_result = dataclasses.replace(
                            repair_result,
                            rgb=rgb_snapshot,
                            routing_counts=routing_snapshot,
                        )
                        _validate_repair_result_binding(
                            acquisition,
                            repair_result,
                            requested_mode=resolved_repair_mode,
                        )
                        hybrid_evidence = None
                        if repair_result.mode_resolved is RepairMode.HYBRID:
                            synthesis_mask_path = _atomic_write_bytes(
                                base_path + "_repaired_SYNTH.png",
                                repair_result.storage_synthesis_mask_png,
                            )
                            hybrid_evidence = _retain_hybrid_repair_evidence(
                                base_path,
                                acquisition,
                                repair_result,
                            )
                            native_synthesis_mask_path = hybrid_evidence["native_mask"]["path"]
                            hybrid_receipt_path = hybrid_evidence["receipt"]["path"]
                    except Exception as error:
                        if transaction is not None:
                            transaction.discard_after(repair_checkpoint)
                        repair_result = None
                        synthesis_mask_path = None
                        native_synthesis_mask_path = None
                        hybrid_receipt_path = None
                        outputs["repaired"] = {
                            "written": False,
                            "status": f"repair evidence failed: {error}",
                        }
                        hybrid_evidence = None
                    if repair_result is None:
                        pass
                    else:
                        acquisition_entry = {
                            "acquisition_id": acquisition.acquisition_id,
                            "slot": acquisition.slot,
                            "reservation_id": acquisition.reservation_id,
                            "capture_attempt_id": acquisition.capture_attempt_id,
                            "evidence_sha256": acquisition.evidence_sha256,
                            "storage_transform": acquisition.storage_transform,
                            "main_rgbi_sha256": acquisition.main_rgbi_sha256,
                            "prepass_rgbi_sha256": acquisition.prepass_rgbi_sha256,
                            "ir_validity_sha256": acquisition.ir_validity_sha256,
                        }
                        entry: dict[str, object] = {
                            "engine": repair_result.engine,
                            "engine_version": repair_result.engine_version,
                            "mode_requested": str(repair_result.mode_requested),
                            "mode_resolved": str(repair_result.mode_resolved),
                            "degraded": repair_result.degraded,
                            "reason": repair_result.reason,
                            "acquisition": acquisition_entry,
                            # Kept at top level for older receipt consumers.
                            "acquisition_id": repair_result.acquisition_id,
                            "slot": repair_result.slot,
                            "reservation_id": repair_result.reservation_id,
                            "evidence_sha256": repair_result.evidence_sha256,
                            "backend_requested": repair_result.backend_requested,
                            "backend_used": repair_result.backend_used,
                            "backend_selection_reason": (repair_result.backend_selection_reason),
                            "native_output_rgb_sha256": (repair_result.native_output_rgb_sha256),
                            "storage_output_rgb_sha256": (repair_result.storage_output_rgb_sha256),
                        }
                        if hybrid_evidence is not None:
                            applied_pixels = int(round(repair_result.synthesis_fraction * acquisition.ir_validity.size))
                            entry["disclosure_mask"] = {
                                "applied_final": {
                                    "fraction": repair_result.synthesis_fraction,
                                    "native": hybrid_evidence["native_mask"],
                                    "pixel_count": applied_pixels,
                                    "storage": {
                                        "bytes": len(repair_result.storage_synthesis_mask_png),
                                        "path": synthesis_mask_path,
                                        "sha256": (repair_result.storage_synthesis_mask_sha256),
                                        "shape": list(repair_result.storage_synthesis_mask_shape),
                                    },
                                    "transform": (repair_result.synthesis_mask_transform),
                                },
                                "routed_raw": {
                                    "native": hybrid_evidence["routed_native_mask"],
                                    "routing_counts": (repair_result.routing_counts),
                                },
                            }
                            entry["hybrid_receipt"] = {
                                **hybrid_evidence["receipt"],
                                "provenance_class": (repair_result.hybrid_provenance_class),
                                "verified_output_rgb_sha256": (repair_result.hybrid_receipt_output_rgb_sha256),
                            }
                            entry["hybrid_evidence_binding"] = hybrid_evidence["binding"]
                        if write_repaired:
                            repaired_rgb_path = _atomic_write_tiff(base_path + "_repaired.tif", repair_result.rgb, photometric="rgb")
                            repaired_ir_path = _atomic_write_tiff(base_path + "_repaired_IR.tif", frame.ir, photometric="minisblack")
                            entry.update(written=True, rgb_path=repaired_rgb_path, ir_path=repaired_ir_path)
                        else:
                            entry.update(written=False, status="not selected (computed in memory for the positive)")
                        outputs["repaired"] = entry

        # -- Tier 3: positive -----------------------------------------------
        outputs["positive"] = {"written": False, "status": "not selected"}
        positive_path = None
        if write_positive:
            if repair_result is None:
                tier2_status = outputs["repaired"].get("status", "unavailable")
                outputs["positive"] = {
                    "written": False,
                    "status": f"unavailable: Tier 2 (repaired) could not be produced ({tier2_status})",
                    "color_mode": positive_mode,
                }
            elif positive_mode == roll_exact_color.PositiveColorMode.NIKON_EXACT.value:
                transaction = _ACTIVE_OUTPUT_TRANSACTION.get()
                exact_checkpoint = transaction.checkpoint() if transaction is not None else frozenset()
                try:
                    exact_icc_profile = roll_nikon_icc.nikon_adobe_rgb_profile()
                    active_receipt = builder_receipt
                    active_builder = self._exact_color_builder
                    active_evaluator = self._exact_color_evaluator
                    if active_receipt is None:
                        if native_receipt_from_frame is not None:
                            active_receipt = native_receipt_from_frame
                        elif native_receipt_error is not None:
                            raise roll_exact_color.ExactColorUnavailable(str(native_receipt_error)) from native_receipt_error
                        else:
                            active_receipt = _build_native_receipt_from_frame(frame)
                        if active_builder is None:
                            from negpy.services.roll.portable_builder import PortableStage1Builder

                            active_builder = PortableStage1Builder()
                        if active_evaluator is None:
                            from negpy.services.roll.portable_cms import PortableCMSOnEvaluator

                            active_evaluator = PortableCMSOnEvaluator()
                    result = roll_exact_color.evaluate_exact_color(
                        repair_result.rgb,
                        builder_receipt=active_receipt,
                        builder=active_builder,
                        evaluator=active_evaluator,
                    )
                    builder_application_receipt = result.builder_application_receipt
                    if builder_application_receipt is None:
                        raise roll_exact_color.ExactColorIntegrityError("exact-color result is missing its builder application receipt")
                    native_builder = type(result.builder_receipt) is roll_exact_color.NativeValidatedBuilderReceipt
                    positive_path = _atomic_write_tiff(
                        base_path + "_positive.tif",
                        result.rgb,
                        photometric="rgb",
                        iccprofile=exact_icc_profile,
                    )
                    exact_tiff_artifact = _verify_exact_positive_tiff(
                        positive_path,
                        expected_rgb=result.rgb,
                        expected_icc=exact_icc_profile,
                    )
                    # Retain replay evidence only after the TIFF itself has
                    # passed verification.  If retention then fails, the
                    # transaction checkpoint below discards both together.
                    if (
                        retained_native_evidence is not None
                        and native_receipt_from_frame is not None
                        and result.builder_receipt.sha256 == native_receipt_from_frame.sha256
                    ):
                        retained_evidence = retained_native_evidence
                    else:
                        retained_evidence = _retain_builder_evidence(
                            base_path,
                            result.builder_receipt,
                        )
                except Exception as error:
                    if transaction is not None:
                        transaction.discard_after(exact_checkpoint)
                    positive_path = None
                    outputs["positive"] = {
                        "written": False,
                        "status": f"unavailable: exact Nikon color: {error}",
                        "color_mode": roll_exact_color.PositiveColorMode.NIKON_EXACT.value,
                    }
                else:
                    outputs["positive"] = {
                        "written": True,
                        "rgb_path": positive_path,
                        "color_mode": roll_exact_color.PositiveColorMode.NIKON_EXACT.value,
                        "exact_nikon_color": True,
                        "inversion_path": (
                            "native-per-acquisition-builder-and-verified-portable-cms"
                            if native_builder
                            else "stage3-evidence-replay-bridge-and-verified-portable-cms"
                        ),
                        "native_per_acquisition_builder": native_builder,
                        "repaired_input_rgb_sha256": result.source_rgb_sha256,
                        "input_rgb_sha256": result.input_rgb_sha256,
                        "stage1_input_rgb_sha256": result.input_rgb_sha256,
                        "output_rgb_sha256": result.output_rgb_sha256,
                        "builder_receipt": roll_exact_color.receipt_payload(result.builder_receipt),
                        "builder_receipt_sha256": result.builder_receipt.sha256,
                        "builder_validated": True,
                        "builder_application_receipt": roll_exact_color.receipt_payload(builder_application_receipt),
                        "builder_application_receipt_sha256": (builder_application_receipt.sha256),
                        "cms_receipt": roll_exact_color.receipt_payload(result.cms_receipt),
                        "cms_receipt_sha256": result.cms_receipt.sha256,
                        "cms_verified": True,
                        "icc_profile": roll_nikon_icc.profile_receipt_binding(),
                        "tiff_artifact": exact_tiff_artifact,
                        "retained_builder_evidence": retained_evidence,
                        "repair_engine": repair_result.engine,
                        "repair_engine_version": repair_result.engine_version,
                        "repair_mode": str(repair_result.mode_resolved),
                    }
                    if native_builder:
                        outputs["positive"]["native_builder_scope"] = roll_exact_color.NATIVE_BUILDER_SCOPE
                    else:
                        outputs["positive"]["replay_bridge_scope"] = roll_exact_color.STAGE3_REPLAY_SCOPE
            elif positive_mode != roll_exact_color.PositiveColorMode.NEGPY_APPROXIMATE.value:
                outputs["positive"] = {
                    "written": False,
                    "status": f"unavailable: unknown positive color mode {positive_mode!r}",
                    "color_mode": positive_mode,
                }
            elif not roll_positive.available():
                outputs["positive"] = {
                    "written": False,
                    "status": "unavailable: inversion path not available",
                    "color_mode": roll_exact_color.PositiveColorMode.NEGPY_APPROXIMATE.value,
                }
            else:
                try:
                    result = roll_positive.render_positive(repair_result.rgb, processor=self._get_image_processor())
                except Exception as error:
                    outputs["positive"] = {
                        "written": False,
                        "status": f"inversion failed: {error}",
                        "color_mode": roll_exact_color.PositiveColorMode.NEGPY_APPROXIMATE.value,
                    }
                else:
                    positive_path = _atomic_write_tiff(base_path + "_positive.tif", result.rgb, photometric="rgb")
                    outputs["positive"] = {
                        "written": True,
                        "rgb_path": positive_path,
                        "color_mode": roll_exact_color.PositiveColorMode.NEGPY_APPROXIMATE.value,
                        "exact_nikon_color": False,
                        "inversion_path": "negpy.services.rendering.image_processor.ImageProcessor.run_pipeline",
                        "render_intent": result.render_intent,
                        "process_mode": result.process_mode,
                        "auto_exposure": result.auto_exposure,
                        "negpy_version": result.negpy_version,
                        "repair_engine": repair_result.engine,
                        "repair_engine_version": repair_result.engine_version,
                        "repair_mode": str(repair_result.mode_resolved),
                    }

        receipt_path = base_path + "_receipt.json"
        receipt_payload = dataclasses.asdict(frame.receipt)
        receipt_artifacts = getattr(frame.receipt, "artifacts", None)
        if receipt_artifacts is not None:
            receipt_payload["artifacts"] = {name: dataclasses.asdict(artifact) for name, artifact in receipt_artifacts.items()}
        receipt_payload["outputs"] = outputs
        _atomic_write_json(receipt_path, receipt_payload)

        return RollFrameOutput(
            slot=frame.slot,
            rgb_path=rgb_path,
            ir_path=ir_path,
            repaired_rgb_path=repaired_rgb_path,
            repaired_ir_path=repaired_ir_path,
            positive_path=positive_path,
            receipt_path=receipt_path,
            synthesis_mask_path=synthesis_mask_path,
            native_synthesis_mask_path=native_synthesis_mask_path,
            hybrid_receipt_path=hybrid_receipt_path,
        )

    def _get_image_processor(self) -> ImageProcessor:
        """Built on first use and reused across a batch: its constructor
        probes for GPU acceleration, worth paying once per service instance
        rather than once per frame, even though Tier-3 rendering itself
        always runs on the CPU engine (see `positive.render_positive`)."""
        if self._image_processor is None:
            self._image_processor = ImageProcessor()
        return self._image_processor

    # -- internals -----------------------------------------------------------

    def _require_roll(self) -> coolscanpy_roll.RollHandle:
        if self._roll is None:
            raise RollScanningError("no roll is open; call open_roll() first")
        return self._roll


def _repair_acquisition_from_frame(frame: object) -> roll_repair.RepairAcquisition:
    prepare = getattr(frame, "prepare_digital_ice", None)
    if not callable(prepare):
        raise ValueError("frame has no bound scanner-native Digital ICE acquisition")
    source = prepare()
    if isinstance(source, roll_repair.RepairAcquisition):
        if source.slot != getattr(frame, "slot", None):
            raise ValueError("Digital ICE acquisition belongs to another frame slot")
        _validate_digital_ice_producer_binding(source)
        return source
    required = (
        "acquisition_id",
        "slot",
        "reservation_id",
        "capture_attempt_id",
        "storage_transform",
        "evidence_sha256",
        "main_rgbi_sha256",
        "meter_rgbi_sha256",
        "ir_validity_sha256",
        "main_rgbi",
        "meter_rgbi",
        "ir_validity",
    )
    missing = [name for name in required if not hasattr(source, name)]
    if missing:
        raise ValueError("Digital ICE acquisition producer omitted: " + ", ".join(missing))
    if source.slot != getattr(frame, "slot", None):
        raise ValueError("Digital ICE acquisition belongs to another frame slot")
    acquisition = roll_repair.RepairAcquisition(
        acquisition_id=source.acquisition_id,
        slot=source.slot,
        reservation_id=source.reservation_id,
        capture_attempt_id=source.capture_attempt_id,
        storage_transform=source.storage_transform,
        evidence_sha256=source.evidence_sha256,
        main_rgbi_sha256=source.main_rgbi_sha256,
        prepass_rgbi_sha256=source.meter_rgbi_sha256,
        ir_validity_sha256=source.ir_validity_sha256,
        main_rgbi=source.main_rgbi,
        prepass_rgbi=source.meter_rgbi,
        ir_validity=source.ir_validity,
    )
    _validate_digital_ice_producer_binding(acquisition)
    return acquisition


def _npy_payload(array: np.ndarray, *, dtype: np.dtype) -> bytes:
    canonical = np.array(array, dtype=dtype, order="C", copy=True)
    stream = io.BytesIO()
    np.save(stream, canonical, allow_pickle=False)
    return stream.getvalue()


def _derive_digital_ice_producer_binding(
    *,
    slot: int,
    reservation_id: str,
    capture_attempt_id: str,
    main_rgbi: np.ndarray,
    prepass_rgbi: np.ndarray,
    ir_validity: np.ndarray,
) -> tuple[str, str]:
    """Reproduce Coolscanpy's v1 DigitalIceAcquisitionEvidence hashes."""

    identity_document = {
        "capture_attempt_id": capture_attempt_id,
        "kind": "coolscanpy.digital-ice-acquisition-identity",
        "reservation_id": reservation_id,
        "slot": slot,
        "version": 1,
    }
    identity_bytes = json.dumps(
        identity_document,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    acquisition_id = "dice-" + hashlib.sha256(identity_bytes).hexdigest()

    native_main = np.array(main_rgbi, dtype="<u2", order="C", copy=True)
    native_prepass = np.array(prepass_rgbi, dtype="<u2", order="C", copy=True)
    native_validity = np.array(
        ir_validity,
        dtype=np.bool_,
        order="C",
        copy=True,
    )
    storage_main = np.ascontiguousarray(np.rot90(native_main, k=1, axes=(0, 1)))
    storage_validity = np.ascontiguousarray(np.rot90(native_validity, k=1, axes=(0, 1)))

    def artifact(array: np.ndarray, *, dtype: np.dtype) -> dict[str, object]:
        canonical = np.array(array, dtype=dtype, order="C", copy=True)
        payload = memoryview(canonical).cast("B")
        return {
            "byte_length": payload.nbytes,
            "dtype": canonical.dtype.str,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "shape": list(canonical.shape),
        }

    evidence_document = {
        "acquisition_id": acquisition_id,
        "artifacts": {
            "scanner_native_ir_validity": artifact(
                native_validity,
                dtype=np.dtype(np.bool_),
            ),
            "scanner_native_main_rgbi": artifact(
                native_main,
                dtype=np.dtype("<u2"),
            ),
            "scanner_native_meter_rgbi": artifact(
                native_prepass,
                dtype=np.dtype("<u2"),
            ),
            "storage_ir": artifact(
                storage_main[..., 3],
                dtype=np.dtype("<u2"),
            ),
            "storage_ir_validity": artifact(
                storage_validity,
                dtype=np.dtype(np.bool_),
            ),
            "storage_rgb": artifact(
                storage_main[..., :3],
                dtype=np.dtype("<u2"),
            ),
        },
        "capture_attempt_id": capture_attempt_id,
        "kind": "coolscanpy.digital-ice-acquisition-evidence",
        "reservation_id": reservation_id,
        "slot": slot,
        "storage_transform": roll_repair.DIGITAL_ICE_STORAGE_TRANSFORM,
        "version": 1,
    }
    evidence_bytes = json.dumps(
        evidence_document,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return acquisition_id, hashlib.sha256(evidence_bytes).hexdigest()


def _validate_digital_ice_producer_binding(
    acquisition: roll_repair.RepairAcquisition,
) -> None:
    expected_acquisition_id, expected_evidence_sha256 = _derive_digital_ice_producer_binding(
        slot=acquisition.slot,
        reservation_id=acquisition.reservation_id,
        capture_attempt_id=acquisition.capture_attempt_id,
        main_rgbi=acquisition.main_rgbi,
        prepass_rgbi=acquisition.prepass_rgbi,
        ir_validity=acquisition.ir_validity,
    )
    if acquisition.acquisition_id != expected_acquisition_id:
        raise ValueError("Digital ICE acquisition identity is not canonical")
    if acquisition.evidence_sha256 != expected_evidence_sha256:
        raise ValueError("Digital ICE producer evidence SHA-256 changed")


def _retain_repair_acquisition_evidence(
    base_path: str,
    acquisition: roll_repair.RepairAcquisition,
    *,
    storage_rgb: np.ndarray,
    storage_ir: np.ndarray,
    rgb_path: str | None,
    ir_path: str | None,
) -> dict[str, object]:
    """Retain the unique inputs needed to replay repair after media moves."""

    if rgb_path is None or ir_path is None:
        raise ValueError("Tier-1 RGB and IR paths are required for repair replay")
    rgb = np.asarray(storage_rgb)
    infrared = np.asarray(storage_ir)
    expected_storage_shape = (
        acquisition.main_rgbi.shape[1],
        acquisition.main_rgbi.shape[0],
    )
    if (
        rgb.dtype != np.uint16
        or rgb.shape != (*expected_storage_shape, 3)
        or infrared.dtype != np.uint16
        or infrared.shape != expected_storage_shape
    ):
        raise ValueError("Tier-1 planes do not match the Digital ICE acquisition")
    storage_rgbi = np.dstack((rgb, infrared))
    reconstructed_native = np.ascontiguousarray(np.rot90(storage_rgbi, k=-1, axes=(0, 1)))
    if _rgb16_sha256(reconstructed_native) != acquisition.main_rgbi_sha256:
        raise ValueError("Tier-1 planes do not reconstruct the captured main RGBI")
    _validate_digital_ice_producer_binding(acquisition)

    evidence_root = os.path.join(
        os.path.dirname(base_path) or ".",
        ".negpy-dice-acquisition",
    )
    evidence_token = hashlib.sha256((acquisition.acquisition_id + "\0" + os.path.basename(base_path)).encode("utf-8")).hexdigest()
    evidence_directory = os.path.join(evidence_root, evidence_token)
    if any(os.path.lexists(path) and os.path.islink(path) for path in (evidence_root, evidence_directory)):
        raise OSError("repair acquisition evidence path is a symlink")
    _prepare_evidence_directory(evidence_directory)

    prepass_payload = _npy_payload(
        acquisition.prepass_rgbi,
        dtype=np.dtype("<u2"),
    )
    validity_payload = _npy_payload(
        acquisition.ir_validity,
        dtype=np.dtype(np.bool_),
    )
    prepass_path = _atomic_write_bytes(
        os.path.join(evidence_directory, "prepass.rgbi16.npy"),
        prepass_payload,
    )
    validity_path = _atomic_write_bytes(
        os.path.join(evidence_directory, "ir-validity.npy"),
        validity_payload,
    )

    def artifact(
        *,
        path: str,
        payload: bytes | None,
        raw_sha256: str,
        shape: tuple[int, ...],
        dtype: str,
    ) -> dict[str, object]:
        row: dict[str, object] = {
            "dtype": dtype,
            "path": path,
            "raw_sha256": raw_sha256,
            "relative_path": os.path.relpath(path, evidence_directory),
            "shape": list(shape),
        }
        if payload is not None:
            row.update(
                bytes=len(payload),
                file_sha256=hashlib.sha256(payload).hexdigest(),
            )
        return row

    sources = {
        "storage_ir_tiff": artifact(
            path=ir_path,
            payload=None,
            raw_sha256=_rgb16_sha256(infrared),
            shape=infrared.shape,
            dtype="<u2",
        ),
        "storage_rgb_tiff": artifact(
            path=rgb_path,
            payload=None,
            raw_sha256=_rgb16_sha256(rgb),
            shape=rgb.shape,
            dtype="<u2",
        ),
    }
    for source in sources.values():
        source["orientation"] = "upright-storage"
    artifacts = {
        "ir_validity": artifact(
            path=validity_path,
            payload=validity_payload,
            raw_sha256=acquisition.ir_validity_sha256,
            shape=acquisition.ir_validity.shape,
            dtype="|b1",
        ),
        "prepass_rgbi": artifact(
            path=prepass_path,
            payload=prepass_payload,
            raw_sha256=acquisition.prepass_rgbi_sha256,
            shape=acquisition.prepass_rgbi.shape,
            dtype="<u2",
        ),
    }
    portable_artifacts = {name: {key: value for key, value in row.items() if key != "path"} for name, row in artifacts.items()}
    portable_sources = {name: {key: value for key, value in row.items() if key != "path"} for name, row in sources.items()}
    replay_contract = {
        "authenticity": "integrity-bound-not-signed",
        "complete": True,
        "reconstruction": "stack-upright-storage-rgb-ir-then-rot90-k-1",
        "requires": [
            "storage_rgb_tiff",
            "storage_ir_tiff",
            "prepass_rgbi",
            "ir_validity",
            "acquisition_provenance",
        ],
        "storage_orientation": "upright-storage",
    }
    binding = {
        "acquisition": {
            "acquisition_id": acquisition.acquisition_id,
            "capture_attempt_id": acquisition.capture_attempt_id,
            "evidence_sha256": acquisition.evidence_sha256,
            "ir_validity_sha256": acquisition.ir_validity_sha256,
            "main_rgbi_sha256": acquisition.main_rgbi_sha256,
            "prepass_rgbi_sha256": acquisition.prepass_rgbi_sha256,
            "reservation_id": acquisition.reservation_id,
            "slot": acquisition.slot,
            "storage_transform": acquisition.storage_transform,
        },
        "artifacts": portable_artifacts,
        "replay": replay_contract,
        "schema": "negpy.dice-acquisition-replay-v1",
        "sources": portable_sources,
    }
    binding_payload = (
        json.dumps(
            binding,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    binding_path = _atomic_write_bytes(
        os.path.join(evidence_directory, "acquisition-binding.json"),
        binding_payload,
    )
    return {
        "acquisition_id": acquisition.acquisition_id,
        "artifacts": artifacts,
        "binding": {
            "bytes": len(binding_payload),
            "path": binding_path,
            "sha256": hashlib.sha256(binding_payload).hexdigest(),
        },
        "replay": replay_contract,
        "replayable": True,
        "schema": "negpy.dice-acquisition-replay-v1",
        "sources": sources,
    }


def _stable_tiff_array(
    path: str,
    *,
    expected_shape: tuple[int, ...],
    label: str,
) -> np.ndarray:
    before = os.lstat(path)
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise ValueError(f"{label} must be a regular non-symlink file")
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    if hasattr(os, "O_NONBLOCK"):
        flags |= os.O_NONBLOCK
    descriptor = os.open(path, flags)

    def identity(metadata: os.stat_result) -> tuple[int, int, int, int, int]:
        return (
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_size,
            metadata.st_mtime_ns,
            metadata.st_ctime_ns,
        )

    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or identity(opened) != identity(before):
            raise ValueError(f"{label} changed while it was opened")
        try:
            with os.fdopen(os.dup(descriptor), "rb") as stream:
                with tifffile.TiffFile(
                    stream,
                    name=os.path.basename(path),
                ) as image:
                    if len(image.pages) != 1:
                        raise ValueError(f"{label} must contain exactly one page")
                    page = image.pages[0]
                    if tuple(page.shape) != expected_shape or page.dtype != np.uint16:
                        raise ValueError(f"{label} geometry or dtype changed")
                    decoded = np.array(page.asarray(), order="C", copy=True)
        except ValueError:
            raise
        except Exception as error:
            raise ValueError(f"cannot decode {label}: {error}") from error
        after_decode = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    after_path = os.lstat(path)
    if identity(after_decode) != identity(opened) or identity(after_path) != identity(opened):
        raise ValueError(f"{label} changed while it was decoded")
    return decoded


def load_repair_acquisition_evidence(
    binding_path: str | os.PathLike[str],
) -> roll_repair.RepairAcquisition:
    """Rebuild a hash-verified RepairAcquisition from a Tier-1 archive."""

    binding_file = os.path.abspath(os.fspath(binding_path))
    payload = _stable_regular_bytes(
        binding_file,
        maximum_bytes=_MAX_RECEIPT_BYTES,
        label="repair acquisition binding",
    )
    try:
        document = _strict_json_loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"repair acquisition binding is invalid JSON: {error}") from error
    if not isinstance(document, dict):
        raise ValueError("repair acquisition binding must contain an object")
    canonical = (
        json.dumps(
            document,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    if canonical != payload or document.get("schema") != "negpy.dice-acquisition-replay-v1":
        raise ValueError("repair acquisition binding is not canonical or supported")
    acquisition = document.get("acquisition")
    artifacts = document.get("artifacts")
    sources = document.get("sources")
    expected_replay_contract = {
        "authenticity": "integrity-bound-not-signed",
        "complete": True,
        "reconstruction": "stack-upright-storage-rgb-ir-then-rot90-k-1",
        "requires": [
            "storage_rgb_tiff",
            "storage_ir_tiff",
            "prepass_rgbi",
            "ir_validity",
            "acquisition_provenance",
        ],
        "storage_orientation": "upright-storage",
    }
    if not all(isinstance(value, dict) for value in (acquisition, artifacts, sources)):
        raise ValueError("repair acquisition binding sections are malformed")
    if document.get("replay") != expected_replay_contract:
        raise ValueError("repair acquisition replay requirements changed")

    evidence_directory = os.path.dirname(binding_file)
    archive_root = os.path.realpath(os.path.dirname(os.path.dirname(evidence_directory)))

    def row_path(row: object, *, label: str) -> tuple[str, dict[str, object]]:
        if not isinstance(row, dict):
            raise ValueError(f"repair acquisition {label} artifact is malformed")
        relative = row.get("relative_path")
        if type(relative) is not str or os.path.isabs(relative):
            raise ValueError(f"repair acquisition {label} relative path is invalid")
        candidate = os.path.abspath(os.path.join(evidence_directory, relative))
        resolved_parent = os.path.realpath(os.path.dirname(candidate))
        resolved_candidate = os.path.join(
            resolved_parent,
            os.path.basename(candidate),
        )
        try:
            inside = os.path.commonpath((archive_root, resolved_candidate)) == archive_root
        except ValueError:
            inside = False
        if not inside:
            raise ValueError(f"repair acquisition {label} path escapes its archive")
        return candidate, cast(dict[str, object], row)

    def shape(row: dict[str, object], *, label: str) -> tuple[int, ...]:
        values = row.get("shape")
        if (
            not isinstance(values, list)
            or not values
            or len(values) > 3
            or any(type(value) is not int or not 1 <= value <= 10_000 for value in values)
        ):
            raise ValueError(f"repair acquisition {label} shape is invalid")
        result = tuple(cast(list[int], values))
        element_count = 1
        for dimension in result:
            element_count *= dimension
        if element_count > 150_000_000:
            raise ValueError(f"repair acquisition {label} shape is too large")
        return result

    def load_npy(
        row: object,
        *,
        label: str,
        expected_dtype: np.dtype,
    ) -> np.ndarray:
        path, artifact_row = row_path(row, label=label)
        expected_shape = shape(artifact_row, label=label)
        if (label == "prepass RGBI" and (len(expected_shape) != 3 or expected_shape[-1] != 4)) or (
            label == "IR validity" and len(expected_shape) != 2
        ):
            raise ValueError(f"repair acquisition {label} shape is invalid")
        if artifact_row.get("dtype") != expected_dtype.str:
            raise ValueError(f"repair acquisition {label} dtype declaration changed")
        encoded = _stable_regular_bytes(
            path,
            maximum_bytes=_MAX_SHARED_EVIDENCE_BYTES,
            label=f"repair acquisition {label}",
        )
        if artifact_row.get("file_sha256") != hashlib.sha256(encoded).hexdigest():
            raise ValueError(f"repair acquisition {label} file SHA-256 changed")
        try:
            stream = io.BytesIO(encoded)
            version = np.lib.format.read_magic(stream)
            if version == (1, 0):
                header_shape, fortran_order, header_dtype = np.lib.format.read_array_header_1_0(stream)
            elif version == (2, 0):
                header_shape, fortran_order, header_dtype = np.lib.format.read_array_header_2_0(stream)
            else:
                raise ValueError(f"unsupported NPY version {version!r}")
            if tuple(header_shape) != expected_shape or header_dtype != expected_dtype or fortran_order:
                raise ValueError("NPY header geometry or dtype changed")
            expected_bytes = int(np.prod(expected_shape, dtype=np.int64)) * expected_dtype.itemsize
            if len(encoded) - stream.tell() != expected_bytes:
                raise ValueError("NPY byte length changed")
            stream.seek(0)
            decoded = np.load(stream, allow_pickle=False)
        except Exception as error:
            raise ValueError(f"repair acquisition {label} NPY is invalid: {error}") from error
        if not isinstance(decoded, np.ndarray) or decoded.dtype != expected_dtype or decoded.shape != expected_shape:
            raise ValueError(f"repair acquisition {label} array changed")
        canonical = np.array(decoded, dtype=expected_dtype, order="C", copy=True)
        digest_dtype = np.dtype(np.bool_) if expected_dtype == np.dtype(np.bool_) else np.dtype("<u2")
        if (
            artifact_row.get("raw_sha256")
            != hashlib.sha256(memoryview(np.array(canonical, dtype=digest_dtype, order="C", copy=True)).cast("B")).hexdigest()
        ):
            raise ValueError(f"repair acquisition {label} raw SHA-256 changed")
        return canonical

    prepass_row = artifacts.get("prepass_rgbi")
    validity_row = artifacts.get("ir_validity")
    prepass = load_npy(
        prepass_row,
        label="prepass RGBI",
        expected_dtype=np.dtype("<u2"),
    )
    validity = load_npy(
        validity_row,
        label="IR validity",
        expected_dtype=np.dtype(np.bool_),
    )
    rgb_path, rgb_row = row_path(
        sources.get("storage_rgb_tiff"),
        label="storage RGB TIFF",
    )
    ir_path, ir_row = row_path(
        sources.get("storage_ir_tiff"),
        label="storage IR TIFF",
    )
    rgb_shape = shape(rgb_row, label="storage RGB TIFF")
    ir_shape = shape(ir_row, label="storage IR TIFF")
    if len(rgb_shape) != 3 or rgb_shape[-1] != 3 or ir_shape != rgb_shape[:2]:
        raise ValueError("repair acquisition Tier-1 source geometry is invalid")
    for source_row in (rgb_row, ir_row):
        if source_row.get("dtype") != "<u2" or source_row.get("orientation") != "upright-storage":
            raise ValueError("repair acquisition Tier-1 source layout changed")
    rgb = _stable_tiff_array(
        rgb_path,
        expected_shape=rgb_shape,
        label="storage RGB TIFF",
    )
    infrared = _stable_tiff_array(
        ir_path,
        expected_shape=ir_shape,
        label="storage IR TIFF",
    )
    if rgb_row.get("raw_sha256") != _rgb16_sha256(rgb):
        raise ValueError("repair acquisition storage RGB SHA-256 changed")
    if ir_row.get("raw_sha256") != _rgb16_sha256(infrared):
        raise ValueError("repair acquisition storage IR SHA-256 changed")
    native = np.ascontiguousarray(np.rot90(np.dstack((rgb, infrared)), k=-1, axes=(0, 1)))
    if _rgb16_sha256(native) != acquisition.get("main_rgbi_sha256"):
        raise ValueError("repair acquisition main RGBI SHA-256 changed")
    try:
        derived_acquisition_id, derived_evidence_sha256 = _derive_digital_ice_producer_binding(
            slot=acquisition["slot"],
            reservation_id=acquisition["reservation_id"],
            capture_attempt_id=acquisition["capture_attempt_id"],
            main_rgbi=native,
            prepass_rgbi=prepass,
            ir_validity=validity,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"repair acquisition provenance is malformed: {error}") from error
    if acquisition.get("acquisition_id") != derived_acquisition_id:
        raise ValueError("repair acquisition identity is not producer-canonical")
    if acquisition.get("evidence_sha256") != derived_evidence_sha256:
        raise ValueError("repair acquisition producer evidence SHA-256 changed")

    try:
        return roll_repair.RepairAcquisition(
            acquisition_id=acquisition["acquisition_id"],
            slot=acquisition["slot"],
            reservation_id=acquisition["reservation_id"],
            capture_attempt_id=acquisition["capture_attempt_id"],
            storage_transform=acquisition["storage_transform"],
            evidence_sha256=acquisition["evidence_sha256"],
            main_rgbi_sha256=acquisition["main_rgbi_sha256"],
            prepass_rgbi_sha256=acquisition["prepass_rgbi_sha256"],
            ir_validity_sha256=acquisition["ir_validity_sha256"],
            main_rgbi=native,
            prepass_rgbi=prepass,
            ir_validity=validity,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"repair acquisition binding is inconsistent: {error}") from error


def _rgb16_sha256(array: np.ndarray) -> str:
    canonical = np.array(array, dtype="<u2", order="C", copy=True)
    return hashlib.sha256(memoryview(canonical).cast("B")).hexdigest()


def _decode_binary_mask(
    payload: bytes,
    *,
    expected_shape: tuple[int, int],
    label: str,
) -> np.ndarray:
    try:
        with Image.open(io.BytesIO(payload)) as image:
            if image.format != "PNG":
                raise ValueError("not a PNG")
            if image.size != (expected_shape[1], expected_shape[0]):
                raise ValueError("geometry changed")
            decoded = np.asarray(image.convert("L"))
    except Exception as error:
        raise ValueError(f"{label} is not a valid PNG: {error}") from error
    if not np.all(np.isin(np.unique(decoded), np.array([0, 255], dtype=np.uint8))):
        raise ValueError(f"{label} is not binary")
    return np.ascontiguousarray(decoded != 0)


def _validate_repair_result_binding(
    acquisition: roll_repair.RepairAcquisition,
    result: roll_repair.RepairResult,
    *,
    requested_mode: RepairMode,
) -> None:
    """Fail closed before any Tier-2 result or evidence is published."""

    for label, actual, expected in (
        ("acquisition ID", result.acquisition_id, acquisition.acquisition_id),
        ("slot", result.slot, acquisition.slot),
        ("reservation ID", result.reservation_id, acquisition.reservation_id),
        ("evidence SHA-256", result.evidence_sha256, acquisition.evidence_sha256),
        ("requested mode", result.mode_requested, requested_mode),
    ):
        if actual != expected:
            raise ValueError(f"repair result {label} disagrees with its acquisition")
    if requested_mode is RepairMode.EXACT and result.mode_resolved is RepairMode.HYBRID:
        raise ValueError("exact repair request cannot resolve to hybrid")
    if result.mode_resolved not in (RepairMode.EXACT, RepairMode.HYBRID):
        raise ValueError("repair result resolved mode is invalid")
    storage_shape = (
        acquisition.main_rgbi.shape[1],
        acquisition.main_rgbi.shape[0],
        3,
    )
    rgb = np.asarray(result.rgb)
    if rgb.dtype != np.uint16 or rgb.shape != storage_shape or not rgb.flags.c_contiguous:
        raise ValueError("repair result has invalid storage-oriented RGB geometry")
    storage_hash = _rgb16_sha256(rgb)
    if result.storage_output_rgb_sha256 != storage_hash:
        raise ValueError("repair result storage RGB SHA-256 changed")
    native_rgb = np.ascontiguousarray(np.rot90(rgb, k=-1, axes=(0, 1)))
    native_hash = _rgb16_sha256(native_rgb)
    if result.native_output_rgb_sha256 != native_hash:
        raise ValueError("repair result scanner-native RGB SHA-256 changed")

    if result.mode_resolved is not RepairMode.HYBRID:
        forbidden = (
            result.native_synthesis_mask_png,
            result.native_synthesis_mask_sha256,
            result.native_synthesis_mask_shape,
            result.routed_native_synthesis_mask_png,
            result.routed_native_synthesis_mask_sha256,
            result.routed_native_synthesis_mask_shape,
            result.storage_synthesis_mask_png,
            result.storage_synthesis_mask_sha256,
            result.storage_synthesis_mask_shape,
            result.synthesis_mask_transform,
            result.synthesis_fraction,
            result.routing_counts,
            result.hybrid_receipt,
            result.hybrid_receipt_sha256,
            result.hybrid_provenance_class,
            result.hybrid_receipt_output_rgb_sha256,
        )
        if any(value is not None for value in forbidden):
            raise ValueError("non-hybrid repair result carries hybrid evidence")
        return

    required = (
        result.native_synthesis_mask_png,
        result.native_synthesis_mask_sha256,
        result.native_synthesis_mask_shape,
        result.routed_native_synthesis_mask_png,
        result.routed_native_synthesis_mask_sha256,
        result.routed_native_synthesis_mask_shape,
        result.storage_synthesis_mask_png,
        result.storage_synthesis_mask_sha256,
        result.storage_synthesis_mask_shape,
        result.synthesis_mask_transform,
        result.synthesis_fraction,
        result.routing_counts,
        result.hybrid_receipt,
        result.hybrid_receipt_sha256,
        result.hybrid_receipt_output_rgb_sha256,
        result.hybrid_provenance_class,
    )
    if any(value is None for value in required):
        raise ValueError("hybrid repair result omitted disclosure evidence")
    native_png = result.native_synthesis_mask_png
    routed_native_png = result.routed_native_synthesis_mask_png
    storage_png = result.storage_synthesis_mask_png
    receipt = result.hybrid_receipt
    if not all(type(value) is bytes for value in (native_png, routed_native_png, storage_png, receipt)):
        raise ValueError("hybrid disclosure evidence must be immutable bytes")
    if hashlib.sha256(native_png).hexdigest() != result.native_synthesis_mask_sha256:
        raise ValueError("scanner-native disclosure mask SHA-256 changed")
    if hashlib.sha256(storage_png).hexdigest() != result.storage_synthesis_mask_sha256:
        raise ValueError("storage disclosure mask SHA-256 changed")
    if hashlib.sha256(routed_native_png).hexdigest() != result.routed_native_synthesis_mask_sha256:
        raise ValueError("routed scanner-native mask SHA-256 changed")
    if hashlib.sha256(receipt).hexdigest() != result.hybrid_receipt_sha256:
        raise ValueError("hybrid receipt SHA-256 changed")
    if len(result.hybrid_receipt_output_rgb_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in result.hybrid_receipt_output_rgb_sha256
    ):
        raise ValueError("hybrid receipt output RGB SHA-256 is malformed")
    try:
        receipt_document = _strict_json_loads(receipt)
    except (UnicodeDecodeError, ValueError, TypeError) as error:
        raise ValueError(f"hybrid receipt JSON is invalid: {error}") from error
    if not isinstance(receipt_document, dict):
        raise ValueError("hybrid receipt must contain a JSON object")
    try:
        canonical_receipt = (
            json.dumps(
                receipt_document,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode("utf-8")
            + b"\n"
        )
    except (TypeError, ValueError, RecursionError) as error:
        raise ValueError(f"hybrid receipt JSON is not canonical: {error}") from error
    if canonical_receipt != receipt:
        raise ValueError("hybrid receipt JSON is not canonical")
    if receipt_document.get("schema") != "fauxce-hybrid-receipt-v2" or result.hybrid_provenance_class != "caller_asserted_bare_npy":
        raise ValueError("hybrid receipt schema or provenance class changed")
    native_mask = _decode_binary_mask(
        native_png,
        expected_shape=acquisition.main_rgbi.shape[:2],
        label="final scanner-native disclosure mask",
    )
    routed_native_mask = _decode_binary_mask(
        routed_native_png,
        expected_shape=acquisition.main_rgbi.shape[:2],
        label="routed scanner-native disclosure mask",
    )
    storage_mask = _decode_binary_mask(
        storage_png,
        expected_shape=(
            acquisition.main_rgbi.shape[1],
            acquisition.main_rgbi.shape[0],
        ),
        label="storage disclosure mask",
    )
    if (
        tuple(native_mask.shape) != result.native_synthesis_mask_shape
        or native_mask.shape != acquisition.main_rgbi.shape[:2]
        or tuple(routed_native_mask.shape) != result.routed_native_synthesis_mask_shape
        or routed_native_mask.shape != acquisition.main_rgbi.shape[:2]
        or not np.array_equal(
            native_mask,
            routed_native_mask & acquisition.ir_validity,
        )
    ):
        raise ValueError("scanner-native disclosure mask geometry changed")
    expected_storage = acquisition.storage_mask(native_mask)
    if (
        tuple(storage_mask.shape) != result.storage_synthesis_mask_shape
        or not np.array_equal(storage_mask, expected_storage)
        or result.synthesis_mask_transform != acquisition.storage_transform
    ):
        raise ValueError("storage disclosure mask transform binding changed")
    synthesis_pixels = int(np.count_nonzero(native_mask))
    routed_pixels = int(np.count_nonzero(routed_native_mask))
    frame_pixels = int(native_mask.size)
    if result.synthesis_fraction != synthesis_pixels / frame_pixels:
        raise ValueError("hybrid synthesis fraction disagrees with disclosure mask")
    counts = result.routing_counts
    routing = receipt_document.get("routing")
    receipt_counts = routing.get("counts") if isinstance(routing, dict) else None
    count_keys = (
        "final_regions",
        "synthesis_pixels",
        "frame_pixels",
        "at_floor_pixels",
    )
    if (
        not isinstance(counts, dict)
        or not isinstance(receipt_counts, dict)
        or any(type(counts.get(key)) is not int for key in count_keys)
        or any(type(receipt_counts.get(key)) is not int for key in count_keys)
        or any(counts[key] != receipt_counts[key] for key in count_keys)
        or any(counts[key] < 0 for key in count_keys)
        or counts.get("synthesis_pixels") != routed_pixels
        or counts.get("frame_pixels") != frame_pixels
        or counts["at_floor_pixels"] > frame_pixels
        or counts["synthesis_pixels"] > counts["at_floor_pixels"]
        or counts["final_regions"] > counts["synthesis_pixels"]
        or (counts["final_regions"] == 0) != (routed_pixels == 0)
    ):
        raise ValueError("hybrid routing counts disagree with disclosure mask")
    synthesis = receipt_document.get("synthesis")
    if (
        not isinstance(synthesis, dict)
        or synthesis.get("pixel_count") != routed_pixels
        or synthesis.get("frame_pixel_count") != frame_pixels
        or synthesis.get("fraction") != routed_pixels / frame_pixels
        or synthesis.get("within_budget") is not True
    ):
        raise ValueError("hybrid receipt synthesis accounting changed")
    artifacts = receipt_document.get("artifacts")
    output_artifacts = (
        [artifact for artifact in artifacts if isinstance(artifact, dict) and artifact.get("role") == "hybrid_output_rgb16"]
        if isinstance(artifacts, list)
        else []
    )
    composite = receipt_document.get("composite")
    if (
        len(output_artifacts) != 1
        or output_artifacts[0].get("raw_sha256") != result.hybrid_receipt_output_rgb_sha256
        or not isinstance(composite, dict)
        or composite.get("hybrid_rgb16_raw_sha256") != result.hybrid_receipt_output_rgb_sha256
    ):
        raise ValueError("hybrid receipt output binding changed")
    mask_artifacts = (
        [artifact for artifact in artifacts if isinstance(artifact, dict) and artifact.get("role") == "synthesis_mask_png"]
        if isinstance(artifacts, list)
        else []
    )
    routed_u8 = np.ascontiguousarray(routed_native_mask.astype(np.uint8) * np.uint8(255))
    if (
        len(mask_artifacts) != 1
        or mask_artifacts[0].get("file_sha256") != result.routed_native_synthesis_mask_sha256
        or mask_artifacts[0].get("raw_sha256") != hashlib.sha256(memoryview(routed_u8).cast("B")).hexdigest()
        or mask_artifacts[0].get("shape") != list(routed_u8.shape)
        or mask_artifacts[0].get("dtype") != "|u1"
    ):
        raise ValueError("hybrid receipt routed mask binding changed")


def _retain_hybrid_repair_evidence(
    base_path: str,
    acquisition: roll_repair.RepairAcquisition,
    result: roll_repair.RepairResult,
) -> dict[str, dict[str, object]]:
    """Retain verbatim verified native evidence beside user-facing output."""

    evidence_root = os.path.join(
        os.path.dirname(base_path) or ".",
        ".negpy-dice-hybrid",
    )
    evidence_directory = os.path.join(evidence_root, result.hybrid_receipt_sha256)
    if any(os.path.lexists(path) and os.path.islink(path) for path in (evidence_root, evidence_directory)):
        raise OSError("hybrid evidence path is a symlink")
    _prepare_evidence_directory(evidence_directory)
    receipt_path = _atomic_write_bytes(
        os.path.join(evidence_directory, "hybrid-receipt.json"),
        result.hybrid_receipt,
    )
    native_mask_path = _atomic_write_bytes(
        os.path.join(
            evidence_directory,
            "synth-mask-applied-scanner-native.png",
        ),
        result.native_synthesis_mask_png,
    )
    routed_native_mask_path = _atomic_write_bytes(
        os.path.join(
            evidence_directory,
            "synth-mask-routed-scanner-native.png",
        ),
        result.routed_native_synthesis_mask_png,
    )
    binding = {
        "acquisition": {
            "acquisition_id": acquisition.acquisition_id,
            "capture_attempt_id": acquisition.capture_attempt_id,
            "evidence_sha256": acquisition.evidence_sha256,
            "ir_validity_sha256": acquisition.ir_validity_sha256,
            "main_rgbi_sha256": acquisition.main_rgbi_sha256,
            "prepass_rgbi_sha256": acquisition.prepass_rgbi_sha256,
            "reservation_id": acquisition.reservation_id,
            "slot": acquisition.slot,
            "storage_transform": acquisition.storage_transform,
        },
        "hybrid_receipt_sha256": result.hybrid_receipt_sha256,
        "hybrid_receipt_output_rgb_sha256": (result.hybrid_receipt_output_rgb_sha256),
        "native_output_rgb_sha256": result.native_output_rgb_sha256,
        "native_synthesis_mask_sha256": result.native_synthesis_mask_sha256,
        "routed_native_synthesis_mask_sha256": (result.routed_native_synthesis_mask_sha256),
        "provenance_class": result.hybrid_provenance_class,
        "schema": "negpy.dice-hybrid-retained-evidence-v2",
        "storage_output_rgb_sha256": result.storage_output_rgb_sha256,
        "storage_synthesis_mask_sha256": result.storage_synthesis_mask_sha256,
        "synthesis_mask_transform": result.synthesis_mask_transform,
    }
    binding_bytes = json.dumps(
        binding,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    binding_path = _atomic_write_bytes(
        os.path.join(evidence_directory, "negpy-binding.json"),
        binding_bytes,
    )
    return {
        "binding": {
            "bytes": len(binding_bytes),
            "path": binding_path,
            "sha256": hashlib.sha256(binding_bytes).hexdigest(),
        },
        "native_mask": {
            "bytes": len(result.native_synthesis_mask_png),
            "path": native_mask_path,
            "sha256": result.native_synthesis_mask_sha256,
            "shape": list(result.native_synthesis_mask_shape),
        },
        "routed_native_mask": {
            "bytes": len(result.routed_native_synthesis_mask_png),
            "path": routed_native_mask_path,
            "sha256": result.routed_native_synthesis_mask_sha256,
            "shape": list(result.routed_native_synthesis_mask_shape),
        },
        "receipt": {
            "bytes": len(result.hybrid_receipt),
            "path": receipt_path,
            "sha256": result.hybrid_receipt_sha256,
        },
    }


def _ensure_uint16(array: np.ndarray) -> np.ndarray:
    return array if array.dtype == np.uint16 else array.astype(np.uint16)


def _atomic_write_tiff(path: str, array: np.ndarray, *, photometric: str, iccprofile: bytes | None = None) -> str:
    """Write `array` to `path` via a temp file + rename, matching the
    atomic-write convention `negpy.services.scanning.writer` already uses
    for the plain scan path."""
    final_path = path
    transaction = _ACTIVE_OUTPUT_TRANSACTION.get()
    if transaction is not None:
        path = transaction.stage_path(path)
    fd, tmp_path = tempfile.mkstemp(suffix=".tif", dir=os.path.dirname(path) or ".")
    os.close(fd)
    try:
        if iccprofile is None:
            tifffile.imwrite(tmp_path, _ensure_uint16(array), photometric=photometric, compression="lzw")
        else:
            tifffile.imwrite(
                tmp_path,
                _ensure_uint16(array),
                photometric=photometric,
                compression="lzw",
                iccprofile=iccprofile,
            )
        sync_descriptor = os.open(tmp_path, os.O_RDONLY)
        try:
            os.fsync(sync_descriptor)
        finally:
            os.close(sync_descriptor)
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise
    return final_path


def _atomic_write_json(path: str, payload: dict) -> str:
    final_path = path
    transaction = _ACTIVE_OUTPUT_TRANSACTION.get()
    if transaction is not None:
        path = transaction.stage_path(path)
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile("w", dir=os.path.dirname(path) or ".", delete=False, suffix=".part", encoding="utf-8") as tmp:
            tmp_path = tmp.name
            json.dump(payload, tmp, indent=2, allow_nan=False)
            tmp.flush()
            os.fsync(tmp.fileno())
        os.replace(tmp_path, path)
    except Exception:
        if tmp_path is not None and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise
    return final_path


def _atomic_write_bytes(path: str, payload: bytes) -> str:
    final_path = path
    transaction = _ACTIVE_OUTPUT_TRANSACTION.get()
    if transaction is not None:
        path = transaction.stage_path(path)
    fd, tmp_path = tempfile.mkstemp(suffix=".part", dir=os.path.dirname(path) or ".")
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise
    return final_path


def _verify_exact_positive_tiff(
    path: str,
    *,
    expected_rgb: np.ndarray,
    expected_icc: bytes,
) -> dict[str, object]:
    """Reopen the staged exact-positive TIFF and bind its real container."""

    transaction = _ACTIVE_OUTPUT_TRANSACTION.get()
    actual_path = transaction.staged_path(path) if transaction is not None else path
    before = os.lstat(actual_path)
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise roll_exact_color.ExactColorIntegrityError("exact-positive TIFF is not a regular file")
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    if hasattr(os, "O_NONBLOCK"):
        flags |= os.O_NONBLOCK
    descriptor = os.open(actual_path, flags)
    file_digest = hashlib.sha256()
    byte_count = 0
    try:
        opened = os.fstat(descriptor)

        def identity(metadata: os.stat_result) -> tuple[int, int, int, int, int]:
            return (
                metadata.st_dev,
                metadata.st_ino,
                metadata.st_size,
                metadata.st_mtime_ns,
                metadata.st_ctime_ns,
            )

        if not stat.S_ISREG(opened.st_mode) or identity(opened) != identity(before):
            raise roll_exact_color.ExactColorIntegrityError("exact-positive TIFF changed while it was opened")
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            file_digest.update(chunk)
            byte_count += len(chunk)
        after_read = os.fstat(descriptor)
        if identity(after_read) != identity(opened) or byte_count != opened.st_size:
            raise roll_exact_color.ExactColorIntegrityError("exact-positive TIFF size changed while it was verified")

        expected = np.asarray(expected_rgb)
        expected_pixel_sha256 = roll_exact_color.rgb16_content_sha256(expected)
        try:
            # Decode through a duplicate of the descriptor whose bytes were
            # just hashed. Reopening the pathname here would let a swap bind
            # one file's hash to another file's pixels and ICC profile.
            os.lseek(descriptor, 0, os.SEEK_SET)
            with os.fdopen(os.dup(descriptor), "rb") as stream:
                # ``fdopen`` exposes the integer descriptor as ``.name``;
                # tifffile normalizes that as a path unless given a benign
                # display hint. The stream remains the descriptor-pinned
                # source of every decoded byte.
                with tifffile.TiffFile(
                    stream,
                    name=os.path.basename(actual_path),
                ) as image:
                    if len(image.pages) != 1:
                        raise roll_exact_color.ExactColorIntegrityError("exact-positive TIFF must contain exactly one page")
                    page = image.pages[0]
                    decoded = np.asarray(page.asarray())
                    samples_per_pixel = int(page.samplesperpixel)
                    bits_value = page.tags[258].value
                    if isinstance(bits_value, (tuple, list)):
                        bits = tuple(int(value) for value in bits_value)
                    else:
                        bits = (int(bits_value),) * samples_per_pixel
                    photometric = int(page.photometric)
                    planar = int(page.planarconfig)
                    orientation_tag = page.tags.get(274)
                    orientation = 1 if orientation_tag is None else int(orientation_tag.value)
                    icc_tag = page.tags.get(34675)
                    icc = None if icc_tag is None else bytes(icc_tag.value)
        except roll_exact_color.ExactColorIntegrityError:
            raise
        except Exception as error:
            raise roll_exact_color.ExactColorIntegrityError(f"cannot verify exact-positive TIFF: {error}") from error
        after_decode = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    after_path = os.lstat(actual_path)
    if identity(after_decode) != identity(opened) or identity(after_path) != identity(opened) or not stat.S_ISREG(after_path.st_mode):
        raise roll_exact_color.ExactColorIntegrityError("exact-positive TIFF changed while it was verified")
    if (
        decoded.dtype != np.uint16
        or decoded.shape != expected.shape
        or expected.shape[-1:] != (3,)
        or samples_per_pixel != 3
        or photometric != 2
        or planar != 1
        or orientation != 1
        or bits != (16, 16, 16)
    ):
        raise roll_exact_color.ExactColorIntegrityError("exact-positive TIFF layout is not one contiguous 16-bit RGB image")
    pixel_sha256 = roll_exact_color.rgb16_content_sha256(decoded)
    if pixel_sha256 != expected_pixel_sha256 or not np.array_equal(decoded, expected):
        raise roll_exact_color.ExactColorIntegrityError("exact-positive TIFF pixels changed during encoding")
    if (
        icc is None
        or len(icc) != len(expected_icc)
        or icc != expected_icc
        or hashlib.sha256(icc).hexdigest() != hashlib.sha256(expected_icc).hexdigest()
    ):
        raise roll_exact_color.ExactColorIntegrityError("exact-positive TIFF Nikon ICC tag is missing or changed")
    return {
        "bits_per_sample": list(bits),
        "bytes": byte_count,
        "dtype": str(decoded.dtype),
        "file_sha256": file_digest.hexdigest(),
        "icc_bytes": len(icc),
        "icc_sha256": hashlib.sha256(icc).hexdigest(),
        "page_count": 1,
        "orientation": "top-left",
        "photometric": "rgb",
        "planar_config": "contiguous",
        "pixel_sha256": pixel_sha256,
        "samples_per_pixel": 3,
        "shape": list(decoded.shape),
    }


def _build_native_receipt_from_frame(
    frame: object,
) -> roll_exact_color.NativeValidatedBuilderReceipt:
    evidence = getattr(frame, "nikon_exact_builder_evidence", None)
    if evidence is None:
        if getattr(frame, "nikon_density_evidence", None) is not None or getattr(frame, "nikon_density_ownership", None) is not None:
            raise roll_exact_color.ExactColorUnavailable("frame-bound Nikon density evidence has no native builder evidence")
        raise roll_exact_color.ExactColorUnavailable(
            "validated Stage-3 builder receipt is not supplied and frame has no native builder evidence"
        )
    evidence = roll_native_builder.adapt_native_builder_evidence(evidence)
    if evidence.slot != getattr(frame, "slot", None):
        raise roll_exact_color.ExactColorUnavailable("frame native builder evidence belongs to a different slot")
    public_receipt = getattr(frame, "receipt", None)
    ownership = getattr(frame, "nikon_density_ownership", None)
    receipt_ownership = None if public_receipt is None else getattr(public_receipt, "nikon_density_ownership", None)
    if ownership is None:
        raise roll_exact_color.ExactColorUnavailable("Nikon density frame ownership receipt is missing")
    if receipt_ownership is None:
        raise roll_exact_color.ExactColorUnavailable("public frame receipt has no Nikon density ownership")
    frame_density_evidence = getattr(frame, "nikon_density_evidence", None)
    if frame_density_evidence is None:
        raise roll_exact_color.ExactColorUnavailable("frame-bound Nikon density evidence is missing")
    ownership_payload = _canonical_component_payload(ownership, label="Nikon density frame ownership")
    receipt_ownership_payload = _canonical_component_payload(
        receipt_ownership,
        label="public-frame Nikon density ownership",
    )
    if receipt_ownership_payload != ownership_payload:
        raise roll_exact_color.ExactColorUnavailable("frame and public receipt disagree on Nikon density ownership")
    density_payload = _canonical_component_payload(frame_density_evidence, label="frame-bound Nikon density evidence")
    if ownership_payload != evidence.frame_ownership_receipt:
        raise roll_exact_color.ExactColorUnavailable("Nikon density frame ownership does not match the native builder evidence")
    if density_payload != evidence.density_evidence_receipt:
        raise roll_exact_color.ExactColorUnavailable("frame-bound Nikon density evidence does not match the native builder evidence")
    validator = getattr(ownership, "validate_evidence", None)
    if callable(validator):
        try:
            validator(frame_density_evidence)
        except (TypeError, ValueError) as error:
            raise roll_exact_color.ExactColorUnavailable(f"Nikon density frame ownership is invalid: {error}") from error
    return roll_native_builder.build_native_builder_receipt(evidence)


def _canonical_component_payload(component: object, *, label: str) -> bytes:
    if type(component) is dict:
        payload = component
    else:
        producer = getattr(component, "to_dict", None)
        if not callable(producer):
            producer = getattr(component, "to_payload", None)
        if callable(producer):
            payload = producer()
        else:
            raise roll_exact_color.ExactColorUnavailable(f"{label} has no canonical receipt payload")
    if type(payload) is not dict:
        raise roll_exact_color.ExactColorUnavailable(f"{label} payload is malformed")
    try:
        return json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, RecursionError) as error:
        raise roll_exact_color.ExactColorUnavailable(f"{label} payload is not canonical JSON: {error}") from error


def _retain_builder_evidence(
    base_path: str,
    receipt: roll_exact_color.BuilderReceipt,
) -> dict[str, object]:
    if isinstance(receipt, roll_exact_color.ValidatedBuilderReceipt):
        return _retain_stage3_replay_evidence(base_path, receipt)
    if isinstance(receipt, roll_exact_color.NativeValidatedBuilderReceipt):
        return _retain_native_builder_evidence(base_path, receipt)
    raise roll_exact_color.ExactColorIntegrityError("builder receipt has an invalid type")


def _retain_stage3_replay_evidence(
    base_path: str,
    receipt: roll_exact_color.ValidatedBuilderReceipt,
) -> dict[str, object]:
    """Atomically retain the raw replay inputs consumed by exact Tier 3."""

    roll_exact_color.builder_receipt_payload(receipt)
    evidence_root = os.path.join(os.path.dirname(base_path) or ".", ".negpy-stage3-replay")
    evidence_directory = os.path.join(evidence_root, receipt.sha256)
    try:
        if any(os.path.lexists(path) and os.path.islink(path) for path in (evidence_root, evidence_directory)):
            raise OSError("evidence path is a symlink")
        _prepare_evidence_directory(evidence_directory)
        report_path = _atomic_write_bytes(
            os.path.join(evidence_directory, "stage3-validation.json"),
            receipt.stage3_receipt,
        )
    except OSError as error:
        raise roll_exact_color.ExactColorUnavailable(f"cannot retain Stage-3 replay evidence: {error}") from error
    report_row = {
        "bytes": len(receipt.stage3_receipt),
        "path": report_path,
        "sha256": receipt.stage3_receipt_sha256,
    }
    lut_rows = []
    for channel, blob, digest in zip(
        ("r", "g", "b"),
        receipt.pre_f_luts,
        receipt.pre_f_lut_sha256,
        strict=True,
    ):
        try:
            path = _atomic_write_bytes(os.path.join(evidence_directory, f"builder-preF-{channel}.bin"), blob)
        except OSError as error:
            raise roll_exact_color.ExactColorUnavailable(f"cannot retain Stage-3 replay evidence: {error}") from error
        lut_rows.append(
            {
                "bytes": len(blob),
                "channel": channel,
                "path": path,
                "sha256": digest,
            }
        )
    return {
        "native_per_acquisition_builder": False,
        "pre_f_luts": lut_rows,
        "scope": roll_exact_color.STAGE3_REPLAY_SCOPE,
        "stage3_report": report_row,
    }


def _retain_native_builder_evidence(
    base_path: str,
    receipt: roll_exact_color.NativeValidatedBuilderReceipt,
) -> dict[str, object]:
    """Atomically retain the native evidence snapshot and derived pre-F LUTs."""

    roll_exact_color.builder_receipt_payload(receipt)
    evidence_root = os.path.join(os.path.dirname(base_path) or ".", ".negpy-native-builder")
    evidence_directory = os.path.join(evidence_root, receipt.sha256)
    try:
        if any(os.path.lexists(path) and os.path.islink(path) for path in (evidence_root, evidence_directory)):
            raise OSError("evidence path is a symlink")
        _prepare_evidence_directory(evidence_directory)
        builder_receipt_path = _atomic_write_bytes(
            os.path.join(evidence_directory, "native-builder-receipt.json"),
            receipt.payload,
        )
        evidence_path = _atomic_write_bytes(
            os.path.join(evidence_directory, "native-builder-evidence.json"),
            receipt.evidence_payload,
        )
        ownership_path = _atomic_write_bytes(
            os.path.join(evidence_directory, "nikon-density-frame-ownership.json"),
            receipt.frame_ownership_receipt,
        )
        density_path = _atomic_write_bytes(
            os.path.join(evidence_directory, "nikon-density-evidence.json"),
            receipt.density_evidence_receipt,
        )
        analyzer_path = _atomic_write_bytes(
            os.path.join(evidence_directory, "analyzer-rgb-u16le.bin"),
            receipt.analyzer_rgb,
        )
    except OSError as error:
        raise roll_exact_color.ExactColorUnavailable(f"cannot retain native builder evidence: {error}") from error
    lut_rows = []
    for channel, blob, digest in zip(("r", "g", "b"), receipt.pre_f_luts, receipt.pre_f_lut_sha256, strict=True):
        try:
            path = _atomic_write_bytes(os.path.join(evidence_directory, f"builder-preF-{channel}.bin"), blob)
        except OSError as error:
            raise roll_exact_color.ExactColorUnavailable(f"cannot retain native builder evidence: {error}") from error
        lut_rows.append({"bytes": len(blob), "channel": channel, "path": path, "sha256": digest})
    return {
        "builder_receipt": {
            "bytes": len(receipt.payload),
            "path": builder_receipt_path,
            "sha256": receipt.sha256,
        },
        "analyzer_rgb": {
            "bytes": len(receipt.analyzer_rgb),
            "path": analyzer_path,
            "sha256": receipt.analyzer_rgb_sha256,
            "shape": list(receipt.analyzer_shape),
        },
        "evidence_receipt": {
            "bytes": len(receipt.evidence_payload),
            "path": evidence_path,
            "sha256": receipt.evidence_sha256,
        },
        "frame_ownership_receipt": {
            "bytes": len(receipt.frame_ownership_receipt),
            "path": ownership_path,
            "sha256": receipt.frame_ownership_receipt_sha256,
        },
        "density_evidence_receipt": {
            "bytes": len(receipt.density_evidence_receipt),
            "path": density_path,
            "sha256": receipt.density_evidence_receipt_sha256,
        },
        "native_per_acquisition_builder": True,
        "pre_f_luts": lut_rows,
        "scope": roll_exact_color.NATIVE_BUILDER_SCOPE,
    }
