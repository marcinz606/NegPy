"""Crash-visible, fail-closed ownership primitives for a live roll run.

This module is deliberately hardware-inert.  It owns only two filesystem
names: the caller-selected run receipt and a fixed lock inside an output
directory.  A live runner can reserve both before opening a scanner, retain
their descriptors for the complete run, and refuse to overwrite names whose
ownership changes underneath it.
"""

from __future__ import annotations

import dataclasses
import json
import os
import stat
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any, Self


OUTPUT_LOCK_NAME = ".negpy-live-acceptance.lock"
_IGNORABLE_FINDER_METADATA_NAMES = frozenset({".DS_Store"})
DEFAULT_RECEIPT_MAX_BYTES = 1024 * 1024
DEFAULT_LOCK_MAX_BYTES = 16 * 1024


class LiveReservationError(RuntimeError):
    """A filesystem ownership boundary could not be established or retained."""


class ReservationConflict(LiveReservationError):
    """The requested receipt or fixed output lock already exists."""


class OwnershipLost(LiveReservationError):
    """A reserved pathname no longer identifies the object we opened."""


class InventoryConflict(LiveReservationError):
    """The output directory contains missing, changed, or unowned entries."""


@dataclasses.dataclass(frozen=True)
class FileIdentity:
    """Stable-enough checkpoint identity for one regular output file."""

    device: int
    inode: int
    size: int
    mtime_ns: int
    ctime_ns: int

    @classmethod
    def from_stat(cls, metadata: os.stat_result) -> Self:
        return cls(
            device=metadata.st_dev,
            inode=metadata.st_ino,
            size=metadata.st_size,
            mtime_ns=metadata.st_mtime_ns,
            ctime_ns=metadata.st_ctime_ns,
        )

    @property
    def object_identity(self) -> tuple[int, int]:
        return self.device, self.inode


@dataclasses.dataclass(frozen=True)
class InventorySnapshot:
    """Exact file/directory names observed at an ownership checkpoint."""

    files: tuple[tuple[str, FileIdentity], ...]
    directories: tuple[str, ...]


def canonical_json_bytes(
    document: Mapping[str, Any],
    *,
    maximum_bytes: int,
    label: str,
) -> bytes:
    """Encode one bounded canonical JSON object, including its final newline."""

    if maximum_bytes <= 0:
        raise ValueError("maximum_bytes must be positive")
    if not isinstance(document, Mapping):
        raise LiveReservationError(f"{label} must be a JSON object")
    try:
        payload = (
            json.dumps(
                dict(document),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode("utf-8")
            + b"\n"
        )
    except (TypeError, ValueError) as error:
        raise LiveReservationError(f"{label} is not canonical JSON: {error}") from error
    if len(payload) > maximum_bytes:
        raise LiveReservationError(f"{label} exceeds its safe size limit ({len(payload)} > {maximum_bytes} bytes)")
    return payload


def _require_posix_descriptor_operations() -> None:
    missing: list[str] = []
    for name in ("O_DIRECTORY", "O_NOFOLLOW"):
        if not hasattr(os, name):
            missing.append(name)
    for function, capability, name in (
        (os.open, os.supports_dir_fd, "open(dir_fd)"),
        (os.stat, os.supports_dir_fd, "stat(dir_fd)"),
        (os.stat, os.supports_follow_symlinks, "stat(follow_symlinks)"),
        (os.unlink, os.supports_dir_fd, "unlink(dir_fd)"),
        (os.listdir, os.supports_fd, "listdir(fd)"),
    ):
        if function not in capability:
            missing.append(name)
    if missing:
        raise LiveReservationError("safe POSIX descriptor operations are unavailable: " + ", ".join(missing))


def _directory_flags() -> int:
    return os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)


def _file_create_flags() -> int:
    return os.O_CREAT | os.O_EXCL | os.O_RDWR | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)


def _same_object(left: os.stat_result, right: os.stat_result) -> bool:
    return (left.st_dev, left.st_ino) == (right.st_dev, right.st_ino)


def _directory_checkpoint(metadata: os.stat_result) -> tuple[int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def is_ignorable_finder_metadata_file(name: str, metadata: os.stat_result) -> bool:
    """Return whether ``name`` is macOS directory metadata, never an artifact.

    Finder creates a regular ``.DS_Store`` merely by browsing a directory. It
    has no capture or output provenance, so it must not make a held receipt
    fail. Keep the exception exact: a directory, symlink, or any other hidden
    name remains an inventory conflict.
    """

    return name in _IGNORABLE_FINDER_METADATA_NAMES and stat.S_ISREG(metadata.st_mode)


def _write_all(descriptor: int, payload: bytes) -> None:
    written = 0
    while written < len(payload):
        count = os.write(descriptor, payload[written:])
        if count <= 0:
            raise OSError("short write while publishing reserved JSON")
        written += count


def _read_exact(descriptor: int, size: int) -> bytes:
    chunks: list[bytes] = []
    offset = 0
    while offset < size:
        chunk = os.pread(descriptor, size - offset, offset)
        if not chunk:
            break
        chunks.append(chunk)
        offset += len(chunk)
    return b"".join(chunks)


def _replace_descriptor_payload(descriptor: int, payload: bytes) -> None:
    os.lseek(descriptor, 0, os.SEEK_SET)
    _write_all(descriptor, payload)
    os.ftruncate(descriptor, len(payload))
    os.fsync(descriptor)
    if _read_exact(descriptor, len(payload)) != payload:
        raise OSError("reserved JSON did not verify after publication")


def _open_visible_directory(path: Path, *, label: str) -> tuple[Path, int]:
    _require_posix_descriptor_operations()
    canonical = path.resolve(strict=True)
    descriptor = os.open(canonical, _directory_flags())
    try:
        opened = os.fstat(descriptor)
        visible = os.stat(canonical, follow_symlinks=False)
        if not stat.S_ISDIR(opened.st_mode) or not stat.S_ISDIR(visible.st_mode) or not _same_object(opened, visible):
            raise OwnershipLost(f"{label} changed while opening")
    except BaseException:
        os.close(descriptor)
        raise
    return canonical, descriptor


def _assert_visible_directory(path: Path, descriptor: int, *, label: str) -> None:
    try:
        opened = os.fstat(descriptor)
        visible = os.stat(path, follow_symlinks=False)
    except OSError as error:
        raise OwnershipLost(f"{label} is no longer visible: {error}") from error
    if not stat.S_ISDIR(opened.st_mode) or not stat.S_ISDIR(visible.st_mode) or not _same_object(opened, visible):
        raise OwnershipLost(f"{label} pathname no longer identifies the opened directory")


def _child_metadata(directory_descriptor: int, name: str) -> os.stat_result:
    return os.stat(
        name,
        dir_fd=directory_descriptor,
        follow_symlinks=False,
    )


class ExclusiveReceiptReservation:
    """An exact run-receipt pathname reserved before any live device open."""

    def __init__(
        self,
        *,
        path: Path,
        parent: Path,
        parent_descriptor: int,
        descriptor: int,
        maximum_bytes: int,
    ) -> None:
        self.path = path
        self._parent = parent
        self._parent_descriptor = parent_descriptor
        self._descriptor = descriptor
        self._maximum_bytes = maximum_bytes
        self._closed = False

    @classmethod
    def reserve(
        cls,
        path: Path,
        initial_document: Mapping[str, Any],
        *,
        maximum_bytes: int = DEFAULT_RECEIPT_MAX_BYTES,
    ) -> Self:
        """Exclusively create and durably publish an in-progress receipt."""

        if initial_document.get("status") != "in_progress":
            raise LiveReservationError("initial run receipt status must be exactly 'in_progress'")
        payload = canonical_json_bytes(
            initial_document,
            maximum_bytes=maximum_bytes,
            label="run receipt",
        )
        absolute = Path(os.path.abspath(os.fspath(path)))
        if absolute.name in {"", ".", ".."}:
            raise LiveReservationError("run receipt must name a file")
        parent, parent_descriptor = _open_visible_directory(
            absolute.parent,
            label="run-receipt parent",
        )
        canonical = parent / absolute.name
        descriptor: int | None = None
        try:
            try:
                descriptor = os.open(
                    absolute.name,
                    _file_create_flags(),
                    0o600,
                    dir_fd=parent_descriptor,
                )
            except FileExistsError as error:
                raise ReservationConflict(f"run receipt already exists: {canonical}") from error
            opened = os.fstat(descriptor)
            visible = _child_metadata(parent_descriptor, absolute.name)
            if not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1 or not _same_object(opened, visible):
                raise OwnershipLost("new run receipt did not retain exclusive regular-file ownership")
            _replace_descriptor_payload(descriptor, payload)
            os.fsync(parent_descriptor)
            reservation = cls(
                path=canonical,
                parent=parent,
                parent_descriptor=parent_descriptor,
                descriptor=descriptor,
                maximum_bytes=maximum_bytes,
            )
            reservation.assert_owned()
            return reservation
        except BaseException:
            if descriptor is not None:
                try:
                    opened = os.fstat(descriptor)
                    visible = _child_metadata(parent_descriptor, absolute.name)
                    if _same_object(opened, visible):
                        os.unlink(absolute.name, dir_fd=parent_descriptor)
                        os.fsync(parent_descriptor)
                except OSError:
                    pass
                os.close(descriptor)
            os.close(parent_descriptor)
            raise

    @property
    def inode(self) -> tuple[int, int]:
        self._require_open()
        metadata = os.fstat(self._descriptor)
        return metadata.st_dev, metadata.st_ino

    def _require_open(self) -> None:
        if self._closed:
            raise LiveReservationError("run-receipt reservation is closed")

    def assert_owned(self) -> None:
        """Require the requested pathname to still name our held receipt inode."""

        self._require_open()
        _assert_visible_directory(
            self._parent,
            self._parent_descriptor,
            label="run-receipt parent",
        )
        try:
            opened = os.fstat(self._descriptor)
            visible = _child_metadata(self._parent_descriptor, self.path.name)
        except OSError as error:
            raise OwnershipLost(f"run receipt is no longer owned: {error}") from error
        if (
            not stat.S_ISREG(opened.st_mode)
            or not stat.S_ISREG(visible.st_mode)
            or opened.st_nlink != 1
            or visible.st_nlink != 1
            or not _same_object(opened, visible)
        ):
            raise OwnershipLost("run-receipt pathname no longer identifies the reserved inode")

    def publish(self, document: Mapping[str, Any]) -> None:
        """Publish new canonical JSON into, and only into, the reserved inode."""

        payload = canonical_json_bytes(
            document,
            maximum_bytes=self._maximum_bytes,
            label="run receipt",
        )
        self.assert_owned()
        _replace_descriptor_payload(self._descriptor, payload)
        self.assert_owned()
        os.fsync(self._parent_descriptor)

    def close(self) -> None:
        """Release descriptors without deleting the durable receipt."""

        if self._closed:
            return
        os.close(self._descriptor)
        os.close(self._parent_descriptor)
        self._closed = True

    def __enter__(self) -> Self:
        self._require_open()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


class FixedOutputLease:
    """A fixed, crash-visible output-directory lease held for one live run."""

    def __init__(
        self,
        *,
        output_dir: Path,
        directory_descriptor: int,
        lock_descriptor: int,
    ) -> None:
        self.output_dir = output_dir
        self.lock_path = output_dir / OUTPUT_LOCK_NAME
        self._directory_descriptor = directory_descriptor
        self._lock_descriptor = lock_descriptor
        self._released = False

    @classmethod
    def acquire(
        cls,
        output_dir: Path,
        lock_document: Mapping[str, Any],
        *,
        maximum_bytes: int = DEFAULT_LOCK_MAX_BYTES,
        require_empty: bool = True,
    ) -> Self:
        """Exclusively own a real output directory through a fixed lock file."""

        payload = canonical_json_bytes(
            lock_document,
            maximum_bytes=maximum_bytes,
            label="output lock",
        )
        absolute = Path(os.path.abspath(os.fspath(output_dir)))
        try:
            linked = absolute.lstat()
        except OSError as error:
            raise LiveReservationError(f"could not inspect output directory: {error}") from error
        if stat.S_ISLNK(linked.st_mode) or not stat.S_ISDIR(linked.st_mode):
            raise LiveReservationError("output directory must be an existing non-symlink directory")
        canonical, directory_descriptor = _open_visible_directory(
            absolute,
            label="output directory",
        )
        lock_descriptor: int | None = None
        try:
            try:
                lock_descriptor = os.open(
                    OUTPUT_LOCK_NAME,
                    _file_create_flags(),
                    0o600,
                    dir_fd=directory_descriptor,
                )
            except FileExistsError as error:
                raise ReservationConflict(f"output directory is already reserved: {canonical / OUTPUT_LOCK_NAME}") from error
            opened = os.fstat(lock_descriptor)
            visible = _child_metadata(directory_descriptor, OUTPUT_LOCK_NAME)
            if not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1 or not _same_object(opened, visible):
                raise OwnershipLost("new output lock did not retain exclusive regular-file ownership")
            _replace_descriptor_payload(lock_descriptor, payload)
            os.fsync(directory_descriptor)
            lease = cls(
                output_dir=canonical,
                directory_descriptor=directory_descriptor,
                lock_descriptor=lock_descriptor,
            )
            lease.assert_owned()
            if require_empty:
                lease.assert_inventory(())
            return lease
        except BaseException:
            if lock_descriptor is not None:
                try:
                    opened = os.fstat(lock_descriptor)
                    visible = _child_metadata(directory_descriptor, OUTPUT_LOCK_NAME)
                    if _same_object(opened, visible):
                        os.unlink(OUTPUT_LOCK_NAME, dir_fd=directory_descriptor)
                        os.fsync(directory_descriptor)
                except OSError:
                    pass
                os.close(lock_descriptor)
            os.close(directory_descriptor)
            raise

    @property
    def released(self) -> bool:
        return self._released

    def _require_active(self) -> None:
        if self._released:
            raise LiveReservationError("output lease is already released")

    def _lock_metadata(self) -> tuple[os.stat_result, os.stat_result]:
        try:
            opened = os.fstat(self._lock_descriptor)
            visible = _child_metadata(self._directory_descriptor, OUTPUT_LOCK_NAME)
        except OSError as error:
            raise OwnershipLost(f"output lock is no longer owned: {error}") from error
        return opened, visible

    def assert_owned(self) -> None:
        """Require both the output pathname and fixed lock to remain ours."""

        self._require_active()
        _assert_visible_directory(
            self.output_dir,
            self._directory_descriptor,
            label="output directory",
        )
        opened, visible = self._lock_metadata()
        if (
            not stat.S_ISREG(opened.st_mode)
            or not stat.S_ISREG(visible.st_mode)
            or opened.st_nlink != 1
            or visible.st_nlink != 1
            or not _same_object(opened, visible)
        ):
            raise OwnershipLost("fixed output-lock pathname no longer identifies the held inode")

    def _relative_entry(self, path: Path | str, *, label: str) -> str:
        candidate = Path(path)
        if candidate.is_absolute():
            absolute = Path(os.path.abspath(os.fspath(candidate)))
        else:
            absolute = Path(os.path.abspath(os.fspath(self.output_dir / candidate)))
        try:
            relative = absolute.relative_to(self.output_dir)
        except ValueError as error:
            raise InventoryConflict(f"{label} escapes the output directory: {path}") from error
        if relative == Path(".") or any(part in {"", ".", ".."} for part in relative.parts):
            raise InventoryConflict(f"{label} does not name a child entry: {path}")
        name = os.fspath(relative)
        if name == OUTPUT_LOCK_NAME:
            raise InventoryConflict("the fixed output lock cannot be declared as an output")
        if Path(name).name in _IGNORABLE_FINDER_METADATA_NAMES:
            raise InventoryConflict("Finder metadata cannot be declared as an output")
        return name

    def _scan_directory(
        self,
        descriptor: int,
        *,
        prefix: str,
        files: dict[str, FileIdentity],
        directories: set[str],
    ) -> None:
        before = os.fstat(descriptor)
        if not stat.S_ISDIR(before.st_mode):
            raise InventoryConflict("an opened output entry stopped being a directory")
        try:
            names = sorted(os.listdir(descriptor))
        except OSError as error:
            raise InventoryConflict(f"could not list output directory: {error}") from error
        for name in names:
            relative = os.path.join(prefix, name) if prefix else name
            try:
                metadata = _child_metadata(descriptor, name)
            except OSError as error:
                raise InventoryConflict(f"output entry changed while inspecting {relative}: {error}") from error
            if is_ignorable_finder_metadata_file(name, metadata):
                continue
            if stat.S_ISLNK(metadata.st_mode):
                raise InventoryConflict(f"output entry is a symbolic link: {relative}")
            if stat.S_ISREG(metadata.st_mode):
                files[relative] = FileIdentity.from_stat(metadata)
                continue
            if not stat.S_ISDIR(metadata.st_mode):
                raise InventoryConflict(f"output entry is not a regular file or directory: {relative}")
            directories.add(relative)
            try:
                child_descriptor = os.open(
                    name,
                    _directory_flags(),
                    dir_fd=descriptor,
                )
            except OSError as error:
                raise InventoryConflict(f"could not open output directory {relative}: {error}") from error
            try:
                opened = os.fstat(child_descriptor)
                if not _same_object(opened, metadata):
                    raise InventoryConflict(f"output directory changed while opening: {relative}")
                self._scan_directory(
                    child_descriptor,
                    prefix=relative,
                    files=files,
                    directories=directories,
                )
            finally:
                os.close(child_descriptor)
        after = os.fstat(descriptor)
        if _directory_checkpoint(before) != _directory_checkpoint(after):
            raise InventoryConflict("output directory changed while its inventory was inspected")

    @staticmethod
    def _ancestor_directories(entries: Iterable[str]) -> set[str]:
        ancestors: set[str] = set()
        for entry in entries:
            parent = os.path.dirname(entry)
            while parent:
                ancestors.add(parent)
                parent = os.path.dirname(parent)
        return ancestors

    def assert_inventory(
        self,
        owned_files: Iterable[Path | str],
        *,
        owned_directories: Iterable[Path | str] = (),
        previous: InventorySnapshot | None = None,
    ) -> InventorySnapshot:
        """Require an exact, regular, non-symlink output inventory.

        ``owned_files`` and ``owned_directories`` exclude the fixed lease file;
        it is included automatically.  Passing the prior checkpoint also
        refuses mutation or replacement of every previously owned file.
        """

        return self._assert_inventory(
            owned_files,
            owned_directories=owned_directories,
            previous=previous,
            include_lock=True,
        )

    def _assert_inventory(
        self,
        owned_files: Iterable[Path | str],
        *,
        owned_directories: Iterable[Path | str] = (),
        previous: InventorySnapshot | None = None,
        include_lock: bool,
    ) -> InventorySnapshot:
        """Check the exact held-directory inventory before or after unlock."""

        self._require_active()
        if include_lock:
            self.assert_owned()
        else:
            _assert_visible_directory(
                self.output_dir,
                self._directory_descriptor,
                label="output directory",
            )
        expected_files = {self._relative_entry(path, label="owned file") for path in owned_files}
        expected_directories = {self._relative_entry(path, label="owned directory") for path in owned_directories}
        if include_lock:
            expected_files.add(OUTPUT_LOCK_NAME)
        expected_directories.update(self._ancestor_directories(expected_files))
        expected_directories.update(self._ancestor_directories(expected_directories))

        actual_files: dict[str, FileIdentity] = {}
        actual_directories: set[str] = set()
        self._scan_directory(
            self._directory_descriptor,
            prefix="",
            files=actual_files,
            directories=actual_directories,
        )
        if include_lock:
            self.assert_owned()
        else:
            _assert_visible_directory(
                self.output_dir,
                self._directory_descriptor,
                label="output directory",
            )

        missing_files = sorted(expected_files - actual_files.keys())
        unexpected_files = sorted(actual_files.keys() - expected_files)
        missing_directories = sorted(expected_directories - actual_directories)
        unexpected_directories = sorted(actual_directories - expected_directories)
        differences: list[str] = []
        for label, entries in (
            ("missing files", missing_files),
            ("unexpected files", unexpected_files),
            ("missing directories", missing_directories),
            ("unexpected directories", unexpected_directories),
        ):
            if entries:
                differences.append(f"{label}: {', '.join(entries)}")
        if differences:
            raise InventoryConflict("output inventory conflict (" + "; ".join(differences) + ")")

        if previous is not None:
            for name, prior_identity in previous.files:
                if not include_lock and name == OUTPUT_LOCK_NAME:
                    continue
                current_identity = actual_files.get(name)
                if current_identity is None:
                    raise InventoryConflict(f"previously owned output disappeared: {name}")
                if current_identity != prior_identity:
                    raise InventoryConflict(f"previously owned output changed: {name}")

        return InventorySnapshot(
            files=tuple(sorted(actual_files.items())),
            directories=tuple(sorted(actual_directories)),
        )

    def _unlink_owned_lock(self) -> None:
        opened, visible = self._lock_metadata()
        if (
            not stat.S_ISREG(opened.st_mode)
            or not stat.S_ISREG(visible.st_mode)
            or opened.st_nlink != 1
            or visible.st_nlink != 1
            or not _same_object(opened, visible)
        ):
            raise OwnershipLost("refusing to remove a fixed output lock that is no longer exclusively owned")
        os.unlink(OUTPUT_LOCK_NAME, dir_fd=self._directory_descriptor)
        os.fsync(self._directory_descriptor)

    def release_verified(
        self,
        owned_files: Iterable[Path | str],
        *,
        previous: InventorySnapshot,
        finalize: Callable[[], None],
    ) -> InventorySnapshot:
        """Unlock and publish success inside two exact pathname checks.

        The fixed lock is removed only after the visible output pathname and
        full inventory pass.  ``finalize`` then publishes the success receipt
        while the directory descriptor remains held.  A second check catches
        any pathname swap or artifact mutation during that publication.  If
        either check fails, the caller can overwrite a prematurely published
        success receipt with a failed receipt before returning.
        """

        self._require_active()
        expected_files = tuple(owned_files)
        error: BaseException | None = None
        result: InventorySnapshot | None = None
        lock_unlinked = False
        try:
            self._assert_inventory(
                expected_files,
                previous=previous,
                include_lock=True,
            )
            self._unlink_owned_lock()
            lock_unlinked = True
            unlocked = self._assert_inventory(
                expected_files,
                previous=previous,
                include_lock=False,
            )
            finalize()
            result = self._assert_inventory(
                expected_files,
                previous=unlocked,
                include_lock=False,
            )
        except BaseException as caught:
            error = caught
        finally:
            if not lock_unlinked:
                try:
                    self._unlink_owned_lock()
                except BaseException as cleanup_error:
                    if error is None:
                        error = cleanup_error
                    else:
                        error = RuntimeError(f"{error}; output lock cleanup failed: {cleanup_error}")
            try:
                os.close(self._lock_descriptor)
            except BaseException as close_error:
                if error is None:
                    error = close_error
                else:
                    error = RuntimeError(f"{error}; output lock descriptor close failed: {close_error}")
            try:
                os.close(self._directory_descriptor)
            except BaseException as close_error:
                if error is None:
                    error = close_error
                else:
                    error = RuntimeError(f"{error}; output directory descriptor close failed: {close_error}")
            self._released = True
        if error is not None:
            if isinstance(error, LiveReservationError):
                raise error
            raise LiveReservationError(f"could not verify and release fixed output lock: {error}") from error
        assert result is not None
        return result

    def release(self) -> None:
        """Remove only the fixed lock inode opened by this lease, then fsync."""

        self._require_active()
        error: BaseException | None = None
        try:
            self._unlink_owned_lock()
        except BaseException as caught:
            error = caught
        finally:
            os.close(self._lock_descriptor)
            os.close(self._directory_descriptor)
            self._released = True
        if error is not None:
            if isinstance(error, LiveReservationError):
                raise error
            raise LiveReservationError(f"could not release fixed output lock: {error}") from error
