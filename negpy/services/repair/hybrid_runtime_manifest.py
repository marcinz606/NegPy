"""Strict operator-provisioned configuration for the external hybrid runtime."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from pathlib import Path

from negpy.kernel.system.config import BASE_USER_DIR
from negpy.services.repair.fauxice_hybrid_runner import HybridRuntimeConfig


RUNTIME_MANIFEST_SCHEMA = "negpy.fauxce-hybrid-runtime.v1"
RUNTIME_MANIFEST_FILENAME = "fauxce-hybrid-runtime.json"
RUNTIME_MANIFEST_MAX_BYTES = 64 * 1024
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_PATH_FIELDS = (
    "hybrid_python",
    "executable",
    "iopaint_python",
    "iopaint_executable",
    "model_dir",
    "model_weights",
)
_HASH_FIELDS = (
    "core_source_manifest_sha256",
    "hybrid_source_manifest_sha256",
    "iopaint_source_manifest_sha256",
    "model_weights_sha256",
)
_REQUIRED_FIELDS = frozenset(
    {
        "schema",
        *_PATH_FIELDS,
        *_HASH_FIELDS,
        "inpaint_device",
        "inpaint_threads",
        "inpaint_seed",
    }
)


class HybridRuntimeManifestError(RuntimeError):
    """The external runtime manifest or its independently retained pin failed."""


def default_hybrid_runtime_manifest_path() -> Path:
    return Path(BASE_USER_DIR) / RUNTIME_MANIFEST_FILENAME


def runtime_manifest_pin_path(manifest_path: Path) -> Path:
    return manifest_path.with_name(manifest_path.name + ".sha256")


def _stable_regular_bytes(path: Path, *, maximum_bytes: int, label: str) -> bytes:
    descriptor: int | None = None
    try:
        linked = os.lstat(path)
        if stat.S_ISLNK(linked.st_mode) or not stat.S_ISREG(linked.st_mode):
            raise HybridRuntimeManifestError(f"{label} must be a regular non-symlink file")
        if linked.st_size < 0 or linked.st_size > maximum_bytes:
            raise HybridRuntimeManifestError(f"{label} exceeds its safe size limit")
        flags = (
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_NONBLOCK", 0)
        )
        descriptor = os.open(path, flags)
        with os.fdopen(descriptor, "rb", closefd=True) as handle:
            descriptor = None
            before = os.fstat(handle.fileno())
            if (
                not stat.S_ISREG(before.st_mode)
                or (before.st_dev, before.st_ino) != (linked.st_dev, linked.st_ino)
                or before.st_size != linked.st_size
            ):
                raise HybridRuntimeManifestError(f"{label} changed while opening")
            payload = handle.read(maximum_bytes + 1)
            after = os.fstat(handle.fileno())
        if (
            len(payload) != before.st_size
            or len(payload) > maximum_bytes
            or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        ):
            raise HybridRuntimeManifestError(f"{label} changed while reading")
        return payload
    except HybridRuntimeManifestError:
        raise
    except OSError as error:
        raise HybridRuntimeManifestError(f"could not read {label}: {error}") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)


def canonical_runtime_manifest_bytes(document: object) -> bytes:
    try:
        return (
            json.dumps(
                document,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode("utf-8")
            + b"\n"
        )
    except (TypeError, ValueError, RecursionError) as error:
        raise HybridRuntimeManifestError(f"hybrid runtime manifest cannot be canonicalized: {error}") from error


def load_hybrid_runtime_manifest(
    manifest_path: str | Path,
    *,
    expected_sha256: str,
) -> HybridRuntimeConfig:
    """Load one canonical manifest whose digest came from a separate boundary."""

    path = Path(manifest_path)
    if _SHA256_RE.fullmatch(expected_sha256) is None:
        raise HybridRuntimeManifestError("hybrid runtime manifest pin must be a lowercase SHA-256 digest")
    payload = _stable_regular_bytes(
        path,
        maximum_bytes=RUNTIME_MANIFEST_MAX_BYTES,
        label="hybrid runtime manifest",
    )
    actual_sha256 = hashlib.sha256(payload).hexdigest()
    if actual_sha256 != expected_sha256:
        raise HybridRuntimeManifestError(f"hybrid runtime manifest SHA-256 mismatch: expected {expected_sha256}, got {actual_sha256}")
    try:
        document = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise HybridRuntimeManifestError(f"hybrid runtime manifest is not valid JSON: {error}") from error
    if not isinstance(document, dict):
        raise HybridRuntimeManifestError("hybrid runtime manifest must be an object")
    if canonical_runtime_manifest_bytes(document) != payload:
        raise HybridRuntimeManifestError("hybrid runtime manifest is not canonical JSON")
    fields = frozenset(document)
    if fields != _REQUIRED_FIELDS:
        missing = sorted(_REQUIRED_FIELDS - fields)
        extra = sorted(fields - _REQUIRED_FIELDS)
        raise HybridRuntimeManifestError(f"hybrid runtime manifest fields changed (missing={missing}, extra={extra})")
    if document["schema"] != RUNTIME_MANIFEST_SCHEMA:
        raise HybridRuntimeManifestError("hybrid runtime manifest schema is unsupported")
    for field in _PATH_FIELDS:
        if not isinstance(document[field], str) or not document[field]:
            raise HybridRuntimeManifestError(f"{field} must be a non-empty path string")
    for field in _HASH_FIELDS:
        if not isinstance(document[field], str) or _SHA256_RE.fullmatch(document[field]) is None:
            raise HybridRuntimeManifestError(f"{field} must be a lowercase SHA-256 digest")
    if document["inpaint_device"] not in {"cpu", "cuda", "mps"}:
        raise HybridRuntimeManifestError("inpaint_device must be cpu, cuda, or mps")
    for field, minimum in (("inpaint_threads", 1), ("inpaint_seed", 0)):
        value = document[field]
        if type(value) is not int or value < minimum:
            raise HybridRuntimeManifestError(f"{field} must be an integer greater than or equal to {minimum}")
    try:
        return HybridRuntimeConfig(
            hybrid_python=Path(document["hybrid_python"]),
            executable=Path(document["executable"]),
            core_source_manifest_sha256=document["core_source_manifest_sha256"],
            hybrid_source_manifest_sha256=document["hybrid_source_manifest_sha256"],
            iopaint_python=Path(document["iopaint_python"]),
            iopaint_executable=Path(document["iopaint_executable"]),
            iopaint_source_manifest_sha256=document["iopaint_source_manifest_sha256"],
            model_dir=Path(document["model_dir"]),
            model_weights=Path(document["model_weights"]),
            model_weights_sha256=document["model_weights_sha256"],
            inpaint_device=document["inpaint_device"],
            inpaint_threads=document["inpaint_threads"],
            inpaint_seed=document["inpaint_seed"],
        )
    except (TypeError, ValueError) as error:
        raise HybridRuntimeManifestError(f"hybrid runtime manifest values are invalid: {error}") from error


def load_default_hybrid_runtime_manifest(
    manifest_path: str | Path | None = None,
) -> HybridRuntimeConfig | None:
    """Load the desktop runtime, or return ``None`` when it was never installed."""

    path = default_hybrid_runtime_manifest_path() if manifest_path is None else Path(manifest_path)
    pin_path = runtime_manifest_pin_path(path)
    if not path.exists() and not pin_path.exists():
        return None
    if not path.exists() or not pin_path.exists():
        raise HybridRuntimeManifestError("hybrid runtime manifest and its .sha256 pin must both exist")
    pin_bytes = _stable_regular_bytes(
        pin_path,
        maximum_bytes=65,
        label="hybrid runtime manifest pin",
    )
    try:
        pin = pin_bytes.decode("ascii").rstrip("\n")
    except UnicodeDecodeError as error:
        raise HybridRuntimeManifestError("hybrid runtime manifest pin is not ASCII") from error
    if pin_bytes != (pin + "\n").encode("ascii") or _SHA256_RE.fullmatch(pin) is None:
        raise HybridRuntimeManifestError("hybrid runtime manifest pin must be one lowercase SHA-256 plus newline")
    return load_hybrid_runtime_manifest(path, expected_sha256=pin)


__all__ = [
    "HybridRuntimeManifestError",
    "RUNTIME_MANIFEST_FILENAME",
    "RUNTIME_MANIFEST_SCHEMA",
    "canonical_runtime_manifest_bytes",
    "default_hybrid_runtime_manifest_path",
    "load_default_hybrid_runtime_manifest",
    "load_hybrid_runtime_manifest",
    "runtime_manifest_pin_path",
]
