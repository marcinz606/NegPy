"""Subprocess bridge to the optional ``fauxce-hybrid`` companion tool.

``fauxce-hybrid`` is a separate, independently optional package from the core
``portable_digital_ice`` engine (see ``fauxice_ir_repair.py``). It has no
importable "run a repair" function of its own; ``src/fauxce_hybrid/cli.py``
in the upstream project keeps that logic private and exposes only the
``fauxce-hybrid`` console script. Shelling out to that script, exactly as its
own docs show, is therefore the calling contract, not a shortcut around one.

The hybrid CLI needs the same paired 285 dpi prepass + 4000 dpi main RGBI
acquisition as the core engine (it runs the core engine internally to get the
``at_floor_mask`` it routes on), so it never relaxes the prepass requirement
described in ``fauxice_ir_repair.py``.

IOPaint and the LaMa model weights are never invoked by NegPy directly. They
run inside the fauxce-hybrid subprocess, pointed at a pinned interpreter and
hash-verified weights that ``HybridRuntimeConfig`` names but never bundles.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import re
import signal
import stat
import subprocess
import tempfile
import threading
import time

import numpy as np
import numpy.typing as npt
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from PIL import Image


class HybridRunError(RuntimeError):
    """The fauxce-hybrid subprocess did not produce a usable result."""


class HybridRunCancelled(HybridRunError):
    """The caller cancelled the external hybrid process and its children."""


def _stable_regular_sha256(path: Path, *, label: str) -> str:
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
    try:
        opened = os.fstat(descriptor)
        identity = (
            opened.st_dev,
            opened.st_ino,
            opened.st_size,
            opened.st_mtime_ns,
            opened.st_ctime_ns,
        )
        if not stat.S_ISREG(opened.st_mode) or identity != (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ):
            raise ValueError(f"{label} changed while it was opened")
        digest = hashlib.sha256()
        remaining = opened.st_size
        while remaining:
            block = os.read(descriptor, min(1024 * 1024, remaining))
            if not block:
                raise ValueError(f"{label} changed while it was hashed")
            digest.update(block)
            remaining -= len(block)
        # Bound the read by the size captured at open. One extra byte detects
        # growth without following an attacker-controlled, never-ending file.
        if os.read(descriptor, 1):
            raise ValueError(f"{label} changed while it was hashed")
        after_read = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    after_path = os.lstat(path)
    for metadata in (after_read, after_path):
        if (
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_size,
            metadata.st_mtime_ns,
            metadata.st_ctime_ns,
        ) != identity:
            raise ValueError(f"{label} changed while it was hashed")
    return digest.hexdigest()


@dataclass(frozen=True)
class HybridRuntimeConfig:
    """Where the pinned IOPaint interpreter and hash-verified weights live.

    None of these paths are discovered automatically. fauxce-hybrid's own
    docs (``hybrid/docs/hybrid-repair.md`` in the digital-fauxice repository)
    require the caller to install IOPaint 1.6.0 into its own virtualenv and
    to supply the measured SHA-256 of ``big-lama.pt`` rather than trusting a
    filename; this config just carries those caller-verified values through.
    """

    hybrid_python: Path
    executable: Path
    core_source_manifest_sha256: str
    hybrid_source_manifest_sha256: str
    iopaint_python: Path
    iopaint_executable: Path
    iopaint_source_manifest_sha256: str
    model_dir: Path
    model_weights: Path
    model_weights_sha256: str
    inpaint_device: str = "cpu"
    inpaint_threads: int = 1
    inpaint_seed: int = 0
    max_synthesis_fraction: float = 0.02

    def __post_init__(self) -> None:
        for field_name in (
            "hybrid_python",
            "executable",
            "iopaint_python",
            "iopaint_executable",
            "model_dir",
            "model_weights",
        ):
            value = Path(getattr(self, field_name))
            if not value.is_absolute():
                raise ValueError(f"{field_name} must be an absolute explicit path")
            object.__setattr__(self, field_name, value)
        for field_name in (
            "core_source_manifest_sha256",
            "hybrid_source_manifest_sha256",
            "iopaint_source_manifest_sha256",
            "model_weights_sha256",
        ):
            if re.fullmatch(r"[0-9a-f]{64}", getattr(self, field_name)) is None:
                raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")
        if self.inpaint_device not in {"cpu", "cuda", "mps"}:
            raise ValueError("inpaint_device must be cpu, cuda, or mps")
        if type(self.inpaint_threads) is not int or self.inpaint_threads < 1:
            raise ValueError("inpaint_threads must be a positive integer")
        if type(self.inpaint_seed) is not int or self.inpaint_seed < 0:
            raise ValueError("inpaint_seed must be a non-negative integer")
        if (
            isinstance(self.max_synthesis_fraction, bool)
            or not isinstance(self.max_synthesis_fraction, (int, float))
            or not math.isfinite(float(self.max_synthesis_fraction))
            or not 0.0 <= float(self.max_synthesis_fraction) <= 1.0
        ):
            raise ValueError("max_synthesis_fraction must be finite and in [0, 1]")
        object.__setattr__(self, "max_synthesis_fraction", float(self.max_synthesis_fraction))

    def validate_files(self) -> None:
        """Fail closed unless every configured runtime artifact is present."""

        executables = (
            ("hybrid_python", self.hybrid_python),
            ("executable", self.executable),
            ("iopaint_python", self.iopaint_python),
            ("iopaint_executable", self.iopaint_executable),
        )
        for label, path in executables:
            try:
                resolved = path.resolve(strict=True)
                mode = resolved.stat().st_mode
            except OSError as error:
                raise ValueError(f"{label} is unavailable: {error}") from error
            if not stat.S_ISREG(mode) or not os.access(path, os.X_OK):
                raise ValueError(f"{label} must resolve to an executable regular file")
        try:
            model_dir = self.model_dir.resolve(strict=True)
        except OSError as error:
            raise ValueError(f"model_dir is unavailable: {error}") from error
        if not model_dir.is_dir():
            raise ValueError("model_dir must resolve to a directory")
        try:
            weights = self.model_weights.resolve(strict=True)
            weights_mode = weights.stat().st_mode
        except OSError as error:
            raise ValueError(f"model_weights is unavailable: {error}") from error
        if not stat.S_ISREG(weights_mode):
            raise ValueError("model_weights must resolve to a regular file")
        try:
            measured_weights_sha256 = _stable_regular_sha256(
                self.model_weights,
                label="model_weights",
            )
        except OSError as error:
            raise ValueError(f"model_weights is unavailable: {error}") from error
        if measured_weights_sha256 != self.model_weights_sha256:
            raise ValueError("model_weights SHA-256 does not match the pinned runtime")


@dataclass(frozen=True)
class HybridRunResult:
    """What one fauxce-hybrid subprocess call produced, read back into memory.

    The mask is carried as bytes rather than a path. The CLI files remain in
    the caller-owned per-run scratch child until the caller removes it; only
    values copied into this result are independent of that directory.
    """

    hybrid_rgb16: npt.NDArray[np.uint16]
    synth_mask_png: bytes
    synth_mask_sha256: str
    synth_mask: npt.NDArray[np.bool_]
    receipt: bytes
    receipt_sha256: str
    acquisition_manifest_sha256: str
    main_rgbi_sha256: str
    prepass_rgbi_sha256: str
    output_rgb16_sha256: str
    provenance_class: str
    synthesis_fraction: float | None
    engine_version: str | None
    backend_requested: str | None
    backend_used: str | None
    backend_selection_reason: str | None
    routing_counts: dict[str, int] | None


ReceiptVerifier = Callable[[Path, HybridRuntimeConfig], Any]
VerificationRunner = Callable[..., "subprocess.CompletedProcess[str]"]
_MAX_SUBPROCESS_LOG_BYTES = 16 * 1024 * 1024


@dataclass(frozen=True)
class _ReceiptVerification:
    receipt_sha256: str
    model_weights_rehashed: bool


_EXTERNAL_RECEIPT_VERIFIER = r"""
import json
import sys
from pathlib import Path

from fauxce_hybrid.receipts import verify_receipt

receipt_path = Path(sys.argv[1])
weights_path = Path(sys.argv[2])
weights_sha256 = sys.argv[3]

def resolve_weights(attestation):
    if getattr(attestation, "sha256", None) != weights_sha256:
        raise ValueError("receipt model weights do not match the pinned runtime")
    return weights_path

verified = verify_receipt(
    receipt_path,
    model_weights_resolver=resolve_weights,
    require_model_weights=True,
)
document = {
    "model_weights_rehashed": bool(verified.model_weights_rehashed),
    "receipt_sha256": verified.receipt_sha256,
    "schema": "negpy.external-fauxce-receipt-verification-v1",
}
sys.stdout.write(json.dumps(document, sort_keys=True, separators=(",", ":")))
""".strip()


def _process_group_exists(process_group: int) -> bool:
    try:
        os.killpg(process_group, 0)
    except ProcessLookupError:
        return False
    except (AttributeError, PermissionError):
        return True
    return True


def _terminate_process_group(process: subprocess.Popen) -> None:
    process_group = process.pid
    try:
        os.killpg(process_group, signal.SIGTERM)
    except (AttributeError, OSError):
        try:
            process.terminate()
        except OSError:
            pass
    deadline = time.monotonic() + 3.0
    while _process_group_exists(process_group) and time.monotonic() < deadline:
        time.sleep(0.05)
    # The group leader may have exited on TERM while an IOPaint/model child
    # ignored it. Kill the group after grace regardless of leader state.
    try:
        os.killpg(process_group, signal.SIGKILL)
    except (AttributeError, OSError):
        try:
            process.kill()
        except OSError:
            pass
    try:
        process.wait(timeout=3.0)
    except (OSError, subprocess.TimeoutExpired):
        pass


def _sanitized_subprocess_environment() -> dict[str, str]:
    environment = dict(os.environ)
    for name in (
        "PYTHONHOME",
        "PYTHONINSPECT",
        "PYTHONPATH",
        "PYTHONSTARTUP",
        "PYTHONUSERBASE",
    ):
        environment.pop(name, None)
    for name in tuple(environment):
        if name.startswith(("DYLD_", "LD_")):
            environment.pop(name, None)
    environment.update(
        {
            "LANG": "C",
            "LC_ALL": "C",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "PYTHONHASHSEED": "0",
            "PYTHONNOUSERSITE": "1",
        }
    )
    return environment


def _raise_if_cancelled(
    cancel: threading.Event | None,
    *,
    phase: str,
) -> None:
    if cancel is not None and cancel.is_set():
        raise HybridRunCancelled(f"fauxce-hybrid cancelled {phase}")


def _run_command(
    argv: Sequence[str],
    *,
    timeout_seconds: float,
    cancel: threading.Event | None,
    runner: VerificationRunner | None,
    label: str,
) -> subprocess.CompletedProcess[str]:
    if cancel is not None and cancel.is_set():
        raise HybridRunCancelled(f"{label} cancelled before launch")
    if runner is not None:
        try:
            completed = runner(
                list(argv),
                capture_output=True,
                env=_sanitized_subprocess_environment(),
                text=True,
                timeout=timeout_seconds,
            )
        except (OSError, subprocess.SubprocessError) as error:
            raise HybridRunError(f"could not launch {label}: {error}") from error
        if cancel is not None and cancel.is_set():
            raise HybridRunCancelled(f"{label} cancelled")
        return completed

    process: subprocess.Popen | None = None
    completed_normally = False
    try:
        with (
            tempfile.TemporaryFile(mode="w+t", encoding="utf-8") as stdout_file,
            tempfile.TemporaryFile(
                mode="w+t",
                encoding="utf-8",
            ) as stderr_file,
        ):
            process = subprocess.Popen(
                list(argv),
                stdin=subprocess.DEVNULL,
                stdout=stdout_file,
                stderr=stderr_file,
                env=_sanitized_subprocess_environment(),
                text=True,
                start_new_session=True,
            )
            deadline = time.monotonic() + timeout_seconds
            while process.poll() is None:
                if cancel is not None and cancel.is_set():
                    _terminate_process_group(process)
                    raise HybridRunCancelled(f"{label} cancelled")
                if time.monotonic() >= deadline:
                    _terminate_process_group(process)
                    raise HybridRunError(f"{label} exceeded its {timeout_seconds:g}s timeout")
                if (
                    os.fstat(stdout_file.fileno()).st_size > _MAX_SUBPROCESS_LOG_BYTES
                    or os.fstat(stderr_file.fileno()).st_size > _MAX_SUBPROCESS_LOG_BYTES
                ):
                    _terminate_process_group(process)
                    raise HybridRunError(f"{label} output exceeded its safe size limit")
                time.sleep(0.05)
            if (
                os.fstat(stdout_file.fileno()).st_size > _MAX_SUBPROCESS_LOG_BYTES
                or os.fstat(stderr_file.fileno()).st_size > _MAX_SUBPROCESS_LOG_BYTES
            ):
                raise HybridRunError(f"{label} output exceeded its safe size limit")
            stdout_file.seek(0)
            stderr_file.seek(0)
            completed = subprocess.CompletedProcess(
                list(argv),
                process.returncode,
                stdout_file.read(),
                stderr_file.read(),
            )
            if _process_group_exists(process.pid):
                _terminate_process_group(process)
                raise HybridRunError(f"{label} left child processes running after exit")
            completed_normally = True
            return completed
    except (HybridRunError, HybridRunCancelled):
        raise
    except (OSError, subprocess.SubprocessError) as error:
        raise HybridRunError(f"could not launch {label}: {error}") from error
    finally:
        if process is not None and not completed_normally:
            _terminate_process_group(process)


def _raw_sha256(array: np.ndarray, *, dtype: np.dtype) -> str:
    canonical = np.array(array, dtype=dtype, order="C", copy=True)
    return hashlib.sha256(memoryview(canonical).cast("B")).hexdigest()


def _canonical_rgbi(array: np.ndarray, *, label: str) -> np.ndarray:
    value = np.asarray(array)
    if value.ndim != 3 or value.shape[2] != 4 or value.dtype != np.uint16:
        raise HybridRunError(f"{label} must be an HxWx4 uint16 array")
    canonical = np.array(value, dtype="<u2", order="C", copy=True)
    canonical.setflags(write=False)
    return canonical


def _canonical_json_bytes(document: object) -> bytes:
    try:
        return json.dumps(
            document,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, RecursionError) as error:
        raise HybridRunError(f"acquisition manifest is not canonical JSON: {error}") from error


def _canonical_receipt_bytes(document: object) -> bytes:
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
        raise HybridRunError(f"hybrid receipt is not canonical JSON: {error}") from error


def _stable_regular_bytes(path: Path, *, maximum_bytes: int, label: str) -> bytes:
    descriptor: int | None = None
    try:
        linked = os.lstat(path)
        if stat.S_ISLNK(linked.st_mode) or not stat.S_ISREG(linked.st_mode):
            raise HybridRunError(f"{label} must be a regular non-symlink file")
        if linked.st_size < 0 or linked.st_size > maximum_bytes:
            raise HybridRunError(f"{label} exceeds its safe size limit")
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
        descriptor = os.open(path, flags)
        with os.fdopen(descriptor, "rb", closefd=True) as handle:
            descriptor = None
            before = os.fstat(handle.fileno())
            if not stat.S_ISREG(before.st_mode) or (
                before.st_dev,
                before.st_ino,
                before.st_size,
                before.st_mtime_ns,
                before.st_ctime_ns,
            ) != (
                linked.st_dev,
                linked.st_ino,
                linked.st_size,
                linked.st_mtime_ns,
                linked.st_ctime_ns,
            ):
                raise HybridRunError(f"{label} changed while being opened")
            payload = handle.read(maximum_bytes + 1)
            after = os.fstat(handle.fileno())
        if (
            len(payload) != before.st_size
            or len(payload) > maximum_bytes
            or (
                before.st_dev,
                before.st_ino,
                before.st_size,
                before.st_mtime_ns,
                before.st_ctime_ns,
            )
            != (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            )
        ):
            raise HybridRunError(f"{label} changed while being read")
        return payload
    except HybridRunError:
        raise
    except OSError as error:
        raise HybridRunError(f"could not securely read {label}: {error}") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _decode_mask_png(payload: bytes, *, expected_shape: tuple[int, int]) -> np.ndarray:
    try:
        with Image.open(io.BytesIO(payload)) as image:
            if image.format != "PNG":
                raise HybridRunError("synthesis disclosure mask is not a PNG")
            if image.size != (expected_shape[1], expected_shape[0]):
                raise HybridRunError("synthesis disclosure mask geometry is invalid")
            decoded = np.asarray(image.convert("L"))
    except HybridRunError:
        raise
    except Exception as error:
        raise HybridRunError(f"synthesis disclosure mask PNG is invalid: {error}") from error
    if decoded.shape != expected_shape or decoded.dtype != np.uint8:
        raise HybridRunError("synthesis disclosure mask geometry is invalid")
    unique = np.unique(decoded)
    if not np.all(np.isin(unique, np.array([0, 255], dtype=np.uint8))):
        raise HybridRunError("synthesis disclosure mask must be binary")
    mask = np.ascontiguousarray(decoded != 0)
    mask.setflags(write=False)
    return mask


def _default_receipt_verifier(
    path: Path,
    runtime: HybridRuntimeConfig,
    *,
    runner: VerificationRunner | None = None,
    cancel: threading.Event | None = None,
) -> _ReceiptVerification:
    """Verify with the explicitly configured Python-3.12 companion.

    NegPy itself supports Python 3.13 while fauxce-hybrid intentionally uses
    a separate Python 3.12 environment.  Importing the verifier in-process
    would therefore make a correctly installed companion appear unavailable.
    The external verifier returns only a small attestation; NegPy then reads
    and checks every result artifact itself before publishing it.
    """

    completed = _run_command(
        [
            str(runtime.hybrid_python),
            "-I",
            "-c",
            _EXTERNAL_RECEIPT_VERIFIER,
            str(path),
            str(runtime.model_weights),
            runtime.model_weights_sha256,
        ],
        timeout_seconds=300.0,
        cancel=cancel,
        runner=runner,
        label="external hybrid receipt verifier",
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()
        raise HybridRunError("external hybrid receipt verification failed: " + detail[-2000:])
    try:
        document = json.loads(completed.stdout)
    except (TypeError, json.JSONDecodeError) as error:
        raise HybridRunError("external hybrid receipt verifier returned invalid JSON") from error
    if (
        not isinstance(document, dict)
        or document.get("schema") != "negpy.external-fauxce-receipt-verification-v1"
        or re.fullmatch(r"[0-9a-f]{64}", document.get("receipt_sha256", "")) is None
        or type(document.get("model_weights_rehashed")) is not bool
    ):
        raise HybridRunError("external hybrid receipt verifier returned an invalid attestation")
    return _ReceiptVerification(
        receipt_sha256=document["receipt_sha256"],
        model_weights_rehashed=document["model_weights_rehashed"],
    )


def _decode_npy_rgb16(
    payload: bytes,
    *,
    expected_shape: tuple[int, int, int],
) -> np.ndarray:
    stream = io.BytesIO(payload)
    try:
        version = np.lib.format.read_magic(stream)
        if version == (1, 0):
            shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(stream)
        elif version == (2, 0):
            shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(stream)
        else:
            raise HybridRunError(f"hybrid output NPY version {version!r} is unsupported")
        if tuple(shape) != expected_shape or dtype != np.dtype("<u2") or fortran_order:
            raise HybridRunError(f"hybrid output must be C-contiguous HxWx3 uint16 with shape {expected_shape}")
        expected_bytes = int(np.prod(expected_shape, dtype=np.int64)) * 2
        if len(payload) - stream.tell() != expected_bytes:
            raise HybridRunError("hybrid output NPY byte length is invalid")
        stream.seek(0)
        decoded = np.load(stream, allow_pickle=False)
    except HybridRunError:
        raise
    except Exception as error:
        raise HybridRunError(f"hybrid output NPY is invalid: {error}") from error
    if stream.tell() != len(payload):
        raise HybridRunError("hybrid output NPY has trailing data")
    if decoded.dtype != np.dtype("<u2") or decoded.shape != expected_shape or not decoded.flags.c_contiguous:
        raise HybridRunError(f"hybrid output must be C-contiguous HxWx3 uint16 with shape {expected_shape}")
    output = np.array(decoded, dtype="<u2", order="C", copy=True)
    output.setflags(write=False)
    return output


def run_hybrid_repair(
    main_rgbi: npt.NDArray[np.uint16],
    prepass_rgbi: npt.NDArray[np.uint16],
    *,
    same_frame_id: str,
    backend: str,
    runtime: HybridRuntimeConfig,
    scratch_dir: Path,
    timeout_seconds: float = 1800.0,
    runner: Callable[..., "subprocess.CompletedProcess[str]"] | None = None,
    receipt_verifier: ReceiptVerifier | None = None,
    verification_runner: VerificationRunner | None = None,
    progress: Callable[[float], None] | None = None,
    cancel: threading.Event | None = None,
) -> HybridRunResult:
    """Run one frame through the fauxce-hybrid CLI and read its outputs back.

    ``scratch_dir`` must resolve to a directory, but it may already contain
    other entries. This function creates one exclusive
    ``negpy-hybrid-run-*`` child and then creates the CLI's ``--out`` path
    inside it (the CLI refuses a pre-existing output path). The caller owns
    both the parent and per-run child lifetime; nothing here deletes them, so
    an ephemeral caller should wrap the parent in
    ``tempfile.TemporaryDirectory()``.
    """

    main = _canonical_rgbi(main_rgbi, label="main_rgbi")
    prepass = _canonical_rgbi(prepass_rgbi, label="prepass_rgbi")
    try:
        runtime.validate_files()
    except ValueError as error:
        raise HybridRunError(f"hybrid runtime is unavailable: {error}") from error
    try:
        scratch_root = Path(scratch_dir).resolve(strict=True)
    except OSError as error:
        raise HybridRunError(f"hybrid scratch directory is unavailable: {error}") from error
    if not scratch_root.is_dir():
        raise HybridRunError("hybrid scratch path must resolve to a directory")
    if type(same_frame_id) is not str or not same_frame_id.strip():
        raise HybridRunError("same_frame_id must be non-empty")
    if cancel is not None and cancel.is_set():
        raise HybridRunCancelled("fauxce-hybrid cancelled before launch")
    if progress is not None:
        progress(0.0)
    main_sha256 = _raw_sha256(main, dtype=np.dtype("<u2"))
    prepass_sha256 = _raw_sha256(prepass, dtype=np.dtype("<u2"))
    acquisition_manifest = {
        "assertions": {
            "focus_exposure_locked": True,
            "same_frame_id": same_frame_id,
        },
        "inputs": {
            "main": {"raw_sha256": main_sha256},
            "prepass": {"raw_sha256": prepass_sha256},
        },
        "provenance_class": "caller_asserted_bare_npy",
        "schema": "negpy.fauxce-hybrid-acquisition-assertion-v1",
    }
    acquisition_manifest_bytes = _canonical_json_bytes(acquisition_manifest)
    acquisition_manifest_sha256 = hashlib.sha256(acquisition_manifest_bytes).hexdigest()
    try:
        work_root = Path(tempfile.mkdtemp(prefix="negpy-hybrid-run-", dir=scratch_root)).resolve(strict=True)
    except OSError as error:
        raise HybridRunError(f"cannot create an exclusive hybrid work directory: {error}") from error
    prepass_path = work_root / "prepass.rgbi16.npy"
    main_path = work_root / "main.rgbi16.npy"
    acquisition_manifest_path = work_root / "acquisition.json"
    out_dir = work_root / "out"
    np.save(prepass_path, prepass, allow_pickle=False)
    np.save(main_path, main, allow_pickle=False)
    acquisition_manifest_path.write_bytes(acquisition_manifest_bytes)

    argv: Sequence[str] = [
        str(runtime.executable),
        "--prepass",
        str(prepass_path),
        "--main",
        str(main_path),
        "--out",
        str(out_dir),
        "--same-frame-id",
        same_frame_id,
        "--assert-focus-exposure-locked",
        "--acquisition-manifest",
        str(acquisition_manifest_path),
        "--backend",
        backend,
        "--max-synth-fraction",
        format(runtime.max_synthesis_fraction, ".12g"),
        "--iopaint-python",
        str(runtime.iopaint_python),
        "--iopaint-executable",
        str(runtime.iopaint_executable),
        "--iopaint-source-manifest-sha256",
        runtime.iopaint_source_manifest_sha256,
        "--model-dir",
        str(runtime.model_dir),
        "--model-weights",
        str(runtime.model_weights),
        "--model-weights-sha256",
        runtime.model_weights_sha256,
        "--inpaint-device",
        runtime.inpaint_device,
        "--inpaint-threads",
        str(runtime.inpaint_threads),
        "--inpaint-seed",
        str(runtime.inpaint_seed),
    ]
    completed = _run_command(
        argv,
        timeout_seconds=timeout_seconds,
        cancel=cancel,
        runner=runner,
        label=f"fauxce-hybrid {runtime.executable!s}",
    )
    _raise_if_cancelled(cancel, phase="after generation")

    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()
        raise HybridRunError(f"fauxce-hybrid exited {completed.returncode}: {detail[-2000:]}")
    if progress is not None:
        progress(0.8)
    _raise_if_cancelled(cancel, phase="after generation progress")

    try:
        hybrid_path = out_dir / "output-hybrid.rgb16.npy"
        mask_path = out_dir / "synth-mask.png"
        receipt_path = out_dir / "hybrid-receipt.json"
        for required in (hybrid_path, mask_path, receipt_path):
            try:
                linked = os.lstat(required)
            except OSError as error:
                raise HybridRunError(f"fauxce-hybrid reported success but {required.name} is missing from {out_dir}") from error
            if stat.S_ISLNK(linked.st_mode) or not stat.S_ISREG(linked.st_mode):
                raise HybridRunError(f"fauxce-hybrid output {required.name} must be a regular non-symlink file")

        expected_output_shape = (*main.shape[:2], 3)
        _raise_if_cancelled(cancel, phase="before receipt verification")
        if receipt_verifier is None:
            verified = _default_receipt_verifier(
                receipt_path,
                runtime,
                runner=verification_runner,
                cancel=cancel,
            )
        else:
            verified = receipt_verifier(receipt_path, runtime)
        _raise_if_cancelled(cancel, phase="after receipt verification")
        if progress is not None:
            progress(0.9)
        _raise_if_cancelled(cancel, phase="after verification progress")

        receipt_bytes = _stable_regular_bytes(
            receipt_path,
            maximum_bytes=16 * 1024 * 1024,
            label="hybrid receipt",
        )
        _raise_if_cancelled(cancel, phase="after receipt read")
        try:
            receipt = json.loads(receipt_bytes)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise HybridRunError(f"hybrid receipt JSON is invalid: {error}") from error
        if _canonical_receipt_bytes(receipt) != receipt_bytes:
            raise HybridRunError("hybrid receipt is not canonical JSON")
        receipt_sha256 = hashlib.sha256(receipt_bytes).hexdigest()
        if receipt_sha256 != getattr(verified, "receipt_sha256", None):
            raise HybridRunError("hybrid receipt changed after verification")

        hybrid_bytes = _stable_regular_bytes(
            hybrid_path,
            maximum_bytes=(int(np.prod(expected_output_shape)) * 2) + 1_048_576,
            label="hybrid output",
        )
        hybrid_rgb16 = _decode_npy_rgb16(
            hybrid_bytes,
            expected_shape=expected_output_shape,
        )
        _raise_if_cancelled(cancel, phase="after output decode")
        output_artifact = _artifact_by_role(receipt, "hybrid_output_rgb16")
        if output_artifact.get("file_sha256") != hashlib.sha256(hybrid_bytes).hexdigest():
            raise HybridRunError("hybrid output file SHA-256 disagrees with receipt")

        mask_bytes = _stable_regular_bytes(
            mask_path,
            maximum_bytes=max(
                1_048_576,
                main.shape[0] * main.shape[1] + 1_048_576,
            ),
            label="synthesis disclosure mask",
        )
        decoded_mask = _decode_mask_png(
            mask_bytes,
            expected_shape=main.shape[:2],
        )
        _raise_if_cancelled(cancel, phase="after mask decode")
        _validate_receipt_bindings(
            receipt,
            runtime=runtime,
            backend=backend,
            acquisition_manifest_sha256=acquisition_manifest_sha256,
            main=main,
            prepass=prepass,
            hybrid_rgb16=hybrid_rgb16,
            synthesis_mask=decoded_mask,
            model_weights_rehashed=bool(getattr(verified, "model_weights_rehashed", False)),
        )
        _raise_if_cancelled(cancel, phase="after artifact binding")

        mask_sha256 = hashlib.sha256(mask_bytes).hexdigest()
        mask_artifact = _artifact_by_role(receipt, "synthesis_mask_png")
        if mask_artifact.get("file_sha256") != mask_sha256:
            raise HybridRunError("synthesis disclosure mask file SHA-256 disagrees with receipt")
        result = HybridRunResult(
            hybrid_rgb16=np.array(
                hybrid_rgb16,
                dtype="<u2",
                order="C",
                copy=True,
            ),
            synth_mask_png=mask_bytes,
            synth_mask_sha256=mask_sha256,
            synth_mask=decoded_mask,
            receipt=receipt_bytes,
            receipt_sha256=receipt_sha256,
            acquisition_manifest_sha256=acquisition_manifest_sha256,
            main_rgbi_sha256=main_sha256,
            prepass_rgbi_sha256=prepass_sha256,
            output_rgb16_sha256=_raw_sha256(hybrid_rgb16, dtype=np.dtype("<u2")),
            provenance_class="caller_asserted_bare_npy",
            **_receipt_fields(receipt),
        )
        result.hybrid_rgb16.setflags(write=False)
        _raise_if_cancelled(cancel, phase="before completion")
        if progress is not None:
            progress(1.0)
        _raise_if_cancelled(cancel, phase="at completion")
        return result
    except HybridRunCancelled:
        raise
    except HybridRunError:
        raise
    except Exception as error:
        raise HybridRunError(f"fauxce-hybrid result validation failed: {error}") from error


def _artifact_by_role(receipt: dict[str, Any], role: str) -> dict[str, Any]:
    artifacts = receipt.get("artifacts")
    if not isinstance(artifacts, list):
        raise HybridRunError("hybrid receipt artifacts are malformed")
    matches = [artifact for artifact in artifacts if isinstance(artifact, dict) and artifact.get("role") == role]
    if len(matches) != 1:
        raise HybridRunError(f"hybrid receipt must contain exactly one {role!r} artifact")
    return matches[0]


def _validate_receipt_bindings(
    receipt: dict[str, Any],
    *,
    runtime: HybridRuntimeConfig,
    backend: str,
    acquisition_manifest_sha256: str,
    main: np.ndarray,
    prepass: np.ndarray,
    hybrid_rgb16: np.ndarray,
    synthesis_mask: np.ndarray,
    model_weights_rehashed: bool,
) -> None:
    if receipt.get("schema") != "fauxce-hybrid-receipt-v2":
        raise HybridRunError("hybrid receipt schema is unsupported")
    inputs = receipt.get("inputs")
    if not isinstance(inputs, dict):
        raise HybridRunError("hybrid receipt inputs are malformed")
    for role, expected in (("main", main), ("prepass", prepass)):
        row = inputs.get(role)
        if not isinstance(row, dict):
            raise HybridRunError(f"hybrid receipt {role} input is missing")
        if row.get("canonical_encoding") != "uint16_little_endian_c_order":
            raise HybridRunError(f"hybrid receipt {role} encoding is unsupported")
        if row.get("shape") != list(expected.shape):
            raise HybridRunError(f"hybrid receipt {role} geometry changed")
        if row.get("raw_sha256") != _raw_sha256(expected, dtype=np.dtype("<u2")):
            raise HybridRunError(f"hybrid receipt {role} SHA-256 changed")
    geometry = inputs.get("geometry")
    if (
        not isinstance(geometry, dict)
        or geometry.get("output_shape") != list(hybrid_rgb16.shape)
        or geometry.get("mask_shape") != list(synthesis_mask.shape)
    ):
        raise HybridRunError("hybrid receipt output geometry changed")
    provenance = inputs.get("provenance")
    if (
        not isinstance(provenance, dict)
        or provenance.get("basis") != "caller_asserted"
        or provenance.get("source_manifest_sha256") != acquisition_manifest_sha256
    ):
        raise HybridRunError("hybrid receipt does not bind the caller-asserted acquisition manifest")

    core = receipt.get("core")
    backend_receipt = core.get("backend") if isinstance(core, dict) else None
    if not isinstance(core, dict) or core.get("source_manifest_sha256") != runtime.core_source_manifest_sha256:
        raise HybridRunError("hybrid receipt core source manifest does not match the pinned runtime")
    generation = receipt.get("generation")
    if not isinstance(generation, dict) or generation.get("hybrid_source_manifest_sha256") != runtime.hybrid_source_manifest_sha256:
        raise HybridRunError("hybrid receipt hybrid source manifest does not match the pinned runtime")
    if (
        not isinstance(backend_receipt, dict)
        or backend_receipt.get("requested") != backend
        or not isinstance(backend_receipt.get("used"), str)
        or not backend_receipt.get("used")
        or not isinstance(backend_receipt.get("reason"), str)
    ):
        raise HybridRunError("hybrid receipt backend binding is invalid")

    synthesis = receipt.get("synthesis")
    if not isinstance(synthesis, dict):
        raise HybridRunError("hybrid receipt synthesis accounting is missing")
    pixel_count = int(np.count_nonzero(synthesis_mask))
    frame_pixels = int(synthesis_mask.size)
    expected_fraction = pixel_count / frame_pixels
    if (
        synthesis.get("pixel_count") != pixel_count
        or synthesis.get("frame_pixel_count") != frame_pixels
        or synthesis.get("fraction") != expected_fraction
        or synthesis.get("within_budget") is not True
        or synthesis.get("maximum_fraction") != runtime.max_synthesis_fraction
    ):
        raise HybridRunError("hybrid receipt synthesis accounting changed")
    routing = receipt.get("routing")
    counts = routing.get("counts") if isinstance(routing, dict) else None
    if (
        not isinstance(counts, dict)
        or any(type(counts.get(key)) is not int for key in _ROUTING_COUNT_KEYS)
        or any(counts[key] < 0 for key in _ROUTING_COUNT_KEYS)
        or counts["synthesis_pixels"] != pixel_count
        or counts["frame_pixels"] != frame_pixels
        or counts["at_floor_pixels"] > frame_pixels
        or counts["synthesis_pixels"] > counts["at_floor_pixels"]
        or counts["final_regions"] > counts["synthesis_pixels"]
        or (counts["final_regions"] == 0) != (pixel_count == 0)
    ):
        raise HybridRunError("hybrid receipt routing counts changed")

    output_hash = _raw_sha256(hybrid_rgb16, dtype=np.dtype("<u2"))
    output_artifact = _artifact_by_role(receipt, "hybrid_output_rgb16")
    if (
        output_artifact.get("shape") != list(hybrid_rgb16.shape)
        or output_artifact.get("dtype") != "<u2"
        or output_artifact.get("raw_sha256") != output_hash
    ):
        raise HybridRunError("hybrid output does not match its receipt artifact")
    composite = receipt.get("composite")
    if not isinstance(composite, dict) or composite.get("hybrid_rgb16_raw_sha256") != output_hash:
        raise HybridRunError("hybrid output does not match its composite receipt")

    inpainting = receipt.get("inpainting")
    if not isinstance(inpainting, dict) or not isinstance(inpainting.get("invoked"), bool):
        raise HybridRunError("hybrid receipt inpainting disclosure is malformed")
    if inpainting["invoked"]:
        model = inpainting.get("model")
        tool = inpainting.get("tool")
        runtime_receipt = inpainting.get("runtime")
        if (
            not model_weights_rehashed
            or not isinstance(model, dict)
            or model.get("weights_sha256") != runtime.model_weights_sha256
            or not isinstance(tool, dict)
            or tool.get("iopaint_source_manifest_sha256") != runtime.iopaint_source_manifest_sha256
            or not isinstance(runtime_receipt, dict)
            or runtime_receipt.get("device") != runtime.inpaint_device
            or runtime_receipt.get("threads") != runtime.inpaint_threads
            or runtime_receipt.get("seed") != runtime.inpaint_seed
        ):
            raise HybridRunError("hybrid receipt does not match the pinned inpainting runtime")


_ROUTING_COUNT_KEYS = (
    "final_regions",
    "synthesis_pixels",
    "frame_pixels",
    "at_floor_pixels",
)


def _receipt_fields(receipt: dict[str, Any]) -> dict[str, Any]:
    """Pull the few fields NegPy's sidecar cares about out of hybrid-receipt.json.

    Reads defensively (``.get`` all the way down): this module is a
    provenance consumer, not the receipt's verifier. ``fauxce-hybrid`` itself
    is the authority on whether a given receipt is internally consistent.

    ``routing_counts`` comes from the receipt's ``routing.counts`` object
    (``fauxce-hybrid-receipt-v2.schema.json``): the disclosed region/pixel
    counts behind the single ``synthesis_fraction`` float, e.g. "13 regions,
    16137 pixels of 22815772". ``None`` when the receipt is missing any of
    the expected keys, rather than publishing a partial count.
    """

    core = receipt.get("core", {}) if isinstance(receipt, dict) else {}
    backend = core.get("backend", {}) if isinstance(core, dict) else {}
    synthesis = receipt.get("synthesis", {}) if isinstance(receipt, dict) else {}
    routing = receipt.get("routing", {}) if isinstance(receipt, dict) else {}
    counts = routing.get("counts", {}) if isinstance(routing, dict) else {}
    routing_counts = (
        {key: counts[key] for key in _ROUTING_COUNT_KEYS}
        if isinstance(counts, dict) and all(key in counts for key in _ROUTING_COUNT_KEYS)
        else None
    )
    return {
        "synthesis_fraction": synthesis.get("fraction"),
        "engine_version": core.get("version"),
        "backend_requested": backend.get("requested"),
        "backend_used": backend.get("used"),
        "backend_selection_reason": backend.get("reason"),
        "routing_counts": routing_counts,
    }


__all__ = [
    "HybridRunError",
    "HybridRunCancelled",
    "HybridRunResult",
    "HybridRuntimeConfig",
    "run_hybrid_repair",
]
