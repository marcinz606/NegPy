import io
import json
import os
import shutil
import struct
import tempfile
from hashlib import sha256
from pathlib import Path
from typing import cast

import numpy as np
import tifffile

from negpy.infrastructure.scanners.result import ScanResult, SplitSourceCapture
from negpy.infrastructure.tiff_contract import LINEAR_SCANNER_RGB_EXTRATAG
from negpy.kernel.system.logging import get_logger
from negpy.services.scanning.quality import inspect_tiff_payload

logger = get_logger(__name__)

_SPLIT_SOURCE_BUNDLE_VERSION = 1
_SPLIT_SOURCE_BUNDLE_KIND = "negpy.full-negative-split-source"


def _fsync_file(path: str | os.PathLike[str]) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path: str | os.PathLike[str]) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _to_uint16(arr: np.ndarray) -> np.ndarray:
    """Convert array to uint16. For uint8, replicate byte (x<<8 | x) so 8-bit
    values span the full 16-bit range instead of being capped at 255."""
    if arr.dtype == np.uint16:
        return arr
    if arr.dtype == np.uint8:
        a16 = arr.astype(np.uint16)
        return (a16 << 8) | a16
    return arr.astype(np.uint16)


def _write_temp_tiff(
    data: np.ndarray,
    target_path: str,
    *,
    photometric: str,
    compression: str = "lzw",
    predictor: bool = False,
    dpi: int | None = None,
) -> str:
    """Write `data` to a temp TIFF next to `target_path`. Returns the temp path.

    Caller commits it (os.replace to the real path) and is responsible for
    cleaning up the temp file if anything downstream fails. On failure here,
    the temp file is cleaned up before re-raising and `target_path` itself is
    never touched.

    `compression`/`predictor` default to the archival bundle/full-negative
    writers' validated lzw-no-predictor setting; `write_tiff_16bit` overrides
    both to match upstream v0.37.0's zlib+predictor tuning for that writer.
    """
    fd, tmp_path = tempfile.mkstemp(suffix=".tif", dir=os.path.dirname(target_path) or ".")
    os.close(fd)
    kwargs: dict = {}
    if dpi and dpi > 0:
        # archival provenance: without this, tifffile stamps XResolution=1
        kwargs["resolution"] = (dpi, dpi)
        kwargs["resolutionunit"] = "INCH"
    if photometric == "rgb":
        # Every RGB payload produced by this module is canonical scanner data,
        # not sRGB-encoded display data.  Stamp a durable, versioned TIFF
        # contract so TiffLoader does not apply a second transfer function.
        kwargs["extratags"] = [LINEAR_SCANNER_RGB_EXTRATAG]
    try:
        tifffile.imwrite(tmp_path, data, photometric=photometric, compression=compression, predictor=predictor, **kwargs)
        _fsync_file(tmp_path)
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise
    return tmp_path


def _write_staged_tiff(data: np.ndarray, path: Path, *, photometric: str, dpi: int) -> None:
    temporary = _write_temp_tiff(data, str(path), photometric=photometric, dpi=dpi)
    os.replace(temporary, path)
    _fsync_directory(path.parent)


def _update_array_digest(digest, role: str, array: np.ndarray) -> None:
    contiguous = np.ascontiguousarray(array)
    header = json.dumps(
        {"role": role, "shape": list(contiguous.shape), "dtype": contiguous.dtype.str},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest.update(len(header).to_bytes(8, "big"))
    digest.update(header)
    digest.update(memoryview(contiguous).cast("B"))


def _artifact_evidence(path: Path) -> dict[str, object]:
    inspection = inspect_tiff_payload(path)
    return {
        "path": path.name,
        "sha256": inspection.sha256,
        "bytes": inspection.byte_length,
        "shape": list(inspection.shape),
        "dtype": inspection.dtype,
        "page_count": inspection.page_count,
        "payload_within_file": inspection.payload_within_file,
        "x_resolution": list(inspection.x_resolution) if inspection.x_resolution is not None else None,
        "y_resolution": list(inspection.y_resolution) if inspection.y_resolution is not None else None,
        "resolution_unit": inspection.resolution_unit,
    }


def _require_committed_bundle_evidence(target: Path, manifest: dict[str, object]) -> None:
    artifacts = manifest.get("artifacts")
    if type(artifacts) is not dict or not artifacts:
        raise RuntimeError(f"committed bundle {target} has invalid artifact evidence")
    for role, raw_evidence in artifacts.items():
        if type(role) is not str or type(raw_evidence) is not dict:
            raise RuntimeError(f"committed bundle {target} has invalid artifact evidence")
        evidence = cast(dict[str, object], raw_evidence)
        relative_path = evidence.get("path")
        if type(relative_path) is not str or Path(relative_path).name != relative_path:
            raise RuntimeError(f"committed bundle {target} has invalid artifact evidence")
        try:
            live = _artifact_evidence(target / relative_path)
        except (OSError, ValueError, tifffile.TiffFileError) as exc:
            raise RuntimeError(f"committed bundle {target} has invalid artifact evidence") from exc
        if live != evidence:
            raise RuntimeError(f"committed bundle {target} has conflicting artifact evidence for {role}")


def write_split_source_bundle(
    source: SplitSourceCapture,
    *,
    aligned_ir: np.ndarray,
    ir_valid_mask: np.ndarray,
    output_dir: str | os.PathLike[str],
    dpi: int,
) -> dict[str, object]:
    """Commit one immutable legacy SANE RGB-plus-IR source bundle.

    The bundle directory is addressed by the exact array content and DPI.
    TIFF payloads are written inside a hidden staging directory; the manifest
    is written last, then the complete directory is renamed into place.
    """

    if type(dpi) is not int or dpi < 1:
        raise ValueError("bundle DPI must be a positive integer")
    rgb4x = np.asarray(source.rgb4x)
    rgb1x_proxy = np.asarray(source.rgb1x_proxy)
    ir1x = np.asarray(source.ir1x)
    aligned = np.asarray(aligned_ir)
    valid = np.asarray(ir_valid_mask)
    if rgb4x.ndim != 3 or rgb4x.shape[2] != 3 or rgb4x.dtype != np.uint16:
        raise ValueError("rgb4x must be a uint16 RGB array")
    if rgb1x_proxy.ndim != 3 or rgb1x_proxy.shape[2] != 3 or rgb1x_proxy.dtype != np.uint16:
        raise ValueError("rgb1x_proxy must be a uint16 RGB array")
    if ir1x.ndim != 2 or ir1x.dtype != np.uint16 or ir1x.shape != rgb1x_proxy.shape[:2]:
        raise ValueError("ir1x must be a uint16 plane matching rgb1x_proxy")
    if aligned.ndim != 2 or aligned.dtype != np.uint16 or aligned.shape != rgb4x.shape[:2]:
        raise ValueError("aligned_ir must be a uint16 plane matching rgb4x")
    if valid.ndim != 2 or valid.shape != aligned.shape or valid.dtype != np.bool_:
        raise ValueError("ir_valid_mask must be a bool plane matching aligned_ir")

    valid_u8 = valid.astype(np.uint8) * np.uint8(255)
    arrays = {
        "rgb4x": rgb4x,
        "rgb1x_proxy": rgb1x_proxy,
        "ir1x": ir1x,
        "aligned_ir": aligned,
        "ir_valid_mask": valid_u8,
    }
    digest = sha256()
    digest.update(f"{_SPLIT_SOURCE_BUNDLE_KIND}:{_SPLIT_SOURCE_BUNDLE_VERSION}:{dpi}".encode("ascii"))
    for role, array in arrays.items():
        _update_array_digest(digest, role, array)
    content_sha256 = digest.hexdigest()
    bundle_name = f"split-source-{content_sha256}"
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    target = root / bundle_name
    manifest_path = target / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("content_sha256") != content_sha256:
            raise RuntimeError(f"committed bundle {target} has conflicting content")
        _require_committed_bundle_evidence(target, manifest)
        return cast(dict[str, object], manifest)
    if target.exists():
        raise RuntimeError(f"bundle path {target} exists without a committed manifest")

    staging = Path(tempfile.mkdtemp(prefix=f".{bundle_name}.", dir=root))
    filenames = {
        "rgb4x": "rgb4x.tif",
        "rgb1x_proxy": "rgb1x-proxy.tif",
        "ir1x": "ir1x.tif",
        "aligned_ir": "aligned-ir.tif",
        "ir_valid_mask": "ir-valid.tif",
    }
    try:
        for role, array in arrays.items():
            _write_staged_tiff(
                array,
                staging / filenames[role],
                photometric="rgb" if array.ndim == 3 else "minisblack",
                dpi=dpi,
            )
        artifacts = {role: _artifact_evidence(staging / filenames[role]) for role in arrays}
        manifest: dict[str, object] = {
            "version": _SPLIT_SOURCE_BUNDLE_VERSION,
            "kind": _SPLIT_SOURCE_BUNDLE_KIND,
            "content_sha256": content_sha256,
            "bundle_path": bundle_name,
            "dpi": dpi,
            "artifacts": artifacts,
        }
        with (staging / "manifest.json").open("x", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        _fsync_directory(staging)
        os.rename(staging, target)
        _fsync_directory(root)
        return manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _unused_sibling_path(target_path: str) -> str:
    """Reserve a unique sibling name until the caller atomically replaces it.

    The returned path is used as a rename target for an existing TIFF while a
    replacement pair is committed. Keeping it in the same directory preserves
    the atomic-rename guarantee of ``os.replace`` for each individual file.
    The placeholder must remain present so another process cannot reserve the
    same backup name before the replace.
    """
    fd, backup_path = tempfile.mkstemp(
        prefix=f".{os.path.basename(target_path)}.",
        suffix=".bak",
        dir=os.path.dirname(target_path) or ".",
    )
    os.close(fd)
    return backup_path


def _unlink_if_present(path: str | None) -> None:
    if path is not None and os.path.exists(path):
        os.unlink(path)


def _commit_tiff_pair(tmp_rgb: str, tmp_ir: str | None, rgb_path: str, ir_path: str) -> None:
    """Commit prepared TIFFs while preserving any prior pair on an error.

    This protects against synchronous rename failures in the running process.
    It is not a two-file filesystem transaction: a crash or power loss between
    the sequential renames can leave destinations missing or mixed, with
    sibling backup files requiring recovery.
    """
    backup_rgb = None
    backup_ir = None
    moved_old_rgb = False
    moved_old_ir = False
    committed_rgb = False
    committed_ir = False

    try:
        backup_rgb = _unused_sibling_path(rgb_path) if os.path.exists(rgb_path) else None
        backup_ir = _unused_sibling_path(ir_path) if os.path.exists(ir_path) else None
        if backup_rgb is not None:
            os.replace(rgb_path, backup_rgb)
            moved_old_rgb = True
        if backup_ir is not None:
            os.replace(ir_path, backup_ir)
            moved_old_ir = True

        os.replace(tmp_rgb, rgb_path)
        committed_rgb = True
        tmp_rgb = ""
        if tmp_ir is not None:
            os.replace(tmp_ir, ir_path)
            committed_ir = True
            tmp_ir = None
    except BaseException as commit_error:
        rollback_errors: list[str] = []

        def _restore(target: str, backup: str | None, moved_old: bool, committed_new: bool) -> None:
            try:
                if moved_old and backup is not None:
                    os.replace(backup, target)
                elif committed_new:
                    _unlink_if_present(target)
            except BaseException as rollback_error:
                rollback_errors.append(f"{target}: {rollback_error}")

        # Restore in reverse commit order. A successful os.replace both removes
        # the new payload and returns the old payload to its original name.
        _restore(ir_path, backup_ir, moved_old_ir, committed_ir)
        _restore(rgb_path, backup_rgb, moved_old_rgb, committed_rgb)
        for label, cleanup_path in (("RGB temp", tmp_rgb), ("IR temp", tmp_ir)):
            try:
                _unlink_if_present(cleanup_path)
            except BaseException as cleanup_error:
                rollback_errors.append(f"{label}: {cleanup_error}")
        for label, backup_path, moved_old in (
            ("RGB backup reservation", backup_rgb, moved_old_rgb),
            ("IR backup reservation", backup_ir, moved_old_ir),
        ):
            if backup_path is not None and not moved_old:
                try:
                    _unlink_if_present(backup_path)
                except BaseException as cleanup_error:
                    rollback_errors.append(f"{label}: {cleanup_error}")

        if rollback_errors:
            details = "; ".join(rollback_errors)
            raise RuntimeError(
                f"TIFF pair commit failed and rollback was incomplete; preserved backup files may remain: {details}"
            ) from commit_error
        raise

    # The replacement pair is complete. Backup deletion is cleanup, not part
    # of the pair commit: if deletion fails, keep the valid new pair and leave
    # the uniquely named backup recoverable instead of attempting a late
    # rollback after one backup may already have been removed.
    for backup_path in (backup_rgb, backup_ir):
        try:
            _unlink_if_present(backup_path)
        except OSError as cleanup_error:
            logger.warning(f"Could not remove TIFF transaction backup {backup_path}: {cleanup_error}")
    _fsync_file(rgb_path)
    if os.path.exists(ir_path):
        _fsync_file(ir_path)
    _fsync_directory(os.path.dirname(rgb_path) or ".")


def _commit_tiff_triplet(temporaries: dict[str, str], targets: dict[str, str]) -> None:
    """Commit a prepared RGB/IR/validity triplet with synchronous rollback."""

    roles = ("rgb", "ir", "ir_valid_mask")
    backups: dict[str, str | None] = {role: None for role in roles}
    moved_old: set[str] = set()
    committed_new: set[str] = set()
    try:
        for role in roles:
            target = targets[role]
            if os.path.exists(target):
                backup = _unused_sibling_path(target)
                backups[role] = backup
                os.replace(target, backup)
                moved_old.add(role)
        for role in roles:
            os.replace(temporaries[role], targets[role])
            temporaries[role] = ""
            committed_new.add(role)
    except BaseException as commit_error:
        rollback_errors: list[str] = []
        for role in reversed(roles):
            try:
                backup = backups[role]
                if role in moved_old and backup is not None:
                    os.replace(backup, targets[role])
                elif role in committed_new:
                    _unlink_if_present(targets[role])
            except BaseException as rollback_error:
                rollback_errors.append(f"{targets[role]}: {rollback_error}")
        for role in roles:
            try:
                _unlink_if_present(temporaries[role])
            except BaseException as cleanup_error:
                rollback_errors.append(f"{role} temp: {cleanup_error}")
        for role in roles:
            backup = backups[role]
            if backup is not None and role not in moved_old:
                try:
                    _unlink_if_present(backup)
                except BaseException as cleanup_error:
                    rollback_errors.append(f"{role} backup reservation: {cleanup_error}")
        if rollback_errors:
            raise RuntimeError("TIFF triplet commit failed and rollback was incomplete; " + "; ".join(rollback_errors)) from commit_error
        raise

    for backup in backups.values():
        try:
            _unlink_if_present(backup)
        except OSError as cleanup_error:
            logger.warning(f"Could not remove TIFF transaction backup {backup}: {cleanup_error}")


def write_full_negative_tiff(
    result: ScanResult,
    *,
    ir_valid_mask: np.ndarray,
    path: str | os.PathLike[str],
) -> dict[str, dict[str, object]]:
    """Transactionally persist a full-negative RGB/IR/validity triplet.

    Invalid aligned-IR pixels are written at sensor white so a legacy
    ``ir < threshold`` dust detector cannot mistake a warp border for dust.
    The separate validity TIFF preserves the exact audit mask.
    """

    target = Path(path).expanduser().resolve()
    if target.suffix.lower() not in {".tif", ".tiff"}:
        target = target.with_suffix(".tif")
    target.parent.mkdir(parents=True, exist_ok=True)
    rgb = np.asarray(result.rgb)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint16:
        raise ValueError("full-negative RGB must be a uint16 RGB array")
    if result.ir is None:
        raise ValueError("full-negative output requires aligned IR")
    ir = np.asarray(result.ir)
    valid = np.asarray(ir_valid_mask)
    if ir.ndim != 2 or ir.dtype != np.uint16 or ir.shape != rgb.shape[:2]:
        raise ValueError("full-negative IR must be a uint16 plane matching RGB")
    if valid.ndim != 2 or valid.dtype != np.bool_ or valid.shape != ir.shape:
        raise ValueError("full-negative IR validity must be a bool plane matching IR")
    if type(result.dpi) is not int or result.dpi < 1:
        raise ValueError("full-negative DPI must be a positive integer")

    safe_ir = ir.copy()
    safe_ir[~valid] = np.iinfo(np.uint16).max
    valid_u8 = valid.astype(np.uint8) * np.uint8(255)
    base = target.with_suffix("")
    targets = {
        "rgb": str(target),
        "ir": f"{base}_IR.tif",
        "ir_valid_mask": f"{base}_IR_VALID.tif",
    }
    temporaries: dict[str, str] = {}
    try:
        temporaries["rgb"] = _write_temp_tiff(rgb, targets["rgb"], photometric="rgb", dpi=result.dpi)
        temporaries["ir"] = _write_temp_tiff(safe_ir, targets["ir"], photometric="minisblack", dpi=result.dpi)
        temporaries["ir_valid_mask"] = _write_temp_tiff(
            valid_u8,
            targets["ir_valid_mask"],
            photometric="minisblack",
            dpi=result.dpi,
        )
    except BaseException:
        for temporary in temporaries.values():
            _unlink_if_present(temporary)
        raise

    try:
        evidence = {role: _artifact_evidence(Path(temporary)) for role, temporary in temporaries.items()}
        for role, artifact in evidence.items():
            artifact["path"] = Path(targets[role]).name
    except BaseException:
        for temporary in temporaries.values():
            _unlink_if_present(temporary)
        raise

    _commit_tiff_triplet(temporaries, targets)
    for committed in targets.values():
        _fsync_file(committed)
    _fsync_directory(target.parent)
    return evidence


def write_tiff_16bit(result: ScanResult, path: str) -> str:
    """Write ScanResult to 16-bit TIFF. IR written as sidecar `<basename>_IR.tif`.

    RGB and (if present) IR are encoded completely before either destination
    changes. A synchronous commit error restores the entire prior RGB/IR pair
    and removes new temporary payloads. This is exception-safe, not a claim of
    power-loss atomicity: two filenames require sequential filesystem renames,
    so a process crash or power loss between them can expose a mixed pair.
    When this write has no IR, a stale `<basename>_IR.tif` sidecar from the
    prior pair is removed as part of the same exception-safe replacement.

    Returns final RGB path.
    """
    if not path.lower().endswith((".tif", ".tiff")):
        path = path + ".tif"

    rgb = _to_uint16(result.rgb)
    base = os.path.splitext(path)[0]
    ir_path = f"{base}_IR.tif"
    has_ir = result.ir is not None

    # Phase 1: write both payloads to temp files. Nothing under `path` or
    # `ir_path` is touched here, so a failure at this stage (bad array,
    # codec error, disk full) leaves the filesystem exactly as it was.
    # zlib+predictor (rather than _write_temp_tiff's archival-default lzw)
    # matches upstream v0.37.0's tuning for this writer specifically.
    tmp_rgb = _write_temp_tiff(rgb, path, photometric="rgb", compression="zlib", predictor=True, dpi=result.dpi)
    tmp_ir = None
    if has_ir:
        try:
            ir_data = _to_uint16(result.ir)
            tmp_ir = _write_temp_tiff(ir_data, ir_path, photometric="minisblack", compression="zlib", predictor=True, dpi=result.dpi)
        except Exception:
            os.unlink(tmp_rgb)
            raise

    # Phase 2: swap the prepared payloads in, retaining any prior pair until
    # every requested destination has been committed.
    _commit_tiff_pair(tmp_rgb, tmp_ir, path, ir_path)

    return path


def write_dng_linear(result: ScanResult, path: str) -> str:
    """Write ScanResult to an uncompressed 16-bit LinearRaw DNG via tifffile.

    A LinearRaw DNG is a single-IFD TIFF plus a few DNG tags. If result.ir is
    present it is stacked as an extra sample. Atomic write; returns final path.
    """
    if not path.lower().endswith(".dng"):
        path = path + ".dng"

    rgb = _to_uint16(result.rgb)

    if result.ir is not None:
        ir = result.ir
        if ir.ndim == 2:
            ir = ir[:, :, np.newaxis]
        ir = _to_uint16(ir)
        full_array = np.dstack([rgb, ir])
    else:
        full_array = np.ascontiguousarray(rgb)

    model = result.device_model
    # (code, dtype, count, value, writeonce); NewSubfileType=0 is required or LibRaw rejects the DNG.
    extratags = [
        (254, 4, 1, 0, True),  # NewSubfileType
        (50706, 1, 4, (1, 4, 0, 0), True),  # DNGVersion
        (50707, 1, 4, (1, 0, 0, 0), True),  # DNGBackwardVersion
        (274, 3, 1, 1, True),  # Orientation
        (271, 2, len(model) + 1, model, True),  # Make
        (272, 2, len(model) + 1, model, True),  # Model
    ]
    payload = _encode_dng(full_array, extratags)

    fd, tmp_path = tempfile.mkstemp(suffix=".dng", dir=os.path.dirname(path) or ".")
    os.close(fd)
    try:
        with open(tmp_path, "wb") as fh:
            fh.write(payload)
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise

    return path


def _encode_dng(full_array: np.ndarray, extratags: list) -> bytes:
    """Encode an RGB(+IR) uint16 array as LinearRaw DNG bytes.

    RGB is written with the RGB photometric so tifffile emits a clean 3 *color*
    samples with no ExtraSamples (matching pidng); the PhotometricInterpretation
    tag is then patched to LinearRaw (34892), which DNG requires. Marking colour
    planes as ExtraSamples instead makes some raw processors treat the file as a
    1-channel sensor + aux planes and mis-demosaic it.

    The IR (4-sample) case keeps the LINEAR_RAW photometric with the extra planes
    declared as extra samples — there the 4th plane genuinely is infrared, and
    tifffile has no clean 4-colour-sample form.
    """
    buf = io.BytesIO()
    if full_array.shape[-1] == 3:
        tifffile.imwrite(buf, full_array, photometric=tifffile.PHOTOMETRIC.RGB, compression=None, metadata=None, extratags=extratags)
        data = bytearray(buf.getvalue())
        with tifffile.TiffFile(io.BytesIO(bytes(data))) as tf:
            page = cast(tifffile.TiffPage, tf.pages[0])
            offset = page.tags["PhotometricInterpretation"].valueoffset
            byteorder = tf.byteorder
        struct.pack_into(byteorder + "H", data, offset, 34892)  # RGB(2) → LinearRaw(34892)
        return bytes(data)

    extrasamples = (0,) * (full_array.shape[-1] - 1)
    tifffile.imwrite(
        buf,
        full_array,
        photometric=tifffile.PHOTOMETRIC.LINEAR_RAW,
        compression=None,
        metadata=None,
        extrasamples=extrasamples,
        extratags=extratags,
    )
    return buf.getvalue()
