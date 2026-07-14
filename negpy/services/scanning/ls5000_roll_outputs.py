"""Promote verified LS-5000 frame artifacts into the user's scan folder.

Single-pass finalization deliberately happens beside its capture journal so a
crash can be resumed and audited.  The desktop workflow still needs ordinary,
discoverable TIFF names in the folder the user selected.  This module creates
that public artifact set without overwriting files and writes a small receipt
binding it back to the verified capture manifest.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date as dt_date
from pathlib import Path
from typing import Any, Protocol

from negpy.services.scanning.templating import (
    render_scan_filename,
    require_sequence_varying_scan_filename,
)


class RollOutputError(RuntimeError):
    """A verified attempt could not be published as an output set."""


class _FinalizationLike(Protocol):
    manifest_path: Path
    output_paths: Mapping[str, Path]
    manifest: Mapping[str, object]


@dataclass(frozen=True)
class PromotedFrame:
    """One collision-free RGB/IR master set in the user's scan folder."""

    rgb_path: Path
    ir_path: Path
    ir_valid_mask_path: Path
    receipt_path: Path
    sequence: int
    selected_slot: int | None = None


def _sha256(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
            size += len(block)
    return digest.hexdigest(), size


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _copy_exclusive(source: Path, destination: Path) -> None:
    """Publish one file without a replace/overwrite race.

    A hard link is instant when capture scratch and output share a filesystem.
    The exclusive-copy fallback covers a separately mounted output folder.
    """

    try:
        os.link(source, destination)
        return
    except FileExistsError:
        raise
    except OSError:
        pass

    try:
        with source.open("rb") as reader, destination.open("xb") as writer:
            shutil.copyfileobj(reader, writer, length=8 * 1024 * 1024)
            writer.flush()
            os.fsync(writer.fileno())
    except BaseException:
        destination.unlink(missing_ok=True)
        raise


def _write_receipt_exclusive(path: Path, payload: dict[str, Any]) -> None:
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    linked = False
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        linked = True
        temporary.unlink()
    finally:
        if not linked:
            temporary.unlink(missing_ok=True)


def promote_single_pass_frame(
    finalization: _FinalizationLike,
    *,
    output_folder: Path | str,
    filename_pattern: str,
    minimum_sequence: int = 1,
    selected_slot: int | None = None,
) -> PromotedFrame:
    """Publish a verified RGB/IR triplet and provenance receipt.

    ``minimum_sequence`` lets a selected-frame queue retain monotonically
    increasing names.  Existing RGB, IR, mask, or receipt files all reserve a
    basename, so a stray sidecar can never be paired with the wrong master.
    """

    if type(minimum_sequence) is not int or minimum_sequence < 1:
        raise ValueError("minimum_sequence must be a positive integer")
    if selected_slot is not None and (
        type(selected_slot) is not int or not 1 <= selected_slot <= 40
    ):
        raise ValueError("selected_slot must be an integer in 1..40 or None")
    source_paths = {key: Path(value).expanduser().resolve() for key, value in finalization.output_paths.items()}
    if set(source_paths) != {"rgb", "ir", "ir_valid_mask"}:
        raise RollOutputError("finalization must provide RGB, IR, and IR-valid-mask outputs")
    manifest_path = Path(finalization.manifest_path).expanduser().resolve()
    if not manifest_path.is_file() or any(not path.is_file() for path in source_paths.values()):
        raise RollOutputError("verified finalization artifacts are missing")
    try:
        committed_manifest = json.loads(manifest_path.read_bytes())
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as error:
        raise RollOutputError(f"verified finalization manifest is unreadable: {error}") from error
    if (
        type(committed_manifest) is not dict
        or dict(finalization.manifest) != committed_manifest
    ):
        raise RollOutputError(
            "verified finalization manifest differs from its in-memory evidence"
        )
    identity = committed_manifest.get("identity")
    manifest_slot = identity.get("selected_slot") if type(identity) is dict else None
    if manifest_slot is not None and (
        type(manifest_slot) is not int or not 1 <= manifest_slot <= 40
    ):
        raise RollOutputError("verified finalization has an invalid selected slot")
    if selected_slot is not None and manifest_slot is None:
        raise RollOutputError(
            "verified finalization has no selected slot to bind the requested output"
        )
    if (
        selected_slot is not None
        and manifest_slot is not None
        and selected_slot != manifest_slot
    ):
        raise RollOutputError(
            "requested output slot differs from the verified finalization"
        )
    effective_slot = selected_slot if selected_slot is not None else manifest_slot

    root = Path(output_folder).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    date = dt_date.today().strftime("%Y%m%d")
    sequence = minimum_sequence
    require_sequence_varying_scan_filename(
        filename_pattern,
        date,
        sequence,
        slot=effective_slot,
    )
    seen_basenames: set[str] = set()
    while True:
        basename = render_scan_filename(
            filename_pattern,
            date,
            sequence,
            slot=effective_slot,
        )
        if basename in seen_basenames:
            raise RollOutputError(
                "filename pattern repeated a basename while resolving an output collision"
            )
        seen_basenames.add(basename)
        candidates = {
            "rgb": root / f"{basename}.tif",
            "ir": root / f"{basename}_IR.tif",
            "ir_valid_mask": root / f"{basename}_IR_VALID.tif",
            "receipt": root / f"{basename}_SCAN.json",
        }
        if not any(path.exists() or path.is_symlink() for path in candidates.values()):
            break
        sequence += 1

    created: list[Path] = []
    try:
        for role in ("rgb", "ir", "ir_valid_mask"):
            source = source_paths[role]
            destination = candidates[role]
            _copy_exclusive(source, destination)
            created.append(destination)
            source_hash = _sha256(source)
            destination_hash = _sha256(destination)
            if destination_hash != source_hash:
                raise RollOutputError(f"promoted {role} differs from its verified source")

        manifest_sha, manifest_bytes = _sha256(manifest_path)
        capture_summary: dict[str, object] = {}
        if type(identity) is dict and type(identity.get("boundary_offset_rows")) is int:
            capture_summary["boundary_offset_rows"] = identity[
                "boundary_offset_rows"
            ]
        quality_control = committed_manifest.get("quality_control")
        if type(quality_control) is dict:
            capture_summary["quality_control"] = quality_control
        frame_evidence = committed_manifest.get("frame_evidence")
        if type(frame_evidence) is dict:
            for key in ("roll_identity", "manual_review_approval"):
                if key in frame_evidence:
                    capture_summary[key] = frame_evidence[key]
        receipt = {
            "version": 1,
            "kind": "negpy.ls5000-promoted-frame",
            "roll_slot": effective_slot,
            "capture_summary": capture_summary,
            "capture_manifest": {
                # Keep machine-specific home-directory paths out of portable
                # scan receipts.  The hash is the durable binding; these names
                # are only a local recovery hint.
                "attempt": manifest_path.parent.parent.name,
                "path": manifest_path.name,
                "sha256": manifest_sha,
                "bytes": manifest_bytes,
            },
            "outputs": {
                role: {
                    "path": candidates[role].name,
                    "sha256": _sha256(candidates[role])[0],
                    "bytes": candidates[role].stat().st_size,
                }
                for role in ("rgb", "ir", "ir_valid_mask")
            },
        }
        _write_receipt_exclusive(candidates["receipt"], receipt)
        created.append(candidates["receipt"])
        _fsync_directory(root)
    except BaseException as error:
        for path in reversed(created):
            path.unlink(missing_ok=True)
        if isinstance(error, (RollOutputError, FileExistsError)):
            raise
        raise RollOutputError(f"could not publish verified frame: {error}") from error

    return PromotedFrame(
        rgb_path=candidates["rgb"],
        ir_path=candidates["ir"],
        ir_valid_mask_path=candidates["ir_valid_mask"],
        receipt_path=candidates["receipt"],
        sequence=sequence,
        selected_slot=effective_slot,
    )


__all__ = [
    "PromotedFrame",
    "RollOutputError",
    "promote_single_pass_frame",
]
