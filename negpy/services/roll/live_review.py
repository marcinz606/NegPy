"""Pinned operator-review evidence for one LS-5000 live acceptance run."""

from __future__ import annotations

import hashlib
import hmac
import io
import json
import os
import stat
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, TypeGuard, cast

import numpy as np
from PIL import Image


SCHEMA = "negpy.ls5000-reviewed-approval.v1"
REVIEW_BASIS = "visual-inspection-of-six-frame-contact-sheet-and-canonical-restored-thumbnails"
APPROVED_SLOTS = (1, 6)
_SHA256_CHARS = frozenset("0123456789abcdef")
_MAX_REVIEW_BYTES = 64 * 1024
_MAX_CONTACT_SHEET_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True)
class ValidatedReviewedApproval:
    """One SHA-pinned visual review, rederived from the restored preview."""

    sha256: str
    byte_length: int
    reviewed_fingerprint_sha256: str
    contact_sheet_path: Path
    contact_sheet_sha256: str
    approvals: Mapping[int, object]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "approvals",
            MappingProxyType(dict(self.approvals)),
        )


def _is_sha256(value: object) -> TypeGuard[str]:
    return type(value) is str and len(value) == 64 and not (set(value) - _SHA256_CHARS)


def _identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _stable_regular_bytes(
    path: Path,
    *,
    maximum_bytes: int,
    label: str,
) -> bytes:
    descriptor: int | None = None
    try:
        before = path.lstat()
        if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
            raise ValueError(f"{label} must be a regular non-symlink file")
        if not 1 <= before.st_size <= maximum_bytes:
            raise ValueError(f"{label} has an invalid byte length")
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
        descriptor = os.open(path, flags)
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or _identity(opened) != _identity(before):
            raise ValueError(f"{label} changed while opening")
        chunks: list[bytes] = []
        total = 0
        while chunk := os.read(descriptor, min(1024 * 1024, maximum_bytes + 1 - total)):
            chunks.append(chunk)
            total += len(chunk)
            if total > maximum_bytes:
                raise ValueError(f"{label} exceeds its safe size limit")
        after = os.fstat(descriptor)
        final = path.lstat()
    except FileNotFoundError as error:
        raise ValueError(f"{label} is missing: {path}") from error
    except OSError as error:
        raise ValueError(f"could not read {label}: {error}") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
    if _identity(opened) != _identity(after) or _identity(opened) != _identity(final):
        raise ValueError(f"{label} changed while reading")
    payload = b"".join(chunks)
    if len(payload) != opened.st_size:
        raise ValueError(f"{label} changed byte length while reading")
    return payload


def _strict_json_object(payload: bytes, *, label: str) -> dict[str, Any]:
    def unique(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate key {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> object:
        raise ValueError(f"non-finite constant {value!r}")

    try:
        document = json.loads(
            payload,
            object_pairs_hook=unique,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"{label} is invalid JSON: {error}") from error
    if type(document) is not dict:
        raise ValueError(f"{label} must contain a JSON object")
    return document


def thumbnail_sha256(image: np.ndarray) -> str:
    """Reproduce CoolscanPy's ManualFrameApproval thumbnail identity."""

    thumbnail = np.asarray(image)
    if thumbnail.dtype != np.uint16 or thumbnail.ndim != 3 or thumbnail.shape[2] != 3:
        raise ValueError("reviewed thumbnail must be HxWx3 uint16")
    digest = hashlib.sha256()
    digest.update(str(thumbnail.shape).encode("ascii"))
    digest.update(thumbnail.dtype.str.encode("ascii"))
    digest.update(np.ascontiguousarray(thumbnail).tobytes())
    return digest.hexdigest()


def approval_payload(value: object) -> dict[str, Any]:
    """Return a full ManualFrameApproval payload without trusting duck types."""

    to_payload = getattr(value, "to_payload", None)
    payload = to_payload() if callable(to_payload) else value
    if type(payload) is not dict or any(type(key) is not str for key in payload):
        raise ValueError("manual approval did not return a payload object")
    return dict(cast(dict[str, Any], payload))


def _default_session_loader(payload: str) -> object:
    from coolscanpy.roll.preview_session import RollPreviewSession

    return RollPreviewSession.from_json(payload)


def _default_approval_parser(payload: object) -> object:
    from coolscanpy.protocol.ls5000_single_pass.capture_process import (
        ManualFrameApproval,
    )

    return ManualFrameApproval.from_payload(payload)


def load_reviewed_approval(
    review_path: Path,
    expected_sha256: str,
    *,
    preview_session_path: Path,
    preview_session_payload: str,
    preview_session_sha256: str,
    session_loader: Callable[[str], object] = _default_session_loader,
    approval_parser: Callable[[object], object] = _default_approval_parser,
) -> ValidatedReviewedApproval:
    """Load, pin, and rederive the exact approvals for selected edge slots."""

    if not _is_sha256(expected_sha256):
        raise ValueError("reviewed-approval pin must be a lowercase SHA-256 digest")
    if not _is_sha256(preview_session_sha256):
        raise ValueError("preview-session SHA-256 is malformed")
    review_bytes = _stable_regular_bytes(
        review_path,
        maximum_bytes=_MAX_REVIEW_BYTES,
        label="reviewed approval",
    )
    actual_sha256 = hashlib.sha256(review_bytes).hexdigest()
    if not hmac.compare_digest(actual_sha256, expected_sha256):
        raise ValueError(f"reviewed-approval SHA-256 mismatch: expected {expected_sha256}, got {actual_sha256}")
    document = _strict_json_object(review_bytes, label="reviewed approval")
    if set(document) != {
        "approvals",
        "contact_sheet",
        "preview_session",
        "review_basis",
        "reviewed_fingerprint_sha256",
        "schema",
    }:
        raise ValueError("reviewed approval fields are incomplete")
    if document.get("schema") != SCHEMA or document.get("review_basis") != REVIEW_BASIS:
        raise ValueError("reviewed approval schema or review basis is unsupported")

    session_row = document.get("preview_session")
    if type(session_row) is not dict or set(session_row) != {"bytes", "path", "sha256"}:
        raise ValueError("reviewed approval preview-session binding is malformed")
    try:
        session_path = Path(session_row["path"]).absolute()
    except TypeError as error:
        raise ValueError("reviewed approval preview-session path is malformed") from error
    if (
        session_path != preview_session_path.absolute()
        or session_row.get("sha256") != preview_session_sha256
        or session_row.get("bytes") != len(preview_session_payload.encode("utf-8"))
    ):
        raise ValueError("reviewed approval belongs to another preview session")

    contact_row = document.get("contact_sheet")
    if type(contact_row) is not dict or set(contact_row) != {"bytes", "path", "sha256"}:
        raise ValueError("reviewed approval contact-sheet binding is malformed")
    if type(contact_row.get("path")) is not str or not _is_sha256(contact_row.get("sha256")):
        raise ValueError("reviewed approval contact-sheet identity is malformed")
    contact_path = Path(contact_row["path"]).absolute()
    contact_bytes = _stable_regular_bytes(
        contact_path,
        maximum_bytes=_MAX_CONTACT_SHEET_BYTES,
        label="reviewed contact sheet",
    )
    if contact_row.get("bytes") != len(contact_bytes) or not hmac.compare_digest(
        hashlib.sha256(contact_bytes).hexdigest(),
        contact_row["sha256"],
    ):
        raise ValueError("reviewed contact-sheet content changed")
    try:
        with Image.open(io.BytesIO(contact_bytes)) as image:
            if image.format != "PNG" or image.width < 6 or image.height < 1:
                raise ValueError("reviewed contact sheet has invalid PNG geometry")
            image.load()
    except Exception as error:
        raise ValueError(f"reviewed contact sheet is not a valid PNG: {error}") from error

    session = session_loader(preview_session_payload)
    reviewed_fingerprint = getattr(session, "reviewed_fingerprint", None)
    if not callable(reviewed_fingerprint):
        raise ValueError("restored preview has no reviewed fingerprint")
    fingerprint = getattr(reviewed_fingerprint(), "binding_sha256", None)
    if not _is_sha256(fingerprint) or document.get("reviewed_fingerprint_sha256") != fingerprint:
        raise ValueError("reviewed approval fingerprint does not match the preview")

    rows = document.get("approvals")
    if type(rows) is not list or len(rows) != len(APPROVED_SLOTS):
        raise ValueError("reviewed approval must contain exactly slots 1 and 6")
    approvals: dict[int, object] = {}
    for expected_slot, row in zip(APPROVED_SLOTS, rows, strict=True):
        parsed = approval_parser(row)
        parsed_payload = approval_payload(parsed)
        if parsed_payload != row or parsed_payload.get("slot") != expected_slot:
            raise ValueError("reviewed approval payload is non-canonical or out of order")
        approve = getattr(session, "approve_manual_origin", None)
        if not callable(approve):
            raise ValueError("restored preview cannot rederive manual approvals")
        derived = approve(expected_slot, parsed_payload["boundary_offset_rows"])
        if approval_payload(derived) != parsed_payload:
            raise ValueError(f"reviewed approval for slot {expected_slot} does not match its thumbnail")
        approvals[expected_slot] = parsed

    return ValidatedReviewedApproval(
        sha256=actual_sha256,
        byte_length=len(review_bytes),
        reviewed_fingerprint_sha256=fingerprint,
        contact_sheet_path=contact_path,
        contact_sheet_sha256=contact_row["sha256"],
        approvals=approvals,
    )


def validate_restored_thumbnails(
    thumbnails: object,
    review: ValidatedReviewedApproval,
) -> None:
    """Cross-check returned thumbnails against the already-pinned review."""

    if not isinstance(thumbnails, (list, tuple)):
        raise ValueError("restored thumbnails must be an ordered sequence")
    by_slot = {getattr(item, "slot", None): item for item in thumbnails}
    if set(by_slot) != set(range(1, 7)):
        raise ValueError("restored thumbnails are not exactly slots 1 through 6")
    for slot, approval in review.approvals.items():
        image = getattr(by_slot[slot], "image", None)
        expected = approval_payload(approval).get("thumbnail_sha256")
        if thumbnail_sha256(np.asarray(image)) != expected:
            raise ValueError(f"restored thumbnail for slot {slot} changed after review")
