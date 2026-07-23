from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from negpy.services.roll.live_review import (
    REVIEW_BASIS,
    SCHEMA,
    load_reviewed_approval,
    thumbnail_sha256,
    validate_restored_thumbnails,
)


class _Approval:
    def __init__(self, payload: dict) -> None:
        self._payload = dict(payload)

    def to_payload(self) -> dict:
        return dict(self._payload)


class _Session:
    def __init__(self, fingerprint: str, approvals: dict[int, dict]) -> None:
        self._fingerprint = fingerprint
        self._approvals = approvals

    def reviewed_fingerprint(self):
        return SimpleNamespace(binding_sha256=self._fingerprint)

    def approve_manual_origin(self, slot: int, offset: int):
        assert self._approvals[slot]["boundary_offset_rows"] == offset
        return _Approval(self._approvals[slot])


def _approval(slot: int, thumbnail: str, fingerprint: str) -> dict:
    payload = {
        "boundary_offset_rows": 0,
        "review_reasons": [f"slot-{slot}-edge-review"],
        "reviewed_fingerprint_sha256": fingerprint,
        "reviewed_lookup_row": slot,
        "reviewed_native_origin": slot * 100,
        "schema_version": 1,
        "slot": slot,
        "thumbnail_sha256": thumbnail,
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()
    return {**payload, "binding_sha256": hashlib.sha256(encoded).hexdigest()}


def _fixture(tmp_path: Path):
    fingerprint = "c" * 64
    images = {slot: np.full((2, 3, 3), slot, dtype=np.uint16) for slot in range(1, 7)}
    approvals = {slot: _approval(slot, thumbnail_sha256(images[slot]), fingerprint) for slot in (1, 6)}
    session_payload = '{"version":1}'
    session_path = tmp_path / "session.json"
    session_path.write_text(session_payload, encoding="utf-8")
    session_sha = hashlib.sha256(session_payload.encode()).hexdigest()
    contact_path = tmp_path / "contact.png"
    Image.new("RGB", (6, 2), "white").save(contact_path)
    contact_bytes = contact_path.read_bytes()
    document = {
        "approvals": [approvals[1], approvals[6]],
        "contact_sheet": {
            "bytes": len(contact_bytes),
            "path": str(contact_path),
            "sha256": hashlib.sha256(contact_bytes).hexdigest(),
        },
        "preview_session": {
            "bytes": len(session_payload.encode()),
            "path": str(session_path),
            "sha256": session_sha,
        },
        "review_basis": REVIEW_BASIS,
        "reviewed_fingerprint_sha256": fingerprint,
        "schema": SCHEMA,
    }
    review_path = tmp_path / "review.json"
    review_path.write_text(json.dumps(document), encoding="utf-8")
    review_sha = hashlib.sha256(review_path.read_bytes()).hexdigest()
    session = _Session(fingerprint, approvals)
    return (
        review_path,
        review_sha,
        session_path,
        session_payload,
        session_sha,
        session,
        images,
        document,
    )


def _load(values):
    review_path, review_sha, session_path, payload, session_sha, session, *_ = values
    return load_reviewed_approval(
        review_path,
        review_sha,
        preview_session_path=session_path,
        preview_session_payload=payload,
        preview_session_sha256=session_sha,
        session_loader=lambda _payload: session,
        approval_parser=lambda row: _Approval(row),
    )


def test_loads_pinned_review_and_rederives_both_approvals(tmp_path: Path) -> None:
    values = _fixture(tmp_path)
    reviewed = _load(values)

    assert reviewed.reviewed_fingerprint_sha256 == "c" * 64
    assert tuple(reviewed.approvals) == (1, 6)

    images = values[6]
    thumbnails = [SimpleNamespace(slot=slot, image=images[slot]) for slot in range(1, 7)]
    validate_restored_thumbnails(thumbnails, reviewed)


def test_derives_required_approvals_from_the_restored_preview(tmp_path: Path) -> None:
    values = list(_fixture(tmp_path))
    approvals = {1: values[7]["approvals"][0]}
    document = dict(values[7])
    document["approvals"] = [approvals[1]]
    values[0].write_text(json.dumps(document), encoding="utf-8")
    values[1] = hashlib.sha256(values[0].read_bytes()).hexdigest()

    session = _Session("c" * 64, approvals)
    session.slots = tuple(
        SimpleNamespace(
            slot_id=slot,
            base_origin=SimpleNamespace(manual_review=(slot in (1, 7))),
        )
        for slot in range(1, 8)
    )
    session.selected_slots = (1, 2, 3, 4, 5, 6)
    values[5] = session

    reviewed = _load(tuple(values))
    assert tuple(reviewed.approvals) == (1,)


def test_wrong_review_hash_fails_before_parsing(tmp_path: Path) -> None:
    values = list(_fixture(tmp_path))
    values[1] = "0" * 64

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        _load(tuple(values))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema", "wrong", "schema or review basis"),
        ("reviewed_fingerprint_sha256", "d" * 64, "fingerprint"),
    ],
)
def test_review_identity_changes_fail_closed(
    tmp_path: Path,
    field: str,
    value: str,
    message: str,
) -> None:
    values = list(_fixture(tmp_path))
    document = dict(values[7])
    document[field] = value
    values[0].write_text(json.dumps(document), encoding="utf-8")
    values[1] = hashlib.sha256(values[0].read_bytes()).hexdigest()

    with pytest.raises(ValueError, match=message):
        _load(tuple(values))


def test_changed_approval_is_not_reaccepted_under_a_new_file_hash(tmp_path: Path) -> None:
    values = list(_fixture(tmp_path))
    document = dict(values[7])
    rows = [dict(row) for row in document["approvals"]]
    rows[0]["thumbnail_sha256"] = "e" * 64
    document["approvals"] = rows
    values[0].write_text(json.dumps(document), encoding="utf-8")
    values[1] = hashlib.sha256(values[0].read_bytes()).hexdigest()

    with pytest.raises(ValueError, match="does not match its thumbnail"):
        _load(tuple(values))


def test_restored_thumbnail_change_fails(tmp_path: Path) -> None:
    values = _fixture(tmp_path)
    reviewed = _load(values)
    images = dict(values[6])
    images[1] = images[1].copy()
    images[1][0, 0, 0] += 1
    thumbnails = [SimpleNamespace(slot=slot, image=images[slot]) for slot in range(1, 7)]

    with pytest.raises(ValueError, match="slot 1 changed"):
        validate_restored_thumbnails(thumbnails, reviewed)


def test_review_mapping_is_immutable(tmp_path: Path) -> None:
    reviewed = _load(_fixture(tmp_path))
    with pytest.raises(TypeError):
        reviewed.approvals[1] = object()  # type: ignore[index]


def test_thumbnail_validator_rejects_non_rgb16() -> None:
    with pytest.raises(ValueError, match="HxWx3 uint16"):
        thumbnail_sha256(np.zeros((2, 2), dtype=np.uint8))
