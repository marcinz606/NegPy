"""Fail-closed native LS-5000 prescan-to-pre-F builder."""

from __future__ import annotations

import base64
import hashlib
import json
import math
import re
import struct
from dataclasses import dataclass
from typing import Final, cast

import numpy as np

from negpy.services.roll import exact_color


CHANNELS: Final = ("r", "g", "b")
ANALYZER_SHAPE: Final = (425, 281, 3)
ANALYZER_RESOLUTION_DPI: Final = 285
DENSITY_SOURCE_RESOLUTION_DPI: Final = 97
# The scanner has two proven 97-dpi roll-preview geometries. The shorter
# variant is produced by the live 37-record full-roll table; accepting it is
# deliberately a closed whitelist, not a variable-length tolerance.
SUPPORTED_DENSITY_SOURCE_WIRE_BYTES: Final = frozenset((6_250_496, 5_804_032))
DENSITY_ARITHMETIC: Final = "ls5000-md3-10088810-layout1-u16-proven-inputs-macos-binary64-exact-v6"
FRAME_OWNERSHIP_STATUS: Final = "proven-exact-reservation-preview-registration-and-transport"
DENSITY_PER_FRAME_BINDING_STATUS: Final = "requires-explicit-frame-ownership-receipt"
RESOURCE_SHA256: Final = exact_color.NATIVE_RESOURCE_SHA256
PARAMETER_ALGORITHM_ID: Final = "ls5000-md3-100100d0-to-1000f470-v1"
CURVE_ALGORITHM_ID: Final = "ls5000-md3-10010c30-pref-v1"

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_RESOURCE_B64 = "AAAAAAAADMCOBvAWSFALwPRsVn2utgnAAAAAAAAACMA/NV66SYwHwGb35GGhVgfAZapgVFInBsDBOSNKe4MFwCegibDh6QTAjgbwFkhQA8D0bFZ9rrYBwAAAAAAAAADAMzMzMzMz+79mZmZmZmb2v5qZmZmZmfG/mpmZmZmZ6b8AAAAAAADgv5qZmZmZmcm/mpmZmZmZuT+amZmZmZnZP2ZmZmZmZuY/AAAAAAAA8D/NzMzMzMz0P5qZmZmZmfk/ZmZmZmZm/j+amZmZmZkBQAAAAAAAAARAZmZmZmZmBkDNzMzMzMwIQDMzMzMzMwtAmpmZmZmZDUAAAAAAAAAQQNNNYhBYOdQ/001iEFg51D/TTWIQWDnUP9NNYhBYOdQ/001iEFg51D/TTWIQWDnUP5hMFYxK6tg/F0hQ/Bhz2z9t5/up8dLdPxSuR+F6FOI/c2iR7Xw/5T+kcD0K16PoPzEIrBxaZO0/30+Nl24S8T+mm8QgsHLzP23n+6nx0vU/MzMzMzMz+D/6fmq8dJP6P8HKoUW28/w/hxbZzvdT/z8nMQisHNoAQArXo3A9CgJA7nw/NV46A0DRItv5fmoEQLTIdr6fmgVAmG4Sg8DKBkB7FK5H4foHQF66SQwCKwlAQmDl0CJbCkAlBoGVQ4sLQAisHFpkuwxA7FG4HoXrDUAv3SQGgZXnPy/dJAaBlec/L90kBoGV5z8v3SQGgZXnPy/dJAaBlec/FK5H4XoU6D8dyeU/pN/qP/yp8dJNYuw/PnlYqDXN7T+1FfvL7snwP8zuycNCrfI/irDh6ZWy9D8r9pfdk4f3P807TtGRXPo/b4EExY8x/T+IY13cRgMAQFmGONbFbQFAKqkT0ETYAkD7y+7Jw0IEQMzuycNCrQVAnRGlvcEXB0BuNIC3QIIIQD9XW7G/7AlAEHo2qz5XC0DgnBGlvcEMQLG/7J48LA5AguLHmLuWD0CqglFJnYAQQBIUP8bcNRFAeqUsQxzrEUDjNhrAW6ASQEvIBz2bVRNAJzEIrBxa7D8nMQisHFrsPycxCKwcWuw/JzEIrBxa7D9JnYAmwobtP1D8GHPXEu4/+n5qvHST8D+cxCCwcmjxPxx8YTJVMPI/MCqpE9BE9D9F2PD0Sln2P/p+arx0k/g/GQRWDi2y+z83iUFg5dD+PyuHFtnO9wBAukkMAiuHAkBKDAIrhxYEQNnO91PjpQVAaJHtfD81B0D4U+Olm8QIQIcW2c73UwpAF9nO91PjC0Cmm8QgsHINQDVeukkMAg9AYhBYObRIEECq8dJNYhARQPLSTWIQ2BFAObTIdr6fEkCBlUOLbGcTQMl2vp8aLxRAEFg5tMj2FEBYObTIdr4VQA=="
_RESOURCE = base64.b64decode(_RESOURCE_B64)
if len(_RESOURCE) != 1_024 or hashlib.sha256(_RESOURCE).hexdigest() != RESOURCE_SHA256:
    raise RuntimeError("packaged Nikon characteristic resource failed its hash pin")
_RESOURCE_TABLES = tuple(np.array(struct.unpack_from("<32d", _RESOURCE, index * 0x100), dtype=np.float64) for index in range(4))


@dataclass(frozen=True)
class NativeBuilderEvidence:
    """Distinct density, analyzer, and final-exposure evidence for one frame."""

    session_id: str
    capture_attempt_id: str
    scan_identity: str
    slot: int
    density_source_wire_sha256: str
    density_source_child_sha256: str
    calibration_numerators: tuple[int, int, int]
    density_f03_denominators: tuple[int, int, int]
    densities: tuple[float, float, float]
    density_arithmetic: str
    frame_ownership_status: str
    frame_ownership_receipt: bytes
    frame_ownership_receipt_sha256: str
    density_evidence_receipt: bytes
    density_evidence_receipt_sha256: str
    reservation_id: str
    batch_session_id: str
    preview_sha256: str
    preview_identity_sha256: str
    transport_table_sha256: str
    transport_identity_sha256: str
    reviewed_fingerprint_sha256: str
    fresh_fingerprint_sha256: str
    frame_index: int
    frame_total: int
    selected_slots: tuple[int, ...]
    analyzer_rgb: np.ndarray
    analyzer_rgb_sha256: str
    analyzer_resolution_dpi: int
    analyzer_rectangle: tuple[int, int, int, int]
    final_f02_denominators: tuple[int, int, int]


@dataclass(frozen=True)
class _AnalyzerOutputs:
    all_lower: tuple[float, float, float]
    all_max: tuple[float, float, float]
    gated_lower: tuple[float, float, float]
    gated_max: tuple[float, float, float]


@dataclass(frozen=True)
class _BuilderArgs:
    A: float
    B: float
    E: float
    C: float
    a: float
    f: float
    g: float


def adapt_native_builder_evidence(evidence: object) -> NativeBuilderEvidence:
    """Convert Coolscanpy's exact neutral producer at the app boundary.

    The optional scanner dependency stays outside NegPy's builder core. Only
    the exact Coolscanpy dataclass is accepted; similarly shaped objects are
    deliberately rejected. ``build_native_builder_receipt`` subsequently
    revalidates every copied identity, receipt digest, analyzer digest, and
    numeric field under NegPy's own contract.
    """

    if type(evidence) is NativeBuilderEvidence:
        return evidence
    try:
        from coolscanpy.protocol.ls5000_single_pass.density import (
            NikonExactBuilderEvidence,
        )
    except ImportError as error:
        raise exact_color.ExactColorUnavailable("frame native builder evidence has an invalid type") from error
    if type(evidence) is not NikonExactBuilderEvidence:
        raise exact_color.ExactColorUnavailable("frame native builder evidence has an invalid type")
    return NativeBuilderEvidence(
        session_id=evidence.session_id,
        capture_attempt_id=evidence.capture_attempt_id,
        scan_identity=evidence.scan_identity,
        slot=evidence.slot,
        density_source_wire_sha256=evidence.density_source_wire_sha256,
        density_source_child_sha256=evidence.density_source_child_sha256,
        calibration_numerators=evidence.calibration_numerators,
        density_f03_denominators=evidence.density_f03_denominators,
        densities=evidence.densities,
        density_arithmetic=evidence.density_arithmetic,
        frame_ownership_status=evidence.frame_ownership_status,
        frame_ownership_receipt=evidence.frame_ownership_receipt,
        frame_ownership_receipt_sha256=evidence.frame_ownership_receipt_sha256,
        density_evidence_receipt=evidence.density_evidence_receipt,
        density_evidence_receipt_sha256=evidence.density_evidence_receipt_sha256,
        reservation_id=evidence.reservation_id,
        batch_session_id=evidence.batch_session_id,
        preview_sha256=evidence.preview_sha256,
        preview_identity_sha256=evidence.preview_identity_sha256,
        transport_table_sha256=evidence.transport_table_sha256,
        transport_identity_sha256=evidence.transport_identity_sha256,
        reviewed_fingerprint_sha256=evidence.reviewed_fingerprint_sha256,
        fresh_fingerprint_sha256=evidence.fresh_fingerprint_sha256,
        frame_index=evidence.frame_index,
        frame_total=evidence.frame_total,
        selected_slots=evidence.selected_slots,
        analyzer_rgb=evidence.analyzer_rgb,
        analyzer_rgb_sha256=evidence.analyzer_rgb_sha256,
        analyzer_resolution_dpi=evidence.analyzer_resolution_dpi,
        analyzer_rectangle=evidence.analyzer_rectangle,
        final_f02_denominators=evidence.final_f02_denominators,
    )


def build_native_builder_receipt(
    evidence: NativeBuilderEvidence,
) -> exact_color.NativeValidatedBuilderReceipt:
    """Derive fresh pre-F LUTs only after every provenance gate closes."""

    analyzer, analyzer_bytes = _validate_evidence(evidence)
    r0, c0, r1, c1 = evidence.analyzer_rectangle
    selected = analyzer[r0 : r1 + 1, c0 : c1 + 1].reshape(-1, 3)
    exposure = (
        evidence.calibration_numerators[0] / evidence.final_f02_denominators[0],
        evidence.calibration_numerators[1] / evidence.final_f02_denominators[1],
        evidence.calibration_numerators[2] / evidence.final_f02_denominators[2],
    )
    args, controls = _derive_parameters(selected, exposure, evidence.densities)
    built_luts = [
        _build_lut(arg, controls[index][0], controls[index][1]).astype("<u2", copy=False).tobytes() for index, arg in enumerate(args)
    ]
    luts = (built_luts[0], built_luts[1], built_luts[2])
    evidence_document = _evidence_payload(evidence, analyzer)
    evidence_payload = _canonical_json(evidence_document)
    pre_f_hashes = tuple(hashlib.sha256(blob).hexdigest() for blob in luts)
    envelope = _canonical_json(
        {
            "algorithm": exact_color.NATIVE_BUILDER_ALGORITHM_ID,
            "analyzer_source": evidence_document["analyzer_source"],
            "density_source": evidence_document["density_source"],
            "evidence_sha256": hashlib.sha256(evidence_payload).hexdigest(),
            "fixed_composition": {
                "lut_sha256": exact_color.FIXED_COMPOSITION_SHA256,
                "order": "F[B_c(i)]",
            },
            "identity": _identity_payload(evidence),
            "native_per_acquisition_builder": True,
            "pre_f_luts": [
                {"bytes": 131_072, "channel": channel, "sha256": digest} for channel, digest in zip(CHANNELS, pre_f_hashes, strict=True)
            ],
            "schema": exact_color.BUILDER_RECEIPT_SCHEMA,
            "scope": exact_color.NATIVE_BUILDER_SCOPE,
            "version": 1,
        }
    )
    return exact_color._issue_native_builder_receipt(
        payload=envelope,
        evidence_payload=evidence_payload,
        frame_ownership_receipt=evidence.frame_ownership_receipt,
        frame_ownership_receipt_sha256=evidence.frame_ownership_receipt_sha256,
        density_evidence_receipt=evidence.density_evidence_receipt,
        density_evidence_receipt_sha256=evidence.density_evidence_receipt_sha256,
        analyzer_rgb=analyzer_bytes,
        analyzer_rgb_sha256=evidence.analyzer_rgb_sha256,
        analyzer_shape=ANALYZER_SHAPE,
        pre_f_luts=luts,
        session_id=evidence.session_id,
        reservation_id=evidence.reservation_id,
        batch_session_id=evidence.batch_session_id,
        preview_sha256=evidence.preview_sha256,
        preview_identity_sha256=evidence.preview_identity_sha256,
        capture_attempt_id=evidence.capture_attempt_id,
        scan_identity=evidence.scan_identity,
        slot=evidence.slot,
    )


def rebuild_retained_native_builder_receipt(
    *,
    envelope_payload: bytes,
    evidence_payload: bytes,
    frame_ownership_receipt: bytes,
    density_evidence_receipt: bytes,
    analyzer_rgb: bytes,
    pre_f_luts: tuple[bytes, bytes, bytes],
) -> exact_color.NativeValidatedBuilderReceipt:
    """Re-derive a retained snapshot instead of trusting its stored LUTs.

    A writable evidence directory is not an attestation boundary. The raw
    analyzer and canonical acquisition receipts therefore reconstruct the
    original ``NativeBuilderEvidence`` and run the pinned builder again. Only
    byte-identical envelope, evidence, analyzer, and pre-F artifacts can be
    promoted back into a trusted receipt.
    """

    try:
        document = _parse_canonical_object(
            evidence_payload,
            label="retained native builder evidence",
        )
        ownership = _parse_canonical_object(
            frame_ownership_receipt,
            label="retained frame ownership",
        )
        _parse_canonical_object(
            density_evidence_receipt,
            label="retained density evidence",
        )
        identity = document["identity"]
        density = document["density_source"]
        analyzer = document["analyzer_source"]
        calibration = document["calibration"]
        if not all(type(value) is dict for value in (identity, density, analyzer, calibration)):
            raise TypeError("retained native builder sections are malformed")
        identity_row = cast(dict[str, object], identity)
        density_row = cast(dict[str, object], density)
        analyzer_row = cast(dict[str, object], analyzer)
        calibration_row = cast(dict[str, object], calibration)
        if document.get("frame_ownership") != ownership:
            raise ValueError("retained native builder evidence disagrees with frame ownership")
        analyzer_array = np.frombuffer(analyzer_rgb, dtype="<u2").reshape(ANALYZER_SHAPE)
        rebuilt_evidence = NativeBuilderEvidence(
            session_id=identity_row["session_id"],
            capture_attempt_id=identity_row["capture_attempt_id"],
            scan_identity=identity_row["scan_identity"],
            slot=identity_row["slot"],
            density_source_wire_sha256=density_row["wire_sha256"],
            density_source_child_sha256=density_row["child_buffer_sha256"],
            calibration_numerators=tuple(calibration_row["read8c_numerators_rgb"]),
            density_f03_denominators=tuple(density_row["f03_denominators_raw_10ns_rgb"]),
            densities=tuple(float.fromhex(value) for value in density_row["densities_binary64_hex_rgb"]),
            density_arithmetic=density_row["arithmetic"],
            frame_ownership_status=ownership["binding_status"],
            frame_ownership_receipt=frame_ownership_receipt,
            frame_ownership_receipt_sha256=hashlib.sha256(frame_ownership_receipt).hexdigest(),
            density_evidence_receipt=density_evidence_receipt,
            density_evidence_receipt_sha256=hashlib.sha256(density_evidence_receipt).hexdigest(),
            reservation_id=identity_row["reservation_id"],
            batch_session_id=identity_row["batch_session_id"],
            preview_sha256=identity_row["preview_sha256"],
            preview_identity_sha256=identity_row["preview_identity_sha256"],
            transport_table_sha256=ownership["transport_table_sha256"],
            transport_identity_sha256=ownership["transport_identity_sha256"],
            reviewed_fingerprint_sha256=ownership["reviewed_fingerprint_sha256"],
            fresh_fingerprint_sha256=ownership["fresh_fingerprint_sha256"],
            frame_index=ownership["frame_index"],
            frame_total=ownership["frame_total"],
            selected_slots=tuple(ownership["selected_slots"]),
            analyzer_rgb=analyzer_array,
            analyzer_rgb_sha256=analyzer_row["rgb_sha256"],
            analyzer_resolution_dpi=analyzer_row["resolution_dpi"],
            analyzer_rectangle=tuple(analyzer_row["rectangle_inclusive"]),
            final_f02_denominators=tuple(analyzer_row["final_f02_denominators_raw_10ns_rgb"]),
        )
        rebuilt = build_native_builder_receipt(rebuilt_evidence)
    except exact_color.ExactColorIntegrityError:
        raise
    except (exact_color.ExactColorUnavailable, KeyError, TypeError, ValueError) as error:
        raise exact_color.ExactColorIntegrityError(f"retained native builder evidence cannot be freshly derived: {error}") from error

    if (
        rebuilt.payload != envelope_payload
        or rebuilt.evidence_payload != evidence_payload
        or rebuilt.frame_ownership_receipt != frame_ownership_receipt
        or rebuilt.density_evidence_receipt != density_evidence_receipt
        or rebuilt.analyzer_rgb != analyzer_rgb
        or rebuilt.pre_f_luts != pre_f_luts
    ):
        raise exact_color.ExactColorIntegrityError("retained native builder artifacts disagree with fresh native derivation")
    return rebuilt


def _validate_evidence(evidence: NativeBuilderEvidence) -> tuple[np.ndarray, bytes]:
    if type(evidence) is not NativeBuilderEvidence:
        raise exact_color.ExactColorUnavailable("native builder evidence has an invalid type")
    for label, value in (
        ("session_id", evidence.session_id),
        ("capture_attempt_id", evidence.capture_attempt_id),
        ("scan_identity", evidence.scan_identity),
    ):
        if type(value) is not str or not value:
            raise exact_color.ExactColorUnavailable(f"native builder {label} is missing")
    if type(evidence.slot) is not int or not 1 <= evidence.slot <= 40:
        raise exact_color.ExactColorUnavailable("native builder slot must be an integer in 1..40")
    for label, digest in (
        ("density source wire", evidence.density_source_wire_sha256),
        ("density source child", evidence.density_source_child_sha256),
        ("frame ownership", evidence.frame_ownership_receipt_sha256),
        ("density evidence", evidence.density_evidence_receipt_sha256),
        ("preview", evidence.preview_sha256),
        ("preview identity", evidence.preview_identity_sha256),
        ("transport table", evidence.transport_table_sha256),
        ("transport identity", evidence.transport_identity_sha256),
        ("reviewed fingerprint", evidence.reviewed_fingerprint_sha256),
        ("fresh fingerprint", evidence.fresh_fingerprint_sha256),
        ("analyzer RGB", evidence.analyzer_rgb_sha256),
    ):
        if type(digest) is not str or not _SHA256.fullmatch(digest):
            raise exact_color.ExactColorUnavailable(f"native builder {label} SHA-256 is malformed")
    if evidence.frame_ownership_status != FRAME_OWNERSHIP_STATUS:
        raise exact_color.ExactColorUnavailable("97-dpi density evidence is not proven to belong to this frame")
    ownership = _parse_canonical_object(evidence.frame_ownership_receipt, label="frame ownership")
    density_receipt = _parse_canonical_object(evidence.density_evidence_receipt, label="density evidence")
    if hashlib.sha256(evidence.frame_ownership_receipt).hexdigest() != evidence.frame_ownership_receipt_sha256:
        raise exact_color.ExactColorUnavailable("frame ownership receipt does not match its SHA-256")
    if hashlib.sha256(evidence.density_evidence_receipt).hexdigest() != evidence.density_evidence_receipt_sha256:
        raise exact_color.ExactColorUnavailable("density evidence receipt does not match its SHA-256")
    _validate_ownership_binding(evidence, ownership, density_receipt)
    if evidence.density_arithmetic != DENSITY_ARITHMETIC:
        raise exact_color.ExactColorUnavailable("density arithmetic is not x87 bit-certified")
    for label, values in (
        ("calibration numerators", evidence.calibration_numerators),
        ("density f03 denominators", evidence.density_f03_denominators),
        ("final f02 denominators", evidence.final_f02_denominators),
    ):
        if type(values) is not tuple or len(values) != 3 or any(type(value) is not int or not 1 <= value <= 0xFFFFFFFF for value in values):
            raise exact_color.ExactColorUnavailable(f"native builder {label} must contain three nonzero uint32 values")
    if evidence.density_f03_denominators == evidence.final_f02_denominators:
        raise exact_color.ExactColorUnavailable("density f03 and final f02 exposure triplets were conflated")
    if (
        type(evidence.densities) is not tuple
        or len(evidence.densities) != 3
        or any(type(value) is not float or not math.isfinite(value) for value in evidence.densities)
    ):
        raise exact_color.ExactColorUnavailable("native builder densities must contain three finite doubles")
    if evidence.analyzer_resolution_dpi != ANALYZER_RESOLUTION_DPI:
        raise exact_color.ExactColorUnavailable("native builder analyzer source must be the 285-dpi pass")
    analyzer = np.asarray(evidence.analyzer_rgb)
    if analyzer.dtype != np.uint16 or analyzer.shape != ANALYZER_SHAPE:
        raise exact_color.ExactColorUnavailable(f"native builder analyzer RGB must be uint16 with shape {ANALYZER_SHAPE}")
    snapshot = np.array(analyzer, dtype="<u2", order="C", copy=True)
    analyzer_bytes = snapshot.tobytes(order="C")
    if hashlib.sha256(analyzer_bytes).hexdigest() != evidence.analyzer_rgb_sha256:
        raise exact_color.ExactColorUnavailable("native builder analyzer RGB does not match its SHA-256")
    rectangle = evidence.analyzer_rectangle
    if type(rectangle) is not tuple or len(rectangle) != 4 or any(type(value) is not int for value in rectangle):
        raise exact_color.ExactColorUnavailable("native builder analyzer rectangle is malformed")
    r0, c0, r1, c1 = rectangle
    if not (0 <= r0 <= r1 < ANALYZER_SHAPE[0] and 0 <= c0 <= c1 < ANALYZER_SHAPE[1]):
        raise exact_color.ExactColorUnavailable("native builder analyzer rectangle is out of bounds")
    snapshot.setflags(write=False)
    return snapshot, analyzer_bytes


def _identity_payload(evidence: NativeBuilderEvidence) -> dict[str, object]:
    return {
        "batch_session_id": evidence.batch_session_id,
        "capture_attempt_id": evidence.capture_attempt_id,
        "preview_identity_sha256": evidence.preview_identity_sha256,
        "preview_sha256": evidence.preview_sha256,
        "reservation_id": evidence.reservation_id,
        "scan_identity": evidence.scan_identity,
        "session_id": evidence.session_id,
        "slot": evidence.slot,
    }


def _evidence_payload(evidence: NativeBuilderEvidence, analyzer: np.ndarray) -> dict[str, object]:
    return {
        "algorithm": {
            "curve": CURVE_ALGORITHM_ID,
            "parameter_derivation": PARAMETER_ALGORITHM_ID,
        },
        "analyzer_source": {
            "final_f02_denominators_raw_10ns_rgb": list(evidence.final_f02_denominators),
            "geometry": list(analyzer.shape),
            "rectangle_inclusive": list(evidence.analyzer_rectangle),
            "resolution_dpi": evidence.analyzer_resolution_dpi,
            "rgb_bytes": len(analyzer.tobytes()),
            "rgb_sha256": evidence.analyzer_rgb_sha256,
        },
        "calibration": {"read8c_numerators_rgb": list(evidence.calibration_numerators)},
        "density_source": {
            "arithmetic": evidence.density_arithmetic,
            "child_buffer_sha256": evidence.density_source_child_sha256,
            "densities_binary64_hex_rgb": [float(value).hex() for value in evidence.densities],
            "f03_denominators_raw_10ns_rgb": list(evidence.density_f03_denominators),
            "resolution_dpi": DENSITY_SOURCE_RESOLUTION_DPI,
            "wire_sha256": evidence.density_source_wire_sha256,
        },
        "density_evidence": {
            "receipt_sha256": evidence.density_evidence_receipt_sha256,
            "scope": "reservation-preview",
        },
        "frame_ownership": json.loads(evidence.frame_ownership_receipt),
        "identity": _identity_payload(evidence),
        "resource": {"bytes": len(_RESOURCE), "sha256": RESOURCE_SHA256},
        "schema": "negpy.native-stage1-builder-evidence",
        "version": 1,
    }


def _canonical_json(value: dict[str, object]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _parse_canonical_object(payload: bytes, *, label: str) -> dict[str, object]:
    if type(payload) is not bytes or not payload:
        raise exact_color.ExactColorUnavailable(f"native builder {label} receipt is missing")
    try:
        parsed = json.loads(payload)
    except (UnicodeDecodeError, ValueError, RecursionError) as error:
        raise exact_color.ExactColorUnavailable(f"native builder {label} receipt is invalid JSON") from error
    if type(parsed) is not dict or _canonical_json(parsed) != payload:
        raise exact_color.ExactColorUnavailable(f"native builder {label} receipt is not canonical")
    return parsed


def _validate_ownership_binding(
    evidence: NativeBuilderEvidence,
    ownership: dict[str, object],
    density_receipt: dict[str, object],
) -> None:
    expected = {
        "schema_version": 1,
        "scope": "reservation-preview-frame",
        "binding_status": FRAME_OWNERSHIP_STATUS,
        "session_reservation_retained": True,
        "reservation_id": evidence.reservation_id,
        "batch_session_id": evidence.batch_session_id,
        "preview_sha256": evidence.preview_sha256,
        "preview_identity_sha256": evidence.preview_identity_sha256,
        "transport_table_sha256": evidence.transport_table_sha256,
        "transport_identity_sha256": evidence.transport_identity_sha256,
        "reviewed_fingerprint_sha256": evidence.reviewed_fingerprint_sha256,
        "fresh_fingerprint_sha256": evidence.fresh_fingerprint_sha256,
        "frame_capture_attempt_id": evidence.capture_attempt_id,
        "frame_index": evidence.frame_index,
        "frame_total": evidence.frame_total,
        "selected_slots": list(evidence.selected_slots),
        "selected_slot": evidence.slot,
    }
    if ownership != expected:
        raise exact_color.ExactColorUnavailable("frame ownership receipt does not bind the native builder acquisition")
    transport_material: dict[str, object] = {
        "reservation_id": evidence.reservation_id,
        "batch_session_id": evidence.batch_session_id,
        "preview_sha256": evidence.preview_sha256,
        "preview_identity_sha256": evidence.preview_identity_sha256,
        "transport_table_sha256": evidence.transport_table_sha256,
        "reviewed_fingerprint_sha256": evidence.reviewed_fingerprint_sha256,
        "fresh_fingerprint_sha256": evidence.fresh_fingerprint_sha256,
        "selected_slots": list(evidence.selected_slots),
    }
    if hashlib.sha256(_canonical_json(transport_material)).hexdigest() != evidence.transport_identity_sha256:
        raise exact_color.ExactColorUnavailable("frame ownership transport identity digest is invalid")
    if (
        type(evidence.reservation_id) is not str
        or not evidence.reservation_id
        or type(evidence.batch_session_id) is not str
        or not evidence.batch_session_id
        or evidence.session_id != evidence.reservation_id
        or evidence.reservation_id != evidence.batch_session_id
        or type(evidence.frame_index) is not int
        or type(evidence.frame_total) is not int
        or not 1 <= evidence.frame_index <= evidence.frame_total
        or type(evidence.selected_slots) is not tuple
        or len(evidence.selected_slots) != evidence.frame_total
        or evidence.slot not in evidence.selected_slots
        or evidence.selected_slots[evidence.frame_index - 1] != evidence.slot
        or any(type(slot) is not int or not 1 <= slot <= 40 for slot in evidence.selected_slots)
        or len(set(evidence.selected_slots)) != len(evidence.selected_slots)
    ):
        raise exact_color.ExactColorUnavailable("frame ownership batch identity is malformed")
    _validate_density_receipt_binding(evidence, density_receipt)


def _validate_density_receipt_binding(
    evidence: NativeBuilderEvidence,
    density_receipt: dict[str, object],
) -> None:
    if set(density_receipt) != {
        "schema_version",
        "scope",
        "per_frame_binding_status",
        "preview_identity_sha256",
        "source_payload_bytes",
        "calibration_binding",
        "source_binding",
        "exposure_binding",
        "result",
    }:
        raise exact_color.ExactColorUnavailable("density evidence receipt fields are incomplete")
    calibration_binding = density_receipt.get("calibration_binding")
    source_binding = density_receipt.get("source_binding")
    exposure_binding = density_receipt.get("exposure_binding")
    result = density_receipt.get("result")
    if not all(type(value) is dict for value in (calibration_binding, source_binding, exposure_binding, result)):
        raise exact_color.ExactColorUnavailable("density evidence bindings are malformed")
    calibration_binding = cast(dict[str, object], calibration_binding)
    source_binding = cast(dict[str, object], source_binding)
    exposure_binding = cast(dict[str, object], exposure_binding)
    result = cast(dict[str, object], result)
    calibration = calibration_binding.get("calibration")
    if type(calibration) is not dict:
        raise exact_color.ExactColorUnavailable("density calibration receipt is malformed")
    calibration = cast(dict[str, object], calibration)
    binding_identities = tuple(
        (binding.get("session_id"), binding.get("capture_attempt_id"), binding.get("scan_identity"))
        for binding in (source_binding, exposure_binding, result)
    )
    calibration_identity = (
        calibration.get("session_id"),
        calibration_binding.get("capture_attempt_id"),
        calibration_binding.get("scan_identity"),
    )
    density_hex = [struct.pack(">d", value).hex() for value in evidence.densities]
    if (
        density_receipt.get("schema_version") != 1
        or density_receipt.get("scope") != "reservation-preview"
        or density_receipt.get("per_frame_binding_status") != DENSITY_PER_FRAME_BINDING_STATUS
        or density_receipt.get("preview_identity_sha256") != evidence.preview_identity_sha256
        or density_receipt.get("source_payload_bytes")
        not in SUPPORTED_DENSITY_SOURCE_WIRE_BYTES
        or any(identity != calibration_identity for identity in binding_identities)
        or calibration_identity[0] != evidence.reservation_id
        or calibration_identity[2] != evidence.scan_identity
        or source_binding.get("resolution_dpi") != DENSITY_SOURCE_RESOLUTION_DPI
        or source_binding.get("wire_sha256") != evidence.preview_sha256
        or source_binding.get("wire_sha256") != evidence.density_source_wire_sha256
        or source_binding.get("child_buffer_sha256") != evidence.density_source_child_sha256
        or calibration.get("numerators_rgb") != list(evidence.calibration_numerators)
        or exposure_binding.get("density_f03_exposures_raw_10ns_rgb") != list(evidence.density_f03_denominators)
        or result.get("algorithm_id") != evidence.density_arithmetic
        or result.get("promotable") is not True
        or result.get("source_wire_sha256") != evidence.density_source_wire_sha256
        or result.get("source_child_buffer_sha256") != evidence.density_source_child_sha256
        or result.get("numerators_rgb") != list(evidence.calibration_numerators)
        or result.get("density_f03_denominators_raw_10ns_rgb") != list(evidence.density_f03_denominators)
        or result.get("densities_rgb") != list(evidence.densities)
        or result.get("density_binary64_be_hex_rgb") != density_hex
    ):
        raise exact_color.ExactColorUnavailable("density evidence belongs to a different preview or reservation")


def _curve(xt: np.ndarray, yt: np.ndarray, value: np.ndarray) -> np.ndarray:
    index = np.full(value.shape, 32, dtype=np.int64)
    for candidate in range(31, 0, -1):
        index = np.where(value < xt[candidate], candidate, index)
    hit = index < 32
    bounded = np.clip(index, 1, 31)
    x1, x0, y1, y0 = xt[bounded], xt[bounded - 1], yt[bounded], yt[bounded - 1]
    denominator = x1 - x0
    interpolated = np.where(
        denominator != 0,
        y0 + (value - x0) * (y1 - y0) / np.where(denominator == 0, 1, denominator),
        y0,
    )
    return np.where(hit, interpolated, yt[31])


def _histogram(values: np.ndarray) -> np.ndarray:
    return (np.bincount(values.astype(np.int64), minlength=65_536) % 65_536).astype(np.int64)


def _lower_cutoff(histogram: np.ndarray, population: int) -> float:
    threshold = math.trunc(0.0042 * population)
    hit = np.flatnonzero(np.cumsum(histogram) >= threshold)
    return float(hit[0]) if hit.size else 0.0


def _maximum(histogram: np.ndarray) -> float:
    hit = np.flatnonzero(histogram)
    return float(hit[-1]) if hit.size else 0.0


def _analyse(pixels: np.ndarray, exposure: tuple[float, float, float], densities: tuple[float, float, float]) -> _AnalyzerOutputs:
    count = pixels.shape[0]
    scaled = pixels.astype(np.float64) * np.asarray(exposure, dtype=np.float64)
    density = np.asarray(densities, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio01 = scaled[:, 0] / (10.0 ** (density[1] - density[0]) * scaled[:, 1])
        ratio02 = scaled[:, 0] / (10.0 ** (density[2] - density[0]) * scaled[:, 2])
        ratio12 = scaled[:, 1] / (10.0 ** (density[2] - density[1]) * scaled[:, 2])

    def good(ratio: np.ndarray) -> np.ndarray:
        return (ratio > 1.0 / 3.0) & (ratio < 3.0)

    good01, good02, good12 = good(ratio01), good(ratio02), good(ratio12)
    passes = (good01 | good02, good01 | good12, good02 | good12)
    all_lower: list[float] = []
    all_max: list[float] = []
    gated_lower: list[float] = []
    gated_max: list[float] = []
    for channel in range(3):
        values = pixels[:, channel]
        histogram = _histogram(values)
        all_lower.append(_lower_cutoff(histogram, count))
        all_max.append(_maximum(histogram))
        keep = passes[channel].copy()
        failing = np.flatnonzero(~keep)
        if failing.size:
            budget = math.ceil(0.9 * count)
            if failing.size > budget:
                keep[failing[budget:]] = True
        gated = values[keep]
        histogram = _histogram(gated)
        gated_lower.append(_lower_cutoff(histogram, gated.size))
        gated_max.append(_maximum(histogram))
    return _AnalyzerOutputs(
        (all_lower[0], all_lower[1], all_lower[2]),
        (all_max[0], all_max[1], all_max[2]),
        (gated_lower[0], gated_lower[1], gated_lower[2]),
        (gated_max[0], gated_max[1], gated_max[2]),
    )


def _scale(outputs: _AnalyzerOutputs, exposure: tuple[float, float, float]) -> tuple[tuple[float, ...], ...]:
    zlo: list[float] = []
    zhi: list[float] = []
    Zlo: list[float] = []
    Zhi: list[float] = []
    for channel in range(3):
        all_lo = max(1.0, outputs.all_lower[channel] * exposure[channel])
        all_hi = max(1.0, outputs.all_max[channel] * exposure[channel])
        gate_lo = outputs.gated_lower[channel] * exposure[channel]
        gate_hi = outputs.gated_max[channel] * exposure[channel]
        zlo.append(math.log10(65_535.0 / gate_lo))
        zhi.append(math.log10(65_535.0 / gate_hi))
        Zlo.append(math.log10(65_535.0 / all_lo))
        Zhi.append(math.log10(65_535.0 / all_hi))
    return tuple(zlo), tuple(zhi), tuple(Zlo), tuple(Zhi)


def _derive_parameters(
    pixels: np.ndarray,
    exposure: tuple[float, float, float],
    densities: tuple[float, float, float],
) -> tuple[list[_BuilderArgs], list[tuple[np.ndarray, np.ndarray]]]:
    outputs = _analyse(pixels, exposure, densities)
    zlo, zhi, Zlo, Zhi = _scale(outputs, exposure)
    D = tuple(min(densities[channel], zhi[channel]) for channel in range(3))
    X = [table + (D[channel] - table[0]) for channel, table in enumerate(_RESOURCE_TABLES[1:])]
    master = _RESOURCE_TABLES[0]

    def first(channel: int, value: float) -> float:
        return float(_curve(X[channel], master, np.array([value], dtype=np.float64))[0])

    reference = [first(channel, D[channel] + 0.15) for channel in range(3)]
    Lo = [first(channel, zlo[channel]) for channel in range(3)]
    Hi = [first(channel, zhi[channel]) for channel in range(3)]
    lo = [Lo[channel] - reference[channel] for channel in range(3)]
    hi = [Hi[channel] - reference[channel] for channel in range(3)]
    wh0 = 10.0 ** (-abs(hi[0] - hi[1]))
    wh2 = 10.0 ** (-abs(hi[2] - hi[1]))
    ws0 = 3.0 ** (-abs((lo[0] - hi[0]) - (lo[1] - hi[1])))
    ws2 = 3.0 ** (-abs((lo[2] - hi[2]) - (lo[1] - hi[1])))
    tone = min(1.0, 10.0 ** (max(lo) - 0.8))
    p = tone * ((1.0 - wh0) + wh0 * ws0)
    q = tone * ((1.0 - wh2) + wh2 * ws2)
    offset0 = p * (Lo[0] - Lo[1]) + wh0 * (1.0 - p) * (Hi[0] - Hi[1])
    offset1 = q * (Lo[1] - Lo[2]) + wh2 * (1.0 - q) * (Hi[1] - Hi[2])
    Y2 = [master - offset0, master.copy(), master + offset1]

    def second(channel: int, value: float) -> float:
        return float(_curve(X[channel], Y2[channel], np.array([value], dtype=np.float64))[0])

    reference2 = [second(channel, D[channel] + 0.15) for channel in range(3)]
    vlo = [second(channel, zlo[channel]) - reference2[channel] for channel in range(3)]
    vhi = [second(channel, zhi[channel]) - reference2[channel] for channel in range(3)]
    E0 = min(vhi)
    a = [min(1.0, 10.0 ** (-(vhi[channel] + 0.23) * 4.0)) for channel in range(3)]
    m = max(a)
    L = max(vlo)
    A0 = max(second(channel, min(Zlo[channel], X[channel][31])) for channel in range(3))
    C = [second(channel, max(Zhi[channel], X[channel][0])) for channel in range(3)]
    B0 = min(C)
    difference0 = A0 - B0
    if difference0 >= 0.6:
        B = B0
        A1 = A0 + 0.05
    else:
        delta = 0.6 - difference0
        B = B0 - (1.0 - m) * delta
        A1 = max(B + 0.6, A0 + 0.05)
    difference = A1 - B
    te = min(1.9, 1.9 * 0.4 ** (E0 + 0.23))
    E = min(1.9, max(difference, te))
    A = A1 + (difference / E) * (1.2 - L) if L < 1.2 else A1
    f = [float(_curve(Y2[channel], X[channel], np.array([A], dtype=np.float64))[0]) for channel in range(3)]
    g = [float(_curve(Y2[channel], X[channel], np.array([B], dtype=np.float64))[0]) for channel in range(3)]
    args = [_BuilderArgs(A, B, E, C[channel], a[channel], f[channel], g[channel]) for channel in range(3)]
    return args, list(zip(X, Y2, strict=True))


def _build_lut(arg: _BuilderArgs, xt: np.ndarray, yt: np.ndarray) -> np.ndarray:
    slope = float(min(max((arg.A - ((1.0 - arg.a) * arg.B + arg.a * arg.C)) / arg.E, 0.01), 4.0))
    clip = max(0, min(int(10.0 ** (-(arg.f - arg.g)) * 65_535.0), 65_535))
    lut = np.zeros(65_536, dtype=np.uint16)
    lut[: clip + 1] = 0xFFFF
    if clip + 1 < 65_536:
        index = np.arange(clip + 1, 65_536, dtype=np.float64)
        tone = _curve(xt, yt, np.log10(65_535.0 / index) + arg.g)
        output = 65_535.0 * np.power(10.0, (arg.A - tone) * (-1.0 / slope))
        lut[clip + 1 :] = (output.astype(np.int64) & 0xFFFF).astype(np.uint16)
    return lut


__all__ = [
    "ANALYZER_RESOLUTION_DPI",
    "DENSITY_SOURCE_RESOLUTION_DPI",
    "NativeBuilderEvidence",
    "adapt_native_builder_evidence",
    "build_native_builder_receipt",
    "rebuild_retained_native_builder_receipt",
]
