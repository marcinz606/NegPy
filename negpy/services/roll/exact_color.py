"""Fail-closed boundary for portable Nikon exact-color evaluation.

This module deliberately contains no builder or CML4 math. It is the narrow
seam between NegPy's repaired uint16 RGB buffer and two separately validated
operations:

* a Stage-3 evidence/replay bridge or a fresh, identity-bound native builder
  receipt using three validated pre-F LUTs and the pinned LS5000.md3 post-F
  composition; and
* a verified portable CMS evaluator, injected by the caller. NegPy ships one
  adapter for the captured CML4 Stage-1/Stage-2 transform.

Receipts are immutable bytes rather than mutable dictionaries. NegPy owns the
aggregate builder envelope; the embedded Stage-3 validation and CMS payloads
retain their producing components' schemas. Every payload has an exact
content hash and explicit attestation. Results are bound to repaired input,
computed Stage-1 input, and final output before any TIFF is written.

The native receipt exists only when distinct 97-dpi density, 285-dpi analyzer,
and final-exposure evidence are explicitly bound to the same frame.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import re
import stat
import struct
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final, Mapping, Protocol, cast

import numpy as np


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MAX_RECEIPT_BYTES = 1024 * 1024
_BUILDER_CHANNELS = ("r", "g", "b")
_PRE_F_LUT_BYTES = 65_536 * 2
# Keep this in lockstep with native_builder: both producer-side and serialized
# receipt validation accept only the two proven LS-5000 97-dpi geometries.
_SUPPORTED_DENSITY_SOURCE_WIRE_BYTES = frozenset((6_250_496, 5_804_032))
BUILDER_RECEIPT_SCHEMA = "negpy.validated-stage1-builder"
STAGE3_REPORT_SCHEMA = "nikonre.ls5000_stage3_validation"
FIXED_COMPOSITION_SHA256 = "8729cae5a7aa551ae35926b80d097d73d92ecb6bde471130d6344c6c10ecbe7a"
STAGE3_REPLAY_SCOPE = "stage3-captured-pref-evidence-replay-bridge"
NATIVE_BUILDER_SCOPE = "ls5000-native-per-acquisition-stage1-builder"
NATIVE_BUILDER_ALGORITHM_ID = "ls5000-md3-prescan-to-pref-v1"
NATIVE_RESOURCE_SHA256 = "cd934185df496f071d307ba4f96a2a2b6ac31c3c85efc62a7fa1e3216fdba70c"

CMS_RECEIPT_KIND: Final = "negpy.portable-cms-on-receipt"
CMS_RECEIPT_VERSION: Final = 1
CMS_ALGORITHM_ID: Final = "cml4-captured-optimized-stage1-stage2-v1"
CMS_SCOPE: Final = "captured-cml4-stage1-stage2-only"
CMS_ORACLE_SOURCE_SHA256: Final = "2a9bad4b89cefb9fcb2bebbc59009ea0248e1ea93897cfa99c2c320c7f675490"
CMS_VALIDATION_RECEIPT_SHA256: Final = "edf6f3f89158810f1de4ce3b4ff8938326bc50e1b3035af59af472258e7d95e8"
CMS_ASSET_SHA256: Mapping[str, str] = MappingProxyType(
    {
        "lch-atan-u16le.bin": "56b8ac82456941a0a8aad6d7de2c79b21785529037b504582731ce6abcd143b1",
        "lch-sincos-i16le.bin": "dc8e71681bc46e60e33448171865fc96770dc17d3cfd0db835f6173e6fed7a35",
        "lch-reciprocal-u16le.bin": "6959ef7deeb57dc96eb4653fe13b4d37fcb4250c28be15654b8e8dd236849042",
        "cml4-stage1-clut0.bin": "a2abbc1e76dc037e6b364a58b483ddf01b9dde7c0e072f85885cb1f8c9dcbf1c",
        "cml4-stage1-input-lut0.bin": "1a487c024ceaf83018c8ab0e405e9c12b25a0500a2ae6e3465e20b29638729d7",
        "cml4-stage1-output-lut0.bin": "fe87ce159ec126597f9fb605b57cebec9b0264da1ca16d1725efe34db2e4fd2b",
        "cml4-stage2-clut0.bin": "d14b7c76091552bf03899327ca6a7c74c712a0423e429de8f8cc8203d5c98da3",
        "cml4-stage2-input-lut0.bin": "60fa510492f2adad2be9b00107b3d7ed7188a798f29748c3401560b70be3a248",
        "cml4-stage2-output-lut0.bin": "b88574377d1a0cf47fe8806335641db2ca35197817cd055a4cd55141708c8291",
    }
)

_STAGE3_MODULE_BYTES = 1_052_672
_STAGE3_MODULE_SHA256 = "45afd6fb61a9517ff95d1896dc4257779c319310c2e8bbe75f3b4f3dada920af"
_STAGE3_RESOURCE_BYTES = 1_024
_STAGE3_RESOURCE_SHA256 = "cd934185df496f071d307ba4f96a2a2b6ac31c3c85efc62a7fa1e3216fdba70c"
_STAGE3_RESOURCE_VA = "0x100ce578"
_STAGE3_FIXED_ARTIFACT_BYTES = {
    "analyzer-desc.bin": 96,
    "analyzer-pixels.bin": 281 * 425 * 3 * 2,
    "builder-args.bin": 204,
    **{f"builder-control-{axis}-{channel}.bin": 256 for channel in _BUILDER_CHANNELS for axis in ("x", "y")},
    **{f"builder-preF-{channel}.bin": _PRE_F_LUT_BYTES for channel in _BUILDER_CHANNELS},
}
_STAGE3_DYNAMIC_ARTIFACTS = {
    "callback-buffer.bin",
    "debugger-session.log",
    "stage3-capture-state.json",
}
_STAGE3_ARTIFACTS = frozenset(_STAGE3_FIXED_ARTIFACT_BYTES) | _STAGE3_DYNAMIC_ARTIFACTS
_STAGE3_CALLBACK_BYTES = frozenset({281 * 425 * 3 * 2, 281 * 425 * 4 * 2})
_TRUSTED_BUILDER_RECEIPT_TOKEN = object()
_TRUSTED_NATIVE_BUILDER_RECEIPT_TOKEN = object()
_TRUSTED_CMS_RECEIPT_TOKEN = object()


class ExactColorUnavailable(RuntimeError):
    """Exact Nikon color cannot be produced from the supplied evidence."""


class ExactColorIntegrityError(ExactColorUnavailable):
    """An evaluator result or receipt is malformed or not content-bound."""


class PositiveColorMode(StrEnum):
    """Receipt-visible Tier-3 color paths; exact never implies fallback."""

    NEGPY_APPROXIMATE = "negpy-approximate"
    NIKON_EXACT = "nikon-exact"


@dataclass(frozen=True)
class ValidatedBuilderReceipt:
    """File-validated Stage-3 replay evidence and exact builder artifacts.

    This is an evidence/replay bridge. It is not the future macOS-native,
    per-acquisition builder, and direct caller construction is never trusted.
    """

    payload: bytes
    sha256: str
    stage3_receipt: bytes
    stage3_receipt_sha256: str
    pre_f_luts: tuple[bytes, bytes, bytes]
    pre_f_lut_sha256: tuple[str, str, str]
    fixed_composition_sha256: str
    evidence_filenames: tuple[str, str, str, str]
    _factory_token: object = field(repr=False, compare=False)

    @property
    def attested(self) -> bool:
        return self._factory_token is _TRUSTED_BUILDER_RECEIPT_TOKEN


@dataclass(frozen=True)
class NativeValidatedBuilderReceipt:
    """Fresh pre-F LUTs derived from identity-bound native scan evidence."""

    payload: bytes
    sha256: str
    evidence_payload: bytes
    evidence_sha256: str
    frame_ownership_receipt: bytes
    frame_ownership_receipt_sha256: str
    density_evidence_receipt: bytes
    density_evidence_receipt_sha256: str
    analyzer_rgb: bytes
    analyzer_rgb_sha256: str
    analyzer_shape: tuple[int, int, int]
    pre_f_luts: tuple[bytes, bytes, bytes]
    pre_f_lut_sha256: tuple[str, str, str]
    fixed_composition_sha256: str
    session_id: str
    reservation_id: str
    batch_session_id: str
    preview_sha256: str
    preview_identity_sha256: str
    capture_attempt_id: str
    scan_identity: str
    slot: int
    _factory_token: object = field(repr=False, compare=False)

    @property
    def attested(self) -> bool:
        return self._factory_token is _TRUSTED_NATIVE_BUILDER_RECEIPT_TOKEN


BuilderReceipt = ValidatedBuilderReceipt | NativeValidatedBuilderReceipt


@dataclass(frozen=True)
class VerifiedBuilderApplicationReceipt:
    """Immutable receipt for applying validated builder artifacts to one frame."""

    payload: bytes
    sha256: str
    attested: bool


@dataclass(frozen=True)
class Stage1BuilderResult:
    """Computed CML Stage-1 input plus bindings to source and builder evidence."""

    rgb: np.ndarray
    source_rgb_sha256: str
    stage1_input_rgb_sha256: str
    builder_receipt: BuilderReceipt
    application_receipt: VerifiedBuilderApplicationReceipt


@dataclass(frozen=True)
class VerifiedCMSReceipt:
    """Opaque receipt issued only by the verified portable CMS adapter."""

    payload: bytes
    sha256: str
    _factory_token: object = field(repr=False, compare=False)

    @property
    def attested(self) -> bool:
        return self._factory_token is _TRUSTED_CMS_RECEIPT_TOKEN


@dataclass(frozen=True)
class ExactColorResult:
    """Final RGB plus receipts and hashes binding both sides of evaluation."""

    rgb: np.ndarray
    input_rgb_sha256: str
    output_rgb_sha256: str
    builder_receipt: BuilderReceipt
    cms_receipt: VerifiedCMSReceipt
    source_rgb_sha256: str | None = None
    builder_application_receipt: VerifiedBuilderApplicationReceipt | None = None


class VerifiedStage1Builder(Protocol):
    """Apply independently validated builder artifacts at the exact boundary."""

    def apply(
        self,
        rgb: np.ndarray,
        *,
        builder_receipt: BuilderReceipt,
    ) -> Stage1BuilderResult: ...


class VerifiedPortableCMSEvaluator(Protocol):
    """Injected adapter around a separately verified portable CMS engine."""

    def evaluate(
        self,
        rgb: np.ndarray,
        *,
        builder_receipt: BuilderReceipt,
    ) -> ExactColorResult: ...


def rgb16_content_sha256(rgb: np.ndarray) -> str:
    """Hash canonical RGB16 content, including its shape and byte order.

    Evaluator adapters should call this helper rather than reimplementing the
    content-identity grammar. Shape is included so equal byte strings with a
    different raster geometry cannot share an identity.
    """

    _validate_rgb16(rgb, label="RGB input")
    canonical = np.asarray(rgb, dtype="<u2", order="C")
    digest = hashlib.sha256()
    digest.update(b"negpy.rgb16.v1\0")
    digest.update(json.dumps(list(canonical.shape), separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


def load_stage3_replay_builder_receipt(
    stage3_report_path: str | os.PathLike[str],
) -> ValidatedBuilderReceipt:
    """Load one real Stage-3 PASS report and its three adjacent pre-F LUTs.

    The four files are read as one fail-closed snapshot using non-symlink
    descriptor reads. The report is then checked against the complete
    Stage-3 PASS/provenance/artifact contract before a trusted replay receipt
    can exist. Callers cannot replace this with an ``attested=True`` claim.
    """

    report_path = Path(stage3_report_path).expanduser().absolute()
    lut_paths = tuple(report_path.parent / f"builder-preF-{channel}.bin" for channel in _BUILDER_CHANNELS)
    report_blob, report_identity = _stable_read_non_symlink(
        report_path,
        label="Stage-3 validation report",
        max_bytes=_MAX_RECEIPT_BYTES,
    )
    lut_snapshots = tuple(
        _stable_read_non_symlink(
            path,
            label=f"Stage-3 pre-F {channel} LUT",
            expected_bytes=_PRE_F_LUT_BYTES,
        )
        for channel, path in zip(_BUILDER_CHANNELS, lut_paths, strict=True)
    )
    all_paths = (report_path, *lut_paths)
    all_identities = (report_identity, *(identity for _, identity in lut_snapshots))
    _assert_paths_unchanged(all_paths, all_identities)

    pre_f_luts = (
        lut_snapshots[0][0],
        lut_snapshots[1][0],
        lut_snapshots[2][0],
    )
    pre_f_hashes = (
        hashlib.sha256(pre_f_luts[0]).hexdigest(),
        hashlib.sha256(pre_f_luts[1]).hexdigest(),
        hashlib.sha256(pre_f_luts[2]).hexdigest(),
    )
    report = _parse_json_object(report_blob, label="Stage-3 validation")
    _validate_stage3_report(report, pre_f_hashes)
    report_hash = hashlib.sha256(report_blob).hexdigest()
    envelope = _canonical_json(
        {
            "fixed_composition": {
                "lut_sha256": FIXED_COMPOSITION_SHA256,
                "order": "F[B_c(i)]",
            },
            "native_per_acquisition_builder": False,
            "pre_f_luts": [
                {
                    "bytes": _PRE_F_LUT_BYTES,
                    "channel": channel,
                    "sha256": digest,
                }
                for channel, digest in zip(_BUILDER_CHANNELS, pre_f_hashes, strict=True)
            ],
            "schema": BUILDER_RECEIPT_SCHEMA,
            "scope": STAGE3_REPLAY_SCOPE,
            "stage3_receipt_sha256": report_hash,
            "version": 1,
        }
    )
    receipt = ValidatedBuilderReceipt(
        payload=envelope,
        sha256=hashlib.sha256(envelope).hexdigest(),
        stage3_receipt=report_blob,
        stage3_receipt_sha256=report_hash,
        pre_f_luts=pre_f_luts,
        pre_f_lut_sha256=pre_f_hashes,
        fixed_composition_sha256=FIXED_COMPOSITION_SHA256,
        evidence_filenames=(report_path.name, lut_paths[0].name, lut_paths[1].name, lut_paths[2].name),
        _factory_token=_TRUSTED_BUILDER_RECEIPT_TOKEN,
    )
    builder_receipt_payload(receipt)
    return receipt


def load_native_builder_receipt(
    native_receipt_path: str | os.PathLike[str],
) -> NativeValidatedBuilderReceipt:
    """Reload one retained native builder snapshot for an exact-color retry.

    Every acquisition-bound input is snapshotted through non-symlink file
    descriptors with strict byte bounds, then revalidated as one native
    receipt. Merely retaining three LUTs is insufficient: ownership, density,
    analyzer, builder envelope, and LUTs all have to remain mutually bound.
    """

    receipt_path = Path(native_receipt_path).expanduser().absolute()
    directory = receipt_path.parent
    evidence_path = directory / "native-builder-evidence.json"
    ownership_path = directory / "nikon-density-frame-ownership.json"
    density_path = directory / "nikon-density-evidence.json"
    analyzer_path = directory / "analyzer-rgb-u16le.bin"
    lut_paths = tuple(directory / f"builder-preF-{channel}.bin" for channel in _BUILDER_CHANNELS)
    paths = (
        receipt_path,
        evidence_path,
        ownership_path,
        density_path,
        analyzer_path,
        *lut_paths,
    )
    reads = (
        _stable_read_non_symlink(
            receipt_path,
            label="native builder receipt",
            max_bytes=_MAX_RECEIPT_BYTES,
        ),
        _stable_read_non_symlink(
            evidence_path,
            label="native builder evidence",
            max_bytes=_MAX_RECEIPT_BYTES,
        ),
        _stable_read_non_symlink(
            ownership_path,
            label="native frame-ownership receipt",
            max_bytes=_MAX_RECEIPT_BYTES,
        ),
        _stable_read_non_symlink(
            density_path,
            label="native density-evidence receipt",
            max_bytes=_MAX_RECEIPT_BYTES,
        ),
        _stable_read_non_symlink(
            analyzer_path,
            label="native analyzer RGB",
            expected_bytes=425 * 281 * 3 * 2,
        ),
        *(
            _stable_read_non_symlink(
                path,
                label=f"native pre-F {channel} LUT",
                expected_bytes=_PRE_F_LUT_BYTES,
            )
            for channel, path in zip(
                _BUILDER_CHANNELS,
                lut_paths,
                strict=True,
            )
        ),
    )
    _assert_paths_unchanged(
        paths,
        tuple(identity for _, identity in reads),
        label="native builder evidence",
    )
    (
        receipt_blob,
        evidence_blob,
        ownership_blob,
        density_blob,
        analyzer_blob,
        pre_f_r,
        pre_f_g,
        pre_f_b,
    ) = (payload for payload, _ in reads)

    envelope = _parse_json_object(receipt_blob, label="native builder receipt")
    if _canonical_json(envelope) != receipt_blob:
        raise ExactColorIntegrityError("native builder receipt is not canonical JSON")
    # Import lazily: native_builder owns the pinned derivation and imports this
    # boundary for its receipt factory. Re-running that derivation is what
    # prevents a writable evidence directory from self-attesting new LUTs.
    from negpy.services.roll import native_builder

    rebuilt = native_builder.rebuild_retained_native_builder_receipt(
        envelope_payload=receipt_blob,
        evidence_payload=evidence_blob,
        frame_ownership_receipt=ownership_blob,
        density_evidence_receipt=density_blob,
        analyzer_rgb=analyzer_blob,
        pre_f_luts=(pre_f_r, pre_f_g, pre_f_b),
    )
    if directory.name != rebuilt.sha256:
        raise ExactColorIntegrityError("native builder evidence is outside its content-addressed directory")
    return rebuilt


def _issue_verified_cms_receipt(payload: bytes) -> VerifiedCMSReceipt:
    """Issue a CMS receipt from the hash-verified production adapter only."""

    if type(payload) is not bytes or not payload or len(payload) > _MAX_RECEIPT_BYTES:
        raise ExactColorIntegrityError("CMS receipt payload is missing or unbounded")
    return VerifiedCMSReceipt(
        payload=payload,
        sha256=hashlib.sha256(payload).hexdigest(),
        _factory_token=_TRUSTED_CMS_RECEIPT_TOKEN,
    )


def _issue_native_builder_receipt(
    *,
    payload: bytes,
    evidence_payload: bytes,
    frame_ownership_receipt: bytes,
    frame_ownership_receipt_sha256: str,
    density_evidence_receipt: bytes,
    density_evidence_receipt_sha256: str,
    analyzer_rgb: bytes,
    analyzer_rgb_sha256: str,
    analyzer_shape: tuple[int, int, int],
    pre_f_luts: tuple[bytes, bytes, bytes],
    session_id: str,
    reservation_id: str,
    batch_session_id: str,
    preview_sha256: str,
    preview_identity_sha256: str,
    capture_attempt_id: str,
    scan_identity: str,
    slot: int,
) -> NativeValidatedBuilderReceipt:
    """Issue a native receipt after the package-local builder closes its gates."""

    receipt = NativeValidatedBuilderReceipt(
        payload=payload,
        sha256=hashlib.sha256(payload).hexdigest(),
        evidence_payload=evidence_payload,
        evidence_sha256=hashlib.sha256(evidence_payload).hexdigest(),
        frame_ownership_receipt=frame_ownership_receipt,
        frame_ownership_receipt_sha256=frame_ownership_receipt_sha256,
        density_evidence_receipt=density_evidence_receipt,
        density_evidence_receipt_sha256=density_evidence_receipt_sha256,
        analyzer_rgb=analyzer_rgb,
        analyzer_rgb_sha256=analyzer_rgb_sha256,
        analyzer_shape=analyzer_shape,
        pre_f_luts=pre_f_luts,
        pre_f_lut_sha256=(
            hashlib.sha256(pre_f_luts[0]).hexdigest(),
            hashlib.sha256(pre_f_luts[1]).hexdigest(),
            hashlib.sha256(pre_f_luts[2]).hexdigest(),
        ),
        fixed_composition_sha256=FIXED_COMPOSITION_SHA256,
        session_id=session_id,
        reservation_id=reservation_id,
        batch_session_id=batch_session_id,
        preview_sha256=preview_sha256,
        preview_identity_sha256=preview_identity_sha256,
        capture_attempt_id=capture_attempt_id,
        scan_identity=scan_identity,
        slot=slot,
        _factory_token=_TRUSTED_NATIVE_BUILDER_RECEIPT_TOKEN,
    )
    builder_receipt_payload(receipt)
    return receipt


def evaluate_exact_color(
    rgb: np.ndarray,
    *,
    builder_receipt: BuilderReceipt | None,
    builder: VerifiedStage1Builder | None,
    evaluator: VerifiedPortableCMSEvaluator | None,
) -> ExactColorResult:
    """Evaluate and independently validate one exact-color result.

    Nothing falls back to NegPy's approximate renderer here. Missing or
    invalid evidence raises ExactColorUnavailable and callers must expose
    that state rather than label any other output exact.
    """

    _validate_rgb16(rgb, label="exact-color input")
    if builder_receipt is None:
        raise ExactColorUnavailable("validated Stage-3 replay or native builder receipt is not supplied")
    if builder is None:
        raise ExactColorUnavailable("verified Stage-1 builder applicator is not supplied")
    if evaluator is None:
        raise ExactColorUnavailable("verified portable CMS evaluator is not supplied")
    builder_receipt_payload(builder_receipt)

    source_snapshot = np.array(rgb, dtype=np.uint16, order="C", copy=True)
    expected_source = rgb16_content_sha256(source_snapshot)
    source_snapshot.setflags(write=False)
    try:
        builder_result = builder.apply(source_snapshot, builder_receipt=builder_receipt)
    except ExactColorUnavailable:
        raise
    except Exception as error:
        raise ExactColorUnavailable(f"Stage-1 builder applicator failed: {error}") from error
    if not isinstance(builder_result, Stage1BuilderResult):
        raise ExactColorIntegrityError("Stage-1 builder returned an invalid result type")
    if builder_result.builder_receipt != builder_receipt:
        raise ExactColorIntegrityError("Stage-1 builder substituted a different builder receipt")
    builder_receipt_payload(builder_result.builder_receipt)
    application_payload = _receipt_payload(
        builder_result.application_receipt,
        VerifiedBuilderApplicationReceipt,
        label="builder application",
    )
    if rgb16_content_sha256(source_snapshot) != expected_source:
        raise ExactColorIntegrityError("Stage-1 builder mutated its repaired RGB input")
    if builder_result.source_rgb_sha256 != expected_source:
        raise ExactColorIntegrityError("builder source hash does not match the repaired RGB content")
    _validate_rgb16(builder_result.rgb, label="Stage-1 builder output")
    if builder_result.rgb.shape != source_snapshot.shape:
        raise ExactColorIntegrityError("Stage-1 builder output geometry differs from its input")
    stage1_snapshot = np.array(builder_result.rgb, dtype=np.uint16, order="C", copy=True)
    expected_stage1 = rgb16_content_sha256(stage1_snapshot)
    if builder_result.stage1_input_rgb_sha256 != expected_stage1:
        raise ExactColorIntegrityError("builder output hash does not match the Stage-1 input content")
    if application_payload.get("builder_receipt_sha256") != builder_receipt.sha256:
        raise ExactColorIntegrityError("builder application receipt does not bind the builder receipt")
    if application_payload.get("source_rgb_sha256") != expected_source:
        raise ExactColorIntegrityError("builder application receipt does not bind its repaired RGB input")
    if application_payload.get("stage1_input_rgb_sha256") != expected_stage1:
        raise ExactColorIntegrityError("builder application receipt does not bind its Stage-1 output")
    _validate_builder_application_receipt(application_payload, builder_receipt)

    stage1_snapshot.setflags(write=False)
    try:
        result = evaluator.evaluate(stage1_snapshot, builder_receipt=builder_receipt)
    except ExactColorUnavailable:
        raise
    except Exception as error:
        raise ExactColorUnavailable(f"portable CMS evaluator failed: {error}") from error
    if not isinstance(result, ExactColorResult):
        raise ExactColorIntegrityError("portable CMS evaluator returned an invalid result type")
    if result.builder_receipt != builder_receipt:
        raise ExactColorIntegrityError("evaluator result substituted a different builder receipt")
    builder_receipt_payload(result.builder_receipt)
    cms_payload = _receipt_payload(result.cms_receipt, VerifiedCMSReceipt, label="CMS")

    if rgb16_content_sha256(stage1_snapshot) != expected_stage1:
        raise ExactColorIntegrityError("portable CMS evaluator mutated its Stage-1 input")
    if result.input_rgb_sha256 != expected_stage1:
        raise ExactColorIntegrityError("evaluator input hash does not match the Stage-1 input content")
    _validate_rgb16(result.rgb, label="exact-color output")
    if result.rgb.shape != stage1_snapshot.shape:
        raise ExactColorIntegrityError("exact-color output geometry differs from its input")
    # Snapshot before hashing: the external evaluator may retain its original
    # array, but it cannot change the content we bind and hand to the writer.
    snapshot = np.array(result.rgb, dtype=np.uint16, order="C", copy=True)
    expected_output = rgb16_content_sha256(snapshot)
    if result.output_rgb_sha256 != expected_output:
        raise ExactColorIntegrityError("evaluator output hash does not match the returned RGB content")
    _validate_cms_receipt(
        cms_payload,
        builder_receipt_sha256=builder_receipt.sha256,
        input_rgb_sha256=expected_stage1,
        output_rgb_sha256=expected_output,
    )

    snapshot.setflags(write=False)
    return ExactColorResult(
        rgb=snapshot,
        input_rgb_sha256=expected_stage1,
        output_rgb_sha256=expected_output,
        builder_receipt=result.builder_receipt,
        cms_receipt=result.cms_receipt,
        source_rgb_sha256=expected_source,
        builder_application_receipt=builder_result.application_receipt,
    )


def receipt_payload(
    receipt: BuilderReceipt | VerifiedBuilderApplicationReceipt | VerifiedCMSReceipt,
) -> dict[str, Any]:
    """Return a JSON-safe copy for the outer NegPy scan receipt."""

    if isinstance(receipt, (ValidatedBuilderReceipt, NativeValidatedBuilderReceipt)):
        return builder_receipt_payload(receipt)
    if type(receipt) is VerifiedBuilderApplicationReceipt:
        expected = VerifiedBuilderApplicationReceipt
    elif type(receipt) is VerifiedCMSReceipt:
        expected = VerifiedCMSReceipt
    else:
        raise ExactColorIntegrityError("receipt has an invalid type")
    return _receipt_payload(receipt, expected, label="receipt")


def builder_receipt_payload(receipt: BuilderReceipt) -> dict[str, Any]:
    """Validate every artifact binding in a replay or native builder receipt."""

    if type(receipt) is NativeValidatedBuilderReceipt:
        return _native_builder_receipt_payload(receipt)

    if type(receipt) is not ValidatedBuilderReceipt or receipt._factory_token is not _TRUSTED_BUILDER_RECEIPT_TOKEN:
        raise ExactColorIntegrityError("builder receipt was not produced by the trusted Stage-3 replay file loader")
    payload = _receipt_payload(receipt, ValidatedBuilderReceipt, label="builder")
    _validate_sha256(receipt.stage3_receipt_sha256, label="Stage-3 receipt")
    if type(receipt.stage3_receipt) is not bytes or not receipt.stage3_receipt:
        raise ExactColorIntegrityError("Stage-3 validation receipt is missing")
    if not hmac.compare_digest(
        hashlib.sha256(receipt.stage3_receipt).hexdigest(),
        receipt.stage3_receipt_sha256,
    ):
        raise ExactColorIntegrityError("Stage-3 validation receipt does not match its SHA-256")
    _validate_pre_f_artifacts(receipt)
    stage3 = _parse_json_object(receipt.stage3_receipt, label="Stage-3 validation")
    _validate_stage3_report(stage3, receipt.pre_f_lut_sha256)

    expected_luts = [
        {
            "bytes": _PRE_F_LUT_BYTES,
            "channel": channel,
            "sha256": digest,
        }
        for channel, digest in zip(_BUILDER_CHANNELS, receipt.pre_f_lut_sha256, strict=True)
    ]
    if payload.get("schema") != BUILDER_RECEIPT_SCHEMA or type(payload.get("version")) is not int or payload.get("version") != 1:
        raise ExactColorIntegrityError("builder receipt envelope schema is unsupported")
    if payload.get("scope") != STAGE3_REPLAY_SCOPE or payload.get("native_per_acquisition_builder") is not False:
        raise ExactColorIntegrityError("builder receipt does not identify the Stage-3 evidence/replay bridge")
    if payload.get("stage3_receipt_sha256") != receipt.stage3_receipt_sha256:
        raise ExactColorIntegrityError("builder receipt envelope does not bind the Stage-3 receipt")
    if payload.get("pre_f_luts") != expected_luts:
        raise ExactColorIntegrityError("builder receipt envelope does not bind the three pre-F LUTs")
    if payload.get("fixed_composition") != {
        "lut_sha256": FIXED_COMPOSITION_SHA256,
        "order": "F[B_c(i)]",
    }:
        raise ExactColorIntegrityError("builder receipt envelope does not bind the fixed post-F composition")
    if (
        type(receipt.evidence_filenames) is not tuple
        or len(receipt.evidence_filenames) != 4
        or type(receipt.evidence_filenames[0]) is not str
        or not receipt.evidence_filenames[0]
        or receipt.evidence_filenames[1:]
        != (
            "builder-preF-r.bin",
            "builder-preF-g.bin",
            "builder-preF-b.bin",
        )
    ):
        raise ExactColorIntegrityError("builder receipt evidence filenames are malformed")
    return payload


def _native_builder_receipt_payload(receipt: NativeValidatedBuilderReceipt) -> dict[str, Any]:
    if receipt._factory_token is not _TRUSTED_NATIVE_BUILDER_RECEIPT_TOKEN:
        raise ExactColorIntegrityError("native builder receipt was not produced by the trusted native factory")
    payload = _receipt_payload(receipt, NativeValidatedBuilderReceipt, label="native builder")
    if type(receipt.evidence_payload) is not bytes or not receipt.evidence_payload:
        raise ExactColorIntegrityError("native builder evidence payload is missing")
    _validate_sha256(receipt.evidence_sha256, label="native builder evidence")
    if not hmac.compare_digest(hashlib.sha256(receipt.evidence_payload).hexdigest(), receipt.evidence_sha256):
        raise ExactColorIntegrityError("native builder evidence does not match its SHA-256")
    evidence = _parse_json_object(receipt.evidence_payload, label="native builder evidence")
    for label, blob, expected_sha256 in (
        ("frame ownership", receipt.frame_ownership_receipt, receipt.frame_ownership_receipt_sha256),
        ("density evidence", receipt.density_evidence_receipt, receipt.density_evidence_receipt_sha256),
    ):
        _validate_sha256(expected_sha256, label=f"native {label}")
        if type(blob) is not bytes or not blob or not hmac.compare_digest(hashlib.sha256(blob).hexdigest(), expected_sha256):
            raise ExactColorIntegrityError(f"native {label} receipt does not match its SHA-256")
    _validate_native_evidence_document(evidence, receipt)
    if (
        type(receipt.analyzer_rgb) is not bytes
        or type(receipt.analyzer_shape) is not tuple
        or receipt.analyzer_shape != (425, 281, 3)
        or len(receipt.analyzer_rgb) != 425 * 281 * 3 * 2
    ):
        raise ExactColorIntegrityError("native builder analyzer snapshot has unsupported geometry")
    _validate_sha256(receipt.analyzer_rgb_sha256, label="native builder analyzer RGB")
    if not hmac.compare_digest(hashlib.sha256(receipt.analyzer_rgb).hexdigest(), receipt.analyzer_rgb_sha256):
        raise ExactColorIntegrityError("native builder analyzer snapshot does not match its SHA-256")
    _validate_pre_f_artifacts(receipt)
    expected_luts = [
        {"bytes": _PRE_F_LUT_BYTES, "channel": channel, "sha256": digest}
        for channel, digest in zip(_BUILDER_CHANNELS, receipt.pre_f_lut_sha256, strict=True)
    ]
    identity = {
        "batch_session_id": receipt.batch_session_id,
        "capture_attempt_id": receipt.capture_attempt_id,
        "preview_identity_sha256": receipt.preview_identity_sha256,
        "preview_sha256": receipt.preview_sha256,
        "reservation_id": receipt.reservation_id,
        "scan_identity": receipt.scan_identity,
        "session_id": receipt.session_id,
        "slot": receipt.slot,
    }
    if (
        set(payload)
        != {
            "algorithm",
            "analyzer_source",
            "density_source",
            "evidence_sha256",
            "fixed_composition",
            "identity",
            "native_per_acquisition_builder",
            "pre_f_luts",
            "schema",
            "scope",
            "version",
        }
        or payload.get("schema") != BUILDER_RECEIPT_SCHEMA
        or type(payload.get("version")) is not int
        or payload.get("version") != 1
        or payload.get("scope") != NATIVE_BUILDER_SCOPE
        or payload.get("native_per_acquisition_builder") is not True
        or payload.get("algorithm") != NATIVE_BUILDER_ALGORITHM_ID
        or payload.get("density_source") != evidence.get("density_source")
        or payload.get("analyzer_source") != evidence.get("analyzer_source")
        or payload.get("identity") != identity
        or payload.get("evidence_sha256") != receipt.evidence_sha256
        or payload.get("pre_f_luts") != expected_luts
        or payload.get("fixed_composition") != {"lut_sha256": FIXED_COMPOSITION_SHA256, "order": "F[B_c(i)]"}
    ):
        raise ExactColorIntegrityError("native builder receipt envelope is malformed or incompletely bound")
    if evidence.get("analyzer_source", {}).get("rgb_sha256") != receipt.analyzer_rgb_sha256:
        raise ExactColorIntegrityError("native builder evidence does not bind its analyzer snapshot")
    return payload


def _validate_native_evidence_document(
    evidence: dict[str, Any],
    receipt: NativeValidatedBuilderReceipt,
) -> None:
    expected_identity = {
        "batch_session_id": receipt.batch_session_id,
        "capture_attempt_id": receipt.capture_attempt_id,
        "preview_identity_sha256": receipt.preview_identity_sha256,
        "preview_sha256": receipt.preview_sha256,
        "reservation_id": receipt.reservation_id,
        "scan_identity": receipt.scan_identity,
        "session_id": receipt.session_id,
        "slot": receipt.slot,
    }
    if set(evidence) != {
        "algorithm",
        "analyzer_source",
        "calibration",
        "density_evidence",
        "density_source",
        "frame_ownership",
        "identity",
        "resource",
        "schema",
        "version",
    }:
        raise ExactColorIntegrityError("native builder evidence fields are incomplete")
    if (
        evidence.get("schema") != "negpy.native-stage1-builder-evidence"
        or type(evidence.get("version")) is not int
        or evidence.get("version") != 1
        or evidence.get("identity") != expected_identity
        or evidence.get("resource") != {"bytes": 1_024, "sha256": NATIVE_RESOURCE_SHA256}
        or evidence.get("algorithm")
        != {
            "curve": "ls5000-md3-10010c30-pref-v1",
            "parameter_derivation": "ls5000-md3-100100d0-to-1000f470-v1",
        }
    ):
        raise ExactColorIntegrityError("native builder evidence provenance is unsupported")
    ownership = evidence.get("frame_ownership")
    if type(ownership) is not dict or set(ownership) != {
        "schema_version",
        "scope",
        "binding_status",
        "session_reservation_retained",
        "reservation_id",
        "batch_session_id",
        "preview_sha256",
        "preview_identity_sha256",
        "transport_table_sha256",
        "transport_identity_sha256",
        "reviewed_fingerprint_sha256",
        "fresh_fingerprint_sha256",
        "frame_capture_attempt_id",
        "frame_index",
        "frame_total",
        "selected_slots",
        "selected_slot",
    }:
        raise ExactColorIntegrityError("native frame-ownership evidence is malformed")
    ownership_blob = _canonical_json(cast(dict[str, Any], ownership))
    if (
        ownership_blob != receipt.frame_ownership_receipt
        or hashlib.sha256(ownership_blob).hexdigest() != receipt.frame_ownership_receipt_sha256
        or ownership.get("schema_version") != 1
        or ownership.get("scope") != "reservation-preview-frame"
        or ownership.get("binding_status") != "proven-exact-reservation-preview-registration-and-transport"
        or ownership.get("session_reservation_retained") is not True
        or ownership.get("reservation_id") != receipt.reservation_id
        or ownership.get("batch_session_id") != receipt.batch_session_id
        or receipt.session_id != receipt.reservation_id
        or receipt.reservation_id != receipt.batch_session_id
        or ownership.get("preview_sha256") != receipt.preview_sha256
        or ownership.get("preview_identity_sha256") != receipt.preview_identity_sha256
        or ownership.get("frame_capture_attempt_id") != receipt.capture_attempt_id
        or ownership.get("selected_slot") != receipt.slot
    ):
        raise ExactColorIntegrityError("native frame ownership belongs to another preview, reservation, or frame")
    _validate_sha256(ownership.get("transport_table_sha256"), label="native transport table")
    _validate_sha256(ownership.get("transport_identity_sha256"), label="native transport identity")
    _validate_sha256(ownership.get("reviewed_fingerprint_sha256"), label="native reviewed fingerprint")
    _validate_sha256(ownership.get("fresh_fingerprint_sha256"), label="native fresh fingerprint")
    selected_slots = ownership.get("selected_slots")
    frame_index = ownership.get("frame_index")
    frame_total = ownership.get("frame_total")
    if (
        type(selected_slots) is not list
        or type(frame_index) is not int
        or type(frame_total) is not int
        or len(selected_slots) != frame_total
        or not 1 <= frame_index <= frame_total
        or selected_slots[frame_index - 1] != receipt.slot
    ):
        raise ExactColorIntegrityError("native frame ownership batch coordinates are malformed")
    transport_material = {
        "reservation_id": receipt.reservation_id,
        "batch_session_id": receipt.batch_session_id,
        "preview_sha256": receipt.preview_sha256,
        "preview_identity_sha256": receipt.preview_identity_sha256,
        "transport_table_sha256": ownership.get("transport_table_sha256"),
        "reviewed_fingerprint_sha256": ownership.get("reviewed_fingerprint_sha256"),
        "fresh_fingerprint_sha256": ownership.get("fresh_fingerprint_sha256"),
        "selected_slots": selected_slots,
    }
    if hashlib.sha256(_canonical_json(transport_material)).hexdigest() != ownership.get("transport_identity_sha256"):
        raise ExactColorIntegrityError("native frame ownership transport identity digest is invalid")
    density_evidence = evidence.get("density_evidence")
    density_evidence_document = _parse_json_object(receipt.density_evidence_receipt, label="native density evidence")
    if (
        density_evidence != {"receipt_sha256": receipt.density_evidence_receipt_sha256, "scope": "reservation-preview"}
        or set(density_evidence_document)
        != {
            "schema_version",
            "scope",
            "per_frame_binding_status",
            "preview_identity_sha256",
            "source_payload_bytes",
            "calibration_binding",
            "source_binding",
            "exposure_binding",
            "result",
        }
        or density_evidence_document.get("schema_version") != 1
        or density_evidence_document.get("scope") != "reservation-preview"
        or density_evidence_document.get("per_frame_binding_status") != "requires-explicit-frame-ownership-receipt"
        or density_evidence_document.get("source_payload_bytes")
        not in _SUPPORTED_DENSITY_SOURCE_WIRE_BYTES
        or density_evidence_document.get("preview_identity_sha256") != receipt.preview_identity_sha256
    ):
        raise ExactColorIntegrityError("native density evidence belongs to another preview or reservation")
    calibration_binding = density_evidence_document.get("calibration_binding")
    source_binding = density_evidence_document.get("source_binding")
    exposure_binding = density_evidence_document.get("exposure_binding")
    density_result = density_evidence_document.get("result")
    if not all(type(value) is dict for value in (calibration_binding, source_binding, exposure_binding, density_result)):
        raise ExactColorIntegrityError("native density evidence bindings are malformed")
    calibration_binding = cast(dict[str, Any], calibration_binding)
    source_binding = cast(dict[str, Any], source_binding)
    exposure_binding = cast(dict[str, Any], exposure_binding)
    density_result = cast(dict[str, Any], density_result)
    density_calibration = calibration_binding.get("calibration")
    if type(density_calibration) is not dict:
        raise ExactColorIntegrityError("native density calibration evidence is malformed")
    binding_identity = (
        source_binding.get("session_id"),
        source_binding.get("capture_attempt_id"),
        source_binding.get("scan_identity"),
    )
    if (
        binding_identity
        != (
            exposure_binding.get("session_id"),
            exposure_binding.get("capture_attempt_id"),
            exposure_binding.get("scan_identity"),
        )
        or binding_identity
        != (
            density_result.get("session_id"),
            density_result.get("capture_attempt_id"),
            density_result.get("scan_identity"),
        )
        or binding_identity
        != (
            density_calibration.get("session_id"),
            calibration_binding.get("capture_attempt_id"),
            calibration_binding.get("scan_identity"),
        )
        or binding_identity[0] != receipt.reservation_id
        or binding_identity[2] != receipt.scan_identity
        or source_binding.get("wire_sha256") != receipt.preview_sha256
        or density_result.get("source_wire_sha256") != receipt.preview_sha256
        or density_result.get("promotable") is not True
    ):
        raise ExactColorIntegrityError("native density evidence identity is inconsistent")
    calibration = evidence.get("calibration")
    if type(calibration) is not dict or set(calibration) != {"read8c_numerators_rgb"}:
        raise ExactColorIntegrityError("native calibration evidence is malformed")
    numerators = calibration.get("read8c_numerators_rgb")
    density = evidence.get("density_source")
    analyzer = evidence.get("analyzer_source")
    if type(density) is not dict or set(density) != {
        "arithmetic",
        "child_buffer_sha256",
        "densities_binary64_hex_rgb",
        "f03_denominators_raw_10ns_rgb",
        "resolution_dpi",
        "wire_sha256",
    }:
        raise ExactColorIntegrityError("native density-source evidence is malformed")
    if type(analyzer) is not dict or set(analyzer) != {
        "final_f02_denominators_raw_10ns_rgb",
        "geometry",
        "rectangle_inclusive",
        "resolution_dpi",
        "rgb_bytes",
        "rgb_sha256",
    }:
        raise ExactColorIntegrityError("native analyzer-source evidence is malformed")
    density_denominators = density.get("f03_denominators_raw_10ns_rgb")
    final_denominators = analyzer.get("final_f02_denominators_raw_10ns_rgb")
    for label, values in (
        ("calibration numerators", numerators),
        ("density f03 denominators", density_denominators),
        ("final f02 denominators", final_denominators),
    ):
        if type(values) is not list or len(values) != 3 or any(type(value) is not int or not 1 <= value <= 0xFFFFFFFF for value in values):
            raise ExactColorIntegrityError(f"native {label} are malformed")
    if density_denominators == final_denominators:
        raise ExactColorIntegrityError("native density f03 and final f02 exposure triplets were conflated")
    density_hex = density.get("densities_binary64_hex_rgb")
    try:
        decoded_density = [float.fromhex(value) for value in density_hex] if type(density_hex) is list else []
    except (TypeError, ValueError) as error:
        raise ExactColorIntegrityError("native density doubles are malformed") from error
    if len(decoded_density) != 3 or any(not math.isfinite(value) for value in decoded_density):
        raise ExactColorIntegrityError("native density doubles are malformed")
    density_binary64 = [struct.pack(">d", value).hex() for value in decoded_density]
    if (
        source_binding.get("resolution_dpi") != 97
        or source_binding.get("wire_sha256") != density.get("wire_sha256")
        or source_binding.get("child_buffer_sha256") != density.get("child_buffer_sha256")
        or density_calibration.get("numerators_rgb") != numerators
        or exposure_binding.get("density_f03_exposures_raw_10ns_rgb") != density_denominators
        or density_result.get("algorithm_id") != density.get("arithmetic")
        or density_result.get("source_wire_sha256") != density.get("wire_sha256")
        or density_result.get("source_child_buffer_sha256") != density.get("child_buffer_sha256")
        or density_result.get("numerators_rgb") != numerators
        or density_result.get("density_f03_denominators_raw_10ns_rgb") != density_denominators
        or density_result.get("densities_rgb") != decoded_density
        or density_result.get("density_binary64_be_hex_rgb") != density_binary64
    ):
        raise ExactColorIntegrityError("native density receipt does not bind the builder inputs")
    _validate_sha256(density.get("wire_sha256"), label="native density source wire")
    _validate_sha256(density.get("child_buffer_sha256"), label="native density source child")
    rectangle = analyzer.get("rectangle_inclusive")
    if (
        density.get("arithmetic") != "ls5000-md3-10088810-layout1-u16-proven-inputs-macos-binary64-exact-v6"
        or density.get("resolution_dpi") != 97
        or analyzer.get("resolution_dpi") != 285
        or analyzer.get("geometry") != [425, 281, 3]
        or analyzer.get("rgb_bytes") != 425 * 281 * 3 * 2
        or type(rectangle) is not list
        or len(rectangle) != 4
        or any(type(value) is not int for value in rectangle)
        or not (0 <= rectangle[0] <= rectangle[2] < 425 and 0 <= rectangle[1] <= rectangle[3] < 281)
    ):
        raise ExactColorIntegrityError("native density/analyzer geometry or arithmetic is unsupported")
    _validate_sha256(analyzer.get("rgb_sha256"), label="native analyzer RGB")


def _validate_builder_application_receipt(
    payload: dict[str, Any],
    receipt: BuilderReceipt,
) -> None:
    common_ok = (
        payload.get("kind") == "negpy.verified-stage1-builder-application"
        and type(payload.get("version")) is int
        and payload.get("version") == 1
        and payload.get("fixed_composition") == {"lut_sha256": FIXED_COMPOSITION_SHA256, "order": "F[B_c(i)]"}
        and payload.get("pre_f_lut_sha256") == dict(zip(_BUILDER_CHANNELS, receipt.pre_f_lut_sha256, strict=True))
    )
    if isinstance(receipt, ValidatedBuilderReceipt):
        specific_ok = (
            payload.get("scope") == STAGE3_REPLAY_SCOPE
            and payload.get("native_per_acquisition_builder") is False
            and payload.get("stage3_receipt_sha256") == receipt.stage3_receipt_sha256
        )
        label = "validated Stage-3 artifacts"
    elif isinstance(receipt, NativeValidatedBuilderReceipt):
        specific_ok = (
            payload.get("scope") == NATIVE_BUILDER_SCOPE
            and payload.get("native_per_acquisition_builder") is True
            and payload.get("native_evidence_sha256") == receipt.evidence_sha256
        )
        label = "validated native builder artifacts"
    else:  # pragma: no cover - BuilderReceipt is a closed union
        raise ExactColorIntegrityError("builder receipt has an invalid type")
    if not common_ok or not specific_ok:
        raise ExactColorIntegrityError(f"builder application receipt does not bind the {label}")


def _receipt_payload(
    receipt: object,
    expected_type: (
        type[ValidatedBuilderReceipt]
        | type[NativeValidatedBuilderReceipt]
        | type[VerifiedBuilderApplicationReceipt]
        | type[VerifiedCMSReceipt]
    ),
    *,
    label: str,
) -> dict[str, Any]:
    if (
        not isinstance(
            receipt,
            (
                ValidatedBuilderReceipt,
                NativeValidatedBuilderReceipt,
                VerifiedBuilderApplicationReceipt,
                VerifiedCMSReceipt,
            ),
        )
        or type(receipt) is not expected_type
    ):
        raise ExactColorIntegrityError(f"{label} receipt has an invalid type")
    payload = receipt.payload
    if type(payload) is not bytes or not payload or len(payload) > _MAX_RECEIPT_BYTES:
        raise ExactColorIntegrityError(f"{label} receipt payload is missing or unbounded")
    if type(receipt) is VerifiedCMSReceipt and receipt._factory_token is not _TRUSTED_CMS_RECEIPT_TOKEN:
        raise ExactColorIntegrityError("CMS receipt was not issued by the trusted portable CMS adapter")
    if receipt.attested is not True:
        raise ExactColorIntegrityError(f"{label} receipt is not attested")
    if type(receipt.sha256) is not str or not _SHA256.fullmatch(receipt.sha256):
        raise ExactColorIntegrityError(f"{label} receipt SHA-256 is malformed")
    if not hmac.compare_digest(hashlib.sha256(payload).hexdigest(), receipt.sha256):
        raise ExactColorIntegrityError(f"{label} receipt payload does not match its SHA-256")
    return _parse_json_object(payload, label=f"{label} receipt")


def _parse_json_object(payload: bytes, *, label: str) -> dict[str, Any]:
    if type(payload) is not bytes or not payload or len(payload) > _MAX_RECEIPT_BYTES:
        raise ExactColorIntegrityError(f"{label} payload is missing or unbounded")
    try:
        parsed = json.loads(
            payload,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, ValueError, RecursionError) as error:
        raise ExactColorIntegrityError(f"{label} is not valid JSON: {error}") from error
    if type(parsed) is not dict:
        raise ExactColorIntegrityError(f"{label} must contain a JSON object")
    return parsed


def _validate_sha256(value: object, *, label: str) -> None:
    if type(value) is not str or not _SHA256.fullmatch(value):
        raise ExactColorIntegrityError(f"{label} SHA-256 is malformed")


def _validate_pre_f_artifacts(receipt: BuilderReceipt) -> None:
    if type(receipt.pre_f_luts) is not tuple or len(receipt.pre_f_luts) != 3:
        raise ExactColorIntegrityError("builder receipt must contain exactly three pre-F LUT byte blobs")
    if type(receipt.pre_f_lut_sha256) is not tuple or len(receipt.pre_f_lut_sha256) != 3:
        raise ExactColorIntegrityError("builder receipt must contain exactly three pre-F LUT hashes")
    for channel, blob, expected in zip(
        _BUILDER_CHANNELS,
        receipt.pre_f_luts,
        receipt.pre_f_lut_sha256,
        strict=True,
    ):
        if type(blob) is not bytes or len(blob) != _PRE_F_LUT_BYTES:
            raise ExactColorIntegrityError(f"builder pre-F {channel} LUT must be exactly {_PRE_F_LUT_BYTES} bytes")
        _validate_sha256(expected, label=f"builder pre-F {channel} LUT")
        if not hmac.compare_digest(hashlib.sha256(blob).hexdigest(), expected):
            raise ExactColorIntegrityError(f"builder pre-F {channel} LUT does not match its SHA-256")
    _validate_sha256(receipt.fixed_composition_sha256, label="fixed composition LUT")
    if receipt.fixed_composition_sha256 != FIXED_COMPOSITION_SHA256:
        raise ExactColorIntegrityError("builder receipt names an unsupported fixed composition LUT")


def _validate_stage3_report(
    report: dict[str, Any],
    pre_f_lut_sha256: tuple[str, str, str],
) -> None:
    if (
        report.get("schema") != STAGE3_REPORT_SCHEMA
        or type(report.get("schema_version")) is not int
        or report.get("schema_version") != 1
        or report.get("status") != "pass"
    ):
        raise ExactColorIntegrityError("Stage-3 validation report is not a supported PASS receipt")
    if report.get("errors") != []:
        raise ExactColorIntegrityError("Stage-3 validation report errors must be exactly empty")
    if type(report.get("capture_directory")) is not str or not report["capture_directory"]:
        raise ExactColorIntegrityError("Stage-3 validation report capture directory is malformed")
    required_summary = {
        "callback_analyzer_exact": True,
        "builder_scalars_exact": 21,
        "builder_scalars_total": 21,
        "builder_controls_exact": 6,
        "builder_controls_total": 6,
        "builder_pref_channels_exact": 3,
        "builder_pref_channels_total": 3,
        "builder_pref_mismatched_u16": 0,
        "builder_pref_total_u16": 196_608,
        "captured_args_replay_channels_exact": 3,
        "lifecycle_exact": True,
    }
    summary = report.get("summary")
    if (
        type(summary) is not dict
        or summary != required_summary
        or any(type(summary[key]) is not type(value) for key, value in required_summary.items())
    ):
        raise ExactColorIntegrityError("Stage-3 validation report does not close every builder gate")

    provenance = report.get("provenance")
    if type(provenance) is not dict or set(provenance) != {
        "module",
        "observer_executable",
        "observer_source",
        "resource",
    }:
        raise ExactColorIntegrityError("Stage-3 validation report provenance fields are incomplete")
    _validate_observer_provenance(provenance.get("observer_source"), label="observer source")
    _validate_observer_provenance(provenance.get("observer_executable"), label="observer executable")
    module = provenance.get("module")
    if (
        type(module) is not dict
        or set(module) != {"path", "bytes", "sha256"}
        or type(module.get("path")) is not str
        or not module["path"]
        or type(module.get("bytes")) is not int
        or module.get("bytes") != _STAGE3_MODULE_BYTES
        or module.get("sha256") != _STAGE3_MODULE_SHA256
    ):
        raise ExactColorIntegrityError("Stage-3 module provenance is not the pinned LS5000.md3")
    resource = provenance.get("resource")
    if (
        type(resource) is not dict
        or resource
        != {
            "bytes": _STAGE3_RESOURCE_BYTES,
            "sha256": _STAGE3_RESOURCE_SHA256,
            "virtual_address": _STAGE3_RESOURCE_VA,
        }
        or type(resource.get("bytes")) is not int
    ):
        raise ExactColorIntegrityError("Stage-3 resource provenance is not the pinned LS5000.md3 block")

    artifacts = report.get("artifacts")
    if type(artifacts) is not list:
        raise ExactColorIntegrityError("Stage-3 validation report has no artifact inventory")
    inventory: dict[str, dict[str, Any]] = {}
    for row in artifacts:
        if type(row) is not dict or set(row) != {"bytes", "name", "sha256"} or type(row.get("name")) is not str:
            raise ExactColorIntegrityError("Stage-3 artifact inventory is malformed")
        name = row["name"]
        if name in inventory:
            raise ExactColorIntegrityError(f"Stage-3 artifact inventory repeats {name!r}")
        if type(row.get("bytes")) is not int or row["bytes"] <= 0:
            raise ExactColorIntegrityError(f"Stage-3 artifact inventory has an invalid byte size for {name!r}")
        _validate_sha256(row.get("sha256"), label=f"Stage-3 artifact {name}")
        inventory[name] = row
    if set(inventory) != _STAGE3_ARTIFACTS:
        raise ExactColorIntegrityError("Stage-3 artifact inventory does not contain the exact required file set")
    for name, expected_bytes in _STAGE3_FIXED_ARTIFACT_BYTES.items():
        if inventory[name]["bytes"] != expected_bytes:
            raise ExactColorIntegrityError(f"Stage-3 artifact inventory has the wrong size for {name}")
    if inventory["callback-buffer.bin"]["bytes"] not in _STAGE3_CALLBACK_BYTES:
        raise ExactColorIntegrityError("Stage-3 callback artifact has unsupported geometry")
    for channel, expected in zip(_BUILDER_CHANNELS, pre_f_lut_sha256, strict=True):
        if inventory.get(f"builder-preF-{channel}.bin") != {
            "bytes": _PRE_F_LUT_BYTES,
            "name": f"builder-preF-{channel}.bin",
            "sha256": expected,
        }:
            raise ExactColorIntegrityError(f"Stage-3 validation report does not bind builder-preF-{channel}.bin")


def _validate_observer_provenance(value: object, *, label: str) -> None:
    if type(value) is not dict:
        raise ExactColorIntegrityError(f"Stage-3 {label} provenance is malformed")
    row = cast(dict[str, Any], value)
    if (
        set(row) != {"path", "bytes", "sha256"}
        or type(row.get("path")) is not str
        or not row["path"]
        or type(row.get("bytes")) is not int
        or row["bytes"] <= 0
    ):
        raise ExactColorIntegrityError(f"Stage-3 {label} provenance is malformed")
    _validate_sha256(row.get("sha256"), label=f"Stage-3 {label}")


def _validate_cms_receipt(
    payload: dict[str, Any],
    *,
    builder_receipt_sha256: str,
    input_rgb_sha256: str,
    output_rgb_sha256: str,
) -> None:
    required_keys = {
        "algorithm",
        "assets",
        "builder_receipt_sha256",
        "chunk_pixels",
        "dll_free",
        "input_rgb_sha256",
        "kind",
        "oracle_source",
        "output_rgb_sha256",
        "scope",
        "stage_order",
        "upstream_builder_included",
        "validation",
        "version",
    }
    if set(payload) != required_keys:
        raise ExactColorIntegrityError("CMS receipt does not contain the exact production contract")
    if (
        payload.get("kind") != CMS_RECEIPT_KIND
        or type(payload.get("version")) is not int
        or payload.get("version") != CMS_RECEIPT_VERSION
        or payload.get("algorithm") != CMS_ALGORITHM_ID
    ):
        raise ExactColorIntegrityError("CMS receipt kind, version, or algorithm is unsupported")
    if payload.get("assets") != dict(CMS_ASSET_SHA256):
        raise ExactColorIntegrityError("CMS receipt does not bind the exact nine transform assets")
    if payload.get("oracle_source") != {
        "path": "portable_oracle_evaluator.py",
        "sha256": CMS_ORACLE_SOURCE_SHA256,
    }:
        raise ExactColorIntegrityError("CMS receipt does not bind the verified oracle source")
    validation = payload.get("validation")
    expected_validation = {
        "events": 12,
        "full_payload_mismatched_bytes": 0,
        "full_payload_total_bytes": 698_880,
        "mismatched_u16": 0,
        "receipt_sha256": CMS_VALIDATION_RECEIPT_SHA256,
        "total_u16": 265_440,
    }
    if (
        type(validation) is not dict
        or validation != expected_validation
        or any(type(validation[key]) is not type(value) for key, value in expected_validation.items())
    ):
        raise ExactColorIntegrityError("CMS receipt does not bind the zero-mismatch 12-event validation")
    if (
        payload.get("scope") != CMS_SCOPE
        or payload.get("stage_order") != ["stage1", "stage2"]
        or payload.get("dll_free") is not True
        or payload.get("upstream_builder_included") is not False
    ):
        raise ExactColorIntegrityError("CMS receipt scope or stage contract is unsupported")
    chunk_pixels = payload.get("chunk_pixels")
    if type(chunk_pixels) is not int or not 1 <= chunk_pixels <= 262_144:
        raise ExactColorIntegrityError("CMS receipt chunk size is invalid")
    if (
        payload.get("builder_receipt_sha256") != builder_receipt_sha256
        or payload.get("input_rgb_sha256") != input_rgb_sha256
        or payload.get("output_rgb_sha256") != output_rgb_sha256
    ):
        raise ExactColorIntegrityError("CMS receipt does not bind its builder, input, and output")


def _stable_read_non_symlink(
    path: Path,
    *,
    label: str,
    expected_bytes: int | None = None,
    max_bytes: int | None = None,
) -> tuple[bytes, tuple[int, int, int, int, int]]:
    byte_limit = expected_bytes if expected_bytes is not None else max_bytes
    if byte_limit is None:
        raise ExactColorIntegrityError(f"{label} read has no byte bound")
    descriptor: int | None = None
    try:
        before = path.lstat()
        if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
            raise OSError("not a regular non-symlink file")
        if expected_bytes is not None and before.st_size != expected_bytes:
            raise ExactColorIntegrityError(f"{label} must be exactly {expected_bytes} bytes")
        if max_bytes is not None and before.st_size > max_bytes:
            raise ExactColorIntegrityError(f"{label} exceeds the bounded receipt size")
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
        descriptor = os.open(path, flags)
        opened = os.fstat(descriptor)
        chunks: list[bytes] = []
        total = 0
        while total <= byte_limit:
            chunk = os.read(
                descriptor,
                min(1024 * 1024, byte_limit + 1 - total),
            )
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
        if total > byte_limit:
            raise ExactColorIntegrityError(f"{label} exceeds its bounded byte length")
        after = os.fstat(descriptor)
        final = path.lstat()
    except ExactColorIntegrityError:
        raise
    except OSError as error:
        raise ExactColorUnavailable(f"{label} is unavailable: {path}: {error}") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
    identities = tuple(_file_identity(item) for item in (before, opened, after, final))
    if len(set(identities)) != 1:
        raise ExactColorIntegrityError(f"{label} changed while being read: {path}")
    payload = b"".join(chunks)
    if len(payload) != before.st_size:
        raise ExactColorIntegrityError(f"{label} changed byte length while being read: {path}")
    return payload, identities[0]


def _assert_paths_unchanged(
    paths: tuple[Path, ...],
    identities: tuple[tuple[int, int, int, int, int], ...],
    *,
    label: str = "Stage-3 replay evidence",
) -> None:
    for path, expected in zip(paths, identities, strict=True):
        try:
            current = path.lstat()
        except OSError as error:
            raise ExactColorUnavailable(f"{label} disappeared: {path}: {error}") from error
        if stat.S_ISLNK(current.st_mode) or not stat.S_ISREG(current.st_mode) or _file_identity(current) != expected:
            raise ExactColorIntegrityError(f"{label} changed during snapshot: {path}")


def _file_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _canonical_json(value: dict[str, Any]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    for key, value in pairs:
        if key in parsed:
            raise ValueError(f"duplicate receipt key {key!r}")
        parsed[key] = value
    return parsed


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant {value!r}")


def _validate_rgb16(rgb: object, *, label: str) -> None:
    if (
        not isinstance(rgb, np.ndarray)
        or rgb.dtype != np.uint16
        or rgb.ndim != 3
        or rgb.shape[2] != 3
        or rgb.shape[0] == 0
        or rgb.shape[1] == 0
    ):
        raise ExactColorIntegrityError(f"{label} must be a non-empty HxWx3 uint16 array")


__all__ = [
    "BUILDER_RECEIPT_SCHEMA",
    "BuilderReceipt",
    "CMS_ALGORITHM_ID",
    "CMS_ASSET_SHA256",
    "CMS_ORACLE_SOURCE_SHA256",
    "CMS_RECEIPT_KIND",
    "CMS_RECEIPT_VERSION",
    "CMS_SCOPE",
    "CMS_VALIDATION_RECEIPT_SHA256",
    "ExactColorIntegrityError",
    "ExactColorResult",
    "ExactColorUnavailable",
    "FIXED_COMPOSITION_SHA256",
    "NATIVE_BUILDER_ALGORITHM_ID",
    "NATIVE_BUILDER_SCOPE",
    "NATIVE_RESOURCE_SHA256",
    "NativeValidatedBuilderReceipt",
    "PositiveColorMode",
    "STAGE3_REPORT_SCHEMA",
    "STAGE3_REPLAY_SCOPE",
    "Stage1BuilderResult",
    "ValidatedBuilderReceipt",
    "VerifiedBuilderApplicationReceipt",
    "VerifiedCMSReceipt",
    "VerifiedPortableCMSEvaluator",
    "VerifiedStage1Builder",
    "builder_receipt_payload",
    "evaluate_exact_color",
    "load_native_builder_receipt",
    "load_stage3_replay_builder_receipt",
    "receipt_payload",
    "rgb16_content_sha256",
]
