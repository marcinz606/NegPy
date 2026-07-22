"""Portable Stage-1 builder tests; all hardware and VM free."""

from __future__ import annotations

import dataclasses
import json
import shutil
from hashlib import sha256
from pathlib import Path

import numpy as np
import pytest

from negpy.services.roll import exact_color
from negpy.services.roll.portable_builder import (
    FIXED_LUT_FILENAME,
    MAX_CHUNK_PIXELS,
    PortableStage1Builder,
)
from negpy.services.roll.portable_cms import PortableCMSOnEvaluator
from tests.roll._exact_fixtures import make_stage3_replay_receipt


CAPTURED_PREF = Path("/Volumes/isos/NikonRE/session20260719/capture-d")
CAPTURED_PREF_SHA256 = (
    "46a0d68ae20c72088e64a1144a0d38bf692f15f506539bbe94eb563fe437c976",
    "23eda81294817e7a2a31f1488544a6f8d3e7ac817f22d43c8a39882565c34b95",
    "3cfc61c06bac49c4c28e69afe99af01366fae6bf5ea88954f688592a8e2756bb",
)
# These are post-F only. The different `final_luts` hashes in the older
# CMS-off per-pixel report include an additional Adobe gamma encoding stage.
CAPTURED_POSTF_SHA256 = {
    "r": "4fcc07a38a64fa004e252705905775952e73a2c5f0ab1f06c56de6f51073a907",
    "g": "e4965e69289dacb21a3241db1fbbee2aa7dba4a1a15f458ed3ca55638b684a30",
    "b": "95ec797e1afed10990ba3609f728af85b7547a33a99a2eee2d67dcae1832bf70",
}


def _receipt(pre_f_arrays: tuple[np.ndarray, np.ndarray, np.ndarray]) -> exact_color.ValidatedBuilderReceipt:
    return make_stage3_replay_receipt(pre_f_arrays)


def _synthetic_luts() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.arange(65_536, dtype=np.uint32)
    return (
        ((values + 17) & 0xFFFF).astype(np.uint16),
        (65_535 - values).astype(np.uint16),
        (values // 3).astype(np.uint16),
    )


def test_builder_composes_fixed_after_each_pref_plane() -> None:
    source = np.array([[[0, 1, 2], [65_535, 40_000, 17]]], dtype=np.uint16)
    pre_f = _synthetic_luts()
    fixed_path = Path(__file__).resolve().parents[2] / "negpy/assets/portable_builder" / FIXED_LUT_FILENAME
    fixed = np.frombuffer(fixed_path.read_bytes(), dtype="<u2")

    result = PortableStage1Builder(chunk_pixels=1).apply(
        source,
        builder_receipt=_receipt(pre_f),
    )

    expected = np.empty_like(source)
    for channel in range(3):
        expected[..., channel] = fixed[pre_f[channel][source[..., channel]]]
    np.testing.assert_array_equal(result.rgb, expected)
    assert result.source_rgb_sha256 == exact_color.rgb16_content_sha256(source)
    assert result.stage1_input_rgb_sha256 == exact_color.rgb16_content_sha256(expected)
    application = exact_color.receipt_payload(result.application_receipt)
    assert application["fixed_composition"]["order"] == "F[B_c(i)]"
    assert application["source_rgb_sha256"] == result.source_rgb_sha256
    assert application["stage1_input_rgb_sha256"] == result.stage1_input_rgb_sha256


def test_builder_chunk_boundaries_are_integer_identical() -> None:
    source = np.random.default_rng(44).integers(0, 65_536, size=(2, 5, 3), dtype=np.uint16)
    receipt = _receipt(_synthetic_luts())
    outputs = [PortableStage1Builder(chunk_pixels=size).apply(source, builder_receipt=receipt).rgb for size in (1, 3, 10, 11)]

    for output in outputs[1:]:
        np.testing.assert_array_equal(output, outputs[0])


def test_exact_boundary_runs_builder_then_cms_and_keeps_receipts_separate() -> None:
    source = np.random.default_rng(46).integers(0, 65_536, size=(2, 4, 3), dtype=np.uint16)
    receipt = _receipt(_synthetic_luts())

    result = exact_color.evaluate_exact_color(
        source,
        builder_receipt=receipt,
        builder=PortableStage1Builder(chunk_pixels=3),
        evaluator=PortableCMSOnEvaluator(chunk_pixels=3),
    )

    assert result.source_rgb_sha256 == exact_color.rgb16_content_sha256(source)
    assert result.builder_application_receipt is not None
    builder_application = exact_color.receipt_payload(result.builder_application_receipt)
    cms = exact_color.receipt_payload(result.cms_receipt)
    assert result.input_rgb_sha256 == builder_application["stage1_input_rgb_sha256"]
    assert cms["input_rgb_sha256"] == result.input_rgb_sha256
    assert cms["output_rgb_sha256"] == result.output_rgb_sha256
    assert builder_application["kind"] == "negpy.verified-stage1-builder-application"
    assert cms["kind"] == "negpy.portable-cms-on-receipt"


@pytest.mark.parametrize(
    "field",
    ["pre_f_blob", "stage3_report", "stage3_artifact_binding", "fixed_identity"],
)
def test_builder_receipt_tampering_fails_closed(field: str) -> None:
    receipt = _receipt(_synthetic_luts())
    if field == "pre_f_blob":
        tampered = bytearray(receipt.pre_f_luts[0])
        tampered[99] ^= 1
        candidate = dataclasses.replace(
            receipt,
            pre_f_luts=(bytes(tampered), *receipt.pre_f_luts[1:]),
        )
    elif field == "stage3_report":
        tampered = bytearray(receipt.stage3_receipt)
        tampered[-2] ^= 1
        candidate = dataclasses.replace(receipt, stage3_receipt=bytes(tampered))
    elif field == "stage3_artifact_binding":
        report = json.loads(receipt.stage3_receipt)
        next(row for row in report["artifacts"] if row["name"] == "builder-preF-r.bin")["sha256"] = "0" * 64
        stage3 = json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
        stage3_hash = sha256(stage3).hexdigest()
        envelope = json.loads(receipt.payload)
        envelope["stage3_receipt_sha256"] = stage3_hash
        payload = json.dumps(envelope, sort_keys=True, separators=(",", ":")).encode()
        candidate = dataclasses.replace(
            receipt,
            payload=payload,
            sha256=sha256(payload).hexdigest(),
            stage3_receipt=stage3,
            stage3_receipt_sha256=stage3_hash,
        )
    else:
        candidate = dataclasses.replace(receipt, fixed_composition_sha256="0" * 64)

    with pytest.raises(exact_color.ExactColorIntegrityError):
        PortableStage1Builder().apply(
            np.zeros((1, 1, 3), dtype=np.uint16),
            builder_receipt=candidate,
        )


def test_stage3_report_with_errors_is_rejected_even_when_summary_claims_pass() -> None:
    receipt = _receipt(_synthetic_luts())
    report = json.loads(receipt.stage3_receipt)
    report["errors"] = [{"code": "suppressed", "message": "must remain fatal"}]
    stage3 = json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
    stage3_hash = sha256(stage3).hexdigest()
    envelope = json.loads(receipt.payload)
    envelope["stage3_receipt_sha256"] = stage3_hash
    payload = json.dumps(envelope, sort_keys=True, separators=(",", ":")).encode()
    candidate = dataclasses.replace(
        receipt,
        payload=payload,
        sha256=sha256(payload).hexdigest(),
        stage3_receipt=stage3,
        stage3_receipt_sha256=stage3_hash,
    )

    with pytest.raises(exact_color.ExactColorIntegrityError, match="errors"):
        exact_color.builder_receipt_payload(candidate)


def test_stage3_replay_receipts_require_the_trusted_file_loader() -> None:
    assert callable(exact_color.load_stage3_replay_builder_receipt)


@pytest.mark.parametrize("failure", ["missing", "tampered"])
def test_fixed_lut_asset_verification_fails_closed(tmp_path: Path, failure: str) -> None:
    packaged = Path(__file__).resolve().parents[2] / "negpy/assets/portable_builder"
    copied = tmp_path / "assets"
    shutil.copytree(packaged, copied)
    target = copied / FIXED_LUT_FILENAME
    if failure == "missing":
        target.unlink()
        expected = exact_color.ExactColorUnavailable
    else:
        payload = bytearray(target.read_bytes())
        payload[1000] ^= 1
        target.write_bytes(payload)
        expected = exact_color.ExactColorIntegrityError

    with pytest.raises(expected):
        PortableStage1Builder(assets_dir=copied)


def test_builder_preserves_input_and_freezes_stage1_output() -> None:
    source = np.random.default_rng(45).integers(0, 65_536, size=(2, 3, 3), dtype=np.uint16)
    original = source.copy()

    result = PortableStage1Builder(chunk_pixels=2).apply(
        source,
        builder_receipt=_receipt(_synthetic_luts()),
    )

    np.testing.assert_array_equal(source, original)
    assert not result.rgb.flags.writeable
    with pytest.raises(ValueError):
        result.rgb[0, 0, 0] ^= 1


@pytest.mark.parametrize(
    "source",
    [
        np.zeros((2, 2, 3), dtype=np.uint8),
        np.zeros((2, 2), dtype=np.uint16),
        np.zeros((2, 2, 4), dtype=np.uint16),
        np.zeros((0, 2, 3), dtype=np.uint16),
    ],
)
def test_builder_rejects_wrong_input_dtype_or_shape(source: np.ndarray) -> None:
    with pytest.raises(exact_color.ExactColorIntegrityError):
        PortableStage1Builder().apply(
            source,
            builder_receipt=_receipt(_synthetic_luts()),
        )


@pytest.mark.parametrize("chunk_pixels", [0, MAX_CHUNK_PIXELS + 1, True, 1.5])
def test_builder_chunk_size_is_bounded(chunk_pixels: object) -> None:
    with pytest.raises(exact_color.ExactColorUnavailable):
        PortableStage1Builder(chunk_pixels=chunk_pixels)  # ty: ignore[invalid-argument-type]


def test_accessible_captured_pref_luts_reproduce_pinned_postf_hashes() -> None:
    paths = [CAPTURED_PREF / f"builder-preF-{channel}.bin" for channel in ("r", "g", "b")]
    if not all(path.is_file() for path in paths):
        pytest.skip("stored capture-d builder LUTs are unavailable")
    blobs = tuple(path.read_bytes() for path in paths)
    assert tuple(sha256(blob).hexdigest() for blob in blobs) == CAPTURED_PREF_SHA256
    luts = (
        np.frombuffer(blobs[0], dtype="<u2"),
        np.frombuffer(blobs[1], dtype="<u2"),
        np.frombuffer(blobs[2], dtype="<u2"),
    )

    result = PortableStage1Builder().apply(
        np.zeros((1, 1, 3), dtype=np.uint16),
        builder_receipt=_receipt(luts),
    )

    application = exact_color.receipt_payload(result.application_receipt)
    assert application["post_f_lut_sha256"] == CAPTURED_POSTF_SHA256
