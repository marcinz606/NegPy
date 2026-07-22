"""Portable CMS-on adapter tests; all hardware and DLL free."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from negpy.services.roll import exact_color, portable_cms
from negpy.services.roll.portable_cms import (
    ASSET_SHA256,
    MAX_CHUNK_PIXELS,
    VALIDATION_RECEIPT_BYTES,
    VALIDATION_RECEIPT_FILENAME,
    VALIDATION_RECEIPT_SHA256,
    PortableCMSOnEvaluator,
)
from tests.roll._exact_fixtures import make_stage3_replay_receipt


WORKSPACE = Path(__file__).resolve().parents[3]
ORACLE_LAB = WORKSPACE / "reverse_engineering/.work-cml-replay/lab"
GEOMETRY = {
    30: (104, 104),
    31: (104, 104),
    34: (54, 104),
    35: (54, 104),
    38: (104, 72),
    39: (104, 72),
    42: (54, 72),
    43: (54, 72),
    47: (104, 104),
    48: (104, 104),
    52: (54, 104),
    53: (54, 104),
}
STAGE1_EVENTS = {30, 34, 38, 42, 47, 52}
IDENTITY_LUT = np.arange(65_536, dtype="<u2").tobytes()


def _event(event: int, kind: str) -> np.ndarray:
    path = ORACLE_LAB / f"event{event}-{kind}.bin"
    if not path.is_file():
        pytest.skip(f"stored CML fixture is unavailable: {path}")
    width, height = GEOMETRY[event]
    raw = np.frombuffer(path.read_bytes(), dtype="<u2")
    assert raw.size == height * 104 * 3
    return raw.reshape(height, 104, 3)[:, :width, :].copy()


def _builder_receipt() -> exact_color.ValidatedBuilderReceipt:
    identity = np.frombuffer(IDENTITY_LUT, dtype="<u2")
    return make_stage3_replay_receipt((identity, identity, identity))


@pytest.mark.parametrize("event", sorted(GEOMETRY))
def test_pinned_integer_runtime_replays_each_stored_cml_event(event: int) -> None:
    source = _event(event, "source")
    expected = _event(event, "expected")
    transforms = PortableCMSOnEvaluator()._transforms  # noqa: SLF001 - oracle-stage verification
    flat_source = source.reshape(-1, 3)

    if event in STAGE1_EVENTS:
        actual = transforms.stage1(flat_source)
    else:
        actual = transforms.stage2(flat_source)

    np.testing.assert_array_equal(actual.reshape(source.shape), expected)


@pytest.mark.parametrize(
    ("stage1_event", "stage2_event"),
    [(30, 31), (34, 35), (38, 39), (42, 43), (47, 48), (52, 53)],
)
def test_adapter_replays_all_stage1_to_stage2_fixture_pairs(
    stage1_event: int,
    stage2_event: int,
) -> None:
    source = _event(stage1_event, "source")
    expected = _event(stage2_event, "expected")
    builder_receipt = _builder_receipt()

    result = PortableCMSOnEvaluator(chunk_pixels=997).evaluate(
        source,
        builder_receipt=builder_receipt,
    )

    np.testing.assert_array_equal(result.rgb, expected)
    cms = exact_color.receipt_payload(result.cms_receipt)
    assert cms["scope"] == "captured-cml4-stage1-stage2-only"
    assert cms["builder_receipt_sha256"] == builder_receipt.sha256
    assert cms["input_rgb_sha256"] == exact_color.rgb16_content_sha256(source)
    assert cms["output_rgb_sha256"] == exact_color.rgb16_content_sha256(expected)
    assert len(cms["assets"]) == 9
    assert cms["validation"] == {
        "events": 12,
        "full_payload_mismatched_bytes": 0,
        "full_payload_total_bytes": 698_880,
        "mismatched_u16": 0,
        "receipt_sha256": VALIDATION_RECEIPT_SHA256,
        "total_u16": 265_440,
    }


def test_chunk_boundaries_do_not_change_integer_results() -> None:
    source = np.random.default_rng(22).integers(0, 65_536, size=(2, 5, 3), dtype=np.uint16)
    builder_receipt = _builder_receipt()
    outputs = []

    for chunk_pixels in (1, 3, 10, 11):
        result = PortableCMSOnEvaluator(chunk_pixels=chunk_pixels).evaluate(
            source,
            builder_receipt=builder_receipt,
        )
        outputs.append(result.rgb)

    for output in outputs[1:]:
        np.testing.assert_array_equal(output, outputs[0])


@pytest.mark.parametrize("failure", ["missing", "tampered"])
def test_asset_verification_fails_closed(tmp_path: Path, failure: str) -> None:
    packaged = Path(__file__).resolve().parents[2] / "negpy/assets/portable_cms"
    copied = tmp_path / "assets"
    shutil.copytree(packaged, copied)
    target = copied / next(iter(ASSET_SHA256))
    if failure == "missing":
        target.unlink()
        expected = exact_color.ExactColorUnavailable
    else:
        payload = bytearray(target.read_bytes())
        payload[len(payload) // 2] ^= 1
        target.write_bytes(payload)
        expected = exact_color.ExactColorIntegrityError

    with pytest.raises(expected):
        PortableCMSOnEvaluator(assets_dir=copied)


def test_packaged_validation_receipt_is_exact_and_closes_the_claimed_totals() -> None:
    packaged = Path(__file__).resolve().parents[2] / "negpy/assets/portable_cms"
    payload = (packaged / VALIDATION_RECEIPT_FILENAME).read_bytes()

    assert len(payload) == VALIDATION_RECEIPT_BYTES
    assert hashlib.sha256(payload).hexdigest() == VALIDATION_RECEIPT_SHA256
    summary = portable_cms._validate_validation_receipt(payload)  # noqa: SLF001 - release provenance contract
    assert summary == {
        "events": 12,
        "mismatched_u16": 0,
        "total_u16": 265_440,
        "full_payload_mismatched_bytes": 0,
        "full_payload_total_bytes": 698_880,
    }


@pytest.mark.parametrize("failure", ["missing", "one_byte"])
def test_validation_receipt_file_failure_is_fatal(tmp_path: Path, failure: str) -> None:
    packaged = Path(__file__).resolve().parents[2] / "negpy/assets/portable_cms"
    copied = tmp_path / "assets"
    shutil.copytree(packaged, copied)
    target = copied / VALIDATION_RECEIPT_FILENAME
    if failure == "missing":
        target.unlink()
        expected = exact_color.ExactColorUnavailable
    else:
        payload = bytearray(target.read_bytes())
        payload[len(payload) // 2] ^= 1
        target.write_bytes(payload)
        expected = exact_color.ExactColorIntegrityError

    with pytest.raises(expected):
        PortableCMSOnEvaluator(assets_dir=copied)


def test_correctly_hashed_but_incomplete_validation_schema_is_fatal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    packaged = Path(__file__).resolve().parents[2] / "negpy/assets/portable_cms"
    copied = tmp_path / "assets"
    shutil.copytree(packaged, copied)
    target = copied / VALIDATION_RECEIPT_FILENAME
    document = json.loads(target.read_bytes())
    del document["stage1"]["per_event"]["30"]
    mutated = json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
    target.write_bytes(mutated)
    monkeypatch.setattr(portable_cms, "VALIDATION_RECEIPT_SHA256", hashlib.sha256(mutated).hexdigest())
    monkeypatch.setattr(portable_cms, "VALIDATION_RECEIPT_BYTES", len(mutated))

    with pytest.raises(exact_color.ExactColorIntegrityError, match="validation receipt"):
        PortableCMSOnEvaluator(assets_dir=copied)


@pytest.mark.parametrize(
    "source",
    [
        np.zeros((2, 2, 3), dtype=np.uint8),
        np.zeros((2, 2), dtype=np.uint16),
        np.zeros((2, 2, 4), dtype=np.uint16),
    ],
)
def test_adapter_rejects_wrong_input_dtype_or_shape(source: np.ndarray) -> None:
    with pytest.raises(exact_color.ExactColorIntegrityError):
        PortableCMSOnEvaluator().evaluate(source, builder_receipt=_builder_receipt())


def test_adapter_rejects_empty_input_geometry() -> None:
    source = np.zeros((0, 2, 3), dtype=np.uint16)

    with pytest.raises(exact_color.ExactColorIntegrityError):
        PortableCMSOnEvaluator().evaluate(source, builder_receipt=_builder_receipt())


def test_adapter_does_not_mutate_input_and_returns_immutable_output() -> None:
    source = np.random.default_rng(23).integers(0, 65_536, size=(2, 3, 3), dtype=np.uint16)
    original = source.copy()

    result = PortableCMSOnEvaluator(chunk_pixels=2).evaluate(
        source,
        builder_receipt=_builder_receipt(),
    )

    np.testing.assert_array_equal(source, original)
    assert not result.rgb.flags.writeable
    with pytest.raises(ValueError):
        result.rgb[0, 0, 0] ^= 1


@pytest.mark.parametrize("chunk_pixels", [0, MAX_CHUNK_PIXELS + 1, True, 1.5])
def test_chunk_size_is_bounded(chunk_pixels: object) -> None:
    with pytest.raises(exact_color.ExactColorUnavailable):
        PortableCMSOnEvaluator(chunk_pixels=chunk_pixels)  # ty: ignore[invalid-argument-type]


def test_preloaded_oracle_module_substitution_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    canonical_name = "negpy.services.roll.portable_oracle_evaluator"
    substitute = types.ModuleType(canonical_name)
    substitute.lookup_three = lambda *_args: pytest.fail("substituted oracle executed")
    monkeypatch.setitem(sys.modules, canonical_name, substitute)

    with pytest.raises(exact_color.ExactColorIntegrityError, match="preloaded"):
        PortableCMSOnEvaluator()


def test_verified_oracle_executes_isolated_without_registering_the_canonical_module() -> None:
    canonical_name = "negpy.services.roll.portable_oracle_evaluator"
    assert canonical_name not in sys.modules

    PortableCMSOnEvaluator()

    assert canonical_name not in sys.modules


def test_oracle_source_hash_mutation_is_rejected(tmp_path: Path) -> None:
    source = Path(portable_cms.__file__).with_name("portable_oracle_evaluator.py")
    copied = tmp_path / source.name
    copied.write_bytes(source.read_bytes() + b"\n# mutation\n")

    with pytest.raises(exact_color.ExactColorIntegrityError, match="source hash mismatch"):
        portable_cms._load_verified_oracle(copied)  # noqa: SLF001 - adversarial loader contract


def test_oracle_source_swap_between_lstat_and_open_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    packaged = Path(portable_cms.__file__).with_name("portable_oracle_evaluator.py")
    copied = tmp_path / packaged.name
    copied.write_bytes(packaged.read_bytes())
    replacement = tmp_path / "replacement.py"
    replacement.write_bytes(packaged.read_bytes() + b"\n# swapped\n")
    real_open = os.open
    swapped = False

    def swap_then_open(path, flags, *args, **kwargs):
        nonlocal swapped
        if not swapped and Path(path) == copied:
            swapped = True
            os.replace(replacement, copied)
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(portable_cms.os, "open", swap_then_open)

    with pytest.raises(exact_color.ExactColorIntegrityError, match="changed while being read"):
        portable_cms._load_verified_oracle(copied)  # noqa: SLF001 - adversarial loader contract
