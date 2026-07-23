"""Production adapter for the verified DLL-free two-stage CML4 evaluator.

The integer implementation is reused byte-for-byte from
'portable_oracle_evaluator.py'. This wrapper owns production concerns:
stable/hash-checked asset loading, strict validation of the packaged 12-event
zero-mismatch receipt, bounded full-frame chunking, validated builder-evidence
binding, and an immutable CMS receipt.

Scope is intentionally narrow. It evaluates only the captured Stage 1 then
Stage 2 CML transforms. It does not implement or invent the upstream
per-frame builder; callers must first apply the separately verified Stage-1
builder and present that computed RGB at this boundary.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import sys
from pathlib import Path
from types import MappingProxyType, ModuleType
from typing import Final, Mapping, cast

import numpy as np

from negpy.services.roll import exact_color
from negpy.services.roll.exact_color import (
    ExactColorIntegrityError,
    ExactColorResult,
    ExactColorUnavailable,
    BuilderReceipt,
)

ORACLE_SOURCE_SHA256: Final = exact_color.CMS_ORACLE_SOURCE_SHA256
ORACLE_VALIDATION_RECEIPT_SHA256: Final = exact_color.CMS_VALIDATION_RECEIPT_SHA256
VALIDATION_RECEIPT_FILENAME: Final = "portable-oracle-receipt.json"
VALIDATION_RECEIPT_BYTES: Final = 4_014
VALIDATION_RECEIPT_SHA256: Final = ORACLE_VALIDATION_RECEIPT_SHA256
ALGORITHM_ID: Final = exact_color.CMS_ALGORITHM_ID
DEFAULT_CHUNK_PIXELS: Final = 65_536
MAX_CHUNK_PIXELS: Final = 262_144
PACKAGED_ASSET_BYTES: Final = 2_506_760
ASSET_SHA256: Mapping[str, str] = MappingProxyType(dict(exact_color.CMS_ASSET_SHA256))
_ORACLE_MODULE_NAME = "negpy.services.roll.portable_oracle_evaluator"

_VALIDATION_EVENTS = {
    "stage1": {
        "30": (32_448, 32_448),
        "34": (16_848, 32_448),
        "38": (22_464, 22_464),
        "42": (11_664, 22_464),
        "47": (32_448, 32_448),
        "52": (16_848, 32_448),
    },
    "stage2": {
        "31": (32_448, 32_448),
        "35": (16_848, 32_448),
        "39": (22_464, 22_464),
        "43": (11_664, 22_464),
        "48": (32_448, 32_448),
        "53": (16_848, 32_448),
    },
}

_ASSET_LAYOUT = {
    "lch-atan-u16le.bin": ("<u2", (8193,)),
    "lch-sincos-i16le.bin": ("<i2", (65537, 2)),
    "lch-reciprocal-u16le.bin": ("<u2", (65537,)),
    "cml4-stage1-clut0.bin": ("<u2", (32, 32, 32, 4)),
    "cml4-stage1-input-lut0.bin": ("<u2", (3, 65536)),
    "cml4-stage1-output-lut0.bin": ("<u2", (3, 65536)),
    "cml4-stage2-clut0.bin": ("<u2", (32, 32, 32, 4)),
    "cml4-stage2-input-lut0.bin": ("<u2", (3, 65536)),
    "cml4-stage2-output-lut0.bin": ("<u2", (3, 65536)),
}


class _LoadedTransforms:
    """The oracle's exact stage order over arrays loaded from verified bytes."""

    def __init__(self, payloads: Mapping[str, bytes], oracle: ModuleType) -> None:
        arrays: dict[str, np.ndarray] = {}
        for name, (dtype, shape) in _ASSET_LAYOUT.items():
            value = np.frombuffer(payloads[name], dtype=dtype).reshape(shape)
            value.setflags(write=False)
            arrays[name] = value
        self.atan = arrays["lch-atan-u16le.bin"]
        self.sincos = arrays["lch-sincos-i16le.bin"]
        self.reciprocal = arrays["lch-reciprocal-u16le.bin"]
        self.s1_clut = arrays["cml4-stage1-clut0.bin"]
        self.s1_input = arrays["cml4-stage1-input-lut0.bin"]
        self.s1_output = arrays["cml4-stage1-output-lut0.bin"]
        self.s2_clut = arrays["cml4-stage2-clut0.bin"]
        self.s2_input = arrays["cml4-stage2-input-lut0.bin"]
        self.s2_output = arrays["cml4-stage2-output-lut0.bin"]
        self._lab_to_lch_codes = oracle.lab_to_lch_codes
        self._lch_to_lab_codes = oracle.lch_to_lab_codes
        self._lookup_three = oracle.lookup_three
        self._optimized_trilinear = oracle.optimized_trilinear

    def stage1(self, source: np.ndarray) -> np.ndarray:
        working = self._lookup_three(self.s1_input, source)
        working = self._optimized_trilinear(self.s1_clut, working)
        working = self._lab_to_lch_codes(working, self.atan)
        return self._lookup_three(self.s1_output, working)

    def stage2(self, source: np.ndarray) -> np.ndarray:
        # Preserve the recovered hue bypass: Stage-2 input LUT plane 2 is not
        # used by the active CML4 path.
        working = source.copy()
        working[:, 0] = self.s2_input[0, working[:, 0]]
        working[:, 1] = self.s2_input[1, working[:, 1]]
        working = self._lch_to_lab_codes(working, self.sincos, self.reciprocal)
        working = self._optimized_trilinear(self.s2_clut, working)
        return self._lookup_three(self.s2_output, working)


class PortableCMSOnEvaluator:
    """Chunked implementation of VerifiedPortableCMSEvaluator.

    Construction verifies the byte-identical oracle source and reads every
    required transform asset plus the exact validation receipt once through a
    stable-file check. No DLL, Wine process, scanner, VM, or external
    repository is used at runtime.
    """

    def __init__(
        self,
        *,
        assets_dir: Path | None = None,
        chunk_pixels: int = DEFAULT_CHUNK_PIXELS,
    ) -> None:
        if type(chunk_pixels) is not int or not 1 <= chunk_pixels <= MAX_CHUNK_PIXELS:
            raise ExactColorUnavailable(f"chunk_pixels must be an integer in 1..{MAX_CHUNK_PIXELS}")
        self._chunk_pixels = chunk_pixels
        self.assets_dir = (
            Path(__file__).resolve().parents[2] / "assets" / "portable_cms"
            if assets_dir is None
            else Path(assets_dir).expanduser().resolve()
        )
        validation_payload = _read_verified_payload(
            self.assets_dir / VALIDATION_RECEIPT_FILENAME,
            label="portable CMS validation receipt",
            expected_bytes=VALIDATION_RECEIPT_BYTES,
        )
        validation_digest = hashlib.sha256(validation_payload).hexdigest()
        if validation_digest != VALIDATION_RECEIPT_SHA256:
            raise ExactColorIntegrityError(
                f"portable CMS validation receipt hash mismatch: {validation_digest} != {VALIDATION_RECEIPT_SHA256}"
            )
        self._validation_summary = MappingProxyType(_validate_validation_receipt(validation_payload))
        oracle = _load_verified_oracle()
        payloads = _read_verified_assets(self.assets_dir)
        self._transforms = _LoadedTransforms(payloads, oracle)

    @property
    def chunk_pixels(self) -> int:
        return self._chunk_pixels

    def evaluate(
        self,
        rgb: np.ndarray,
        *,
        builder_receipt: BuilderReceipt,
    ) -> ExactColorResult:
        input_hash = exact_color.rgb16_content_sha256(rgb)
        exact_color.builder_receipt_payload(builder_receipt)

        source = np.array(rgb, dtype=np.uint16, order="C", copy=True)
        source.setflags(write=False)
        flat_source = source.reshape(-1, 3)
        output = np.empty_like(source)
        flat_output = output.reshape(-1, 3)
        for start in range(0, flat_source.shape[0], self.chunk_pixels):
            stop = min(start + self.chunk_pixels, flat_source.shape[0])
            stage1 = self._transforms.stage1(flat_source[start:stop])
            flat_output[start:stop] = self._transforms.stage2(stage1)

        if exact_color.rgb16_content_sha256(source) != input_hash:
            raise ExactColorIntegrityError("portable CMS evaluator mutated its Stage-1 input")
        output_hash = exact_color.rgb16_content_sha256(output)
        output.setflags(write=False)
        cms_payload = _canonical_json(
            {
                "algorithm": ALGORITHM_ID,
                "assets": dict(sorted(ASSET_SHA256.items())),
                "builder_receipt_sha256": builder_receipt.sha256,
                "chunk_pixels": self.chunk_pixels,
                "dll_free": True,
                "input_rgb_sha256": input_hash,
                "kind": exact_color.CMS_RECEIPT_KIND,
                "oracle_source": {
                    "path": "portable_oracle_evaluator.py",
                    "sha256": ORACLE_SOURCE_SHA256,
                },
                "output_rgb_sha256": output_hash,
                "scope": exact_color.CMS_SCOPE,
                "stage_order": ["stage1", "stage2"],
                "upstream_builder_included": False,
                "validation": {
                    "events": self._validation_summary["events"],
                    "full_payload_mismatched_bytes": self._validation_summary["full_payload_mismatched_bytes"],
                    "full_payload_total_bytes": self._validation_summary["full_payload_total_bytes"],
                    "mismatched_u16": self._validation_summary["mismatched_u16"],
                    "receipt_sha256": ORACLE_VALIDATION_RECEIPT_SHA256,
                    "total_u16": self._validation_summary["total_u16"],
                },
                "version": exact_color.CMS_RECEIPT_VERSION,
            }
        )
        cms_receipt = exact_color._issue_verified_cms_receipt(cms_payload)  # noqa: SLF001 - trusted adapter boundary
        return ExactColorResult(
            rgb=output,
            input_rgb_sha256=input_hash,
            output_rgb_sha256=output_hash,
            builder_receipt=builder_receipt,
            cms_receipt=cms_receipt,
        )


def _load_verified_oracle(source_path: Path | None = None) -> ModuleType:
    if _ORACLE_MODULE_NAME in sys.modules:
        raise ExactColorIntegrityError("portable CMS oracle module is preloaded; verified isolated loading is required")
    path = Path(__file__).with_name("portable_oracle_evaluator.py") if source_path is None else Path(source_path)
    payload = _read_verified_payload(path, label="portable CMS oracle source")
    if _ORACLE_MODULE_NAME in sys.modules:
        raise ExactColorIntegrityError("portable CMS oracle module was substituted during verification")
    actual = hashlib.sha256(payload).hexdigest()
    if actual != ORACLE_SOURCE_SHA256:
        raise ExactColorIntegrityError(f"portable CMS oracle source hash mismatch: {actual} != {ORACLE_SOURCE_SHA256}")
    isolated_name = f"_negpy_verified_portable_oracle_{actual[:16]}"
    oracle = ModuleType(isolated_name)
    oracle.__file__ = str(path)
    oracle.__package__ = "negpy.services.roll"
    try:
        code = compile(payload, str(path), "exec")
        exec(code, oracle.__dict__)
    except Exception as error:
        raise ExactColorIntegrityError(f"verified portable CMS oracle could not execute: {error}") from error
    if _ORACLE_MODULE_NAME in sys.modules:
        raise ExactColorIntegrityError("portable CMS oracle module was substituted during isolated execution")
    exports = ("lab_to_lch_codes", "lch_to_lab_codes", "lookup_three", "optimized_trilinear")
    if oracle.__dict__.get("ASSET_SHA256") != dict(ASSET_SHA256) or any(not callable(oracle.__dict__.get(name)) for name in exports):
        raise ExactColorIntegrityError("verified portable CMS oracle exports do not match the production contract")
    return oracle


def _read_verified_assets(directory: Path) -> Mapping[str, bytes]:
    payloads: dict[str, bytes] = {}
    for name, expected in ASSET_SHA256.items():
        path = directory / name
        dtype, shape = _ASSET_LAYOUT[name]
        expected_bytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
        payload = _read_verified_payload(
            path,
            label="portable CMS asset",
            expected_bytes=expected_bytes,
        )
        actual = hashlib.sha256(payload).hexdigest()
        if actual != expected:
            raise ExactColorIntegrityError(f"portable CMS asset hash mismatch: {path}: {actual} != {expected}")
        payloads[name] = payload
    if sum(len(payload) for payload in payloads.values()) != PACKAGED_ASSET_BYTES:
        raise ExactColorIntegrityError("portable CMS asset byte total is inconsistent")
    return MappingProxyType(payloads)


def _read_verified_payload(
    path: Path,
    *,
    label: str,
    expected_bytes: int | None = None,
) -> bytes:
    descriptor: int | None = None
    try:
        before = path.lstat()
        if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
            raise OSError("not a regular non-symlink file")
        if expected_bytes is not None and before.st_size != expected_bytes:
            raise ExactColorIntegrityError(f"{label} byte size mismatch: {before.st_size} != {expected_bytes}")
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        opened = os.fstat(descriptor)
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
        after = os.fstat(descriptor)
        final = path.lstat()
    except OSError as error:
        raise ExactColorUnavailable(f"{label} is unavailable: {path}: {error}") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
    if len({_identity(item) for item in (before, opened, after, final)}) != 1:
        raise ExactColorIntegrityError(f"{label} changed while being read: {path}")
    payload = b"".join(chunks)
    if len(payload) != before.st_size:
        raise ExactColorIntegrityError(f"{label} changed byte length while being read: {path}")
    return payload


def _validate_validation_receipt(payload: bytes) -> dict[str, int]:
    """Close the packaged zero-mismatch claim over its exact 12-event receipt."""

    try:
        document = json.loads(
            payload,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ExactColorIntegrityError(f"portable CMS validation receipt is invalid JSON: {error}") from error
    if type(document) is not dict or set(document) != {
        "all_12_events",
        "all_12_full_payloads",
        "stage1",
        "stage2",
    }:
        raise ExactColorIntegrityError("portable CMS validation receipt top-level schema is incomplete")

    stage_totals: list[int] = []
    stage_full_totals: list[int] = []
    for stage_name, expected_events in _VALIDATION_EVENTS.items():
        stage = document.get(stage_name)
        if type(stage) is not dict or set(stage) != {
            "full_payload_total",
            "per_event",
            "total",
        }:
            raise ExactColorIntegrityError(f"portable CMS validation receipt {stage_name} schema is incomplete")
        per_event = stage.get("per_event")
        if type(per_event) is not dict or set(per_event) != set(expected_events):
            raise ExactColorIntegrityError(f"portable CMS validation receipt {stage_name} event inventory is incomplete")
        event_total = 0
        event_full_total = 0
        for event, (expected_total, expected_full_total) in expected_events.items():
            row = per_event.get(event)
            if type(row) is not dict or set(row) != {
                "full_payload",
                "mae",
                "max_abs",
                "mismatched_u16",
                "total_u16",
            }:
                raise ExactColorIntegrityError(f"portable CMS validation receipt event {event} schema is incomplete")
            _require_zero_metrics(
                {key: row[key] for key in ("mae", "max_abs", "mismatched_u16", "total_u16")},
                expected_total=expected_total,
                label=f"event {event}",
            )
            full_payload = row.get("full_payload")
            _require_zero_metrics(
                full_payload,
                expected_total=expected_full_total,
                label=f"event {event} full payload",
            )
            event_total += expected_total
            event_full_total += expected_full_total
        _require_zero_metrics(
            stage.get("total"),
            expected_total=event_total,
            label=f"{stage_name} total",
        )
        _require_zero_metrics(
            stage.get("full_payload_total"),
            expected_total=event_full_total,
            label=f"{stage_name} full-payload total",
        )
        stage_totals.append(event_total)
        stage_full_totals.append(event_full_total)

    total_u16 = sum(stage_totals)
    full_total_u16 = sum(stage_full_totals)
    _require_zero_metrics(
        document.get("all_12_events"),
        expected_total=total_u16,
        label="all-12-event total",
    )
    full_payload = document.get("all_12_full_payloads")
    if type(full_payload) is not dict or set(full_payload) != {
        "mae",
        "max_abs",
        "mismatched_bytes",
        "mismatched_u16",
        "total_bytes",
        "total_u16",
    }:
        raise ExactColorIntegrityError("portable CMS validation receipt full-payload schema is incomplete")
    _require_zero_metrics(
        {key: full_payload[key] for key in ("mae", "max_abs", "mismatched_u16", "total_u16")},
        expected_total=full_total_u16,
        label="all-12 full-payload total",
    )
    if (
        type(full_payload.get("total_bytes")) is not int
        or full_payload.get("total_bytes") != full_total_u16 * 2
        or type(full_payload.get("mismatched_bytes")) is not int
        or full_payload.get("mismatched_bytes") != 0
    ):
        raise ExactColorIntegrityError("portable CMS validation receipt byte totals are inconsistent")
    return {
        "events": sum(len(events) for events in _VALIDATION_EVENTS.values()),
        "mismatched_u16": 0,
        "total_u16": total_u16,
        "full_payload_mismatched_bytes": 0,
        "full_payload_total_bytes": full_total_u16 * 2,
    }


def _require_zero_metrics(
    value: object,
    *,
    expected_total: int,
    label: str,
) -> None:
    if type(value) is not dict:
        raise ExactColorIntegrityError(f"portable CMS validation receipt {label} is not an exact zero-mismatch result")
    metrics = cast(dict[str, object], value)
    if (
        set(metrics) != {"mae", "max_abs", "mismatched_u16", "total_u16"}
        or type(metrics.get("mismatched_u16")) is not int
        or metrics.get("mismatched_u16") != 0
        or type(metrics.get("total_u16")) is not int
        or metrics.get("total_u16") != expected_total
        or type(metrics.get("max_abs")) is not int
        or metrics.get("max_abs") != 0
        or type(metrics.get("mae")) is not float
        or metrics.get("mae") != 0.0
    ):
        raise ExactColorIntegrityError(f"portable CMS validation receipt {label} is not an exact zero-mismatch result")


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    parsed: dict[str, object] = {}
    for key, value in pairs:
        if key in parsed:
            raise ValueError(f"duplicate key {key!r}")
        parsed[key] = value
    return parsed


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant {value!r}")


def _identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _canonical_json(value: dict) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


__all__ = [
    "ALGORITHM_ID",
    "ASSET_SHA256",
    "DEFAULT_CHUNK_PIXELS",
    "MAX_CHUNK_PIXELS",
    "ORACLE_SOURCE_SHA256",
    "ORACLE_VALIDATION_RECEIPT_SHA256",
    "PACKAGED_ASSET_BYTES",
    "PortableCMSOnEvaluator",
    "VALIDATION_RECEIPT_BYTES",
    "VALIDATION_RECEIPT_FILENAME",
    "VALIDATION_RECEIPT_SHA256",
]
