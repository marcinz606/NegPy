"""Explicit test-only files for the production exact-color trust boundary."""

from __future__ import annotations

import json
import tempfile
from hashlib import sha256
from pathlib import Path

import numpy as np

from negpy.services.roll import exact_color


CHANNELS = ("r", "g", "b")
PRE_F_BYTES = 65_536 * 2
FIXED_ARTIFACT_BYTES = {
    "analyzer-desc.bin": 96,
    "analyzer-pixels.bin": 281 * 425 * 3 * 2,
    "builder-args.bin": 204,
    **{f"builder-control-{axis}-{channel}.bin": 256 for channel in CHANNELS for axis in ("x", "y")},
}


def write_stage3_replay_fixture(
    root: Path,
    pre_f_arrays: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> Path:
    """Write a complete real-shaped PASS JSON plus the three consumed LUTs."""

    root.mkdir(parents=True, exist_ok=True)
    pre_f_luts = tuple(np.asarray(array, dtype="<u2").tobytes() for array in pre_f_arrays)
    if any(len(blob) != PRE_F_BYTES for blob in pre_f_luts):
        raise ValueError("test pre-F arrays must each contain exactly 65,536 u16 values")
    pre_f_hashes = tuple(sha256(blob).hexdigest() for blob in pre_f_luts)
    for channel, blob in zip(CHANNELS, pre_f_luts, strict=True):
        (root / f"builder-preF-{channel}.bin").write_bytes(blob)

    artifact_sizes = {
        **FIXED_ARTIFACT_BYTES,
        **{f"builder-preF-{channel}.bin": PRE_F_BYTES for channel in CHANNELS},
        "callback-buffer.bin": 281 * 425 * 4 * 2,
        "debugger-session.log": 1_024,
        "stage3-capture-state.json": 2_048,
    }
    artifacts = []
    for name, size in artifact_sizes.items():
        digest = (
            pre_f_hashes[CHANNELS.index(name.removeprefix("builder-preF-").removesuffix(".bin"))]
            if name.startswith("builder-preF-")
            else sha256(f"test-only:{name}".encode()).hexdigest()
        )
        artifacts.append({"bytes": size, "name": name, "sha256": digest})

    report = {
        "artifacts": sorted(artifacts, key=lambda row: row["name"]),
        "capture_directory": str(root),
        "errors": [],
        "provenance": {
            "module": {
                "bytes": 1_052_672,
                "path": "C:/Program Files/Common Files/Nikon/MaidMods/Scanners/LS5000.md3",
                "sha256": "45afd6fb61a9517ff95d1896dc4257779c319310c2e8bbe75f3b4f3dada920af",
            },
            "observer_executable": {
                "bytes": 24_576,
                "path": "C:/NikonRE/ls5000_parity_capture_v6.exe",
                "sha256": sha256(b"test-only-observer-executable").hexdigest(),
            },
            "observer_source": {
                "bytes": 113_412,
                "path": "reverse_engineering/live/ls5000_parity_capture_v6.c",
                "sha256": sha256(b"test-only-observer-source").hexdigest(),
            },
            "resource": {
                "bytes": 1_024,
                "sha256": "cd934185df496f071d307ba4f96a2a2b6ac31c3c85efc62a7fa1e3216fdba70c",
                "virtual_address": "0x100ce578",
            },
        },
        "schema": exact_color.STAGE3_REPORT_SCHEMA,
        "schema_version": 1,
        "status": "pass",
        "summary": {
            "builder_controls_exact": 6,
            "builder_controls_total": 6,
            "builder_pref_channels_exact": 3,
            "builder_pref_channels_total": 3,
            "builder_pref_mismatched_u16": 0,
            "builder_pref_total_u16": 196_608,
            "builder_scalars_exact": 21,
            "builder_scalars_total": 21,
            "callback_analyzer_exact": True,
            "captured_args_replay_channels_exact": 3,
            "lifecycle_exact": True,
        },
    }
    report_path = root / "stage3-validation.json"
    report_path.write_bytes(json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())
    return report_path


def make_stage3_replay_receipt(
    pre_f_arrays: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> exact_color.ValidatedBuilderReceipt:
    with tempfile.TemporaryDirectory(prefix="negpy-stage3-test-") as directory:
        report_path = write_stage3_replay_fixture(Path(directory), pre_f_arrays)
        return exact_color.load_stage3_replay_builder_receipt(report_path)


def production_cms_payload(
    *,
    builder_receipt_sha256: str,
    input_rgb_sha256: str,
    output_rgb_sha256: str,
    chunk_pixels: int = 17,
) -> dict[str, object]:
    return {
        "algorithm": exact_color.CMS_ALGORITHM_ID,
        "assets": dict(exact_color.CMS_ASSET_SHA256),
        "builder_receipt_sha256": builder_receipt_sha256,
        "chunk_pixels": chunk_pixels,
        "dll_free": True,
        "input_rgb_sha256": input_rgb_sha256,
        "kind": exact_color.CMS_RECEIPT_KIND,
        "oracle_source": {
            "path": "portable_oracle_evaluator.py",
            "sha256": exact_color.CMS_ORACLE_SOURCE_SHA256,
        },
        "output_rgb_sha256": output_rgb_sha256,
        "scope": exact_color.CMS_SCOPE,
        "stage_order": ["stage1", "stage2"],
        "upstream_builder_included": False,
        "validation": {
            "events": 12,
            "full_payload_mismatched_bytes": 0,
            "full_payload_total_bytes": 698_880,
            "mismatched_u16": 0,
            "receipt_sha256": exact_color.CMS_VALIDATION_RECEIPT_SHA256,
            "total_u16": 265_440,
        },
        "version": exact_color.CMS_RECEIPT_VERSION,
    }
