"""Portable application of validated Nikon Stage-1 builder artifacts.

The pre-F LUTs come from either the validated Stage-3 replay bridge or the
identity-bound native builder. This applicator composes each plane in the
recovered order ``F[B_c(i)]`` with the pinned LS5000.md3 fixed output LUT,
then applies the three composed u16 LUTs to repaired RGB in bounded chunks.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Final

import numpy as np

from negpy.services.roll import exact_color
from negpy.services.roll.exact_color import (
    BuilderReceipt,
    ExactColorIntegrityError,
    ExactColorUnavailable,
    Stage1BuilderResult,
    VerifiedBuilderApplicationReceipt,
)


ALGORITHM_ID: Final = "ls5000-md3-pref-fixed-postf-v1"
FIXED_LUT_FILENAME: Final = "ls5000-fixed-output-lut-u16le.bin"
FIXED_LUT_SHA256: Final = exact_color.FIXED_COMPOSITION_SHA256
FIXED_LUT_BYTES: Final = 65_536 * 2
DEFAULT_CHUNK_PIXELS: Final = 65_536
MAX_CHUNK_PIXELS: Final = 262_144


class PortableStage1Builder:
    """Verified, chunked pre-F → fixed post-F Stage-1 input builder."""

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
            Path(__file__).resolve().parents[2] / "assets" / "portable_builder"
            if assets_dir is None
            else Path(assets_dir).expanduser().resolve()
        )
        fixed_payload = _read_verified_file(
            self.assets_dir / FIXED_LUT_FILENAME,
            expected_sha256=FIXED_LUT_SHA256,
            expected_bytes=FIXED_LUT_BYTES,
        )
        self._fixed_lut = np.frombuffer(fixed_payload, dtype="<u2")
        self._fixed_lut.setflags(write=False)

    @property
    def chunk_pixels(self) -> int:
        return self._chunk_pixels

    def apply(
        self,
        rgb: np.ndarray,
        *,
        builder_receipt: BuilderReceipt,
    ) -> Stage1BuilderResult:
        exact_color.builder_receipt_payload(builder_receipt)
        source_hash = exact_color.rgb16_content_sha256(rgb)
        source = np.array(rgb, dtype=np.uint16, order="C", copy=True)
        source.setflags(write=False)
        flat_source = source.reshape(-1, 3)

        pre_f = [np.frombuffer(blob, dtype="<u2") for blob in builder_receipt.pre_f_luts]
        post_f = [self._fixed_lut[plane] for plane in pre_f]
        for plane in post_f:
            plane.setflags(write=False)

        output = np.empty_like(source)
        flat_output = output.reshape(-1, 3)
        for start in range(0, flat_source.shape[0], self.chunk_pixels):
            stop = min(start + self.chunk_pixels, flat_source.shape[0])
            for channel in range(3):
                flat_output[start:stop, channel] = post_f[channel][flat_source[start:stop, channel]]

        if exact_color.rgb16_content_sha256(source) != source_hash:
            raise ExactColorIntegrityError("Stage-1 builder mutated its repaired RGB input")
        stage1_hash = exact_color.rgb16_content_sha256(output)
        post_f_hashes = [hashlib.sha256(np.asarray(plane, dtype="<u2").tobytes()).hexdigest() for plane in post_f]
        application: dict[str, object] = {
            "algorithm": ALGORITHM_ID,
            "builder_receipt_sha256": builder_receipt.sha256,
            "chunk_pixels": self.chunk_pixels,
            "fixed_composition": {
                "lut_sha256": FIXED_LUT_SHA256,
                "order": "F[B_c(i)]",
            },
            "kind": "negpy.verified-stage1-builder-application",
            "post_f_lut_sha256": dict(zip(("r", "g", "b"), post_f_hashes, strict=True)),
            "pre_f_lut_sha256": dict(zip(("r", "g", "b"), builder_receipt.pre_f_lut_sha256, strict=True)),
            "source_rgb_sha256": source_hash,
            "stage1_input_rgb_sha256": stage1_hash,
            "version": 1,
        }
        if isinstance(builder_receipt, exact_color.ValidatedBuilderReceipt):
            application.update(
                native_per_acquisition_builder=False,
                scope=exact_color.STAGE3_REPLAY_SCOPE,
                stage3_receipt_sha256=builder_receipt.stage3_receipt_sha256,
            )
        elif isinstance(builder_receipt, exact_color.NativeValidatedBuilderReceipt):
            application.update(
                native_per_acquisition_builder=True,
                scope=exact_color.NATIVE_BUILDER_SCOPE,
                native_evidence_sha256=builder_receipt.evidence_sha256,
            )
        else:  # pragma: no cover - BuilderReceipt is a closed union
            raise ExactColorIntegrityError("builder receipt has an invalid type")
        application_payload = _canonical_json(application)
        application_receipt = VerifiedBuilderApplicationReceipt(
            payload=application_payload,
            sha256=hashlib.sha256(application_payload).hexdigest(),
            attested=True,
        )
        output.setflags(write=False)
        return Stage1BuilderResult(
            rgb=output,
            source_rgb_sha256=source_hash,
            stage1_input_rgb_sha256=stage1_hash,
            builder_receipt=builder_receipt,
            application_receipt=application_receipt,
        )


def _read_verified_file(
    path: Path,
    *,
    expected_sha256: str,
    expected_bytes: int,
) -> bytes:
    try:
        before = path.lstat()
        if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
            raise OSError("not a regular non-symlink file")
        with path.open("rb") as stream:
            opened = os.fstat(stream.fileno())
            payload = stream.read()
            after = os.fstat(stream.fileno())
    except OSError as error:
        raise ExactColorUnavailable(f"fixed composition LUT is unavailable: {path}: {error}") from error
    if not (_identity(before) == _identity(opened) == _identity(after)):
        raise ExactColorIntegrityError(f"fixed composition LUT changed while being read: {path}")
    if len(payload) != expected_bytes:
        raise ExactColorIntegrityError(f"fixed composition LUT byte size mismatch: {len(payload)} != {expected_bytes}")
    actual = hashlib.sha256(payload).hexdigest()
    if actual != expected_sha256:
        raise ExactColorIntegrityError(f"fixed composition LUT hash mismatch: {path}: {actual} != {expected_sha256}")
    return payload


def _identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _canonical_json(value: dict[str, object]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


__all__ = [
    "ALGORITHM_ID",
    "DEFAULT_CHUNK_PIXELS",
    "FIXED_LUT_BYTES",
    "FIXED_LUT_FILENAME",
    "FIXED_LUT_SHA256",
    "MAX_CHUNK_PIXELS",
    "PortableStage1Builder",
]
