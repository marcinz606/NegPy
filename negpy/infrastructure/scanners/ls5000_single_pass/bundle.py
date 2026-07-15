"""Integrity identity for the package-owned LS-5000 capture bundle."""

from __future__ import annotations

import hashlib
import json
from importlib.resources import files
from pathlib import Path

from .continuation_plan import (
    CANONICAL_CONTINUATION_PLAN_FILENAME,
    CANONICAL_CONTINUATION_PLAN_SHA256,
    canonical_continuation_plan_bytes,
)
from .plan import CANONICAL_PLAN_SHA256, canonical_plan_bytes


DATA_PACKAGE = "negpy.infrastructure.scanners.ls5000_single_pass.data"
CANONICAL_MANIFEST_FILENAME = "replay-first-rgbi4-manifest.json"

# These hashes bind the scanner-facing implementation and both wire resources.
# Update them only after the corresponding hardware-free regression suite passes.
CAPTURE_BUNDLE_COMPONENT_SHA256 = {
    "capture_process.py": "5133c2ccb6dccccaa41d2fec11e0aecccedb96a5d2a66bcd9146adf534813e5e",
    "worker.py": "41dde02011d9c857c92de6dbc3935720dddf6cc6cae604a5960b7c8609f0a561",
    "continuation_plan.py": "bfdebfaa28075c708f3e8ef070083edce36a28b497bba622173cbb6d1466a282",
    "meter.py": "c7d00c9c8796b7264a553848106a1fe075ab4a25315fbe5a05d05bc35515ca10",
    "roll_index.py": "5cec9535322847e34aa980ace0f91f589579486464c68be643bd9910ae90d252",
    "window.py": "5edd64a2f55cb3c968bb380d548d0d9002b41b26f5f4713e5d9b889910d5ed4f",
    "data/replay-first-rgbi4-plan.jsonl": CANONICAL_PLAN_SHA256,
    f"data/{CANONICAL_CONTINUATION_PLAN_FILENAME}": (
        CANONICAL_CONTINUATION_PLAN_SHA256
    ),
    "data/replay-first-rgbi4-manifest.json": "a87faad5aa4cb458d6044cc218ab4fc13ce84f03d5355e7ea47c04c76f290e5f",
}
CAPTURE_WORKER_SHA256 = CAPTURE_BUNDLE_COMPONENT_SHA256["worker.py"]
CAPTURE_BUNDLE_SHA256 = hashlib.sha256(
    json.dumps(
        CAPTURE_BUNDLE_COMPONENT_SHA256,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
).hexdigest()


class CaptureBundleIntegrityError(RuntimeError):
    """The installed scanner capture bundle differs from its pinned identity."""


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_manifest_bytes() -> bytes:
    """Return the bundled Nikon wire manifest after validating its plan binding."""

    payload = files(DATA_PACKAGE).joinpath(CANONICAL_MANIFEST_FILENAME).read_bytes()
    expected = CAPTURE_BUNDLE_COMPONENT_SHA256[
        f"data/{CANONICAL_MANIFEST_FILENAME}"
    ]
    actual = _sha256(payload)
    if actual != expected:
        raise CaptureBundleIntegrityError(
            f"canonical capture manifest SHA-256 mismatch: expected {expected}, got {actual}"
        )
    try:
        manifest = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CaptureBundleIntegrityError(
            f"canonical capture manifest is not valid JSON: {error}"
        ) from error
    if not isinstance(manifest, dict) or manifest.get("plan_sha256") != CANONICAL_PLAN_SHA256:
        raise CaptureBundleIntegrityError(
            "canonical capture manifest is not bound to the packaged replay plan"
        )
    return payload


def verify_capture_bundle(*, require_python_sources: bool) -> str:
    """Verify package resources and, when available, every Python source file.

    Installed wheels and source checkouts expose their ``.py`` files directly.
    A frozen app executes modules from PyInstaller's signed archive, so it can
    only revalidate the separately bundled wire resources at runtime; its code
    identity is the app bundle signature plus ``CAPTURE_BUNDLE_SHA256``.
    """

    if _sha256(canonical_plan_bytes()) != CANONICAL_PLAN_SHA256:
        raise CaptureBundleIntegrityError("canonical capture plan changed")
    if (
        _sha256(canonical_continuation_plan_bytes())
        != CANONICAL_CONTINUATION_PLAN_SHA256
    ):
        raise CaptureBundleIntegrityError("canonical continuation plan changed")
    canonical_manifest_bytes()
    if require_python_sources:
        package_root = Path(__file__).resolve().parent
        for relative, expected in CAPTURE_BUNDLE_COMPONENT_SHA256.items():
            if relative.startswith("data/"):
                continue
            path = package_root / relative
            if not path.is_file():
                raise CaptureBundleIntegrityError(
                    f"capture component is not a regular file: {path}"
                )
            actual = _sha256(path.read_bytes())
            if actual != expected:
                raise CaptureBundleIntegrityError(
                    f"capture component {relative} SHA-256 mismatch: "
                    f"expected {expected}, got {actual}"
                )
    return CAPTURE_BUNDLE_SHA256


__all__ = [
    "CANONICAL_MANIFEST_FILENAME",
    "CAPTURE_BUNDLE_COMPONENT_SHA256",
    "CAPTURE_BUNDLE_SHA256",
    "CAPTURE_WORKER_SHA256",
    "CaptureBundleIntegrityError",
    "canonical_manifest_bytes",
    "verify_capture_bundle",
]
