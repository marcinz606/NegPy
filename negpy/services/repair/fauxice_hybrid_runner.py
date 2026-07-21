"""Subprocess bridge to the optional ``fauxce-hybrid`` companion tool.

``fauxce-hybrid`` is a separate, independently optional package from the core
``portable_digital_ice`` engine (see ``fauxice_ir_repair.py``). It has no
importable "run a repair" function of its own; ``src/fauxce_hybrid/cli.py``
in the upstream project keeps that logic private and exposes only the
``fauxce-hybrid`` console script. Shelling out to that script, exactly as its
own docs show, is therefore the calling contract, not a shortcut around one.

The hybrid CLI needs the same paired 285 dpi prepass + 4000 dpi main RGBI
acquisition as the core engine (it runs the core engine internally to get the
``at_floor_mask`` it routes on), so it never relaxes the prepass requirement
described in ``fauxice_ir_repair.py``.

IOPaint and the LaMa model weights are never invoked by NegPy directly. They
run inside the fauxce-hybrid subprocess, pointed at a pinned interpreter and
hash-verified weights that ``HybridRuntimeConfig`` names but never bundles.
"""

from __future__ import annotations

import hashlib
import json
import numpy as np
import numpy.typing as npt
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import subprocess


class HybridRunError(RuntimeError):
    """The fauxce-hybrid subprocess did not produce a usable result."""


@dataclass(frozen=True)
class HybridRuntimeConfig:
    """Where the pinned IOPaint interpreter and hash-verified weights live.

    None of these paths are discovered automatically. fauxce-hybrid's own
    docs (``hybrid/docs/hybrid-repair.md`` in the digital-fauxice repository)
    require the caller to install IOPaint 1.6.0 into its own virtualenv and
    to supply the measured SHA-256 of ``big-lama.pt`` rather than trusting a
    filename; this config just carries those caller-verified values through.
    """

    iopaint_python: Path
    iopaint_executable: Path
    iopaint_source_manifest_sha256: str
    model_dir: Path
    model_weights: Path
    model_weights_sha256: str
    inpaint_device: str = "cpu"
    inpaint_threads: int = 1
    inpaint_seed: int = 0
    executable: str = "fauxce-hybrid"


@dataclass(frozen=True)
class HybridRunResult:
    """What one fauxce-hybrid subprocess call produced, read back into memory.

    The mask is carried as bytes rather than a path: the CLI's ``--out``
    directory must not already exist and is treated as scratch space by this
    module, so nothing durable survives the call except what the caller
    persists from this result.
    """

    hybrid_rgb16: npt.NDArray[np.uint16]
    synth_mask_png: bytes
    synth_mask_sha256: str
    synthesis_fraction: float | None
    engine_version: str | None
    backend_requested: str | None
    backend_used: str | None
    backend_selection_reason: str | None
    routing_counts: dict[str, int] | None


def run_hybrid_repair(
    main_rgbi: npt.NDArray[np.uint16],
    prepass_rgbi: npt.NDArray[np.uint16],
    *,
    same_frame_id: str,
    backend: str,
    runtime: HybridRuntimeConfig,
    scratch_dir: Path,
    timeout_seconds: float = 1800.0,
    runner: Callable[..., "subprocess.CompletedProcess[str]"] = subprocess.run,
) -> HybridRunResult:
    """Run one frame through the fauxce-hybrid CLI and read its outputs back.

    ``scratch_dir`` must exist and be empty; this function creates the CLI's
    ``--out`` subdirectory itself (the CLI refuses a pre-existing one). The
    caller owns ``scratch_dir``'s lifetime; nothing here deletes it, so a
    caller that wants ephemeral scratch space should wrap the call in its own
    ``tempfile.TemporaryDirectory()``.
    """

    prepass_path = scratch_dir / "prepass.rgbi16.npy"
    main_path = scratch_dir / "main.rgbi16.npy"
    out_dir = scratch_dir / "out"
    np.save(prepass_path, prepass_rgbi, allow_pickle=False)
    np.save(main_path, main_rgbi, allow_pickle=False)

    argv: Sequence[str] = [
        runtime.executable,
        "--prepass",
        str(prepass_path),
        "--main",
        str(main_path),
        "--out",
        str(out_dir),
        "--same-frame-id",
        same_frame_id,
        "--assert-focus-exposure-locked",
        "--backend",
        backend,
        "--iopaint-python",
        str(runtime.iopaint_python),
        "--iopaint-executable",
        str(runtime.iopaint_executable),
        "--iopaint-source-manifest-sha256",
        runtime.iopaint_source_manifest_sha256,
        "--model-dir",
        str(runtime.model_dir),
        "--model-weights",
        str(runtime.model_weights),
        "--model-weights-sha256",
        runtime.model_weights_sha256,
        "--inpaint-device",
        runtime.inpaint_device,
        "--inpaint-threads",
        str(runtime.inpaint_threads),
        "--inpaint-seed",
        str(runtime.inpaint_seed),
    ]
    try:
        completed = runner(list(argv), capture_output=True, text=True, timeout=timeout_seconds)
    except OSError as error:
        raise HybridRunError(f"could not launch {runtime.executable!r}: {error}") from error

    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()
        raise HybridRunError(f"fauxce-hybrid exited {completed.returncode}: {detail[-2000:]}")

    hybrid_path = out_dir / "output-hybrid.rgb16.npy"
    mask_path = out_dir / "synth-mask.png"
    receipt_path = out_dir / "hybrid-receipt.json"
    for required in (hybrid_path, mask_path, receipt_path):
        if not required.is_file():
            raise HybridRunError(f"fauxce-hybrid reported success but {required.name} is missing from {out_dir}")

    hybrid_rgb16 = np.load(hybrid_path, allow_pickle=False)
    mask_bytes = mask_path.read_bytes()
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    return HybridRunResult(
        hybrid_rgb16=hybrid_rgb16,
        synth_mask_png=mask_bytes,
        synth_mask_sha256=hashlib.sha256(mask_bytes).hexdigest(),
        **_receipt_fields(receipt),
    )


_ROUTING_COUNT_KEYS = ("final_regions", "synthesis_pixels", "frame_pixels", "at_floor_pixels")


def _receipt_fields(receipt: dict[str, Any]) -> dict[str, Any]:
    """Pull the few fields NegPy's sidecar cares about out of hybrid-receipt.json.

    Reads defensively (``.get`` all the way down): this module is a
    provenance consumer, not the receipt's verifier. ``fauxce-hybrid`` itself
    is the authority on whether a given receipt is internally consistent.

    ``routing_counts`` comes from the receipt's ``routing.counts`` object
    (``fauxce-hybrid-receipt-v2.schema.json``): the disclosed region/pixel
    counts behind the single ``synthesis_fraction`` float, e.g. "13 regions,
    16137 pixels of 22815772". ``None`` when the receipt is missing any of
    the expected keys, rather than publishing a partial count.
    """

    core = receipt.get("core", {}) if isinstance(receipt, dict) else {}
    backend = core.get("backend", {}) if isinstance(core, dict) else {}
    synthesis = receipt.get("synthesis", {}) if isinstance(receipt, dict) else {}
    routing = receipt.get("routing", {}) if isinstance(receipt, dict) else {}
    counts = routing.get("counts", {}) if isinstance(routing, dict) else {}
    routing_counts = (
        {key: counts[key] for key in _ROUTING_COUNT_KEYS}
        if isinstance(counts, dict) and all(key in counts for key in _ROUTING_COUNT_KEYS)
        else None
    )
    return {
        "synthesis_fraction": synthesis.get("fraction"),
        "engine_version": core.get("version"),
        "backend_requested": backend.get("requested"),
        "backend_used": backend.get("used"),
        "backend_selection_reason": backend.get("reason"),
        "routing_counts": routing_counts,
    }


__all__ = [
    "HybridRunError",
    "HybridRunResult",
    "HybridRuntimeConfig",
    "run_hybrid_repair",
]
