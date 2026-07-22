"""Minimal frozen auxiliary entry points for scanner and packaging checks.

This module intentionally imports only the standard library at module load.
The real executable dispatches here before importing the Qt desktop, so a
desktop-only dependency or startup regression cannot block the isolated
LS-5000 worker or the offline post-build smoke.
"""

from __future__ import annotations

import sys


CAPTURE_HELPER_FLAG = "--ls5000-capture-helper"
LIVE_ACCEPTANCE_FLAG = "--ls5000-live-acceptance"
PACKAGING_SMOKE_FLAG = "--negpy-packaging-smoke"
PACKAGING_SMOKE_OK = "NegPy frozen scanner packaging smoke passed"


def _require_registered_repair_engine() -> None:
    """Import the production bridge and reject a silently unavailable engine."""

    from negpy.infrastructure.roll import fauxice_bridge
    from negpy.infrastructure.roll import repair as roll_repair

    if not fauxice_bridge.engine_available() or not roll_repair.available():
        raise RuntimeError("the frozen Digital ICE repair bridge did not register an engine")


def _require_pinned_hybrid_runtime(*, loader=None) -> str:
    """Load and independently rehash the configured hybrid model weights."""

    import hashlib
    import os
    import re
    import stat
    from pathlib import Path

    if loader is None:
        from negpy.services.repair.hybrid_runtime_manifest import (
            load_default_hybrid_runtime_manifest,
        )

        loader = load_default_hybrid_runtime_manifest

    runtime = loader()
    if runtime is None:
        raise RuntimeError("the pinned hybrid runtime is not installed")
    try:
        runtime.validate_files()
    except ValueError as error:
        raise RuntimeError(f"the pinned hybrid runtime is invalid: {error}") from error

    expected = runtime.model_weights_sha256
    if not isinstance(expected, str) or re.fullmatch(r"[0-9a-f]{64}", expected) is None:
        raise RuntimeError("the hybrid model weights pin is not a lowercase SHA-256")
    try:
        weights = Path(runtime.model_weights)
        linked = os.lstat(weights)
        if stat.S_ISLNK(linked.st_mode) or not stat.S_ISREG(linked.st_mode):
            raise RuntimeError("the hybrid model weights must be a regular non-symlink file")
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(weights, flags)
        try:
            opened = os.fstat(descriptor)
            identity = (
                opened.st_dev,
                opened.st_ino,
                opened.st_size,
                opened.st_mtime_ns,
                opened.st_ctime_ns,
            )
            if not stat.S_ISREG(opened.st_mode) or identity != (
                linked.st_dev,
                linked.st_ino,
                linked.st_size,
                linked.st_mtime_ns,
                linked.st_ctime_ns,
            ):
                raise RuntimeError("the hybrid model weights changed while opening")
            digest = hashlib.sha256()
            while block := os.read(descriptor, 1024 * 1024):
                digest.update(block)
            after_read = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        after_path = os.lstat(weights)
    except RuntimeError:
        raise
    except (OSError, TypeError, ValueError) as error:
        raise RuntimeError(f"the hybrid model weights could not be verified: {error}") from error

    for metadata in (after_read, after_path):
        if (
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_size,
            metadata.st_mtime_ns,
            metadata.st_ctime_ns,
        ) != identity:
            raise RuntimeError("the hybrid model weights changed while hashing")
    actual = digest.hexdigest()
    if actual != expected:
        raise RuntimeError(f"the hybrid model weights SHA-256 does not match the pin: expected {expected}, got {actual}")
    return actual


def run_packaging_smoke_checks() -> None:
    """Validate frozen scanner/color/DICE resources without opening a device."""

    import ctypes
    import hashlib
    import importlib
    from pathlib import Path

    from llvmlite.binding import check_jit_execution

    if not getattr(sys, "frozen", False):
        raise RuntimeError("the packaging smoke is only valid in a frozen application")
    meipass = getattr(sys, "_MEIPASS", None)
    if not isinstance(meipass, str) or not meipass:
        raise RuntimeError("the frozen application has no PyInstaller resource root")
    resource_root = Path(meipass)

    # This is the exact hardened-runtime operation used by Numba's LLVM JIT:
    # allocate an anonymous RW page and transition it to RX. Keep it explicit
    # so the post-sign smoke catches missing executable-memory entitlements.
    check_jit_execution()

    from coolscanpy import (
        DigitalIceAcquisition,
        DigitalIceAcquisitionEvidence,
        build_digital_ice_acquisition_evidence,
    )
    from coolscanpy.protocol.ls5000_single_pass.bundle import verify_capture_bundle
    from coolscanpy.protocol.ls5000_single_pass.density import (
        NikonDensityEvidence,
        NikonExactBuilderEvidence,
        build_nikon_density_evidence,
        build_nikon_exact_builder_evidence,
    )
    from coolscanpy.protocol.ls5000_single_pass.usb_backend import bundled_libusb_path
    from negpy.services.repair.fauxice_hybrid_runner import (
        HybridRuntimeConfig,
        run_hybrid_repair,
    )
    from negpy.services.roll.nikon_icc import (
        NIKON_ADOBE_RGB_PROFILE_BYTES,
        NIKON_ADOBE_RGB_PROFILE_SHA256,
        nikon_adobe_rgb_profile,
    )
    from negpy.services.roll.portable_builder import PortableStage1Builder
    from negpy.services.roll.portable_cms import PortableCMSOnEvaluator

    required_callables = (
        DigitalIceAcquisition,
        DigitalIceAcquisitionEvidence,
        build_digital_ice_acquisition_evidence,
        NikonDensityEvidence,
        NikonExactBuilderEvidence,
        build_nikon_density_evidence,
        build_nikon_exact_builder_evidence,
        HybridRuntimeConfig,
        run_hybrid_repair,
    )
    if not all(callable(value) for value in required_callables):
        raise RuntimeError("a required Coolscan or DICE runtime API is unavailable")
    for module_name in (
        "portable_digital_ice.backend",
        "portable_digital_ice.fast_cpu.engine",
        "portable_digital_ice.metal_backend.engine",
    ):
        importlib.import_module(module_name)
    _require_registered_repair_engine()
    _require_pinned_hybrid_runtime()

    bundle_sha256 = verify_capture_bundle(require_python_sources=False)
    if len(bundle_sha256) != 64:
        raise RuntimeError("the packaged Coolscan capture bundle identity is invalid")

    expected_libusb = resource_root / "coolscanpy" / "_native" / "libusb-1.0.dylib"
    libusb_path = bundled_libusb_path()
    if libusb_path != expected_libusb.resolve(strict=True):
        raise RuntimeError(f"Coolscan resolved an unexpected frozen libusb: {libusb_path}")
    libusb = ctypes.CDLL(str(libusb_path))
    if not all(hasattr(libusb, symbol) for symbol in ("libusb_init", "libusb_exit")):
        raise RuntimeError("the packaged libusb is missing required entry points")

    # Constructors validate every exact LUT's size and pinned SHA-256, the
    # portable evaluator source identity, and the strict 12-event receipt.
    PortableStage1Builder()
    PortableCMSOnEvaluator()

    profile = nikon_adobe_rgb_profile()
    if len(profile) != NIKON_ADOBE_RGB_PROFILE_BYTES or hashlib.sha256(profile).hexdigest() != NIKON_ADOBE_RGB_PROFILE_SHA256:
        raise RuntimeError("the exact Nikon output ICC profile failed validation")

    expected_icc = {
        "AdobeCompat-v4.icc",
        "DisplayP3-v4.icc",
        "GrayGamma2.2.icc",
        "ProPhoto-v4.icc",
        "RGBScan.icc",
        "Rec2020-v4.icc",
        "sRGB-v4.icc",
    }
    icc_dir = resource_root / "icc"
    packaged_icc = {path.name for path in icc_dir.glob("*.icc") if path.is_file()}
    if packaged_icc != expected_icc:
        raise RuntimeError("the frozen app's ICC asset set is incomplete or unexpected")


def dispatch_packaging_smoke(argv: list[str]) -> bool:
    """Run the private post-build smoke before desktop initialization."""

    if len(argv) < 2 or argv[1] != PACKAGING_SMOKE_FLAG:
        return False
    if len(argv) != 2:
        raise RuntimeError("the packaging smoke does not accept arguments")
    run_packaging_smoke_checks()
    print(PACKAGING_SMOKE_OK, flush=True)
    return True


def dispatch_capture_helper(argv: list[str]) -> bool:
    """Run the frozen LS-5000 worker before desktop initialization."""

    if len(argv) < 2 or argv[1] != CAPTURE_HELPER_FLAG:
        return False

    from coolscanpy.protocol.ls5000_single_pass.capture_process import (
        CAPTURE_HELPER_FLAG as COOLSCAN_CAPTURE_HELPER_FLAG,
    )
    from coolscanpy.protocol.ls5000_single_pass.worker import main as worker_main

    if COOLSCAN_CAPTURE_HELPER_FLAG != CAPTURE_HELPER_FLAG:
        raise RuntimeError("NegPy and Coolscan helper flags do not match")
    worker_main(argv[2:])
    return True


def dispatch_live_acceptance(argv: list[str]) -> bool:
    """Run the one-shot LS-5000 acceptance before desktop initialization."""

    if len(argv) < 2 or argv[1] != LIVE_ACCEPTANCE_FLAG:
        return False

    from negpy.services.roll.live_acceptance import main as live_acceptance_main

    exit_code = live_acceptance_main(argv[2:])
    if exit_code:
        raise SystemExit(exit_code)
    return True


def dispatch_frozen_auxiliary(argv: list[str]) -> bool:
    """Dispatch a helper/smoke command, or leave normal desktop startup alone."""

    return dispatch_packaging_smoke(argv) or dispatch_live_acceptance(argv) or dispatch_capture_helper(argv)


__all__ = [
    "CAPTURE_HELPER_FLAG",
    "LIVE_ACCEPTANCE_FLAG",
    "PACKAGING_SMOKE_FLAG",
    "PACKAGING_SMOKE_OK",
    "dispatch_capture_helper",
    "dispatch_frozen_auxiliary",
    "dispatch_live_acceptance",
    "dispatch_packaging_smoke",
    "run_packaging_smoke_checks",
]
