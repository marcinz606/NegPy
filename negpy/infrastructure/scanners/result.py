from dataclasses import dataclass

import numpy as np

from negpy.infrastructure.scanners.params import ScannerCaptureState


@dataclass(frozen=True)
class SplitIrAlignment:
    """Measured transform + confidence for a split RGB/IR Coolscan capture.

    ``mode`` is "identity" when the registration proxy was byte-identical to
    the multisampled RGB, "phase-ecc" when both estimators corroborated the
    translation, "phase-only" for the strict near-zero three-channel phase
    fallback used when ECC finds a conflicting local optimum, or
    "tiled-phase" for the legacy single-scale periodic fallback,
    "multiscale-global" for a short-wide global consensus, or
    "multiscale-tiled" when a periodic global alias is rejected by stable
    local evidence at two resolutions. Confidence fields are None only in
    identity mode. New multiscale fields default empty so historical source
    checkpoints remain readable.
    """

    mode: str
    dx_px: float
    dy_px: float
    phase_responses: tuple[float, ...] = ()
    channel_spread_px: float | None = None
    ecc_coefficient: float | None = None
    tile_support_counts: tuple[int, ...] = ()
    tile_shift_spread_px: float | None = None
    estimator_version: int | None = None
    multiscale_max_dimensions: tuple[int, ...] = ()
    multiscale_channel_shifts_px: tuple[tuple[tuple[float, float], ...], ...] = ()
    multiscale_responses: tuple[tuple[float, ...], ...] = ()
    multiscale_tile_support_counts: tuple[tuple[int, ...], ...] = ()
    multiscale_tile_shift_spreads_px: tuple[tuple[float, ...], ...] = ()
    multiscale_global_alias_shifts_px: tuple[tuple[tuple[float, float], ...], ...] = ()


@dataclass(frozen=True)
class SplitSourceCapture:
    """Unmodified arrays returned by one RGB4x + RGBI1x reservation."""

    rgb4x: np.ndarray
    rgb1x_proxy: np.ndarray
    ir1x: np.ndarray


@dataclass(frozen=True)
class ScanResult:
    rgb: np.ndarray
    ir: np.ndarray | None
    dpi: int
    device_model: str
    # Present only for split Coolscan RGB+IR captures: how IR was registered
    # to the multisampled RGB, with the measured confidence that acceptance
    # tooling must be able to audit.
    ir_alignment: SplitIrAlignment | None = None
    # True where the IR sample comes from the scanner.  Split captures keep
    # the complete RGB canvas and mark warp/tail borders false instead of
    # cropping real photograph or letting zero-fill masquerade as dust.
    ir_valid_mask: np.ndarray | None = None
    # Replayable focus/exposure values actually used by the scanner.  A second
    # adjacent source window consumes this state with AF/AE disabled.
    capture_state: ScannerCaptureState | None = None
    split_source: SplitSourceCapture | None = None
