"""Dependency-neutral acceptance policy for split RGB/IR alignment."""

SPLIT_MIN_PHASE_RESPONSE = 0.10  # Every RGB channel must lock at least this hard.
SPLIT_MAX_CHANNEL_SPREAD_PX = 1.0  # R/G/B translations must agree at estimation scale.
SPLIT_MIN_ECC = 0.65  # Translation-only ECC coefficient on the mean-RGB proxy.
SPLIT_MAX_ECC_DIVERGENCE_PX = 2.0  # ECC refinement must corroborate the phase shift.
SPLIT_MAX_ECC_REFINEMENT_DELTA_PX = 0.5  # ECC may refine phase sub-pixel, never move it further.
SPLIT_MAX_HORIZONTAL_SHIFT_PX = 2.0  # Same-reservation across-film displacement.
SPLIT_MAX_VERTICAL_SHIFT_PX = 8.0  # Conservative transport-axis displacement.
SPLIT_PHASE_ONLY_MIN_RESPONSE = 0.50  # Strict fallback: every RGB phase lock.
SPLIT_PHASE_ONLY_MAX_CHANNEL_SPREAD_PX = 0.10  # Strict fallback: RGB agreement at estimate scale.
SPLIT_PHASE_ONLY_MAX_SHIFT_PX = 2.0  # Strict fallback: same-reservation output displacement.
SPLIT_TILED_PHASE_MIN_TILE_RESPONSE = 0.30  # Ignore tiles that did not lock.
SPLIT_TILED_PHASE_MIN_MEDIAN_RESPONSE = 0.40  # Per-channel supported-tile confidence.
SPLIT_TILED_PHASE_MIN_SUPPORT = 6  # At least six of the 2x4 tiles per channel.
SPLIT_TILED_PHASE_MAX_TILE_SPREAD_PX = 0.35  # Within-channel tile consensus at estimate scale.
SPLIT_TILED_PHASE_MAX_CHANNEL_SPREAD_PX = 0.15  # Cross-channel median agreement.
SPLIT_TILED_PHASE_MAX_SHIFT_PX = 2.0  # Same-reservation displacement after scaling.
SPLIT_MULTISCALE_MIN_ASPECT_RATIO = 2.0  # Short, wide windows are vulnerable to sprocket aliases.
SPLIT_MULTISCALE_LOW_MAX_DIMENSION = 1024
SPLIT_MULTISCALE_HIGH_MAX_DIMENSION = 2048
SPLIT_MULTISCALE_LOW_MIN_RESPONSE = 0.30
SPLIT_MULTISCALE_HIGH_MIN_RESPONSE = 0.40
SPLIT_MULTISCALE_MAX_CHANNEL_SHIFT_PX = 0.50  # Native/output pixels, every channel and axis.
SPLIT_MULTISCALE_MAX_CHANNEL_SPREAD_PX = 0.50  # Native/output pixels within one scale.
SPLIT_MULTISCALE_MAX_CROSS_SCALE_DELTA_PX = 0.25  # Native/output aggregate disagreement.
SPLIT_MULTISCALE_PERIODIC_ALIAS_MIN_HORIZONTAL_PX = 32.0
SPLIT_MULTISCALE_PERIODIC_ALIAS_MIN_CHANNELS = 2
SPLIT_MULTISCALE_PERIODIC_ALIAS_DOMINANCE = 4.0  # Horizontal displacement versus vertical.
SPLIT_MIN_TEXTURE_STD = 1e-4  # Full-scale dead/uniform-capture guard.
SPLIT_MIN_OVERLAP_FRACTION = 0.5  # Phase-predicted overlap required on each axis.


__all__ = [
    "SPLIT_MAX_CHANNEL_SPREAD_PX",
    "SPLIT_MAX_ECC_DIVERGENCE_PX",
    "SPLIT_MAX_ECC_REFINEMENT_DELTA_PX",
    "SPLIT_MAX_HORIZONTAL_SHIFT_PX",
    "SPLIT_MAX_VERTICAL_SHIFT_PX",
    "SPLIT_MIN_ECC",
    "SPLIT_MIN_OVERLAP_FRACTION",
    "SPLIT_MIN_PHASE_RESPONSE",
    "SPLIT_MIN_TEXTURE_STD",
    "SPLIT_MULTISCALE_HIGH_MAX_DIMENSION",
    "SPLIT_MULTISCALE_HIGH_MIN_RESPONSE",
    "SPLIT_MULTISCALE_LOW_MAX_DIMENSION",
    "SPLIT_MULTISCALE_LOW_MIN_RESPONSE",
    "SPLIT_MULTISCALE_MAX_CHANNEL_SHIFT_PX",
    "SPLIT_MULTISCALE_MAX_CHANNEL_SPREAD_PX",
    "SPLIT_MULTISCALE_MAX_CROSS_SCALE_DELTA_PX",
    "SPLIT_MULTISCALE_MIN_ASPECT_RATIO",
    "SPLIT_MULTISCALE_PERIODIC_ALIAS_DOMINANCE",
    "SPLIT_MULTISCALE_PERIODIC_ALIAS_MIN_CHANNELS",
    "SPLIT_MULTISCALE_PERIODIC_ALIAS_MIN_HORIZONTAL_PX",
    "SPLIT_PHASE_ONLY_MAX_CHANNEL_SPREAD_PX",
    "SPLIT_PHASE_ONLY_MAX_SHIFT_PX",
    "SPLIT_PHASE_ONLY_MIN_RESPONSE",
    "SPLIT_TILED_PHASE_MAX_CHANNEL_SPREAD_PX",
    "SPLIT_TILED_PHASE_MAX_SHIFT_PX",
    "SPLIT_TILED_PHASE_MAX_TILE_SPREAD_PX",
    "SPLIT_TILED_PHASE_MIN_MEDIAN_RESPONSE",
    "SPLIT_TILED_PHASE_MIN_SUPPORT",
    "SPLIT_TILED_PHASE_MIN_TILE_RESPONSE",
]
