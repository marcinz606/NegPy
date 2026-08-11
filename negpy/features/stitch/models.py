import hashlib
import os
from dataclasses import dataclass
from typing import Sequence

from negpy.domain.tokens import composite_token


@dataclass(frozen=True)
class StitchConfig:
    """Panorama stitch: one frame assembled from overlapping part scans.

    The primary part is the asset's own file; the remaining parts ride here.
    Transforms are full-res part->canvas affines fixed once at registration
    time; decode replays them at whatever scale it works at.
    """

    stitch_enabled: bool = False
    stitch_paths: tuple[str, ...] = ()  # non-primary parts, registration order
    stitch_transforms: tuple[tuple[float, ...], ...] = ()  # per part incl. primary, 2x3 row-major
    stitch_canvas: tuple[int, int] = (0, 0)  # full-res (W, H)
    stitch_sizes: tuple[tuple[int, int], ...] = ()  # full-res decoded (W, H) per part
    stitch_triplets: tuple[tuple[str, str], ...] = ()  # RGB-scan (green, blue) per part incl. primary
    stitch_align: bool = True  # sub-pixel registration within each part's triplet


def stitch_has_triplets(config: StitchConfig) -> bool:
    """True when any part is assembled from an R/G/B exposure triplet."""
    return any(green and blue for green, blue in config.stitch_triplets)


def stitch_active(config: StitchConfig) -> bool:
    """The predicate the decode paths use to decide to assemble a composite."""
    return bool(config.stitch_enabled and config.stitch_paths)


def stitch_token(config: StitchConfig) -> str:
    """Identity of the active stitch, folded into the render source hash. Empty when inactive."""
    if not stitch_active(config):
        return ""
    triplets = [path for pair in config.stitch_triplets for path in pair]
    return composite_token("stitch", config, [*config.stitch_paths, *triplets])


def stitch_name(part_paths: Sequence[str]) -> str:
    """Display name of a composite: joined part stems, elided beyond three parts."""
    stems = [os.path.splitext(os.path.basename(p))[0] for p in part_paths]
    if len(stems) > 3:
        return f"{stems[0]} +{len(stems) - 1} (Stitch)"
    return f"{'+'.join(stems)} (Stitch)"


def stitch_hash(part_hashes: Sequence[str]) -> str:
    """Edit-key identity of a composite: content hashes of the parts, order-sensitive
    (order defines the reference frame). The ``#`` suffix follows the half-frame
    convention so ``base_hash`` strips to a plain digest."""
    digest = hashlib.sha256("|".join(part_hashes).encode()).hexdigest()
    return f"{digest}#stitch"
