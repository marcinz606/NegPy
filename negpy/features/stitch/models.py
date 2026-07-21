import hashlib
import os
from dataclasses import dataclass
from typing import Sequence


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


def stitch_token(config: StitchConfig) -> str:
    """Identity of the active stitch, folded into the render source hash. Empty when inactive."""
    if not config.stitch_enabled or not config.stitch_paths:
        return ""
    parts = []
    for path in config.stitch_paths:
        try:
            parts.append(f"{path}:{os.path.getmtime(path)}")
        except OSError:
            return ""
    geometry = repr((config.stitch_transforms, config.stitch_canvas, config.stitch_sizes))
    digest = hashlib.sha256(geometry.encode()).hexdigest()[:12]
    return f"|stitch:{':'.join(parts)}:{digest}"


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
