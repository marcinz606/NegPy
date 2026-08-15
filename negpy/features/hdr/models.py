import hashlib
import os
from dataclasses import dataclass, replace
from typing import Sequence

from negpy.domain.tokens import composite_token


#: Sentinel for "the render exposure was not set as a value". See HdrConfig.hdr_anchor_ev.
ANCHOR_EV_UNSET = 1.0


@dataclass(frozen=True)
class HdrConfig:
    """Bracketed exposures of one frame, merged into a single linear source.

    The primary frame is the asset's own file and is the exposure *reference*: the merge
    expresses radiance in its units, so it is the frame that defines white. The remaining
    exposures ride here the way stitch parts do. Ratios are solved once at merge time and
    stored, so a reopened edit merges identically.

    Every field is `hdr_`-prefixed on purpose. `WorkspaceConfig.to_dict` flattens all
    sub-configs into one namespace and RgbScanConfig already owns the bare `enabled` and
    `align`; a duplicate name would silently clobber.
    """

    hdr_enabled: bool = False
    hdr_paths: tuple[str, ...] = ()  # non-reference exposures
    hdr_ratios: tuple[float, ...] = ()  # per frame incl. the reference, which is 1.0
    hdr_align: bool = True  # sub-pixel registration of each frame to the reference
    #: Path of the frame the merge renders at. The reference defines *white*, and this decides
    #: which exposure the picture opens at. They are not the same choice: see
    #: `logic.output_scale`. Empty falls back to the bracket's middle exposure.
    hdr_anchor: str = ""
    #: Render exposure in stops below the reference, when the user set it as a value rather
    #: than by naming a frame. Takes precedence over `hdr_anchor`.
    #:
    #: Any value at or above ANCHOR_EV_UNSET means "not set": `output_scale` clamps the scale
    #: to 1.0 or less, so a positive EV renders identically to 0 and is unreachable as a
    #: setting. That leaves the whole usable range, 0 and below, free to mean what it says,
    #: which a 0.0 sentinel would not, since 0 EV is a real choice distinct from the default.
    hdr_anchor_ev: float = 1.0


def hdr_active(config: HdrConfig) -> bool:
    """The predicate the decode paths use to decide to merge a bracket."""
    return bool(config.hdr_enabled and config.hdr_paths)


def hdr_merge_token(config: HdrConfig) -> str:
    """Identity of the *merge itself*, excluding the exposure it renders at.

    The merge costs seconds; the render exposure is one multiply on its result. Keyed apart
    so dragging the exposure re-uses the merged buffer instead of decoding the bracket
    again. `hdr_token` still carries the exposure, because a buffer that has been scaled is
    a different buffer.
    """
    if not hdr_active(config):
        return ""
    neutral = replace(config, hdr_anchor="", hdr_anchor_ev=ANCHOR_EV_UNSET)
    return composite_token("hdrmerge", neutral, config.hdr_paths)


def hdr_token(config: HdrConfig) -> str:
    """Identity of the active bracket, folded into the render source hash. Empty when inactive."""
    if not hdr_active(config):
        return ""
    return composite_token("hdr", config, config.hdr_paths)


def hdr_stem(frame_paths: Sequence[str]) -> str:
    """The bracket's naming stem: its alphabetically first frame.

    Not the reference frame's. The reference is whichever exposure the *images* say is
    longest-without-clipping, so naming after it makes the output name depend on picture
    content — a bracket of _DSC1715..19 exporting as "_DSC1716" reads as arbitrary. The
    first frame in filename order is stable, and matches how a stitch composite is named.
    """
    if not frame_paths:
        return ""
    return os.path.splitext(os.path.basename(min(frame_paths, key=lambda p: os.path.basename(p).lower())))[0]


def hdr_name(frame_paths: Sequence[str]) -> str:
    """Display name of a merge: the naming stem plus the extra frame count."""
    if not frame_paths:
        return "(HDR)"
    return f"{hdr_stem(frame_paths)} +{len(frame_paths) - 1} (HDR)"


def hdr_frame_paths(asset: dict) -> list:
    """Every frame of a merged asset, reference first. Empty for a non-merged asset."""
    others = asset.get("hdr_paths") or ()
    return [asset["path"], *others] if others else []


def hdr_hash(frame_hashes: Sequence[str]) -> str:
    """Edit-key identity of a merge: content hashes of the frames, order-sensitive (order
    defines which frame is the exposure reference). The ``#`` suffix follows the half-frame
    convention so ``base_hash`` strips it to a plain digest."""
    digest = hashlib.sha256("|".join(frame_hashes).encode()).hexdigest()
    return f"{digest}#hdr"
