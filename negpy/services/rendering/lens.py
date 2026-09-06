from typing import TYPE_CHECKING

from negpy.domain.types import ImageBuffer
from negpy.features.flatfield.logic import apply_flatfield, flatfield_token
from negpy.features.flatfield.models import FlatFieldConfig
from negpy.features.lens.logic import apply_lens
from negpy.features.lens.models import LensMetadata
from negpy.features.hdr.models import hdr_active
from negpy.features.rgbscan.models import is_rgb_triplet
from negpy.features.stitch.models import stitch_active

if TYPE_CHECKING:
    from negpy.domain.models import WorkspaceConfig


def metadata_lens_enabled(config: "WorkspaceConfig") -> bool:
    """Composite registrations refer to the unwarped component images."""
    return config.geometry.lens_from_metadata and not (
        stitch_active(config.stitch) or hdr_active(config.hdr) or is_rgb_triplet(config.rgbscan)
    )


def lens_decode_token(enabled: bool, flatfield: FlatFieldConfig) -> str:
    return "|embedded-lens-v1" + flatfield_token(flatfield) if enabled else ""


def prepare_lens_source(img: ImageBuffer, metadata: dict, flatfield: FlatFieldConfig) -> ImageBuffer:
    """Flat-field in sensor positions before a lens warp moves the samples."""
    img = apply_flatfield(img, flatfield)
    lens = metadata.get("lens_correction")
    if isinstance(lens, LensMetadata) and metadata.get("ir") is None:
        img = apply_lens(img, lens, metadata.get("orientation", 1))
    return img
