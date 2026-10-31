"""A Slide of a scanner file or a finished image starts Positive: the as-captured transfer reads a
camera RAW's sensor-white decode, which a scan does not have. The user can turn it off."""

import functools
import os

from negpy.domain.models import WorkspaceConfig
from negpy.features.hdr.models import hdr_active
from negpy.features.process.models import ProcessMode, with_positive_source
from negpy.features.rgbscan.models import is_rgb_triplet
from negpy.features.stitch.models import stitch_active
from negpy.infrastructure.loaders.factory import loader_factory


@functools.lru_cache(maxsize=4096)
def _checked_scanned_positive_source(path: str) -> bool:
    return not loader_factory.is_camera_raw(path)


def is_scanned_positive_source(path: str) -> bool:
    """A TIFF, JPEG, JXL or a film scanner's own raw (Pakon, Noritsu, Coolscan NEF, Flextight
    FFF). Cached per path once the file is readable, because a Coolscan NEF or Flextight FFF is told
    apart by opening it; an unreadable file is not cached, so a volume that comes back is read again."""
    if not path:
        return False
    if not os.access(path, os.R_OK):
        return not loader_factory.is_camera_raw(path)
    return _checked_scanned_positive_source(path)


is_scanned_positive_source.cache_clear = _checked_scanned_positive_source.cache_clear


def _is_composite(config: WorkspaceConfig) -> bool:
    """A merged or assembled frame keeps its Positive choice: it renders on the Transfer path."""
    return hdr_active(config.hdr) or stitch_active(config.stitch) or is_rgb_triplet(config.rgbscan)


def with_scanner_positive_default(config: WorkspaceConfig, path: str) -> WorkspaceConfig:
    """*config* with Positive on when it is a Slide of a scanned source. Call it only where the
    frame takes its Film Mode without a Positive choice of its own: a first open, or a switch to Slide."""
    if (
        config.process.process_mode != ProcessMode.E6
        or config.process.positive_source
        or _is_composite(config)
        or not is_scanned_positive_source(path)
    ):
        return config
    return with_positive_source(config, True)
