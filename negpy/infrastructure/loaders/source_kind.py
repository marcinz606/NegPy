"""A Slide of a scanner file or a finished image starts Positive: the as-captured transfer reads a
camera RAW's sensor-white decode, which a scan does not have. The user can turn it off."""

import functools
import os

from negpy.domain.models import WorkspaceConfig
from negpy.features.hdr.models import hdr_active
from negpy.features.process.models import ProcessMode, with_positive_source
from negpy.features.rgbscan.models import is_rgb_triplet
from negpy.features.stitch.models import stitch_active, stitch_has_triplets
from negpy.infrastructure.loaders.factory import loader_factory
from negpy.infrastructure.loaders.helpers import is_scanned_dng


@functools.lru_cache(maxsize=4096)
def _checked_scanned_positive_source(path: str) -> bool:
    return not loader_factory.is_camera_raw(path)


def is_scanned_positive_source(path: str) -> bool:
    """A TIFF, JPEG, JXL, a film scanner's own raw (Pakon, Noritsu, Coolscan NEF, Flextight FFF) or
    a scanner DNG. Cached per path once the file is readable, because a Coolscan NEF or Flextight FFF
    is told apart by opening it; an unreadable file is not cached, so a volume that comes back is read
    again. `is_scanned_dng` caches per file revision, so a DNG read while still being written is read again."""
    if not path:
        return False
    if not os.access(path, os.R_OK):
        return not loader_factory.is_camera_raw(path)
    return _checked_scanned_positive_source(path) or is_scanned_dng(path)


is_scanned_positive_source.cache_clear = _checked_scanned_positive_source.cache_clear


def _is_trichrome(config: WorkspaceConfig) -> bool:
    """A Trichrome frame, or a stitch of them, keeps its Positive choice: each file holds one color channel."""
    return is_rgb_triplet(config.rgbscan) or stitch_has_triplets(config.stitch)


def _source_paths(config: WorkspaceConfig, path: str) -> list[str]:
    """The files the frame decodes from: every exposure of an HDR merge or part of a stitch, else *path* alone."""
    if hdr_active(config.hdr):
        return [path, *config.hdr.hdr_paths]
    if stitch_active(config.stitch):
        return [path, *config.stitch.stitch_paths]
    return [path]


def with_scanner_positive_default(config: WorkspaceConfig, path: str) -> WorkspaceConfig:
    """*config* with Positive on when it is a Slide of a scanned source. Call it only where the
    frame takes its Film Mode without a Positive choice of its own: a first open, or a switch to Slide.
    An HDR merge or a stitch starts Positive only when every file in it is a scanned source."""
    if (
        config.process.process_mode != ProcessMode.E6
        or config.process.positive_source
        or _is_trichrome(config)
        or not all(is_scanned_positive_source(p) for p in _source_paths(config, path))
    ):
        return config
    return with_positive_source(config, True)
