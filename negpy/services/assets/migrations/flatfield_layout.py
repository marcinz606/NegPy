"""Rewrite flat-field profiles baked in their reference's EXIF orientation into sensor layout.

A profile whose bake orientation cannot be told stays unmarked, applies as baked and is retried
every launch. Nothing raises into app startup.
"""

import os
import uuid
from typing import Dict, Optional

import numpy as np

from negpy.features.flatfield.logic import gain_token, invalidate_gain, to_sensor_layout
from negpy.infrastructure.loaders.constants import SUPPORTED_JXL_EXTENSIONS
from negpy.infrastructure.loaders.helpers import read_exif_from_file, read_orientation
from negpy.kernel.system.config import APP_CONFIG
from negpy.kernel.system.logging import get_logger
from negpy.services.assets.flatfield import SENSOR_LAYOUT

logger = get_logger(__name__)

_EXT = ".npz"
_TIFF_MARKERS = (b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+")


def _baked_orientation(source: str, gain: np.ndarray) -> Optional[int]:
    """The EXIF orientation the bake applied to *source*, or None when it cannot be told.
    A LibRaw-only tag (CR3, RAF) baked as 1, so the gain's aspect tells a quarter turn; a half
    turn or a mirror leaves no trace."""
    if os.path.splitext(source)[1].lower() in SUPPORTED_JXL_EXTENSIONS:
        return 1
    with open(source, "rb") as fh:
        tiff = fh.read(4) in _TIFF_MARKERS
    tag = read_orientation(source)
    if tag == 1 or tiff or (read_exif_from_file(source) or {}).get("0th", {}).get(274) is not None:
        return tag
    if tag not in (5, 6, 7, 8):
        return None
    import rawpy

    with rawpy.imread(source) as raw:
        sensor_portrait = raw.sizes.height > raw.sizes.width
    return tag if (gain.shape[0] > gain.shape[1]) != sensor_portrait else 1


def _migrate_file(path: str) -> None:
    with np.load(path, allow_pickle=False) as data:
        if "layout" in data.files or "gain" not in data.files:
            return
        members: Dict[str, np.ndarray] = {k: data[k] for k in data.files}
    source = str(members.get("source", ""))
    if not source or not os.path.exists(source):
        return
    try:
        orientation = _baked_orientation(source, members["gain"])
    except Exception:
        orientation = None
    if orientation is None:
        return
    gain = np.ascontiguousarray(to_sensor_layout(members["gain"], orientation), dtype=np.float32)
    members["gain"] = gain
    members["token"] = np.array(gain_token(gain))
    if "reference" in members:
        members["reference"] = to_sensor_layout(members["reference"], orientation)
    members["layout"] = np.array(SENSOR_LAYOUT)
    # Unique and without the profile extension, so a concurrent launch never shares it and the
    # profile list never shows one a crash left behind.
    temp = f"{path}.{uuid.uuid4().hex}.part"
    try:
        with open(temp, "wb") as fh:
            np.savez_compressed(fh, **members)
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.remove(temp)
    invalidate_gain(os.path.basename(path)[: -len(_EXT)])
    if orientation != 1:
        logger.info("Flat-field: %s turned to sensor layout from EXIF orientation %d", os.path.basename(path), orientation)


def migrate_flatfield_sensor_layout() -> None:
    """Turn every unmarked profile whose reference can be read into sensor layout."""
    directory = APP_CONFIG.flatfield_dir
    if not os.path.isdir(directory):
        return
    for name in sorted(os.listdir(directory)):
        if not name.endswith(_EXT):
            continue
        try:
            _migrate_file(os.path.join(directory, name))
        except Exception:
            logger.exception("Flat-field: could not move %s to sensor layout", name)
