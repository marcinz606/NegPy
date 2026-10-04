"""Merge an assembled frame into one linear TIFF negative that replaces its source files.
The buffer is the render decode (`ImageProcessor._load_source_f32`), not Linear Output's."""

import hashlib
import os
import tempfile
from dataclasses import replace
from typing import Callable, Dict, List, Optional

import numpy as np
import tifffile

from negpy.domain.models import TiffCompression, WorkspaceConfig
from negpy.features.flatfield.models import FlatFieldConfig
from negpy.features.hdr.models import HdrConfig
from negpy.features.metadata.fsdate import sync_export_filesystem_dates
from negpy.features.process.path import RenderPath, render_path
from negpy.features.stitch.models import StitchConfig
from negpy.kernel.image.logic import _to_uint16_jit
from negpy.services.export.linear_output import (
    _is_camera_raw,
    _linear_resolution,
    _read_source_meta_tiff,
    _source_format_label,
    _write_tiff,
)

#: No "hdr": a bracket's recovered detail sits below a 16-bit step.
MERGEABLE_KINDS = frozenset({"rgb", "stitch"})

MERGED_SUFFIX: Dict[str, str] = {"rgb": "_RGB", "stitch": "_STITCH"}
MERGED_EXT = ".tif"

#: Must match the labels `_source_format_label` gives a merged file.
_MERGED_FORMATS = ("camera RAW (RGB triplet)", "camera RAW (stitch ")


class MergeVerifyError(RuntimeError):
    """The file on disk does not hold the buffer that was written."""


class MergeCancelled(RuntimeError):
    """Abort was pressed; nothing was written."""


def part_files(asset: dict, kind: str) -> List[str]:
    """Every source file of *asset* but its primary, in a stable order."""
    if kind == "rgb":
        return [p for p in (asset.get("green_path"), asset.get("blue_path")) if p]
    if kind != "stitch":
        return []
    primary = asset.get("path") or ""
    paths = [*(asset.get("stitch_paths") or ())]
    for green, blue in asset.get("stitch_triplets") or ():
        if green and blue:
            paths.extend((green, blue))
    return [p for p in dict.fromkeys(paths) if p and p != primary]


def frame_files(asset: dict, kind: str) -> List[str]:
    """The sources that were frames in their own right, so can carry a `.negpy` sidecar."""
    primary = asset.get("path") or ""
    if kind == "stitch":
        return [primary, *(asset.get("stitch_paths") or ())]
    return [primary]


def can_merge(asset: dict, kind: str) -> bool:
    if kind not in MERGEABLE_KINDS:
        return False
    primary = asset.get("path") or ""
    if not primary:
        return False
    if kind == "stitch" and not registration_complete(asset):
        return False
    # Every source, not just the primary: a TIFF part is labeled plain "TIFF".
    sources = (primary, *part_files(asset, kind))
    return all(os.path.exists(p) and _is_camera_raw(p) for p in sources)


def describes_a_merge(primary_path: str, params: WorkspaceConfig) -> bool:
    """Whether the written file would carry the merge label `is_merged_source` reads back.
    A TIFF or LinearRaw DNG source is labeled for its own format, so it cannot merge."""
    return is_merge_description(_source_format_label(primary_path, params.rgbscan, params.stitch))


def is_merge_description(label: str) -> bool:
    return any(mark in label for mark in _MERGED_FORMATS)


def registration_complete(asset: dict) -> bool:
    n = 1 + len(asset.get("stitch_paths") or ())
    canvas = asset.get("stitch_canvas") or (0, 0)
    return len(asset.get("stitch_transforms") or ()) == n and len(asset.get("stitch_sizes") or ()) == n and all(canvas)


def merged_path_for(primary_path: str, kind: str, taken: frozenset = frozenset()) -> str:
    """`<primary stem><suffix>.tif` beside the primary, numbered past any existing file."""
    folder = os.path.dirname(primary_path)
    stem = os.path.splitext(os.path.basename(primary_path))[0] + MERGED_SUFFIX[kind]
    path = os.path.join(folder, stem + MERGED_EXT)
    counter = 2
    while path in taken or os.path.exists(path):
        path = os.path.join(folder, f"{stem}_{counter}{MERGED_EXT}")
        counter += 1
    return path


def decode_params(params: WorkspaceConfig, kind: str) -> WorkspaceConfig:
    """The params to decode *kind* with; `merged_edit` clears whatever this bakes.
    A stitch bakes its flat field: the registration is valid only on flat-fielded parts."""
    if kind == "rgb":
        return replace(params, flatfield=FlatFieldConfig(), stitch=StitchConfig(), hdr=HdrConfig())
    if kind == "stitch":
        return replace(params, hdr=HdrConfig())
    raise ValueError(f"Cannot merge a {kind or 'plain'} frame")


def needs_camera_matrix(params: WorkspaceConfig) -> bool:
    """Whether the frame renders through the RAW's camera color matrix, which a TIFF cannot carry."""
    return render_path(params.process) in (RenderPath.TRANSFER, RenderPath.POSITIVE)


def is_merged_source(path: str) -> bool:
    """Whether *path* is a TIFF NegPy wrote from an assembled frame."""
    if not path.lower().endswith((".tif", ".tiff")):
        return False
    try:
        with tifffile.TiffFile(path) as tif:
            description = tif.pages[0].description or ""
    except Exception:
        return False
    return description.startswith("NegPy Linear Output") and is_merge_description(description)


def _digest(u16: np.ndarray) -> str:
    return hashlib.blake2b(np.ascontiguousarray(u16).tobytes(), digest_size=16).hexdigest()


def write_merged_frame(
    f32: np.ndarray,
    primary_path: str,
    out_path: str,
    params: WorkspaceConfig,
    compression: TiffCompression = TiffCompression.ZIP,
    should_cancel: Optional[Callable[[], bool]] = None,
) -> None:
    """Write *f32* as an untagged 16-bit TIFF, verified under a temporary name, then renamed to
    *out_path*. An existing *out_path* is never replaced; *should_cancel* is polled just before the rename."""
    expected = _digest(_to_uint16_jit(np.ascontiguousarray(f32, dtype=np.float32)))
    folder = os.path.dirname(out_path) or "."
    tmp_path: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(dir=folder, delete=False, suffix=".part") as tmp:
            tmp_path = tmp.name
            _write_tiff(
                f32,
                tmp,
                os.path.basename(primary_path),
                source_path=primary_path,
                source_meta=_read_source_meta_tiff(primary_path),
                source_format=_source_format_label(primary_path, params.rgbscan, params.stitch),
                resolution=_linear_resolution(primary_path),
                compression=compression,
            )
        written = tifffile.imread(tmp_path)
        if written.dtype != np.uint16 or written.shape != f32.shape or _digest(written) != expected:
            raise MergeVerifyError(f"{os.path.basename(out_path)} did not read back as written")
        if should_cancel is not None and should_cancel():
            raise MergeCancelled(os.path.basename(out_path))
        if os.path.exists(out_path):
            raise FileExistsError(out_path)
        os.replace(tmp_path, out_path)
        tmp_path = None
    finally:
        if tmp_path is not None and os.path.exists(tmp_path):
            os.unlink(tmp_path)
    sync_export_filesystem_dates(out_path, primary_path)
