"""Reading an infrared plane out of a TIFF-structured container.

Shared by `TiffLoader` and `RawpyLoader`: SilverFast stores the iSRD infrared record as a
separate full-resolution grayscale page in both its HDRi TIFFs and its HDRi DNGs, so the
page search is one implementation with the candidate pages supplied by the caller (which
page counts as the main image differs between the two containers).
"""

from typing import Any, Iterable, Optional

import negpy_core
import numpy as np

_NEW_SUBFILE_TYPE = 254
_TRANSPARENCY_MASK = 4  # NewSubfileType bit SilverFast marks the IR page with
_MIN_IS_BLACK = 1


def normalize_ir_to_float32(ir: np.ndarray) -> np.ndarray:
    """Single-channel uint8/uint16/float ndarray → float32 in [0,1]."""
    if ir.ndim == 3:
        ir = ir[:, :, 0]
    if ir.dtype in (np.uint8, np.uint16):
        return negpy_core.int_to_unit_float32(np.ascontiguousarray(ir))
    return np.clip(ir.astype(np.float32), 0.0, 1.0)


def find_ir_plane(pages: Iterable[Any], main_h: int, main_w: int) -> Optional[np.ndarray]:
    """First page at exactly (main_h, main_w) that reads as an IR plane, normalized to [0,1].

    A page qualifies on either marker scanner stacks use: NewSubfileType=4 (transparency
    mask) or a grayscale photometric. The exact-dims match is what keeps thumbnails and
    reduced-resolution previews out, and it also implies a single sample per pixel — a
    multi-sample page's `shape` carries a third axis and can never compare equal.
    """
    for page in pages:
        if getattr(page, "shape", None) != (main_h, main_w):
            continue
        tags = getattr(page, "tags", None) or {}
        nst = tags.get(_NEW_SUBFILE_TYPE)
        photometric = getattr(page, "photometric", None)
        is_mask_page = nst is not None and nst.value == _TRANSPARENCY_MASK
        is_grayscale = photometric is not None and int(photometric) == _MIN_IS_BLACK
        if is_mask_page or is_grayscale:
            return normalize_ir_to_float32(page.asarray())
    return None
