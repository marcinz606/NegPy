import os
import tempfile

import numpy as np
import tifffile

from negpy.infrastructure.scanners.result import ScanResult
from negpy.kernel.system.logging import get_logger

logger = get_logger(__name__)

# Extension for the file an atomic write goes to before its rename, the same one export and
# the sidecar writer use. Deliberately not one a loader accepts: the output folder may be a
# watched hot folder, which indexes on extension and would take a half-written scan for a
# new asset.
_PART_SUFFIX = ".part"


def _to_uint16(arr: np.ndarray) -> np.ndarray:
    """Convert array to uint16. For uint8, replicate byte (x<<8 | x) so 8-bit
    values span the full 16-bit range instead of being capped at 255."""
    if arr.dtype == np.uint16:
        return arr
    if arr.dtype == np.uint8:
        a16 = arr.astype(np.uint16)
        return (a16 << 8) | a16
    return arr.astype(np.uint16)


def _to_grey(rgb: np.ndarray) -> np.ndarray:
    """One plane from three, by their mean.

    Film with a single record is metered with its channels locked, so the three planes are one
    density read three times and their mean is the least noisy of them. Summing in uint32 keeps
    the accumulator off the source dtype; `(s + 1) // 3` rounds rather than floors.
    """
    if rgb.ndim == 2:
        return rgb
    return ((np.sum(rgb, axis=-1, dtype=np.uint32) + 1) // 3).astype(rgb.dtype)


def write_tiff_16bit(result: ScanResult, path: str, *, mono: bool = False) -> str:
    """Write ScanResult to 16-bit TIFF. IR written as sidecar `<basename>_IR.tif`.

    `mono` writes one grey plane instead of three. Uses atomic write (write to a part file,
    then rename) to avoid partial files. Returns final RGB path.
    """
    if not path.lower().endswith((".tif", ".tiff")):
        path = path + ".tif"

    rgb = _to_uint16(result.rgb)
    photometric = "rgb"
    if mono:
        rgb = _to_grey(rgb)
        photometric = "minisblack"

    fd, tmp_path = tempfile.mkstemp(suffix=_PART_SUFFIX, dir=os.path.dirname(path) or ".")
    os.close(fd)
    try:
        tifffile.imwrite(tmp_path, rgb, photometric=photometric, compression="zlib", predictor=True)
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise

    if result.ir is not None:
        base = os.path.splitext(path)[0]
        ir_path = f"{base}_IR.tif"
        ir_data = _to_uint16(result.ir)
        fd_ir, tmp_ir = tempfile.mkstemp(suffix=_PART_SUFFIX, dir=os.path.dirname(ir_path) or ".")
        os.close(fd_ir)
        try:
            tifffile.imwrite(tmp_ir, ir_data, photometric="minisblack", compression="zlib", predictor=True)
            os.replace(tmp_ir, ir_path)
        except Exception:
            if os.path.exists(tmp_ir):
                os.unlink(tmp_ir)
            raise

    # Mask marking which IR pixels the scanner actually sampled. The loader fails closed on a
    # malformed one, so write the {0,255} the reader accepts.
    if result.ir_valid_mask is not None:
        base = os.path.splitext(path)[0]
        valid_path = f"{base}_IR_VALID.tif"
        valid_data = np.asarray(result.ir_valid_mask).astype(np.uint8) * 255
        fd_v, tmp_v = tempfile.mkstemp(suffix=_PART_SUFFIX, dir=os.path.dirname(valid_path) or ".")
        os.close(fd_v)
        try:
            tifffile.imwrite(tmp_v, valid_data, photometric="minisblack", compression="zlib", predictor=True)
            os.replace(tmp_v, valid_path)
        except Exception:
            if os.path.exists(tmp_v):
                os.unlink(tmp_v)
            raise

    return path
