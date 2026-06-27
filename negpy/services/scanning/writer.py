import os
import tempfile

import numpy as np
import tifffile

from negpy.infrastructure.scanners.result import ScanResult
from negpy.kernel.system.logging import get_logger

logger = get_logger(__name__)


def _to_uint16(arr: np.ndarray) -> np.ndarray:
    """Convert array to uint16. For uint8, replicate byte (x<<8 | x) so 8-bit
    values span the full 16-bit range instead of being capped at 255."""
    if arr.dtype == np.uint16:
        return arr
    if arr.dtype == np.uint8:
        a16 = arr.astype(np.uint16)
        return (a16 << 8) | a16
    return arr.astype(np.uint16)


def write_tiff_16bit(result: ScanResult, path: str) -> str:
    """Write ScanResult to 16-bit TIFF. IR written as sidecar `<basename>_IR.tif`.

    Uses atomic write (write to .tmp then rename) to avoid partial files.
    Returns final RGB path.
    """
    if not path.lower().endswith((".tif", ".tiff")):
        path = path + ".tif"

    rgb = _to_uint16(result.rgb)

    fd, tmp_path = tempfile.mkstemp(suffix=".tif", dir=os.path.dirname(path) or ".")
    os.close(fd)
    try:
        tifffile.imwrite(tmp_path, rgb, photometric="rgb", compression="lzw")
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise

    if result.ir is not None:
        base = os.path.splitext(path)[0]
        ir_path = f"{base}_IR.tif"
        ir_data = _to_uint16(result.ir)
        fd_ir, tmp_ir = tempfile.mkstemp(suffix=".tif", dir=os.path.dirname(ir_path) or ".")
        os.close(fd_ir)
        try:
            tifffile.imwrite(tmp_ir, ir_data, photometric="minisblack", compression="lzw")
            os.replace(tmp_ir, ir_path)
        except Exception:
            if os.path.exists(tmp_ir):
                os.unlink(tmp_ir)
            raise

    return path


def write_dng_linear(result: ScanResult, path: str) -> str:
    """Write ScanResult to an uncompressed 16-bit LinearRaw DNG.

    If result.ir is present, it is stacked as an extra sample channel.
    Uses atomic write (write to .tmp then rename). Returns final path.

    A LinearRaw DNG is a single-IFD TIFF plus a few DNG tags, so this is
    written with tifffile (no native deps) rather than a DNG-specific library.
    """
    if not path.lower().endswith(".dng"):
        path = path + ".dng"

    rgb = _to_uint16(result.rgb)

    if result.ir is not None:
        ir = result.ir
        if ir.ndim == 2:
            ir = ir[:, :, np.newaxis]
        ir = _to_uint16(ir)
        full_array = np.dstack([rgb, ir])
    else:
        full_array = np.ascontiguousarray(rgb)

    model = result.device_model
    # extratags: (code, dtype, count, value, writeonce). dtype 1=BYTE, 2=ASCII, 3=SHORT, 4=LONG.
    # NewSubfileType=0 marks the main IFD as the raw image — LibRaw rejects the DNG without it.
    extratags = [
        (254, 4, 1, 0, True),  # NewSubfileType = primary image
        (50706, 1, 4, (1, 4, 0, 0), True),  # DNGVersion 1.4.0.0
        (50707, 1, 4, (1, 0, 0, 0), True),  # DNGBackwardVersion 1.0.0.0
        (274, 3, 1, 1, True),  # Orientation = top-left
        (271, 2, len(model) + 1, model, True),  # Make
        (272, 2, len(model) + 1, model, True),  # Model
    ]
    # tifffile counts only the first sample for LINEAR_RAW; the rest (RGB, +IR) are
    # declared as unspecified extra samples so SamplesPerPixel comes out 3 (or 4).
    extrasamples = (0,) * (full_array.shape[-1] - 1)

    fd, tmp_path = tempfile.mkstemp(suffix=".dng", dir=os.path.dirname(path) or ".")
    os.close(fd)
    try:
        tifffile.imwrite(
            tmp_path,
            full_array,
            photometric=tifffile.PHOTOMETRIC.LINEAR_RAW,
            compression=None,
            metadata=None,  # no ImageDescription JSON — keep the DNG IFD clean
            extrasamples=extrasamples,
            extratags=extratags,
        )
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise

    return path
