import io
import os
import struct
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
        tifffile.imwrite(tmp_path, rgb, photometric="rgb", compression="zlib", predictor=True)
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
            tifffile.imwrite(tmp_ir, ir_data, photometric="minisblack", compression="zlib", predictor=True)
            os.replace(tmp_ir, ir_path)
        except Exception:
            if os.path.exists(tmp_ir):
                os.unlink(tmp_ir)
            raise

    return path


def write_dng_linear(result: ScanResult, path: str) -> str:
    """Write ScanResult to an uncompressed 16-bit LinearRaw DNG via tifffile.

    A LinearRaw DNG is a single-IFD TIFF plus a few DNG tags. If result.ir is
    present it is stacked as an extra sample. Atomic write; returns final path.
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
    # (code, dtype, count, value, writeonce); NewSubfileType=0 is required or LibRaw rejects the DNG.
    extratags = [
        (254, 4, 1, 0, True),  # NewSubfileType
        (50706, 1, 4, (1, 4, 0, 0), True),  # DNGVersion
        (50707, 1, 4, (1, 0, 0, 0), True),  # DNGBackwardVersion
        (274, 3, 1, 1, True),  # Orientation
        (271, 2, len(model) + 1, model, True),  # Make
        (272, 2, len(model) + 1, model, True),  # Model
    ]
    payload = _encode_dng(full_array, extratags)

    fd, tmp_path = tempfile.mkstemp(suffix=".dng", dir=os.path.dirname(path) or ".")
    os.close(fd)
    try:
        with open(tmp_path, "wb") as fh:
            fh.write(payload)
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise

    return path


def _encode_dng(full_array: np.ndarray, extratags: list) -> bytes:
    """Encode an RGB(+IR) uint16 array as LinearRaw DNG bytes.

    RGB is written with the RGB photometric so tifffile emits a clean 3 *color*
    samples with no ExtraSamples (matching pidng); the PhotometricInterpretation
    tag is then patched to LinearRaw (34892), which DNG requires. Marking colour
    planes as ExtraSamples instead makes some raw processors treat the file as a
    1-channel sensor + aux planes and mis-demosaic it.

    The IR (4-sample) case keeps the LINEAR_RAW photometric with the extra planes
    declared as extra samples — there the 4th plane genuinely is infrared, and
    tifffile has no clean 4-colour-sample form.
    """
    buf = io.BytesIO()
    if full_array.shape[-1] == 3:
        tifffile.imwrite(buf, full_array, photometric=tifffile.PHOTOMETRIC.RGB, compression=None, metadata=None, extratags=extratags)
        data = bytearray(buf.getvalue())
        with tifffile.TiffFile(io.BytesIO(bytes(data))) as tf:
            offset = tf.pages[0].tags["PhotometricInterpretation"].valueoffset
            byteorder = tf.byteorder
        struct.pack_into(byteorder + "H", data, offset, 34892)  # RGB(2) → LinearRaw(34892)
        return bytes(data)

    extrasamples = (0,) * (full_array.shape[-1] - 1)
    tifffile.imwrite(
        buf,
        full_array,
        photometric=tifffile.PHOTOMETRIC.LINEAR_RAW,
        compression=None,
        metadata=None,
        extrasamples=extrasamples,
        extratags=extratags,
    )
    return buf.getvalue()
