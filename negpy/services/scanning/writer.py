import os
import shutil
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
    """Write ScanResult to DNG LinearRaw format.

    If result.ir is present, it is stacked as an extra sample channel.
    Uses atomic write via tempdir + rename.

    Returns final path.
    """
    from pidng.core import RAW2DNG, DNGTags
    from pidng.defs import PhotometricInterpretation
    from pidng.dng import Tag

    if not path.lower().endswith(".dng"):
        path = path + ".dng"

    rgb = _to_uint16(result.rgb)

    if result.ir is not None:
        ir = result.ir
        if ir.ndim == 2:
            ir = ir[:, :, np.newaxis]
        ir = _to_uint16(ir)
        full_array = np.dstack([rgb, ir])
        samples_per_pixel = 4
    else:
        full_array = rgb
        samples_per_pixel = 3

    h, w = full_array.shape[:2]

    tags = DNGTags()
    tags.set(Tag.ImageWidth, w)
    tags.set(Tag.ImageLength, h)
    tags.set(Tag.BitsPerSample, 16)
    tags.set(Tag.SamplesPerPixel, samples_per_pixel)
    tags.set(Tag.PhotometricInterpretation, PhotometricInterpretation.Linear_Raw)
    tags.set(Tag.Orientation, 1)
    tags.set(Tag.Make, result.device_model)
    tags.set(Tag.Model, result.device_model)

    converter = RAW2DNG()

    # Write to temp dir, then atomic rename
    target_dir = os.path.dirname(path) or "."
    tmpdir = tempfile.mkdtemp(dir=target_dir)
    tmp_basename = "tmp_" + os.path.basename(tmpdir)
    try:
        converter.options(tags, path=tmpdir, compress=False)
        written = converter.convert(full_array, filename=tmp_basename)
        # pidng appends .dng if not present
        if not os.path.exists(written) and not written.endswith(".dng"):
            written = written + ".dng"
        os.replace(written, path)
    except Exception:
        shutil.rmtree(tmpdir, ignore_errors=True)
        raise
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    return path
