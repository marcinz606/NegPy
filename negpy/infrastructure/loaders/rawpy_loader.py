import os
from typing import Any, ContextManager, Optional, Tuple

import numpy as np
import rawpy
import tifffile

from negpy.domain.interfaces import IImageLoader
from negpy.infrastructure.loaders.helpers import NonStandardFileWrapper, read_orientation
from negpy.infrastructure.loaders.ir_planes import find_ir_plane
from negpy.kernel.system.logging import get_logger

logger = get_logger(__name__)

# DNG PhotometricInterpretation value for LinearRaw (TIFF/EP §6.10.4).
_LINEAR_RAW = 34892


def _find_linearraw_page(tif: "tifffile.TiffFile", samples: int) -> Optional[Any]:
    """Return the page carrying `samples`-sample LinearRaw data: page 0 itself (NegPy's own
    single-IFD DNGs) or one of its SubIFDs (VueScan/SilverFast-style thumbnail + SubIFD DNGs)."""
    page0 = tif.pages[0]
    for page in (page0, *(page0.pages or [])):
        tags = getattr(page, "tags", None)
        if tags is None:
            continue
        spp_tag = tags.get("SamplesPerPixel")
        photo_tag = tags.get("PhotometricInterpretation")
        spp = int(spp_tag.value) if spp_tag is not None else 0
        photo = int(photo_tag.value) if photo_tag is not None else 0
        if spp == samples and photo == _LINEAR_RAW:
            return page
    return None


def _is_dng(file_path: str) -> bool:
    return os.path.splitext(file_path)[1].lower() == ".dng"


def _tag_floats(tag: Optional[Any]) -> Tuple[float, ...]:
    """Flatten a TIFF tag's value to floats. tifffile returns RATIONAL/SRATIONAL tags
    (dtype 5/10) as raw numerator/denominator pairs, not pre-divided values."""
    if tag is None:
        return ()
    value = tag.value
    if not isinstance(value, (tuple, list)):
        value = (value,)
    if int(tag.dtype) in (5, 10):
        return tuple(n / d if d else 0.0 for n, d in zip(value[0::2], value[1::2]))
    return tuple(float(v) for v in value)


def _broadcast3(values: Tuple[float, ...], default: float) -> np.ndarray:
    """First 3 entries of `values` as an (R, G, B) array; a single entry broadcasts to all
    three (some writers give one BlackLevel/WhiteLevel for the whole frame); empty falls
    back to `default`."""
    if len(values) >= 3:
        return np.asarray(values[:3], dtype=np.float64)
    if len(values) == 1:
        return np.full(3, values[0], dtype=np.float64)
    return np.full(3, default, dtype=np.float64)


def _peek_linear_dng_rgb(file_path: str) -> Optional[Tuple[np.ndarray, Optional[Tuple[float, float, float]]]]:
    """Decode a 3-sample LinearRaw DNG that libraw can't read (DNG 1.7 JPEG-XL from DxO
    PhotoLab/PureRAW and Lightroom Enhance, and similar) directly via tifffile/imagecodecs,
    replaying the linearization/black-white steps libraw's postprocess() would otherwise
    apply. Returns (rgb, wb_gains) as float32 [0,1] sensor-native RGB plus the as-shot white
    balance gains (R, G, B; None if AsShotNeutral is absent) — gains are not applied here,
    since NonStandardFileWrapper.postprocess() decides based on use_camera_wb, same as a
    real rawpy object would. No camera-to-XYZ color matrix: NegPy keeps every RAW source in
    sensor-native RGB (see _decode_sensor_rgb's output_color=raw), so this mirrors that
    rather than doing a color-managed decode.
    """
    if not _is_dng(file_path):
        return None
    try:
        with tifffile.TiffFile(file_path) as tif:
            page0 = tif.pages[0]
            main = _find_linearraw_page(tif, samples=3)
            if main is None:
                return None
            arr = main.asarray()  # type: ignore[attr-defined]
            if arr.ndim != 3 or arr.shape[2] != 3:
                return None

            def tag(name: str) -> Optional[Any]:
                return main.tags.get(name) or page0.tags.get(name)

            # Resolved to plain values here, while the file is still open. TiffTag.value is read
            # lazily from the file handle, and tifffile only warns, reopening the path, rather than
            # erroring if that happens after this `with` exits.
            lin_tag = tag("LinearizationTable")
            lin_table = np.asarray(lin_tag.value, dtype=np.float64) if lin_tag is not None else None
            black = _tag_floats(tag("BlackLevel"))
            white = _tag_floats(tag("WhiteLevel"))
            neutral = _tag_floats(page0.tags.get("AsShotNeutral"))
            crop_origin = _tag_floats(tag("DefaultCropOrigin"))
            crop_size = _tag_floats(tag("DefaultCropSize"))
    except Exception as e:
        logger.warning(f"Linear DNG peek failed for {file_path}: {e}")
        return None

    dtype_max = float(np.iinfo(arr.dtype).max) if np.issubdtype(arr.dtype, np.integer) else 1.0
    data = arr.astype(np.float64)

    if lin_table is not None:
        idx = np.clip(data, 0, len(lin_table) - 1).astype(np.int64)
        data = lin_table[idx]

    black3 = _broadcast3(black, 0.0)
    white3 = _broadcast3(white, dtype_max)
    data = (data - black3) / np.maximum(white3 - black3, 1e-6)
    data = np.clip(data, 0.0, 1.0)

    if len(crop_origin) >= 2 and len(crop_size) >= 2:
        ox, oy = int(round(crop_origin[0])), int(round(crop_origin[1]))
        cw, ch = int(round(crop_size[0])), int(round(crop_size[1]))
        h, w = data.shape[:2]
        if 0 <= oy < h and 0 <= ox < w and cw > 0 and ch > 0 and (cw, ch) != (w, h):
            data = data[oy : oy + ch, ox : ox + cw]

    wb_gains: Optional[Tuple[float, float, float]] = None
    if len(neutral) >= 3 and all(n > 0 for n in neutral[:3]):
        r, g, b = neutral[:3]
        wb_gains = (g / r, 1.0, g / b)

    return np.ascontiguousarray(data.astype(np.float32)), wb_gains


def _peek_linearraw_4ch(file_path: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Inspect a DNG. If it carries 4 linear samples (RGB + IR), return (rgb, ir) as float32 [0,1].

    NegPy's own `write_dng_linear` produces a single-IFD DNG; VueScan and Adobe-style DNGs
    put the full-res data in a SubIFD behind a reduced-resolution thumbnail IFD0 — both are
    checked. Returns None for camera DNGs (Bayer, 3-channel, etc.) so rawpy can handle them.
    """
    if not _is_dng(file_path):
        return None
    try:
        with tifffile.TiffFile(file_path) as tif:
            page = _find_linearraw_page(tif, samples=4)
            if page is None:
                return None
            arr = page.asarray()  # type: ignore[attr-defined]
    except Exception as e:
        logger.warning(f"DNG peek failed for {file_path}: {e}")
        return None

    if arr.ndim != 3 or arr.shape[2] != 4:
        return None

    if arr.dtype == np.uint16:
        scale = 1.0 / 65535.0
    elif arr.dtype == np.uint8:
        scale = 1.0 / 255.0
    else:
        scale = 1.0
    full = np.clip(arr.astype(np.float32) * scale, 0.0, 1.0)
    rgb = np.ascontiguousarray(full[:, :, :3])
    ir = np.ascontiguousarray(full[:, :, 3])
    return rgb, ir


def _peek_hdri_ir_page(file_path: str) -> Optional[np.ndarray]:
    """IR plane of a SilverFast HDRi DNG, or None.

    SilverFast writes the frame as a *3*-sample LinearRaw SubIFD behind a thumbnail IFD0 and
    puts the infrared record in its own full-resolution grayscale page (NewSubfileType=4) —
    the same layout as its HDRi TIFFs, so only the IR needs reading here: libraw decodes
    3-sample LinearRaw faithfully (a pass-through at `user_wb=[1,1,1,1]`), and keeping it on
    the rawpy path preserves the embedded-thumbnail splash and the half-size fast preview.
    """
    if not _is_dng(file_path):
        return None
    try:
        with tifffile.TiffFile(file_path) as tif:
            main = _find_linearraw_page(tif, samples=3)
            if main is None:
                return None
            main_h, main_w = int(main.shape[0]), int(main.shape[1])
            # Top-level pages are where SilverFast puts it, and SubIFDs are searched too for tools
            # that nest it. The main page carries 3 samples, so a 2-D dims match cannot select the
            # image itself.
            candidates = [*tif.pages, *(tif.pages[0].pages or [])]
            return find_ir_plane(candidates, main_h, main_w)
    except Exception as e:
        logger.warning(f"DNG IR-page peek failed for {file_path}: {e}")
    return None


class RawpyLoader(IImageLoader):
    """
    Standard RAW loader (libraw). For LinearRaw 4-channel DNGs (RGB + IR), bypasses
    rawpy and reads via tifffile so the IR plane is preserved. SilverFast HDRi DNGs keep
    the libraw decode and get their IR from a separate grayscale page. A 3-channel
    LinearRaw DNG libraw can't unpack (DNG 1.7 JPEG-XL from DxO PhotoLab/PureRAW and
    Lightroom Enhance: libraw's DNG-SDK-gated support isn't linked into rawpy's wheels,
    see rawpy#207) falls back to the same tifffile decode as the SilverFast case.
    """

    def load(self, file_path: str) -> Tuple[ContextManager[Any], dict]:
        peeked = _peek_linearraw_4ch(file_path)
        if peeked is not None:
            rgb, ir = peeked
            metadata = {
                "orientation": read_orientation(file_path),
                "raw_flip": 0,
                # Sensor-native linear samples; no ColorSpace names them.
                "color_space": None,
                "ir": ir,
            }
            return NonStandardFileWrapper(rgb), metadata

        if _is_dng(file_path):
            try:
                raw = rawpy.imread(file_path)
                raw.unpack()  # force now: postprocess() would hit the same error later
            except rawpy.LibRawError:
                fallback = _peek_linear_dng_rgb(file_path)
                if fallback is None:
                    raise
                rgb, wb_gains = fallback
                metadata = {
                    "orientation": read_orientation(file_path),
                    "raw_flip": 0,
                    "color_space": None,
                    "ir": None,
                }
                return NonStandardFileWrapper(rgb, wb_gains=wb_gains), metadata
        else:
            raw = rawpy.imread(file_path)

        metadata = {
            "orientation": read_orientation(file_path),
            "raw_flip": 0,
            # Decoded output_color=raw, so the file's own tags characterise nothing here.
            "color_space": None,
            "ir": _peek_hdri_ir_page(file_path),
        }

        return raw, metadata
