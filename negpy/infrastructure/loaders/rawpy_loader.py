import os
from collections.abc import Callable
from typing import Any, ContextManager, Optional, Tuple

import cv2
import numpy as np
import rawpy
import tifffile
from PIL import Image

from negpy.domain.interfaces import IImageLoader
from negpy.infrastructure.loaders.helpers import (
    NonStandardFileWrapper,
    dng_bounded_preview,
    dng_quick_preview,
    embedded_preview,
    fit_bounded_preview,
    read_orientation,
)
from negpy.infrastructure.loaders.ir_planes import find_ir_plane
from negpy.infrastructure.loaders.memory import PreviewMemoryEstimate
from negpy.kernel.system.logging import get_logger

logger = get_logger(__name__)

# DNG PhotometricInterpretation value for LinearRaw (TIFF/EP §6.10.4).
_LINEAR_RAW = 34892
_JPEG_XL_COMPRESSIONS = {50002, 52546}
_PREVIEW_SEGMENT_MAX_BYTES = 64 * 1024 * 1024
_EMBEDDED_PREVIEW_MAX_BYTES = 64 * 1024 * 1024
_NORMALIZED_SEGMENT_WORKING_MULTIPLIER = 16
_CONVERTED_SEGMENT_WORKING_MULTIPLIER = 6


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


def _stream_linearraw_preview(
    page: Any,
    page0: Any,
    max_edge: int,
    *,
    normalize_tags: bool,
    min_preserve_ir: bool = False,
    should_cancel: Optional[Callable[[], bool]] = None,
) -> Optional[Tuple[np.ndarray, Tuple[int, int]]]:
    """Decode LinearRaw segments into a preview-size float32 buffer."""
    shape = tuple(int(v) for v in page.shape)
    if not _linearraw_page_is_streamable(page):
        return None
    height, width, samples = shape
    if should_cancel is not None and should_cancel():
        raise InterruptedError("preview load cancelled")

    scale = min(1.0, max(1, int(max_edge)) / max(height, width))
    out_height = max(1, int(round(height * scale)))
    out_width = max(1, int(round(width * scale)))
    output = np.zeros((out_height, out_width, samples), dtype=np.float32)

    def tag(name: str) -> Optional[Any]:
        return page.tags.get(name) or page0.tags.get(name)

    dtype_max = float(np.iinfo(page.dtype).max)
    linearization_tag = tag("LinearizationTable") if normalize_tags else None
    linearization = np.asarray(linearization_tag.value, dtype=np.float32) if linearization_tag is not None else None
    black = _broadcast3(_tag_floats(tag("BlackLevel")), 0.0).astype(np.float32).reshape(1, 1, 3)
    white = _broadcast3(_tag_floats(tag("WhiteLevel")), dtype_max).astype(np.float32).reshape(1, 1, 3)
    ir_kernel = None
    if min_preserve_ir and samples == 4 and scale < 1.0:
        size = max(1, int(round(1.0 / scale)) | 1)
        if size > 1:
            ir_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))

    for decoded, position, _shape in page.segments(maxworkers=1):
        if should_cancel is not None and should_cancel():
            raise InterruptedError("preview load cancelled")
        if decoded is None:
            continue
        tile = decoded[0] if decoded.ndim == 4 else decoded
        if tile.ndim != 3 or tile.shape[2] < samples:
            return None
        y, x = int(position[2]), int(position[3])
        valid_height = min(tile.shape[0], height - y)
        valid_width = min(tile.shape[1], width - x)
        if valid_height <= 0 or valid_width <= 0:
            continue
        source = tile[:valid_height, :valid_width, :samples]
        if normalize_tags:
            source_rgb = source[:, :, :3]
            if linearization is not None:
                indices = np.clip(source_rgb, 0, linearization.size - 1).astype(np.int32)
                data = linearization[indices]
            else:
                data = source_rgb.astype(np.float32)
            data = np.clip((data - black) / np.maximum(white - black, 1e-6), 0.0, 1.0)
        else:
            data = source.astype(np.float32) / dtype_max

        left = int(round(x * out_width / width))
        top = int(round(y * out_height / height))
        right = int(round((x + valid_width) * out_width / width))
        bottom = int(round((y + valid_height) * out_height / height))
        if right <= left or bottom <= top:
            continue
        target = (right - left, bottom - top)
        if min_preserve_ir and samples == 4:
            output[top:bottom, left:right, :3] = cv2.resize(data[:, :, :3], target, interpolation=cv2.INTER_AREA)
            ir = data[:, :, 3]
            if ir_kernel is not None:
                # A boundary defect needs its inward footprint without decoding adjacent segments.
                ir = cv2.erode(ir, ir_kernel, borderType=cv2.BORDER_REPLICATE)
            output[top:bottom, left:right, 3] = cv2.resize(ir, target, interpolation=cv2.INTER_AREA)
        else:
            output[top:bottom, left:right] = cv2.resize(data, target, interpolation=cv2.INTER_AREA)

    full_dims = (height, width)
    if normalize_tags:
        crop_origin = _tag_floats(tag("DefaultCropOrigin"))
        crop_size = _tag_floats(tag("DefaultCropSize"))
        if len(crop_origin) >= 2 and len(crop_size) >= 2:
            ox, oy = crop_origin[:2]
            crop_width, crop_height = crop_size[:2]
            box = (
                max(0, int(round(ox * out_width / width))),
                max(0, int(round(oy * out_height / height))),
                min(out_width, int(round((ox + crop_width) * out_width / width))),
                min(out_height, int(round((oy + crop_height) * out_height / height))),
            )
            if box[2] > box[0] and box[3] > box[1]:
                output = output[box[1] : box[3], box[0] : box[2]]
                full_dims = (int(round(crop_height)), int(round(crop_width)))
    return np.ascontiguousarray(output), full_dims


def _linearraw_page_is_streamable(page: Any) -> bool:
    segment_bytes = _linearraw_segment_bytes(page)
    return segment_bytes is not None and segment_bytes <= _PREVIEW_SEGMENT_MAX_BYTES


def _linearraw_segment_bytes(page: Any) -> Optional[int]:
    shape = tuple(int(value) for value in page.shape)
    if len(shape) != 3 or shape[2] not in (3, 4) or page.dtype not in (np.uint8, np.uint16):
        return None
    height, width, samples = shape
    segment_height = int(page.tilelength if page.is_tiled else page.rowsperstrip or height)
    segment_width = int(page.tilewidth if page.is_tiled else width)
    return segment_height * segment_width * samples * int(np.dtype(page.dtype).itemsize)


def _streaming_linearraw_memory_estimate(
    page: Any,
    page0: Any,
    max_edge: int,
    *,
    normalize_tags: bool,
) -> Optional[PreviewMemoryEstimate]:
    """Estimate peak memory for the segmented LinearRaw preview path."""
    segment_bytes = _linearraw_segment_bytes(page)
    if segment_bytes is None or segment_bytes > _PREVIEW_SEGMENT_MAX_BYTES:
        return None

    height, width, samples = (int(value) for value in page.shape)
    scale = min(1.0, max(1, int(max_edge)) / max(height, width))
    preview_width = max(1, int(round(width * scale)))
    preview_height = max(1, int(round(height * scale)))
    cached_bytes = preview_width * preview_height * samples * np.dtype(np.float32).itemsize

    linearization_bytes = 0
    if normalize_tags:
        linearization_tag = page.tags.get("LinearizationTable") or page0.tags.get("LinearizationTable")
        if linearization_tag is not None:
            linearization_bytes = max(0, int(getattr(linearization_tag, "count", 0))) * np.dtype(np.float32).itemsize
    segment_multiplier = _NORMALIZED_SEGMENT_WORKING_MULTIPLIER if normalize_tags else _CONVERTED_SEGMENT_WORKING_MULTIPLIER
    decode_working_bytes = segment_bytes * segment_multiplier + cached_bytes + linearization_bytes
    downstream_working_bytes = cached_bytes * 3 + linearization_bytes
    return PreviewMemoryEstimate(
        cached_bytes=int(cached_bytes),
        temporary_bytes=int(max(decode_working_bytes, downstream_working_bytes)),
        source_dimensions=(width, height),
    )


def _peek_linear_dng_rgb_preview(
    file_path: str,
    max_edge: int,
    should_cancel: Optional[Callable[[], bool]] = None,
) -> tuple[bool, Optional[Tuple[np.ndarray, Optional[Tuple[float, float, float]], Tuple[int, int]]]]:
    """Read a JPEG XL LinearRaw DNG directly into a bounded RGB preview."""
    if not _is_dng(file_path):
        return False, None
    handled = False
    try:
        with tifffile.TiffFile(file_path) as tif:
            page0 = tif.pages[0]
            main = _find_linearraw_page(tif, samples=3)
            if main is None or int(main.compression) not in _JPEG_XL_COMPRESSIONS:
                return False, None
            handled = True
            streamed = _stream_linearraw_preview(
                main,
                page0,
                max_edge,
                normalize_tags=True,
                should_cancel=should_cancel,
            )
            neutral = _tag_floats(page0.tags.get("AsShotNeutral"))
    except InterruptedError:
        raise
    except Exception as e:
        logger.warning(f"Linear DNG preview failed for {file_path}: {e}")
        return handled, None
    if streamed is None:
        return True, None
    rgb, full_dims = streamed
    wb_gains: Optional[Tuple[float, float, float]] = None
    if len(neutral) >= 3 and all(n > 0 for n in neutral[:3]):
        red, green, blue = neutral[:3]
        wb_gains = (green / red, 1.0, green / blue)
    return True, (rgb, wb_gains, full_dims)


def _peek_linearraw_4ch_preview(
    file_path: str,
    max_edge: int,
    should_cancel: Optional[Callable[[], bool]] = None,
) -> tuple[bool, Optional[Tuple[np.ndarray, np.ndarray, Tuple[int, int]]]]:
    """Read a four-sample LinearRaw DNG directly into bounded RGB and IR previews."""
    if not _is_dng(file_path):
        return False, None
    handled = False
    try:
        with tifffile.TiffFile(file_path) as tif:
            page0 = tif.pages[0]
            page = _find_linearraw_page(tif, samples=4)
            if page is None:
                return False, None
            handled = True
            streamed = _stream_linearraw_preview(
                page,
                page0,
                max_edge,
                normalize_tags=False,
                min_preserve_ir=True,
                should_cancel=should_cancel,
            )
    except InterruptedError:
        raise
    except Exception as e:
        logger.warning(f"DNG RGB+IR preview failed for {file_path}: {e}")
        return handled, None
    if streamed is None:
        return True, None
    full, full_dims = streamed
    return True, (np.ascontiguousarray(full[:, :, :3]), np.ascontiguousarray(full[:, :, 3]), full_dims)


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
    black3 = _broadcast3(black, 0.0)
    white3 = _broadcast3(white, dtype_max)
    denominator = np.maximum(white3 - black3, 1e-6)

    if len(crop_origin) >= 2 and len(crop_size) >= 2:
        ox, oy = int(round(crop_origin[0])), int(round(crop_origin[1]))
        cw, ch = int(round(crop_size[0])), int(round(crop_size[1]))
        h, w = arr.shape[:2]
        if 0 <= oy < h and 0 <= ox < w and cw > 0 and ch > 0 and (cw, ch) != (w, h):
            arr = arr[oy : oy + ch, ox : ox + cw]

    data = np.empty(arr.shape, dtype=np.float32)
    # Keep float64 arithmetic, but bound temporary storage to one row block.
    rows = max(1, (8 * 1024 * 1024) // (arr.shape[1] * 3 * 8))
    for start in range(0, arr.shape[0], rows):
        block = arr[start : start + rows].astype(np.float64)
        if lin_table is not None:
            np.clip(block, 0, len(lin_table) - 1, out=block)
            block = lin_table[block.astype(np.int64)]
        np.subtract(block, black3, out=block)
        np.divide(block, denominator, out=block)
        np.clip(block, 0.0, 1.0, out=block)
        data[start : start + rows] = block

    wb_gains: Optional[Tuple[float, float, float]] = None
    if len(neutral) >= 3 and all(n > 0 for n in neutral[:3]):
        r, g, b = neutral[:3]
        wb_gains = (g / r, 1.0, g / b)

    return data, wb_gains


def _peek_linearraw_4ch(file_path: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Inspect a DNG. If it carries 4 linear samples (RGB + IR), return (rgb, ir) as float32 [0,1].

    Checks IFD0 and, for VueScan and Adobe-style DNGs, a SubIFD behind a thumbnail IFD0.
    Returns None for camera DNGs (Bayer, 3-channel, etc.) so rawpy can handle them.
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
    the libraw decode and get their IR from a separate grayscale page. JPEG XL LinearRaw
    previews stream through tifffile. Full-resolution 3-channel loads fall back to a
    tag-aware tifffile decode when libraw cannot unpack the file.
    """

    @staticmethod
    def supports_cancellable_linear_preview(file_path: str) -> bool:
        """Return whether preview decode is bounded and checks cancellation between segments."""
        return RawpyLoader.estimate_cancellable_linear_preview_memory(file_path, 1) is not None

    @staticmethod
    def estimate_cancellable_linear_preview_memory(file_path: str, max_edge: int) -> Optional[PreviewMemoryEstimate]:
        """Estimate a segmented LinearRaw preview from TIFF metadata only."""
        if not _is_dng(file_path):
            return None
        try:
            with tifffile.TiffFile(file_path) as tif:
                page0 = tif.pages[0]
                page_4ch = _find_linearraw_page(tif, samples=4)
                if page_4ch is not None:
                    return _streaming_linearraw_memory_estimate(page_4ch, page0, max_edge, normalize_tags=False)
                page_3ch = _find_linearraw_page(tif, samples=3)
                if page_3ch is None or int(page_3ch.compression) not in _JPEG_XL_COMPRESSIONS:
                    return None
                return _streaming_linearraw_memory_estimate(
                    page_3ch,
                    page0,
                    max_edge,
                    normalize_tags=True,
                )
        except Exception:
            return None

    def load(
        self,
        file_path: str,
        preview_max_edge: Optional[int] = None,
        should_cancel: Optional[Callable[[], bool]] = None,
    ) -> Tuple[ContextManager[Any], dict]:
        preview_dims: Optional[Tuple[int, int]] = None
        preview_ir: Optional[np.ndarray] = None
        if preview_max_edge is not None:
            handled_4ch, preview_4ch = _peek_linearraw_4ch_preview(file_path, preview_max_edge, should_cancel)
            if handled_4ch:
                if preview_4ch is None:
                    raise RuntimeError("LinearRaw DNG cannot be decoded within the preview memory limit")
                rgb, preview_ir, preview_dims = preview_4ch
                peeked = None
            else:
                handled_3ch, preview_3ch = _peek_linear_dng_rgb_preview(file_path, preview_max_edge, should_cancel)
                if handled_3ch:
                    if preview_3ch is None:
                        raise RuntimeError("LinearRaw DNG cannot be decoded within the preview memory limit")
                    rgb, wb_gains, preview_dims = preview_3ch
                    metadata = {
                        "orientation": read_orientation(file_path),
                        "raw_flip": 0,
                        "color_space": None,
                        "ir": None,
                    }
                    return NonStandardFileWrapper(rgb, full_output_hw=preview_dims, wb_gains=wb_gains), metadata
                peeked = None
        else:
            peeked = _peek_linearraw_4ch(file_path)

        if preview_ir is not None and preview_dims is not None:
            metadata = {
                "orientation": read_orientation(file_path),
                "raw_flip": 0,
                "color_space": None,
                "ir": preview_ir,
            }
            return NonStandardFileWrapper(rgb, full_output_hw=preview_dims), metadata

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
            raw = None
            try:
                raw = rawpy.imread(file_path)
                if should_cancel is not None and should_cancel():
                    raw.close()
                    raise InterruptedError("preview load cancelled")
                raw.unpack()  # force now: postprocess() would hit the same error later
                if should_cancel is not None and should_cancel():
                    raw.close()
                    raise InterruptedError("preview load cancelled")
            except rawpy.LibRawError:
                if raw is not None:
                    raw.close()
                fallback = _peek_linear_dng_rgb(file_path) if preview_max_edge is None else None
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
            if should_cancel is not None and should_cancel():
                raw.close()
                raise InterruptedError("preview load cancelled")

        metadata = {
            "orientation": read_orientation(file_path),
            "raw_flip": 0,
            # Decoded output_color=raw, so the file's own tags characterise nothing here.
            "color_space": None,
            "ir": _peek_hdri_ir_page(file_path),
        }

        return raw, metadata

    def load_bounded_preview(
        self,
        file_path: str,
        max_edge: int,
        *,
        fast_only: bool = False,
        should_cancel: Optional[Callable[[], bool]] = None,
    ) -> Optional[Image.Image]:
        if should_cancel is not None and should_cancel():
            raise InterruptedError("preview cancelled")

        quick = dng_quick_preview(file_path)
        if quick is not None:
            return fit_bounded_preview(quick, max_edge)
        if not fast_only:
            handled, preview = dng_bounded_preview(file_path, max_edge, should_cancel=should_cancel)
            if handled:
                return preview

        try:
            with rawpy.imread(file_path) as raw:
                preview = embedded_preview(raw, file_path)
        except Exception:
            return None
        if preview is None:
            return None
        preview.draft("RGB", (max_edge, max_edge))
        if preview.width * preview.height * 3 > _EMBEDDED_PREVIEW_MAX_BYTES:
            return None
        return fit_bounded_preview(preview, max_edge, read_orientation(file_path))
