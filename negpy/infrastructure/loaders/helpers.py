import os
import io
from types import SimpleNamespace
from typing import Any, Callable, Optional, Tuple

import numpy as np
import rawpy
from PIL import Image, ImageCms

from negpy.domain.models import ColorSpace
from negpy.features.process.models import DemosaicMode
from negpy.infrastructure.loaders.constants import SUPPORTED_RAW_EXTENSIONS
from negpy.kernel.image.logic import apply_exif_orientation, ensure_rgb
from negpy.kernel.system.logging import get_logger

logger = get_logger(__name__)


# (path) -> (mtime_ns, size, exif). A navigation reads the same file's EXIF for the metadata panel
# and again for the orientation tag; a RAW parse is tens of ms, a large TIFF hundreds.
_exif_cache: dict[str, tuple[int, int, Optional[dict]]] = {}
_EXIF_CACHE_MAX = 64


def read_exif_from_file(file_path: str) -> Optional[dict]:
    """Read EXIF data from a file as a piexif-format dict. Returns None on failure.
    Callers get their own copy, so mutating the result never leaks into a later read."""
    import copy

    try:
        st = os.stat(file_path)
    except OSError:
        return _read_exif_uncached(file_path)
    hit = _exif_cache.get(file_path)
    if hit is None or hit[0] != st.st_mtime_ns or hit[1] != st.st_size:
        if len(_exif_cache) >= _EXIF_CACHE_MAX:
            _exif_cache.clear()
        hit = (st.st_mtime_ns, st.st_size, _read_exif_uncached(file_path))
        _exif_cache[file_path] = hit
    return copy.deepcopy(hit[2])


def _read_exif_uncached(file_path: str) -> Optional[dict]:
    import piexif

    # Try piexif first (works for JPEG, TIFF)
    try:
        return piexif.load(file_path)
    except Exception:
        pass

    # Fallback: try to read EXIF via PIL from RAW by opening the file
    try:
        from PIL import Image

        with Image.open(file_path) as img:
            exif_bytes = img.info.get("exif")
            if exif_bytes:
                return piexif.load(exif_bytes)
    except Exception:
        pass

    # PIL cannot open JPEG XL, so its EXIF comes out of the container by hand.
    try:
        from negpy.infrastructure.loaders.jxl_boxes import is_jxl, read_jxl_exif

        with open(file_path, "rb") as fh:
            data = fh.read()
        if is_jxl(data):
            exif_bytes = read_jxl_exif(data)
            if exif_bytes:
                return piexif.load(exif_bytes)
    except Exception:
        pass

    return None


def read_orientation(file_path: str) -> int:
    """Read the EXIF orientation tag (1-8) from a file. Returns 1 (normal) when absent."""
    import piexif

    # piexif reads entire TIFF files; orientation needs only the first IFD.
    try:
        with open(file_path, "rb") as source:
            marker = source.read(4)
            if marker in (b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+"):
                import tifffile

                source.seek(0)
                with tifffile.TiffFile(source) as tif:
                    tag = tif.pages[0].tags.get("Orientation")
                    value = int(tag.value) if tag is not None else 1
                    return value if 1 <= value <= 8 else 1
    except (OSError, ValueError, IndexError, TypeError):
        return 1

    exif = read_exif_from_file(file_path)
    if not exif:
        return 1
    try:
        val = exif.get("0th", {}).get(piexif.ImageIFD.Orientation)
    except Exception:
        return 1
    if isinstance(val, int) and 1 <= val <= 8:
        return val
    return 1


def identify_color_space_from_icc(icc_bytes: Optional[bytes]) -> Optional[str]:
    """
    Resolve a ColorSpace enum value from an embedded ICC profile's description.
    Returns None when bytes are missing or the description doesn't match a known space.
    """
    if not icc_bytes:
        return None
    try:
        profile = ImageCms.getOpenProfile(io.BytesIO(icc_bytes))
        desc = (ImageCms.getProfileDescription(profile) or "").lower()
    except Exception as e:
        logger.warning(f"Could not parse embedded ICC profile: {e}")
        return None

    # Order matters: more specific matches first.
    if "prophoto" in desc:
        return ColorSpace.PROPHOTO.value
    if "rec. 2020" in desc or "rec2020" in desc or "bt.2020" in desc:
        return ColorSpace.REC2020.value
    if "display p3" in desc or "p3 d65" in desc:
        return ColorSpace.P3_D65.value
    if "aces" in desc:
        return ColorSpace.ACES.value
    if "adobe rgb" in desc or "adobe compat" in desc:
        return ColorSpace.ADOBE_RGB.value
    if "srgb" in desc or "iec 61966" in desc or "iec61966" in desc:
        return ColorSpace.SRGB.value
    return None


def _tiff_preview_page(file_path: str) -> Optional[Image.Image]:
    """Reduced-resolution preview page of a TIFF-based raw, or None.

    Page 0 holds the preview only when the full-res data sits in SubIFDs, which is how
    DNG writers lay it out. Scanner DNGs write that page 16-bit, so both depths count.
    """
    try:
        import tifffile

        with tifffile.TiffFile(file_path) as tif:
            page = tif.pages[0]
            if not page.pages or page.dtype not in (np.uint8, np.uint16):  # type: ignore[union-attr]
                return None
            decoded_bytes = int(np.prod(page.shape)) * int(np.dtype(page.dtype).itemsize)
            if decoded_bytes > _QUICK_PREVIEW_MAX_BYTES:
                return None
            arr = page.asarray()  # type: ignore[attr-defined]
    except Exception as e:
        logger.warning(f"TIFF preview page read failed for {file_path}: {e}")
        return None

    if arr.dtype == np.uint16:
        arr = (arr >> 8).astype(np.uint8)
    return Image.fromarray(ensure_rgb(arr))


_QUICK_PREVIEW_MAX_BYTES = 64 * 1024 * 1024
_DNG_STREAM_PREVIEW_MAX_BYTES = 64 * 1024 * 1024
_TIFF_STREAM_PREVIEW_MAX_BYTES = 64 * 1024 * 1024
_DNG_LINEAR_RAW = 34892
_DNG_CFA = 32803

_linear_u16_levels = np.arange(65536, dtype=np.float32) / 65535.0
_linear_u16_low = _linear_u16_levels < 0.018
_linear_u16_levels[_linear_u16_low] *= 4.5
_linear_u16_levels[~_linear_u16_low] = 1.099 * np.power(_linear_u16_levels[~_linear_u16_low], 1.0 / 2.222) - 0.099
_LINEAR_U16_DISPLAY_LUT = np.clip(_linear_u16_levels * 255.0, 0, 255).astype(np.uint8)
del _linear_u16_levels, _linear_u16_low


def linear_uint16_to_display_uint8(values: np.ndarray) -> np.ndarray:
    """Apply the loader display curve to linear uint16 preview samples."""
    return _LINEAR_U16_DISPLAY_LUT[values]


def _collect_dng_pages(tif: Any) -> list[Any]:
    pages: list[Any] = []
    seen: set[int] = set()

    def collect(page: Any) -> None:
        offset = int(getattr(page, "offset", id(page)))
        if offset in seen:
            return
        seen.add(offset)
        pages.append(page)
        for child in page.pages or ():
            collect(child)

    for root in tif.pages:
        collect(root)
    return pages


def _dng_tag_floats(tag: Any) -> np.ndarray:
    if tag is None:
        return np.empty(0, dtype=np.float32)
    try:
        values = np.asarray(tag.value, dtype=np.float32).reshape(-1)
        if int(tag.dtype) in (5, 10):
            if values.size % 2:
                return np.empty(0, dtype=np.float32)
            numerators = values[0::2]
            denominators = values[1::2]
            return np.divide(
                numerators,
                denominators,
                out=np.zeros_like(numerators),
                where=denominators != 0,
            )
        return values
    except (AttributeError, TypeError, ValueError):
        return np.empty(0, dtype=np.float32)


def _dng_rgb_values(values: np.ndarray, default: float) -> np.ndarray:
    if values.size >= 3:
        return values[:3].reshape(1, 1, 3)
    if values.size == 1:
        return np.full((1, 1, 3), float(values[0]), dtype=np.float32)
    return np.full((1, 1, 3), default, dtype=np.float32)


def dng_quick_preview(file_path: str) -> Optional[Image.Image]:
    """Decode the smallest usable reduced DNG IFD without reading the main image."""
    if os.path.splitext(file_path)[1].lower() != ".dng":
        return None
    try:
        import tifffile

        with tifffile.TiffFile(file_path) as tif:
            pages = _collect_dng_pages(tif)

            if not pages:
                return None
            largest_pixels = max(int(np.prod(page.shape[:2])) for page in pages if len(page.shape) >= 2)
            candidates: list[tuple[int, int, Any]] = []
            for page in pages:
                shape = tuple(int(v) for v in page.shape)
                if len(shape) not in (2, 3) or (len(shape) == 3 and shape[2] not in (1, 3, 4)):
                    continue
                tags = page.tags
                subfile = tags.get("NewSubfileType")
                reduced = bool(int(subfile.value) & 1) if subfile is not None else False
                pixels = shape[0] * shape[1]
                if not reduced and not (page is pages[0] and bool(page.pages) and pixels < largest_pixels):
                    continue
                photo_tag = tags.get("PhotometricInterpretation")
                photo = int(photo_tag.value) if photo_tag is not None else 0
                samples = shape[2] if len(shape) == 3 else 1
                if photo == _DNG_CFA or (photo == _DNG_LINEAR_RAW and samples < 3):
                    continue
                decoded_bytes = int(np.prod(shape)) * int(np.dtype(page.dtype).itemsize)
                if decoded_bytes > _QUICK_PREVIEW_MAX_BYTES or page.dtype not in (np.uint8, np.uint16):
                    continue
                long_edge = max(shape[:2])
                below_target = int(long_edge < 256)
                candidates.append((below_target, pixels if not below_target else -pixels, page))

            if not candidates:
                return None
            page = min(candidates, key=lambda item: (item[0], item[1]))[2]
            arr = page.asarray()  # type: ignore[attr-defined]
            orientation_tag = page.tags.get("Orientation") or pages[0].tags.get("Orientation")
            orientation = int(orientation_tag.value) if orientation_tag is not None else 1
    except Exception as e:
        logger.warning(f"DNG quick preview read failed for {file_path}: {e}")
        return None

    if arr.dtype == np.uint16:
        arr = (arr >> 8).astype(np.uint8)
    arr = ensure_rgb(arr)
    if arr.ndim == 3 and arr.shape[2] > 3:
        arr = arr[:, :, :3]
    if orientation != 1:
        arr = apply_exif_orientation(arr, orientation)
    return Image.fromarray(np.ascontiguousarray(arr))


def dng_bounded_preview(
    file_path: str,
    max_edge: int,
    *,
    should_cancel: Optional[Callable[[], bool]] = None,
) -> tuple[bool, Optional[Image.Image]]:
    """Stream a LinearRaw DNG into a small preview with bounded segment memory.

    The boolean is true when the DNG is LinearRaw and must not fall through to a
    full-array loader, including when its layout prevents a bounded preview.
    """
    if os.path.splitext(file_path)[1].lower() != ".dng":
        return False, None

    handled = False
    try:
        import tifffile

        with tifffile.TiffFile(file_path) as tif:
            pages = _collect_dng_pages(tif)
            candidates = []
            for page in pages:
                shape = tuple(int(v) for v in page.shape)
                photo_tag = page.tags.get("PhotometricInterpretation")
                photo = int(photo_tag.value) if photo_tag is not None else 0
                if len(shape) == 3 and shape[2] in (3, 4) and photo == _DNG_LINEAR_RAW:
                    candidates.append(page)
            if not candidates:
                return False, None

            handled = True
            page = max(candidates, key=lambda item: int(np.prod(item.shape[:2])))
            height, width, _samples = (int(v) for v in page.shape)
            itemsize = int(np.dtype(page.dtype).itemsize)
            segment_height = int(page.tilelength if page.is_tiled else page.rowsperstrip or height)
            segment_width = int(page.tilewidth if page.is_tiled else width)
            segment_bytes = segment_height * segment_width * int(page.samplesperpixel) * itemsize
            if page.dtype not in (np.uint8, np.uint16) or segment_bytes > _DNG_STREAM_PREVIEW_MAX_BYTES:
                return True, None

            scale = min(1.0, max(1, int(max_edge)) / max(height, width))
            out_height = max(1, int(round(height * scale)))
            out_width = max(1, int(round(width * scale)))
            output = np.zeros((out_height, out_width, 3), dtype=np.uint8)

            def tag(name: str) -> Any:
                return page.tags.get(name) or pages[0].tags.get(name)

            dtype_max = float(np.iinfo(page.dtype).max)
            black = _dng_rgb_values(_dng_tag_floats(tag("BlackLevel")), 0.0)
            white = _dng_rgb_values(_dng_tag_floats(tag("WhiteLevel")), dtype_max)
            neutral = _dng_tag_floats(pages[0].tags.get("AsShotNeutral"))
            wb = np.ones((1, 1, 3), dtype=np.float32)
            if neutral.size >= 3 and np.all(neutral[:3] > 0):
                wb[0, 0] = (neutral[1] / neutral[0], 1.0, neutral[1] / neutral[2])
            linearization_tag = tag("LinearizationTable")
            linearization = np.asarray(linearization_tag.value, dtype=np.float32) if linearization_tag is not None else None
            crop_origin = _dng_tag_floats(tag("DefaultCropOrigin"))
            crop_size = _dng_tag_floats(tag("DefaultCropSize"))
            orientation_tag = tag("Orientation")
            orientation = int(orientation_tag.value) if orientation_tag is not None else 1
            for decoded, position, _shape in page.segments(maxworkers=1):
                if should_cancel is not None and should_cancel():
                    raise InterruptedError("thumbnail cancelled")
                if decoded is None:
                    continue
                tile = decoded[0] if decoded.ndim == 4 else decoded
                if tile.ndim != 3 or tile.shape[2] < 3:
                    return True, None
                y, x = int(position[2]), int(position[3])
                valid_height = min(tile.shape[0], height - y)
                valid_width = min(tile.shape[1], width - x)
                if valid_height <= 0 or valid_width <= 0:
                    continue
                source = tile[:valid_height, :valid_width, :3]
                if linearization is not None:
                    indices = np.clip(source, 0, linearization.size - 1).astype(np.int32)
                    data = linearization[indices]
                else:
                    data = source.astype(np.float32)
                data = np.clip((data - black) / np.maximum(white - black, 1e-6), 0.0, 1.0)
                data = np.clip(data * wb, 0.0, 1.0)
                low = data < 0.018
                data[low] *= 4.5
                data[~low] = 1.099 * np.power(data[~low], 1.0 / 2.222) - 0.099
                tile_u8 = np.clip(data * 255.0, 0, 255).astype(np.uint8)

                left = int(round(x * out_width / width))
                top = int(round(y * out_height / height))
                right = int(round((x + valid_width) * out_width / width))
                bottom = int(round((y + valid_height) * out_height / height))
                if right <= left or bottom <= top:
                    continue
                small = Image.fromarray(tile_u8).resize((right - left, bottom - top), Image.Resampling.BOX)
                output[top:bottom, left:right] = np.asarray(small)

            image = Image.fromarray(output)
            if crop_origin.size >= 2 and crop_size.size >= 2:
                ox, oy = float(crop_origin[0]), float(crop_origin[1])
                crop_width, crop_height = float(crop_size[0]), float(crop_size[1])
                box = (
                    max(0, int(round(ox * out_width / width))),
                    max(0, int(round(oy * out_height / height))),
                    min(out_width, int(round((ox + crop_width) * out_width / width))),
                    min(out_height, int(round((oy + crop_height) * out_height / height))),
                )
                if box[2] > box[0] and box[3] > box[1]:
                    image = image.crop(box)
            if orientation != 1:
                image = Image.fromarray(apply_exif_orientation(np.asarray(image), orientation))
            return True, image
    except InterruptedError:
        raise
    except Exception as e:
        logger.warning(f"DNG bounded preview read failed for {file_path}: {e}")
        return handled, None


def fit_bounded_preview(image: Image.Image, max_edge: int, orientation: int = 1) -> Image.Image:
    """Return a loaded RGB preview with orientation and size applied."""
    result = image.convert("RGB")
    if orientation != 1:
        result = Image.fromarray(apply_exif_orientation(np.asarray(result), orientation))
    result.thumbnail((max(1, max_edge), max(1, max_edge)), Image.Resampling.LANCZOS)
    return result.copy()


def bounded_tiff_page_preview(
    page: Any,
    max_edge: int,
    *,
    should_cancel: Optional[Callable[[], bool]] = None,
) -> Optional[Image.Image]:
    """Stream a chunky grayscale or RGB TIFF page into a bounded preview."""
    shape = tuple(int(value) for value in page.shape)
    if len(shape) not in (2, 3) or (len(shape) == 3 and shape[2] not in (1, 3, 4)):
        return None
    if page.dtype not in (np.uint8, np.uint16) or int(getattr(page, "planarconfig", 1)) != 1:
        return None

    height, width = shape[:2]
    samples = shape[2] if len(shape) == 3 else 1
    itemsize = int(np.dtype(page.dtype).itemsize)
    segment_height = int(page.tilelength if page.is_tiled else page.rowsperstrip or height)
    segment_width = int(page.tilewidth if page.is_tiled else width)
    if segment_height * segment_width * samples * itemsize > _TIFF_STREAM_PREVIEW_MAX_BYTES:
        return None

    scale = min(1.0, max(1, max_edge) / max(height, width))
    out_height = max(1, int(round(height * scale)))
    out_width = max(1, int(round(width * scale)))
    output = np.zeros((out_height, out_width, 3), dtype=np.uint8)
    for decoded, position, _shape in page.segments(maxworkers=1):
        if should_cancel is not None and should_cancel():
            raise InterruptedError("preview cancelled")
        if decoded is None:
            continue
        tile = decoded[0] if decoded.ndim == 4 else decoded
        if tile.ndim == 2:
            tile = tile[:, :, None]
        if tile.ndim != 3 or tile.shape[2] not in (1, 3, 4):
            return None
        y, x = int(position[2]), int(position[3])
        valid_height = min(tile.shape[0], height - y)
        valid_width = min(tile.shape[1], width - x)
        if valid_height <= 0 or valid_width <= 0:
            continue
        source = tile[:valid_height, :valid_width]
        if source.dtype == np.uint16:
            source = linear_uint16_to_display_uint8(source)
        if source.shape[2] == 1:
            source = np.repeat(source, 3, axis=2)
        elif source.shape[2] == 4:
            source = source[:, :, :3]

        left = int(round(x * out_width / width))
        top = int(round(y * out_height / height))
        right = int(round((x + valid_width) * out_width / width))
        bottom = int(round((y + valid_height) * out_height / height))
        if right <= left or bottom <= top:
            continue
        small = Image.fromarray(source).resize((right - left, bottom - top), Image.Resampling.BOX)
        output[top:bottom, left:right] = np.asarray(small)

    return Image.fromarray(output)


def embedded_preview(raw: Any, file_path: str) -> Optional[Image.Image]:
    """Embedded preview image of a raw file, or None when it has none to give.

    Only a JPEG thumb is taken from libraw. For a BITMAP thumb rawpy shapes the array
    (h, w, 3) from libraw's hardcoded colors=3 while the allocation holds only data_size
    bytes, and rawpy exposes no data_size — a grayscale thumb is then read three times
    past its end, which segfaults whenever the heap tail is unmapped. A TIFF-based file
    carries that same preview as its reduced-resolution page 0, so it is read from the
    file instead. Never touch `thumb.data` on a BITMAP thumb.
    """
    if not hasattr(raw, "extract_thumb"):
        return None
    try:
        thumb = raw.extract_thumb()
        if thumb.format == rawpy.ThumbFormat.JPEG:
            return Image.open(io.BytesIO(thumb.data))
        if thumb.format == rawpy.ThumbFormat.BITMAP:
            return _tiff_preview_page(file_path)
    except Exception:
        return None
    return None


class NonStandardFileWrapper:
    """
    numpy -> rawpy-like interface.
    """

    def __init__(
        self,
        data: np.ndarray,
        full_output_hw: Optional[Tuple[int, int]] = None,
        wb_gains: Optional[Tuple[float, float, float]] = None,
    ) -> None:
        self.data = data
        # If set, `sizes` reports this (h, w) for full image; else derived from `data` shape.
        self._full_output_hw: Optional[Tuple[int, int]] = full_output_hw
        # As-shot (R, G, B) white balance gains, applied by postprocess() when the caller asks
        # for camera WB. None means the source has no WB to offer, as with NegPy's own scanner
        # DNGs, which are always neutral. postprocess() then leaves the data untouched.
        self.wb_gains: Optional[Tuple[float, float, float]] = wb_gains

    @property
    def sizes(self) -> Any:
        if self._full_output_hw is not None:
            h, w = self._full_output_hw
        else:
            h, w = self.data.shape[0], self.data.shape[1]
        return SimpleNamespace(raw_height=int(h), raw_width=int(w))

    def __enter__(self) -> "NonStandardFileWrapper":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        pass

    def postprocess(self, **kwargs: Any) -> np.ndarray:
        bps = kwargs.get("output_bps", 8)
        half_size = kwargs.get("half_size", False)
        gamma = kwargs.get("gamma")
        data = self.data
        if half_size:
            data = data[::2, ::2]

        if kwargs.get("use_camera_wb") and self.wb_gains is not None:
            r, g, b = self.wb_gains
            data = data.astype(np.float32, copy=True)
            data[..., 0] *= r
            data[..., 2] *= b
            data = np.clip(data, 0.0, 1.0)

        if gamma is None or tuple(gamma) != (1, 1):
            # LibRaw's default BT.709 display gamma, or linear thumbnails go near-black.
            data = np.where(data < 0.018, data * 4.5, 1.099 * np.power(np.maximum(data, 0.0), 1.0 / 2.222) - 0.099)

        if bps == 16:
            return (data * 65535.0).astype(np.uint16)
        return (data * 255.0).astype(np.uint8)


_DEMOSAIC_ALGORITHMS: dict[str, Any] = {
    DemosaicMode.LINEAR: rawpy.DemosaicAlgorithm.LINEAR,
    DemosaicMode.VNG: rawpy.DemosaicAlgorithm.VNG,
    DemosaicMode.PPG: rawpy.DemosaicAlgorithm.PPG,
    DemosaicMode.AHD: rawpy.DemosaicAlgorithm.AHD,
    DemosaicMode.DCB: rawpy.DemosaicAlgorithm.DCB,
    DemosaicMode.DHT: rawpy.DemosaicAlgorithm.DHT,
    DemosaicMode.AAHD: rawpy.DemosaicAlgorithm.AAHD,
}


def supported_demosaic_modes() -> list:
    """AUTO plus every algorithm this libraw build compiled in. The GPL demosaic packs
    (AMAZE, LMMSE, VCD) are absent from a permissive build and render as something else."""
    return [DemosaicMode.AUTO] + [m for m, algo in _DEMOSAIC_ALGORITHMS.items() if algo.isSupported]


def resolve_demosaic(raw: Any, mode: str) -> tuple[Any, Optional[str]]:
    """The rawpy algorithm to run, paired with a display label for what actually ran
    (None for a source with no CFA, where the mode is ignored).

    On a 6x6 X-Trans CFA no algorithm reaches ahd_interpolate: LibRaw routes filters==9
    to Markesteijn ahead of the quality dispatch, 3-pass above PPG and 1-pass at PPG or
    below in the rawpy DemosaicAlgorithm quality ordering.
    """
    if isinstance(raw, NonStandardFileWrapper):
        return rawpy.DemosaicAlgorithm.LINEAR, None

    try:
        # A 2x2 CFA block is Bayer, 6x6 is X-Trans. Anything else (Stack: Linear DNG, Foveon,
        # sRAW) arrives de-mosaiced and only LINEAR is meaningful.
        if raw.raw_type == rawpy.RawType.Flat and raw.raw_pattern.shape[0] in (2, 6):
            chosen = _DEMOSAIC_ALGORITHMS.get(DemosaicMode(mode))
            if chosen is None or not chosen.isSupported:
                chosen = rawpy.DemosaicAlgorithm.AHD
            if raw.raw_pattern.shape[0] == 6:
                label = "Markesteijn 3-pass" if chosen.value > rawpy.DemosaicAlgorithm.PPG.value else "Markesteijn 1-pass"
            else:
                label = chosen.name
            return chosen, label
    except (AttributeError, ValueError) as e:
        logger.exception(f"Failed to determine sensor CFA pattern: {e}. Falling back to LINEAR.")

    return rawpy.DemosaicAlgorithm.LINEAR, None


def get_best_demosaic_algorithm(raw: Any, mode: str = DemosaicMode.AUTO) -> Any:
    """The user's `mode` where it is meaningful, else AHD for a mosaiced sensor and LINEAR
    for anything that arrives de-mosaiced. A source with no CFA has nothing to interpolate,
    so the mode is ignored there."""
    return resolve_demosaic(raw, mode)[0]


def is_xtrans(raw: Any) -> bool:
    """True for a Fuji X-Trans sensor (6x6 CFA). half_size aliases its mosaic."""
    try:
        return raw.raw_pattern.shape[0] == 6
    except (AttributeError, ValueError):
        return False


def camera_xyz_matrix(raw: Any) -> Optional[list]:
    """The decoder's XYZ->camera matrix as plain nested lists, or None if it carries none.

    Serialized out of the rawpy object deliberately: the metadata dict outlives the `with`
    block that owns the decoder, and the numpy view libraw hands back is backed by freed
    memory once it closes.
    """
    try:
        m = np.asarray(raw.rgb_xyz_matrix, dtype=np.float64)
    except Exception:
        return None
    if m.ndim != 2 or m.shape[0] < 3 or m.shape[1] != 3 or not np.all(np.isfinite(m[:3])):
        return None
    # All-zero is libraw's "no color data" sentinel, not a valid transform.
    if float(np.abs(m[:3]).max()) < 1e-12:
        return None
    return [[float(v) for v in row] for row in m[:3]]


def camera_wb_multipliers(raw: Any) -> Optional[list]:
    """The as-shot white balance as [R, G, B] multipliers, or None if absent.

    Needed only when a buffer is decoded WITHOUT white balance (Linear RAW): the camera
    matrix is row-normalized, so it assumes a neutral camera signal, and an unbalanced
    one renders with a heavy cast. Folding these back in reconstructs the balanced
    signal the matrix expects. Serialized out of the rawpy object for the same
    lifetime reason as camera_xyz_matrix.
    """
    try:
        wb = [float(v) for v in raw.camera_whitebalance[:3]]
    except Exception:
        return None
    if len(wb) != 3 or not all(np.isfinite(wb)) or min(wb) <= 0.0:
        return None
    return wb


#: Nikon's High Efficiency (HE / HE*) raw on the Z 8 and Z 9 is intoPIX TicoRAW carrying a
#: plain-text vendor marker at the head of the strip. The TIFF tag still reads 34713
#: ("Nikon NEF Compressed"), the same value a lossless NEF uses, so only the payload can
#: tell them apart.
_TICORAW_MARKER = b"INTOPIX"
_NEF_COMPRESSED = 34713


def unsupported_raw_reason(file_path: str) -> Optional[str]:
    """Why libraw cannot decode this raw, in words a photographer can act on.

    None when nothing recognised is wrong -- the caller then reports libraw's own error,
    which is right for a genuinely corrupt or unknown file. This exists because the useful
    cases are indistinguishable from corruption by their tags: a High Efficiency NEF parses
    perfectly, reports full sensor dimensions, and only fails when the payload is unpacked.
    """
    if os.path.splitext(file_path)[1].lower() != ".nef":
        return None
    try:
        import tifffile

        with tifffile.TiffFile(file_path) as tif:
            for sub in tif.pages[0].pages or []:
                tags = getattr(sub, "tags", None)
                compression = tags.get("Compression") if tags else None
                if compression is None or int(compression.value) != _NEF_COMPRESSED:
                    continue
                offsets = tags.get("StripOffsets")
                if offsets is None:
                    continue
                offset = offsets.value[0] if isinstance(offsets.value, (tuple, list)) else int(offsets.value)
                with open(file_path, "rb") as f:
                    f.seek(int(offset))
                    if _TICORAW_MARKER in f.read(64):
                        return (
                            "Nikon High Efficiency (HE) raw — NegPy cannot decode this format. "
                            "Re-shoot as Lossless Compressed, or convert to DNG."
                        )
    except Exception:
        return None
    return None


def get_supported_raw_wildcards() -> str:
    """
    Returns raw formats as string for file dialogs.
    """
    wildcards = []
    for ext in sorted(SUPPORTED_RAW_EXTENSIONS):
        base = ext.lstrip(".")
        wildcards.append(f"*.{base}")
        wildcards.append(f"*.{base.upper()}")

    return " ".join(wildcards)
