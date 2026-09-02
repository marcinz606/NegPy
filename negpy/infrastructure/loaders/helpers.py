import os
import io
from types import SimpleNamespace
from typing import Any, Optional, Tuple

import numpy as np
import rawpy
from PIL import Image, ImageCms

from negpy.domain.models import ColorSpace
from negpy.features.process.models import DemosaicMode
from negpy.infrastructure.loaders.constants import SUPPORTED_RAW_EXTENSIONS
from negpy.kernel.image.logic import ensure_rgb
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
            arr = page.asarray()  # type: ignore[attr-defined]
    except Exception as e:
        logger.warning(f"TIFF preview page read failed for {file_path}: {e}")
        return None

    if arr.dtype == np.uint16:
        arr = (arr >> 8).astype(np.uint8)
    return Image.fromarray(ensure_rgb(arr))


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


def get_best_demosaic_algorithm(raw: Any, mode: str = DemosaicMode.AUTO) -> Any:
    """The user's `mode` where it is meaningful, else AHD for a mosaiced sensor and LINEAR
    for anything that arrives de-mosaiced.

    A source with no CFA has nothing to interpolate, so the mode is ignored there. On a 6x6
    X-Trans CFA no value reaches ahd_interpolate either: LibRaw routes filters==9 to
    Markesteijn ahead of the quality dispatch, 3-pass above PPG and 1-pass at PPG or below.
    """
    if isinstance(raw, NonStandardFileWrapper):
        return rawpy.DemosaicAlgorithm.LINEAR

    try:
        # A 2x2 CFA block is Bayer, 6x6 is X-Trans. Anything else (Stack: Linear DNG, Foveon,
        # sRAW) arrives de-mosaiced and only LINEAR is meaningful.
        if raw.raw_type == rawpy.RawType.Flat and raw.raw_pattern.shape[0] in (2, 6):
            chosen = _DEMOSAIC_ALGORITHMS.get(DemosaicMode(mode))
            if chosen is not None and chosen.isSupported:
                return chosen
            return rawpy.DemosaicAlgorithm.AHD
    except (AttributeError, ValueError) as e:
        logger.exception(f"Failed to determine sensor CFA pattern: {e}. Falling back to LINEAR.")

    return rawpy.DemosaicAlgorithm.LINEAR


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
