import os
import io
from types import SimpleNamespace
from typing import Any, Optional, Tuple

import numpy as np
import rawpy
from PIL import Image, ImageCms

from negpy.domain.models import ColorSpace
from negpy.infrastructure.loaders.constants import SUPPORTED_RAW_EXTENSIONS
from negpy.kernel.image.logic import ensure_rgb
from negpy.kernel.system.logging import get_logger

logger = get_logger(__name__)


def read_exif_from_file(file_path: str) -> Optional[dict]:
    """Read EXIF data from a file as a piexif-format dict. Returns None on failure."""
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


def get_best_demosaic_algorithm(raw: Any, for_preview: bool = False) -> Any:
    """
    Selects optimal demosaicing algorithm based on sensor type and CFA pattern.
    Exclusively uses algorithms packaged in the standard permissive (LGPL) rawpy build.

    When for_preview=True, selects faster algorithms appropriate for downsampled
    preview rendering (PPG for Bayer, LINEAR for X-Trans).
    """
    selected_algo = rawpy.DemosaicAlgorithm.LINEAR

    if isinstance(raw, NonStandardFileWrapper):
        return selected_algo

    try:
        # Stacked sensors (Linear DNG, Foveon, sRAW)
        if raw.raw_type == rawpy.RawType.Stack:
            selected_algo = rawpy.DemosaicAlgorithm.LINEAR

        # Flat sensors (Bayer, X-Trans)
        elif raw.raw_type == rawpy.RawType.Flat:
            cfa_block_size = raw.raw_pattern.shape[0]

            if cfa_block_size == 6:
                # A 6x6 block means a Fujifilm X-Trans sensor. LINEAR is much faster than DHT and its
                # artifacts are invisible at preview scale. VNG was used before, but it produces dot and
                # maze artifacts on X-Trans's 6x6 pattern in high-contrast regions (see #272). DHT was
                # added to dcraw and LibRaw specifically to handle X-Trans, and is also LGPL-clean.
                selected_algo = rawpy.DemosaicAlgorithm.LINEAR if for_preview else rawpy.DemosaicAlgorithm.DHT

            elif cfa_block_size == 2:
                # A 2x2 block means a standard Bayer sensor. PPG is faster than AHD with negligible
                # quality difference at preview scale.
                selected_algo = rawpy.DemosaicAlgorithm.PPG if for_preview else rawpy.DemosaicAlgorithm.AHD

    except (AttributeError, ValueError) as e:
        logger.exception(f"Failed to determine sensor CFA pattern: {e}. Falling back to LINEAR.")
        selected_algo = rawpy.DemosaicAlgorithm.LINEAR

    return selected_algo


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
