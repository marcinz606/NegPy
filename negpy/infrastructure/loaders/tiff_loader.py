import os
import imageio.v3 as iio
import numpy as np
import tifffile
from typing import Any, ContextManager, Optional, Tuple
from negpy.domain.interfaces import IImageLoader
from negpy.domain.models import ColorSpace
from negpy.kernel.image.logic import srgb_to_linear, uint8_to_float32, uint16_to_float32
from negpy.infrastructure.loaders.helpers import NonStandardFileWrapper, identify_color_space_from_icc
from negpy.kernel.system.logging import get_logger

logger = get_logger(__name__)


def _normalize_ir_to_float32(ir: np.ndarray) -> np.ndarray:
    """Single-channel uint8/uint16/float ndarray → float32 in [0,1]."""
    if ir.ndim == 3:
        ir = ir[:, :, 0]
    if ir.dtype == np.uint8:
        return ir.astype(np.float32) * (1.0 / 255.0)
    if ir.dtype == np.uint16:
        return ir.astype(np.float32) * (1.0 / 65535.0)
    return np.clip(ir.astype(np.float32), 0.0, 1.0)


def _read_sidecar_ir(file_path: str) -> Optional[np.ndarray]:
    """Looks for `<basename>_IR.tif(f)` sidecar produced by write_tiff_16bit."""
    base, _ = os.path.splitext(file_path)
    for ext in ("_IR.tif", "_IR.tiff"):
        candidate = base + ext
        if os.path.exists(candidate):
            try:
                arr = tifffile.imread(candidate)
                return _normalize_ir_to_float32(np.asarray(arr))
            except Exception as e:
                logger.warning(f"Failed to read IR sidecar {candidate}: {e}")
    return None


def _read_ir_from_extra_page(file_path: str, main_h: int, main_w: int) -> Optional[np.ndarray]:
    """Finds a grayscale page at full resolution — SilverFast iSRD stores IR as page 2 with NewSubfileType=4."""
    try:
        with tifffile.TiffFile(file_path) as tif:
            for page in tif.pages[1:]:
                if page.shape != (main_h, main_w):
                    continue
                tags = getattr(page, "tags", None) or {}
                nst = tags.get(254)
                photometric = getattr(page, "photometric", None)
                is_mask_page = nst is not None and nst.value == 4
                is_grayscale = photometric is not None and int(photometric) == 1
                if is_mask_page or is_grayscale:
                    return _normalize_ir_to_float32(page.asarray())
    except Exception as e:
        logger.warning(f"Failed to read extra-page IR from {file_path}: {e}")
    return None


def _extract_ir_from_extrasamples(file_path: str, img: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Inspects ExtraSamples; returns (rgb, ir_or_none).

    Convention:
    - ExtraSamples[0] == 0 (UNSPECIFIED) → 4th plane is IR.
    - ExtraSamples[0] in (1, 2) (associated/unassociated alpha) → drop as alpha.
    - ExtraSamples tag missing → treat 4th plane as IR. Many scanner stacks (Nikon
      Coolscan via Nikon Scan, some VueScan profiles) emit 4-sample TIFFs without
      tagging the extra plane. A trailing alpha channel from a scanner is
      vanishingly rare; IR is the overwhelmingly common case.
    """
    if img.ndim != 3 or img.shape[2] != 4:
        return img, None

    extrasamples_kind: Optional[int] = None
    tag_present = False
    try:
        with tifffile.TiffFile(file_path) as tif:
            page = tif.pages[0]
            tags = getattr(page, "tags", None)
            tag = tags.get("ExtraSamples") if tags is not None else None
            if tag is not None and tag.value is not None:
                tag_present = True
                val = tag.value
                if isinstance(val, (tuple, list, np.ndarray)):
                    extrasamples_kind = int(val[0]) if len(val) > 0 else None
                else:
                    extrasamples_kind = int(val)
    except Exception as e:
        logger.warning(f"Failed to read ExtraSamples tag from {file_path}: {e}")

    is_ir = extrasamples_kind == 0 or not tag_present
    if is_ir:
        return np.ascontiguousarray(img[:, :, :3]), _normalize_ir_to_float32(img[:, :, 3])
    return np.ascontiguousarray(img[:, :, :3]), None


class TiffLoader(IImageLoader):
    """
    Loader for TIFF scans. Surfaces an IR channel via `metadata["ir"]` when present
    (either as a 4th sample with ExtraSamples=UNSPECIFIED, or via a `_IR.tif` sidecar).
    """

    def load(self, file_path: str) -> Tuple[ContextManager[Any], dict]:
        img = iio.imread(file_path)
        ir: Optional[np.ndarray] = None

        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        elif img.ndim == 3 and img.shape[2] == 4:
            img, ir = _extract_ir_from_extrasamples(file_path, img)
        elif img.ndim == 3 and img.shape[2] > 4:
            img = img[:, :, :3]

        if ir is None:
            ir = _read_sidecar_ir(file_path)

        if ir is None:
            ir = _read_ir_from_extra_page(file_path, img.shape[0], img.shape[1])

        if img.dtype == np.uint8:
            f32 = uint8_to_float32(np.ascontiguousarray(img))
        elif img.dtype == np.uint16:
            f32 = uint16_to_float32(np.ascontiguousarray(img))
        else:
            f32 = np.clip(img.astype(np.float32), 0, 1)

        icc_bytes: bytes | None = None
        try:
            with tifffile.TiffFile(file_path) as tif:
                page = tif.pages[0]
                tags = getattr(page, "tags", None)
                tag = tags.get("InterColorProfile") if tags is not None else None
                if tag is not None and tag.value:
                    icc_bytes = bytes(tag.value)
        except Exception:
            icc_bytes = None

        color_space = identify_color_space_from_icc(icc_bytes) or ColorSpace.SRGB.value
        if color_space == ColorSpace.SRGB.value:
            f32 = srgb_to_linear(f32)
        metadata = {"orientation": 0, "color_space": color_space, "icc_profile": icc_bytes, "ir": ir}
        return NonStandardFileWrapper(f32), metadata
