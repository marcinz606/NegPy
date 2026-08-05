"""Linear Output: export a loader's decoded buffer as an untagged 16-bit TIFF.

For single files, bypasses the entire darkroom pipeline — no normalization,
exposure, lab, toning, finish, flatfield, or sensor-crosstalk correction.

For composites (stitch / RGB-scan triplets), flatfield and sensor correction
are applied per-part before assembly so the output is physically correct
(no vignetting seams or channel crosstalk).
"""

import io
import os
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import rawpy
import tifffile as _tifffile

from negpy.features.flatfield.logic import apply_flatfield as _apply_flatfield_correction
from negpy.features.retouch.models import IR_METHOD_OPENICE, RetouchConfig
from negpy.features.flatfield.models import FlatFieldConfig
from negpy.features.geometry.models import GeometryConfig
from negpy.features.process.models import ProcessConfig
from negpy.features.process.sensor import apply_sensor_correction
from negpy.features.rgbscan.logic import merge_rgb_triplet
from negpy.features.rgbscan.models import RgbScanConfig, is_rgb_triplet
from negpy.features.stitch.logic import stitch_composite
from negpy.features.stitch.models import StitchConfig, stitch_has_triplets
from negpy.infrastructure.loaders.constants import SUPPORTED_JPEG_EXTENSIONS, SUPPORTED_RAW_EXTENSIONS, SUPPORTED_TIFF_EXTENSIONS
from negpy.infrastructure.loaders.helpers import NonStandardFileWrapper, get_best_demosaic_algorithm, read_orientation
from negpy.infrastructure.loaders.pakon_loader import PakonLoader
from negpy.infrastructure.loaders.fff_loader import is_flextight_fff
from negpy.infrastructure.loaders.nef_loader import is_coolscan_nef
from negpy.infrastructure.loaders.noritsu_loader import is_noritsu_raw, detect_noritsu_dims
from negpy.infrastructure.loaders.rawpy_loader import (
    _find_linearraw_page,
    _is_dng,
    _peek_hdri_ir_page,
    _peek_linearraw_4ch,
)
from negpy.kernel.image.logic import _to_uint16_jit, apply_exif_orientation, ensure_rgb, uint16_to_float32


@dataclass(frozen=True)
class _CameraWB:
    """Camera white balance multipliers extracted before the unity-WB decode."""

    as_shot: tuple[float, float, float, float]
    daylight: tuple[float, float, float, float]


@dataclass(frozen=True)
class _SourceMeta:
    """Device and timestamp metadata from the source file."""

    make: Optional[str] = None
    model: Optional[str] = None
    datetime: Optional[str] = None


def _read_source_meta_tiff(file_path: str) -> _SourceMeta:
    try:
        with _tifffile.TiffFile(file_path) as tif:
            tags = tif.pages[0].tags
            make = tags.get("Make")
            model = tags.get("Model")
            dt = tags.get("DateTime")
            return _SourceMeta(
                make=str(make.value).strip() if make else None,
                model=str(model.value).strip() if model else None,
                datetime=str(dt.value).strip() if dt else None,
            )
    except Exception:
        return _read_source_meta_exif(file_path)


def _read_source_meta_exif(file_path: str) -> _SourceMeta:
    """Fallback: scan for an embedded EXIF TIFF block (works for RAF, ORF, etc.)."""
    try:
        with open(file_path, "rb") as f:
            header = f.read(4096)
        marker = header.find(b"Exif\x00\x00")
        if marker < 0:
            return _SourceMeta()
        tiff_bytes = header[marker + 6 :]
        import logging

        prev = logging.getLogger("tifffile").level
        logging.getLogger("tifffile").setLevel(logging.CRITICAL)
        try:
            tif = _tifffile.TiffFile(io.BytesIO(tiff_bytes))
        finally:
            logging.getLogger("tifffile").setLevel(prev)
            tags = tif.pages[0].tags
            make = tags.get("Make")
            model = tags.get("Model")
            dt = tags.get("DateTime")
            return _SourceMeta(
                make=str(make.value).strip() if make else None,
                model=str(model.value).strip() if model else None,
                datetime=str(dt.value).strip() if dt else None,
            )
    except Exception:
        return _SourceMeta()


def _read_fff_meta(file_path: str) -> _SourceMeta:
    """Build _SourceMeta from FFF proprietary tags (plist + firmware)."""
    try:
        from negpy.infrastructure.loaders.fff_loader import _parse_fff_firmware, _parse_fff_plist

        with _tifffile.TiffFile(file_path) as tif:
            p0_tags = getattr(tif.pages[0], "tags", None)
            if p0_tags is None:
                return _SourceMeta()
            parts: dict = {}
            plist_tag = p0_tags.get(50457)
            if plist_tag is not None and isinstance(plist_tag.value, bytes):
                parts.update(_parse_fff_plist(plist_tag.value))
            fw_tag = p0_tags.get(46279)
            if fw_tag is not None and isinstance(fw_tag.value, bytes):
                parts.update(_parse_fff_firmware(fw_tag.value))
        make = "Imacon/Hasselblad"
        serial = parts.get("scanner_serial", "")
        model_parts = [s for s in ("Flextight", serial) if s]
        model = " ".join(model_parts) if model_parts else None
        film = parts.get("film_stock")
        film_type = parts.get("film_type")
        if film and film_type:
            make = f"{make} ({film}, {film_type})"
        elif film:
            make = f"{make} ({film})"
        return _SourceMeta(make=make, model=model, datetime=parts.get("scan_date"))
    except Exception:
        return _SourceMeta(make="Imacon/Hasselblad")


def _is_camera_raw(file_path: str) -> bool:
    ext = os.path.splitext(file_path)[1].lower()
    if ext in SUPPORTED_TIFF_EXTENSIONS | SUPPORTED_JPEG_EXTENSIONS:
        return False
    if PakonLoader.can_handle(file_path):
        return False
    if is_coolscan_nef(file_path):
        return False
    if is_flextight_fff(file_path):
        return False
    if is_noritsu_raw(file_path):
        return False
    return ext in SUPPORTED_RAW_EXTENSIONS


def _is_tiff(file_path: str) -> bool:
    return os.path.splitext(file_path)[1].lower() in SUPPORTED_TIFF_EXTENSIONS


def is_linear_output_supported(file_path: str) -> bool:
    if PakonLoader.can_handle(file_path):
        return True
    if _is_dng(file_path):
        return _is_linearraw_dng(file_path) or _is_camera_raw(file_path)
    if is_coolscan_nef(file_path):
        return True
    if is_flextight_fff(file_path):
        return True
    if is_noritsu_raw(file_path):
        return True
    if _is_camera_raw(file_path):
        return True
    if _is_tiff(file_path):
        return True
    return False


def linear_output_source_type(file_path: str) -> str:
    """Classify a file for Linear Output expansion options.

    Returns ``"pakon"``, ``"dng"``, ``"camera"``, ``"nef"``, ``"fff"``, ``"tiff"``, or ``"unsupported"``.
    """
    if PakonLoader.can_handle(file_path):
        return "pakon_f335" if _is_pakon_f335(file_path) else "pakon"
    if _is_dng(file_path) and _is_linearraw_dng(file_path):
        return "dng"
    if is_coolscan_nef(file_path):
        return "nef"
    if is_flextight_fff(file_path):
        return "fff"
    if is_noritsu_raw(file_path):
        return "noritsu"
    if _is_camera_raw(file_path):
        return "camera"
    if _is_tiff(file_path):
        return "tiff"
    return "unsupported"


def _is_linearraw_dng(file_path: str) -> bool:
    """True if the DNG contains a LinearRaw IFD (3 or 4 samples)."""
    try:
        with _tifffile.TiffFile(file_path) as tif:
            return _find_linearraw_page(tif, samples=4) is not None or _find_linearraw_page(tif, samples=3) is not None
    except Exception:
        return False


def _apply_user_geometry(f32: np.ndarray, geometry: GeometryConfig) -> np.ndarray:
    if geometry.rotation != 0:
        f32 = np.rot90(f32, k=geometry.rotation)
    if geometry.flip_horizontal:
        f32 = np.ascontiguousarray(np.fliplr(f32))
    if geometry.flip_vertical:
        f32 = np.ascontiguousarray(np.flipud(f32))
    return f32


def _apply_geometry(f32: np.ndarray, orientation: int, geometry: Optional[GeometryConfig]) -> np.ndarray:
    f32 = apply_exif_orientation(f32, orientation)
    if geometry is not None:
        f32 = _apply_user_geometry(f32, geometry)
    return f32


TIFF_GAMMA_OPTIONS: list[tuple[str, str]] = [
    ("linear", "Linear (1.0)"),
    ("1.8", "Gamma 1.8"),
    ("2.2", "Gamma 2.2"),
    ("2.4", "Gamma 2.4"),
    ("2.6", "Gamma 2.6"),
    ("srgb", "sRGB"),
    ("lstar", "L*"),
    ("rec709", "Rec.709"),
]


def _linearize(f32: np.ndarray, gamma_key: str) -> np.ndarray:
    """Reverse a gamma encoding to recover linear-light values."""
    if gamma_key == "linear":
        return f32
    f32 = np.clip(f32, 0.0, 1.0)
    if gamma_key in ("1.8", "2.2", "2.4", "2.6"):
        g = float(gamma_key)
        return np.power(f32, g, dtype=np.float32)
    if gamma_key == "srgb":
        lo = f32 / 12.92
        hi = np.power((f32 + 0.055) / 1.055, 2.4, dtype=np.float32)
        return np.where(f32 <= 0.04045, lo, hi).astype(np.float32)
    if gamma_key == "rec709":
        lo = f32 / 4.5
        hi = np.power((f32 + 0.099) / 1.099, 1.0 / 0.45, dtype=np.float32)
        return np.where(f32 <= 0.081, lo, hi).astype(np.float32)
    if gamma_key == "lstar":
        lo = f32 / 9.0329
        hi = np.power((f32 + 0.16) / 1.16, 3.0, dtype=np.float32)
        return np.where(f32 <= 0.08, lo, hi).astype(np.float32)
    return f32


def _apply_white_balance(f32: np.ndarray, wb: _CameraWB) -> np.ndarray:
    """Multiply a linear RGB buffer by the as-shot white-balance gains."""
    r, _g, b = _normalize_wb_rgb(wb.as_shot)
    f32 = f32.copy()
    f32[:, :, 0] *= r
    f32[:, :, 2] *= b
    np.clip(f32, 0.0, 1.0, out=f32)
    return f32


def _apply_ice(rgb: np.ndarray, ir: np.ndarray, retouch: RetouchConfig) -> np.ndarray:
    """Apply IR dust correction to a linear RGB buffer using the IR channel."""
    if retouch.ir_method == IR_METHOD_OPENICE:
        from negpy.features.retouch import openice

        corrected, _, _, _ = openice.run(rgb, ir, float(retouch.ir_threshold), None)
        return corrected

    from negpy.features.retouch.logic import (
        apply_ir_attenuation,
        apply_ir_reconstruction,
        downsample_ir,
        ir_defect_score,
        ir_detect_cutoff,
        ir_detect_target,
        ir_ratio_and_gain,
    )

    target = ir_detect_target(max(rgb.shape[:2]), max(rgb.shape[:2]))
    ir_det = downsample_ir(np.ascontiguousarray(ir, dtype=np.float32), target)
    h, w = rgb.shape[:2]
    if max(h, w) > target:
        import cv2

        s = target / max(h, w)
        rgb_det = cv2.resize(rgb, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)
    else:
        rgb_det = rgb
    ratio_det, gain_det, degenerate, _ = ir_ratio_and_gain(ir_det, rgb_det)
    if degenerate:
        return rgb
    score_det = ir_defect_score(ratio_det, ir_detect_cutoff(retouch.ir_threshold, retouch.ir_attenuation))
    out = apply_ir_attenuation(rgb, gain_det) if retouch.ir_attenuation else rgb
    out = apply_ir_reconstruction(out, score_det)
    return out


def _decode_linear(
    file_path: str,
    geometry: Optional[GeometryConfig] = None,
    expansion: Optional[float] = None,
    rgbscan: Optional[RgbScanConfig] = None,
    stitch: Optional[StitchConfig] = None,
    flatfield: Optional[FlatFieldConfig] = None,
    process: Optional[ProcessConfig] = None,
    apply_wb: bool = False,
    apply_flatfield: bool = False,
    apply_sensor: bool = False,
    gamma_key: str = "linear",
) -> tuple[np.ndarray, Optional[np.ndarray], Optional[_CameraWB], _SourceMeta]:
    """Decode to an oriented float32 buffer. Returns (rgb, ir_or_none, camera_wb_or_none, source_meta)."""
    if stitch is not None and stitch.stitch_enabled and stitch.stitch_paths:
        rgb, ir, wb, meta = _decode_stitch(file_path, stitch, geometry, flatfield, process)
        if apply_wb and wb is not None:
            rgb = _apply_white_balance(rgb, wb)
        return rgb, ir, wb, meta
    if PakonLoader.can_handle(file_path):
        rgb, ir = _decode_pakon(file_path, geometry, expansion=expansion)
        meta = _SourceMeta(make="Pakon", model=_pakon_spec_desc(file_path))
        return rgb, ir, None, meta
    if _is_dng(file_path):
        meta = _read_source_meta_tiff(file_path)
        if _is_linearraw_dng(file_path):
            rgb, ir = _decode_dng(file_path, geometry, expansion=expansion)
            return rgb, ir, None, meta
        if _is_camera_raw(file_path):
            rgb, ir, wb = _decode_camera_raw(file_path, geometry)
            return rgb, ir, wb, meta
    if is_coolscan_nef(file_path):
        meta = _read_source_meta_tiff(file_path)
        rgb, ir = _decode_nef(file_path, geometry, gamma_key=gamma_key)
        return rgb, ir, None, meta
    if is_flextight_fff(file_path):
        meta = _read_fff_meta(file_path)
        rgb, ir = _decode_fff(file_path, geometry)
        return rgb, ir, None, meta
    if is_noritsu_raw(file_path):
        rgb, ir = _decode_noritsu(file_path, geometry, expansion=expansion)
        meta = _SourceMeta(make="Noritsu")
        return rgb, ir, None, meta
    if _is_camera_raw(file_path):
        if rgbscan is not None and is_rgb_triplet(rgbscan):
            rgb, ir, wb, meta = _decode_camera_raw_triplet(file_path, rgbscan, geometry)
            if apply_flatfield and flatfield is not None:
                rgb = _apply_flatfield_correction(rgb, flatfield)
            if apply_wb and wb is not None:
                rgb = _apply_white_balance(rgb, wb)
            return rgb, ir, wb, meta
        meta = _read_source_meta_tiff(file_path)
        rgb, ir, wb, decode_meta = _decode_camera_raw(file_path, geometry)
        merged = _SourceMeta(
            make=meta.make or decode_meta.make,
            model=meta.model or decode_meta.model,
            datetime=meta.datetime or decode_meta.datetime,
        )
        if apply_flatfield and flatfield is not None:
            rgb = _apply_flatfield_correction(rgb, flatfield)
        if apply_sensor and process is not None and process.sensor_matrix is not None:
            rgb = apply_sensor_correction(rgb, process.sensor_matrix)
        if apply_wb and wb is not None:
            rgb = _apply_white_balance(rgb, wb)
        return rgb, ir, wb, merged
    if _is_tiff(file_path):
        meta = _read_source_meta_tiff(file_path)
        rgb, ir = _decode_tiff(file_path, geometry, gamma_key=gamma_key, expansion=expansion)
        return rgb, ir, None, meta
    raise ValueError(f"Linear Output is not supported for this file type: {file_path}")


PAKON_EXPANSION = 4.0
NORITSU_EXPANSION = 16.0
_F335_SIZE = 72000000


def _is_pakon_f335(file_path: str) -> bool:
    try:
        return abs(os.path.getsize(file_path) - _F335_SIZE) < 1024
    except OSError:
        return False


def _default_pakon_expansion(file_path: str) -> float:
    # F335 is 16-bit; all others assumed 14-bit (confirmed for F135, unverified for 2k Square / Panoram).
    return 1.0 if _is_pakon_f335(file_path) else PAKON_EXPANSION


def _pakon_spec_desc(file_path: str) -> str:
    try:
        file_size = os.path.getsize(file_path)
        spec = next((s for s in PakonLoader.PAKON_SPECS if abs(file_size - s["size"]) < 1024), None)
        return spec["desc"] if spec else "Unknown"
    except OSError:
        return "Unknown"


def _decode_tiff(
    file_path: str,
    geometry: Optional[GeometryConfig] = None,
    gamma_key: str = "linear",
    expansion: Optional[float] = None,
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """Read a TIFF, optionally linearize. Returns (rgb, ir_or_none)."""
    from negpy.infrastructure.loaders.ir_planes import find_ir_plane
    from negpy.infrastructure.loaders.tiff_loader import _extract_ir_from_extrasamples, _read_sidecar_ir

    with _tifffile.TiffFile(file_path) as tif:
        page = tif.pages[0]
        arr = page.asarray()
    if arr.dtype == np.uint16:
        scale = 1.0 / 65535.0
    elif arr.dtype == np.uint8:
        scale = 1.0 / 255.0
    elif arr.dtype == np.float32:
        scale = 1.0
    else:
        scale = 1.0 / float(np.iinfo(arr.dtype).max) if np.issubdtype(arr.dtype, np.integer) else 1.0
    f32 = arr.astype(np.float32) * scale

    ir: Optional[np.ndarray] = None
    if f32.ndim == 3 and f32.shape[2] == 4:
        f32, ir = _extract_ir_from_extrasamples(file_path, f32)
    elif f32.ndim == 2:
        f32 = np.stack([f32, f32, f32], axis=2)

    if ir is None:
        try:
            with _tifffile.TiffFile(file_path) as tif:
                ir = find_ir_plane(tif.pages[1:], f32.shape[0], f32.shape[1])
        except Exception:
            pass

    if ir is None:
        ir_result, _mask = _read_sidecar_ir(file_path)
        ir = ir_result

    f32 = np.clip(f32, 0.0, 1.0)
    if gamma_key != "linear":
        f32 = _linearize(f32, gamma_key)
    if expansion is not None and expansion > 1.0:
        f32 = np.clip(f32 * expansion, 0.0, 1.0)
    orientation = read_orientation(file_path)
    f32 = _apply_geometry(f32, orientation, geometry)
    if ir is not None:
        ir = _apply_geometry(ir, orientation, geometry)
    return f32, ir


def _decode_pakon(file_path: str, geometry: Optional[GeometryConfig] = None, expansion: Optional[float] = None) -> tuple[np.ndarray, None]:
    loader = PakonLoader()
    ctx_mgr, metadata = loader.load(file_path)
    with ctx_mgr as wrapper:
        if not isinstance(wrapper, NonStandardFileWrapper):
            raise TypeError("Expected NonStandardFileWrapper from PakonLoader")
        f32 = wrapper.data
    factor = expansion if expansion is not None else _default_pakon_expansion(file_path)
    if factor > 1.0:
        f32 = np.clip(f32 * factor, 0.0, 1.0)
    f32 = _apply_geometry(f32, metadata.get("orientation", 0), geometry)
    return f32, None


def _decode_via_loader(
    loader: Any,
    file_path: str,
    geometry: Optional[GeometryConfig] = None,
    gamma_key: str = "linear",
    expansion: Optional[float] = None,
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """Decode through a main-path loader with linear_raw=True, then apply geometry."""
    ctx_mgr, metadata = loader.load(file_path, linear_raw=True)
    with ctx_mgr as wrapper:
        f32 = wrapper.data if isinstance(wrapper, NonStandardFileWrapper) else np.asarray(wrapper)
    ir = metadata.get("ir")
    f32 = np.clip(f32, 0.0, 1.0)
    if gamma_key != "linear":
        f32 = _linearize(f32, gamma_key)
    if expansion is not None and expansion > 1.0:
        f32 = np.clip(f32 * expansion, 0.0, 1.0)
    orientation = metadata.get("orientation", 0)
    f32 = _apply_geometry(f32, orientation, geometry)
    if ir is not None:
        ir = _apply_geometry(ir, orientation, geometry)
    return f32, ir


def _decode_nef(
    file_path: str,
    geometry: Optional[GeometryConfig] = None,
    gamma_key: str = "linear",
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """Read a Coolscan NEF via the main loader. Returns (rgb, ir_or_none)."""
    from negpy.infrastructure.loaders.nef_loader import NefLoader

    return _decode_via_loader(NefLoader(), file_path, geometry, gamma_key)


def _decode_fff(
    file_path: str,
    geometry: Optional[GeometryConfig] = None,
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """Read a Flextight FFF via the main loader. Returns (rgb, ir_or_none)."""
    from negpy.infrastructure.loaders.fff_loader import FffLoader

    return _decode_via_loader(FffLoader(), file_path, geometry)


def _decode_noritsu(
    file_path: str,
    geometry: Optional[GeometryConfig] = None,
    expansion: Optional[float] = None,
) -> tuple[np.ndarray, None]:
    """Read a headerless Noritsu EZController RAW. Returns (rgb, None)."""
    dims = detect_noritsu_dims(file_path)
    if dims is None:
        raise ValueError(f"Unknown Noritsu dimensions for {file_path}")
    w, h = dims
    with open(file_path, "rb") as f:
        data = np.fromfile(f, dtype="<u2", count=w * h * 3)
    arr = data.reshape(h, w, 3)[:, :, ::-1]  # BGR → RGB
    f32 = arr.astype(np.float32) / 65535.0
    factor = expansion if expansion is not None else NORITSU_EXPANSION
    if factor > 1.0:
        f32 = np.clip(f32 * factor, 0.0, 1.0)
    f32 = _apply_geometry(f32, 0, geometry)
    return f32, None


def _decode_dng(
    file_path: str, geometry: Optional[GeometryConfig] = None, expansion: Optional[float] = None
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    peeked_4ch = _peek_linearraw_4ch(file_path)
    if peeked_4ch is not None:
        rgb, ir = peeked_4ch
        if expansion is not None and expansion > 1.0:
            rgb = np.clip(rgb * expansion, 0.0, 1.0)
        orientation = read_orientation(file_path)
        rgb = _apply_geometry(rgb, orientation, geometry)
        ir = _apply_geometry(ir, orientation, geometry)
        return rgb, ir

    # 3-channel LinearRaw (SilverFast HDRi): read directly via tifffile.
    try:
        with _tifffile.TiffFile(file_path) as tif:
            page = _find_linearraw_page(tif, samples=3)
            if page is None:
                raise ValueError(f"No LinearRaw IFD found in {file_path}")
            arr = page.asarray()
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"Failed to read LinearRaw data from {file_path}: {e}") from e

    if arr.dtype == np.uint16:
        scale = 1.0 / 65535.0
    elif arr.dtype == np.uint8:
        scale = 1.0 / 255.0
    else:
        scale = 1.0
    rgb = np.clip(arr.astype(np.float32) * scale, 0.0, 1.0)
    if expansion is not None and expansion > 1.0:
        rgb = np.clip(rgb * expansion, 0.0, 1.0)

    ir = _peek_hdri_ir_page(file_path)
    orientation = read_orientation(file_path)
    rgb = _apply_geometry(rgb, orientation, geometry)
    if ir is not None:
        ir = _apply_geometry(ir, orientation, geometry)
    return rgb, ir


def _decode_camera_raw_buffer(file_path: str) -> tuple[np.ndarray, _CameraWB, _SourceMeta]:
    """Decode a camera RAW to an oriented float32 buffer without applying user geometry.

    Returns (f32, camera_wb, source_meta).  EXIF orientation *is* applied (lossless,
    baked into the file) but user rotation/flip is not — the caller decides that.
    """
    raw = rawpy.imread(file_path)
    wb = _CameraWB(
        as_shot=tuple(raw.camera_whitebalance),  # type: ignore[arg-type]
        daylight=tuple(raw.daylight_whitebalance),  # type: ignore[arg-type]
    )
    ts = raw.other.timestamp
    dt_str = ts.strftime("%Y:%m:%d %H:%M:%S") if ts else None
    algo = get_best_demosaic_algorithm(raw)
    rgb = raw.postprocess(
        gamma=(1, 1),
        no_auto_bright=True,
        user_wb=[1, 1, 1, 1],
        output_bps=16,
        output_color=rawpy.ColorSpace.raw,
        demosaic_algorithm=algo,
        user_flip=0,
        adjust_maximum_thr=0.0,
    )
    raw.close()
    rgb = ensure_rgb(rgb)
    f32 = uint16_to_float32(rgb)
    orientation = read_orientation(file_path)
    f32 = apply_exif_orientation(f32, orientation)
    meta = _SourceMeta(datetime=dt_str)
    return f32, wb, meta


def _decode_camera_raw(file_path: str, geometry: Optional[GeometryConfig] = None) -> tuple[np.ndarray, None, _CameraWB, _SourceMeta]:
    f32, wb, meta = _decode_camera_raw_buffer(file_path)
    if geometry is not None:
        f32 = _apply_user_geometry(f32, geometry)
    return f32, None, wb, meta


def _decode_camera_raw_triplet(
    file_path: str, rgbscan: RgbScanConfig, geometry: Optional[GeometryConfig] = None
) -> tuple[np.ndarray, None, Optional[_CameraWB], _SourceMeta]:
    """Decode three narrowband exposures and merge into one RGB buffer."""
    primary_f32, wb, meta = _decode_camera_raw_buffer(file_path)
    file_meta = _read_source_meta_tiff(file_path)
    merged_meta = _SourceMeta(
        make=file_meta.make or meta.make,
        model=file_meta.model or meta.model,
        datetime=file_meta.datetime or meta.datetime,
    )

    cache: dict[str, np.ndarray] = {file_path: primary_f32}

    def _decode(path: str) -> np.ndarray:
        if path in cache:
            return cache[path]
        buf, _, _ = _decode_camera_raw_buffer(path)
        cache[path] = buf
        return buf

    f32 = merge_rgb_triplet(_decode, file_path, rgbscan.green_path, rgbscan.blue_path, align=rgbscan.align)
    if geometry is not None:
        f32 = _apply_user_geometry(f32, geometry)
    return f32, None, wb, merged_meta


def _decode_stitch_part(
    file_path: str,
    rgbscan: Optional[RgbScanConfig],
    flatfield: Optional[FlatFieldConfig],
    process: Optional[ProcessConfig],
) -> np.ndarray:
    """Decode one stitch part with flatfield and sensor correction applied.

    Triplet merge is performed when *rgbscan* is a valid triplet config.
    Sensor correction is skipped for triplets (no cross-channel leakage
    with narrowband exposures).
    """
    is_triplet = rgbscan is not None and is_rgb_triplet(rgbscan)

    if is_triplet:
        primary_f32, _, _ = _decode_camera_raw_buffer(file_path)
        cache: dict[str, np.ndarray] = {file_path: primary_f32}

        def _decode(path: str) -> np.ndarray:
            if path in cache:
                return cache[path]
            buf, _, _ = _decode_camera_raw_buffer(path)
            cache[path] = buf
            return buf

        f32 = merge_rgb_triplet(_decode, file_path, rgbscan.green_path, rgbscan.blue_path, align=rgbscan.align)
    else:
        f32, _, _ = _decode_camera_raw_buffer(file_path)

    if flatfield is not None:
        f32 = _apply_flatfield_correction(f32, flatfield)
    if not is_triplet and process is not None and process.sensor_matrix is not None:
        f32 = apply_sensor_correction(f32, process.sensor_matrix)
    return f32


def _decode_stitch(
    file_path: str,
    stitch: StitchConfig,
    geometry: Optional[GeometryConfig],
    flatfield: Optional[FlatFieldConfig],
    process: Optional[ProcessConfig],
) -> tuple[np.ndarray, None, Optional[_CameraWB], _SourceMeta]:
    """Decode all stitch parts, apply per-part corrections, and assemble."""
    all_paths = [file_path, *stitch.stitch_paths]
    has_triplets = stitch_has_triplets(stitch)

    primary_meta = _read_source_meta_tiff(file_path)
    _, wb, decode_meta = _decode_camera_raw_buffer(file_path)
    merged_meta = _SourceMeta(
        make=primary_meta.make or decode_meta.make,
        model=primary_meta.model or decode_meta.model,
        datetime=primary_meta.datetime or decode_meta.datetime,
    )

    parts: list[np.ndarray] = []
    for i, path in enumerate(all_paths):
        part_rgbscan: Optional[RgbScanConfig] = None
        if i < len(stitch.stitch_triplets):
            green, blue = stitch.stitch_triplets[i]
            if green and blue:
                part_rgbscan = RgbScanConfig(enabled=True, green_path=green, blue_path=blue, align=stitch.stitch_align)
        parts.append(_decode_stitch_part(path, part_rgbscan, flatfield, process))

    irs: list[None] = [None] * len(parts)
    f32, _ = stitch_composite(parts, irs, stitch)
    if geometry is not None:
        f32 = _apply_user_geometry(f32, geometry)
    return f32, None, wb if not has_triplets else None, merged_meta


def _normalize_wb_rgb(wb: tuple[float, float, float, float]) -> tuple[float, float, float]:
    """Normalize RGGB multipliers to green=1, return (R, G, B)."""
    g = (wb[1] + wb[3]) / 2.0 if (wb[1] + wb[3]) > 0 else 1.0
    return (wb[0] / g, 1.0, wb[2] / g)


def _build_xmp(source_path: str, wb: _CameraWB, title: str = "", wb_applied: bool = False) -> bytes:
    raw_name = os.path.basename(source_path)
    title_block = ""
    if title:
        title_block = f"  <dc:title>\n   <rdf:Alt>\n    <rdf:li xml:lang='x-default'>{title}</rdf:li>\n   </rdf:Alt>\n  </dc:title>\n"
    desc_block = ""
    if not wb_applied:
        r, g, b = _normalize_wb_rgb(wb.as_shot)
        desc_block = (
            "  <dc:description>\n"
            "   <rdf:Alt>\n"
            f"    <rdf:li xml:lang='x-default'>RAW-WB: {r:.6f} {g:.6f} {b:.6f}</rdf:li>\n"
            "   </rdf:Alt>\n"
            "  </dc:description>\n"
        )
    xmp = (
        "<?xpacket begin='﻿' id='W5M0MpCehiHzreSzNTczkc9d'?>\n"
        "<x:xmpmeta xmlns:x='adobe:ns:meta/'>\n"
        "<rdf:RDF xmlns:rdf='http://www.w3.org/1999/02/22-rdf-syntax-ns#'>\n"
        " <rdf:Description rdf:about=''\n"
        "  xmlns:crs='http://ns.adobe.com/camera-raw-settings/1.0/'>\n"
        f"  <crs:RawFileName>{raw_name}</crs:RawFileName>\n"
        " </rdf:Description>\n"
        " <rdf:Description rdf:about=''\n"
        "  xmlns:dc='http://purl.org/dc/elements/1.1/'>\n"
        f"{title_block}"
        f"{desc_block}"
        " </rdf:Description>\n"
        "</rdf:RDF>\n"
        "</x:xmpmeta>\n"
        "<?xpacket end='w'?>"
    )
    return xmp.encode("utf-8")


def _effective_expansion(file_path: str, expansion: Optional[float]) -> float:
    if PakonLoader.can_handle(file_path):
        factor = expansion if expansion is not None else _default_pakon_expansion(file_path)
        return factor if factor > 1.0 else 1.0
    if is_noritsu_raw(file_path):
        factor = expansion if expansion is not None else NORITSU_EXPANSION
        return factor if factor > 1.0 else 1.0
    if _is_dng(file_path) and _is_linearraw_dng(file_path):
        return expansion if (expansion is not None and expansion > 1.0) else 1.0
    if _is_tiff(file_path):
        return expansion if (expansion is not None and expansion > 1.0) else 1.0
    return 1.0


def _source_format_label(
    file_path: str,
    rgbscan: Optional[RgbScanConfig] = None,
    stitch: Optional[StitchConfig] = None,
) -> str:
    is_stitch = stitch is not None and stitch.stitch_enabled and stitch.stitch_paths
    if PakonLoader.can_handle(file_path):
        return f"Pakon {_pakon_spec_desc(file_path)}"
    if _is_dng(file_path) and _is_linearraw_dng(file_path):
        return "DNG LinearRaw"
    if is_coolscan_nef(file_path):
        return "Coolscan NEF"
    if is_flextight_fff(file_path):
        return "Flextight FFF"
    if is_noritsu_raw(file_path):
        return "Noritsu RAW"
    if _is_camera_raw(file_path):
        if is_stitch and stitch_has_triplets(stitch):
            n = 1 + len(stitch.stitch_paths)
            return f"camera RAW (stitch {n}-part, RGB triplet)"
        if is_stitch:
            n = 1 + len(stitch.stitch_paths)
            return f"camera RAW (stitch {n}-part)"
        if rgbscan is not None and is_rgb_triplet(rgbscan):
            return "camera RAW (RGB triplet)"
        return "camera RAW"
    if _is_tiff(file_path):
        return "TIFF"
    return "unknown"


def _parse_tiff_datetime(dt_str: Optional[str]) -> Optional[str]:
    """Validate/normalise a TIFF DateTime string to ``YYYY:MM:DD HH:MM:SS``."""
    if not dt_str:
        return None
    from datetime import datetime

    for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y.%m.%d %H.%M.%S", "%Y:%m:%d", "%Y-%m-%d"):
        try:
            parsed = datetime.strptime(dt_str.strip(), fmt)
            return parsed.strftime("%Y:%m:%d %H:%M:%S")
        except ValueError:
            continue
    return None


def _write_tiff(
    f32: np.ndarray,
    dest,
    source_name: str,
    camera_wb: Optional[_CameraWB] = None,
    source_path: Optional[str] = None,
    source_meta: Optional[_SourceMeta] = None,
    expansion: float = 1.0,
    source_format: str = "",
    wb_applied: bool = False,
    flatfield_applied: bool = False,
    sensor_applied: bool = False,
    ice_applied: bool = False,
    gamma_key: str = "linear",
) -> None:
    """Write a float32 buffer as an untagged 16-bit TIFF to *dest* (path or file-like)."""
    u16 = _to_uint16_jit(np.ascontiguousarray(f32, dtype=np.float32))
    photometric = "rgb" if f32.ndim == 3 else "minisblack"
    parts = [f"source: {source_format or source_name}"]
    if expansion > 1.0:
        parts.append(f"expansion: x{expansion:g}")
    else:
        parts.append("no scaling")
    if gamma_key != "linear":
        gamma_labels = dict(TIFF_GAMMA_OPTIONS)
        parts.append(f"linearized from {gamma_labels.get(gamma_key, gamma_key)}")
    if camera_wb is not None:
        r, g, b = _normalize_wb_rgb(camera_wb.as_shot)
        if wb_applied:
            parts.append(f"WB applied (as-shot: {r:.3f} {g:.3f} {b:.3f})")
        else:
            parts.append(f"no WB applied (as-shot: {r:.3f} {g:.3f} {b:.3f})")
    else:
        parts.append("no WB applied")
    corrections = [s for s, on in (("flatfield", flatfield_applied), ("sensor", sensor_applied), ("ICE", ice_applied)) if on]
    if corrections:
        parts.append(f"corrections: {', '.join(corrections)}")
    parts.append("no color management")
    description = f"NegPy Linear Output -- {', '.join(parts)}."

    extratags: list[tuple] = []
    dt: Optional[str] = None
    try:
        if camera_wb is not None and source_path is not None:
            xmp_bytes = _build_xmp(source_path, camera_wb, title=description, wb_applied=wb_applied)
            extratags.append((700, 1, len(xmp_bytes), xmp_bytes, True))
        if source_meta is not None:
            if source_meta.make:
                extratags.append((271, 2, 0, source_meta.make, True))
            if source_meta.model:
                extratags.append((272, 2, 0, source_meta.model, True))
        dt = _parse_tiff_datetime((source_meta.datetime if source_meta else None) or None)
    except Exception:
        extratags = []
        dt = None

    _tifffile.imwrite(
        dest,
        u16,
        photometric=photometric,
        compression="zlib",
        predictor=True,
        description=description,
        software="NegPy",
        datetime=dt,
        extratags=extratags or None,
        metadata=None,
    )


def _write_ir_tiff(ir: np.ndarray, dest, source_name: str) -> None:
    """Write a single-channel IR buffer as an untagged 16-bit grayscale TIFF."""
    u16 = _to_uint16_jit(np.ascontiguousarray(ir[:, :, np.newaxis] if ir.ndim == 2 else ir, dtype=np.float32))
    if u16.ndim == 3 and u16.shape[2] == 1:
        u16 = u16[:, :, 0]
    description = f"NegPy Linear Output -- infrared channel. Source: {source_name}"
    _tifffile.imwrite(
        dest,
        u16,
        photometric="minisblack",
        compression="zlib",
        predictor=True,
        description=description,
    )


def export_linear_output(
    file_path: str,
    output_path: str,
    geometry: Optional[GeometryConfig] = None,
    expansion: Optional[float] = None,
    rgbscan: Optional[RgbScanConfig] = None,
    stitch: Optional[StitchConfig] = None,
    flatfield: Optional[FlatFieldConfig] = None,
    process: Optional[ProcessConfig] = None,
    apply_wb: bool = False,
    apply_flatfield: bool = False,
    apply_sensor: bool = False,
    apply_ice: bool = False,
    retouch: Optional[RetouchConfig] = None,
    gamma_key: str = "linear",
) -> None:
    """Decode *file_path* and write an untagged linear 16-bit TIFF to *output_path*.

    Lossless geometry (90-degree rotation, horizontal/vertical flip) from *geometry*
    is applied; fine rotation is ignored (it resamples).

    *expansion* scales the linear data before writing (e.g. 4.0 for Pakon's 14-bit
    sensor → 16-bit range). ``None`` uses the source-type default; values <= 1.0 disable.

    *rgbscan*, when a valid triplet config, merges three narrowband exposures into
    one combined RGB buffer before writing.

    *stitch*, when active, decodes all parts (with per-part triplet merge if
    applicable), applies flatfield and sensor correction per-part, then assembles
    via stitch_composite.

    *apply_wb*, *apply_flatfield*, *apply_sensor*, *apply_ice*: optional per-step
    corrections.  When False (default), the raw dump is written unchanged.  When
    True, the corresponding correction is applied before writing.  *apply_ice*
    requires an IR channel in the source and a *retouch* config; it uses the
    configured IR method and threshold.

    If the source has an IR channel, it is written as a separate grayscale TIFF
    with an ``_ir`` suffix next to the RGB output.
    """
    eff = _effective_expansion(file_path, expansion)
    fmt = _source_format_label(file_path, rgbscan, stitch)
    f32, ir, camera_wb, meta = _decode_linear(
        file_path,
        geometry,
        expansion=expansion,
        rgbscan=rgbscan,
        stitch=stitch,
        flatfield=flatfield,
        process=process,
        apply_wb=apply_wb,
        apply_flatfield=apply_flatfield,
        apply_sensor=apply_sensor,
        gamma_key=gamma_key,
    )
    ice_applied = False
    if apply_ice and ir is not None:
        ret = retouch if retouch is not None else RetouchConfig()
        f32 = _apply_ice(f32, ir, ret)
        ice_applied = True
    is_stitch = stitch is not None and stitch.stitch_enabled and stitch.stitch_paths
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    _write_tiff(
        f32,
        output_path,
        os.path.basename(file_path),
        camera_wb,
        source_path=file_path,
        source_meta=meta,
        expansion=eff,
        source_format=fmt,
        wb_applied=apply_wb,
        flatfield_applied=apply_flatfield or is_stitch,
        sensor_applied=apply_sensor or is_stitch,
        ice_applied=ice_applied,
        gamma_key=gamma_key,
    )

    if ir is not None and not ice_applied:
        stem, ext = os.path.splitext(output_path)
        ir_path = f"{stem}_ir{ext}"
        _write_ir_tiff(ir, ir_path, os.path.basename(file_path))


def export_linear_output_bytes(file_path: str, geometry: Optional[GeometryConfig] = None) -> tuple[bytes, str]:
    """Like export_linear_output but returns (tiff_bytes, filename_stem) for in-memory use.

    IR is not included in the returned bytes (use export_linear_output for IR).
    """
    eff = _effective_expansion(file_path, None)
    fmt = _source_format_label(file_path)
    f32, _ir, camera_wb, meta = _decode_linear(file_path, geometry)
    buf = io.BytesIO()
    _write_tiff(f32, buf, os.path.basename(file_path), camera_wb, source_path=file_path, source_meta=meta, expansion=eff, source_format=fmt)
    stem = os.path.splitext(os.path.basename(file_path))[0]
    return buf.getvalue(), stem
