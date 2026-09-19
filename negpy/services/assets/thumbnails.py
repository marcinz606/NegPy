from typing import Any, Callable, Dict, Optional, Tuple

from PIL import Image
import rawpy
from negpy.kernel.system.config import APP_CONFIG
import numpy as np
from negpy.kernel.image.logic import apply_exif_orientation, ensure_rgb, float_to_uint8, prepare_thumbnail, srgb_to_linear, uint8_to_float32
from negpy.infrastructure.capture.raw_demosaic import _user_sat
from negpy.infrastructure.loaders.factory import loader_factory
from negpy.infrastructure.loaders.helpers import NonStandardFileWrapper, embedded_preview
from negpy.infrastructure.display.color_spaces import WORKING_COLOR_SPACE
from negpy.kernel.system.logging import get_logger

logger = get_logger(__name__)


def asset_thumbnail_key(asset: Dict[str, Any]) -> str:
    """The one key an asset's thumbnail lives under, in memory and on disk.

    Keyed by identity rather than display name: two same-named files from different
    folders would otherwise overwrite each other's thumbnail, and a triplet's merged
    thumb would be shadowed by its red exposure's.
    """
    return thumbnail_cache_key(asset["hash"], bool(asset.get("green_path") and asset.get("blue_path")))


def thumbnail_cache_key(file_hash: str, is_triplet: bool) -> str:
    """Disk cache key for a frame's thumbnail.

    A triplet caches under a distinct key so (a) a red-only thumbnail cached before
    merge support can't shadow the corrected one, and (b) the batch (source) path and
    the rendered-positive path agree on where the triplet's thumbnail lives — the
    rendered positive is what makes the filmstrip match the canvas.

    The version suffix retires caches written before the batch path inverted negatives;
    without it an existing library keeps serving its stored negatives forever. v3 retires
    the ones written before the batch path read the frame's saved film process, which is
    where a slide got cached as an inverted negative."""
    base = f"{file_hash}-rgb" if is_triplet else file_hash
    return f"{base}-v3"


def _fast_demosaic(raw: Any) -> np.ndarray:
    """Fast half-size linear demosaic used for preview thumbnails."""
    # NonStandardFileWrapper has no camera calibration to read; its postprocess ignores user_sat anyway.
    user_sat = None if isinstance(raw, NonStandardFileWrapper) else _user_sat(raw)
    return ensure_rgb(
        raw.postprocess(
            use_camera_wb=True,
            user_wb=None,
            adjust_maximum_thr=0.0,
            user_sat=user_sat,  # calibrated linearity limit, not the format's generic max
            half_size=True,
            no_auto_bright=True,
            bright=1.0,
            demosaic_algorithm=rawpy.DemosaicAlgorithm.LINEAR,
            user_flip=0,
        )
    )


def _decode_triplet_preview(red_path: str, green_path: str, blue_path: str) -> Optional[Image.Image]:
    """Merge an RGB-scan triplet's three narrowband exposures into one preview.

    The red file's embedded thumbnail (and a lone decode of it) shows only the red
    channel, so the filmstrip would disagree with the canvas. Alignment is skipped:
    it's imperceptible at thumbnail scale and avoids a phase-correlation pass per file."""
    from negpy.features.rgbscan.logic import assemble_rgb

    def _decode(path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        ctx_mgr, metadata = loader_factory.get_loader(path)
        with ctx_mgr as raw:
            return _fast_demosaic(raw), metadata

    r, red_meta = _decode(red_path)
    g, _ = _decode(green_path)
    b, _ = _decode(blue_path)
    merged = assemble_rgb(r, g, b, align=False)

    img = Image.fromarray(merged)
    orientation = red_meta.get("orientation", 1)
    if orientation and orientation != 1:
        img = Image.fromarray(apply_exif_orientation(np.asarray(img), orientation))
    return img


def decode_source_image(
    file_path: str,
    green_path: str = "",
    blue_path: str = "",
) -> Optional[Image.Image]:
    """Small EXIF-oriented preview of a source file (embedded thumb, else fast decode).

    An RGB-scan triplet (green_path/blue_path given) is merged from its three
    exposures so the thumbnail matches the canvas rather than showing red only."""
    if green_path and blue_path:
        return _decode_triplet_preview(file_path, green_path, blue_path)

    ctx_mgr, metadata = loader_factory.get_loader(file_path)
    with ctx_mgr as raw:
        img: Optional[Image.Image] = embedded_preview(raw, file_path)

        if img is None:
            img = Image.fromarray(_fast_demosaic(raw))

        orientation = metadata.get("orientation", 1)
        if orientation and orientation != 1:
            img = Image.fromarray(apply_exif_orientation(np.asarray(img), orientation))

        return img


def decode_bounded_source_preview(
    file_path: str,
    green_path: str = "",
    blue_path: str = "",
    *,
    max_edge: int,
    fast_only: bool = False,
    should_cancel: Optional[Callable[[], bool]] = None,
) -> Optional[Image.Image]:
    """Load a source through its format loader's bounded-preview contract."""

    def decode_one(path: str) -> Optional[Image.Image]:
        if should_cancel is not None and should_cancel():
            raise InterruptedError("thumbnail cancelled")
        return loader_factory.load_bounded_preview(
            path,
            max_edge,
            fast_only=fast_only,
            should_cancel=should_cancel,
        )

    if not green_path or not blue_path:
        return decode_one(file_path)

    from negpy.features.rgbscan.logic import assemble_rgb

    red_image = decode_one(file_path)
    green_image = decode_one(green_path)
    blue_image = decode_one(blue_path)
    if red_image is None or green_image is None or blue_image is None:
        return None
    red = np.asarray(red_image.convert("RGB"))
    green = np.asarray(green_image.convert("RGB"))
    blue = np.asarray(blue_image.convert("RGB"))
    return Image.fromarray(assemble_rgb(red, green, blue, align=False))


def preview_positive(img: Image.Image, process_mode: str = "") -> Image.Image:
    """Invert a negative preview so the filmstrip reads as photographs before a frame
    has ever been opened.

    ``process_mode`` is the frame's *stored* film process; it decides outright, and the
    heuristic runs only when nothing has decided yet. Detection here reads an 8-bit
    preview — for a slide that is an already-positive embedded JPEG, and warm content
    trips the orange-mask test into calling it a negative, which is how a transparency
    reached the filmstrip inverted while the canvas rendered it right.

    Deliberately not the pipeline: per-channel log-density bounds over an 8-bit preview,
    with no config to key on. It is a placeholder that get_rendered_thumbnail supersedes
    the moment the frame renders, so drift from the engine costs nothing.
    """
    from negpy.features.process.logic import detect_process_mode
    from negpy.features.process.models import ProcessMode

    arr = np.asarray(img.convert("RGB"), dtype=np.uint8)
    linear = srgb_to_linear(uint8_to_float32(arr))
    # ProcessMode("") would resolve to C41 via _missing_, so an unknown mode must not reach it.
    mode = ProcessMode(process_mode) if process_mode else detect_process_mode(linear)
    if mode is ProcessMode.E6:
        return img

    # A lone narrowband exposure carries its picture in one channel, and the other two hold
    # only near-black noise that the log below would stretch into a solid color cast.
    # Borrowing the channel that has the signal renders it as the monochrome it is.
    peaks = np.percentile(linear, 99.0, axis=(0, 1))
    strongest = int(np.argmax(peaks))
    empty = peaks < peaks[strongest] * 0.05
    if empty.any():
        linear = linear.copy()
        linear[:, :, empty] = linear[:, :, strongest : strongest + 1]

    density = -np.log10(np.clip(linear, 1e-4, None))
    lo = np.percentile(density, 1.0, axis=(0, 1))
    hi = np.percentile(density, 99.0, axis=(0, 1))
    positive = (density - lo) / np.maximum(hi - lo, 1e-6)
    return Image.fromarray(float_to_uint8(np.clip(positive, 0.0, 1.0)))


def get_thumbnail_worker(
    file_path: str,
    file_hash: str,
    asset_store: Any = None,
    half: int = 0,
    split_x: float = 0.5,
    green_path: str = "",
    blue_path: str = "",
    crop_rect: Optional[tuple[float, float, float, float]] = None,
    gutter_thickness: float = 0.0,
    process_mode: str = "",
    *,
    fast_only: bool = False,
    should_cancel: Optional[Callable[[], bool]] = None,
) -> Optional[Image.Image]:
    """
    Checks cache -> extracts/renders -> resize.
    """
    try:
        cache_key = thumbnail_cache_key(file_hash, bool(green_path and blue_path))
        if asset_store:
            cached = asset_store.get_thumbnail(cache_key)
            if isinstance(cached, Image.Image):
                return cached

        ts = APP_CONFIG.thumbnail_size
        img = decode_bounded_source_preview(
            file_path,
            green_path,
            blue_path,
            max_edge=ts * 2,
            fast_only=fast_only,
            should_cancel=should_cancel,
        )
        if img is None:
            return None

        if half:
            from negpy.services.assets.half_frame import slice_half

            img = Image.fromarray(slice_half(np.asarray(img), half, split_x, crop_rect=crop_rect, gutter_thickness=gutter_thickness))

        # Shrink before the inversion, not after: preview_positive is float math over every
        # pixel, and on a full-size decode its temporaries cost a gigabyte per worker.
        img.thumbnail((ts, ts), Image.Resampling.LANCZOS)
        square_img: Image.Image = prepare_thumbnail(preview_positive(img, process_mode), ts)

        if asset_store:
            asset_store.save_thumbnail(cache_key, square_img)

        return square_img
    except InterruptedError:
        return None
    except Exception as e:
        logger.error(f"Thumbnail Error for {file_path}: {e}")
        return None


def get_rendered_thumbnail(
    buffer: Any,
    file_hash: str,
    asset_store: Any = None,
    color_space: str = WORKING_COLOR_SPACE,
    monitor_icc_bytes: Optional[bytes] = None,
    proof: Optional[tuple] = None,
) -> Optional[Image.Image]:
    """
    Creates a thumbnail from a rendered float32 buffer, applying the same display
    transform the canvas applied to that buffer (mirrors ImageConverter.to_qimage).

    ``color_space``/``monitor_icc_bytes``/``proof`` must come from
    ``AppController.display_transform_params``. Rendered buffers are always in the
    working space: a soft proof rides the display LUT rather than the buffer, so
    dropping ``proof`` here leaves the filmstrip unproofed beside a proofed canvas.
    """
    try:
        from negpy.infrastructure.display.color_mgmt import apply_display_transform
        from negpy.kernel.image.logic import float_to_uint8

        ts = APP_CONFIG.thumbnail_size
        if isinstance(buffer, np.ndarray) and buffer.ndim == 3 and buffer.shape[2] == 4:
            buffer = buffer[:, :, :3]
        if isinstance(buffer, np.ndarray) and buffer.dtype == np.float32:
            buffer = apply_display_transform(buffer, color_space, monitor_icc_bytes, proof)
        u8_arr = float_to_uint8(buffer)
        img = Image.fromarray(u8_arr)

        square_img: Image.Image = prepare_thumbnail(img, ts)

        if asset_store:
            asset_store.save_thumbnail(file_hash, square_img)

        return square_img
    except Exception as e:
        logger.error(f"Rendered Thumbnail Error: {e}")
        return None
