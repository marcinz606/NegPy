import asyncio
from typing import Optional, Any, List, Dict, Tuple
from PIL import Image
import rawpy
from negpy.kernel.system.config import APP_CONFIG
import numpy as np
from negpy.kernel.image.logic import apply_exif_orientation, ensure_rgb, prepare_thumbnail
from negpy.infrastructure.loaders.factory import loader_factory
from negpy.infrastructure.display.color_spaces import WORKING_COLOR_SPACE
from negpy.kernel.system.logging import get_logger

logger = get_logger(__name__)


async def generate_batch_thumbnails(
    files: List[Dict[str, str]],
    asset_store: Any,
    progress_callback: Optional[Any] = None,
) -> Dict[str, Image.Image]:
    """
    Parallel thumbnail generation with progress reporting.
    """

    semaphore = asyncio.Semaphore(APP_CONFIG.max_workers)
    completed = 0

    async def _worker(f_info: Dict[str, str]) -> Tuple[str, Optional[Image.Image]]:
        nonlocal completed
        async with semaphore:
            thumb = await asyncio.to_thread(
                get_thumbnail_worker,
                f_info["path"],
                f_info["hash"],
                asset_store,
                int(f_info.get("half") or 0),
                float(f_info.get("split_x") or 0.5),
                f_info.get("green_path") or "",
                f_info.get("blue_path") or "",
            )
            completed += 1
            if progress_callback:
                if asyncio.iscoroutinefunction(progress_callback):
                    await progress_callback(completed, f_info["name"])
                else:
                    progress_callback(completed, f_info["name"])
            return f_info["name"], thumb

    tasks = [_worker(f) for f in files]
    results = await asyncio.gather(*tasks)

    return {name: thumb for name, thumb in results if isinstance(thumb, Image.Image)}


def thumbnail_cache_key(file_hash: str, is_triplet: bool) -> str:
    """Disk cache key for a frame's thumbnail.

    A triplet caches under a distinct key so (a) a red-only thumbnail cached before
    merge support can't shadow the corrected one, and (b) the batch (source) path and
    the rendered-positive path agree on where the triplet's thumbnail lives — the
    rendered positive is what makes the filmstrip match the canvas."""
    return f"{file_hash}-rgb" if is_triplet else file_hash


def _fast_demosaic(raw: Any) -> np.ndarray:
    """Fast half-size linear demosaic used for preview thumbnails."""
    return ensure_rgb(
        raw.postprocess(
            use_camera_wb=True,
            user_wb=None,
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


def decode_source_image(file_path: str, green_path: str = "", blue_path: str = "") -> Optional[Image.Image]:
    """Small EXIF-oriented preview of a source file (embedded thumb, else fast decode).

    An RGB-scan triplet (green_path/blue_path given) is merged from its three
    exposures so the thumbnail matches the canvas rather than showing red only."""
    if green_path and blue_path:
        return _decode_triplet_preview(file_path, green_path, blue_path)

    ctx_mgr, metadata = loader_factory.get_loader(file_path)
    with ctx_mgr as raw:
        img: Optional[Image.Image] = None

        if hasattr(raw, "extract_thumb"):
            try:
                thumb = raw.extract_thumb()
                if thumb.format == rawpy.ThumbFormat.JPEG:
                    import io

                    img = Image.open(io.BytesIO(thumb.data))
                elif thumb.format == rawpy.ThumbFormat.BITMAP:
                    img = Image.fromarray(thumb.data)
            except Exception:
                pass

        if img is None:
            img = Image.fromarray(_fast_demosaic(raw))

        orientation = metadata.get("orientation", 1)
        if orientation and orientation != 1:
            img = Image.fromarray(apply_exif_orientation(np.asarray(img), orientation))

        return img


def get_thumbnail_worker(
    file_path: str,
    file_hash: str,
    asset_store: Any = None,
    half: int = 0,
    split_x: float = 0.5,
    green_path: str = "",
    blue_path: str = "",
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
        img = decode_source_image(file_path, green_path, blue_path)
        if img is None:
            return None

        if half:
            from negpy.services.assets.half_frame import slice_half

            img = Image.fromarray(slice_half(np.asarray(img), half, split_x))

        square_img: Image.Image = prepare_thumbnail(img, ts)

        if asset_store:
            asset_store.save_thumbnail(cache_key, square_img)

        return square_img
    except Exception as e:
        logger.error(f"Thumbnail Error for {file_path}: {e}")
        return None


def get_rendered_thumbnail(
    buffer: Any,
    file_hash: str,
    asset_store: Any = None,
    color_space: str = WORKING_COLOR_SPACE,
    monitor_icc_bytes: Optional[bytes] = None,
) -> Optional[Image.Image]:
    """
    Creates a thumbnail from a rendered float32 buffer, applying the same display
    transform the canvas applied to that buffer (mirrors ImageConverter.to_qimage).

    ``color_space``/``monitor_icc_bytes`` must come from
    ``AppController.display_transform_params`` — they encode whether the buffer is
    still in the working space or has already been baked into display space by a
    soft proof. Assuming working space unconditionally re-applies the working->sRGB
    conversion to an already-converted buffer, which clips channels and leaves the
    filmstrip visibly more saturated than the canvas.
    """
    try:
        from negpy.infrastructure.display.color_mgmt import apply_display_transform
        from negpy.kernel.image.logic import float_to_uint8

        ts = APP_CONFIG.thumbnail_size
        if isinstance(buffer, np.ndarray) and buffer.ndim == 3 and buffer.shape[2] == 4:
            buffer = buffer[:, :, :3]
        if isinstance(buffer, np.ndarray) and buffer.dtype == np.float32:
            buffer = apply_display_transform(buffer, color_space, monitor_icc_bytes)
        u8_arr = float_to_uint8(buffer)
        img = Image.fromarray(u8_arr)

        square_img: Image.Image = prepare_thumbnail(img, ts)

        if asset_store:
            asset_store.save_thumbnail(file_hash, square_img)

        return square_img
    except Exception as e:
        logger.error(f"Rendered Thumbnail Error: {e}")
        return None
