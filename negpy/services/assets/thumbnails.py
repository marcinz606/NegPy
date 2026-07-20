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


def decode_source_image(file_path: str) -> Optional[Image.Image]:
    """Small EXIF-oriented preview of a source file (embedded thumb, else fast decode)."""
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
            algo = rawpy.DemosaicAlgorithm.LINEAR

            rgb = raw.postprocess(
                use_camera_wb=True,
                user_wb=None,
                half_size=True,
                no_auto_bright=True,
                bright=1.0,
                demosaic_algorithm=algo,
                user_flip=0,
            )
            rgb = ensure_rgb(rgb)
            img = Image.fromarray(rgb)

        orientation = metadata.get("orientation", 1)
        if orientation and orientation != 1:
            img = Image.fromarray(apply_exif_orientation(np.asarray(img), orientation))

        return img


def get_thumbnail_worker(
    file_path: str, file_hash: str, asset_store: Any = None, half: int = 0, split_x: float = 0.5
) -> Optional[Image.Image]:
    """
    Checks cache -> extracts/renders -> resize.
    """
    try:
        if asset_store:
            cached = asset_store.get_thumbnail(file_hash)
            if isinstance(cached, Image.Image):
                return cached

        ts = APP_CONFIG.thumbnail_size
        img = decode_source_image(file_path)
        if img is None:
            return None

        if half:
            from negpy.services.assets.half_frame import slice_half

            img = Image.fromarray(slice_half(np.asarray(img), half, split_x))

        square_img: Image.Image = prepare_thumbnail(img, ts)

        if asset_store:
            asset_store.save_thumbnail(file_hash, square_img)

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
