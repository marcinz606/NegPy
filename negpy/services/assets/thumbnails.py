import asyncio
from typing import Optional, Any, List, Dict, Tuple
from PIL import Image
import rawpy
from negpy.kernel.system.config import APP_CONFIG
from negpy.kernel.image.logic import ensure_rgb, prepare_thumbnail
from negpy.infrastructure.loaders.factory import loader_factory
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
            thumb = await asyncio.to_thread(get_thumbnail_worker, f_info["path"], f_info["hash"], asset_store)
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


def get_thumbnail_worker(file_path: str, file_hash: str, asset_store: Any = None) -> Optional[Image.Image]:
    """
    Checks cache -> extracts/renders -> resize.
    """
    try:
        if asset_store:
            cached = asset_store.get_thumbnail(file_hash)
            if isinstance(cached, Image.Image):
                return cached

        ts = APP_CONFIG.thumbnail_size
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
                    use_camera_wb=False,
                    user_wb=[1, 1, 1, 1],
                    half_size=True,
                    no_auto_bright=True,
                    bright=1.0,
                    demosaic_algorithm=algo,
                )
                rgb = ensure_rgb(rgb)
                img = Image.fromarray(rgb)

            rot = metadata.get("orientation", 0)
            if rot != 0:
                img = img.rotate(rot * -90, expand=True)

            square_img: Image.Image = prepare_thumbnail(img, ts)

            if asset_store:
                asset_store.save_thumbnail(file_hash, square_img)

            return square_img
    except Exception as e:
        logger.error(f"Thumbnail Error for {file_path}: {e}")
        return None


def get_rendered_thumbnail(buffer: Any, file_hash: str, asset_store: Any = None) -> Optional[Image.Image]:
    """
    Creates a thumbnail from a rendered float32 buffer.
    """
    try:
        from negpy.kernel.image.logic import float_to_uint8

        ts = APP_CONFIG.thumbnail_size
        u8_arr = float_to_uint8(buffer)
        img = Image.fromarray(u8_arr)

        square_img: Image.Image = prepare_thumbnail(img, ts)

        if asset_store:
            asset_store.save_thumbnail(file_hash, square_img)

        return square_img
    except Exception as e:
        logger.error(f"Rendered Thumbnail Error: {e}")
        return None
