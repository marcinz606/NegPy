import asyncio
from typing import Optional, Any, List, Dict, Tuple
from PIL import Image
import rawpy
from src.kernel.system.config import APP_CONFIG
from src.kernel.image.logic import ensure_rgb
from src.infrastructure.loaders.factory import loader_factory
from src.kernel.system.logging import get_logger

logger = get_logger(__name__)


async def generate_batch_thumbnails(
    files: List[Dict[str, str]], asset_store: Any
) -> Dict[str, Image.Image]:
    """
    Parallel thumbnail generation (throttled).
    """

    # Limit concurrency to half of available cores
    limit = max(1, APP_CONFIG.max_workers // 2)
    semaphore = asyncio.Semaphore(limit)

    async def _worker(f_info: Dict[str, str]) -> Tuple[str, Optional[Image.Image]]:
        async with semaphore:
            thumb = await asyncio.to_thread(
                get_thumbnail_worker, f_info["path"], f_info["hash"], asset_store
            )
            return f_info["name"], thumb

    tasks = [_worker(f) for f in files]
    results = await asyncio.gather(*tasks)

    return {name: thumb for name, thumb in results if isinstance(thumb, Image.Image)}


def get_thumbnail_worker(
    file_path: str, file_hash: str, asset_store: Any = None
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
        with loader_factory.get_loader(file_path) as raw:
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

            img.thumbnail((ts, ts))
            square_img = Image.new("RGB", (ts, ts), (14, 17, 23))
            square_img.paste(img, ((ts - img.width) // 2, (ts - img.height) // 2))

            if asset_store:
                asset_store.save_thumbnail(file_hash, square_img)

            return square_img
    except Exception as e:
        logger.error(f"Thumbnail Error for {file_path}: {e}")
        return None
