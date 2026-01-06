from typing import Optional
from PIL import Image
import rawpy
from src.config import APP_CONFIG
from src.helpers import ensure_rgb
from src.infrastructure.loaders.factory import loader_factory


def get_thumbnail_worker(file_path: str) -> Optional[Image.Image]:
    """
    Worker function for parallel thumbnail generation from RAW file path.
    """
    try:
        ts = APP_CONFIG.thumbnail_size
        with loader_factory.get_loader(file_path) as raw:
            # Use fastest possible demosaicing for thumbnails
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
            return square_img
    except Exception as e:
        print(f"Thumbnail Error for {file_path}: {e}")
        return None
