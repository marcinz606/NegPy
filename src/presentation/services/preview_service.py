import numpy as np
import rawpy
import cv2
from typing import Tuple
from src.config import APP_CONFIG
from src.helpers import ensure_rgb, ensure_array
from src.infrastructure.loaders.factory import loader_factory
from src.core.types import ImageBuffer, Dimensions
from src.core.validation import ensure_image


class PreviewService:
    """
    Service responsible for loading and preparing linear RAW data for the UI.
    """

    @staticmethod
    def load_linear_preview(
        file_path: str, color_space: str
    ) -> Tuple[ImageBuffer, Dimensions]:
        """
        Reads a RAW file, demosaics to linear space, and returns a downsampled ImageBuffer.
        """
        raw_color_space = rawpy.ColorSpace.sRGB
        if color_space == "Adobe RGB":
            raw_color_space = rawpy.ColorSpace.Adobe

        with loader_factory.get_loader(file_path) as raw:
            # PURE RAW: Essential for darkroom-style analytical WB.

            rgb = raw.postprocess(
                gamma=(1, 1),
                no_auto_bright=True,
                use_camera_wb=False,
                user_wb=[1, 1, 1, 1],
                output_bps=16,
                output_color=raw_color_space,
                demosaic_algorithm=rawpy.DemosaicAlgorithm.LINEAR,
            )
            rgb = ensure_rgb(rgb)

            full_linear = rgb.astype(np.float32) / 65535.0
            h_orig, w_orig = full_linear.shape[:2]

            max_res = APP_CONFIG.preview_render_size
            if max(h_orig, w_orig) > max_res:
                scale = max_res / max(h_orig, w_orig)
                target_w = int(w_orig * scale)
                target_h = int(h_orig * scale)

                # Use INTER_AREA for downsampling to minimize aliasing
                preview_raw = ensure_array(
                    cv2.resize(
                        full_linear,
                        (target_w, target_h),
                        interpolation=cv2.INTER_AREA,
                    )
                )
            else:
                preview_raw = full_linear.copy()

            return ensure_image(preview_raw), (h_orig, w_orig)
