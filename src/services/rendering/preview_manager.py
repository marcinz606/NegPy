import numpy as np
import rawpy
import cv2
from typing import Tuple
from src.kernel.system.config import APP_CONFIG
from src.kernel.image.logic import ensure_rgb, uint16_to_float32
from src.infrastructure.loaders.factory import loader_factory
from src.domain.types import ImageBuffer, Dimensions
from src.kernel.image.validation import ensure_image


class PreviewManager:
    """
    Loads RAW files for UI preview.
    """

    @staticmethod
    def load_linear_preview(
        file_path: str, color_space: str
    ) -> Tuple[ImageBuffer, Dimensions]:
        """
        Loads linear RGB, downsamples for display.
        """
        raw_color_space = rawpy.ColorSpace.sRGB
        if color_space == "Adobe RGB":
            raw_color_space = rawpy.ColorSpace.Adobe

        with loader_factory.get_loader(file_path) as raw:
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

            full_linear = uint16_to_float32(np.ascontiguousarray(rgb))
            h_orig, w_orig = full_linear.shape[:2]

            max_res = APP_CONFIG.preview_render_size
            if max(h_orig, w_orig) > max_res:
                scale = max_res / max(h_orig, w_orig)
                target_w = int(w_orig * scale)
                target_h = int(h_orig * scale)

                preview_raw = ensure_image(
                    cv2.resize(
                        full_linear,
                        (target_w, target_h),
                        interpolation=cv2.INTER_AREA,
                    )
                )
            else:
                preview_raw = full_linear.copy()

            return ensure_image(preview_raw), (h_orig, w_orig)
