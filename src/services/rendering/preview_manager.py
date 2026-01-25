import numpy as np
import cv2
from typing import Tuple
from src.kernel.system.config import APP_CONFIG
from src.kernel.image.logic import ensure_rgb, uint16_to_float32
from src.infrastructure.loaders.factory import loader_factory
from src.infrastructure.loaders.helpers import get_best_demosaic_algorithm
from src.domain.types import ImageBuffer, Dimensions
from src.kernel.image.validation import ensure_image
from src.infrastructure.display.color_spaces import ColorSpaceRegistry


class PreviewManager:
    """
    Loads RAW files for UI preview.
    """

    @staticmethod
    def load_linear_preview(
        file_path: str,
        color_space: str | None = None,
        use_camera_wb: bool = False,
    ) -> Tuple[ImageBuffer, Dimensions, dict]:
        """
        Loads linear RGB, downsamples for display.
        If color_space is None, uses the source's declared space (metadata).
        """
        ctx_mgr, metadata = loader_factory.get_loader(file_path)

        # Resolve target color space
        if color_space is None:
            color_space = metadata.get("color_space", "Adobe RGB")

        raw_color_space = ColorSpaceRegistry.get_rawpy_space(color_space)

        with ctx_mgr as raw:
            algo = get_best_demosaic_algorithm(raw)
            user_wb = None if use_camera_wb else [1, 1, 1, 1]

            rgb = raw.postprocess(
                gamma=(1, 1),
                no_auto_bright=True,
                use_camera_wb=use_camera_wb,
                user_wb=user_wb,
                output_bps=16,
                output_color=raw_color_space,
                demosaic_algorithm=algo,
                user_flip=0,
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

            return ensure_image(preview_raw), (h_orig, w_orig), metadata
