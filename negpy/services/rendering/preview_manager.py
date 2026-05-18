from typing import Tuple

import cv2
import numpy as np
import rawpy

from negpy.domain.types import Dimensions, ImageBuffer
from negpy.infrastructure.loaders.factory import loader_factory
from negpy.infrastructure.loaders.helpers import get_best_demosaic_algorithm
from negpy.kernel.image.logic import ensure_rgb, uint16_to_float32
from negpy.kernel.image.validation import ensure_image
from negpy.kernel.system.config import APP_CONFIG


# Pre-warm the Numba JIT so the first actual preview load doesn't pay the compile cost.
_warmup = np.zeros((2, 2, 3), dtype=np.uint16)
uint16_to_float32(_warmup)
del _warmup


class PreviewManager:
    """
    Loads RAW files for UI preview.
    """

    @staticmethod
    def load_linear_preview(
        file_path: str,
        color_space: str | None = None,
        linear_raw: bool = False,
        full_resolution: bool = False,
    ) -> Tuple[ImageBuffer, Dimensions, dict]:
        """
        Loads linear RGB, downsamples for display.
        If color_space is None, uses the source's declared space (metadata).
        """
        ctx_mgr, metadata = loader_factory.get_loader(file_path)

        with ctx_mgr as raw:
            algo = get_best_demosaic_algorithm(raw, for_preview=True)
            user_wb = [1, 1, 1, 1] if linear_raw else None

            rgb = raw.postprocess(
                gamma=(1, 1),
                no_auto_bright=True,
                use_camera_wb=not linear_raw,
                user_wb=user_wb,
                output_bps=16,
                output_color=rawpy.ColorSpace.raw,
                demosaic_algorithm=algo,
                user_flip=0,
            )
            rgb = ensure_rgb(rgb)

            full_linear = uint16_to_float32(np.ascontiguousarray(rgb))
            h_orig, w_orig = full_linear.shape[:2]

            max_res = APP_CONFIG.preview_render_size
            if max(h_orig, w_orig) > max_res and not full_resolution:
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
