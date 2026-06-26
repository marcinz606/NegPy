import io
import time
from typing import Any, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

import rawpy

from negpy.domain.types import Dimensions, ImageBuffer
from negpy.infrastructure.loaders.factory import loader_factory
from negpy.infrastructure.loaders.helpers import NonStandardFileWrapper, get_best_demosaic_algorithm
from negpy.kernel.image.logic import apply_exif_orientation, ensure_rgb, uint16_to_float32
from negpy.kernel.image.validation import ensure_image
from negpy.kernel.system.config import APP_CONFIG
from negpy.features.rgbscan.logic import assemble_rgb
from negpy.kernel.system.logging import get_logger
from negpy.services.rendering.preview_cache import PreviewBufferCache, PreviewCacheKey

logger = get_logger(__name__)


def _output_dimensions_from_raw(raw: Any, postprocessed_h: int, postprocessed_w: int) -> Tuple[int, int]:
    """
    Returns (height, width) of the full-resolution image in image space, not the half_size postprocess output.
    """
    try:
        s = raw.sizes
        for pair in (("iheight", "iwidth"), ("raw_height", "raw_width"), ("height", "width")):
            h_attr, w_attr = pair
            if hasattr(s, h_attr) and hasattr(s, w_attr):
                h = int(getattr(s, h_attr))
                w = int(getattr(s, w_attr))
                if h > 0 and w > 0:
                    return (h, w)
    except Exception:
        pass
    return (postprocessed_h, postprocessed_w)


# Pre-warm the Numba JIT so the first actual preview load doesn't pay the compile cost.
_warmup = np.zeros((2, 2, 3), dtype=np.uint16)
uint16_to_float32(_warmup)
del _warmup


class PreviewManager:
    """
    Loads RAW (and other) files for UI preview, with in-memory LRU and fast decode.
    """

    def __init__(self) -> None:
        self._cache = PreviewBufferCache(APP_CONFIG)

    # ------------------------------------------------------------------
    # Internal helpers — operate on an already-open raw object so that
    # callers that need both splash and linear can share a single file open.
    # ------------------------------------------------------------------

    @staticmethod
    def _try_splash_from_open_raw(raw: Any, file_path: str) -> Optional[Tuple[ImageBuffer, Dimensions]]:
        """
        Extract a splash preview from an already-open raw object.
        Returns None if a thumb cannot be extracted or converted.
        """
        t0 = time.perf_counter()
        if not hasattr(raw, "extract_thumb"):
            return None
        try:
            thumb = raw.extract_thumb()
            img: Optional[Image.Image] = None
            if thumb.format == rawpy.ThumbFormat.JPEG:
                img = Image.open(io.BytesIO(thumb.data))
            elif thumb.format == rawpy.ThumbFormat.BITMAP:
                img = Image.fromarray(ensure_rgb(thumb.data))
            if img is None:
                return None
            img = img.convert("RGB")
        except Exception:
            return None
        arr = np.ascontiguousarray(np.array(img, dtype=np.float32) / 255.0)
        h, w = arr.shape[:2]
        if max(h, w) > APP_CONFIG.preview_render_size:
            scale = APP_CONFIG.preview_render_size / max(h, w)
            tw, th = int(w * scale), int(h * scale)
            arr = ensure_image(cv2.resize(arr, (tw, th), interpolation=cv2.INTER_AREA).astype(np.float32))
        dh, dw = arr.shape[:2]
        full_dims = _output_dimensions_from_raw(raw, dh, dw)
        logger.debug("preview _try_splash_from_open_raw ok %.3fs for %s", time.perf_counter() - t0, file_path)
        return ensure_image(arr), full_dims

    def _load_from_open_raw(
        self,
        raw: Any,
        metadata: dict,
        file_path: str,
        color_space: str,
        use_camera_wb: bool,
        full_resolution: bool,
        file_hash: str | None,
    ) -> Tuple[ImageBuffer, Dimensions, dict]:
        """
        Decode and resize a linear preview from an already-open raw object.
        Handles cache write on completion.
        """
        t_decode = time.perf_counter()

        use_fast = (not full_resolution) and (not isinstance(raw, NonStandardFileWrapper))
        if use_fast:
            demosaic = rawpy.DemosaicAlgorithm.LINEAR
            post_kw: dict = {"half_size": True}
        else:
            demosaic = get_best_demosaic_algorithm(raw)
            post_kw = {}

        # Read full-resolution dims before postprocess — rawpy/libraw mutates
        # raw.sizes.iheight/iwidth when half_size=True, so reading after gives wrong dims.
        full_dims_pre = _output_dimensions_from_raw(raw, 0, 0)

        user_wb = None if use_camera_wb else [1, 1, 1, 1]

        t_pp = time.perf_counter()
        rgb = raw.postprocess(
            gamma=(1, 1),
            no_auto_bright=True,
            use_camera_wb=use_camera_wb,
            user_wb=user_wb,
            output_bps=16,
            output_color=rawpy.ColorSpace.raw,
            demosaic_algorithm=demosaic,
            user_flip=0,
            **post_kw,
        )
        logger.debug("raw.postprocess %.3fs (fast=%s)", time.perf_counter() - t_pp, use_fast)
        rgb = ensure_rgb(rgb)

        # Bake EXIF orientation into the buffer (postprocess runs with user_flip=0).
        orientation = metadata.get("orientation", 1)
        full_linear = apply_exif_orientation(uint16_to_float32(np.ascontiguousarray(rgb)), orientation)
        ir_full = metadata.get("ir")
        if ir_full is not None:
            ir_full = apply_exif_orientation(ir_full, orientation)

        h_p, w_p = full_linear.shape[:2]
        # Use pre-postprocess dims if valid; fall back to buffer shape (e.g. NonStandardFileWrapper).
        if full_dims_pre[0] > 0:
            h_orig, w_orig = full_dims_pre
            # Sensor dims are pre-orientation; swap to match the oriented buffer for 90° rotations.
            if orientation in (5, 6, 7, 8):
                h_orig, w_orig = w_orig, h_orig
        else:
            h_orig, w_orig = _output_dimensions_from_raw(raw, h_p, w_p)
        t_resize0 = time.perf_counter()
        max_res = APP_CONFIG.preview_render_size
        if max(h_p, w_p) > max_res and not full_resolution:
            scale = max_res / max(h_p, w_p)
            target_w = int(w_p * scale)
            target_h = int(h_p * scale)
            preview_raw = ensure_image(
                cv2.resize(
                    full_linear,
                    (target_w, target_h),
                    interpolation=cv2.INTER_AREA,
                )
            )
        else:
            preview_raw = full_linear.copy()
        logger.debug("preview resize+convert %.3fs", time.perf_counter() - t_resize0)

        # IR channel travels with the preview; resize it to match the final preview dims.
        if ir_full is not None and ir_full.shape[:2] == full_linear.shape[:2]:
            ph, pw = preview_raw.shape[:2]
            if (ph, pw) != ir_full.shape[:2]:
                metadata["ir_preview"] = cv2.resize(
                    ir_full.astype(np.float32),
                    (pw, ph),
                    interpolation=cv2.INTER_AREA,
                ).astype(np.float32)
            else:
                metadata["ir_preview"] = ir_full.astype(np.float32).copy()
        else:
            metadata["ir_preview"] = None

        out = ensure_image(preview_raw)
        logger.debug(
            "PreviewManager._load_from_open_raw decode+resize %.3fs",
            time.perf_counter() - t_decode,
        )
        if file_hash:
            ck = PreviewCacheKey(
                file_hash=file_hash,
                use_camera_wb=use_camera_wb,
                workspace_color_space=color_space,
                full_resolution=full_resolution,
            )
            self._cache.put(ck, out.copy(), (h_orig, w_orig), dict(metadata))
        return out, (h_orig, w_orig), metadata

    # ------------------------------------------------------------------
    # Public API — thin wrappers; kept for all existing callers.
    # ------------------------------------------------------------------

    @staticmethod
    def try_splash_preview(file_path: str) -> Optional[Tuple[ImageBuffer, Dimensions]]:
        """
        Quick embedded-JPEG (or half-size) RGB for first paint. Returns None if not available.
        """
        try:
            ctx_mgr, _metadata = loader_factory.get_loader(file_path)
        except Exception:
            return None
        try:
            with ctx_mgr as raw:
                return PreviewManager._try_splash_from_open_raw(raw, file_path)
        except Exception as e:
            logger.debug("preview splash skip: %s", e)
        return None

    def load_linear_preview(
        self,
        file_path: str,
        color_space: str | None = None,
        use_camera_wb: bool = False,
        full_resolution: bool = False,
        file_hash: str | None = None,
    ) -> Tuple[ImageBuffer, Dimensions, dict]:
        """
        Loads linear RGB, downsamples for display.
        If color_space is None, uses the source's declared space (metadata).
        """
        t_all = time.perf_counter()

        # Fast path: skip file open entirely when all cache-key params are known upfront.
        if file_hash and color_space is not None:
            ck = PreviewCacheKey(
                file_hash=file_hash,
                use_camera_wb=use_camera_wb,
                workspace_color_space=color_space,
                full_resolution=full_resolution,
            )
            hit = self._cache.get(ck)
            if hit is not None:
                logger.debug("preview cache hit %.3fs for %s", time.perf_counter() - t_all, file_path)
                return hit  # cache hit — caller must not mutate this buffer

        ctx_mgr, metadata = loader_factory.get_loader(file_path)

        if color_space is None:
            color_space = metadata.get("color_space", "Adobe RGB")
            # Re-check now that color_space is resolved from metadata.
            if file_hash:
                ck = PreviewCacheKey(
                    file_hash=file_hash,
                    use_camera_wb=use_camera_wb,
                    workspace_color_space=color_space,
                    full_resolution=full_resolution,
                )
                hit = self._cache.get(ck)
                if hit is not None:
                    logger.debug("preview cache hit %.3fs for %s", time.perf_counter() - t_all, file_path)
                    return hit  # cache hit — caller must not mutate this buffer

        t_decode = time.perf_counter()
        with ctx_mgr as raw:
            out, dims, meta = self._load_from_open_raw(
                raw,
                metadata,
                file_path,
                color_space,
                use_camera_wb,
                full_resolution,
                file_hash,
            )
        logger.debug(
            "PreviewManager.load_linear_preview decode+resize %.3fs (total %.3fs)",
            time.perf_counter() - t_decode,
            time.perf_counter() - t_all,
        )
        return out, dims, meta

    def load_linear_preview_rgb(
        self,
        red_path: str,
        green_path: str,
        blue_path: str,
        color_space: str | None = None,
        use_camera_wb: bool = False,
        full_resolution: bool = False,
        file_hash: str | None = None,
        align: bool = True,
    ) -> Tuple[ImageBuffer, Dimensions, dict]:
        """Merge a narrowband R/G/B triplet into one linear preview: red channel from the
        red shot, green from green, blue from blue. Each exposure is decoded (and cached)
        through the normal preview path, then channels are combined."""
        red_out, dims, meta = self.load_linear_preview(red_path, color_space, use_camera_wb, full_resolution, file_hash)
        green_out, _, _ = self.load_linear_preview(green_path, color_space, use_camera_wb, full_resolution, None)
        blue_out, _, _ = self.load_linear_preview(blue_path, color_space, use_camera_wb, full_resolution, None)

        red = np.asarray(red_out, dtype=np.float32)

        def _match(buf: ImageBuffer) -> np.ndarray:
            arr = np.asarray(buf, dtype=np.float32)
            if arr.shape[:2] != red.shape[:2]:
                arr = cv2.resize(arr, (red.shape[1], red.shape[0]), interpolation=cv2.INTER_AREA)
            return arr

        merged = assemble_rgb(red, _match(green_out), _match(blue_out), align=align)
        return ensure_image(merged), dims, meta

    def load_splash_and_linear(
        self,
        file_path: str,
        color_space: str | None = None,
        use_camera_wb: bool = False,
        full_resolution: bool = False,
        file_hash: str | None = None,
    ) -> Tuple[Optional[Tuple[ImageBuffer, Dimensions]], Tuple[ImageBuffer, Dimensions, dict]]:
        """
        Open the RAW file once and return both the splash preview and the linear
        preview in a single call.  This avoids the double file-open cost that
        occurs when ``try_splash_preview`` and ``load_linear_preview`` are called
        back-to-back.

        Returns ``(splash_result, linear_result)`` where *splash_result* is the
        same type as ``try_splash_preview`` (may be ``None``) and *linear_result*
        is the same type as ``load_linear_preview``.
        """
        t_all = time.perf_counter()

        # Fast path: skip file open entirely when all cache-key params are known upfront.
        if file_hash and color_space is not None:
            ck = PreviewCacheKey(
                file_hash=file_hash,
                use_camera_wb=use_camera_wb,
                workspace_color_space=color_space,
                full_resolution=full_resolution,
            )
            hit = self._cache.get(ck)
            if hit is not None:
                logger.debug("preview cache hit %.3fs for %s", time.perf_counter() - t_all, file_path)
                return None, hit  # no splash on cache hit — linear is already fast

        try:
            ctx_mgr, metadata = loader_factory.get_loader(file_path)
        except Exception as e:
            logger.debug("preview load_splash_and_linear open failed: %s", e)
            raise

        if color_space is None:
            color_space = metadata.get("color_space", "Adobe RGB")
            # Re-check now that color_space is resolved from metadata.
            if file_hash:
                ck = PreviewCacheKey(
                    file_hash=file_hash,
                    use_camera_wb=use_camera_wb,
                    workspace_color_space=color_space,
                    full_resolution=full_resolution,
                )
                hit = self._cache.get(ck)
                if hit is not None:
                    logger.debug("preview cache hit %.3fs for %s", time.perf_counter() - t_all, file_path)
                    return None, hit  # no splash on cache hit — linear is already fast

        t_decode = time.perf_counter()
        splash_result: Optional[Tuple[ImageBuffer, Dimensions]] = None
        with ctx_mgr as raw:
            if not full_resolution:
                splash_result = self._try_splash_from_open_raw(raw, file_path)
            linear_result = self._load_from_open_raw(
                raw,
                metadata,
                file_path,
                color_space,
                use_camera_wb,
                full_resolution,
                file_hash,
            )
        logger.debug(
            "PreviewManager.load_splash_and_linear %.3fs (total %.3fs)",
            time.perf_counter() - t_decode,
            time.perf_counter() - t_all,
        )
        return splash_result, linear_result
