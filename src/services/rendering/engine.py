from typing import Optional, Any, Callable, Tuple
from src.domain.types import ImageBuffer
from src.domain.interfaces import PipelineContext
from src.domain.models import WorkspaceConfig
from src.kernel.caching.manager import PipelineCache
from src.kernel.caching.logic import calculate_config_hash, CacheEntry
from src.kernel.image.validation import ensure_image
from src.kernel.system.logging import get_logger
from src.features.geometry.processor import GeometryProcessor, CropProcessor
from src.features.exposure.processor import NormalizationProcessor, PhotometricProcessor
from src.features.toning.processor import ToningProcessor
from src.features.lab.processor import PhotoLabProcessor
from src.features.retouch.processor import RetouchProcessor
from src.kernel.system.config import APP_CONFIG
from src.services.view.coordinate_mapping import CoordinateMapping

logger = get_logger(__name__)


class DarkroomEngine:
    """
    Runs the pipeline. Handles stage caching.
    """

    def __init__(self) -> None:
        self.config = APP_CONFIG
        self.cache = PipelineCache()

    def _run_stage(
        self,
        img: ImageBuffer,
        config: Any,
        cache_field: str,
        processor_fn: Callable[[ImageBuffer, PipelineContext], ImageBuffer],
        context: PipelineContext,
        pipeline_changed: bool,
    ) -> Tuple[ImageBuffer, bool]:
        conf_hash = calculate_config_hash(config)
        cached_entry = getattr(self.cache, cache_field)

        if (
            not pipeline_changed
            and cached_entry
            and cached_entry.config_hash == conf_hash
        ):
            context.metrics.update(cached_entry.metrics)
            context.active_roi = cached_entry.active_roi
            return cached_entry.data, False

        new_img = processor_fn(img, context)
        new_entry = CacheEntry(
            conf_hash, new_img, context.metrics.copy(), context.active_roi
        )
        setattr(self.cache, cache_field, new_entry)

        return new_img, True

    def process(
        self,
        img: ImageBuffer,
        settings: WorkspaceConfig,
        source_hash: str,
        context: Optional[PipelineContext] = None,
    ) -> ImageBuffer:
        img = ensure_image(img)
        h_orig, w_cols = img.shape[:2]

        if context is None:
            context = PipelineContext(
                scale_factor=max(h_orig, w_cols)
                / float(self.config.preview_render_size),
                original_size=(h_orig, w_cols),
                process_mode=settings.process_mode,
            )

        pipeline_changed = False
        if self.cache.source_hash != source_hash:
            self.cache.clear()
            self.cache.source_hash = source_hash
            pipeline_changed = True

        if self.cache.process_mode != settings.process_mode:
            self.cache.process_mode = settings.process_mode
            self.cache.exposure = None
            self.cache.retouch = None
            self.cache.lab = None
            pipeline_changed = True

        current_img = img

        if settings.geometry.manual_crop_rect:
            logger.debug(
                f"Engine process with manual_crop_rect: {settings.geometry.manual_crop_rect}"
            )

        def run_base(img_in: ImageBuffer, ctx: PipelineContext) -> ImageBuffer:
            img_in = GeometryProcessor(settings.geometry).process(img_in, ctx)
            return NormalizationProcessor(settings.exposure).process(img_in, ctx)

        base_key = (settings.geometry, settings.exposure.analysis_buffer)
        current_img, pipeline_changed = self._run_stage(
            current_img, base_key, "base", run_base, context, pipeline_changed
        )

        def run_exposure(img_in: ImageBuffer, ctx: PipelineContext) -> ImageBuffer:
            img_out = PhotometricProcessor(settings.exposure).process(img_in, ctx)
            return img_out

        current_img, pipeline_changed = self._run_stage(
            current_img,
            settings.exposure,
            "exposure",
            run_exposure,
            context,
            pipeline_changed,
        )

        context.metrics["retouch_source"] = current_img.copy()

        def run_retouch(img_in: ImageBuffer, ctx: PipelineContext) -> ImageBuffer:
            return RetouchProcessor(settings.retouch).process(img_in, ctx)

        current_img, pipeline_changed = self._run_stage(
            current_img,
            settings.retouch,
            "retouch",
            run_retouch,
            context,
            pipeline_changed,
        )

        def run_lab(img_in: ImageBuffer, ctx: PipelineContext) -> ImageBuffer:
            return PhotoLabProcessor(settings.lab).process(img_in, ctx)

        current_img, pipeline_changed = self._run_stage(
            current_img, settings.lab, "lab", run_lab, context, pipeline_changed
        )

        current_img = ToningProcessor(settings.toning).process(current_img, context)
        current_img = CropProcessor(settings.geometry).process(current_img, context)

        try:
            uv_grid = CoordinateMapping.create_uv_grid(
                rh_orig=h_orig,
                rw_orig=w_cols,
                rotation=settings.geometry.rotation,
                fine_rot=settings.geometry.fine_rotation,
                flip_h=settings.geometry.flip_horizontal,
                flip_v=settings.geometry.flip_vertical,
                autocrop=True,
                autocrop_params={"roi": context.active_roi}
                if context.active_roi
                else None,
            )
            context.metrics["uv_grid"] = uv_grid
        except Exception as e:
            logger.error(f"Failed to generate UV grid: {e}")

        context.metrics["base_positive"] = current_img.copy()

        return current_img
