from typing import Optional, Any, Callable, Tuple
from src.core.types import ImageBuffer
from src.core.interfaces import PipelineContext
from src.core.models import WorkspaceConfig
from src.core.caching import PipelineCache, calculate_config_hash, CacheEntry
from src.core.validation import ensure_image
from src.logging_config import get_logger
from src.features.geometry.processor import GeometryProcessor, CropProcessor
from src.features.exposure.processor import NormalizationProcessor, PhotometricProcessor
from src.features.toning.processor import ToningProcessor
from src.features.lab.processor import PhotoLabProcessor
from src.features.retouch.processor import RetouchProcessor
from src.config import APP_CONFIG

logger = get_logger(__name__)


class DarkroomEngine:
    """
    The orchestrator that assembles the modular pipeline with smart caching.
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
        """
        Generic helper to execute a pipeline stage with caching logic.

        Args:
            img: Input image buffer.
            config: Configuration object for this stage (used for hashing).
            cache_field: Name of the attribute in PipelineCache to store result.
            processor_fn: Lambda/Function to execute the actual processing logic.
            context: Pipeline context.
            pipeline_changed: Boolean flag indicating if previous stages changed.

        Returns:
            Tuple[ImageBuffer, bool]: (Resulting Image, is_changed flag)
        """
        conf_hash = calculate_config_hash(config)

        cached_entry = getattr(self.cache, cache_field)
        if (
            not pipeline_changed
            and cached_entry
            and cached_entry.config_hash == conf_hash
        ):
            context.metrics.update(cached_entry.metrics)
            return cached_entry.data, False  # Not changed

        new_img = processor_fn(img, context)

        new_entry = CacheEntry(conf_hash, new_img, context.metrics.copy())
        setattr(self.cache, cache_field, new_entry)

        return new_img, True

    def process(
        self,
        img: ImageBuffer,
        settings: WorkspaceConfig,
        source_hash: str,
        context: Optional[PipelineContext] = None,
    ) -> ImageBuffer:
        """
        Executes the processing pipeline with stage-level caching.
        """
        img = ensure_image(img)

        h_orig, w_cols = img.shape[:2]

        if context is None:
            context = PipelineContext(
                scale_factor=max(h_orig, w_cols)
                / float(self.config.preview_render_size),
                original_size=(h_orig, w_cols),
                process_mode=settings.process_mode,
            )

        # Invalidate cache if source file changed
        if self.cache.source_hash != source_hash:
            self.cache.clear()
            self.cache.source_hash = source_hash

        current_img = img
        pipeline_changed = False

        def run_base(img_in: ImageBuffer, ctx: PipelineContext) -> ImageBuffer:
            img_in = GeometryProcessor(settings.geometry).process(img_in, ctx)
            return NormalizationProcessor().process(img_in, ctx)

        current_img, pipeline_changed = self._run_stage(
            current_img,
            settings.geometry,
            "base",
            run_base,
            context,
            pipeline_changed,
        )

        def run_exposure(img_in: ImageBuffer, ctx: PipelineContext) -> ImageBuffer:
            img_in = PhotometricProcessor(settings.exposure).process(img_in, ctx)
            ctx.metrics["base_positive"] = img_in.copy()
            return img_in

        current_img, pipeline_changed = self._run_stage(
            current_img,
            settings.exposure,
            "exposure",
            run_exposure,
            context,
            pipeline_changed,
        )

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
            current_img,
            settings.lab,
            "lab",
            run_lab,
            context,
            pipeline_changed,
        )

        current_img = ToningProcessor(settings.toning).process(current_img, context)
        current_img = CropProcessor().process(current_img, context)

        return current_img
