from typing import List, Optional
import numpy as np

# Core
from src.core.types import ImageBuffer
from src.core.interfaces import IProcessor, PipelineContext

from src.core.models import WorkspaceConfig
from src.logging_config import get_logger

# Features
from src.features.geometry.processor import GeometryProcessor, CropProcessor
from src.features.exposure.processor import NormalizationProcessor, PhotometricProcessor
from src.features.toning.processor import ToningProcessor
from src.features.lab.processor import PhotoLabProcessor
from src.features.retouch.processor import RetouchProcessor

# Legacy Support
from src.config import APP_CONFIG

logger = get_logger(__name__)


class DarkroomEngine:
    """
    The new orchestrator that assembles the modular pipeline.
    """

    def __init__(self) -> None:
        self.config = APP_CONFIG

    def _build_pipeline(self, settings: WorkspaceConfig) -> List[IProcessor]:
        # Assemble Pipeline directly from composed configs
        return [
            GeometryProcessor(settings.geometry),
            NormalizationProcessor(),
            PhotometricProcessor(settings.exposure),
            RetouchProcessor(settings.retouch),
            PhotoLabProcessor(settings.lab),
            ToningProcessor(settings.toning),
            CropProcessor(),
        ]

    def process(
        self,
        img: ImageBuffer,
        settings: WorkspaceConfig,
        context: Optional[PipelineContext] = None,
    ) -> ImageBuffer:
        """
        Executes the processing pipeline.
        """
        # Ensure input is float32
        if img.dtype != np.float32:
            img = img.astype(np.float32)

        h_orig, w_cols = img.shape[:2]

        # Shared execution context
        if context is None:
            context = PipelineContext(
                scale_factor=max(h_orig, w_cols)
                / float(self.config.preview_render_size),
                original_size=(h_orig, w_cols),
                process_mode=settings.process_mode,
            )

        pipeline = self._build_pipeline(settings)

        # Run each component in sequence
        for processor in pipeline:
            img = processor.process(img, context)

            if isinstance(processor, PhotometricProcessor):
                # Keep positive from this point in the pipeline
                # for the purpose of computing masks for dodge & burn
                context.metrics["base_positive"] = img.copy()

        return img
