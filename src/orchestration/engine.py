from typing import List
import numpy as np

# Core
from src.core.types import ImageBuffer
from src.core.interfaces import IProcessor, PipelineContext

from src.core.session.models import WorkspaceConfig

# Features
from src.features.geometry.processor import GeometryProcessor, CropProcessor
from src.features.exposure.processor import NormalizationProcessor, PhotometricProcessor
from src.features.toning.processor import ToningProcessor
from src.features.lab.processor import PhotoLabProcessor
from src.features.retouch.processor import RetouchProcessor

# Legacy Support
from src.config import APP_CONFIG


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

    def process(self, img: ImageBuffer, settings: WorkspaceConfig) -> ImageBuffer:
        """
        Executes the processing pipeline.
        """
        # Ensure input is float32
        if img.dtype != np.float32:
            img = img.astype(np.float32)

        h_orig, w_cols = img.shape[:2]

        # Shared execution context
        context = PipelineContext(
            scale_factor=max(h_orig, w_cols) / float(self.config.preview_render_size),
            original_size=(h_orig, w_cols),
            process_mode=settings.process_mode,
        )

        pipeline = self._build_pipeline(settings)

        # Run each component in sequence
        for processor in pipeline:
            img = processor.process(img, context)

        return img
