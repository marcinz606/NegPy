import numpy as np
from typing import List
from src.config import APP_CONFIG
from src.domain_objects import ImageSettings, PipelineContext
from src.helpers import ensure_rgb
from src.backend.pipeline import (
    Processor,
    NormalizationProcessor,
    PhotometricProcessor,
    ToningProcessor,
    LocalRetouchProcessor,
    PhotoLabProcessor,
    GeometryProcessor,
    OutputCropProcessor,
)


class DarkroomEngine:
    """
    The core photometric engine.
    Orchestrates a sequence of modular processing components.
    """

    def __init__(self) -> None:
        self.config = APP_CONFIG
        # Define the default processing chain
        self.pipeline: List[Processor] = [
            GeometryProcessor(),
            NormalizationProcessor(),
            PhotometricProcessor(),
            LocalRetouchProcessor(),
            PhotoLabProcessor(),
            ToningProcessor(),
            OutputCropProcessor(),
        ]

    def process(self, img: np.ndarray, settings: ImageSettings) -> np.ndarray:
        """
        Executes the processing pipeline.
        """
        img = ensure_rgb(img)
        h_orig, w_cols = img.shape[:2]

        # Shared execution context for the pipeline
        context = PipelineContext(
            scale_factor=max(h_orig, w_cols) / float(self.config.preview_max_res),
            original_size=(h_orig, w_cols),
        )

        # Run each component in sequence
        for processor in self.pipeline:
            img = processor.process(img, settings, context)

        return img
