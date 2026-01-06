from typing import List
import numpy as np

# Core
from src.core.types import ImageBuffer
from src.core.interfaces import IProcessor, PipelineContext

# Features
from src.features.geometry.models import GeometryConfig
from src.features.geometry.processor import GeometryProcessor, CropProcessor
from src.features.exposure.models import ExposureConfig
from src.features.exposure.processor import NormalizationProcessor, PhotometricProcessor
from src.features.toning.models import ToningConfig
from src.features.toning.processor import ToningProcessor
from src.features.lab.models import LabConfig
from src.features.lab.processor import PhotoLabProcessor
from src.features.retouch.models import RetouchConfig, LocalAdjustmentConfig
from src.features.retouch.processor import RetouchProcessor

# Legacy Support
from src.domain_objects import ImageSettings
from src.config import APP_CONFIG


class DarkroomEngine:
    """
    The new orchestrator that assembles the modular pipeline.
    """

    def __init__(self) -> None:
        self.config = APP_CONFIG

    def _build_pipeline(self, settings: ImageSettings) -> List[IProcessor]:
        # 1. Geometry Config
        geo_conf = GeometryConfig(
            rotation=settings.rotation,
            fine_rotation=settings.fine_rotation,
            autocrop=settings.autocrop,
            autocrop_offset=settings.autocrop_offset,
            autocrop_ratio=settings.autocrop_ratio,
        )

        # 2. Exposure Config
        exp_conf = ExposureConfig(
            density=settings.density,
            grade=settings.grade,
            wb_cyan=settings.wb_cyan,
            wb_magenta=settings.wb_magenta,
            wb_yellow=settings.wb_yellow,
            toe=settings.toe,
            toe_width=settings.toe_width,
            toe_hardness=settings.toe_hardness,
            shoulder=settings.shoulder,
            shoulder_width=settings.shoulder_width,
            shoulder_hardness=settings.shoulder_hardness,
            process_mode=settings.process_mode,
        )

        # 3. Lab Config
        lab_conf = LabConfig(
            color_separation=settings.color_separation,
            hypertone_strength=settings.hypertone_strength,
            c_noise_strength=settings.c_noise_strength,
            sharpen=settings.sharpen,
            crosstalk_matrix=settings.crosstalk_matrix,
            exposure=settings.exposure,
        )

        # 4. Toning Config
        tone_conf = ToningConfig(
            paper_profile=settings.paper_profile,
            selenium_strength=settings.selenium_strength,
            sepia_strength=settings.sepia_strength,
            process_mode=settings.process_mode,
        )

        # 5. Retouch Config
        # Convert legacy LocalAdjustment to LocalAdjustmentConfig
        local_adjs = [
            LocalAdjustmentConfig(
                points=adj.points,
                strength=adj.strength,
                radius=adj.radius,
                feather=adj.feather,
                luma_range=adj.luma_range,
                luma_softness=adj.luma_softness,
            )
            for adj in settings.local_adjustments
        ]

        retouch_conf = RetouchConfig(
            dust_remove=settings.dust_remove,
            dust_threshold=settings.dust_threshold,
            dust_size=settings.dust_size,
            manual_dust_spots=settings.manual_dust_spots,
            manual_dust_size=settings.manual_dust_size,
            local_adjustments=local_adjs,
        )

        # Assemble Pipeline
        return [
            GeometryProcessor(geo_conf),
            NormalizationProcessor(),
            PhotometricProcessor(exp_conf),
            RetouchProcessor(retouch_conf),
            PhotoLabProcessor(lab_conf),
            ToningProcessor(tone_conf),
            CropProcessor(),
        ]

    def process(self, img: ImageBuffer, settings: ImageSettings) -> ImageBuffer:
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
        )

        pipeline = self._build_pipeline(settings)

        # Run each component in sequence
        for processor in pipeline:
            img = processor.process(img, context)

        return img
