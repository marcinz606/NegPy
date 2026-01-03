import numpy as np
from abc import ABC, abstractmethod
from src.domain_objects import ImageSettings, PipelineContext
from src.helpers import cmy_to_density
from src.config import PIPELINE_CONSTANTS
from src.backend.image_logic.exposure import (
    measure_log_negative_bounds,
    apply_film_characteristic_curve,
    apply_chromaticity_preserving_black_point,
)
from src.backend.image_logic.color import (
    apply_paper_warmth,
    apply_shadow_highlight_grading,
    convert_to_monochrome,
    apply_color_separation,
    apply_shadow_desaturation,
)
from src.backend.image_logic.retouch import (
    apply_autocrop,
    apply_dust_removal,
    apply_chroma_noise_removal,
    apply_fine_rotation,
)
from src.backend.image_logic.local import apply_local_adjustments


class Processor(ABC):
    """
    Base class for a single stage in the Darkroom pipeline.
    """

    @abstractmethod
    def process(
        self, img: np.ndarray, settings: ImageSettings, context: PipelineContext
    ) -> np.ndarray: ...


class NormalizationProcessor(Processor):
    """
    Simulates scanner gain by performing a per-channel log-normalization.
    Stores the measured bounds in the context for downstream processors.
    """

    def process(
        self, img: np.ndarray, settings: ImageSettings, context: PipelineContext
    ) -> np.ndarray:
        epsilon = 1e-6
        img_log = np.log10(np.clip(img, epsilon, 1.0))

        bounds = measure_log_negative_bounds(img)
        context.bounds = bounds

        for ch in range(3):
            f, c = bounds.floors[ch], bounds.ceils[ch]
            img_log[:, :, ch] = np.clip(
                (img_log[:, :, ch] - f) / (max(c - f, epsilon)), 0, 1
            )
        res: np.ndarray = img_log
        return res


class PhotometricProcessor(Processor):
    """
    The heart of the engine: Inverts the negative using the H&D Characteristic Curve.
    """

    def process(
        self, img: np.ndarray, settings: ImageSettings, context: PipelineContext
    ) -> np.ndarray:
        bounds = context.bounds
        if not bounds:
            return img

        master_ref = 1.0
        exposure_shift = 0.1 + (
            settings.density * PIPELINE_CONSTANTS["density_multiplier"]
        )
        slope = 1.0 + (settings.grade * PIPELINE_CONSTANTS["grade_multiplier"])

        # Calculate per-channel pivots
        pivots = []
        cmy_vals = [settings.wb_cyan, settings.wb_magenta, settings.wb_yellow]
        for ch in range(3):
            rng = max(bounds.ceils[ch] - bounds.floors[ch], 1e-6)
            shift = cmy_to_density(cmy_vals[ch], rng)
            pivots.append(master_ref - exposure_shift - shift)

        img_pos = apply_film_characteristic_curve(
            img,  # This is the img_log from NormalizationProcessor
            (pivots[0], slope),
            (pivots[1], slope),
            (pivots[2], slope),
            toe=settings.toe,
            toe_width=settings.toe_width,
            toe_hardness=settings.toe_hardness,
            shoulder=settings.shoulder,
            shoulder_width=settings.shoulder_width,
            shoulder_hardness=settings.shoulder_hardness,
            pre_logged=True,
        )

        res = np.clip(img_pos, 0, 1)
        if settings.process_mode == "B&W":
            res = convert_to_monochrome(res)
        return res


class ToningProcessor(Processor):
    """
    Applies paper stock warmth and chemical toning effects.
    """

    def process(
        self, img: np.ndarray, settings: ImageSettings, context: PipelineContext
    ) -> np.ndarray:
        img = apply_paper_warmth(img, settings.temperature)
        img = apply_shadow_highlight_grading(img, settings)

        if settings.process_mode == "B&W":
            img = apply_chromaticity_preserving_black_point(img, 0.05)
        return img


class LocalRetouchProcessor(Processor):
    """
    Handles cleanup (dust/noise) and targeted dodge/burn adjustments.
    """

    def process(
        self, img: np.ndarray, settings: ImageSettings, context: PipelineContext
    ) -> np.ndarray:
        scale_factor = context.scale_factor

        # 1. Automated Cleanup
        img = apply_dust_removal(img, settings, scale_factor)
        img = apply_chroma_noise_removal(img, settings, scale_factor)

        # 2. Manual Dodge & Burn
        img = apply_local_adjustments(img, settings.local_adjustments, scale_factor)
        return img


class ColorFinishingProcessor(Processor):
    """
    Final look adjustments like saturation and shadow desaturation.
    """

    def process(
        self, img: np.ndarray, settings: ImageSettings, context: PipelineContext
    ) -> np.ndarray:
        if settings.process_mode != "B&W":
            img = apply_color_separation(img, settings.color_separation)
            img *= settings.saturation

        img = apply_shadow_desaturation(img, settings.shadow_desat_strength)
        img *= 2.0**settings.exposure  # Linear exposure trim
        return np.clip(img, 0, 1)


class GeometryProcessor(Processor):
    """
    Handles rotations and cropping as the final step.
    """

    def process(
        self, img: np.ndarray, settings: ImageSettings, context: PipelineContext
    ) -> np.ndarray:
        scale_factor = context.scale_factor

        if settings.rotation != 0:
            img = np.rot90(img, k=settings.rotation)

        if settings.fine_rotation != 0.0:
            img = apply_fine_rotation(img, settings.fine_rotation)

        if settings.autocrop:
            img = apply_autocrop(
                img,
                offset_px=settings.autocrop_offset,
                scale_factor=scale_factor,
                ratio=settings.autocrop_ratio,
            )
        return img
