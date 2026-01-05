import numpy as np
import copy
from abc import ABC, abstractmethod
from typing import Tuple
from src.logging_config import get_logger
from src.domain_objects import ImageSettings, PipelineContext
from src.helpers import cmy_to_density
from src.backend.utils import convert_to_monochrome
from src.config import PIPELINE_CONSTANTS
from src.backend.image_logic.exposure import (
    measure_log_negative_bounds,
    apply_film_characteristic_curve,
    apply_chromaticity_preserving_black_point,
)
from src.backend.image_logic.paper_toning import (
    simulate_paper_substrate,
    apply_chemical_toning,
)
from src.backend.image_logic.lab_scanner import (
    apply_spectral_crosstalk,
    apply_hypertone,
    apply_chroma_noise_removal,
    apply_output_sharpening,
)
from src.backend.image_logic.retouch import (
    apply_dust_removal,
)
from src.backend.image_logic.geometry import (
    get_autocrop_coords,
    apply_fine_rotation,
    map_coords_to_geometry,
)
from src.backend.image_logic.local_adjustments import apply_local_adjustments

logger = get_logger(__name__)


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

        # Use the active ROI for analysis if it exists
        analysis_img = img
        if context.active_crop_roi:
            y1, y2, x1, x2 = context.active_crop_roi
            analysis_img = img[y1:y2, x1:x2]

        bounds = measure_log_negative_bounds(analysis_img)
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

        # Calculate per-channel pivots (Base pivot without WB shift)
        pivots = [master_ref - exposure_shift] * 3

        # Convert UI CMY values to density offsets
        cmy_vals = [settings.wb_cyan, settings.wb_magenta, settings.wb_yellow]
        offsets_list = [cmy_to_density(val) for val in cmy_vals]
        cmy_offsets: Tuple[float, float, float] = (
            offsets_list[0],
            offsets_list[1],
            offsets_list[2],
        )

        logger.debug(f"CMY UI: {cmy_vals} -> Offsets: {cmy_offsets}")

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
            cmy_offsets=cmy_offsets,
        )

        res = np.clip(img_pos, 0, 1)
        if settings.process_mode == "B&W":
            res = convert_to_monochrome(res)
        return res


class ToningProcessor(Processor):
    """
    Simulates physical paper substrates and chemical toning processes.
    """

    def process(
        self, img: np.ndarray, settings: ImageSettings, context: PipelineContext
    ) -> np.ndarray:
        # 1. Simulate Physical Paper Substrate
        img = simulate_paper_substrate(img, settings.paper_profile)

        # 2. Apply Chemical Toning Simulation (B&W only)
        if settings.process_mode == "B&W":
            img = apply_chemical_toning(
                img,
                selenium_strength=settings.selenium_strength,
                sepia_strength=settings.sepia_strength,
            )

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

        h_img, w_img = img.shape[:2]

        # Map coordinates to current geometric state (rotated)

        # Note: LocalRetouchProcessor currently runs on the FULL (rotated) image,

        # so we don't pass context.active_crop_roi to map_coords_to_geometry here.

        orig_shape = context.original_size

        rotation = settings.rotation
        fine_rotation = settings.fine_rotation

        # 1. Map Manual Dust Spots
        mapped_settings = settings

        if settings.manual_dust_spots:
            # We must not mutate the original settings object

            new_spots = []

            for nx, ny, size in settings.manual_dust_spots:
                mnx, mny = map_coords_to_geometry(
                    nx, ny, orig_shape, rotation, fine_rotation, roi=None
                )

                new_spots.append((mnx, mny, size))

            # Use a temporary settings-like object or pass them directly if possible

            # For simplicity in this engine, we'll create a shallow copy and override

            mapped_settings = copy.copy(settings)

            mapped_settings.manual_dust_spots = new_spots

        # 2. Map Local Adjustments

        if settings.local_adjustments:
            if mapped_settings is settings:  # if not already copied
                mapped_settings = copy.copy(settings)

            new_adjustments = []

            for adj in settings.local_adjustments:
                new_adj = copy.copy(adj)

                new_points = []

                for nx, ny in adj.points:
                    mnx, mny = map_coords_to_geometry(
                        nx, ny, orig_shape, rotation, fine_rotation, roi=None
                    )

                    new_points.append((mnx, mny))

                new_adj.points = new_points

                new_adjustments.append(new_adj)

            mapped_settings.local_adjustments = new_adjustments

        # 1. Automated Cleanup

        img = apply_dust_removal(img, mapped_settings, scale_factor)

        # 2. Manual Dodge & Burn

        img = apply_local_adjustments(
            img, mapped_settings.local_adjustments, scale_factor
        )

        return img


class PhotoLabProcessor(Processor):
    """
    Simulates high-end laboratory scanner processing:
    - Spectral Crosstalk (Density space)
    - Hypertone (Local contrast)
    - Chroma Noise Removal
    - Output Sharpening
    """

    def process(
        self, img: np.ndarray, settings: ImageSettings, context: PipelineContext
    ) -> np.ndarray:
        scale_factor = context.scale_factor

        # 1. Spectral Crosstalk (Needs Density Space)
        epsilon = 1e-6
        img_dens = -np.log10(np.clip(img, epsilon, 1.0))

        crosstalk_strength = max(0.0, settings.color_separation - 1.0)
        img_dens = apply_spectral_crosstalk(
            img_dens, crosstalk_strength, settings.crosstalk_matrix
        )
        img = np.power(10.0, -img_dens)

        # 2. Scanner Emulation (Hypertone)
        if settings.hypertone_strength > 0:
            img = apply_hypertone(img, settings.hypertone_strength)

        # 3. Chroma Noise Removal
        img = apply_chroma_noise_removal(img, settings, scale_factor)

        # 4. Output Sharpening
        img = apply_output_sharpening(img, settings.sharpen)

        img *= 2.0**settings.exposure  # Linear exposure trim
        return np.clip(img, 0, 1)


class GeometryProcessor(Processor):
    """
    Handles rotations and defines the active crop ROI.
    """

    def process(
        self, img: np.ndarray, settings: ImageSettings, context: PipelineContext
    ) -> np.ndarray:
        scale_factor = context.scale_factor

        # 1. Apply Rotations immediately so the grid is aligned
        if settings.rotation != 0:
            img = np.rot90(img, k=settings.rotation)

        if settings.fine_rotation != 0.0:
            img = apply_fine_rotation(img, settings.fine_rotation)

        # 2. Calculate ROI but do not crop yet
        if settings.autocrop:
            roi = get_autocrop_coords(
                img,
                offset_px=settings.autocrop_offset,
                scale_factor=scale_factor,
                target_ratio_str=settings.autocrop_ratio,
                detect_res=settings.working_copy_size,
            )
            context.active_crop_roi = roi
        else:
            context.active_crop_roi = None

        return img


class OutputCropProcessor(Processor):
    """
    Final step: Applies the crop ROI defined earlier in the pipeline.
    """

    def process(
        self, img: np.ndarray, settings: ImageSettings, context: PipelineContext
    ) -> np.ndarray:
        if context.active_crop_roi:
            y1, y2, x1, x2 = context.active_crop_roi
            return img[y1:y2, x1:x2]
        return img
