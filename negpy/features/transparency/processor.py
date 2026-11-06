"""The slide renders: a Slide as captured and a Positive frame. The engine routes here by
process.path.render_path."""

from typing import Optional

import negpy_core
import numpy as np

from negpy.domain.interfaces import PipelineContext
from negpy.domain.types import ImageBuffer
from negpy.features.exposure.analysis import density_histogram, density_parade, scan_full_scale_columns
from negpy.features.exposure.logic import filtration_offsets
from negpy.features.exposure.models import EXPOSURE_CONSTANTS, ExposureConfig
from negpy.features.exposure.normalization import (
    LogNegativeBounds,
    luminance_density_range,
    measure_anchor_from_log,
    measure_clip_fractions,
    measure_highlight_point_from_log,
    measure_shadow_point_from_log,
    measure_textural_range_from_log,
    normalize_log_image,
    prefilter_log_grid,
    relog_matrix_image,
    resolve_analysis_region,
    unmix_log_image,
)
from negpy.features.process.fade import (
    apply_crossover,
    crossover_offsets,
    crossover_space,
    fade_density,
    render_white_space,
    resolve_crossover,
    resolve_fade_white,
    with_fade_white,
)
from negpy.features.process.capture_color import apply_camera_matrix
from negpy.features.process.models import ProcessConfig
from negpy.features.process.path import RenderPath, render_path
from negpy.features.transparency.logic import (
    TRANSFER_DENSITY_RANGE,
    apply_transfer_curve,
    fade_white_grid,
    density_stage,
    measure_transfer_neutral_axis,
    transfer_assumed_anchor,
    transfer_auto_terms,
    transfer_bounds,
    transfer_cast_affine,
    transfer_curve_params,
    transfer_point_offsets,
    transfer_widths,
)


class TransparencyBaseProcessor:
    """Linear capture -> normalized log density over the fixed transfer window."""

    def __init__(self, config: ProcessConfig, cast_strength: float = 0.0):
        self.config = config
        # The neutral axis is measured only when Cast Removal can use it. `base_key` in
        # engine.py carries the gate.
        self.cast_strength = cast_strength

    def process(self, image: ImageBuffer, context: PipelineContext) -> ImageBuffer:
        """
        Camera or source primaries -> working space, then a FIXED log-density window.

        No meter shapes the window itself. Measured bounds are exactly what makes two
        exposures of one slide render alike, and a raw slide was exposed deliberately
        — so the window is anchored to the decoder's white level, identical for every
        frame, and a brighter capture stays brighter. The meters Auto Density/Auto Grade
        read are stored below; they move the render only when those toggles are on, and a
        slide starts with both off (auto_meter_for_mode).
        """
        epsilon = 1e-6
        image = np.ascontiguousarray(image, dtype=np.float32)
        an_roi, an_buffer = resolve_analysis_region(
            image.shape, context.active_roi, self.config.analysis_buffer, self.config.analysis_rect, self.config.meter_rect
        )

        image_grid: Optional[ImageBuffer] = None

        def image_log_grid() -> ImageBuffer:
            """The prefiltered *image*, built once: the white and, with no matrix before the log, the meters read it."""
            nonlocal image_grid
            if image_grid is None:
                image_grid = prefilter_log_grid(image, an_roi, an_buffer)
            return image_grid

        def white_grid() -> ImageBuffer:
            return fade_white_grid(
                self.config, context.process_mode, context.cam_xyz, context.camera_wb, image, an_roi, an_buffer, image_log_grid
            )

        white_space = render_white_space(self.config, context.process_mode, context.cam_xyz, context.camera_wb)
        measured = resolve_fade_white(self.config, white_space, white_grid)
        if measured is not None:
            context.metrics["fade_white"] = measured
        # Honoured here too: a rig-calibrated matrix is a capture correction like Hue Trim, and
        # the mode gate keeps a negative's profile from touching a slide. Inert at the shipped
        # default, so the as-captured render is unperturbed.
        density = fade_density(with_fade_white(self.config, measured), context.process_mode, white_space)
        stage = density_stage(self.config, context.cam_xyz, context.camera_wb, density.matrix, density.anchor)
        linear = image if stage.pre is None else apply_camera_matrix(negpy_core.nan_to_num(image, epsilon, 1.0, epsilon), stage.pre)
        img_log = negpy_core.log_density_open(np.ascontiguousarray(linear, dtype=np.float32))
        log_grid = image_log_grid() if stage.pre is None else prefilter_log_grid(linear, an_roi, an_buffer)
        faded_grid = unmix_log_image(log_grid, stage.unmix, stage.anchor)
        # The crossover table acts in the dye channels after the fade, before the primaries mix them.
        table, measured_table = resolve_crossover(self.config, crossover_space(density, white_space, self.config), lambda: faded_grid)
        if measured_table is not None:
            context.metrics["fade_crossover"] = measured_table
        offsets = crossover_offsets(table, self.config.fade_crossover)
        img_log = relog_matrix_image(apply_crossover(unmix_log_image(img_log, stage.unmix, stage.anchor), offsets), stage.post)
        floors, ceils = transfer_bounds()
        pre_trim_bounds = LogNegativeBounds(floors=floors, ceils=ceils)
        # White/Black Point manually deviate the fixed window, same technique as the measured
        # path below: a user-driven nudge, not a meter, so it does not reopen what the fixed
        # window exists to prevent (see render_path).
        wp3, bp3 = transfer_point_offsets(self.config)
        if any(v != 0.0 for v in wp3 + bp3):
            floors = (floors[0] + wp3[0], floors[1] + wp3[1], floors[2] + wp3[2])
            ceils = (ceils[0] + bp3[0], ceils[1] + bp3[1], ceils[2] + bp3[2])
        bounds = LogNegativeBounds(floors=floors, ceils=ceils)
        res = normalize_log_image(img_log, bounds)

        # Shared prefilter for the neutral axis and Auto Density/Grade's four meters --
        # working-space (post camera matrix) and post-unmix, since that is what this curve
        # itself consumes.
        prefiltered = relog_matrix_image(apply_crossover(faded_grid, offsets), stage.post)
        if context.wants_fade_grid:
            context.metrics["transfer_grid"] = white_grid()

        # Cast Removal's neutral axis is metered on the working-space log image the curve
        # consumes, since the camera matrix above would leave a meter reading a different
        # space than the GPU's. Pre-trim bounds, so a White/Black Point nudge cannot
        # perturb it.
        if self.cast_strength > 0.0:
            positive = render_path(self.config) is RenderPath.POSITIVE
            context.metrics["neutral_axis_refs"] = measure_transfer_neutral_axis(prefiltered, positive)

        # Auto Density and Auto Grade meter against the fixed pre-trim window.
        context.metrics["metered_anchor"] = measure_anchor_from_log(
            prefiltered, pre_trim_bounds, None, 0.0, assumed=transfer_assumed_anchor()
        )
        context.metrics["textural_range"] = measure_textural_range_from_log(prefiltered, None, 0.0)
        context.metrics["shadow_point"] = measure_shadow_point_from_log(prefiltered, pre_trim_bounds, None, 0.0)
        context.metrics["highlight_point"] = measure_highlight_point_from_log(prefiltered, pre_trim_bounds, None, 0.0)

        context.metrics["log_bounds"] = bounds
        context.metrics["log_bounds_base"] = bounds
        context.metrics["final_bounds"] = bounds
        context.metrics["norm_density_range"] = luminance_density_range(bounds)
        context.metrics["density_above_base"] = None  # a slide has no clear base to read above
        context.metrics["scan_clip_fractions"] = measure_clip_fractions(image, an_roi, an_buffer)
        context.metrics["normalized_log"] = res
        context.metrics["histogram_density"] = density_histogram(res, context.active_roi)
        # The Crop tool shows the whole frame, so the parade's columns span it too (the GPU's full_frame).
        # Export and batch thumbnails render with stage caching off and skip it.
        if context.cache_stages:
            parade_roi = None if context.crop_preview_full else context.active_roi
            context.metrics["density_parade"] = density_parade(img_log, parade_roi)
            context.metrics["scan_full_scale"] = scan_full_scale_columns(image, parade_roi)
        return res


class TransferProcessor:
    """Normalized log density -> scene-linear positive through the transfer curve."""

    def __init__(self, config: ExposureConfig, positive_source: bool = False, white_target: Optional[float] = None):
        self.config = config
        self.positive_source = positive_source
        self.white_target = white_target

    def process(self, image: ImageBuffer, context: PipelineContext) -> ImageBuffer:
        """
        The exact inverse of the fixed-bounds normalization,
        deviated only by what the user has actually moved -- plus Auto Density/Auto
        Grade, restated on this curve (transfer_auto_terms).

        The autos follow their toggles, which a slide starts with off: reading the frame to
        decide a look is the opposite of starting from the capture. Cast Removal starts at 0
        on a slide too: what it corrects here is a faded original's crossover, and a
        deliberate colour cast is the photograph.
        """
        exposure_offset, contrast, toe3, sh3 = transfer_curve_params(self.config)
        exposure_offset, contrast, highlight_auto, shadow_auto = transfer_auto_terms(
            self.config,
            exposure_offset,
            contrast,
            context.metrics.get("textural_range"),
            context.metrics.get("metered_anchor"),
            context.metrics.get("shadow_point"),
            context.metrics.get("highlight_point"),
        )
        final_bounds = context.metrics.get("final_bounds")
        cmy_offsets = filtration_offsets(
            (self.config.wb_cyan, self.config.wb_magenta, self.config.wb_yellow),
            final_bounds,
        )
        cmy_max = EXPOSURE_CONSTANTS["cmy_max_density"]
        shadow_cmy = (
            self.config.shadow_cyan * cmy_max,
            self.config.shadow_magenta * cmy_max,
            self.config.shadow_yellow * cmy_max,
        )
        highlight_cmy = (
            self.config.highlight_cyan * cmy_max,
            self.config.highlight_magenta * cmy_max,
            self.config.highlight_yellow * cmy_max,
        )
        # With no neutral axis this solves to the identity and the capture passes through.
        cast_gain, cast_offset_norm = transfer_cast_affine(
            final_bounds,
            context.metrics.get("neutral_axis_refs"),
            self.config.cast_removal_strength,
            self.white_target,
        )
        cast_offset = tuple(o * TRANSFER_DENSITY_RANGE for o in cast_offset_norm)

        tw3, sw3 = transfer_widths(self.config)
        return apply_transfer_curve(
            image,
            exposure_offset,
            contrast,
            toe3,
            sh3,
            cmy_offsets,
            tw3,
            sw3,
            shadow_density=self.config.shadow_density + shadow_auto,
            highlight_density=self.config.highlight_density + highlight_auto,
            shadow_cmy=shadow_cmy,
            highlight_cmy=highlight_cmy,
            cast_gain=cast_gain,
            cast_offset=cast_offset,
            positive_source=self.positive_source,
            separation=self.config.dye_separation,
            separation_trims=(
                self.config.dye_separation_trim_red,
                self.config.dye_separation_trim_green,
                self.config.dye_separation_trim_blue,
            ),
            damping=self.config.separation_damping,
        )
