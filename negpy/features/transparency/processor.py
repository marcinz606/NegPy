"""The slide renders: a Slide as captured and a Positive frame. The engine routes here by
process.path.render_path."""

import numpy as np

from negpy.domain.interfaces import PipelineContext
from negpy.domain.types import ImageBuffer
from negpy.features.exposure.analysis import density_histogram
from negpy.features.exposure.logic import cast_solve_inputs, filtration_offsets, neutral_axis_affine
from negpy.features.exposure.models import EXPOSURE_CONSTANTS, ExposureConfig
from negpy.features.exposure.normalization import (
    LogNegativeBounds,
    effective_crosstalk_matrix,
    luminance_density_range,
    measure_anchor_from_log,
    measure_clip_fractions,
    measure_highlight_point_from_log,
    measure_neutral_axis_from_log,
    measure_shadow_point_from_log,
    measure_textural_range_from_log,
    normalize_log_image,
    prefilter_log_grid,
    resolve_analysis_region,
    unmix_log_image,
)
from negpy.features.process.capture_color import apply_camera_matrix, camera_to_working_matrix
from negpy.features.process.logic import should_fold_camera_wb
from negpy.features.process.models import ProcessConfig
from negpy.features.transparency.logic import (
    TRANSFER_DENSITY_RANGE,
    apply_transfer_curve,
    transfer_assumed_anchor,
    transfer_auto_terms,
    transfer_bounds,
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
        Camera primaries -> working space, then a FIXED log-density window.

        No meter shapes the window itself. Measured bounds are exactly what makes two
        exposures of one slide render alike, and a raw slide was exposed deliberately
        — so the window is anchored to the decoder's white level, identical for every
        frame, and a brighter capture stays brighter. The meters Auto Density/Auto Grade
        read are stored below; they move the render only when those toggles are on, and a
        slide starts with both off (auto_meter_for_mode).
        """
        epsilon = 1e-6
        # Linear RAW decodes without white balance, which the row-normalized camera matrix
        # assumes. Folding the as-shot multipliers back in makes this render independent of which
        # decode produced the buffer, so the toggle cannot cast the image here — except under
        # narrowband light, where the fold never runs: an as-shot WB estimate describes a
        # continuous-spectrum scene, and there is no such scene to describe (see
        # should_fold_camera_wb).
        matrix = camera_to_working_matrix(context.cam_xyz, context.camera_wb if should_fold_camera_wb(self.config) else None)
        linear = apply_camera_matrix(np.nan_to_num(image, nan=epsilon, posinf=1.0, neginf=epsilon), matrix)

        img_log = np.log10(np.clip(linear, epsilon, None))
        # Honoured here too: a rig-calibrated matrix is a capture correction like Hue Trim, and
        # the mode gate keeps a negative's profile from touching a slide. Inert at the shipped
        # default, so the as-captured render is unperturbed.
        unmix = effective_crosstalk_matrix(self.config, context.process_mode)
        img_log = unmix_log_image(img_log, unmix)
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
        an_roi, an_buffer = resolve_analysis_region(
            linear.shape, context.active_roi, self.config.analysis_buffer, self.config.analysis_rect
        )
        prefiltered = unmix_log_image(prefilter_log_grid(linear, an_roi, an_buffer), unmix)

        # Cast Removal's neutral axis is metered on the working-space log image the curve
        # consumes, since the camera matrix above would leave a meter reading a different
        # space than the GPU's. Pre-trim bounds, so a White/Black Point nudge cannot
        # perturb it.
        if self.cast_strength > 0.0:
            context.metrics["neutral_axis_refs"] = measure_neutral_axis_from_log(prefiltered, pre_trim_bounds, None, 0.0)

        # Auto Density and Auto Grade meter against the same fixed pre-trim window.
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
        context.metrics["scan_clip_fractions"] = measure_clip_fractions(image, context.active_roi, self.config.analysis_buffer)
        context.metrics["normalized_log"] = res
        context.metrics["histogram_density"] = density_histogram(res, context.active_roi)
        return res


class TransferProcessor:
    """Normalized log density -> scene-linear positive through the transfer curve."""

    def __init__(self, config: ExposureConfig, positive_source: bool = False):
        self.config = config
        self.positive_source = positive_source

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
        exposure_offset, contrast, highlight_auto = transfer_auto_terms(
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
        # Shadow refs stay out: the P98 tie is calibrated for a negative. With no neutral
        # axis this solves to the identity and the capture passes through.
        strength, _shadow_refs_norm, neutral_axis_norm = cast_solve_inputs(
            final_bounds,
            None,
            context.metrics.get("neutral_axis_refs"),
            self.config.cast_removal_strength,
        )
        cast_gain, cast_offset_norm = neutral_axis_affine(neutral_axis_norm, strength)
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
            shadow_density=self.config.shadow_density,
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
