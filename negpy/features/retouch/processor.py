from negpy.domain.interfaces import PipelineContext
from negpy.domain.types import ImageBuffer
from negpy.features.retouch.ir_heal import IRSupportError, apply_ir_dust_removal, detect_ir_repair_support
from negpy.features.retouch.models import RetouchConfig
from negpy.features.retouch.logic import apply_manual_heals, build_heal_regions
from negpy.kernel.system.logging import get_logger


logger = get_logger(__name__)


class RetouchProcessor:
    """
    Applies the IR pixel-mask healer, then membrane-clones manual and
    luminance-detected regions injected by ImageProcessor._augment_retouch.
    """

    def __init__(self, config: RetouchConfig):
        self.config = config

    def process(self, image: ImageBuffer, context: PipelineContext) -> ImageBuffer:
        img = image

        ir = context.metrics.get("ir_post_geometry")
        if self.config.ir_dust_remove:
            if ir is None:
                context.metrics["ir_dust_status"] = "skipped"
                context.metrics["ir_dust_skipped"] = (
                    "No IR channel is available for this scan"
                )
                logger.warning(
                    "IR dust repair skipped: %s",
                    context.metrics["ir_dust_skipped"],
                )
            elif ir.shape[:2] != img.shape[:2]:
                context.metrics["ir_dust_status"] = "skipped"
                context.metrics["ir_dust_skipped"] = f"RGB {img.shape[:2]} and IR {ir.shape[:2]} shapes differ"
                logger.warning("IR dust repair skipped: %s", context.metrics["ir_dust_skipped"])
            else:
                threshold = 1.0 - self.config.ir_threshold
                try:
                    support = detect_ir_repair_support(ir, threshold=threshold)
                except IRSupportError as exc:
                    # Fail closed: a bad/opaque IR plane must never turn the
                    # whole photograph into one giant heal operation.
                    context.metrics["ir_dust_status"] = "skipped"
                    context.metrics["ir_dust_skipped"] = str(exc)
                    logger.warning("IR dust repair skipped: %s", exc)
                else:
                    img, mask = apply_ir_dust_removal(
                        img,
                        ir,
                        threshold,
                        self.config.ir_inpaint_radius,
                        context.scale_factor,
                        support_bounds=support.bounds,
                    )
                    context.metrics["ir_dust_support"] = support.to_dict()
                    context.metrics["ir_dust_mask_pixels"] = int((mask > 0).sum())
                    context.metrics["ir_dust_status"] = "applied"

        if not (self.config.manual_heal_strokes or self.config.manual_dust_spots):
            return img

        orig_h, orig_w = context.original_size
        rot_params = context.metrics.get(
            "geometry_params",
            {
                "rotation": 0,
                "fine_rotation": 0.0,
                "flip_horizontal": False,
                "flip_vertical": False,
            },
        )
        distortion_k1 = context.metrics.get("distortion_k1", 0.0)

        heal_regions = build_heal_regions(
            self.config.manual_heal_strokes,
            self.config.manual_dust_spots,
            (orig_h, orig_w),
            rot_params.get("rotation", 0),
            rot_params.get("fine_rotation", 0.0),
            rot_params.get("flip_horizontal", False),
            rot_params.get("flip_vertical", False),
            distortion_k1,
            (img.shape[1], img.shape[0]),
        )
        return apply_manual_heals(img, *heal_regions)
