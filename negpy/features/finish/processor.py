from dataclasses import replace
from typing import TYPE_CHECKING, Any, Optional

import numpy as np

from negpy.domain.interfaces import PipelineContext
from negpy.domain.types import ImageBuffer
from negpy.features.altprocess.models import AltProcess
from negpy.features.exposure.models import RenderIntent
from negpy.features.exposure.processor import PhotometricProcessor
from negpy.features.finish.logic import apply_carrier, apply_vignette, carrier_tone_exposures, linear_carrier_tone
from negpy.features.finish.models import FinishConfig
from negpy.features.lab.logic import apply_saturation
from negpy.features.process.models import ProcessMode
from negpy.features.toning.processor import ToningProcessor

if TYPE_CHECKING:
    from negpy.domain.models import WorkspaceConfig

# log10 by which the film base is clearer than the normalization's thin bound.
REBATE_BASE_MARGIN = 0.15


def carrier_width_px(carrier_width_mm: float, print_size_cm: float, long_edge_px: float) -> float:
    """Rebate width in image pixels for a given print long edge in cm."""
    return (carrier_width_mm / max(print_size_cm * 10.0, 1.0)) * long_edge_px


def rebate_tone(settings: "WorkspaceConfig", metrics: Any) -> np.ndarray:
    """
    (CARRIER_TONE_SAMPLES, 3) paper-relative print color of a neutral density ramp off the
    film base, through this frame's curves, saturation and toning. Plain light where there
    is no negative print model.
    """
    mode = settings.process.process_mode
    bounds = metrics.get("final_bounds")
    if (
        bounds is None
        or mode not in (ProcessMode.C41, ProcessMode.BW)
        or settings.process.positive_source
        or settings.exposure.render_intent == RenderIntent.FLAT
        or settings.altproc.alt_process != AltProcess.NONE
    ):
        return linear_carrier_tone()
    floors = np.asarray(bounds.floors, dtype=np.float32)
    ceils = np.asarray(bounds.ceils, dtype=np.float32)
    t = np.maximum(carrier_tone_exposures(), np.float32(1e-6))
    strip = 1.0 + (REBATE_BASE_MARGIN + np.log10(t)[:, None]) / np.maximum(ceils - floors, 1e-6)[None, :]
    ctx = PipelineContext(original_size=(1, len(t)), scale_factor=1.0, process_mode=mode, metrics=dict(metrics))
    exposure = replace(settings.exposure, contrast_mask=0.0)
    printed = PhotometricProcessor(exposure).process(strip[None].astype(np.float32), ctx)
    if settings.lab.saturation != 1.0 or settings.lab.skin_protection > 0:
        printed = apply_saturation(printed, settings.lab.saturation, settings.lab.skin_protection)
    printed = np.asarray(ToningProcessor(settings.toning, settings.altproc.alt_process).process(printed, ctx))[0]
    return np.ascontiguousarray(printed / np.maximum(printed[:1], 1e-6), dtype=np.float32)


class FinishProcessor:
    def __init__(
        self,
        config: FinishConfig,
        print_size_cm: float = 30.0,
        paper: tuple[float, float, float] = (1.0, 1.0, 1.0),
        tone: Optional[np.ndarray] = None,
    ):
        self.config = config
        self.print_size_cm = print_size_cm
        self.paper = paper
        self.tone = tone

    def process(self, image: ImageBuffer, context: PipelineContext) -> ImageBuffer:
        if self.config.vignette_stops != 0.0:
            image = apply_vignette(image, self.config.vignette_stops, self.config.vignette_size, self.config.vignette_roundness)
        if self.config.carrier_width > 0.0:
            width = carrier_width_px(self.config.carrier_width, self.print_size_cm, float(max(image.shape[:2])))
            image = apply_carrier(
                image,
                width,
                self.config.carrier_rough,
                self.config.carrier_flare,
                self.config.carrier_corner,
                self.paper,
                self.tone,
            )
        return image
