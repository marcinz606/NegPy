from typing import Protocol, Optional, Any, runtime_checkable
from dataclasses import dataclass, field
from src.core.types import ImageBuffer, ROI, Dimensions


@dataclass
class PipelineContext:
    """


    Shared state passed through the pipeline.


    """

    original_size: Dimensions

    scale_factor: float

    process_mode: str = "C41"

    # ROI detected by geometry step, applied by crop step

    active_roi: Optional[ROI] = None
    # Metrics gathered by analysis steps (e.g., histogram bounds)
    metrics: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class IProcessor(Protocol):
    """
    Interface for any image processing step.
    """

    def process(self, image: ImageBuffer, context: PipelineContext) -> ImageBuffer: ...


class IImageSource(Protocol):
    """
    Interface for loading images.
    """

    def read(self) -> ImageBuffer: ...
