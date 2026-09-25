from enum import StrEnum
from typing import Optional

from negpy.features.exposure.models import RenderIntent
from negpy.features.process.models import ProcessConfig, ProcessMode


class RenderPath(StrEnum):
    PRINT = "print"
    TRANSFER = "transfer"
    POSITIVE = "positive"


def render_path(process: ProcessConfig, render_intent: Optional[str] = None) -> RenderPath:
    """Which renderer a frame goes through. Every caller asks this, so CPU, GPU, decode and UI
    cannot disagree.

    PRINT is the negative's print curve: every negative, a Slide with Normalize on, and any
    Flat master. TRANSFER is a Slide as captured. POSITIVE is a Slide marked Positive: a file
    already positivized before NegPy saw it, metered but with no display rendering. Only a
    Slide config carries Positive (ProcessConfig.__post_init__), and Normalize outranks it.
    """
    if render_intent == RenderIntent.FLAT or process.process_mode != ProcessMode.E6 or process.e6_normalize:
        return RenderPath.PRINT
    return RenderPath.POSITIVE if process.positive_source else RenderPath.TRANSFER
