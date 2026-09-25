from enum import StrEnum
from negpy.features.process.models import ProcessConfig, ProcessMode


class RenderPath(StrEnum):
    PRINT = "print"
    TRANSFER = "transfer"
    POSITIVE = "positive"


def render_path(process: ProcessConfig) -> RenderPath:
    """Which renderer a frame goes through. Every caller asks this, so CPU, GPU, decode and UI
    cannot disagree.

    PRINT is the negative's print curve. TRANSFER is a Slide as captured. POSITIVE is a Slide
    marked Positive: a file already positivized before NegPy saw it, decoded on its own profile
    and shown with no display rendering. Only a Slide config carries Positive
    (ProcessConfig.__post_init__). A Flat master takes the same base as its print.
    """
    if process.process_mode != ProcessMode.E6:
        return RenderPath.PRINT
    return RenderPath.POSITIVE if process.positive_source else RenderPath.TRANSFER
