from dataclasses import dataclass
from typing import Optional
from enum import StrEnum


class ProcessMode(StrEnum):
    C41 = "C41"
    BW = "B&W"


@dataclass(frozen=True)
class ProcessConfig:
    """
    Core film/sensor processing parameters.
    """

    process_mode: str = ProcessMode.C41
    analysis_buffer: float = 0.07
    use_roll_average: bool = False
    locked_floors: tuple[float, float, float] = (0.0, 0.0, 0.0)
    locked_ceils: tuple[float, float, float] = (1.0, 1.0, 1.0)
    local_floors: tuple[float, float, float] = (0.0, 0.0, 0.0)
    local_ceils: tuple[float, float, float] = (1.0, 1.0, 1.0)
    roll_name: Optional[str] = None
