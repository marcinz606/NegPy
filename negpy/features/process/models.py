from dataclasses import dataclass
from enum import StrEnum
from typing import Optional


class ProcessMode(StrEnum):
    C41 = "C41"
    BW = "B&W"
    E6 = "E-6"


@dataclass(frozen=True)
class ProcessConfig:
    """
    Core film/sensor processing parameters.
    """

    process_mode: ProcessMode = ProcessMode.C41
    analysis_buffer: float = 0.05
    drange_clip: float = 0.0
    e6_normalize: bool = True
    use_roll_average: bool = False
    locked_floors: tuple[float, float, float] = (0.0, 0.0, 0.0)
    locked_ceils: tuple[float, float, float] = (0.0, 0.0, 0.0)
    local_floors: tuple[float, float, float] = (0.0, 0.0, 0.0)
    local_ceils: tuple[float, float, float] = (0.0, 0.0, 0.0)

    white_point_offset: float = 0.0
    black_point_offset: float = 0.0

    lock_bounds: bool = False

    roll_name: Optional[str] = None

    def __post_init__(self) -> None:
        """
        Ensure JSON-loaded lists are converted back to tuples.
        """
        object.__setattr__(self, "locked_floors", tuple(self.locked_floors))
        object.__setattr__(self, "locked_ceils", tuple(self.locked_ceils))
        object.__setattr__(self, "local_floors", tuple(self.local_floors))
        object.__setattr__(self, "local_ceils", tuple(self.local_ceils))

    @property
    def is_local_initialized(self) -> bool:
        """Checks if per-file auto-exposure has been performed."""
        return any(v != 0.0 for v in self.local_floors)

    @property
    def is_locked_initialized(self) -> bool:
        """Checks if a roll-wide baseline is available."""
        return any(v != 0.0 for v in self.locked_floors)


def invalidate_local_bounds(process: ProcessConfig) -> dict:
    """Returns kwargs for dataclasses.replace that clear local bounds; no-op when lock_bounds=True."""
    if process.lock_bounds:
        return {}
    return {"local_floors": (0.0, 0.0, 0.0), "local_ceils": (0.0, 0.0, 0.0)}
