from dataclasses import dataclass
from enum import StrEnum

from negpy.kernel.system.i18n import mark


class Phase(StrEnum):
    """Which iteration phase the proposal is in."""

    FIXED = "fixed"
    REFINE = "refine"


@dataclass(frozen=True)
class PickParam:
    """One tunable Exposure control the picker proposes.

    `strong_step` is the bold step a first selection moves; `floor` is the smallest step;
    `lo`/`hi` clamp a candidate to its slider range. `compound` marks a control spanning a
    field pair (Temperature moves wb_magenta/wb_yellow; its "value" is Kelvin).
    """

    field: str
    name: str
    strong_step: float
    floor: float
    lo: float
    hi: float
    compound: bool = False


# The eleven tunable Exposure controls, by parameter index (0 to 10): Contrast (grade) and
# Exposure (density) first, then the split grade and split density trims, then Preflash, Dye
# Separation, Snap, Toe and Temperature. The excluded controls (Luma/Color Clip, Crop,
# Crosstalk, Cast Removal, Hue Trim, Black/White Point) never appear here.
PICK_PARAMS: tuple[PickParam, ...] = (
    PickParam("grade", mark("ISO-R Grade"), 20.0, 1.0, 50.0, 180.0),
    PickParam("density", mark("Print Density"), 0.3, 0.01, 0.0, 2.0),
    PickParam("highlight_grade", mark("Highlights Grade"), 15.0, 1.0, -50.0, 50.0),
    PickParam("shadow_grade", mark("Shadows Grade"), 15.0, 1.0, -50.0, 50.0),
    PickParam("highlight_density", mark("Highlights Density"), 0.2, 0.01, -0.5, 0.5),
    PickParam("shadow_density", mark("Shadows Density"), 0.3, 0.02, -0.9, 0.9),
    PickParam("preflash", mark("Preflash"), 0.1, 0.01, 0.0, 1.0),
    PickParam("dye_separation", mark("Dye Separation"), 0.1, 0.01, 0.5, 1.5),
    PickParam("midtone_gamma", mark("Snap"), 0.05, 0.005, -0.5, 0.5),
    PickParam("toe", mark("Toe"), 0.1, 0.01, -1.0, 1.0),
    PickParam("temperature", mark("Temperature"), 500.0, 50.0, 3000.0, 12000.0, compound=True),
)


@dataclass(frozen=True)
class PickBetterProgress:
    """The iteration cursor the controller advances after each pick.

    `selection_counts` holds, per parameter, how many times that parameter has been committed;
    the proposed step shrinks with the count (bold at first, smaller on each later visit of the
    same parameter). `round_number` is the 1-based count of rounds run, shown at the head of the
    in-flight parameter name. `param_order` is the Custom Routing route (the order the rounds visit
    the eleven parameters); `randomize_order` is a shuffled copy used for picking when `randomize`
    is on, so the route keeps the randomize-off order.
    """

    phase: Phase
    param_index: int
    round_number: int
    selection_counts: tuple[int, ...]
    # The Custom Routing route: the order the rounds visit the eleven parameters. A drag-and-drop
    # reorder sets it; a randomized run shuffles a copy into randomize_order and leaves it untouched.
    # selection_counts stays keyed by the real parameter index, so a reorder reorders rounds but each
    # parameter's step still shrinks by its own count.
    param_order: tuple[int, ...] = tuple(range(len(PICK_PARAMS)))
    randomize: bool = False
    randomize_order: tuple[int, ...] = tuple(range(len(PICK_PARAMS)))
