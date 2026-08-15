from dataclasses import dataclass, field
from typing import List, Tuple

# Heal `size` is a diameter at this reference long edge (px). Pinned, so stored heals keep
# their footprint when preview_render_size changes.
HEAL_SIZE_REF = 1600

# IR reconstruction methods: the ratio/score/fill chain in logic.py, and the Digital ICE
# port in openice.py. Two implementations of one stage, side by side until scans decide.
IR_METHOD_NEGPY = "negpy"
IR_METHOD_OPENICE = "openice"
IR_METHODS = (IR_METHOD_NEGPY, IR_METHOD_OPENICE)


@dataclass(frozen=True)
class RetouchConfig:
    dust_remove: bool = False
    dust_threshold: float = 0.66
    dust_size: int = 4
    manual_dust_spots: List[Tuple[float, float, float]] = field(default_factory=list)
    # Each stroke: (points, size, src_dx, src_dy). points = [[nx, ny], ...] source-normalized,
    # size = diameter at HEAL_SIZE_REF scale, (src_dx, src_dy) = the source-normalized offset
    # to the clone source. A single-point stroke is a spot. manual_dust_spots is the legacy
    # pre-stroke format.
    manual_heal_strokes: List[Tuple] = field(default_factory=list)
    # Each line: (nx0, ny0, nx1, ny1, width), the source-normalized endpoints of a transport
    # scratch plus the width the guide drew, at HEAL_SIZE_REF scale. Traced from one click.
    # What gets repaired is re-measured from the line's own evidence at render time.
    scratch_lines: List[Tuple] = field(default_factory=list)
    # How readily a transport scratch is followed, where higher is conservative, as with the
    # other thresholds. Sets the ridge bar for both the extent along the line and the band
    # grown out from it, so it trades reach against picking up film either side.
    scratch_threshold: float = 0.5
    manual_dust_size: int = 6
    ir_dust_remove: bool = False
    # Which reconstruction runs (IR_METHODS). ir_attenuation belongs to the NegPy method
    # alone: OpenICE folds that tier into its own base term.
    ir_method: str = IR_METHOD_NEGPY
    # Sensitivity on the normalized IR ratio, where higher is conservative. The default gives
    # a cutoff of 0.59 with attenuation on: division fixes shallow dust and the fill rebuilds
    # cores. OpenICE reads it as a bias on its per-frame weight ramp.
    ir_threshold: float = 0.66
    # IR-division tier: recover the image under semi-transparent dust, with no cloning.
    # Tracks ir_dust_remove from the single "IR Removal" control. B&W and Kodachrome frames
    # are auto-skipped by the degenerate guard, not by this flag.
    ir_attenuation: bool = True
