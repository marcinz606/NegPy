"""Printer's notes: the print recipe and the dodge/burn map as a darkroom printer writes them.

Everything here is text, in the *exposure* domain a printing record uses: a burn adds
exposure and reads `+`, a dodge withholds it and reads `−`. `LocalMask.stops` carries
that same convention, so nothing here re-signs it.
"""

from dataclasses import dataclass
from typing import List

from negpy.features.exposure.models import EXPOSURE_CONSTANTS, ExposureConfig
from negpy.features.exposure.papers import resolve_paper
from negpy.features.finish.models import FinishConfig
from negpy.features.local.models import LocalAdjustmentsConfig

# Vulgar fractions a printer would actually write; anything else prints as a decimal.
_FRACTIONS = ((0.25, "¼"), (1.0 / 3.0, "⅓"), (0.5, "½"), (2.0 / 3.0, "⅔"), (0.75, "¾"))
_FRAC_TOLERANCE = 0.02


def stops_label(stops: float) -> str:
    """Exposure difference in stops, darkroom-signed: + = more exposure (burn)."""
    mag = abs(float(stops))
    if mag < 0.005:
        return "0"
    sign = "−" if stops < 0 else "+"
    whole = int(mag)
    frac = mag - whole
    for value, glyph in _FRACTIONS:
        if abs(frac - value) <= _FRAC_TOLERANCE:
            return f"{sign}{whole or ''}{glyph}"
    if frac <= _FRAC_TOLERANCE:
        return f"{sign}{whole}"
    return f"{sign}{mag:.2f}"


def local_grade_label(grade: float, delta: float) -> str:
    """The grade a mask actually prints at, as "R95"; empty when it prints at the frame's."""
    if not delta or grade <= 0.0:
        return ""
    c = EXPOSURE_CONSTANTS
    r = min(max(grade + delta, float(c["iso_r_min"])), float(c["iso_r_max"]))
    return f"R{r:.0f}"


@dataclass(frozen=True)
class MaskNote:
    number: int  # 1-based, matching the Dodge & Burn mask list
    is_burn: bool
    stops: str  # stops_label's "0" when the mask changes nothing but grade
    local_r: str = ""

    @property
    def kind(self) -> str:
        if self.stops == "0":
            return "Grade"
        return "Burn" if self.is_burn else "Dodge"

    @property
    def badge(self) -> str:
        """Short form for the map drawn on the print."""
        parts = [str(self.number)]
        if self.stops != "0":
            parts.append(self.stops)
        if self.local_r:
            parts.append(self.local_r)
        return " ".join(parts)

    @property
    def summary(self) -> str:
        """Long form for the record card."""
        text = f"{self.number} {self.kind}"
        if self.stops != "0":
            text += f" {self.stops}"
        if self.local_r:
            text += f" @ {self.local_r}"
        return text


def mask_notes(local: LocalAdjustmentsConfig, grade: float = 0.0) -> List[MaskNote]:
    """One note per mask, in list order. `grade` is the frame's ISO R, which turns a
    mask's grade delta into the grade it prints at."""
    return [
        MaskNote(
            number=i + 1,
            is_burn=m.stops > 0,
            stops=stops_label(m.stops),
            local_r=local_grade_label(grade, m.grade),
        )
        for i, m in enumerate(local.masks)
    ]


def recipe_lines(exposure: ExposureConfig, local: LocalAdjustmentsConfig, finish: FinishConfig, *, frame: str = "") -> List[str]:
    """The printing record: one line per decision that is not at its default."""
    lines: List[str] = []
    if frame:
        lines.append(frame)

    paper = resolve_paper(exposure.paper_profile).label
    flags = [name for name, on in (("Paper White", exposure.paper_dmin), ("Paper Black", exposure.paper_black)) if on]
    lines.append(" · ".join([paper, *flags]))

    density = f"Print Density {exposure.density:.2f}"
    if exposure.auto_exposure:
        density += " (auto)"
    lines.append(density)

    if exposure.shadow_density or exposure.highlight_density:
        lines.append(f"Zone density: shadows {exposure.shadow_density:+.2f} · highlights {exposure.highlight_density:+.2f}")

    grade = f"Grade ISO-R {exposure.grade:.0f}"
    if exposure.auto_normalize_contrast:
        grade += " (auto)"
    if exposure.shadow_grade or exposure.highlight_grade:
        grade += f" · split {exposure.shadow_grade:+.0f}/{exposure.highlight_grade:+.0f}"
    lines.append(grade)

    if exposure.wb_cyan or exposure.wb_magenta or exposure.wb_yellow:
        lines.append(f"Filtration C{exposure.wb_cyan:+.2f} M{exposure.wb_magenta:+.2f} Y{exposure.wb_yellow:+.2f}")

    if exposure.toe or exposure.shoulder:
        lines.append(f"Toe {exposure.toe:+.2f} · Shoulder {exposure.shoulder:+.2f}")

    if exposure.midtone_gamma:
        lines.append(f"Snap {exposure.midtone_gamma:+.2f}")

    if finish.vignette_stops:
        lines.append(f"Edge burn {stops_label(finish.vignette_stops)} stop")

    notes = mask_notes(local, exposure.grade)
    if notes:
        lines.append("Dodge & burn: " + " · ".join(n.summary for n in notes))

    return lines
