"""Printer's notes: the print recipe and the dodge/burn map as a darkroom printer writes them.

Everything here is text, in the *exposure* domain a printing record uses: a burn adds
exposure and reads `+`, a dodge withholds it and reads `−`. `LocalMask.stops` carries
that same convention, so nothing here re-signs it.
"""

from dataclasses import dataclass
from typing import List

from negpy.features.exposure.densitometer import zone_roman
from negpy.features.exposure.models import EXPOSURE_CONSTANTS, ExposureConfig
from negpy.features.exposure.papers import resolve_paper
from negpy.features.finish.models import FinishConfig
from negpy.features.local.logic import limited_indices
from negpy.features.local.models import LocalAdjustmentsConfig, LocalMask, MaskKey
from negpy.kernel.system.i18n import mark, tr

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


def tone_limit_label(mask: LocalMask) -> str:
    """The tones a limited mask acts on, as "≥VI" or "≤III⅓"; empty when it acts on all."""
    if mask.key == MaskKey.OFF:
        return ""
    return ("≥" if mask.key == MaskKey.HIGHLIGHTS else "≤") + zone_roman(mask.key_zone)


@dataclass(frozen=True)
class MaskNote:
    number: int  # 1-based, matching the Dodge & Burn mask list
    is_burn: bool
    stops: str  # stops_label's "0" when the mask changes nothing but grade
    local_r: str = ""
    key: str = ""
    flash: str = ""

    @property
    def kind(self) -> str:
        if self.stops == "0":
            return tr(mark("Flash") if self.flash and not self.local_r else mark("Grade"))
        return tr(mark("Burn") if self.is_burn else mark("Dodge"))

    @property
    def badge(self) -> str:
        """Short form for the map drawn on the print."""
        parts = [str(self.number)]
        if self.stops != "0":
            parts.append(self.stops)
        if self.local_r:
            parts.append(self.local_r)
        if self.flash:
            parts.append(self.flash)
        if self.key:
            parts.append(self.key)
        return " ".join(parts)

    @property
    def summary(self) -> str:
        """Long form for the record card."""
        text = f"{self.number} {self.kind}"
        if self.stops != "0":
            text += f" {self.stops}"
        if self.local_r:
            text += f" @ {self.local_r}"
        if self.flash:
            text += f" {self.flash}"
        if self.key:
            text += " " + tr("on {key}", key=self.key)
        return text


def mask_notes(local: LocalAdjustmentsConfig, grade: float = 0.0) -> List[MaskNote]:
    """One note per enabled mask, numbered by its position in the mask list. `grade` is the
    frame's ISO R, which turns a mask's grade delta into the grade it prints at. A disabled
    mask burns nothing, so it carries no note. A limited mask past the fourth prints
    unlimited, so it carries no tone note."""
    limited = limited_indices(local)
    return [
        MaskNote(
            number=i + 1,
            is_burn=m.stops > 0,
            stops=stops_label(m.stops),
            local_r=local_grade_label(grade, m.grade),
            flash=tr("flash {value:.2f}", value=m.flash) if m.flash and i not in limited else "",
            key=tone_limit_label(m) if i in limited else "",
        )
        for i, m in enumerate(local.masks)
        if m.enabled
    ]


def recipe_lines(
    exposure: ExposureConfig, local: LocalAdjustmentsConfig, finish: FinishConfig, *, frame: str = "", brief: bool = False
) -> List[str]:
    """The printing record: one line per decision that is not at its default. brief keeps the
    three every print has: paper, Print Density and grade."""
    lines: List[str] = []
    if frame:
        lines.append(frame)

    paper = tr(resolve_paper(exposure.paper_profile).label)
    flags = [tr(name) for name, on in ((mark("Paper White"), exposure.paper_dmin), (mark("Paper Black"), exposure.paper_black)) if on]
    lines.append(" · ".join([paper, *flags]))

    density = tr("Print Density {value}", value=f"{exposure.density:.2f}")
    if exposure.auto_exposure:
        density += tr(" (auto)")
    lines.append(density)

    if not brief and (exposure.shadow_density or exposure.highlight_density):
        lines.append(
            tr(
                "Zone density: shadows {shadows} · highlights {highlights}",
                shadows=f"{exposure.shadow_density:+.2f}",
                highlights=f"{exposure.highlight_density:+.2f}",
            )
        )

    grade = tr("Grade ISO-R {value}", value=f"{exposure.grade:.0f}")
    if exposure.auto_normalize_contrast:
        grade += tr(" (auto)")
    if exposure.shadow_grade or exposure.highlight_grade:
        grade += " · " + tr(
            "split {shadows}/{highlights}", shadows=f"{exposure.shadow_grade:+.0f}", highlights=f"{exposure.highlight_grade:+.0f}"
        )
    lines.append(grade)
    if brief:
        return lines

    if exposure.wb_cyan or exposure.wb_magenta or exposure.wb_yellow:
        lines.append(
            tr("Filtration C{c} M{m} Y{y}", c=f"{exposure.wb_cyan:+.2f}", m=f"{exposure.wb_magenta:+.2f}", y=f"{exposure.wb_yellow:+.2f}")
        )

    if exposure.toe or exposure.shoulder:
        lines.append(tr("Toe {toe} · Shoulder {shoulder}", toe=f"{exposure.toe:+.2f}", shoulder=f"{exposure.shoulder:+.2f}"))

    if exposure.midtone_gamma:
        lines.append(tr("Snap {value}", value=f"{exposure.midtone_gamma:+.2f}"))

    if exposure.preflash:
        lines.append(tr("Preflash {value} of threshold", value=f"{exposure.preflash:.2f}"))

    if finish.vignette_stops:
        lines.append(tr("Edge burn {value} stop", value=stops_label(finish.vignette_stops)))

    notes = mask_notes(local, exposure.grade)
    if notes:
        lines.append(tr("Dodge & burn: {value}", value=" · ".join(n.summary for n in notes)))

    return lines
