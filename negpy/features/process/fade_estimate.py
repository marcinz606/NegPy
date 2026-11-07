"""Fade Estimate as the Fade Restoration card runs it: the survivals, white, Gray Mix, Restore Range
and crossover table one press or Gray Mix change writes, and the hints that go with them."""

from dataclasses import dataclass, field, replace
from typing import Any, Callable, Optional

from negpy.domain.types import ImageBuffer
from negpy.features.process.fade import (
    CROSSOVER_FLOOR,
    FadeEstimate,
    NeutralAxisRefs,
    blend_gray_readings,
    crossover_amount,
    crossover_space,
    estimate_fade_with,
    fade_density,
    gray_readings,
    pick_neutrals,
    render_white_space,
    stain_suggested,
    survival_crossover,
    tone_bands_only,
    toward_no_change,
)
from negpy.features.process.fade_auto_red import AUTO_RED_HINT_RATIO, auto_red, survival_points
from negpy.features.process.models import ProcessConfig
from negpy.kernel.system.i18n import tr


class GrayMemo:
    """The gray readings of the last Estimate grid. They depend on the grid and the anchor, never on
    Gray Mix, so a Gray Mix drag reads them once."""

    def __init__(self) -> None:
        self._memo: Optional[tuple[ImageBuffer, tuple[bool, str], Any]] = None

    def reader(
        self, grid: ImageBuffer, anchor: bool, space: str, mix: float
    ) -> Callable[[ImageBuffer], tuple[Optional[NeutralAxisRefs], str]]:
        """Estimate's gray reader on *grid* at Gray Mix *mix*."""
        key = (anchor, space)

        def read(gray_grid: ImageBuffer) -> tuple[Optional[NeutralAxisRefs], str]:
            if self._memo is None or self._memo[0] is not grid or self._memo[1] != key:
                self._memo = (grid, key, gray_readings(gray_grid))
            return blend_gray_readings(self._memo[2], mix)

        return read

    def readings(self, grid: ImageBuffer) -> Any:
        """The readings last taken on *grid*, None with none."""
        return self._memo[2] if self._memo is not None and self._memo[0] is grid else None

    def tone_bands_only(self, grid: ImageBuffer) -> bool:
        """Whether the last reading on *grid* found grays only in the tone bands."""
        readings = self.readings(grid)
        return readings is not None and tone_bands_only(readings)


@dataclass(frozen=True)
class EstimateOutcome:
    """What one Estimate writes and shows. `changes` go to the roll's Fade card and `fade_white` to the
    frame alone; `survivals` (red, green, blue) and `mix` are the sliders' new values, None to keep them.
    `red_hint` is None to leave the Restore Range hint as it is."""

    hint: str
    stain: str = ""
    changes: dict = field(default_factory=dict)
    fade_white: dict = field(default_factory=dict)
    survivals: Optional[tuple[float, float, float]] = None
    mix: Optional[float] = None
    tone_bands: bool = False
    red_hint: Optional[str] = None
    suggest_red: bool = False


def estimate_hint(chosen: FadeEstimate) -> str:
    """The result, or why there is none."""
    if chosen.reason and chosen.green == 1.0 and chosen.blue == 1.0:
        return chosen.reason
    red = chosen.red if chosen.red is not None else 1.0
    text = tr("Estimated: red {r}, green {g}, blue {b}.", r=f"{red:.2f}", g=f"{chosen.green:.2f}", b=f"{chosen.blue:.2f}")
    if chosen.reason:
        text = chosen.reason + " " + text
    return text


def auto_red_changes(conf: ProcessConfig, cam_xyz: Any, camera_wb: Any, grid: ImageBuffer) -> tuple[Optional[dict], str]:
    """`fade_auto_red.auto_red` for the survivals and shift in *conf*."""
    space = render_white_space(conf, conf.process_mode, cam_xyz, camera_wb)
    g, b, shift = conf.fade_ratio_g, conf.fade_ratio_b, conf.fade_density_shift

    def points_at(red: float):
        return survival_points(grid, conf, (red, g, b), conf.fade_white_anchor, bool(space), space, shift)

    return auto_red(points_at, conf.fade_ratio_r, shift)


def short_range(conf: ProcessConfig, cam_xyz: Any, camera_wb: Any, grid: ImageBuffer) -> bool:
    """Whether Restore Range would set Red Survival at AUTO_RED_HINT_RATIO of *conf*'s or less."""
    changes, _ = auto_red_changes(conf, cam_xyz, camera_wb, grid)
    return changes is not None and changes["fade_ratio_r"] <= conf.fade_ratio_r * AUTO_RED_HINT_RATIO


def measured_crossover(
    conf: ProcessConfig, new_conf: ProcessConfig, cam_xyz: Any, camera_wb: Any, grid: ImageBuffer, memo: GrayMemo, mix: float
) -> dict:
    """The crossover table and its tag measured after *new_conf*'s survivals, {} with none. *conf*
    gives the anchor and film mode the grays were read with."""
    space = render_white_space(new_conf, conf.process_mode, cam_xyz, camera_wb)
    anchor = conf.fade_white_anchor
    rgb = (new_conf.fade_ratio_r, new_conf.fade_ratio_g, new_conf.fade_ratio_b)
    table = survival_crossover(
        grid,
        memo.reader(grid, anchor, space, mix),
        rgb,
        white_anchor=anchor,
        camera_raw=bool(space),
        white_space=space,
        shift=new_conf.fade_density_shift,
    )
    if table is None:
        return {}
    tag = crossover_space(fade_density(new_conf, conf.process_mode, space), space, new_conf)
    return {"fade_crossover_table": table, "fade_crossover_space": tag}


def run_estimate(
    conf: ProcessConfig,
    grid: ImageBuffer,
    cam_xyz: Any,
    camera_wb: Any,
    memo: GrayMemo,
    *,
    white_anchor: bool,
    mix: Optional[float] = None,
    live: bool = False,
    keep_table: bool = False,
) -> EstimateOutcome:
    """Estimate on *grid* with Gray Mix *mix* (the stored one when None). *live* is a Gray Mix drag: no
    crossover table and no Restore Range hint. *keep_table* writes no table: only a press measures one."""
    mix_changes = {} if mix is None else {"fade_neutrals_mix": mix}
    mix = conf.fade_neutrals_mix if mix is None else mix
    anchor = white_anchor
    space = render_white_space(conf, conf.process_mode, cam_xyz, camera_wb)
    reader = memo.reader(grid, anchor, space, mix)
    # A press of Estimate picks Gray Mix and measures the tables then; a drag measures none.
    pick = not live and not mix_changes
    chosen = estimate_fade_with(
        grid, reader, white_anchor=anchor, camera_raw=bool(space), white_space=space, crossover=not live and not pick and not keep_table
    )
    # Where the gray finder finds no grays, Estimate reads the tone bands at full Gray Mix, and Gray
    # Mix moves from no change to that estimate.
    tone_bands = memo.tone_bands_only(grid)
    picked, slider_mix = None, None
    if pick and chosen.red is not None:
        readings = memo.readings(grid)
        found = None if tone_bands or readings is None else pick_neutrals(grid, readings, anchor, bool(space), space)
        if found is None:
            chosen = estimate_fade_with(grid, reader, white_anchor=anchor, camera_raw=bool(space), white_space=space)
        else:
            picked, chosen = found
            mix, mix_changes, slider_mix = picked, {"fade_neutrals_mix": picked}, picked
    if tone_bands:
        if not mix_changes:
            mix, mix_changes, slider_mix = 1.0, {"fade_neutrals_mix": 1.0}, 1.0
        chosen = toward_no_change(chosen, mix)
    failed = bool(chosen.reason) and chosen.green == 1.0 and chosen.blue == 1.0
    # A failed reading stores the white only where it is plainly a stain on clear film. A camera RAW's
    # render scales density above its white, so it stores the white with Remove Stain off too.
    white = {}
    if (anchor or space) and chosen.white is not None and (not failed or space or stain_suggested(chosen.white)):
        white = dict(zip(("fade_white_r", "fade_white_g", "fade_white_b"), chosen.white), fade_white_space=space)
    stain = ""
    if not anchor and chosen.white is not None and stain_suggested(chosen.white):
        stain = tr("The clear areas have a yellow stain: turn on Remove Stain.")
    hint = estimate_hint(chosen)
    if picked is not None:
        hint = tr("Gray Mix set to {n} %.", n=f"{picked * 100:.0f}") + " " + hint
    if tone_bands:
        hint = tr("This frame has few clear grays. Drag Gray Mix to set how much Estimate changes.") + " " + hint
    if anchor and not white and (conf.fade_white_r, conf.fade_white_g, conf.fade_white_b) == (0.0, 0.0, 0.0):
        hint += " " + tr("No stain found to remove.")
    if failed:
        frame_white = white if space and not live else {}
        return EstimateOutcome(hint, stain, {**({} if space else white), **mix_changes}, frame_white, mix=slider_mix, tone_bands=tone_bands)

    red = chosen.red if chosen.red is not None else conf.fade_ratio_r
    # Estimate's table is read with no shift, and a new red makes Restore Range's shift stale.
    survivals: dict[str, Any] = dict(
        fade_ratio_r=red, fade_ratio_g=chosen.green, fade_ratio_b=chosen.blue, fade_density_shift=0.0, **white, **mix_changes
    )
    # Restore Range keeps its Red: it runs again on the new grays, and the table is measured after it.
    short = not live and not conf.fade_auto_red and short_range(replace(conf, **survivals), cam_xyz, camera_wb, grid)
    auto = None
    if conf.fade_auto_red and live:
        # A Gray Mix drag keeps the fit, and the release fits Red again.
        survivals.update(fade_ratio_r=conf.fade_ratio_r, fade_density_shift=conf.fade_density_shift)
    elif conf.fade_auto_red:
        auto, _ = auto_red_changes(replace(conf, **survivals), cam_xyz, camera_wb, grid)
    if auto is not None:
        survivals.update(auto)
    crossover, measured = {}, {}
    if not keep_table and (auto is not None or (tone_bands and chosen.crossover is None)) and not live:
        measured = measured_crossover(conf, replace(conf, **survivals), cam_xyz, camera_wb, grid, memo, mix)
        survivals = {**survivals, **measured}
    if not keep_table and not measured and chosen.crossover is not None:
        new_conf = replace(conf, **survivals)
        tag = crossover_space(fade_density(new_conf, conf.process_mode, space), space, new_conf)
        crossover = dict(fade_crossover_table=chosen.crossover, fade_crossover_space=tag)
    red_hint, suggest_red = None, False
    if not live:
        table = crossover.get("fade_crossover_table", survivals.get("fade_crossover_table"))
        if conf.fade_crossover == 0.0 and isinstance(table, tuple) and crossover_amount(table) >= CROSSOVER_FLOOR:
            hint += " " + tr("The colors shift from shadows to highlights: raise Crossover.")
        suggest_red = short and not conf.fade_auto_red
        red_hint = tr("This frame has a short range: turn on Restore Range.") if suggest_red else ""
    return EstimateOutcome(
        hint,
        stain,
        {**survivals, **crossover},
        survivals=(survivals["fade_ratio_r"], chosen.green, chosen.blue),
        mix=slider_mix,
        tone_bands=tone_bands,
        red_hint=red_hint,
        suggest_red=suggest_red,
    )
