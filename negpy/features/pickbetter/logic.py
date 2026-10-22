import random
from dataclasses import replace

from negpy.domain.models import WorkspaceConfig
from negpy.features.exposure.logic import kelvin_to_wb, wb_to_kelvin
from negpy.kernel.system.i18n import tr
from negpy.features.pickbetter.models import PICK_PARAMS, PickBetterProgress, PickParam, Phase


def effective_order(progress: PickBetterProgress) -> tuple[int, ...]:
    """The order the rounds actually visit: the shuffled randomize_order when randomize is on,
    else the drag-and-drop route param_order. Both are permutations of the eleven parameters, so
    consecutive rounds (param_index + 1, wrapping) always name a different parameter.
    """
    return progress.randomize_order if progress.randomize else progress.param_order


def shuffle_order() -> tuple[int, ...]:
    """A fresh full shuffle of the eleven parameters, used to seed randomize_order."""
    order = list(range(len(PICK_PARAMS)))
    random.shuffle(order)
    return tuple(order)


def _clamp(value: float, param: PickParam) -> float:
    """Clamp a candidate to the field's slider range."""
    return max(param.lo, min(param.hi, value))


def param_value(config: WorkspaceConfig, param: PickParam) -> float:
    """Read a control's current value: the field itself, or Kelvin for the compound Temperature"""
    if param.compound:
        return wb_to_kelvin(config.exposure.wb_magenta, config.exposure.wb_yellow)
    return float(getattr(config.exposure, param.field))


def set_param(config: WorkspaceConfig, param: PickParam, value: float) -> WorkspaceConfig:
    """Return a config with the control set to `value`, clamped to its slider range.

    A single-field control writes its field; the compound Temperature moves wb_magenta/
    wb_yellow along the Planckian direction to land on the Kelvin value.
    """
    value = _clamp(value, param)
    if param.compound:
        m2, y2 = kelvin_to_wb(value, config.exposure.wb_magenta, config.exposure.wb_yellow)
        return replace(config, exposure=replace(config.exposure, wb_magenta=m2, wb_yellow=y2))
    return replace(config, exposure=replace(config.exposure, **{param.field: value}))


def step_for(param: PickParam, selections: int) -> float:
    """The step a round proposes: bold at the parameter's first selection, ×2/3 per later
    selection of the same parameter, never below the floor.
    """
    return max(param.strong_step * (2 / 3) ** selections, param.floor)


#: The initial route: Grade and Density first, then Temperature and Dye Separation, then the
#: remaining parameters in PICK_PARAMS order.
DEFAULT_ROUTE: tuple[int, ...] = (0, 1, 10, 7, 2, 3, 4, 5, 6, 8, 9)


def start_pick_better(config: WorkspaceConfig) -> PickBetterProgress:
    """The first round: position 0 at its full (bold) step, round 1 of the fixed phase.

    The route is DEFAULT_ROUTE; the user reorders it (or turns on randomize) later via the
    Custom Routing window, which sets param_order and randomize_order without touching this start.
    """
    return PickBetterProgress(
        phase=Phase.FIXED,
        param_index=0,
        round_number=1,
        selection_counts=tuple(0 for _ in PICK_PARAMS),
        param_order=DEFAULT_ROUTE,
        randomize=False,
        randomize_order=tuple(range(len(PICK_PARAMS))),
    )


def current_pair(
    config: WorkspaceConfig,
    progress: PickBetterProgress,
    scale: float = 1.0,
) -> tuple[WorkspaceConfig, WorkspaceConfig]:
    """The two candidates to render, differing in one ExposureConfig field.

    A round is center +/- step (left stronger, right weaker), the step bold at the parameter's
    first selection and smaller on each later one. `scale` multiplies the step so a Difference
    slider widens or narrows the A/B spread without changing the committed center.
    """
    real = effective_order(progress)[progress.param_index]
    param = PICK_PARAMS[real]
    current = param_value(config, param)
    step = step_for(param, progress.selection_counts[real]) * scale
    left_val = _clamp(current + step, param)
    right_val = _clamp(current - step, param)
    # At a bound a candidate clamps back onto current, so a pick of it is a no-op that
    # shows only the parameter name. Nudge it onto the open side so a pick always moves.
    if left_val == current:
        left_val = _clamp(current - step / 2, param)
    if right_val == current:
        right_val = _clamp(current + step / 2, param)
    left = set_param(config, param, left_val)
    right = set_param(config, param, right_val)
    return left, right


def apply_pick(
    config: WorkspaceConfig,
    progress: PickBetterProgress,
    side: int,
    scale: float = 1.0,
) -> tuple[WorkspaceConfig, PickBetterProgress]:
    """Commit the chosen half (side 0 = left, 1 = right) and advance to the next round.

    A pick counts as one selection of the in-flight parameter, so its next visit proposes a
    smaller step.
    """
    left, right = current_pair(config, progress, scale=scale)
    new_config = left if side == 0 else right
    real = effective_order(progress)[progress.param_index]
    counts = progress.selection_counts
    counts = counts[:real] + (counts[real] + 1,) + counts[real + 1 :]
    return new_config, _advance(progress, counts)


def reject_and_advance(
    config: WorkspaceConfig,
    progress: PickBetterProgress,
) -> tuple[WorkspaceConfig, PickBetterProgress]:
    """Reject both candidates (keep the middle current photo) and advance to the next round.

    A reject is not a selection: the config and the selection counts stay put, so the next
    parameter's first visit still opens at its bold step. Both phases move to the next
    parameter; the fixed phase begins the refinement after its eleventh round.
    """
    if progress.phase == Phase.FIXED:
        next_index = progress.param_index + 1
        if next_index < len(PICK_PARAMS):
            return config, _carry(progress, counts=progress.selection_counts, phase=Phase.FIXED, param_index=next_index)
        return config, _carry(progress, counts=progress.selection_counts, phase=Phase.REFINE, param_index=0)
    next_index = (progress.param_index + 1) % len(PICK_PARAMS)
    return config, _carry(progress, counts=progress.selection_counts, phase=Phase.REFINE, param_index=next_index)


def _advance(
    progress: PickBetterProgress,
    counts: tuple[int, ...],
) -> PickBetterProgress:
    """Advance to the next round after a pick. The fixed phase moves one parameter per round
    (bold at first, then the next parameter); after the eleventh it begins the refinement. The
    refinement runs the parameters in round-robin, each visit 2/3 of the prior step.
    """
    if progress.phase == Phase.FIXED:
        next_index = progress.param_index + 1
        if next_index < len(PICK_PARAMS):
            return _carry(progress, counts=counts, phase=Phase.FIXED, param_index=next_index)
        return _carry(progress, counts=counts, phase=Phase.REFINE, param_index=0)
    return _advance_refine(progress, counts)


def _advance_refine(
    progress: PickBetterProgress,
    counts: tuple[int, ...],
) -> PickBetterProgress:
    """The refinement runs the parameters in round-robin; each visit 2/3 of the prior step.

    Every round visits a different parameter, so two consecutive rounds never share one.
    """
    next_index = (progress.param_index + 1) % len(PICK_PARAMS)
    return _carry(progress, counts=counts, phase=Phase.REFINE, param_index=next_index)


def _carry(
    progress: PickBetterProgress,
    *,
    counts: tuple[int, ...],
    phase: Phase,
    param_index: int,
) -> PickBetterProgress:
    """Advance the cursor (phase, position, round number) and carry the route and counts forward."""
    return PickBetterProgress(
        phase=phase,
        param_index=param_index,
        round_number=progress.round_number + 1,
        selection_counts=counts,
        param_order=progress.param_order,
        randomize=progress.randomize,
        randomize_order=progress.randomize_order,
    )


def is_finished(progress: PickBetterProgress) -> bool:
    """The refinement is endless; the user ends via the Finish button or the toggle, not here."""
    return False


def move_summary(old: WorkspaceConfig, new: WorkspaceConfig, progress: PickBetterProgress) -> str:
    """The "Name +delta" line shown for a few seconds after a left or right pick.

    The delta is the committed value minus the pre-pick value (a clamp shows the actual
    movement); a pick that keeps the value shows the name alone, with no delta.
    """
    real = effective_order(progress)[progress.param_index]
    param = PICK_PARAMS[real]
    delta = param_value(new, param) - param_value(old, param)
    if abs(delta) < 1e-9:
        return tr(param.name)
    return f"{tr(param.name)} {delta:+g}"
