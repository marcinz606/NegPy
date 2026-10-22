from dataclasses import replace

import pytest

from negpy.domain.models import WorkspaceConfig
from negpy.features.pickbetter.logic import (
    DEFAULT_ROUTE,
    apply_pick,
    current_pair,
    effective_order,
    is_finished,
    move_summary,
    param_value,
    reject_and_advance,
    set_param,
    shuffle_order,
    start_pick_better,
    step_for,
)
from negpy.features.pickbetter.models import Phase, PICK_PARAMS, PickBetterProgress


def _config(**exposure_fields) -> WorkspaceConfig:
    return replace(WorkspaceConfig(), exposure=replace(WorkspaceConfig().exposure, **exposure_fields))


def _zeros() -> tuple[int, ...]:
    return tuple(0 for _ in PICK_PARAMS)


def _count_at(index: int, value: int) -> tuple[int, ...]:
    counts = [0] * len(PICK_PARAMS)
    counts[index] = value
    return tuple(counts)


class TestStart:
    def test_first_round_is_grade_strong_weak(self):
        progress = start_pick_better(WorkspaceConfig())
        assert progress.phase == Phase.FIXED
        assert progress.param_index == 0
        # round_number is 1-based and counts the rounds run; the first round is 1.
        assert progress.round_number == 1
        # First selection of a parameter is bold: the step is its full strong_step.
        assert step_for(PICK_PARAMS[0], progress.selection_counts[0]) == PICK_PARAMS[0].strong_step
        # The route is DEFAULT_ROUTE; randomize is off and its shuffled copy is the identity too.
        assert progress.param_order == DEFAULT_ROUTE
        assert progress.randomize is False
        assert progress.randomize_order == tuple(range(len(PICK_PARAMS)))


class TestFixedSequence:
    def test_first_pair_is_center_plus_minus_step(self):
        progress = start_pick_better(WorkspaceConfig())
        left, right = current_pair(WorkspaceConfig(), progress)
        assert left.exposure.grade == 135.0  # 115 + 20
        assert right.exposure.grade == 95.0  # 115 - 20

    def test_strong_weak_advances_to_next_param(self):
        progress = start_pick_better(WorkspaceConfig())
        new_config, next_progress = apply_pick(WorkspaceConfig(), progress, side=0)
        assert new_config.exposure.grade == 135.0  # left committed
        # A fixed pick moves to the next parameter, bumps the round count, and counts one selection
        # of the in-flight parameter.
        assert next_progress.phase == Phase.FIXED
        assert next_progress.param_index == 1
        assert next_progress.round_number == 2
        assert next_progress.selection_counts[0] == 1

    def test_full_fixed_sequence_visits_each_param_once(self):
        # The fixed phase runs one round per parameter, in route order; after the eleventh it opens
        # the refinement back at the first parameter.
        config = WorkspaceConfig()
        progress = start_pick_better(config)
        seen = []
        for _ in range(len(PICK_PARAMS)):
            seen.append(effective_order(progress)[progress.param_index])
            config, progress = apply_pick(config, progress, side=0)
        assert sorted(seen) == list(range(len(PICK_PARAMS)))
        assert progress.phase == Phase.REFINE
        assert progress.param_index == 0
        assert progress.round_number == len(PICK_PARAMS) + 1

    def test_no_consecutive_same_param_in_fixed(self):
        # Two consecutive fixed rounds name different parameters: the route is a permutation and the
        # cursor advances one each round.
        config = WorkspaceConfig()
        progress = start_pick_better(config)
        prev = effective_order(progress)[progress.param_index]
        for _ in range(len(PICK_PARAMS)):
            config, progress = apply_pick(config, progress, side=0)
            real = effective_order(progress)[progress.param_index]
            assert real != prev
            prev = real


class TestClamping:
    def test_candidate_clamps_to_hi(self):
        # grade 175 + 20 = 195 clamps to hi 180; 175 - 20 = 155 stays in range.
        config = _config(grade=175.0)
        progress = start_pick_better(config)
        left, right = current_pair(config, progress)
        assert left.exposure.grade == 180.0
        assert right.exposure.grade == 155.0

    def test_candidate_clamps_to_lo(self):
        # highlight_density -0.4 - 0.2 = -0.6 clamps to lo -0.5.
        config = _config(highlight_density=-0.4)
        progress = PickBetterProgress(Phase.FIXED, 4, 5, _zeros())
        left, right = current_pair(config, progress)
        assert left.exposure.highlight_density == -0.2
        assert right.exposure.highlight_density == -0.5

    def test_dye_separation_clamps_to_hi(self):
        # dye_separation 1.45 + 0.1 = 1.55 clamps to hi 1.5; 1.45 - 0.1 = 1.35 stays in range.
        config = _config(dye_separation=1.45)
        progress = PickBetterProgress(Phase.FIXED, 7, 8, _zeros())
        left, right = current_pair(config, progress)
        assert left.exposure.dye_separation == 1.5
        assert right.exposure.dye_separation == pytest.approx(1.35)

    def test_preflash_clamps_to_lo(self):
        # preflash 0.05 - 0.1 = -0.05 clamps to lo 0.0; 0.05 + 0.1 = 0.15 stays in range.
        config = _config(preflash=0.05)
        progress = PickBetterProgress(Phase.FIXED, 6, 7, _zeros())
        left, right = current_pair(config, progress)
        assert left.exposure.preflash == pytest.approx(0.15)
        assert right.exposure.preflash == 0.0

    def test_candidate_at_hi_moves_on_the_open_side(self):
        # grade 180 is at the hi bound: 180 + 20 clamps back onto 180 (a no-op pick), so the
        # left candidate is nudged onto the open side (180 - 10 = 170); the right stays 160.
        config = _config(grade=180.0)
        progress = start_pick_better(config)
        left, right = current_pair(config, progress)
        assert left.exposure.grade == 170.0  # nudged below the hi bound so the pick moves
        assert right.exposure.grade == 160.0
        assert left.exposure.grade != right.exposure.grade

    def test_candidate_at_lo_moves_on_the_open_side(self):
        # grade 50 is at the lo bound: 50 - 20 clamps back onto 50 (a no-op pick), so the
        # right candidate is nudged onto the open side (50 + 10 = 60); the left stays 70.
        config = _config(grade=50.0)
        progress = start_pick_better(config)
        left, right = current_pair(config, progress)
        assert left.exposure.grade == 70.0
        assert right.exposure.grade == 60.0  # nudged above the lo bound so the pick moves
        assert left.exposure.grade != right.exposure.grade


class TestCompoundTemperature:
    # Temperature is a compound control: its value is Kelvin, read and written through the
    # wb_magenta/wb_yellow pair, not an ExposureConfig field of its own.
    def test_param_value_reads_kelvin_from_wb(self):
        # Neutral wb (magenta=0, yellow=0) reads back as the 5500K reference.
        assert param_value(WorkspaceConfig(), PICK_PARAMS[10]) == 5500.0

    def test_set_param_moves_the_wb_pair_to_kelvin(self):
        param = PICK_PARAMS[10]
        config = WorkspaceConfig()
        new_config = set_param(config, param, 6000.0)
        # The wb pair moved off neutral, the read-back lands on the target Kelvin, and the
        # original frozen config is untouched.
        assert new_config.exposure.wb_magenta != 0.0
        assert new_config.exposure.wb_yellow != 0.0
        assert param_value(new_config, param) == pytest.approx(6000.0)
        assert config.exposure.wb_magenta == 0.0
        assert config.exposure.wb_yellow == 0.0

    def test_temperature_clamps_to_kelvin_range(self):
        param = PICK_PARAMS[10]
        config = WorkspaceConfig()
        assert param_value(set_param(config, param, 2000.0), param) == pytest.approx(3000.0)
        assert param_value(set_param(config, param, 20000.0), param) == pytest.approx(12000.0)


class TestStepShrinksWithSelections:
    # A step is bold on a parameter's first selection and ×2/3 on each later selection of the
    # same parameter, never below the floor.
    def test_grade_step_reduces_to_two_thirds_per_selection(self):
        # grade strong_step 20, floor 1: ×2/3 each selection; the floor (1.0) first holds at selection 8.
        param = PICK_PARAMS[0]
        assert step_for(param, 0) == 20.0
        assert step_for(param, 1) == pytest.approx(20 * 2 / 3)
        assert step_for(param, 2) == pytest.approx(20 * (2 / 3) ** 2)
        assert step_for(param, 3) == pytest.approx(20 * (2 / 3) ** 3)
        assert step_for(param, 4) == pytest.approx(20 * (2 / 3) ** 4)
        assert step_for(param, 8) == 1.0  # 20*(2/3)^8 is below the floor, so 1.0 holds
        assert step_for(param, 9) == 1.0  # the floor holds for any larger count

    def test_second_density_selection_is_smaller_than_the_first(self):
        # After one density selection, the next density proposal is bolder-then-smaller.
        param = PICK_PARAMS[1]  # density
        assert step_for(param, 0) > step_for(param, 1)
        assert step_for(param, 0) == 0.3
        assert step_for(param, 1) == pytest.approx(0.2)


class TestRefinement:
    # The refinement runs the eleven parameters in round-robin, one per round, wrapping after the
    # last; each visit 2/3 of the prior step, keyed by that parameter's own selection count.
    def test_refine_visits_params_in_round_robin(self):
        # From param_index 0, the round-robin visits 0..10 then wraps back to 0.
        config = WorkspaceConfig()
        progress = PickBetterProgress(Phase.REFINE, 0, 12, _zeros())
        seen = []
        for _ in range(13):
            seen.append(effective_order(progress)[progress.param_index])
            config, progress = apply_pick(config, progress, side=0)
        assert seen == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 1]
        assert progress.phase == Phase.REFINE

    def test_refine_step_reduces_to_two_thirds_per_visit(self):
        # A parameter's step is its bold step on the first visit and ×2/3 on the second, since the
        # round-robin revisits it once every full cycle and the count grows once per visit.
        config = WorkspaceConfig()
        progress = PickBetterProgress(Phase.REFINE, 0, 12, _zeros())
        steps = []
        for _ in range(12):
            real = effective_order(progress)[progress.param_index]
            steps.append(step_for(PICK_PARAMS[real], progress.selection_counts[real]))
            config, progress = apply_pick(config, progress, side=0)
        assert steps[0] == PICK_PARAMS[0].strong_step  # first visit of grade
        assert steps[11] == pytest.approx(PICK_PARAMS[0].strong_step * 2 / 3)  # second visit of grade

    def test_refine_wraps_to_zero(self):
        config = WorkspaceConfig()
        progress = PickBetterProgress(Phase.REFINE, 10, 12, _zeros())
        _, next_progress = apply_pick(config, progress, side=0)
        assert next_progress.phase == Phase.REFINE
        assert next_progress.param_index == 0

    def test_no_consecutive_same_param_in_refine(self):
        # Two consecutive refine rounds never share a parameter: the round-robin moves one each time.
        config = WorkspaceConfig()
        progress = PickBetterProgress(Phase.REFINE, 0, 12, _zeros())
        prev = effective_order(progress)[progress.param_index]
        for _ in range(20):
            config, progress = apply_pick(config, progress, side=0)
            real = effective_order(progress)[progress.param_index]
            assert real != prev
            prev = real


class TestReject:
    # A click on the centre current photo rejects both candidates: the config stays put, the
    # selection counts do not increment, and the cursor advances to the next parameter's round,
    # uniform across the fixed and refine phases.
    def test_config_unchanged_and_counts_do_not_increment(self):
        # Rejecting the grade round keeps the current grade and leaves the count at 0.
        config = _config(grade=135.0)
        progress = PickBetterProgress(Phase.FIXED, 0, 1, _zeros())
        new_config, next_progress = reject_and_advance(config, progress)
        assert new_config.exposure.grade == 135.0
        assert next_progress.selection_counts == _zeros()
        assert next_progress.selection_counts[0] == 0

    def test_fixed_advances_to_next_param(self):
        config = WorkspaceConfig()
        progress = start_pick_better(config)
        _, next_progress = reject_and_advance(config, progress)
        assert next_progress.phase == Phase.FIXED
        assert next_progress.param_index == 1
        assert next_progress.round_number == 2
        # The next param opens at its bold step, since the reject added no selection.
        assert step_for(PICK_PARAMS[1], next_progress.selection_counts[1]) == PICK_PARAMS[1].strong_step

    def test_fixed_last_param_enters_refine(self):
        # Rejecting the last fixed param (temperature, index 10) opens the refine at param 0.
        config = WorkspaceConfig()
        progress = PickBetterProgress(Phase.FIXED, 10, 11, _zeros())
        _, next_progress = reject_and_advance(config, progress)
        assert next_progress.phase == Phase.REFINE
        assert next_progress.param_index == 0

    def test_refine_advances_to_next_param_wrapping(self):
        # In the refine, a reject moves to the next parameter; at the last param it wraps to 0.
        config = WorkspaceConfig()
        progress = PickBetterProgress(Phase.REFINE, 9, 12, _zeros())
        _, next_progress = reject_and_advance(config, progress)
        assert next_progress.phase == Phase.REFINE
        assert next_progress.param_index == 10
        progress_last = PickBetterProgress(Phase.REFINE, 10, 12, _zeros())
        _, wrapped = reject_and_advance(config, progress_last)
        assert wrapped.param_index == 0

    def test_does_not_mutate_inputs(self):
        config = _config(grade=135.0)
        progress = PickBetterProgress(Phase.FIXED, 0, 1, _count_at(0, 2))
        reject_and_advance(config, progress)
        # The frozen config and progress are returned unchanged; only the new progress differs.
        assert config.exposure.grade == 135.0
        assert progress.selection_counts == _count_at(0, 2)

    def test_fixed_to_refine_boundary_is_no_consecutive(self):
        # The last fixed round (param_index 10) hands off to the first refine round (param_index 0);
        # both read distinct entries of the route, so they name different parameters.
        progress = PickBetterProgress(Phase.FIXED, 10, 11, _zeros())
        last_fixed = effective_order(progress)[progress.param_index]
        refine = PickBetterProgress(Phase.REFINE, 0, 12, _zeros())
        first_refine = effective_order(refine)[refine.param_index]
        assert last_fixed != first_refine


class TestUndo:
    # The controller's Undo restores the pre-pick (config, progress) and re-renders; at the logic
    # level that is current_pair being a pure function of (config, progress) and apply_pick never
    # mutating its frozen inputs, so re-rendering the snapshot reproduces the exact pair.
    def test_undo_reproposes_previous_pair(self):
        config = WorkspaceConfig()
        progress = start_pick_better(config)
        # Render the pair, commit the left (grade 135), then restore the snapshot and re-render.
        left1, right1 = current_pair(config, progress)
        apply_pick(config, progress, side=0)
        left2, right2 = current_pair(config, progress)
        assert left1.exposure.grade == left2.exposure.grade == 135.0
        assert right1.exposure.grade == right2.exposure.grade == 95.0

    def test_apply_pick_does_not_mutate_inputs(self):
        config = _config(grade=135.0)
        progress = PickBetterProgress(Phase.FIXED, 0, 1, _zeros())
        apply_pick(config, progress, side=0)
        # The frozen config and progress are returned unchanged; only the new pair differs.
        assert config.exposure.grade == 135.0
        assert progress.selection_counts == _zeros()


class TestFinished:
    def test_is_finished_is_always_false(self):
        assert is_finished(start_pick_better(WorkspaceConfig())) is False
        assert is_finished(PickBetterProgress(Phase.REFINE, 5, 12, _zeros())) is False


class TestDifferenceScale:
    # The Difference slider multiplies the A/B step; the two candidates stay symmetric about the
    # current (Skip) value, so a larger spread moves them further out without moving the center.
    def test_scale_doubles_the_spread(self):
        # grade 115, bold step 20: at scale 2.0 the step is 40, so A is 155 and B is 75.
        progress = start_pick_better(WorkspaceConfig())
        left, right = current_pair(WorkspaceConfig(), progress, scale=2.0)
        assert left.exposure.grade == 155.0
        assert right.exposure.grade == 75.0

    def test_scale_keeps_the_pair_symmetric_about_current(self):
        # A bigger spread moves both candidates out but the midpoint stays the current value.
        config = _config(grade=115.0)
        progress = start_pick_better(config)
        for scale in (0.5, 1.0, 2.0, 4.0):
            left, right = current_pair(config, progress, scale=scale)
            assert (left.exposure.grade + right.exposure.grade) / 2 == pytest.approx(115.0)

    def test_apply_pick_commits_the_scaled_candidate(self):
        # A left pick at scale 2.0 commits the further-out value (155), not the unit-scale 135.
        config = WorkspaceConfig()
        progress = start_pick_better(config)
        new_config, _ = apply_pick(config, progress, side=0, scale=2.0)
        assert new_config.exposure.grade == 155.0


class TestRandomize:
    # randomize_order is the shuffled copy used when randomize is on; param_order is the
    # drag-and-drop route used when randomize is off. effective_order picks between them, so
    # turning randomize off restores the drag-and-drop route, and a fresh shuffle seeds a new copy.
    def test_effective_order_uses_param_order_when_randomize_off(self):
        progress = start_pick_better(WorkspaceConfig())
        assert effective_order(progress) == progress.param_order

    def test_effective_order_uses_randomize_order_when_randomize_on(self):
        order = shuffle_order()
        progress = PickBetterProgress(Phase.FIXED, 0, 1, _zeros(), randomize=True, randomize_order=order)
        assert effective_order(progress) == order

    def test_turning_randomize_off_restores_the_route(self):
        # The randomize flag alone decides which order effective_order reads; clearing it keeps the
        # shuffled copy but reads the drag-and-drop route again.
        order = shuffle_order()
        progress = PickBetterProgress(Phase.FIXED, 0, 1, _zeros(), randomize=True, randomize_order=order)
        restored = replace(progress, randomize=False)
        assert effective_order(restored) == progress.param_order
        assert restored.randomize_order == order

    def test_shuffle_order_is_a_full_permutation(self):
        order = shuffle_order()
        assert len(order) == len(PICK_PARAMS)
        assert sorted(order) == list(range(len(PICK_PARAMS)))

    def test_shuffled_fixed_run_visits_all_params(self):
        # 11 fixed rounds over a shuffled route still cover every one of the eleven parameters once.
        config = WorkspaceConfig()
        order = shuffle_order()
        progress = PickBetterProgress(Phase.FIXED, 0, 1, _zeros(), randomize=True, randomize_order=order)
        seen = []
        for _ in range(len(PICK_PARAMS)):
            seen.append(effective_order(progress)[progress.param_index])
            config, progress = apply_pick(config, progress, side=0)
        assert sorted(seen) == list(range(len(PICK_PARAMS)))

    def test_selection_count_tracks_the_real_param(self):
        # After a pick at a shuffled position, the count increments on that position's real parameter,
        # not on the position index.
        config = WorkspaceConfig()
        order = shuffle_order()
        progress = PickBetterProgress(Phase.FIXED, 0, 1, _zeros(), randomize=True, randomize_order=order)
        real = effective_order(progress)[progress.param_index]
        _, next_progress = apply_pick(config, progress, side=0)
        assert next_progress.selection_counts[real] == 1
        assert sum(next_progress.selection_counts) == 1


class TestMoveSummary:
    # move_summary renders the "Name +delta" line shown briefly after a left or right pick, reading
    # the in-flight parameter through the active order so a shuffled round still names the right control.
    def test_grade_move_shows_name_and_delta(self):
        # grade 115 -> 135 is a +20 move; the line names the control and its signed delta.
        old = WorkspaceConfig()
        new = _config(grade=135.0)
        progress = start_pick_better(old)
        assert move_summary(old, new, progress) == "ISO-R Grade +20"

    def test_zero_delta_shows_name_only(self):
        # A pick that keeps the value (delta 0) shows the name alone, with no delta.
        config = _config(grade=135.0)
        progress = start_pick_better(config)
        assert move_summary(config, config, progress) == "ISO-R Grade"

    def test_names_the_real_param_at_the_position(self):
        # With param_order putting density (index 1) at position 0, the summary reads the real
        # parameter, not the position index (grade, index 0).
        old = WorkspaceConfig()
        new = _config(density=1.3)
        progress = PickBetterProgress(
            Phase.FIXED,
            0,
            1,
            _zeros(),
            param_order=(1, 0, 2, 3, 4, 5, 6, 7, 8, 9, 10),
        )
        assert move_summary(old, new, progress) == "Print Density +0.3"
