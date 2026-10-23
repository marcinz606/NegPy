"""Detect Scenes: groups of a roll's frames under one light, from their near-gray axes."""

from negpy.features.exposure.normalization import detect_light_groups


def _frame(dr: float, db: float, conf: float = 0.9) -> tuple:
    """A frame on one film's R/B-vs-G axis, its light shifting the grays by (dr, db) log D."""
    band = lambda g: (g + 0.1 + 0.05 * g * g + dr, g, g - 0.1 - 0.03 * g * g + db)  # noqa: E731
    return (band(-0.8), band(-0.4), band(-1.4), conf)


def test_frames_under_another_light_are_one_group():
    axes = [_frame(0.0, 0.0)] * 5 + [_frame(0.2, -0.1)] * 3

    assert detect_light_groups(axes) == [[5, 6, 7]]


def test_one_light_finds_nothing():
    assert detect_light_groups([_frame(0.01 * i, 0.0) for i in range(6)]) == []


def test_the_largest_light_stays_with_the_roll_and_singles_stay_alone():
    axes = [_frame(0.0, 0.0)] * 4 + [_frame(0.3, 0.0)] * 2 + [_frame(-0.3, 0.2)] * 3 + [_frame(0.0, 0.4)]

    assert detect_light_groups(axes) == [[6, 7, 8], [4, 5]]


def test_frames_without_trustworthy_grays_stay_out():
    axes = [_frame(0.0, 0.0)] * 4 + [_frame(0.2, -0.1), None, _frame(0.2, -0.1, conf=0.1), _frame(0.2, -0.1)]

    assert detect_light_groups(axes) == [[4, 7]]


def test_a_frame_off_the_scenes_center_stays_out():
    """Within the distance of every member, but its own light is off the scene's center."""
    axes = [_frame(0.0, 0.0)] * 6 + [_frame(0.2, -0.1)] * 4 + [_frame(0.265, -0.1)]

    assert detect_light_groups(axes) == [[6, 7, 8, 9]]


def test_in_scan_order_a_drifting_light_stays_one_scene():
    """Three frames whose light drifts past the distance end to end: neighbours in order hold them together."""
    run = [_frame(0.15, 0.0), _frame(0.19, 0.0), _frame(0.235, 0.0)]
    in_order = [_frame(0.0, 0.0)] * 6 + run

    assert detect_light_groups(in_order) == [[6, 7, 8]]


def test_frames_out_of_scan_order_get_no_help_from_it():
    """The same frames, each separated by another light: neighbours agree no more than any pair."""
    main = _frame(0.0, 0.0)
    mixed = [_frame(0.15, 0.0), main, _frame(0.19, 0.0), main, _frame(0.235, 0.0), main, main, main, main]

    assert detect_light_groups(mixed) == [[0, 2]]
