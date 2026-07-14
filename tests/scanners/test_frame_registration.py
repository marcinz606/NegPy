"""Pure-array tests for roll frame registration."""

import numpy as np
import pytest

from negpy.infrastructure.scanners.frame_registration import (
    TailObservation,
    fit_transport_tail,
    learn_film_base,
    measure_leading_margin,
    measure_transport_tails,
    measure_target_start,
    observe_transport_tail,
    robust_pitch_prediction,
    shifted_reference_prediction,
)


def _preview(
    *,
    edge: int,
    base_rgb: tuple[int, int, int],
    scene_rgb: tuple[int, int, int],
    height: int = 72,
    width: int = 32,
    seed: int = 0,
) -> np.ndarray:
    """Make a small wide preview with a uniform gap followed by a scene."""
    rng = np.random.default_rng(seed)
    image = np.empty((height, width, 3), dtype=np.float64)
    image[:edge] = np.asarray(base_rgb) + rng.normal(0.0, 8.0, (edge, width, 3))
    y, x = np.mgrid[: height - edge, :width]
    texture = 700.0 * np.sin(x / 2.7) + 450.0 * np.cos(y / 3.1)
    image[edge:] = np.asarray(scene_rgb) + texture[..., None] + rng.normal(0.0, 35.0, (height - edge, width, 3))
    return np.clip(image, 0, 65535).astype(np.uint16)


def _tail_preview(*, tail_start: int = 40, height: int = 52, width: int = 24) -> np.ndarray:
    y, x = np.mgrid[:height, :width]
    image = np.empty((height, width, 3), dtype=np.float64)
    texture = 900.0 * np.sin(x / 2.3)
    for row in range(tail_start):
        image[row] = 47000.0 + texture[row, :, None] + row * 17.0
    burst_row = tail_start - 5
    image[burst_row + 1 : tail_start] -= 5000.0
    repeated = 36000.0 + texture[tail_start, :, None]
    image[tail_start:] = repeated
    return np.clip(image, 0, 65535).astype(np.uint16)


def test_learns_roll_specific_film_base_without_a_fixed_rgb_value() -> None:
    base = (41200, 17600, 9300)
    previews = [
        _preview(edge=edge, base_rgb=base, scene_rgb=scene, seed=index)
        for index, (edge, scene) in enumerate(((13, (8000, 7000, 5000)), (19, (58000, 55000, 50000)), (9, (24000, 20000, 15000))))
    ]

    model = learn_film_base(previews)

    np.testing.assert_allclose(model.rgb, base, atol=25)


def test_film_base_tie_prefers_high_transmission_over_a_repeated_smooth_scene() -> None:
    """C-41 base is the least-dense recurring run, not the first tied colour."""

    rng = np.random.default_rng(20260712)
    base = np.asarray((59000, 22800, 14500), dtype=np.float64)
    false_scene = np.asarray((24000, 8500, 6500), dtype=np.float64)
    previews = []
    for index, edge in enumerate((90, 84, 100)):
        image = np.empty((130, 32, 3), dtype=np.float64)
        image[:40] = false_scene + rng.normal(0, 5, size=(40, 32, 3))
        image[40 : edge - 20] = np.asarray((9000, 5000, 3000)) + rng.normal(
            0,
            500,
            size=(edge - 60, 32, 3),
        )
        image[edge - 20 : edge] = base
        y, x = np.mgrid[: 130 - edge, :32]
        texture = 900 * np.sin(x / 2.3) + 600 * np.cos(y / 3.1)
        image[edge:] = np.asarray((13000 + index * 5000, 7000, 4000)) + texture[..., None]
        previews.append(np.clip(image, 0, 65535).astype(np.uint16))

    model = learn_film_base(previews)

    np.testing.assert_allclose(model.rgb, base, atol=40)
    assert [measure_target_start(preview, model).row for preview in previews] == [90, 84, 100]


def test_finds_the_gap_edge_for_both_bright_and_dark_frames() -> None:
    base = (41200, 17600, 9300)
    training = [
        _preview(edge=11, base_rgb=base, scene_rgb=(10000, 8000, 5000), seed=1),
        _preview(edge=18, base_rgb=base, scene_rgb=(57000, 53000, 49000), seed=2),
        _preview(edge=14, base_rgb=base, scene_rgb=(26000, 22000, 17000), seed=3),
    ]
    model = learn_film_base(training)

    dark = measure_target_start(_preview(edge=17, base_rgb=base, scene_rgb=(2500, 1800, 900), seed=4), model)
    bright = measure_target_start(_preview(edge=8, base_rgb=base, scene_rgb=(62500, 60000, 56000), seed=5), model)

    assert (dark.row, bright.row) == (17, 8)


def test_edge_prior_disambiguates_a_long_smooth_patch_inside_the_scene() -> None:
    base = (41200, 17600, 9300)
    training = [
        _preview(edge=10 + index, base_rgb=base, scene_rgb=scene, seed=index)
        for index, scene in enumerate(((4000, 3000, 2000), (61000, 58000, 54000), (26000, 21000, 17000)))
    ]
    model = learn_film_base(training)
    preview = _preview(edge=16, base_rgb=base, scene_rgb=(3000, 2200, 1300), seed=7)
    preview[38:57] = np.asarray(base, dtype=np.uint16)

    edge = measure_target_start(preview, model, expected_row=15.0, maximum_prediction_error=6.0)

    assert edge.row == 16


def test_edge_prior_can_recover_a_direct_short_prefix_at_the_transport_boundary() -> None:
    base = (41200, 17600, 9300)
    training = [
        _preview(edge=10 + index, base_rgb=base, scene_rgb=scene, seed=index)
        for index, scene in enumerate(((4000, 3000, 2000), (61000, 58000, 54000), (26000, 21000, 17000)))
    ]
    model = learn_film_base(training)
    clipped = _preview(edge=1, base_rgb=base, scene_rgb=(7000, 5000, 3000), seed=11)

    edge = measure_target_start(
        clipped,
        model,
        expected_row=17.0,
        maximum_prediction_error=12.0,
        allow_short_prefix=True,
    )

    assert edge.row == 1
    assert edge.confidence == "medium"


def test_model_covers_normal_film_base_drift_across_a_roll() -> None:
    nominal = np.asarray((40000, 17000, 9000))
    offsets = ((-900, -380, -700), (0, 0, 0), (700, 290, 600), (2200, 900, 1800))
    scenes = ((3000, 2200, 1200), (60000, 31000, 18000), (19000, 9000, 4500), (52000, 26000, 14000))
    training = [
        _preview(
            edge=10 + index,
            base_rgb=tuple(nominal + offset),
            scene_rgb=scene,
            seed=20 + index,
        )
        for index, (offset, scene) in enumerate(zip(offsets, scenes, strict=True))
    ]
    model = learn_film_base(training)
    live_base = tuple(nominal + np.asarray((1800, 750, 1500)))

    edge = measure_target_start(
        _preview(edge=15, base_rgb=live_base, scene_rgb=(1800, 1300, 700), seed=30),
        model,
    )

    assert edge.row == 15


def test_transport_tail_detector_is_invariant_to_bright_and_dark_scenes() -> None:
    bright = _tail_preview()
    dark = (bright // 5 + 400).astype(np.uint16)

    bright_tail = observe_transport_tail(bright)
    dark_tail = observe_transport_tail(dark)

    assert bright_tail is not None
    assert dark_tail is not None
    assert (bright_tail.burst_row, bright_tail.tail_start_row, bright_tail.confidence) == (35, 40, "high")
    assert (dark_tail.burst_row, dark_tail.tail_start_row, dark_tail.confidence) == (35, 40, "high")


def test_transport_tail_fit_uses_position_keys_and_rejects_scene_outliers() -> None:
    observations = {
        position: TailObservation(
            burst_row=555 - 2 * position,
            tail_start_row=600 - 2 * position,
            burst_ratio=12.0,
            stationarity_ratio=1.1,
            confidence="high",
        )
        for position in range(20, 30)
    }
    observations[22] = TailObservation(511, 586, 12.0, 1.1, "high")
    observations[28] = TailObservation(499, 524, 12.0, 1.1, "high")

    model = fit_transport_tail(observations, residual_tolerance_rows=1.5)

    assert (model.slope, model.intercept) == (-2.0, 600.0)
    assert model.predict(35) == 530.0
    assert 22 not in model.inlier_positions
    assert 28 not in model.inlier_positions


def test_joint_roll_tail_measurement_marks_direct_inferred_and_outside_boundaries() -> None:
    previews = {position: _tail_preview(tail_start=80 - 2 * position) for position in range(20, 28)}
    no_tail = _preview(
        edge=5,
        base_rgb=(40000, 17000, 9000),
        scene_rgb=(7000, 5000, 3000),
        height=52,
        seed=40,
    )
    previews[15] = no_tail
    previews[10] = no_tail

    boundaries = measure_transport_tails(previews)

    assert (boundaries[20].tail_start_row, boundaries[20].source) == (40.0, "direct")
    assert (boundaries[15].tail_start_row, boundaries[15].source) == (50.0, "inferred")
    assert (boundaries[10].tail_start_row, boundaries[10].source) == (60.0, "outside")


def test_joint_tail_measurement_fails_when_a_visible_predicted_tail_is_missing() -> None:
    previews = {position: _tail_preview(tail_start=80 - 2 * position) for position in range(20, 28)}
    previews[30] = _preview(
        edge=5,
        base_rgb=(40000, 17000, 9000),
        scene_rgb=(7000, 5000, 3000),
        height=52,
        seed=41,
    )

    with pytest.raises(ValueError, match="visible transport tail"):
        measure_transport_tails(previews)


def test_aligned_margin_only_uses_the_film_base_run_at_the_prefix() -> None:
    base = (41200, 17600, 9300)
    training = [
        _preview(edge=12 + index, base_rgb=base, scene_rgb=scene, seed=index)
        for index, scene in enumerate(((5000, 4000, 3000), (59000, 56000, 51000), (22000, 19000, 15000)))
    ]
    model = learn_film_base(training)
    aligned = _preview(edge=6, base_rgb=base, scene_rgb=(8000, 7000, 6000), seed=9)
    # A later smooth patch happens to look exactly like film base.  It is not
    # the registration margin because it is disconnected from row zero.
    aligned[30:44] = np.asarray(base, dtype=np.uint16)

    margin = measure_leading_margin(aligned, model)

    assert margin.row == 6


def test_pitch_prior_ignores_an_excluded_cut_frame_and_a_measurement_outlier() -> None:
    observed = {frame: 121.0 - 4.0 * frame for frame in range(1, 9)}
    observed[1] += 47.0  # The first exposure was physically cut short.
    observed[6] -= 31.0  # One local CV lock chose a dark patch in the scene.

    prediction = robust_pitch_prediction(observed, frame=12, excluded_frames={1})

    assert prediction == 73.0


def test_shifted_reference_prior_uses_the_robust_refeed_offset() -> None:
    reference = {1: 118, 2: 113, 3: 109, 4: 106, 5: 100, 6: 97, 12: 73}
    observed = {1: 179, 2: 120, 3: 116, 4: 149, 5: 107, 6: 104}

    prediction = shifted_reference_prediction(reference, observed, frame=12, excluded_frames={1})

    assert prediction == 80.0
