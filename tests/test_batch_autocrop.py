from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import negpy.features.geometry.batch_autocrop as batch_autocrop
from negpy.features.geometry.batch_autocrop import (
    CropEvidence,
    _inset_rect_by_border,
    _map_rect_between_rotations,
    _pixel_roi,
    _resolve_border,
    _roll_border,
    _top_edge_slope,
    add_uniform_safety_border,
    build_roll_template,
    detect_crop_candidate,
    resolve_roll_crops,
)


_LANDSCAPE_SHAPE = (1000, 1000)


def _evidence(
    key: str,
    *,
    canvas_shape: tuple[int, int] = _LANDSCAPE_SHAPE,
    roi: tuple[int, int, int, int] | None = (100, 900, 100, 900),
    correction_angle: float = 1.0,
    confidence: float = 0.9,
    target_ratio: str = "3:2",
    supported_sides: frozenset[str] = frozenset({"left", "right"}),
    supported_corners: frozenset[str] = frozenset(),
    geometry_score: float = 0.8,
    border: tuple[float, ...] = (),
    rebate_trim: float = 1.0,
    angle_confident: bool = False,
    vertical_edge_profile: np.ndarray | None = None,
    vertical_edge_contrast: float | None = None,
    reason: str = "",
) -> CropEvidence:
    profile = np.empty(0, dtype=np.float32) if vertical_edge_profile is None else np.asarray(vertical_edge_profile, dtype=np.float32)
    profile_contrast = float(np.max(profile)) if vertical_edge_contrast is None and profile.size else float(vertical_edge_contrast or 0.0)
    return CropEvidence(
        key=key,
        canvas_shape=canvas_shape,
        roi=roi,
        correction_angle=correction_angle,
        confidence=confidence,
        target_ratio=target_ratio,
        supported_sides=supported_sides,
        supported_corners=supported_corners,
        geometry_score=geometry_score,
        border=border,
        rebate_trim=rebate_trim,
        angle_confident=angle_confident,
        vertical_edge_contrast=profile_contrast,
        vertical_edge_profile=profile,
        reason=reason,
    )


def _trusted_roll(
    angles: tuple[float, float, float] = (0.9, 1.0, 1.1),
) -> list[CropEvidence]:
    return [_evidence(f"trusted-{index}", correction_angle=angle) for index, angle in enumerate(angles)]


def _resolved_by_key(evidence: list[CropEvidence]) -> dict[str, object]:
    return {item.key: item for item in resolve_roll_crops(evidence, safety_border=0.0)}


def test_roll_template_rejects_width_and_angle_outlier_deterministically() -> None:
    evidence = [
        _evidence("frame-a", roi=(100, 900, 100, 900), correction_angle=0.9),
        _evidence("frame-b", roi=(100, 900, 105, 905), correction_angle=1.0),
        _evidence("frame-c", roi=(100, 900, 95, 895), correction_angle=1.1),
        _evidence("frame-d", roi=(100, 900, 100, 900), correction_angle=1.0),
        _evidence("outlier", roi=(250, 750, 300, 700), correction_angle=6.0),
    ]

    forward_template = build_roll_template(evidence)
    reverse_template = build_roll_template(list(reversed(evidence)))

    assert forward_template is not None
    assert reverse_template == forward_template
    assert forward_template.sample_count == 4
    assert forward_template.width == pytest.approx(0.8)
    assert forward_template.correction_angle == pytest.approx(1.0)

    forward_results = resolve_roll_crops(evidence, safety_border=0.0)
    reverse_results = resolve_roll_crops(list(reversed(evidence)), safety_border=0.0)
    assert [item.key for item in forward_results] == [item.key for item in evidence]
    assert [item.key for item in reverse_results] == [item.key for item in reversed(evidence)]
    assert {item.key: item for item in reverse_results} == {item.key: item for item in forward_results}


def test_short_detection_expands_to_roll_width_from_supported_left_edge() -> None:
    short = _evidence(
        "short",
        roi=(100, 900, 150, 870),
        supported_sides=frozenset({"left"}),
    )

    resolved = _resolved_by_key([*_trusted_roll(), short])["short"]

    assert resolved.manual_crop_rect == pytest.approx((0.15, 0.1, 0.95, 0.9))
    assert resolved.calibrated is True


def test_weak_frame_resolves_from_profile_edges_near_roll_template() -> None:
    profile = np.zeros(101, dtype=np.float32)
    profile[10] = 1.0
    profile[90] = 1.0
    weak = _evidence(
        "weak-profile",
        roi=None,
        correction_angle=0.0,
        confidence=0.0,
        supported_sides=frozenset(),
        geometry_score=0.0,
        vertical_edge_profile=profile,
        reason="no_consensus",
    )

    resolved = _resolved_by_key([*_trusted_roll(), weak])["weak-profile"]

    expected = _map_rect_between_rotations((0.1, 0.1, 0.9, 0.9), _LANDSCAPE_SHAPE, 0.0, 1.0)
    assert resolved.manual_crop_rect == pytest.approx(expected, abs=6e-4)
    assert resolved.correction_angle == pytest.approx(1.0)
    assert resolved.confidence == pytest.approx(0.55)
    assert resolved.calibrated is True


def test_weak_frame_without_profile_edges_abstains() -> None:
    weak = _evidence(
        "no-edges",
        roi=None,
        confidence=0.0,
        supported_sides=frozenset(),
        geometry_score=0.0,
        vertical_edge_profile=np.zeros(101, dtype=np.float32),
        reason="no_consensus",
    )

    assert "no-edges" not in _resolved_by_key([*_trusted_roll(), weak])


def test_near_flat_noise_profile_abstains_even_with_a_valid_template() -> None:
    rng = np.random.default_rng(17)
    noise = np.clip(0.5 + rng.normal(0.0, 1e-5, (200, 300, 3)), 0.0, 1.0).astype(np.float32)
    weak = detect_crop_candidate("near-flat", noise)

    assert weak.roi is None
    assert weak.vertical_edge_contrast < 0.06
    assert "near-flat" not in _resolved_by_key([*_trusted_roll(), weak])


def test_post_deskew_abstention_does_not_inherit_initial_confidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = np.zeros(300, dtype=np.float32)
    initial = SimpleNamespace(
        roi=(20, 180, 30, 270),
        correction_angle=2.0,
        confidence=0.9,
        supported_sides=frozenset({"top", "right", "bottom", "left"}),
        supported_corners=frozenset({"top_left"}),
        evidence_sources=("adaptive-dark",),
        geometry_score=0.9,
        vertical_edge_contrast=0.8,
        vertical_edge_profile=profile,
    )
    final = SimpleNamespace(
        roi=None,
        correction_angle=0.0,
        confidence=0.0,
        supported_sides=frozenset(),
        supported_corners=frozenset(),
        evidence_sources=(),
        geometry_score=0.0,
        vertical_edge_contrast=0.0,
        vertical_edge_profile=profile,
    )
    detections = iter((initial, final))
    monkeypatch.setattr(batch_autocrop, "detect_film_bounds_with_confidence", lambda _image: next(detections))
    monkeypatch.setattr(
        batch_autocrop,
        "measure_film_border",
        lambda *_args, **_kwargs: pytest.fail("fallback geometry must not be assigned trusted confidence"),
    )

    evidence = detect_crop_candidate("deskew-failed", np.ones((200, 300, 3), dtype=np.float32))

    assert evidence.roi is None
    assert evidence.confidence == 0.0
    assert evidence.correction_angle == pytest.approx(2.0)
    assert evidence.reason == "deskew_no_consensus"


def test_portrait_detection_and_resolution_abstain_before_detector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_detector(_image: np.ndarray) -> None:
        pytest.fail("portrait input should not reach the landscape film detector")

    monkeypatch.setattr(
        batch_autocrop,
        "detect_film_bounds_with_confidence",
        unexpected_detector,
    )
    portrait = detect_crop_candidate(
        "portrait",
        np.zeros((120, 80, 3), dtype=np.float32),
    )

    assert portrait.roi is None
    assert portrait.confidence == 0.0
    assert portrait.reason == "unsupported_orientation"
    assert "portrait" not in _resolved_by_key([*_trusted_roll(), portrait])


def test_safety_border_uses_equal_one_percent_padding_or_common_edge_limit() -> None:
    unconstrained = (100, 900, 200, 1800)
    assert add_uniform_safety_border(unconstrained, (1000, 2000)) == (
        90,
        910,
        190,
        1810,
    )

    edge_limited = (4, 900, 20, 1900)
    padded = add_uniform_safety_border(edge_limited, (1000, 2000))
    assert padded == (0, 904, 16, 1904)
    assert (
        edge_limited[0] - padded[0],
        padded[1] - edge_limited[1],
        edge_limited[2] - padded[2],
        padded[3] - edge_limited[3],
    ) == (4, 4, 4, 4)


def test_divergent_frame_maps_crop_before_using_roll_median_angle() -> None:
    divergent = _evidence(
        "divergent-angle",
        correction_angle=4.0,
        supported_sides=frozenset({"left"}),
    )

    resolved = _resolved_by_key([*_trusted_roll(), divergent])["divergent-angle"]

    expected = _map_rect_between_rotations((0.1, 0.1, 0.9, 0.9), _LANDSCAPE_SHAPE, 4.0, 1.0)
    assert resolved.manual_crop_rect == pytest.approx(expected, abs=6e-4)
    assert resolved.correction_angle == pytest.approx(1.0)
    assert resolved.calibrated is True


def test_rotation_mapping_quantizes_half_open_bounds_outward() -> None:
    shape = (101, 203)
    source_roi = (10, 101, 99, 184)
    source_rect = (source_roi[2] / 203, source_roi[0] / 101, source_roi[3] / 203, source_roi[1] / 101)

    mapped = _map_rect_between_rotations(source_rect, shape, 0.0, 4.144)

    assert _pixel_roi(mapped, shape) == (4, 101, 96, 188)


def test_roll_templates_do_not_mix_different_target_ratios() -> None:
    three_two = _trusted_roll()
    four_three = _evidence(
        "four-three",
        roi=(100, 900, 145, 855),
        target_ratio="4:3",
    )

    resolved = _resolved_by_key([*three_two, four_three])["four-three"]

    assert resolved.manual_crop_rect == pytest.approx((0.145, 0.1, 0.855, 0.9))


def test_resolved_rect_preserves_half_open_coordinates_when_normalized() -> None:
    evidence = _evidence(
        "exclusive",
        canvas_shape=(101, 203),
        roi=(7, 97, 11, 199),
        correction_angle=0.25,
    )

    resolved = _resolved_by_key([evidence])["exclusive"]

    assert resolved.manual_crop_rect == pytest.approx((11 / 203, 7 / 101, 199 / 203, 97 / 101))
    assert (resolved.manual_crop_rect[2] - resolved.manual_crop_rect[0]) * 203 == pytest.approx(188)
    assert (resolved.manual_crop_rect[3] - resolved.manual_crop_rect[1]) * 101 == pytest.approx(90)


_NAN = float("nan")


def test_roll_border_medians_each_side_and_outvotes_a_wild_frame() -> None:
    evidence = [_evidence(f"f{index}", border=(0.01, 0.02, 0.015, 0.012)) for index in range(5)]
    evidence.append(_evidence("wild", border=(0.30, 0.30, 0.30, 0.30)))

    border, samples = _roll_border(evidence)

    assert samples == 6
    assert border == pytest.approx((0.01, 0.02, 0.015, 0.012))


def test_roll_border_excludes_abstaining_sides_rather_than_counting_them_as_zero() -> None:
    evidence = [_evidence(f"f{index}", border=(0.01, 0.02, 0.015, 0.012)) for index in range(5)]
    evidence.append(_evidence("partial", border=(_NAN, 0.9, 0.015, 0.012)))

    border, samples = _roll_border(evidence)

    assert border[0] == pytest.approx(0.01)
    assert samples == 5  # limited by the side that abstained


def test_roll_border_requires_a_minimum_sample_count() -> None:
    evidence = [_evidence(f"f{index}", border=(0.01, 0.02, 0.015, 0.012)) for index in range(4)]

    assert _roll_border(evidence) == ((), 4)


def test_frame_border_wins_over_the_roll_median() -> None:
    # How much bed the detector leaves inside the film box varies per frame, so a
    # frame that measured its own border must not be overridden by the roll.
    own = (0.05, 0.02, 0.015, 0.012)

    assert _resolve_border(own, (0.01, 0.02, 0.015, 0.012)) == pytest.approx(own)


def test_abstaining_side_falls_back_to_the_roll_median() -> None:
    resolved = _resolve_border((_NAN, 0.02, 0.015, 0.012), (0.01, 0.09, 0.09, 0.09))

    assert resolved[0] == pytest.approx(0.01)
    assert resolved[1] == pytest.approx(0.02)


def test_no_roll_border_means_no_trim() -> None:
    assert _resolve_border((0.05, 0.02, 0.015, 0.012), ()) == ()


def test_border_inset_trims_each_side_as_a_fraction_of_the_film_box() -> None:
    inset = _inset_rect_by_border((0.1, 0.1, 0.9, 0.9), (0.25, 0.125, 0.5, 0.25))

    assert inset == pytest.approx((0.5, 0.3, 0.7, 0.8))


def test_border_inset_keeps_the_rect_when_it_would_collapse() -> None:
    rect = (0.1, 0.1, 0.9, 0.9)

    assert _inset_rect_by_border(rect, (0.6, 0.6, 0.1, 0.1)) == rect


def test_resolved_crop_trims_the_rebate_when_the_roll_measured_one() -> None:
    evidence = [_evidence(f"f{index}", border=(0.02, 0.02, 0.02, 0.02)) for index in range(6)]

    x1, y1, x2, y2 = _resolved_by_key(evidence)["f0"].manual_crop_rect

    assert x1 > 0.1 and y1 > 0.1
    assert x2 < 0.9 and y2 < 0.9


def test_resolved_crop_passes_through_untouched_without_a_roll_border() -> None:
    resolved = _resolved_by_key(_trusted_roll())["trusted-0"]

    assert resolved.manual_crop_rect == pytest.approx((0.1, 0.1, 0.9, 0.9))


def _trimmed_roll(rebate_trim: float) -> tuple[float, float, float, float]:
    evidence = [_evidence(f"f{index}", border=(0.02, 0.02, 0.02, 0.02), rebate_trim=rebate_trim) for index in range(6)]
    return _resolved_by_key(evidence)["f0"].manual_crop_rect


def test_rebate_trim_zero_keeps_the_whole_film_box() -> None:
    assert _trimmed_roll(0.0) == pytest.approx((0.1, 0.1, 0.9, 0.9))


def test_rebate_trim_cuts_further_the_higher_it_goes() -> None:
    full, over = _trimmed_roll(1.0), _trimmed_roll(1.5)

    assert over[0] > full[0] > 0.1
    assert over[2] < full[2] < 0.9


def _tilted_film_box(angle: float) -> np.ndarray:
    from negpy.features.geometry.logic import apply_fine_rotation

    box = np.full((400, 600), 0.2, dtype=np.float32)
    box[100:300, 50:550] = 0.9
    return apply_fine_rotation(box, angle) if angle else box


@pytest.mark.parametrize("tilt", [-0.7, -0.3, 0.0, 0.4, 1.0])
def test_top_edge_slope_returns_the_correction_that_levels_the_edge(tilt: float) -> None:
    # The fit is added to the consensus angle, so a frame tilted by `tilt` must measure
    # -tilt. Both signs pinned: a flipped convention doubles the error.
    assert _top_edge_slope(_tilted_film_box(tilt), (100, 300, 50, 550)) == pytest.approx(-tilt, abs=0.05)


def test_top_edge_slope_abstains_when_no_edge_reads_straight() -> None:
    noise = np.asarray(np.random.default_rng(7).random((400, 600)), dtype=np.float32)

    assert _top_edge_slope(noise, (100, 300, 50, 550)) is None


def test_top_edge_slope_abstains_on_a_flat_band_rather_than_reporting_zero() -> None:
    assert _top_edge_slope(np.ones((400, 600), dtype=np.float32), (100, 300, 50, 550)) is None


def test_detect_candidate_folds_the_edge_fit_into_the_consensus_angle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = np.zeros(600, dtype=np.float32)
    detection = SimpleNamespace(
        roi=(100, 300, 50, 550),
        correction_angle=0.2,
        confidence=0.9,
        supported_sides=frozenset({"top", "right", "bottom", "left"}),
        supported_corners=frozenset({"top_left"}),
        evidence_sources=("adaptive-dark",),
        geometry_score=0.9,
        vertical_edge_contrast=0.8,
        vertical_edge_profile=profile,
    )
    monkeypatch.setattr(batch_autocrop, "detect_film_bounds_with_confidence", lambda _image: detection)
    monkeypatch.setattr(batch_autocrop, "apply_fine_rotation", lambda image, _angle: image)
    monkeypatch.setattr(batch_autocrop, "_trim_opaque_border", lambda _lum, roi: roi)
    monkeypatch.setattr(batch_autocrop, "measure_film_border", lambda *_a, **_k: dict.fromkeys(batch_autocrop.BORDER_SIDES, 0.0))

    evidence = detect_crop_candidate("fitted", np.repeat(_tilted_film_box(-0.3)[:, :, None], 3, axis=2))

    assert evidence.angle_confident is True
    assert evidence.correction_angle == pytest.approx(0.5, abs=0.05)  # 0.2 consensus + 0.3 measured


def test_fitted_angle_survives_a_mediocre_box_score() -> None:
    # Box confidence and edge tilt measure different things.
    weak = _evidence("weak", correction_angle=1.4, confidence=0.4, angle_confident=True)

    assert _resolved_by_key([*_trusted_roll(), weak])["weak"].correction_angle == pytest.approx(1.4)


def test_unfitted_weak_frame_still_takes_the_roll_angle() -> None:
    weak = _evidence("weak", correction_angle=1.4, confidence=0.4, angle_confident=False)

    assert _resolved_by_key([*_trusted_roll(), weak])["weak"].correction_angle == pytest.approx(1.0)


def test_fitted_angle_still_yields_when_it_diverges_beyond_tolerance() -> None:
    divergent = _evidence("divergent", correction_angle=5.0, confidence=0.4, angle_confident=True)

    assert _resolved_by_key([*_trusted_roll(), divergent])["divergent"].correction_angle == pytest.approx(1.0)


def test_roll_angle_comes_from_the_fitted_frames_once_enough_carry_one() -> None:
    evidence = [
        _evidence("fit-a", correction_angle=2.0, angle_confident=True),
        _evidence("fit-b", correction_angle=2.1, angle_confident=True),
        _evidence("fit-c", correction_angle=1.9, angle_confident=True),
        _evidence("blob-a", correction_angle=1.0),
        _evidence("blob-b", correction_angle=1.0),
    ]

    template = build_roll_template(evidence)

    assert template is not None
    assert template.correction_angle == pytest.approx(2.0)


def test_roll_angle_keeps_every_frame_below_the_fitted_minimum() -> None:
    evidence = [
        _evidence("fit-a", correction_angle=2.0, angle_confident=True),
        _evidence("fit-b", correction_angle=2.0, angle_confident=True),
        _evidence("blob-a", correction_angle=1.0),
        _evidence("blob-b", correction_angle=1.0),
        _evidence("blob-c", correction_angle=1.0),
    ]

    template = build_roll_template(evidence)

    assert template is not None
    assert template.correction_angle == pytest.approx(1.0)
