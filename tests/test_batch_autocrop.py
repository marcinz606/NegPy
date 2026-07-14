from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import negpy.features.geometry.batch_autocrop as batch_autocrop
from negpy.features.geometry.batch_autocrop import (
    CropEvidence,
    _map_rect_between_rotations,
    _pixel_roi,
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
        "get_autocrop_coords",
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
