"""Scanner-independent registration policy and geometry tests."""

import math

import numpy as np
import pytest

from negpy.services.scanning.quality import StoppedTransportSmearAssessment
from negpy.infrastructure.scanners.params import RegisteredScanGeometry
from negpy.infrastructure.scanners.frame_registration import TailBoundary
from negpy.services.scanning import roll_registration as roll_registration_module
from negpy.services.scanning.roll_registration import (
    PreviewRollRegistration,
    RollRegistrationConfig,
    _qc_transport_boundaries,
    registered_scan_geometry,
)


def test_measured_frame_three_edge_and_tail_reproduce_the_verified_scan_geometry() -> None:
    geometry = registered_scan_geometry(
        frame=3,
        target_start_row=109,
        tail_start_row=602.5,
        config=RollRegistrationConfig(),
    )
    assert geometry == RegisteredScanGeometry(
        frame=3,
        subframe_mm=6.35,
        br_y_device_px=5003,
    )


def test_clipped_last_frame_rolls_back_to_the_previous_transport_position() -> None:
    geometry = registered_scan_geometry(
        frame=37,
        target_start_row=1,
        tail_start_row=518.0,
        config=RollRegistrationConfig(),
    )

    assert geometry == RegisteredScanGeometry(
        frame=36,
        subframe_mm=37.82,
        br_y_device_px=5162,
    )


def test_ls5000_registration_stride_matches_the_declared_full_window() -> None:
    config = RollRegistrationConfig()

    assert config.frame_pitch_mm == pytest.approx(5959 * 25.4 / 4000)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"preview_dpi": 0},
        {"device_dpi": True},
        {"retained_margin_mm": -0.01},
        {"retained_margin_mm": math.inf},
        {"tail_guard_device_px": -1},
        {"inclusive_endpoint_adjustment_px": -1},
        {"alignment_tolerance_rows": -1},
        {"maximum_edge_prediction_error_rows": math.nan},
        {"excluded_frames": frozenset({0})},
    ],
)
def test_registration_config_rejects_impossible_values(kwargs) -> None:
    with pytest.raises(ValueError):
        RollRegistrationConfig(**kwargs)


def test_real_policy_declares_and_defensively_checks_three_frame_minimum() -> None:
    policy = PreviewRollRegistration(RollRegistrationConfig())
    previews = {
        2: np.zeros((12, 16, 3), dtype=np.uint16),
        3: np.zeros((12, 16, 3), dtype=np.uint16),
    }

    assert policy.minimum_preview_count == 3
    with pytest.raises(ValueError, match="at least 3"):
        policy.calibrate(previews)


def _wide_preview(*, edge: int, tail: int, seed: int) -> np.ndarray:
    height, width = 72, 32
    rng = np.random.default_rng(seed)
    image = np.empty((height, width, 3), dtype=np.float64)
    base = np.asarray((42000, 18000, 9500))
    image[:edge] = base + rng.normal(0, 6, size=(edge, width, 3))
    y, x = np.mgrid[: tail - edge, :width]
    scene = np.asarray((12000, 8000, 4500))
    texture = 800 * np.sin(x / 2.5) + 350 * np.cos(y / 3.0)
    image[edge:tail] = scene + texture[..., None] + rng.normal(0, 20, size=(tail - edge, width, 3))
    image[tail - 4 : tail] -= 5000
    image[tail:] = np.asarray((35000, 30000, 24000)) + 500 * np.sin(np.arange(width) / 2.5)[:, None]
    return np.clip(image, 0, 65535).astype(np.uint16)


def _aligned_preview(*, margin: int, seed: int) -> np.ndarray:
    image = _wide_preview(edge=margin, tail=60, seed=seed)
    return image[:52].copy()  # shortened before the synthetic transport tail


def test_policy_jointly_measures_edges_and_transport_tails_then_verifies_margin() -> None:
    frames = range(10, 16)
    edges = {frame: 10 + frame % 3 for frame in frames}
    tails = {frame: 80 - 2 * frame for frame in frames}
    previews = {frame: _wide_preview(edge=edges[frame], tail=tails[frame], seed=frame) for frame in frames}
    policy = PreviewRollRegistration(RollRegistrationConfig())

    registrations = policy.calibrate(previews)

    assert set(registrations) == set(frames)
    for frame in frames:
        registration = registrations[frame]
        assert registration.target_start_row == edges[frame]
        assert registration.usable_tail_row == tails[frame]
        assert registration.geometry.frame == frame

    verification = policy.verify(
        12,
        _aligned_preview(margin=9, seed=50),
        registrations[12],
    )
    assert verification.passed
    assert verification.leading_margin_rows == 9


def test_policy_recovers_and_verifies_a_last_frame_clipped_by_the_nominal_origin() -> None:
    frames = range(32, 38)
    edges = {32: 14, 33: 12, 34: 10, 35: 8, 36: 6, 37: 1}
    previews = {frame: _wide_preview(edge=edges[frame], tail=60, seed=300 + frame) for frame in frames}
    policy = PreviewRollRegistration(RollRegistrationConfig())

    registrations = policy.calibrate(previews)
    frame_37 = registrations[37]
    verification = policy.verify(
        37,
        _aligned_preview(margin=4, seed=401),
        frame_37,
    )

    assert frame_37.target_start_row == 1
    assert frame_37.geometry.frame == 36
    assert frame_37.geometry.subframe_mm == 37.82
    assert verification.target_margin_rows == 1
    assert verification.passed


def test_excluded_cut_frame_keeps_its_local_edge_instead_of_normal_pitch_prior() -> None:
    previews = {
        frame: _wide_preview(
            edge=30 if frame == 1 else 10,
            tail=65 - 2 * frame,
            seed=70 + frame,
        )
        for frame in range(1, 7)
    }
    policy = PreviewRollRegistration(RollRegistrationConfig(excluded_frames=frozenset({1})))

    registrations = policy.calibrate(previews)

    assert registrations[1].target_start_row == 30
    assert all(registrations[frame].target_start_row == 10 for frame in range(2, 7))


def test_roll_prediction_recovers_a_scene_that_only_exposes_a_false_local_edge() -> None:
    previews = {
        frame: _wide_preview(
            edge=55 if frame == 4 else 10,
            tail=65,
            seed=170 + frame,
        )
        for frame in range(2, 8)
    }
    policy = PreviewRollRegistration(RollRegistrationConfig())

    registrations = policy.calibrate(previews)

    assert registrations[4].target_start_row == 10
    assert registrations[4].confidence == "medium"


def test_uncertain_aligned_preview_is_recorded_for_human_review_instead_of_aborting() -> None:
    previews = {frame: _wide_preview(edge=10, tail=80 - 2 * frame, seed=90 + frame) for frame in range(10, 16)}
    policy = PreviewRollRegistration(RollRegistrationConfig())
    registrations = policy.calibrate(previews)
    no_visible_film_base = np.full((52, 32, 3), 7000, dtype=np.uint16)

    verification = policy.verify(10, no_visible_film_base, registrations[10])

    assert verification.leading_margin_rows is None
    assert verification.confidence == "unresolved"
    assert not verification.passed


def test_qc_tail_fallback_keeps_full_height_and_lowers_confidence_when_either_band_is_indeterminate(
    monkeypatch,
) -> None:
    previews = {frame: np.full((12, 32, 3), frame, dtype=np.uint16) for frame in (2, 3, 4)}

    def assessment(verdict: str, *, start_row: int | None = None) -> StoppedTransportSmearAssessment:
        return StoppedTransportSmearAssessment(
            verdict=verdict,
            start_row=start_row,
            suffix_rows=0 if start_row is None else 12 - start_row,
            minimum_matches=7,
            tail_median_rms=None,
            tail_min_corr=None,
            pre_tail_median_rms=None,
            texture_span=5000.0,
            reason="test",
        )

    def fake_smear(image, *, dpi):
        assert dpi == 400
        frame = int(image[0, 0, 0])
        is_outer_band = image.shape[1] == 16
        if frame == 2 and is_outer_band:
            return assessment("indeterminate", start_row=10)
        return assessment("clean")

    monkeypatch.setattr(roll_registration_module, "assess_stopped_transport_smear", fake_smear)

    boundaries = _qc_transport_boundaries(
        previews,
        config=RollRegistrationConfig(expected_full_preview_rows=12),
    )

    assert boundaries == {
        2: TailBoundary(tail_start_row=12.0, source="complete", confidence="medium"),
        3: TailBoundary(tail_start_row=12.0, source="complete", confidence="high"),
        4: TailBoundary(tail_start_row=12.0, source="complete", confidence="high"),
    }


def test_qc_tail_fallback_keeps_complete_frame_when_indeterminate_starts_disagree_like_frame_6(monkeypatch) -> None:
    starts = iter((593, 562))

    def fake_smear(_image, *, dpi):
        assert dpi == 400
        start = next(starts)
        return StoppedTransportSmearAssessment(
            "indeterminate",
            start,
            595 - start,
            7,
            100.0,
            0.99991,
            None,
            22000.0,
            "scene-driven partial EOF signature",
        )

    monkeypatch.setattr(roll_registration_module, "assess_stopped_transport_smear", fake_smear)

    boundary = _qc_transport_boundaries(
        {6: np.zeros((595, 394, 3), dtype=np.uint16)},
        config=RollRegistrationConfig(),
    )[6]

    assert boundary == TailBoundary(
        tail_start_row=595.0,
        source="complete",
        confidence="medium",
    )


def test_qc_tail_fallback_keeps_complete_frame_when_indeterminate_starts_disagree_like_frame_25(monkeypatch) -> None:
    starts = iter((585, 587))

    def fake_smear(_image, *, dpi):
        assert dpi == 400
        start = next(starts)
        return StoppedTransportSmearAssessment(
            "indeterminate",
            start,
            595 - start,
            7,
            130.0,
            0.99994,
            None,
            36000.0,
            "scene-driven partial EOF signature",
        )

    monkeypatch.setattr(roll_registration_module, "assess_stopped_transport_smear", fake_smear)

    boundary = _qc_transport_boundaries(
        {25: np.zeros((595, 394, 3), dtype=np.uint16)},
        config=RollRegistrationConfig(),
    )[25]

    assert boundary == TailBoundary(
        tail_start_row=595.0,
        source="complete",
        confidence="medium",
    )


def test_qc_tail_fallback_rejects_a_short_preview_before_smear_assessment(monkeypatch) -> None:
    monkeypatch.setattr(
        roll_registration_module,
        "assess_stopped_transport_smear",
        lambda *_args, **_kwargs: pytest.fail("short preview reached smear QC"),
    )

    with pytest.raises(ValueError, match="expected 12 rows"):
        _qc_transport_boundaries(
            {2: np.zeros((11, 32, 3), dtype=np.uint16)},
            config=RollRegistrationConfig(expected_full_preview_rows=12),
        )


def test_qc_tail_fallback_never_promotes_an_indeterminate_boundary_to_high(monkeypatch) -> None:
    calls = 0

    def fake_smear(_image, *, dpi):
        nonlocal calls
        calls += 1
        if calls == 1:
            return StoppedTransportSmearAssessment("smear", 10, 2, 7, 50.0, 0.99999, 500.0, 5000.0, "test")
        return StoppedTransportSmearAssessment("indeterminate", 4, 8, 7, 90.0, 0.99995, None, 5000.0, "test")

    monkeypatch.setattr(roll_registration_module, "assess_stopped_transport_smear", fake_smear)

    boundary = _qc_transport_boundaries(
        {2: np.zeros((12, 32, 3), dtype=np.uint16)},
        config=RollRegistrationConfig(expected_full_preview_rows=12),
    )[2]

    assert boundary == TailBoundary(tail_start_row=10.0, source="qc", confidence="medium")


def test_qc_tail_fallback_rejects_disagreeing_full_and_outer_smear_boundaries(monkeypatch) -> None:
    starts = iter((10, 4))

    def fake_smear(_image, *, dpi):
        start = next(starts)
        return StoppedTransportSmearAssessment("smear", start, 12 - start, 7, 50.0, 0.99999, 500.0, 5000.0, "test")

    monkeypatch.setattr(roll_registration_module, "assess_stopped_transport_smear", fake_smear)

    with pytest.raises(ValueError, match="boundaries disagree"):
        _qc_transport_boundaries(
            {2: np.zeros((12, 32, 3), dtype=np.uint16)},
            config=RollRegistrationConfig(expected_full_preview_rows=12),
        )
