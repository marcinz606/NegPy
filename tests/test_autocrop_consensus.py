from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from negpy.features.geometry.logic import (
    _FilmCandidate,
    _select_consensus_cluster,
    apply_fine_rotation,
    detect_film_bounds_with_confidence,
)


def _two_tier_scan(
    height: int = 360,
    width: int = 540,
    *,
    surround: float = 1.0,
    film: float = 0.55,
) -> np.ndarray:
    image = np.full((height, width, 3), surround, dtype=np.float32)
    y1, y2 = round(height / 6), round(5 * height / 6)
    x1, x2 = round(4 * width / 27), round(23 * width / 27)
    image[y1:y2, x1:x2] = film
    return image


def _normalized_roi(roi, shape):
    h, w = shape[:2]
    y1, y2, x1, x2 = roi
    return y1 / h, y2 / h, x1 / w, x2 / w


def _candidate(
    roi,
    *,
    threshold_index,
    source="adaptive-dark",
    geometry_score=0.9,
    boundary_score=0.9,
):
    return _FilmCandidate(
        roi=roi,
        correction_angle=0.0,
        source=source,
        threshold_index=threshold_index,
        polarity="dark",
        boundary_score=boundary_score,
        geometry_score=geometry_score,
        supported_sides=frozenset({"top", "bottom", "left", "right"}),
        supported_corners=frozenset({"top_left", "top_right", "bottom_left", "bottom_right"}),
    )


def test_consensus_rejects_a_high_scoring_single_source_outlier():
    stable = [
        _candidate((79, 301, 99, 441), threshold_index=0),
        _candidate((80, 300, 100, 440), threshold_index=1),
        _candidate((81, 300, 101, 439), threshold_index=2),
        _candidate((80, 299, 100, 441), threshold_index=3),
        _candidate((79, 300, 100, 440), threshold_index=4),
    ]
    outlier = _candidate(
        (20, 340, 20, 520),
        threshold_index=None,
        source="edges",
        geometry_score=1.0,
        boundary_score=1.0,
    )

    consensus = _select_consensus_cluster([outlier, *stable], (360, 540))

    assert consensus is not None
    assert all(abs(actual - expected) <= 1 for actual, expected in zip(consensus.roi, (80, 300, 100, 440)))
    assert "adaptive-dark" in consensus.evidence_sources
    assert "edges" not in consensus.evidence_sources


def test_consensus_abstains_on_a_single_structural_mask():
    candidate = _candidate((80, 300, 100, 440), threshold_index=None, source="edges")
    assert _select_consensus_cluster([candidate], (360, 540)) is None


def test_detects_dark_film_on_light_bed_and_returns_immutable_evidence():
    detection = detect_film_bounds_with_confidence(_two_tier_scan())

    assert detection.roi is not None
    assert np.allclose(_normalized_roi(detection.roi, (360, 540)), (1 / 6, 5 / 6, 4 / 27, 23 / 27), atol=0.02)
    assert detection.confidence >= 0.55
    assert detection.geometry_score >= 0.5
    assert "adaptive-dark" in detection.evidence_sources
    assert detection.supported_sides == frozenset({"top", "bottom", "left", "right"})
    assert detection.vertical_edge_profile.shape == (540,)
    assert not detection.vertical_edge_profile.flags.writeable
    with pytest.raises(FrozenInstanceError):
        detection.confidence = 0.0


def test_detects_bright_film_in_dark_holder():
    detection = detect_film_bounds_with_confidence(_two_tier_scan(surround=0.04, film=0.72))

    assert detection.roi is not None
    assert np.allclose(_normalized_roi(detection.roi, (360, 540)), (1 / 6, 5 / 6, 4 / 27, 23 / 27), atol=0.02)
    assert detection.confidence >= 0.55
    assert "adaptive-bright" in detection.evidence_sources


def test_rejects_a_content_rectangle_in_a_borderless_scan():
    h, w = 360, 540
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    texture = 0.48 + 0.17 * np.sin(xx / 13.0) * np.cos(yy / 19.0)
    image = np.repeat(texture[..., None], 3, axis=2).astype(np.float32)
    image[75:285, 110:430] = 0.14

    detection = detect_film_bounds_with_confidence(image)

    assert detection.roi is None
    assert detection.confidence < 0.55


def test_correction_angle_has_the_sign_needed_by_fine_rotation():
    skewed = apply_fine_rotation(_two_tier_scan(), 4.0)

    detection = detect_film_bounds_with_confidence(skewed)

    assert detection.roi is not None
    assert detection.correction_angle == pytest.approx(-4.0, abs=0.8)
    corrected = apply_fine_rotation(skewed, detection.correction_angle)
    residual = detect_film_bounds_with_confidence(corrected)
    assert residual.roi is not None
    assert residual.correction_angle == pytest.approx(0.0, abs=0.5)


def test_borderless_texture_abstains():
    h, w = 360, 540
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    texture = 0.45 + 0.22 * np.sin(xx / 11.0) * np.cos(yy / 17.0)
    image = np.repeat(texture[..., None], 3, axis=2).astype(np.float32)

    detection = detect_film_bounds_with_confidence(image)

    assert detection.roi is None
    assert detection.supported_sides == frozenset()
    assert detection.supported_corners == frozenset()


def test_detection_is_stable_across_resolutions():
    small = _two_tier_scan(360, 540)
    large = _two_tier_scan(960, 1440)

    small_detection = detect_film_bounds_with_confidence(small)
    large_detection = detect_film_bounds_with_confidence(large)

    assert small_detection.roi is not None
    assert large_detection.roi is not None
    small_norm = _normalized_roi(small_detection.roi, small.shape)
    large_norm = _normalized_roi(large_detection.roi, large.shape)
    assert np.allclose(small_norm, large_norm, atol=0.01)
    assert abs(small_detection.confidence - large_detection.confidence) < 0.12
