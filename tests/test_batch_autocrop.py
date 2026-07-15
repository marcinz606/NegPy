import cv2
import numpy as np

from negpy.features.geometry.batch_autocrop import (
    AutocropCandidate,
    AutocropResult,
    BatchDetectParams,
    detect_autocrop_candidate,
    finalize_autocrop_batch,
    geometry_from_autocrop_result,
)
from negpy.features.geometry.models import GeometryConfig


def _candidate(
    source_id: str,
    roi: tuple[int, int, int, int],
    *,
    angle: float = 0.3,
    plausible: bool = True,
    top: bool = True,
    left: bool = True,
    right: bool = True,
    left_primary: bool = False,
    right_primary: bool = False,
) -> AutocropCandidate:
    return AutocropCandidate(
        source_id=source_id,
        roi=roi,
        canvas=(200, 300),
        angle=angle,
        plausible_geometry=plausible,
        top_found=top,
        left_found=left,
        right_found=right,
        left_primary=left_primary,
        right_primary=right_primary,
    )


def test_detect_candidate_accepts_in_memory_float_image() -> None:
    image = np.full((400, 600, 3), 0.5, dtype=np.float32)
    image[:30] = 0.0
    image[-25:] = 1.0
    image[:, :40] = 0.0
    image[:, -45:] = 0.0

    candidate = detect_autocrop_candidate(image, "scan", BatchDetectParams(frame_ratio=1.5))

    assert candidate.source_id == "scan"
    assert candidate.canvas == (400, 600)
    assert candidate.plausible_geometry
    assert candidate.top_found
    assert candidate.left_found
    assert candidate.right_found


def test_detect_candidate_estimates_rotation() -> None:
    image = np.full((400, 600, 3), 0.5, dtype=np.float32)
    image[:30] = 0.0
    image[-25:] = 1.0
    image[:, :40] = 0.0
    image[:, -45:] = 0.0
    matrix = cv2.getRotationMatrix2D((300, 200), -2.0, 1.0)
    tilted = cv2.warpAffine(image, matrix, (600, 400), borderMode=cv2.BORDER_REPLICATE)

    candidate = detect_autocrop_candidate(tilted, "scan")

    assert abs(candidate.angle - 2.0) <= 0.15


def test_batch_preserves_one_corner_and_unsupported_image() -> None:
    reference = _candidate("reference", (14, 184, 22, 279))
    one_corner = _candidate("one-corner", (14, 184, 24, 300), right=False)
    unsupported = _candidate(
        "unsupported",
        (0, 200, 0, 300),
        angle=0.0,
        plausible=False,
        top=False,
        left=False,
        right=False,
    )

    results = finalize_autocrop_batch([reference, one_corner, unsupported])

    assert results["reference"].method == "two-corner"
    assert results["one-corner"].method == "one-corner"
    assert results["one-corner"].roi[2] == 22
    assert results["unsupported"].method == "manual"
    assert results["unsupported"].roi == (0, 200, 0, 300)


def test_batch_rejects_width_and_angle_outliers() -> None:
    references = [
        _candidate(f"reference-{index}", (14, 184, 20, 20 + width), angle=angle)
        for index, (width, angle) in enumerate(((255, 0.2), (257, 0.3), (259, 0.4)))
    ]
    outlier = _candidate("outlier", (14, 184, 4, 296), angle=-0.5)

    result = finalize_autocrop_batch([*references, outlier])["outlier"]

    assert result.method == "batch-template"
    assert result.roi[3] - result.roi[2] < 292
    assert result.angle == 0.3


def test_batch_uses_only_primary_corner_for_false_edge() -> None:
    references = [
        _candidate(
            f"reference-{index}",
            (14, 184, 20, 20 + width),
            left_primary=True,
            right_primary=True,
        )
        for index, width in enumerate((255, 257, 259))
    ]
    false_right = _candidate(
        "false-right",
        (14, 184, 12, 296),
        left_primary=True,
        right_primary=False,
    )

    result = finalize_autocrop_batch([*references, false_right])["false-right"]

    assert result.method == "one-corner"
    assert result.roi[2] == 10


def test_batch_keeps_equal_outward_safety_margin() -> None:
    candidate = _candidate("reference", (14, 184, 22, 279))

    exact = finalize_autocrop_batch(
        [candidate],
        BatchDetectParams(outward_frac=0),
    )["reference"]
    padded = finalize_autocrop_batch([candidate])["reference"]

    assert padded.roi == (
        exact.roi[0] - 2,
        exact.roi[1] + 2,
        exact.roi[2] - 2,
        exact.roi[3] + 2,
    )


def test_result_becomes_persistent_manual_crop_and_rotation() -> None:
    geometry = GeometryConfig(fine_rotation=0.5, auto_crop_enabled=True)
    result = AutocropResult((20, 180, 30, 270), 1.25, "two-corner")

    updated = geometry_from_autocrop_result(geometry, result, (200, 300))

    assert updated is not None
    assert updated.fine_rotation == 1.75
    assert updated.manual_crop_rect == (0.1, 0.1, 0.9, 0.9)
    assert not updated.auto_crop_enabled


def test_manual_result_leaves_geometry_unchanged() -> None:
    result = AutocropResult((0, 200, 0, 300), 0.0, "manual")

    assert geometry_from_autocrop_result(GeometryConfig(), result, (200, 300)) is None
