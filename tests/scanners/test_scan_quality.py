"""Public scanner-quality contracts shared by acceptance and roll workflows."""

from dataclasses import replace

import cv2
import numpy as np
import tifffile

from negpy.infrastructure.scanners.result import SplitIrAlignment
from negpy.services.scanning.quality import (
    assess_stopped_transport_smear,
    file_sha256,
    inspect_tiff_payload,
    measure_focus_detail,
    measure_scan_clipping,
    split_alignment_metrics_confident,
)


def _textured_rgb_with_repeated_suffix(*, height: int = 96, width: int = 80, repeated_rows: int = 21) -> np.ndarray:
    rng = np.random.default_rng(20260712)
    rgb = rng.integers(1000, 64000, size=(height, width, 3), dtype=np.uint16)
    anchor = rng.integers(4000, 60000, size=(width, 3), dtype=np.uint16)
    rgb[-repeated_rows:] = anchor
    return rgb


def test_stopped_transport_assessment_detects_the_established_smear_signature() -> None:
    assessment = assess_stopped_transport_smear(
        _textured_rgb_with_repeated_suffix(),
        dpi=1000,
    )

    assert assessment.verdict == "smear"
    assert assessment.minimum_matches == 16
    assert assessment.start_row == 76
    assert assessment.suffix_rows == 20
    assert assessment.texture_span is not None
    assert assessment.texture_span > 2048


def test_uniform_colored_repeated_band_is_not_mistaken_for_spatial_texture() -> None:
    rng = np.random.default_rng(20260714)
    rgb = rng.integers(1000, 64000, size=(96, 80, 3), dtype=np.uint16)
    rgb[-21:] = np.array([10_000, 30_000, 50_000], dtype=np.uint16)

    assessment = assess_stopped_transport_smear(rgb, dpi=1000)

    assert assessment.verdict == "indeterminate"
    assert assessment.suffix_rows == 20
    assert assessment.texture_span == 0.0
    assert "lacks enough texture" in assessment.reason


def test_file_sha256_hashes_the_exact_payload(tmp_path) -> None:
    payload = tmp_path / "payload.bin"
    payload.write_bytes(b"NegPy scanner QC\x00\xff")

    assert file_sha256(payload) == "eee0bef6a63e83f133b24ab4120259da907ff1a5e7e0aa5d8fba41e1bcb56719"


def test_tiff_payload_inspection_reports_file_truth_including_actual_dpi(
    tmp_path,
) -> None:
    path = tmp_path / "frame.tif"
    image = np.arange(4 * 5 * 3, dtype=np.uint16).reshape(4, 5, 3)
    tifffile.imwrite(
        path,
        image,
        photometric="rgb",
        resolution=(4000, 4000),
        resolutionunit="INCH",
    )

    inspection = inspect_tiff_payload(path)

    assert inspection.shape == (4, 5, 3)
    assert inspection.dtype == "uint16"
    assert inspection.byte_length == path.stat().st_size
    assert inspection.sha256 == file_sha256(path)
    assert inspection.page_count == 1
    assert inspection.x_resolution == (4000, 1)
    assert inspection.y_resolution == (4000, 1)
    assert inspection.resolution_unit == "INCH"
    assert inspection.payload_within_file is True


def test_scan_clipping_reports_per_channel_sensor_white_fractions() -> None:
    rgb = np.full((64, 64, 3), 32768, dtype=np.uint16)
    rgb[:16, :, 0] = 65535

    telemetry = measure_scan_clipping(rgb)

    assert telemetry.fractions == (0.25, 0.0, 0.0)
    assert telemetry.clip_level == 0.99
    assert telemetry.warning_fraction == 0.01
    assert telemetry.warning is True


def test_scan_clipping_observes_pixels_in_every_sampling_phase() -> None:
    rgb = np.full((64, 64, 3), 32768, dtype=np.uint16)
    rgb[1::4, 2::4, 0] = 65535

    telemetry = measure_scan_clipping(rgb)

    assert telemetry.fractions == (0.0625, 0.0, 0.0)
    assert telemetry.warning is True


def test_scan_clipping_rejects_zero_area_rgb() -> None:
    rgb = np.empty((0, 64, 3), dtype=np.uint16)

    with np.testing.assert_raises_regex(ValueError, "RGB must have non-zero area"):
        measure_scan_clipping(rgb)


def test_scan_clipping_rejects_non_finite_float_rgb() -> None:
    rgb = np.full((16, 16, 3), 0.5, dtype=np.float32)
    rgb[5, 7, 1] = np.nan

    with np.testing.assert_raises_regex(ValueError, "float RGB must contain only finite values"):
        measure_scan_clipping(rgb)


def test_focus_detail_is_indeterminate_for_a_uniform_frame() -> None:
    rgb = np.full((128, 160, 3), 32000, dtype=np.uint16)

    telemetry = measure_focus_detail(rgb)

    assert telemetry.verdict == "indeterminate"
    assert telemetry.score is None
    assert telemetry.texture_span == 0.0


def test_focus_detail_scores_sharp_texture_above_its_blurred_copy() -> None:
    rng = np.random.default_rng(47)
    sharp = rng.integers(4000, 62000, size=(256, 320, 3), dtype=np.uint16)
    blurred = cv2.GaussianBlur(sharp, (0, 0), sigmaX=3.0, sigmaY=3.0)

    sharp_telemetry = measure_focus_detail(sharp)
    blurred_telemetry = measure_focus_detail(blurred)

    assert sharp_telemetry.verdict == "measured"
    assert blurred_telemetry.verdict == "measured"
    assert sharp_telemetry.score is not None
    assert blurred_telemetry.score is not None
    assert sharp_telemetry.score > blurred_telemetry.score


def test_split_alignment_metric_confidence_matches_the_recorded_capture_metrics() -> None:
    accepted_phase = SplitIrAlignment(
        mode="phase-ecc",
        dx_px=0.25,
        dy_px=-0.5,
        phase_responses=(0.10, 0.20, 0.30),
        channel_spread_px=1.0,
        ecc_coefficient=0.65,
    )
    identity = SplitIrAlignment(mode="identity", dx_px=0.0, dy_px=0.0)

    assert split_alignment_metrics_confident(accepted_phase) is True
    assert split_alignment_metrics_confident(identity) is True
    assert split_alignment_metrics_confident(None) is False
    assert (
        split_alignment_metrics_confident(
            SplitIrAlignment(
                mode="phase-ecc",
                dx_px=0.25,
                dy_px=-0.5,
                phase_responses=(0.099, 0.20, 0.30),
                channel_spread_px=1.0,
                ecc_coefficient=0.65,
            )
        )
        is False
    )
    assert (
        split_alignment_metrics_confident(
            SplitIrAlignment(
                mode="phase-ecc",
                dx_px=0.25,
                dy_px=-0.5,
                phase_responses=(0.10, 0.20, 0.30),
                channel_spread_px=1.001,
                ecc_coefficient=0.65,
            )
        )
        is False
    )
    assert (
        split_alignment_metrics_confident(
            SplitIrAlignment(
                mode="phase-ecc",
                dx_px=0.25,
                dy_px=-0.5,
                phase_responses=(0.10, 0.20, 0.30),
                channel_spread_px=1.0,
                ecc_coefficient=0.649,
            )
        )
        is False
    )
    assert (
        split_alignment_metrics_confident(
            SplitIrAlignment(
                mode="phase-ecc",
                dx_px=float("nan"),
                dy_px=-0.5,
                phase_responses=(0.10, 0.20, 0.30),
                channel_spread_px=1.0,
                ecc_coefficient=0.65,
            )
        )
        is False
    )
    assert split_alignment_metrics_confident(SplitIrAlignment(mode="identity", dx_px=0.01, dy_px=0.0)) is False


def test_split_alignment_metric_confidence_accepts_only_strict_near_zero_phase_only_evidence() -> None:
    accepted = SplitIrAlignment(
        mode="phase-only",
        dx_px=0.2,
        dy_px=-0.1,
        phase_responses=(0.52, 0.54, 0.55),
        channel_spread_px=0.03,
        ecc_coefficient=0.90,
    )

    assert split_alignment_metrics_confident(accepted, image_width=3946) is True
    assert split_alignment_metrics_confident(replace(accepted, phase_responses=(0.49, 0.54, 0.55)), image_width=3946) is False
    assert split_alignment_metrics_confident(replace(accepted, channel_spread_px=0.101), image_width=3946) is False
    assert split_alignment_metrics_confident(replace(accepted, dx_px=2.01), image_width=3946) is False


def test_split_alignment_metric_confidence_accepts_only_supported_tiled_phase_consensus() -> None:
    accepted = SplitIrAlignment(
        mode="tiled-phase",
        dx_px=-0.2,
        dy_px=0.1,
        phase_responses=(0.45, 0.46, 0.44),
        channel_spread_px=0.10,
        tile_support_counts=(8, 6, 8),
        tile_shift_spread_px=0.25,
    )

    assert split_alignment_metrics_confident(accepted, image_width=3946) is True
    assert split_alignment_metrics_confident(replace(accepted, tile_support_counts=(8, 5, 8)), image_width=3946) is False
    assert split_alignment_metrics_confident(replace(accepted, tile_shift_spread_px=0.351), image_width=3946) is False
    assert split_alignment_metrics_confident(replace(accepted, phase_responses=(0.39, 0.46, 0.44)), image_width=3946) is False


def test_split_alignment_metric_confidence_revalidates_multiscale_global_evidence() -> None:
    accepted = SplitIrAlignment(
        mode="multiscale-global",
        dx_px=0.10,
        dy_px=0.0,
        phase_responses=(0.45, 0.46, 0.44),
        channel_spread_px=0.02,
        estimator_version=2,
        multiscale_max_dimensions=(1024, 2048),
        multiscale_channel_shifts_px=(
            ((0.18, -0.06), (0.16, -0.04), (0.17, -0.05)),
            ((0.10, -0.05), (0.11, -0.04), (0.09, -0.06)),
        ),
        multiscale_responses=((0.35, 0.36, 0.34), (0.45, 0.46, 0.44)),
    )

    assert split_alignment_metrics_confident(accepted, image_width=3946) is True
    assert (
        split_alignment_metrics_confident(
            replace(accepted, multiscale_responses=((0.29, 0.36, 0.34), (0.45, 0.46, 0.44))),
            image_width=3946,
        )
        is False
    )


def test_split_alignment_metric_confidence_revalidates_multiscale_tiled_and_alias_evidence() -> None:
    accepted = SplitIrAlignment(
        mode="multiscale-tiled",
        dx_px=0.10,
        dy_px=0.0,
        phase_responses=(0.45, 0.46, 0.44),
        channel_spread_px=0.02,
        tile_support_counts=(8, 6, 7),
        tile_shift_spread_px=0.30,
        estimator_version=2,
        multiscale_max_dimensions=(1024, 2048),
        multiscale_channel_shifts_px=(
            ((0.18, -0.06), (0.16, -0.04), (0.17, -0.05)),
            ((0.10, -0.05), (0.11, -0.04), (0.09, -0.06)),
        ),
        multiscale_responses=((0.45, 0.46, 0.44), (0.45, 0.46, 0.44)),
        multiscale_tile_support_counts=((8, 7, 8), (8, 6, 7)),
        multiscale_tile_shift_spreads_px=((0.25, 0.30, 0.28), (0.20, 0.30, 0.25)),
        multiscale_global_alias_shifts_px=(
            ((-582.0, 0.1), (-581.5, -0.1), (0.1, 0.0)),
            ((-582.1, 0.0), (-581.7, -0.1), (0.0, 0.0)),
        ),
    )

    assert split_alignment_metrics_confident(accepted, image_width=3946) is True
    assert (
        split_alignment_metrics_confident(
            replace(accepted, multiscale_tile_support_counts=((8, 5, 8), (8, 6, 7))),
            image_width=3946,
        )
        is False
    )
    assert (
        split_alignment_metrics_confident(
            replace(accepted, multiscale_global_alias_shifts_px=()),
            image_width=3946,
        )
        is False
    )
    assert (
        split_alignment_metrics_confident(
            replace(
                accepted,
                multiscale_channel_shifts_px=(
                    ((0.40, -0.06), (0.39, -0.04), (0.41, -0.05)),
                    ((0.10, -0.05), (0.11, -0.04), (0.09, -0.06)),
                ),
            ),
            image_width=3946,
        )
        is False
    )
    numerically_snapped = replace(
        accepted,
        dx_px=0.0,
        dy_px=0.0,
        phase_responses=(0.45, 0.46, 0.44),
        channel_spread_px=0.02,
        multiscale_channel_shifts_px=(
            ((0.13, 0.00), (0.11, 0.01), (0.12, -0.01)),
            ((0.03, 0.00), (0.04, 0.01), (0.02, -0.01)),
        ),
    )
    assert split_alignment_metrics_confident(numerically_snapped, image_width=3946) is True


def test_split_alignment_metric_confidence_enforces_same_reservation_caps_without_width() -> None:
    alignment = SplitIrAlignment(
        mode="phase-ecc",
        dx_px=1_000_000.0,
        dy_px=-1_000_000.0,
        phase_responses=(0.20, 0.25, 0.30),
        channel_spread_px=0.5,
        ecc_coefficient=0.9,
    )

    assert split_alignment_metrics_confident(alignment) is False


def test_split_alignment_metric_confidence_with_width_enforces_production_shift_bound() -> None:
    def _alignment(dx_px: float, dy_px: float = 0.0) -> SplitIrAlignment:
        return SplitIrAlignment(
            mode="phase-ecc",
            dx_px=dx_px,
            dy_px=dy_px,
            phase_responses=(0.20, 0.25, 0.30),
            channel_spread_px=0.5,
            ecc_coefficient=0.9,
        )

    assert split_alignment_metrics_confident(_alignment(2.0), image_width=1000) is True
    assert split_alignment_metrics_confident(_alignment(2.01), image_width=1000) is False
    assert split_alignment_metrics_confident(_alignment(0.0, 8.0), image_width=1000) is True
    assert split_alignment_metrics_confident(_alignment(0.0, 8.01), image_width=1000) is False
    assert split_alignment_metrics_confident(_alignment(1_000_000.0), image_width=1000) is False
