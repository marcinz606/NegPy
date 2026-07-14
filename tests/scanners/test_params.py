"""Tests for ScanParams dataclass and ScanMode validation."""

import math

import pytest
from negpy.infrastructure.scanners.params import RegisteredScanGeometry, ScanMode, ScanParams, parse_registration_manifest
from negpy.infrastructure.scanners.base import ScannerCapabilities


class TestScanMode:
    def test_enum_values(self) -> None:
        assert ScanMode.NEGATIVE.value == "Negative"
        assert ScanMode.POSITIVE.value == "Positive"
        assert ScanMode.TRANSPARENCY.value == "Transparency"

    def test_from_value(self) -> None:
        assert ScanMode("Negative") == ScanMode.NEGATIVE
        assert ScanMode("Positive") == ScanMode.POSITIVE


class TestScanParams:
    def test_default_construction(self) -> None:
        params = ScanParams(dpi=3600, depth=16, capture_ir=False)
        assert params.dpi == 3600
        assert params.depth == 16
        assert params.capture_ir is False
        assert params.area is None
        assert params.auto_exposure is False
        assert params.frame is None
        assert params.registered_geometry is None

    def test_with_area(self) -> None:
        params = ScanParams(dpi=2400, depth=8, capture_ir=True, area=(0, 0, 36, 24))
        assert params.area == (0, 0, 36, 24)

    def test_registered_geometry_keeps_position_and_window_typed_together(self) -> None:
        geometry = RegisteredScanGeometry(frame=3, subframe_mm=6.35, br_y_device_px=5003)
        current_position = RegisteredScanGeometry(subframe_mm=6.35, br_y_device_px=5003)
        params = ScanParams(dpi=4000, depth=16, capture_ir=True, registered_geometry=geometry)

        assert params.registered_geometry == geometry
        assert geometry.frame == 3
        assert geometry.subframe_mm == 6.35
        assert geometry.br_y_device_px == 5003
        assert current_position.frame is None

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"frame": 0, "subframe_mm": 1.0, "br_y_device_px": 100},
            {"frame": 1, "subframe_mm": -0.01, "br_y_device_px": 100},
            {"frame": 1, "subframe_mm": math.nan, "br_y_device_px": 100},
            {"frame": 1, "subframe_mm": 1.0, "br_y_device_px": -1},
            {"frame": 1, "subframe_mm": 1.0, "br_y_device_px": True},
        ],
    )
    def test_registered_geometry_rejects_impossible_values(self, kwargs) -> None:
        with pytest.raises(ValueError):
            RegisteredScanGeometry(**kwargs)

    def test_frozen(self) -> None:
        params = ScanParams(dpi=1200, depth=16, capture_ir=False)
        with pytest.raises(Exception):
            params.dpi = 2400  # type: ignore[misc]


class TestParseRegistrationManifest:
    """Schema matches roll_scan_runner's --registration-json. Numeric/geometric
    validation is intentionally delegated to RegisteredScanGeometry.__post_init__
    rather than duplicated here."""

    def test_extracts_matching_frame_entry(self) -> None:
        data = {"frames": [{"frame": 3, "subframe_mm": 6.35, "br_y": 5003}]}
        geometry = parse_registration_manifest(data, frame=3)
        assert geometry == RegisteredScanGeometry(subframe_mm=6.35, br_y_device_px=5003, frame=3)

    def test_picks_correct_entry_among_several(self) -> None:
        data = {
            "frames": [
                {"frame": 1, "subframe_mm": 1.0, "br_y": 100},
                {"frame": 3, "subframe_mm": 6.35, "br_y": 5003},
            ]
        }
        geometry = parse_registration_manifest(data, frame=3)
        assert geometry.subframe_mm == 6.35
        assert geometry.br_y_device_px == 5003

    def test_missing_frames_key_raises(self) -> None:
        with pytest.raises(ValueError, match="frames"):
            parse_registration_manifest({}, frame=1)

    def test_non_dict_payload_raises(self) -> None:
        with pytest.raises(ValueError):
            parse_registration_manifest([{"frame": 1}], frame=1)

    def test_no_matching_frame_raises(self) -> None:
        data = {"frames": [{"frame": 1, "subframe_mm": 1.0, "br_y": 100}]}
        with pytest.raises(ValueError, match="no entry for frame 3"):
            parse_registration_manifest(data, frame=3)

    def test_duplicate_frame_entries_raise(self) -> None:
        data = {
            "frames": [
                {"frame": 3, "subframe_mm": 6.35, "br_y": 5003},
                {"frame": 3, "subframe_mm": 6.4, "br_y": 5010},
            ]
        }
        with pytest.raises(ValueError, match="duplicate"):
            parse_registration_manifest(data, frame=3)

    def test_missing_subframe_mm_raises(self) -> None:
        data = {"frames": [{"frame": 3, "br_y": 5003}]}
        with pytest.raises(ValueError, match="subframe_mm"):
            parse_registration_manifest(data, frame=3)

    def test_missing_br_y_raises(self) -> None:
        data = {"frames": [{"frame": 3, "subframe_mm": 6.35}]}
        with pytest.raises(ValueError, match="br_y"):
            parse_registration_manifest(data, frame=3)

    def test_bool_frame_does_not_masquerade_as_int_frame(self) -> None:
        # JSON `true`/`false` decode to Python bool, and True == 1 / False == 0 —
        # type(...) is int (not isinstance) keeps a boolean entry from matching.
        data = {"frames": [{"frame": True, "subframe_mm": 6.35, "br_y": 5003}]}
        with pytest.raises(ValueError, match="no entry for frame 1"):
            parse_registration_manifest(data, frame=1)

    def test_invalid_geometry_values_delegate_to_dataclass_validation(self) -> None:
        data = {"frames": [{"frame": 3, "subframe_mm": -1.0, "br_y": 5003}]}
        with pytest.raises(ValueError):
            parse_registration_manifest(data, frame=3)


class TestCapabilityFiltering:
    def test_sources_filtered_by_caps(self) -> None:
        caps = ScannerCapabilities(
            ir_channel=False,
            supported_dpi=(300, 600, 1200, 2400),
            supported_depths=(8, 16),
            sources=(ScanMode.NEGATIVE, ScanMode.POSITIVE),
            max_area_mm=(36.0, 25.0),
        )
        assert ScanMode.TRANSPARENCY not in caps.sources
        assert ScanMode.NEGATIVE in caps.sources
        assert len(caps.sources) == 2

    def test_empty_sources_means_no_film(self) -> None:
        caps = ScannerCapabilities(
            ir_channel=False,
            supported_dpi=(),
            supported_depths=(),
            sources=(),
            max_area_mm=(0, 0),
        )
        assert len(caps.sources) == 0

    def test_dpi_range_from_caps(self) -> None:
        caps = ScannerCapabilities(
            ir_channel=True,
            supported_dpi=(300, 600, 1200, 2400, 3600),
            supported_depths=(16,),
            sources=(ScanMode.NEGATIVE, ScanMode.TRANSPARENCY),
            max_area_mm=(36, 25),
        )
        assert 3600 in caps.supported_dpi
        assert 75 not in caps.supported_dpi
