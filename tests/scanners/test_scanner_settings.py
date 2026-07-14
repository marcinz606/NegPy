"""Persistence contracts for scanner hardware and capture preferences."""

from dataclasses import asdict

from negpy.infrastructure.scanners.settings import ScannerSettings


def test_sa30_profile_is_default_off_for_legacy_settings_and_round_trips() -> None:
    legacy_payload = {
        "last_device_id": "coolscan3:usb:libusb:001:007",
        "dpi": 4_000,
        "depth": 16,
        "capture_ir": True,
        "autofocus": True,
        "samples_per_scan": 4,
        "auto_exposure": True,
        "archival_split_capture": False,
        "output_folder": "/tmp/scans",
        "output_format": "TIFF",
        "filename_pattern": 'roll_{{ "%03d" % seq }}',
    }

    migrated = ScannerSettings(**legacy_payload)
    opted_in = ScannerSettings(**legacy_payload, sa30_compatible_roll_feeder=True)
    reloaded = ScannerSettings(**asdict(opted_in))

    assert migrated.sa30_compatible_roll_feeder is False
    assert migrated.preview_meter_inset_percent == 10
    assert asdict(opted_in)["sa30_compatible_roll_feeder"] is True
    assert reloaded == opted_in


def test_malformed_sa30_profile_values_fail_closed_without_losing_other_settings() -> None:
    for malformed in (0, 1, "false", "true"):
        settings = ScannerSettings(
            dpi=4_000,
            output_folder="/tmp/scans",
            sa30_compatible_roll_feeder=malformed,  # type: ignore[arg-type]
        )

        assert settings.sa30_compatible_roll_feeder is False
        assert settings.dpi == 4_000
        assert settings.output_folder == "/tmp/scans"


def test_preview_meter_inset_clamps_integer_bounds_and_defaults_malformed_values() -> None:
    cases = (
        (-5, 0),
        (31, 30),
        (100, 30),
        (True, 10),
        (10.0, 10),
        ("20", 10),
        (None, 10),
    )

    for persisted, expected in cases:
        settings = ScannerSettings(
            dpi=4_000,
            output_folder="/tmp/scans",
            preview_meter_inset_percent=persisted,  # type: ignore[arg-type]
        )

        assert settings.preview_meter_inset_percent == expected
        assert settings.dpi == 4_000
        assert settings.output_folder == "/tmp/scans"


def test_persisted_archival_split_recipe_always_restores_at_16_bit() -> None:
    settings = ScannerSettings(
        depth=8,
        capture_ir=True,
        samples_per_scan=4,
        archival_split_capture=True,
    )

    assert settings.depth == 16


def test_malformed_archival_split_values_fail_closed() -> None:
    for malformed in (0, 1, "false", "true", None):
        settings = ScannerSettings(
            depth=8,
            archival_split_capture=malformed,  # type: ignore[arg-type]
        )

        assert settings.archival_split_capture is False
        assert settings.depth == 8
