"""Tests for developer recipes, scan setups and the Process & Scan preset."""

import os

import pytest

from negpy.features.metadata.gear_logic import gear_search_text, matches_gear_filter, metadata_from_gear
from negpy.features.metadata.gear_models import (
    DeveloperRecipe,
    DevProcess,
    GearLibrary,
    GearPreset,
    ProcessScanPreset,
    ScanMethod,
    ScanSetup,
)
from negpy.features.metadata.models import MetadataConfig
from negpy.features.metadata.payload import build_metadata_payload, has_capture_gear
from negpy.services.assets.gear import GearProfiles


@pytest.fixture
def gear_dir(tmp_path, monkeypatch):
    monkeypatch.setattr("negpy.services.assets.gear.APP_CONFIG.gear_dir", str(tmp_path))
    monkeypatch.setattr("negpy.services.assets.gear.get_resource_path", lambda _: str(tmp_path / "_no_bundled"))
    return str(tmp_path)


@pytest.fixture
def library():
    return GearLibrary(
        developers=[
            DeveloperRecipe(id="d1", developer="D-76", dilution="1+1", time="9:30", temperature_c=20),
            DeveloperRecipe(id="d2", developer="HC-110", dilution="dil. B", time="6:00"),
        ],
        scan_setups=[
            ScanSetup(id="s1", device="Sony A7RIV", light_source="Scanlight narrowband"),
            ScanSetup(id="s2", method=ScanMethod.FLATBED, device="Epson V850"),
        ],
        process_scan_presets=[ProcessScanPreset(id="ps1", display_name="Home B&W", developer_id="d1", scan_setup_id="s1")],
    )


def test_developer_label_renders_chemistry_and_conditions():
    recipe = DeveloperRecipe(developer="D-76", dilution="1+1", time="9:30", temperature_c=20)

    assert recipe.full_developer_label == "D-76 1+1, 9:30 @ 20 °C"
    assert recipe.resolved_display_name == "D-76 1+1"


def test_developer_label_omits_missing_parts():
    assert DeveloperRecipe(developer="HC-110", dilution="dil. B").full_developer_label == "HC-110 dil. B"
    assert DeveloperRecipe(developer="C-41", lab="Carmencita").full_developer_label == "C-41 (Carmencita)"
    assert DeveloperRecipe(developer="Rodinal", time="13:00").full_developer_label == "Rodinal, 13:00"


def test_scan_label_renders_method_and_rig():
    rig = ScanSetup(device="Sony A7RIV", light_source="Scanlight narrowband")

    assert rig.full_scan_label == "Copy stand — Sony A7RIV, Scanlight narrowband"
    assert ScanSetup(method=ScanMethod.FLATBED).full_scan_label == "Flatbed"


def test_enums_fall_back_to_other_on_unknown_storage_value():
    assert DevProcess.from_storage("C41") is DevProcess.C41
    assert DevProcess.from_storage("Caffenol") is DevProcess.OTHER
    assert ScanMethod.from_storage("CopyStand") is ScanMethod.COPY_STAND
    assert ScanMethod.from_storage("Contact print") is ScanMethod.OTHER


def test_round_trip_through_json(gear_dir):
    saved = GearLibrary(
        developers=[DeveloperRecipe(id="d1", developer="XTOL", dilution="1+1", time="9:45", temperature_c=20)],
        scan_setups=[ScanSetup(id="s1", method=ScanMethod.DRUM, device="Howtek 4500")],
        process_scan_presets=[ProcessScanPreset(id="ps1", display_name="Archive", developer_id="d1", scan_setup_id="s1")],
    )

    GearProfiles.save_library(saved)
    loaded = GearProfiles.load_library()

    assert os.path.isfile(os.path.join(gear_dir, "developers.json"))
    assert loaded.developers[0].full_developer_label == "XTOL 1+1, 9:45 @ 20 °C"
    assert loaded.scan_setups[0].method is ScanMethod.DRUM
    assert loaded.process_scan_presets[0].developer_id == "d1"


def test_selecting_a_developer_fills_the_text_field(library):
    applied = metadata_from_gear(MetadataConfig(), library, developer_id="d1")

    assert applied.developer_id == "d1"
    assert applied.developer == "D-76 1+1, 9:30 @ 20 °C"


def test_selecting_a_scan_setup_fills_the_text_field(library):
    applied = metadata_from_gear(MetadataConfig(), library, scan_setup_id="s2")

    assert applied.scan_setup_id == "s2"
    assert applied.scanning == "Flatbed — Epson V850"


def test_process_scan_preset_sets_both_slots(library):
    applied = metadata_from_gear(MetadataConfig(), library, process_scan_preset_id="ps1")

    assert applied.developer_id == "d1"
    assert applied.scan_setup_id == "s1"
    assert applied.developer == "D-76 1+1, 9:30 @ 20 °C"
    assert applied.scanning == "Copy stand — Sony A7RIV, Scanlight narrowband"


def test_gear_preset_developer_slot_is_additive(library):
    """An empty gear-preset slot must not wipe a developer picked by hand."""
    library.gear_presets = [GearPreset(id="p1", display_name="Bodies only")]
    picked = metadata_from_gear(MetadataConfig(), library, developer_id="d1")

    applied = metadata_from_gear(picked, library, gear_preset_id="p1")

    assert applied.developer_id == "d1"
    assert applied.developer == "D-76 1+1, 9:30 @ 20 °C"


def test_gear_preset_slot_applies_when_set(library):
    library.gear_presets = [GearPreset(id="p1", display_name="Full kit", developer_id="d2", scan_setup_id="s2")]

    applied = metadata_from_gear(MetadataConfig(), library, gear_preset_id="p1")

    assert applied.developer == "HC-110 dil. B, 6:00"
    assert applied.scanning == "Flatbed — Epson V850"


def test_process_scan_preset_is_authoritative_over_its_slots(library):
    library.process_scan_presets = [ProcessScanPreset(id="ps2", display_name="Dev only", developer_id="d2")]
    picked = metadata_from_gear(MetadataConfig(), library, scan_setup_id="s1")

    applied = metadata_from_gear(picked, library, process_scan_preset_id="ps2")

    assert applied.developer_id == "d2"
    assert applied.scan_setup_id == ""


def test_typed_override_survives_an_unrelated_gear_change(library):
    picked = metadata_from_gear(MetadataConfig(), library, developer_id="d1")
    edited = picked.__class__(**{**picked.__dict__, "developer": "D-76 1+1, stand 20 min"})

    applied = metadata_from_gear(edited, library, camera_id="")

    assert applied.developer == "D-76 1+1, stand 20 min"


def test_empty_id_leaves_the_text_alone(library):
    config = MetadataConfig(developer="Caffenol-C", scanning="Phone snap")

    applied = metadata_from_gear(config, library, developer_id="", scan_setup_id="")

    assert applied.developer == "Caffenol-C"
    assert applied.scanning == "Phone snap"


def test_search_text_covers_the_new_item_types(library):
    recipe = library.developers[0]
    rig = library.scan_setups[0]

    assert matches_gear_filter(recipe, "d-76")
    assert matches_gear_filter(recipe, "20")
    assert matches_gear_filter(rig, "scanlight")
    assert matches_gear_filter(rig, "copy stand")
    assert not matches_gear_filter(rig, "drum")


def test_preset_search_text_includes_linked_labels(library):
    text = gear_search_text(library.process_scan_presets[0], library)

    assert "d-76" in text
    assert "sony a7riv" in text


def test_chemistry_alone_does_not_claim_the_scanner_exif(library):
    """Developer and rig are not capture gear, so they must not strip scan EXIF."""
    config = metadata_from_gear(MetadataConfig(), library, process_scan_preset_id="ps1")

    assert has_capture_gear(config) is False

    payload = build_metadata_payload(config, library)

    assert payload.exif_flags.strip_scan_residuals is False
    assert payload.developer == "D-76 1+1, 9:30 @ 20 °C"
    assert payload.scan_method == "Copy stand — Sony A7RIV, Scanlight narrowband"
