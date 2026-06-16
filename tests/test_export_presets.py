"""Tests for export preset serialization, persistence, and worker path resolution."""
import os
import uuid
import pytest

from negpy.domain.models import (
    ExportFormat,
    ExportPreset,
    ExportPresetOutputMode,
    ExportResolutionMode,
    AspectRatio,
    ColorSpace,
)
from negpy.infrastructure.storage.repository import StorageRepository


# ---------------------------------------------------------------------------
# ExportPreset serialization
# ---------------------------------------------------------------------------

def _make_preset(**kwargs) -> ExportPreset:
    defaults = dict(
        id=str(uuid.uuid4()),
        name="Test Preset",
        enabled=True,
        export_fmt=ExportFormat.TIFF,
        jpeg_quality=90,
        export_resolution_mode=ExportResolutionMode.ORIGINAL.value,
        paper_aspect_ratio=AspectRatio.ORIGINAL,
        export_print_size=30.0,
        export_dpi=300,
        export_target_long_edge_px=2000,
        output_mode=ExportPresetOutputMode.SAME_AS_SOURCE,
        output_subfolder="",
        output_path="",
        overwrite=True,
        filename_pattern="{{ original_name }}",
        export_color_space=ColorSpace.ADOBE_RGB.value,
        icc_input_path=None,
        icc_output_path=None,
    )
    defaults.update(kwargs)
    return ExportPreset(**defaults)


def test_preset_round_trip_tiff():
    p = _make_preset(name="TIFF Archive", export_fmt=ExportFormat.TIFF)
    p2 = ExportPreset.from_dict(p.to_dict())
    assert p2.name == "TIFF Archive"
    assert p2.export_fmt == ExportFormat.TIFF
    assert p2.id == p.id


def test_preset_round_trip_jpeg():
    p = _make_preset(name="JPEG Preview", export_fmt=ExportFormat.JPEG, jpeg_quality=75)
    p2 = ExportPreset.from_dict(p.to_dict())
    assert p2.export_fmt == ExportFormat.JPEG
    assert p2.jpeg_quality == 75


def test_preset_round_trip_png():
    p = _make_preset(name="PNG Full Size", export_fmt=ExportFormat.PNG)
    p2 = ExportPreset.from_dict(p.to_dict())
    assert p2.export_fmt == ExportFormat.PNG


def test_preset_subfolder_of_source():
    p = _make_preset(
        output_mode=ExportPresetOutputMode.SUBFOLDER_OF_SOURCE,
        output_subfolder="TIFF",
    )
    p2 = ExportPreset.from_dict(p.to_dict())
    assert p2.output_mode == ExportPresetOutputMode.SUBFOLDER_OF_SOURCE
    assert p2.output_subfolder == "TIFF"


def test_preset_absolute_path():
    p = _make_preset(
        output_mode=ExportPresetOutputMode.ABSOLUTE,
        output_path="/some/export/path",
    )
    p2 = ExportPreset.from_dict(p.to_dict())
    assert p2.output_mode == ExportPresetOutputMode.ABSOLUTE
    assert p2.output_path == "/some/export/path"


def test_preset_unknown_keys_dropped():
    d = _make_preset().to_dict()
    d["unknown_future_field"] = "should be ignored"
    p = ExportPreset.from_dict(d)
    assert not hasattr(p, "unknown_future_field")


# ---------------------------------------------------------------------------
# ExportFormat enum
# ---------------------------------------------------------------------------

def test_export_format_png_exists():
    assert ExportFormat.PNG == "PNG"
    assert ExportFormat.TIFF == "TIFF"
    assert ExportFormat.JPEG == "JPEG"


# ---------------------------------------------------------------------------
# Output path resolution (mirroring worker logic)
# ---------------------------------------------------------------------------

def _resolve_output_dir(preset: ExportPreset, source_path: str) -> str:
    source_dir = os.path.dirname(source_path)
    if preset.output_mode == ExportPresetOutputMode.SUBFOLDER_OF_SOURCE:
        subfolder = preset.output_subfolder or ""
        return os.path.join(source_dir, subfolder) if subfolder else source_dir
    elif preset.output_mode == ExportPresetOutputMode.ABSOLUTE:
        return preset.output_path or source_dir
    else:
        return source_dir


def test_output_dir_same_as_source():
    p = _make_preset(output_mode=ExportPresetOutputMode.SAME_AS_SOURCE)
    source = "/photos/roll/IMG_001.RAF"
    assert _resolve_output_dir(p, source) == "/photos/roll"


def test_output_dir_subfolder_of_source():
    p = _make_preset(
        output_mode=ExportPresetOutputMode.SUBFOLDER_OF_SOURCE,
        output_subfolder="TIFF",
    )
    source = "/photos/roll/IMG_001.RAF"
    assert _resolve_output_dir(p, source) == "/photos/roll/TIFF"


def test_output_dir_subfolder_empty_falls_back_to_source():
    p = _make_preset(
        output_mode=ExportPresetOutputMode.SUBFOLDER_OF_SOURCE,
        output_subfolder="",
    )
    source = "/photos/roll/IMG_001.RAF"
    assert _resolve_output_dir(p, source) == "/photos/roll"


def test_output_dir_absolute():
    p = _make_preset(
        output_mode=ExportPresetOutputMode.ABSOLUTE,
        output_path="/mnt/export/archive",
    )
    source = "/photos/roll/IMG_001.RAF"
    assert _resolve_output_dir(p, source) == "/mnt/export/archive"


# ---------------------------------------------------------------------------
# Extension mapping
# ---------------------------------------------------------------------------

_EXT_MAP = {ExportFormat.JPEG: "jpg", ExportFormat.TIFF: "tiff", ExportFormat.PNG: "png"}


def test_extension_jpeg():
    assert _EXT_MAP[ExportFormat.JPEG] == "jpg"


def test_extension_tiff():
    assert _EXT_MAP[ExportFormat.TIFF] == "tiff"


def test_extension_png():
    assert _EXT_MAP[ExportFormat.PNG] == "png"


# ---------------------------------------------------------------------------
# Repository persistence
# ---------------------------------------------------------------------------

@pytest.fixture()
def repo(tmp_path):
    edits_db = str(tmp_path / "edits.db")
    settings_db = str(tmp_path / "settings.db")
    r = StorageRepository(edits_db, settings_db)
    r.initialize()
    return r


def test_save_and_load_presets(repo):
    presets = [
        _make_preset(name="TIFF Archive", export_fmt=ExportFormat.TIFF),
        _make_preset(name="PNG Preview", export_fmt=ExportFormat.PNG, enabled=False),
    ]
    repo.save_export_presets(presets)
    loaded = repo.load_export_presets()
    assert len(loaded) == 2
    assert loaded[0].name == "TIFF Archive"
    assert loaded[0].export_fmt == ExportFormat.TIFF
    assert loaded[1].name == "PNG Preview"
    assert loaded[1].enabled is False


def test_load_presets_empty(repo):
    assert repo.load_export_presets() == []


def test_save_empty_presets_clears(repo):
    repo.save_export_presets([_make_preset(name="Old")])
    repo.save_export_presets([])
    assert repo.load_export_presets() == []


def test_preset_order_preserved(repo):
    presets = [_make_preset(name=f"Preset {i}") for i in range(5)]
    repo.save_export_presets(presets)
    loaded = repo.load_export_presets()
    assert [p.name for p in loaded] == [f"Preset {i}" for i in range(5)]
