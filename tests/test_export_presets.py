"""Tests for export preset serialization, persistence, and format encoding."""

import io
import os
import uuid

import numpy as np
import pytest
import tifffile
from PIL import Image

from negpy.domain.models import (
    ExportConfig,
    ExportFormat,
    ExportPreset,
    ExportPresetOutputMode,
    ExportResolutionMode,
    AspectRatio,
    ColorSpace,
    WorkspaceConfig,
    preset_from_export_config,
)
from negpy.infrastructure.display.color_spaces import WORKING_COLOR_SPACE
from negpy.infrastructure.storage.repository import StorageRepository
from negpy.services.rendering.image_processor import ImageProcessor


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
# ExportConfig -> ExportPreset passthrough
# ---------------------------------------------------------------------------


def test_preset_from_export_config_passthrough():
    conf = ExportConfig(
        export_fmt=ExportFormat.JPEG,
        jpeg_quality=72,
        output_mode=ExportPresetOutputMode.SUBFOLDER_OF_SOURCE,
        output_subfolder="web",
        export_path="/abs/out",
    )
    preset = preset_from_export_config(conf)
    assert preset.jpeg_quality == 72
    assert preset.output_mode == ExportPresetOutputMode.SUBFOLDER_OF_SOURCE
    assert preset.output_subfolder == "web"
    # ExportConfig's absolute path maps to the preset's output_path.
    assert preset.output_path == "/abs/out"


# ---------------------------------------------------------------------------
# from_flat_dict back-compat: same_as_source -> output_mode
# ---------------------------------------------------------------------------


def test_from_flat_dict_same_as_source_true_maps_to_same():
    data = WorkspaceConfig().to_dict()
    data.pop("output_mode", None)
    data["same_as_source"] = True
    cfg = WorkspaceConfig.from_flat_dict(data)
    assert cfg.export.output_mode == ExportPresetOutputMode.SAME_AS_SOURCE


def test_from_flat_dict_same_as_source_false_maps_to_absolute():
    data = WorkspaceConfig().to_dict()
    data.pop("output_mode", None)
    data["same_as_source"] = False
    cfg = WorkspaceConfig.from_flat_dict(data)
    assert cfg.export.output_mode == ExportPresetOutputMode.ABSOLUTE


def test_from_flat_dict_output_mode_wins_over_legacy_flag():
    data = WorkspaceConfig().to_dict()
    data["output_mode"] = ExportPresetOutputMode.SUBFOLDER_OF_SOURCE
    data["same_as_source"] = True  # legacy flag ignored when output_mode present
    cfg = WorkspaceConfig.from_flat_dict(data)
    assert cfg.export.output_mode == ExportPresetOutputMode.SUBFOLDER_OF_SOURCE


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


def test_load_presets_defaults_when_unset(repo):
    # A fresh repo (never saved) ships starter JPEG/TIFF/PNG presets.
    loaded = repo.load_export_presets()
    assert [p.name for p in loaded] == ["JPEG", "TIFF", "PNG"]
    assert [p.export_fmt for p in loaded] == [ExportFormat.JPEG, ExportFormat.TIFF, ExportFormat.PNG]
    assert loaded[0].enabled is True


def test_save_empty_presets_clears(repo):
    repo.save_export_presets([_make_preset(name="Old")])
    repo.save_export_presets([])
    assert repo.load_export_presets() == []


def test_preset_order_preserved(repo):
    presets = [_make_preset(name=f"Preset {i}") for i in range(5)]
    repo.save_export_presets(presets)
    loaded = repo.load_export_presets()
    assert [p.name for p in loaded] == [f"Preset {i}" for i in range(5)]


# ---------------------------------------------------------------------------
# Format encoding (real bytes for every format) — guards the PNG RGB crash
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def proc():
    return ImageProcessor()


def _rgb_buffer(h=8, w=12):
    # A simple gradient well inside every gamut so color management never clips.
    x = np.linspace(0.2, 0.8, w, dtype=np.float32)
    buf = np.stack([np.tile(x, (h, 1))] * 3, axis=-1)
    return np.ascontiguousarray(buf)


def test_encode_png_rgb_produces_valid_image(proc):
    """PNG export of a color image must not crash and must round-trip as RGB."""
    buf = _rgb_buffer()
    preset = _make_preset(export_fmt=ExportFormat.PNG)
    data, ext = proc._encode_export(buf, preset, ColorSpace.ADOBE_RGB.value, WORKING_COLOR_SPACE)
    assert ext == "png"
    img = Image.open(io.BytesIO(data))
    assert img.mode == "RGB"
    assert img.size == (buf.shape[1], buf.shape[0])


def test_encode_png_greyscale_keeps_16bit(proc):
    buf = _rgb_buffer()
    preset = _make_preset(export_fmt=ExportFormat.PNG)
    data, ext = proc._encode_export(buf, preset, ColorSpace.GREYSCALE.value, WORKING_COLOR_SPACE)
    assert ext == "png"
    img = Image.open(io.BytesIO(data))
    assert img.mode.startswith("I")  # 16-bit greyscale
    assert img.size == (buf.shape[1], buf.shape[0])


def test_encode_tiff_rgb_is_16bit(proc):
    buf = _rgb_buffer()
    preset = _make_preset(export_fmt=ExportFormat.TIFF)
    data, ext = proc._encode_export(buf, preset, ColorSpace.ADOBE_RGB.value, WORKING_COLOR_SPACE)
    assert ext == "tiff"
    arr = tifffile.imread(io.BytesIO(data))
    assert arr.dtype == np.uint16
    assert arr.shape == (buf.shape[0], buf.shape[1], 3)


def test_encode_jpeg_rgb(proc):
    buf = _rgb_buffer()
    preset = _make_preset(export_fmt=ExportFormat.JPEG)
    data, ext = proc._encode_export(buf, preset, ColorSpace.ADOBE_RGB.value, WORKING_COLOR_SPACE)
    assert ext == "jpg"
    img = Image.open(io.BytesIO(data))
    assert img.format == "JPEG"
    assert img.size == (buf.shape[1], buf.shape[0])
