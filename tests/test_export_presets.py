"""Tests for export preset serialization, persistence, and format encoding."""

import io
import os
import uuid

import numpy as np
import pytest
import tifffile
from dataclasses import replace
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
    resolve_preset_export,
)
from negpy.features.exposure.models import RenderIntent
from negpy.infrastructure.display.color_spaces import WORKING_COLOR_SPACE
from negpy.infrastructure.storage.repository import StorageRepository
from negpy.kernel.image.logic import float_to_uint16
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


def test_preset_flat_render_intent_round_trip():
    p = _make_preset(name="Flat Master", render_intent=RenderIntent.FLAT, export_fmt=ExportFormat.DNG)
    p2 = ExportPreset.from_dict(p.to_dict())
    assert p2.render_intent == RenderIntent.FLAT
    assert p2.export_fmt == ExportFormat.DNG


def test_preset_render_intent_defaults_to_print():
    p = ExportPreset.from_dict(_make_preset().to_dict())
    assert p.render_intent == RenderIntent.PRINT


def test_resolve_preset_export_flat_applies_master_pipeline():
    loud = replace(
        WorkspaceConfig(),
        lab=replace(WorkspaceConfig().lab, saturation=2.0),
        toning=replace(WorkspaceConfig().toning, sepia_strength=1.0),
    )
    preset = _make_preset(render_intent=RenderIntent.FLAT, export_fmt=ExportFormat.TIFF)
    params, delivery = resolve_preset_export(preset, loud)
    assert params.exposure.render_intent == RenderIntent.FLAT
    assert params.exposure.auto_exposure is False
    assert delivery.export_fmt == ExportFormat.TIFF
    assert delivery.export_resolution_mode == ExportResolutionMode.ORIGINAL.value


def test_resolve_preset_export_print_passthrough():
    preset = _make_preset(export_fmt=ExportFormat.JPEG)
    params = WorkspaceConfig()
    out_params, delivery = resolve_preset_export(preset, params)
    assert out_params is params
    assert delivery.export_fmt == ExportFormat.JPEG
    assert delivery.render_intent == RenderIntent.PRINT


def test_flat_export_preset_coerces_delivery_format():
    preset = _make_preset(render_intent=RenderIntent.FLAT, export_fmt=ExportFormat.JPEG)
    _, delivery = resolve_preset_export(preset, WorkspaceConfig())
    assert delivery.export_fmt == ExportFormat.TIFF


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


def _off_axis_buffer(h: int = 8, w: int = 12) -> np.ndarray:
    """RGB buffer that spans off-axis colours so the 3D LUT's interpolation
    is exercised beyond the neutral (R=G=B) diagonal."""
    rng = np.random.default_rng(42)
    buf = rng.uniform(0.2, 0.8, size=(h, w, 3)).astype(np.float32)
    return np.ascontiguousarray(buf)


@pytest.mark.parametrize(
    "target_cs",
    [ColorSpace.SRGB.value, ColorSpace.ADOBE_RGB.value],
)
def test_tiff_cross_space_is_16bit(proc, target_cs):
    """Cross-space TIFF CMS uses lcms2 at 16-bit precision, so the output
    must contain values that are NOT all multiples of 257 — an 8-bit PIL
    path expanded with *257 would produce only multiples-of-257 values.

    This guards against accidentally reverting to the 8-bit CMS path
    (which was the first attempt at fixing #311 before switching to
    imagecodecs for true 16-bit CMS).
    """
    buf = _off_axis_buffer()
    preset = _make_preset(export_fmt=ExportFormat.TIFF, export_color_space=target_cs)
    tiff_data, tiff_status = proc._encode_export(buf, preset, target_cs, WORKING_COLOR_SPACE)
    assert tiff_data is not None, f"export returned None: {tiff_status}"

    tiff_arr = tifffile.imread(io.BytesIO(tiff_data))
    assert tiff_arr.dtype == np.uint16
    assert tiff_arr.shape == (buf.shape[0], buf.shape[1], 3)

    # At least some pixels must not be a multiple of 257 — proves the
    # 16-bit lcms2 path was used, not the 8-bit PIL + *257 expansion.
    not_expanded = np.any((tiff_arr.astype(np.int32) % 257) != 0)
    assert not_expanded, (
        "cross-space TIFF output contains only multiples of 257 — "
        "the 8-bit PIL CMS path is being used instead of the 16-bit "
        "imagecodecs path"
    )


def test_tiff_same_space_preserves_16bit(proc):
    """Same-space TIFF exports (working == target, no custom ICC) must
    preserve full 16-bit precision — no round-trip through 8-bit PIL."""
    buf = _off_axis_buffer()
    target = ColorSpace.PROPHOTO.value  # working space is ProPhoto RGB

    preset = _make_preset(export_fmt=ExportFormat.TIFF, export_color_space=target)
    tiff_data, tiff_status = proc._encode_export(buf, preset, target, WORKING_COLOR_SPACE)
    assert tiff_data is not None, f"export returned None: {tiff_status}"

    tiff_arr = tifffile.imread(io.BytesIO(tiff_data))
    expected = float_to_uint16(buf)

    assert tiff_arr.dtype == np.uint16
    assert tiff_arr.shape == expected.shape
    diff = np.abs(tiff_arr.astype(np.int32) - expected.astype(np.int32))
    assert diff.max() == 0, (
        f"Same-space 16-bit precision lost: max_diff={diff.max()}"
    )


def test_dng_cross_space_uses_16bit_cms(proc):
    """DNG export uses _apply_color_management_u16 (lcms2 16-bit), not
    the old 3D LUT.  DNG output isn't trivially decodable to check
    pixel values, but we verify the export succeeds and returns bytes."""
    buf = _off_axis_buffer()
    preset = _make_preset(export_fmt=ExportFormat.DNG, export_color_space=ColorSpace.SRGB.value)
    data, ext = proc._encode_export(buf, preset, ColorSpace.SRGB.value, WORKING_COLOR_SPACE)
    assert data is not None, "DNG export returned None"
    assert ext == "dng"
    assert isinstance(data, bytes) and len(data) > 0, "DNG export returned empty bytes"


def test_jxl_cross_space_is_16bit(proc):
    """JXL export uses lcms2 at 16-bit precision — decode the output
    and verify it's not just 8-bit CMS expanded to 16-bit."""
    import imagecodecs

    buf = _off_axis_buffer()
    preset = _make_preset(export_fmt=ExportFormat.JXL, export_color_space=ColorSpace.SRGB.value)
    data, ext = proc._encode_export(buf, preset, ColorSpace.SRGB.value, WORKING_COLOR_SPACE)
    assert data is not None, "JXL export returned None"
    assert ext == "jxl"

    decoded = imagecodecs.jpegxl_decode(data)
    assert decoded.dtype == np.uint16, (
        f"Expected uint16 JXL decode, got {decoded.dtype}"
    )
    assert decoded.shape == (buf.shape[0], buf.shape[1], 3)

    not_expanded = np.any((decoded.astype(np.int32) % 257) != 0)
    assert not_expanded, (
        "JXL 16-bit output contains only multiples of 257 — "
        "the 8-bit CMS path is being used instead of the 16-bit "
        "imagecodecs path"
    )
