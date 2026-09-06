import struct
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import tifffile

from negpy.desktop.session import AppState
from negpy.desktop.view.sidebar.geometry import GeometrySidebar
from negpy.domain.models import WorkspaceConfig
from negpy.features.flatfield.models import FlatFieldConfig
from negpy.features.geometry.models import GeometryConfig
from negpy.features.lens.logic import _dng_maps, _sony_maps, apply_lens
from negpy.features.lens.models import IDENTITY, LensMetadata, RectilinearWarp, SonyWarp
from negpy.infrastructure.loaders.lens_metadata import bind_decode, parse_opcodes, read_lens_metadata
from negpy.kernel.image.logic import apply_exif_orientation
from negpy.services.rendering.lens import lens_decode_token, metadata_lens_enabled, prepare_lens_source
from negpy.services.rendering.preview_cache import PreviewCacheKey
from negpy.services.rendering.source_identity import source_token


def opcode(coefficients=(IDENTITY,), center=(0.5, 0.5), code=1, flags=0):
    values = tuple(v for plane in coefficients for v in plane) + center
    payload = struct.pack(">I", len(coefficients)) + struct.pack(f">{len(values)}d", *values)
    return struct.pack(">5I", 1, code, 0x01030000, flags, len(payload)) + payload


def raw_file(path, tags=(), *, dng=True, subifd=False, byteorder="<"):
    image = np.zeros((40, 60), dtype=np.uint16)
    extra = [(50706, "B", 4, (1, 7, 1, 0), False)] if dng else []
    extra += list(tags)
    with tifffile.TiffWriter(path, byteorder=byteorder) as tif:
        if subifd:
            tif.write(np.zeros((8, 12, 3), dtype=np.uint8), photometric="rgb", subifds=1, metadata=None)
        tif.write(image, photometric=32803, extratags=extra, metadata=None)
    return str(path)


def dng_file(tmp_path, data, **kwargs):
    return raw_file(tmp_path / "source.dng", [(51022, "B", len(data), data, False)], **kwargs)


def test_dng_identity_and_separate_capabilities():
    assert parse_opcodes(opcode()) == ()
    ca = ((1.002, 0, 0, 0, 0, 0), IDENTITY, (0.998, 0, 0, 0, 0, 0))
    lens = LensMetadata("DNG", parse_opcodes(opcode(ca)))
    assert lens.available and lens.ca and not lens.distortion
    distortion = LensMetadata("DNG", parse_opcodes(opcode(((1, -0.1, 0, 0, 0, 0),))))
    assert distortion.distortion and not distortion.ca


@pytest.mark.parametrize("byteorder", ["<", ">"])
@pytest.mark.parametrize("subifd", [False, True])
def test_dng_17_and_tiff_byte_order_do_not_change_opcode_endianness(tmp_path, byteorder, subifd):
    data = opcode(((1, -0.08, 0.02, 0, 0, 0),))
    lens = read_lens_metadata(dng_file(tmp_path, data, byteorder=byteorder, subifd=subifd))
    assert lens.distortion
    assert lens.active_area == (0, 0, 40, 60)


@pytest.mark.parametrize(
    "data",
    [
        b"",
        b"\0\0",
        opcode()[:-1],
        opcode() + b"extra",
        struct.pack(">I", 1000000),
        opcode(code=14),
        opcode(code=6),
        opcode(flags=4),
        opcode(((float("nan"), 0, 0, 0, 0, 0),)),
        opcode(((1, -1, 0, 0, 0, 0),)),
        opcode(center=(-0.01, 0.5)),
        opcode((IDENTITY, IDENTITY)),
    ],
)
def test_bad_or_unsupported_dng_metadata_is_unavailable_without_breaking_load(tmp_path, data):
    lens = read_lens_metadata(dng_file(tmp_path, data))
    assert not lens.available
    assert lens.reason


def test_plain_exif_and_rendered_images_never_enable_embedded_correction(tmp_path):
    assert not read_lens_metadata(raw_file(tmp_path / "no-profile.dng")).available
    jpeg = tmp_path / "image.jpg"
    jpeg.write_bytes(b"not needed")
    assert not read_lens_metadata(str(jpeg)).available
    rendered = tmp_path / "rendered.dng"
    data = opcode(((1, -0.1, 0, 0, 0, 0),))
    tifffile.imwrite(rendered, np.zeros((40, 60, 3), np.uint16), photometric="rgb", extratags=[(51022, "B", len(data), data, False)])
    assert not read_lens_metadata(str(rendered)).available


def test_sony_padded_arrays_and_ca_only_are_independent(tmp_path):
    dist = (11, *range(-100, 10, 10), *([0] * 5))
    ca = (22, *([100] * 11), *([-100] * 11), *([0] * 10))
    path = raw_file(tmp_path / "source.arw", [(0x7037, "h", 17, dist, False), (0x7035, "h", 33, ca, False)], dng=False, subifd=True)
    lens = read_lens_metadata(path)
    assert lens.distortion and lens.ca
    assert len(lens.warps[0].distortion) == 11
    ca_path = raw_file(tmp_path / "ca.arw", [(0x7035, "h", 33, ca, False)], dng=False)
    ca_lens = read_lens_metadata(ca_path)
    assert ca_lens.ca and not ca_lens.distortion


def test_sony_unavailable_flag_overrides_leftover_coefficients(tmp_path):
    data = (16, *([100] * 16))
    path = raw_file(tmp_path / "source.arw", [(0x7037, "h", 17, data, False), (0x7036, "H", 1, 255, False)], dng=False)
    assert not read_lens_metadata(path).available


def test_dng_does_not_reuse_inherited_sony_coefficients(tmp_path):
    data = (16, *([100] * 16))
    path = raw_file(tmp_path / "converted.dng", [(0x7037, "h", 17, data, False)])
    assert not read_lens_metadata(path).available


def test_metadata_cache_tracks_file_revision(tmp_path):
    path = dng_file(tmp_path, opcode(((1, -0.1, 0, 0, 0, 0),)))
    assert read_lens_metadata(path).available
    dng_file(tmp_path, struct.pack(">I", 0))
    assert not read_lens_metadata(path).available


def test_dng_coordinate_map_matches_spec_with_offset_center_and_tangential_terms():
    warp = RectilinearWarp(((1, -0.03, 0.01, 0, 0.001, -0.002),), (0.4, 0.6))
    lens = LensMetadata("DNG", (warp,), active_area=(10, 20, 110, 220), buffer_area=(15, 25, 105, 215))
    mx, my = _dng_maps(lens, warp, (90, 190, 3), 0, 90, 0)
    x, y = 170, 75
    cx, cy = 20 + 0.4 * 199, 10 + 0.6 * 99
    radius = np.hypot(max(cx - 20, 219 - cx), max(cy - 10, 109 - cy))
    dx, dy = (25 + x - cx) / radius, (15 + y - cy) / radius
    r2 = dx * dx + dy * dy
    factor = 1 - 0.03 * r2 + 0.01 * r2**2
    expected_x = cx + radius * (dx * factor + 0.002 * dx * dy - 0.002 * (r2 + 2 * dx**2)) - 25
    expected_y = cy + radius * (dy * factor - 0.004 * dx * dy + 0.001 * (r2 + 2 * dy**2)) - 15
    assert mx[y, x] == pytest.approx(expected_x, abs=2e-5)
    assert my[y, x] == pytest.approx(expected_y, abs=2e-5)


def test_sony_known_scale_and_ca_units():
    warp = SonyWarp((-819.2,) * 16, (2097.152,) * 16, (-2097.152,) * 16)
    for channel, ca in enumerate((1.001, 1.0, 0.999)):
        mx, my = _sony_maps(warp, (80, 120, 3), 0, 80, channel)
        assert mx[10, 15] == pytest.approx(60 + (15 - 60) * 0.95 * ca, abs=1e-5)
        assert my[10, 15] == pytest.approx(40 + (10 - 40) * 0.95 * ca, abs=1e-5)


@pytest.mark.parametrize("orientation", range(1, 9))
def test_orientation_and_tca_keep_green_unchanged(orientation):
    ramp = np.tile(np.linspace(0.1, 0.8, 120, dtype=np.float32), (80, 1))
    image = np.repeat(ramp[..., None], 3, axis=2)
    coefficients = ((1.01, 0, 0, 0, 0, 0), IDENTITY, (0.99, 0, 0, 0, 0, 0))
    lens = LensMetadata("DNG", (RectilinearWarp(coefficients, (0.4, 0.6)),))
    expected = apply_exif_orientation(apply_lens(image, lens), orientation)
    oriented = apply_exif_orientation(image, orientation)
    result = apply_lens(oriented, lens, orientation)
    np.testing.assert_allclose(result, expected, atol=1e-6)
    np.testing.assert_array_equal(result[..., 1], oriented[..., 1])
    assert not np.array_equal(result[..., 0], oriented[..., 0])


def test_noop_preserves_source_and_distortion_preserves_flat_color():
    image = np.full((40, 60, 3), 0.37, np.float32)
    assert apply_lens(image, LensMetadata()) is image
    lens = LensMetadata("DNG", (RectilinearWarp(((1, -0.1, 0.02, 0, 0, 0),)),))
    np.testing.assert_allclose(apply_lens(image, lens), image, atol=1e-6)


def test_bind_decode_checks_coordinate_compatibility():
    lens = LensMetadata("DNG", (RectilinearWarp(((1, -0.1, 0, 0, 0, 0),)),), active_area=(4, 8, 104, 208))
    raw = SimpleNamespace(sizes=SimpleNamespace(top_margin=4, left_margin=8, height=100, width=200))
    assert bind_decode(lens, raw).buffer_area == lens.active_area
    raw.sizes.width = 300
    assert not bind_decode(lens, raw).available
    cropped_fallback = SimpleNamespace(sizes=SimpleNamespace(raw_width=180, raw_height=90))
    assert not bind_decode(lens, cropped_fallback, fallback=True).available


def test_flatfield_is_applied_before_the_lens_warp(monkeypatch):
    from negpy.services.rendering import lens as service

    image = np.ones((40, 60, 3), np.float32)
    gain = np.broadcast_to(np.linspace(0.2, 0.8, 60, dtype=np.float32)[None, :, None], image.shape)
    metadata = {"lens_correction": LensMetadata("Sony", (SonyWarp((-1000,) * 16),)), "orientation": 1}
    monkeypatch.setattr(service, "apply_flatfield", lambda img, config: img * gain)
    out = prepare_lens_source(image, metadata, FlatFieldConfig())
    np.testing.assert_array_equal(out, apply_lens(image * gain, metadata["lens_correction"]))
    np.testing.assert_array_equal(image, 1.0)


def test_setting_roundtrip_and_source_cache_identity(monkeypatch):
    from negpy.features.flatfield import logic as ff

    monkeypatch.setitem(ff._GAIN_CACHE, "reference", (np.ones((4, 6, 3), np.float32), "gain-token"))
    base = WorkspaceConfig()
    enabled = replace(base, geometry=GeometryConfig(lens_from_metadata=True, distortion_k1=0.05))
    restored = WorkspaceConfig.from_flat_dict(enabled.to_dict())
    assert restored.geometry.lens_from_metadata
    assert restored.geometry.distortion_k1 == 0
    assert source_token(base) != source_token(enabled)
    flat = FlatFieldConfig(apply=True, profile_id="reference")
    assert source_token(enabled) != source_token(replace(enabled, flatfield=flat))
    assert source_token(base) == source_token(replace(base, flatfield=flat))
    off = PreviewCacheKey("file", False, "sRGB", False)
    on = replace(off, lens_token=lens_decode_token(True, flat))
    assert off.as_tuple() != on.as_tuple()


def test_sidebar_uses_source_capabilities_and_can_clear_unavailable_saved_mode(qapp, monkeypatch):
    from negpy.desktop.view.sidebar import geometry

    controller = MagicMock()
    controller.state = AppState()
    monkeypatch.setattr(geometry, "read_lens_metadata", lambda path: LensMetadata())
    sidebar = GeometrySidebar(controller)
    sidebar.sync_ui()
    assert not sidebar.metadata_lens_btn.isEnabled()
    assert sidebar.distortion_slider.isEnabled()
    ca = LensMetadata("Sony", (SonyWarp(ca_red=(100,) * 16, ca_blue=(-100,) * 16),))
    monkeypatch.setattr(geometry, "read_lens_metadata", lambda path: ca)
    sidebar.sync_ui()
    assert sidebar.metadata_lens_btn.isEnabled()
    assert "lateral CA" in sidebar.lens_hint.text()
    assert "distortion" not in sidebar.lens_hint.text()
    sidebar.metadata_lens_btn.click()
    requested = controller.apply_config.call_args.args[0]
    assert requested.geometry.lens_from_metadata
    controller.state.config = requested
    monkeypatch.setattr(geometry, "read_lens_metadata", lambda path: LensMetadata())
    sidebar.sync_ui()
    assert sidebar.metadata_lens_btn.isEnabled()
    assert sidebar.metadata_lens_btn.isChecked()
    assert "Unavailable" in sidebar.lens_hint.text()
    sidebar.metadata_lens_btn.click()
    assert not controller.apply_config.call_args.args[0].geometry.lens_from_metadata


def test_composites_do_not_apply_primary_lens_metadata():
    from negpy.features.hdr.models import HdrConfig
    from negpy.features.rgbscan.models import RgbScanConfig
    from negpy.features.stitch.models import StitchConfig

    config = replace(WorkspaceConfig(), geometry=GeometryConfig(lens_from_metadata=True))
    assert metadata_lens_enabled(config)
    assert not metadata_lens_enabled(replace(config, hdr=HdrConfig(hdr_enabled=True, hdr_paths=("b.arw",))))
    assert not metadata_lens_enabled(replace(config, rgbscan=RgbScanConfig(enabled=True, green_path="g.arw", blue_path="b.arw")))
    assert not metadata_lens_enabled(replace(config, stitch=StitchConfig(stitch_enabled=True, stitch_paths=("b.arw",))))


def test_preview_and_export_share_warp_flatfield_and_per_file_coefficients(tmp_path, monkeypatch):
    from negpy.features.flatfield import logic as ff
    from negpy.infrastructure.loaders import factory
    from negpy.infrastructure.loaders.helpers import NonStandardFileWrapper
    from negpy.services.rendering.image_processor import ImageProcessor
    from negpy.services.rendering.preview_manager import PreviewManager

    ramp = np.tile(np.linspace(0.1, 0.7, 120, dtype=np.float32), (80, 1))
    image = np.repeat(ramp[..., None], 3, axis=2)
    first = tmp_path / "one.arw"
    second = tmp_path / "two.arw"
    first.touch()
    second.touch()
    lenses = {
        str(first): LensMetadata("Sony", (SonyWarp((-1000,) * 16),)),
        str(second): LensMetadata("Sony", (SonyWarp((800,) * 16),)),
    }
    monkeypatch.setattr(
        factory.loader_factory,
        "get_loader",
        lambda path, **kw: (
            NonStandardFileWrapper(image.copy()),
            {"orientation": 6, "lens_correction": lenses[path], "ir": None},
        ),
    )
    monkeypatch.setitem(ff._GAIN_CACHE, "test-gain", (np.full((8, 12, 3), 1.1, np.float32), "gain"))
    config = WorkspaceConfig()
    config = replace(
        config,
        geometry=GeometryConfig(lens_from_metadata=True),
        process=replace(config.process, linear_raw=True),
        flatfield=FlatFieldConfig(apply=True, profile_id="test-gain"),
    )
    preview = PreviewManager()
    processor = ImageProcessor()
    outputs = []
    for path in (str(first), str(second), str(first)):
        out, _, _ = preview.load_linear_preview(
            path,
            color_space="Adobe RGB",
            full_resolution=True,
            file_hash=path,
            lens_from_metadata=True,
            lens_flatfield=config.flatfield,
        )
        exported, _, _ = processor._load_source_f32(path, config)
        np.testing.assert_allclose(out, exported, atol=1e-6)
        outputs.append(out)
    assert not np.array_equal(outputs[0], outputs[1])
    np.testing.assert_array_equal(outputs[0], outputs[2])
    unwarped, _, _ = preview.load_linear_preview(str(first), color_space="Adobe RGB", full_resolution=True, file_hash=str(first))
    assert not np.array_equal(unwarped, outputs[0])


def test_linear_dng_fallback_crop_is_relative_to_active_area(tmp_path):
    from negpy.infrastructure.loaders.rawpy_loader import _peek_linear_dng_rgb

    image = np.arange(60 * 80 * 3, dtype=np.uint16).reshape(60, 80, 3)
    data = opcode(((1, -0.05, 0, 0, 0, 0),))
    path = tmp_path / "linear.dng"
    tifffile.imwrite(
        path,
        image,
        photometric=34892,
        planarconfig="contig",
        metadata=None,
        extratags=[
            (50706, "B", 4, (1, 7, 1, 0), False),
            (50829, "I", 4, (4, 8, 56, 72), False),
            (50719, "I", 2, (3, 2), False),
            (50720, "I", 2, (58, 48), False),
            (51022, "B", len(data), data, False),
        ],
    )
    result = _peek_linear_dng_rgb(str(path))
    assert result is not None
    np.testing.assert_allclose(result[0], image[6:54, 11:69] / 65535, atol=1e-7)
    lens = read_lens_metadata(str(path))
    bound = bind_decode(lens, SimpleNamespace(sizes=SimpleNamespace(raw_height=48, raw_width=58)), fallback=True)
    assert bound.available and bound.buffer_area == (6, 11, 54, 69)


def test_dng_17_jpegxl_fallback_keeps_preview_export_and_optical_center_in_sync(tmp_path):
    from negpy.infrastructure.loaders.rawpy_loader import RawpyLoader
    from negpy.services.rendering.image_processor import ImageProcessor
    from negpy.services.rendering.preview_manager import PreviewManager

    ramp = np.tile(np.linspace(4000, 45000, 160).astype(np.uint16), (100, 1))
    image = np.repeat(ramp[..., None], 3, axis=2)
    data = opcode(((1, -0.05, 0.01, 0, 0.001, -0.001),), center=(0.4, 0.6))
    path = tmp_path / "jpegxl.dng"
    tifffile.imwrite(
        path,
        image,
        photometric=34892,
        planarconfig="contig",
        compression=52546,
        metadata=None,
        extratags=[
            (50706, "B", 4, (1, 7, 1, 0), False),
            (50707, "B", 4, (1, 4, 0, 0), False),
            (50829, "I", 4, (4, 8, 96, 152), False),
            (50719, "I", 2, (3, 2), False),
            (50720, "I", 2, (138, 88), False),
            (51022, "B", len(data), data, False),
        ],
    )
    raw, metadata = RawpyLoader().load(str(path))
    with raw:
        assert metadata["lens_correction"].available
        assert metadata["lens_correction"].buffer_area == (6, 11, 94, 149)
    config = WorkspaceConfig()
    config = replace(config, geometry=GeometryConfig(lens_from_metadata=True), process=replace(config.process, linear_raw=True))
    preview, _, _ = PreviewManager().load_linear_preview(str(path), full_resolution=True, lens_from_metadata=True)
    exported, _, _ = ImageProcessor()._load_source_f32(str(path), config)
    assert preview.shape == (88, 138, 3)
    np.testing.assert_array_equal(preview, exported)


@pytest.mark.parametrize("enabled", [False, True])
def test_history_or_reset_reloads_pixels_when_metadata_mode_changes(enabled):
    from negpy.desktop.controller import AppController

    state = AppState(current_file_path="scan.arw")
    state.config = replace(state.config, geometry=GeometryConfig(lens_from_metadata=enabled))
    state.preview_lens_token = lens_decode_token(not enabled, state.config.flatfield)
    controller = SimpleNamespace(state=state, _render_debounce=MagicMock(), load_file=MagicMock())
    AppController.request_render(controller)
    controller.load_file.assert_called_once_with("scan.arw", preserve_zoom=True)
