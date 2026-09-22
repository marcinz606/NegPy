import struct
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import tifffile

from negpy.desktop.session import AppState
from negpy.domain.models import WorkspaceConfig
from negpy.features.flatfield.models import FlatFieldConfig
from negpy.features.geometry.models import GeometryConfig
from negpy.features.lens.logic import apply_lens
from negpy.features.process.models import ProcessMode
from negpy.features.lens.models import LensCorrections, LensMetadata, LensWarp
from negpy.features.lens.warps import IDENTITY, RectilinearWarp, SonyWarp
from negpy.infrastructure.loaders.lens_metadata import bind_decode, parse_opcodes, read_lens_metadata
from negpy.kernel.image.logic import apply_exif_orientation
from negpy.services.rendering.lens import lens_decode_token, metadata_lens_corrections, prepare_lens_source
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


@pytest.mark.parametrize("distortion,ca", [(False, False), (True, False), (False, True), (True, True)])
def test_sony_components_are_independent(distortion, ca):
    shape = (40, 60, 3)
    warp = SonyWarp((-1024,) * 16, (32768,) * 16, (-16384,) * 16)
    lens = LensMetadata("Sony", (warp,))
    y, x = np.mgrid[:40, :60].astype(np.float32)
    for channel, ca_gain in enumerate((1 + 1 / 64, 1, 1 - 1 / 128)):
        mx, my = warp.remap(lens, shape, 0, 40, channel, LensCorrections(distortion, ca))
        factor = (1 - 1 / 16 if distortion else 1) * (ca_gain if ca else 1)
        np.testing.assert_allclose(mx, (x - 30) * factor + 30, atol=1e-5)
        np.testing.assert_allclose(my, (y - 20) * factor + 20, atol=1e-5)


@pytest.mark.parametrize("common", [(0.9, 0, 0, 0, 0, 0), (1, -0.3, 0, 0, 0, 0), (1.02, 0.1, 0.02, 0.001, 0.003, -0.002)])
def test_dng_ca_only_preserves_green_geometry_with_crop_and_off_center_lens(common):
    warp = RectilinearWarp(tuple(tuple(v * scale for v in common) for scale in (1.02, 1, 0.98)), (0.37, 0.61))
    lens = LensMetadata("DNG", (warp,), active_area=(4, 8, 104, 168), buffer_area=(10, 20, 90, 140))
    y, x = np.mgrid[:40, :60].astype(np.float32)
    cx = (8 + 0.37 * 159 - 20 + 0.5) / 2 - 0.5
    cy = (4 + 0.61 * 99 - 10 + 0.5) / 2 - 0.5
    for channel, factor in enumerate((1.02, 1, 0.98)):
        mx, my = warp.remap(lens, (40, 60, 3), 0, 40, channel, LensCorrections(ca=True))
        np.testing.assert_allclose(mx, (x - cx) * factor + cx, atol=1e-4)
        np.testing.assert_allclose(my, (y - cy) * factor + cy, atol=1e-4)
        dx, dy = warp.remap(lens, (40, 60, 3), 0, 40, channel, LensCorrections(distortion=True))
        gx, gy = warp.remap(lens, (40, 60, 3), 0, 40, 1)
        np.testing.assert_array_equal(dx, gx)
        np.testing.assert_array_equal(dy, gy)


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
    mx, my = warp.remap(lens, (90, 190, 3), 0, 90, 0)
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
        mx, my = warp.remap(LensMetadata(), (80, 120, 3), 0, 80, channel)
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


@pytest.mark.parametrize(
    "warp",
    [
        SonyWarp(ca_red=(100,) * 16, ca_blue=(-100,) * 16),
        RectilinearWarp(((1.01, 0, 0, 0, 0, 0), IDENTITY, (0.99, 0, 0, 0, 0, 0))),
    ],
)
def test_lens_preserves_flatfield_values_above_one_for_sensor_unmix(monkeypatch, warp):
    from negpy.features.flatfield import logic as ff
    from negpy.features.process.sensor import apply_sensor_correction, build_sensor_matrix

    image = np.full((40, 60, 3), 0.8, np.float32)
    gain = np.full_like(image, 1.6)
    monkeypatch.setitem(ff._GAIN_CACHE, "reference", (gain, "gain-token"))
    flatfield = FlatFieldConfig(apply=True, profile_id="reference")
    lens = LensMetadata("Test", (warp,))
    out = prepare_lens_source(image, {"lens_correction": lens, "orientation": 1}, flatfield)
    np.testing.assert_allclose(out, image * gain, atol=1e-6)

    matrix = build_sensor_matrix((1, 0.25, 0.25), (0.25, 1, 0.25), (0.25, 0.25, 1))
    expected = apply_sensor_correction(image * gain, matrix)
    assert expected.max() < 1.0
    np.testing.assert_allclose(apply_sensor_correction(out, matrix), expected, atol=1e-6)
    np.testing.assert_array_equal(image, np.float32(0.8))


@pytest.mark.parametrize("distortion,ca", [(False, False), (True, False), (False, True), (True, True)])
def test_setting_roundtrip_and_source_cache_identity(monkeypatch, distortion, ca):
    from negpy.features.flatfield import logic as ff

    monkeypatch.setitem(ff._GAIN_CACHE, "reference", (np.ones((4, 6, 3), np.float32), "gain-token"))
    base = WorkspaceConfig()
    enabled = replace(base, geometry=GeometryConfig(lens_distortion_from_metadata=distortion, lens_ca_from_metadata=ca, distortion_k1=0.05))
    restored = WorkspaceConfig.from_flat_dict(enabled.to_dict())
    assert restored == enabled
    assert restored.geometry.distortion_k1 == (0 if distortion else 0.05)
    assert (source_token(base) != source_token(enabled)) == (distortion or ca)
    flat = FlatFieldConfig(apply=True, profile_id="reference")
    assert (source_token(enabled) != source_token(replace(enabled, flatfield=flat))) == (distortion or ca)
    assert source_token(base) == source_token(replace(base, flatfield=flat))
    off = PreviewCacheKey("file", False, "sRGB", False)
    on = replace(off, lens_token=lens_decode_token(LensCorrections(distortion, ca), flat))
    assert (off.as_tuple() != on.as_tuple()) == (distortion or ca)


@pytest.mark.parametrize("enabled", [False, True])
def test_combined_saved_lens_mode_migrates_without_overriding_split_settings(enabled):
    config = WorkspaceConfig.from_flat_dict({"lens_from_metadata": enabled})
    assert config.geometry.lens_distortion_from_metadata is enabled
    assert config.geometry.lens_ca_from_metadata is enabled
    explicit = WorkspaceConfig.from_flat_dict({"lens_from_metadata": enabled, "lens_ca_from_metadata": not enabled})
    assert explicit.geometry.lens_ca_from_metadata is not enabled
    assert "lens_from_metadata" not in config.to_dict()


def test_sidebar_uses_source_capabilities_and_can_clear_unavailable_saved_mode(qapp, monkeypatch):
    from negpy.desktop.view.sidebar import lens as lens_panel

    controller = MagicMock()
    controller.state = AppState()

    def _edit(_card, persist=True, readback_metrics=True, **changes):
        cfg = controller.state.config
        controller.state.config = replace(cfg, geometry=replace(cfg.geometry, **changes))

    controller.set_roll_default.side_effect = _edit
    monkeypatch.setattr(lens_panel, "read_lens_metadata", lambda path: LensMetadata())
    sidebar = lens_panel.LensSidebar(controller)
    sidebar.sync_ui()
    assert not sidebar.metadata_distortion_btn.isEnabled()
    assert not sidebar.metadata_ca_btn.isEnabled()
    assert sidebar.distortion_slider.isEnabled()
    ca = LensMetadata("Sony", (SonyWarp(ca_red=(100,) * 16, ca_blue=(-100,) * 16),))
    monkeypatch.setattr(lens_panel, "read_lens_metadata", lambda path: ca)
    sidebar.sync_ui()
    assert not sidebar.metadata_distortion_btn.isEnabled()
    assert sidebar.metadata_ca_btn.isEnabled()
    assert "lateral CA" in sidebar.lens_hint.text()
    assert "distortion" not in sidebar.lens_hint.text()
    sidebar.metadata_ca_btn.click()
    assert controller.state.config.geometry.lens_ca_from_metadata
    assert not controller.state.config.geometry.lens_distortion_from_metadata
    sidebar.sync_ui()
    assert sidebar.distortion_slider.isEnabled()
    monkeypatch.setattr(lens_panel, "read_lens_metadata", lambda path: LensMetadata())
    sidebar.sync_ui()
    assert sidebar.metadata_ca_btn.isEnabled()
    assert sidebar.metadata_ca_btn.isChecked()
    assert "Unavailable" in sidebar.lens_hint.text()
    sidebar.metadata_ca_btn.click()
    assert not controller.state.config.geometry.lens_ca_from_metadata

    distortion = LensMetadata("Sony", (SonyWarp(distortion=(100,) * 16),))
    controller.state.config = WorkspaceConfig()
    monkeypatch.setattr(lens_panel, "read_lens_metadata", lambda path: distortion)
    sidebar.sync_ui()
    assert sidebar.metadata_distortion_btn.isEnabled()
    assert not sidebar.metadata_ca_btn.isEnabled()
    sidebar.metadata_distortion_btn.click()
    sidebar.sync_ui()
    assert not sidebar.distortion_slider.isEnabled()


def test_composites_do_not_apply_primary_lens_metadata():
    from negpy.features.hdr.models import HdrConfig
    from negpy.features.rgbscan.models import RgbScanConfig
    from negpy.features.stitch.models import StitchConfig

    config = replace(WorkspaceConfig(), geometry=GeometryConfig(lens_distortion_from_metadata=True, lens_ca_from_metadata=True))
    assert metadata_lens_corrections(config)
    assert not metadata_lens_corrections(replace(config, hdr=HdrConfig(hdr_enabled=True, hdr_paths=("b.arw",))))
    assert not metadata_lens_corrections(replace(config, rgbscan=RgbScanConfig(enabled=True, green_path="g.arw", blue_path="b.arw")))
    assert not metadata_lens_corrections(replace(config, stitch=StitchConfig(stitch_enabled=True, stitch_paths=("b.arw",))))


@pytest.mark.parametrize("kind", ["stitch", "hdr"])
def test_composite_solve_uses_unwarped_sources(monkeypatch, kind):
    from negpy.desktop.workers import hdr, stitch
    from negpy.features.flatfield import logic as ff
    from negpy.services.rendering.image_processor import ImageProcessor

    rng = np.random.default_rng(7)
    sources = {path: rng.integers(6500, 40000, (60, 80, 3), dtype=np.uint16) for path in ("a.dng", "b.dng")}
    lens = LensMetadata("DNG", (RectilinearWarp(((1, -0.1, 0, 0, 0, 0),)),))
    monkeypatch.setattr(
        ImageProcessor,
        "_decode_sensor_rgb",
        lambda self, path, *args, **kwargs: (sources[path].copy(), {"lens_correction": lens, "orientation": 1}),
    )
    monkeypatch.setitem(ff._GAIN_CACHE, "reference", (np.full((60, 80, 3), 1.1, np.float32), "gain-token"))
    config = replace(
        WorkspaceConfig(),
        geometry=GeometryConfig(lens_distortion_from_metadata=True, lens_ca_from_metadata=True),
        flatfield=FlatFieldConfig(apply=True, profile_id="reference"),
    )
    seen = []

    def solve(buffers, *args, **kwargs):
        seen.extend(buffer.copy() for buffer in buffers)
        if kind == "stitch":
            return [np.eye(2, 3), np.eye(2, 3)], (80, 60)
        return [1.0, 2.0]

    if kind == "stitch":
        monkeypatch.setattr(stitch, "register_parts", solve)
        worker, task_type = stitch.StitchWorker(), stitch.StitchTask
        completed = worker.registered
    else:
        monkeypatch.setattr(hdr, "solve_ratios", solve)
        worker, task_type = hdr.HdrWorker(), hdr.HdrTask
        completed = worker.solved
    results, errors = [], []
    completed.connect(results.append)
    worker.error.connect(errors.append)
    worker.run(
        task_type(
            files=tuple({"path": path, "name": path} for path in sources),
            params_by_path={path: config for path in sources},
        )
    )
    assert not errors and len(results) == 1
    assert len(seen) == len(sources)
    for actual, source in zip(seen, sources.values()):
        expected = source.astype(np.float32) / 65535.0
        if kind == "stitch":
            expected *= 1.1
        np.testing.assert_array_equal(actual, expected)
    assert config.geometry.lens_distortion_from_metadata


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
        geometry=GeometryConfig(lens_distortion_from_metadata=True, lens_ca_from_metadata=True),
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
            lens_corrections=LensCorrections(True, True),
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
    config = replace(
        config,
        geometry=GeometryConfig(lens_distortion_from_metadata=True, lens_ca_from_metadata=True),
        process=replace(config.process, linear_raw=True),
    )
    preview, _, _ = PreviewManager().load_linear_preview(str(path), full_resolution=True, lens_corrections=LensCorrections(True, True))
    exported, _, _ = ImageProcessor()._load_source_f32(str(path), config)
    assert preview.shape == (88, 138, 3)
    np.testing.assert_array_equal(preview, exported)


@pytest.mark.parametrize("enabled", [False, True])
def test_history_or_reset_reloads_pixels_when_metadata_mode_changes(enabled):
    from negpy.desktop.controller import AppController

    state = AppState(current_file_path="scan.arw")
    state.config = replace(state.config, geometry=GeometryConfig(lens_distortion_from_metadata=enabled, lens_ca_from_metadata=enabled))
    state.preview_lens_token = lens_decode_token(LensCorrections(not enabled, not enabled), state.config.flatfield)
    controller = SimpleNamespace(state=state, _render_debounce=MagicMock(), load_file=MagicMock())
    AppController.request_render(controller)
    controller.load_file.assert_called_once_with("scan.arw", preserve_zoom=True)


@pytest.mark.parametrize("splash", [False, True])
@pytest.mark.parametrize("color_space", [None, "Adobe RGB"])
def test_positive_source_and_lens_mode_have_independent_preview_cache_entries(tmp_path, monkeypatch, splash, color_space):
    from negpy.infrastructure.loaders import factory
    from negpy.infrastructure.loaders.helpers import NonStandardFileWrapper
    from negpy.services.rendering.image_processor import ImageProcessor
    from negpy.services.rendering.preview_manager import PreviewManager

    path = str(tmp_path / "source.arw")
    ramp = np.tile(np.linspace(0.1, 0.7, 120, dtype=np.float32), (80, 1))
    image = np.repeat(ramp[..., None], 3, axis=2)
    lens = LensMetadata("Sony", (SonyWarp((-1000,) * 16, (32768,) * 16, (-16384,) * 16),))

    def get_loader(file_path, *, linear_raw=False, positive_source=False, preview_max_edge=None, should_cancel=None):
        pixels = image * (0.5 if positive_source else 1.0)
        return NonStandardFileWrapper(pixels), {"orientation": 1, "color_space": "Adobe RGB", "lens_correction": lens}

    monkeypatch.setattr(factory.loader_factory, "get_loader", get_loader)
    monkeypatch.setattr("negpy.services.rendering.preview_manager.APP_CONFIG.preview_cache_max_full_res_entries", 8)
    manager = PreviewManager()
    processor = ImageProcessor()
    load = manager.load_splash_and_linear if splash else manager.load_linear_preview
    outputs = {}
    modes = [LensCorrections(d, ca) for d in (False, True) for ca in (False, True)]
    for positive, corrections in [(positive, mode) for positive in (False, True) for mode in modes] * 2:
        result = load(
            path,
            color_space=color_space,
            use_camera_wb=False,
            full_resolution=True,
            file_hash="source",
            positive_source=positive,
            lens_corrections=corrections,
        )
        preview = result[1][0] if splash else result[0]
        config = WorkspaceConfig()
        config = replace(
            config,
            geometry=GeometryConfig(lens_distortion_from_metadata=corrections.distortion, lens_ca_from_metadata=corrections.ca),
            process=replace(config.process, process_mode=ProcessMode.E6, linear_raw=True, positive_source=positive),
        )
        exported, _, _ = processor._load_source_f32(path, config)
        np.testing.assert_allclose(preview, exported, atol=1e-6)
        if (positive, corrections) in outputs:
            assert preview is outputs[positive, corrections]
        outputs[positive, corrections] = preview
    assert len({out.tobytes() for out in outputs.values()}) == 8


@pytest.mark.parametrize("mode", ["linear", "splash", "warm"])
def test_preview_worker_forwards_positive_source_and_lens_settings(mode):
    from negpy.desktop.workers.render import PreviewLoadTask, PreviewLoadWorker

    service = MagicMock()
    result = (np.full((8, 12, 3), 0.5, np.float32), (8, 12), {})
    service.load_linear_preview.return_value = result
    service.load_splash_and_linear.return_value = (None, result)
    service.prefetch_linear_preview.return_value = True
    task = PreviewLoadTask(
        file_path="source.arw",
        workspace_color_space="Adobe RGB",
        use_camera_wb=False,
        positive_source=True,
        lens_corrections=LensCorrections(True, True),
        lens_flatfield=FlatFieldConfig(apply=True, profile_id="gain"),
        use_splash=mode == "splash",
        for_cache_warm=mode == "warm",
    )
    worker = PreviewLoadWorker(service)
    errors = []
    worker.error.connect(errors.append)
    worker.process(task)
    if mode == "splash":
        call = service.load_splash_and_linear
    elif mode == "warm":
        call = service.prefetch_linear_preview
    else:
        call = service.load_linear_preview
    assert call.call_count == 1
    assert call.call_args.kwargs["positive_source"] is True
    assert call.call_args.kwargs["lens_corrections"] == LensCorrections(True, True)
    assert call.call_args.kwargs["lens_flatfield"] == task.lens_flatfield
    assert not errors


@pytest.mark.parametrize("orientation", [1, 6])
def test_registered_reader_and_structural_warp_use_shared_rendering(tmp_path, monkeypatch, orientation):
    from negpy.infrastructure.loaders import lens_metadata as reader

    calls = []
    reads = []

    @dataclass(frozen=True)
    class OffsetWarp:
        offsets: tuple[int, ...]

        @property
        def has_distortion(self) -> bool:
            return self.offsets[1] != 0

        @property
        def has_ca(self) -> bool:
            return self.offsets[0] != self.offsets[1] or self.offsets[2] != self.offsets[1]

        def remap(self, lens, shape, start, stop, channel, corrections=LensCorrections(True, True)):
            calls.append((lens, shape, start, stop, channel))
            y, x = np.mgrid[start:stop, : shape[1]].astype(np.float32)
            return x + self.offsets[channel], y

    def read_offsets(file_path: str) -> LensMetadata:
        reads.append(file_path)
        offsets = tuple(int(v) for v in Path(file_path).read_text().split(","))
        warp: LensWarp = OffsetWarp(offsets)
        return LensMetadata("Offset reader", (warp,))

    path = tmp_path / "source.CUSTOM"
    path.write_text("2,1,-1")
    monkeypatch.setitem(reader._READERS, ".custom", read_offsets)
    lens = read_lens_metadata(str(path))
    assert read_lens_metadata(str(path)) is lens
    assert reads == [str(path)]
    assert lens.description == "Offset reader: distortion + lateral CA"

    image = np.random.default_rng(7).uniform(0.1, 0.9, (521, 35, 3)).astype(np.float32)
    expected = np.stack([image[:, np.clip(np.arange(35) + shift, 0, 34), ch] for ch, shift in enumerate((2, 1, -1))], axis=-1)
    oriented = apply_exif_orientation(image, orientation)
    result = apply_lens(oriented, lens, orientation)
    np.testing.assert_array_equal(result, apply_exif_orientation(expected, orientation))
    assert all(context is lens and shape == image.shape for context, shape, *_ in calls)
    assert [(start, stop, channel) for _, _, start, stop, channel in calls] == [
        (start, min(start + 256, 521), channel) for channel in range(3) for start in range(0, 521, 256)
    ]


@pytest.mark.parametrize(
    "warp, distortion, ca",
    [
        (RectilinearWarp((IDENTITY,)), False, False),
        (RectilinearWarp((IDENTITY,) * 3), False, False),
        (RectilinearWarp(((1, -0.1, 0, 0, 0, 0),)), True, False),
        (RectilinearWarp(((1.01, 0, 0, 0, 0, 0), IDENTITY, IDENTITY)), False, True),
        (SonyWarp(), False, False),
        (SonyWarp((0,) * 16, (0,) * 16, (0,) * 16), False, False),
        (SonyWarp(distortion=(100,) * 16), True, False),
        (SonyWarp(ca_red=(100,) * 16, ca_blue=(-100,) * 16), False, True),
        (SonyWarp((100,) * 16, (100,) * 16, (-100,) * 16), True, True),
    ],
)
def test_warp_capabilities_drive_availability(warp: LensWarp, distortion, ca):
    lens = LensMetadata("Test", (warp,))
    assert lens.distortion is distortion
    assert lens.ca is ca
    assert lens.available is (distortion or ca)
    if not lens.available:
        image = np.full((16, 24, 3), 0.5, np.float32)
        assert apply_lens(image, lens) is image
