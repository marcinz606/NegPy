"""A Positive TIFF decodes by Source Encoding, whatever Linear RAW says."""

import io
import os
from dataclasses import replace

import numpy as np
import pytest
import tifffile
from PIL import Image

from negpy.domain.models import ColorSpace, ExportFormat, WorkspaceConfig
from negpy.features.process.capture_color import apply_camera_matrix, camera_to_working_matrix
from negpy.features.process.logic import linear_raw_token
from negpy.features.process.models import ProcessConfig, ProcessMode, SourceEncoding, with_positive_source
from negpy.features.process.path import positive_decode
from negpy.infrastructure.loaders import source_encoding as se
from negpy.infrastructure.loaders.helpers import camera_xyz_matrix
from negpy.infrastructure.loaders.source_kind import is_scanned_positive_source, with_scanner_positive_default
from negpy.infrastructure.loaders.tiff_loader import TiffLoader
from negpy.kernel.image.logic import working_oetf_decode
from negpy.services.rendering.source_identity import source_token
from tests.test_tiff_loader_encoding import _ADOBE_RGB_ICC

_ICC_DIR = os.path.join(os.path.dirname(__file__), "..", "icc")


def _icc(name: str) -> bytes:
    with open(os.path.join(_ICC_DIR, name), "rb") as fh:
        return fh.read()


def _rgb16(seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 30000, (16, 24, 3), dtype=np.uint16)


def _write(tmp_path, data: np.ndarray, icc: bytes | None = None) -> str:
    path = str(tmp_path / "scan.tif")
    extratags = [(34675, 7, len(icc), icc, True)] if icc else []
    tifffile.imwrite(path, data, photometric="rgb", extratags=extratags)
    return path


def _load(path: str, encoding: str, linear_raw: bool = True) -> tuple[np.ndarray, dict]:
    ctx, metadata = TiffLoader().load(path, linear_raw=linear_raw, positive_decode=encoding)
    with ctx as raw:
        return raw.data, metadata


@pytest.mark.parametrize("linear_raw", [True, False])
def test_auto_reads_an_untagged_16_bit_scan_as_linear(tmp_path, linear_raw) -> None:
    data = _rgb16()
    f32, metadata = _load(_write(tmp_path, data), SourceEncoding.AUTO, linear_raw)
    np.testing.assert_allclose(f32, data / 65535.0, atol=1e-6)
    assert metadata["source_encoding"] == "linear"


@pytest.mark.parametrize("linear_raw", [True, False])
def test_auto_decodes_an_adobe_rgb_profile_even_with_linear_raw(tmp_path, linear_raw) -> None:
    data = _rgb16()
    f32, metadata = _load(_write(tmp_path, data, _ADOBE_RGB_ICC), SourceEncoding.AUTO, linear_raw)
    np.testing.assert_allclose(f32, working_oetf_decode(data.astype(np.float32) / 65535.0), atol=2e-5)
    assert metadata["source_encoding"] == "profile"
    assert metadata["color_space"] == ColorSpace.ADOBE_RGB.value


def _load_working(path: str, encoding: str) -> np.ndarray:
    """The decode as the engines see it: the source primaries the loader keeps, then the
    source-to-working matrix it carries in the camera-matrix slot."""
    ctx, _ = TiffLoader().load(path, linear_raw=True, positive_decode=encoding)
    with ctx as raw:
        return apply_camera_matrix(raw.data, camera_to_working_matrix(camera_xyz_matrix(raw)))


def test_an_srgb_profile_converts_to_adobe_primaries(tmp_path) -> None:
    """Pure sRGB red is inside Adobe RGB, so it lands at a smaller red and a nonzero green."""
    data = np.zeros((4, 4, 3), dtype=np.uint16)
    data[..., 0] = 65535
    path = _write(tmp_path, data, _icc("sRGB-v4.icc"))
    source, _ = _load(path, SourceEncoding.AUTO)
    np.testing.assert_allclose(source[0, 0], (1.0, 0.0, 0.0), atol=1e-6)
    r, g, b = _load_working(path, SourceEncoding.AUTO)[0, 0]
    assert 0.70 < r < 0.73 and abs(g) < 0.01 and abs(b) < 0.01
    data[..., :] = 65535
    np.testing.assert_allclose(_load_working(_write(tmp_path, data, _icc("sRGB-v4.icc")), SourceEncoding.AUTO)[0, 0], 1.0, atol=2e-3)


def test_linear_and_gamma_override_the_profile(tmp_path) -> None:
    data = _rgb16()
    path = _write(tmp_path, data, _icc("sRGB-v4.icc"))
    linear, _ = _load(path, SourceEncoding.LINEAR)
    np.testing.assert_allclose(linear, data / 65535.0, atol=1e-6)
    gamma, metadata = _load(path, SourceEncoding.GAMMA_22)
    np.testing.assert_allclose(gamma, np.power(data / 65535.0, 2.2), atol=1e-6)
    assert metadata["color_space"] is None


def test_auto_without_a_profile_takes_the_untagged_rule(tmp_path) -> None:
    data8 = np.linspace(0, 255, 16 * 24 * 3).reshape(16, 24, 3).astype(np.uint8)
    path = str(tmp_path / "photo.tif")
    tifffile.imwrite(path, data8, photometric="rgb")
    f32, metadata = _load(path, SourceEncoding.AUTO)
    assert metadata["source_encoding"] == "srgb"
    # Mid-gray sRGB 128 decodes to about 0.216 linear before the primaries move it.
    assert f32.mean() < data8.mean() / 255.0


def test_the_negative_side_keeps_linear_raw(tmp_path) -> None:
    """Without positive_decode the loader is unchanged: Linear RAW still skips the profile."""
    data = _rgb16()
    ctx, metadata = TiffLoader().load(_write(tmp_path, data, _icc("AdobeCompat-v4.icc")), linear_raw=True)
    with ctx as raw:
        np.testing.assert_allclose(raw.data, data / 65535.0, atol=1e-7)
    assert metadata["color_space"] is None


def test_bundled_working_colorants_match_the_constant() -> None:
    profile = se.parse_profile(_icc("AdobeCompat-v4.icc"), 256)
    assert profile is not None and profile.to_working is None
    tags = se._tag_table(_icc("AdobeCompat-v4.icc"))
    cols = [[se._s15f16(tags[sig][8 + 4 * i : 12 + 4 * i]) for i in range(3)] for sig in (b"rXYZ", b"gXYZ", b"bXYZ")]
    np.testing.assert_allclose(np.array(cols).T, se._WORKING_COLORANTS_D50, atol=1e-5)


def test_positive_decode_and_tokens_follow_the_encoding() -> None:
    slide = ProcessConfig(process_mode=ProcessMode.E6, positive_source=True, source_encoding=SourceEncoding.GAMMA_22)
    assert positive_decode(slide) == "gamma22"
    assert positive_decode(replace(slide, positive_source=False)) == ""
    assert positive_decode(replace(slide, process_mode=ProcessMode.C41)) == ""
    assert linear_raw_token(slide) != linear_raw_token(replace(slide, source_encoding=SourceEncoding.AUTO))
    a = WorkspaceConfig(process=slide)
    assert source_token(a) != source_token(replace(a, process=replace(slide, source_encoding=SourceEncoding.LINEAR)))
    negative = WorkspaceConfig()
    assert "|pd" not in source_token(negative) and "|pd" not in linear_raw_token(negative.process)


def test_an_unknown_encoding_reads_as_auto() -> None:
    assert ProcessConfig(source_encoding="nonsense").source_encoding is SourceEncoding.AUTO


# camera.nef is a camera RAW; a Coolscan NEF is a scanner raw (test_a_slide_of_a_scanner_raw_starts_positive).
@pytest.mark.parametrize(
    "path, forced", [("a.tif", True), ("a.TIFF", True), ("a.jpg", True), ("a.jxl", True), ("camera.nef", False), ("a.dng", False)]
)
def test_a_slide_of_a_finished_image_starts_positive(path, forced) -> None:
    slide = replace(WorkspaceConfig(), process=ProcessConfig(process_mode=ProcessMode.E6))
    assert with_scanner_positive_default(slide, path).process.positive_source is forced
    assert with_scanner_positive_default(WorkspaceConfig(), path).process.positive_source is False


@pytest.mark.parametrize("loader", ["_nef", "_fff", "_pakon", "_noritsu"])
def test_a_slide_of_a_scanner_raw_starts_positive(monkeypatch, loader) -> None:
    from negpy.infrastructure.loaders import factory

    monkeypatch.setattr(factory.loader_factory, "_select_loader", lambda _path: getattr(factory.loader_factory, loader))
    is_scanned_positive_source.cache_clear()
    slide = replace(WorkspaceConfig(), process=ProcessConfig(process_mode=ProcessMode.E6))
    assert with_scanner_positive_default(slide, f"/scans/{loader}.raw").process.positive_source is True
    is_scanned_positive_source.cache_clear()


def test_a_linear_16_bit_decode_passes_the_samples_through(tmp_path) -> None:
    data = _rgb16()
    ctx, _ = TiffLoader().load(_write(tmp_path, data), positive_decode=SourceEncoding.LINEAR)
    assert ctx._source_u16 is not None
    with ctx as raw:
        np.testing.assert_array_equal(raw.postprocess(gamma=(1, 1), output_bps=16), data)


def test_out_of_gamut_colors_keep_their_hue() -> None:
    """A ProPhoto-like red outside Adobe RGB moves toward its luminance, not one channel at a time."""
    rgb = np.array([[[1.2, 0.3, -0.2]]], dtype=np.float32)
    out = se._into_gamut(rgb)
    assert out.min() >= 0.0 and out.max() <= 1.0
    y = float(rgb[0, 0] @ se._WORKING_COLORANTS_D50[1])
    d_in, d_out = rgb[0, 0] - y, out[0, 0] - y
    np.testing.assert_allclose(d_out / np.linalg.norm(d_out), d_in / np.linalg.norm(d_in), atol=1e-5)
    inside = np.array([[[0.5, 0.4, 0.2]]], dtype=np.float32)
    np.testing.assert_allclose(se._into_gamut(inside), inside, atol=1e-7)


def test_a_stale_deferred_reload_does_nothing() -> None:
    from types import SimpleNamespace

    from negpy.desktop.controller import AppController

    loads = []
    fake = SimpleNamespace(state=SimpleNamespace(current_file_path="/b.tif"), load_file=lambda path, **_: loads.append(path))
    AppController._reload_if_open(fake, "/a.tif")
    AppController._reload_if_open(fake, "/b.tif")
    assert loads == ["/b.tif"]


def test_auto_skips_a_scanners_own_profile(tmp_path) -> None:
    """A scnr-class profile describes the scanner, as SilverFast embeds in linear raw output."""
    icc = bytearray(_icc("AdobeCompat-v4.icc"))
    icc[12:16] = b"scnr"
    data = _rgb16()
    f32, metadata = _load(_write(tmp_path, data, bytes(icc)), SourceEncoding.AUTO)
    np.testing.assert_allclose(f32, data / 65535.0, atol=1e-6)
    assert metadata["source_encoding"] == "linear" and metadata["color_space"] is None
    gamma, _ = _load(_write(tmp_path, data, bytes(icc)), SourceEncoding.GAMMA_22)
    np.testing.assert_allclose(gamma, np.power(data / 65535.0, 2.2), atol=1e-6)


def test_encoding_follows_the_roll_and_old_rolls_get_auto(tmp_path) -> None:
    from negpy.infrastructure.storage.repository import StorageRepository
    from negpy.services.assets import rolls
    from negpy.services.assets.migrations.roll_fields import migrate_film_card_source_encoding

    repo = StorageRepository(str(tmp_path / "edits.db"), str(tmp_path / "settings.db"))
    repo.initialize()
    old = rolls.create_virtual_roll(repo, "Old", [])
    rolls.set_roll_defaults(repo, old, process_mode=ProcessMode.E6, positive_source=True)
    linear = rolls.create_virtual_roll(repo, "Linear", [])
    rolls.set_roll_defaults(repo, linear, process_mode=ProcessMode.E6, positive_source=True, source_encoding="linear")
    migrate_film_card_source_encoding(repo)
    assert rolls.roll_defaults(repo, old)["source_encoding"] == "auto"
    assert rolls.roll_defaults(repo, linear)["source_encoding"] == "linear"
    slide = replace(WorkspaceConfig(), process=ProcessConfig(process_mode=ProcessMode.E6, positive_source=True))
    assert rolls.resolve_roll_config(repo, linear, "h1", slide).process.source_encoding is SourceEncoding.LINEAR


def test_an_in_range_image_skips_the_gamut_pass(monkeypatch) -> None:
    inside = np.random.default_rng(1).random((8, 8, 3), dtype=np.float32)
    monkeypatch.setattr(se.np, "clip", lambda *a, **k: pytest.fail("the gamut pass ran"))
    assert se._into_gamut(inside) is inside
    monkeypatch.undo()
    outside = inside.copy()
    outside[0, 0] = (1.2, 0.3, -0.2)
    out = se._into_gamut(outside)
    y = float(outside[0, 0] @ se._WORKING_COLORANTS_D50[1])
    assert out.min() >= 0.0 and out.max() <= 1.0
    assert abs(out[0, 0, 0] - y) < abs(outside[0, 0, 0] - y)
    np.testing.assert_allclose(out[1:], inside[1:], atol=1e-6)


@pytest.mark.parametrize("composite", ["hdr", "stitch", "triplet"])
def test_a_composite_slide_takes_no_scanner_default(composite) -> None:
    from negpy.features.hdr.models import HdrConfig
    from negpy.features.rgbscan.models import RgbScanConfig
    from negpy.features.stitch.models import StitchConfig

    parts = {
        "hdr": {"hdr": HdrConfig(hdr_enabled=True, hdr_paths=("/x/a.tif", "/x/b.tif"))},
        "stitch": {"stitch": StitchConfig(stitch_enabled=True, stitch_paths=("/x/a.tif", "/x/b.tif"))},
        "triplet": {"rgbscan": RgbScanConfig(enabled=True, green_path="/x/g.tif", blue_path="/x/b.tif")},
    }[composite]
    slide = replace(WorkspaceConfig(), process=ProcessConfig(process_mode=ProcessMode.E6), **parts)
    assert with_scanner_positive_default(slide, "/x/a.tif").process.positive_source is False


def test_an_unreadable_file_is_not_cached_as_a_camera_raw(monkeypatch, tmp_path) -> None:
    from negpy.infrastructure.loaders import factory

    is_scanned_positive_source.cache_clear()
    path = str(tmp_path / "gone.nef")
    monkeypatch.setattr(factory.loader_factory, "_select_loader", lambda _path: factory.loader_factory._rawpy)
    assert is_scanned_positive_source(path) is False
    open(path, "wb").close()
    monkeypatch.setattr(factory.loader_factory, "_select_loader", lambda _path: factory.loader_factory._nef)
    assert is_scanned_positive_source(path) is True
    is_scanned_positive_source.cache_clear()


def _hydrating_session(monkeypatch, tmp_path, saved=None):
    """A session over a real repository that hydrates a frame in Slide mode, with *saved* as its edit."""
    import negpy.desktop.session as session_mod
    from negpy.desktop.session import AppState, DesktopSessionManager
    from negpy.features.process.models import with_process_mode
    from negpy.infrastructure.storage.repository import StorageRepository

    repo = StorageRepository(str(tmp_path / "edits.db"), str(tmp_path / "settings.db"))
    repo.initialize()
    session = DesktopSessionManager.__new__(DesktopSessionManager)
    session.state = AppState()
    session.repo = repo
    session._apply_sticky_settings = lambda c, only_global=False, path="": c if only_global else with_process_mode(c, ProcessMode.E6)
    monkeypatch.setattr(session_mod, "load_or_promote", lambda *a, **k: saved)
    return session


@pytest.mark.parametrize("path, positive", [("/x/scan.tif", True), ("/x/camera.nef", False)])
def test_a_first_open_scanner_slide_starts_positive(monkeypatch, tmp_path, path, positive) -> None:
    session = _hydrating_session(monkeypatch, tmp_path)
    assert session._hydrate_asset_config({"hash": "h1", "path": path}).process.positive_source is positive


def test_a_reopened_frame_keeps_positive_off(monkeypatch, tmp_path) -> None:
    off = replace(WorkspaceConfig(), process=ProcessConfig(process_mode=ProcessMode.E6, positive_source=False))
    session = _hydrating_session(monkeypatch, tmp_path)
    session.repo.save_file_settings("h1", off, file_path="/x/scan.tif")
    saved = session.repo.load_file_settings("h1")
    assert saved is not None and saved.process.positive_source is False
    session = _hydrating_session(monkeypatch, tmp_path, saved=saved)
    assert session._hydrate_asset_config({"hash": "h1", "path": "/x/scan.tif"}).process.positive_source is False


@pytest.mark.parametrize("roll_positive, expected", [(False, False), (True, True), (None, True)])
def test_a_roll_keeps_its_positive_choice(monkeypatch, tmp_path, roll_positive, expected) -> None:
    from negpy.services.assets import rolls

    session = _hydrating_session(monkeypatch, tmp_path)
    roll_id = rolls.create_virtual_roll(session.repo, "Faded", ["/x/scan.tif"])
    film = {"process_mode": ProcessMode.E6} | ({} if roll_positive is None else {"positive_source": roll_positive})
    rolls.set_roll_defaults(session.repo, roll_id, **film)
    session.state.active_roll_id = roll_id
    config = session._hydrate_asset_config({"hash": "h1", "path": "/x/scan.tif"})
    assert config.process.positive_source is expected


def _fake_controller(config: WorkspaceConfig, path: str):
    from types import SimpleNamespace

    applied: list[WorkspaceConfig] = []
    fake = SimpleNamespace(
        state=SimpleNamespace(config=config, current_file_path=path),
        apply_config=lambda c, persist=False: applied.append(c),
        _lock_roll_card=lambda *_args: None,
    )
    return fake, applied


@pytest.mark.parametrize("path, positive", [("/x/scan.tif", True), ("/x/camera.nef", False)])
def test_a_switch_to_slide_starts_a_scanner_file_positive(path, positive) -> None:
    from negpy.desktop.controller import AppController

    fake, applied = _fake_controller(WorkspaceConfig(), path)
    AppController.set_process_mode(fake, ProcessMode.E6)
    assert applied[-1].process.positive_source is positive


def test_slide_to_slide_keeps_positive_off() -> None:
    from negpy.desktop.controller import AppController

    off = replace(WorkspaceConfig(), process=ProcessConfig(process_mode=ProcessMode.E6, positive_source=False))
    fake, applied = _fake_controller(off, "/x/scan.tif")
    AppController.set_process_mode(fake, ProcessMode.E6)
    assert applied[-1].process.positive_source is False


def test_a_scanner_slide_can_turn_positive_off() -> None:
    from negpy.desktop.controller import AppController

    on = replace(WorkspaceConfig(), process=ProcessConfig(process_mode=ProcessMode.E6, positive_source=True))
    fake, applied = _fake_controller(on, "/x/scan.tif")
    AppController.set_positive_source(fake, False)
    assert applied[-1].process.positive_source is False


def _neutral_positive(fmt: ExportFormat, space: str) -> WorkspaceConfig:
    config = with_positive_source(replace(WorkspaceConfig(), process=ProcessConfig(process_mode=ProcessMode.E6)), True)
    return replace(
        config,
        exposure=replace(config.exposure, auto_exposure=False, auto_normalize_contrast=False, cast_removal_strength=0.0),
        lab=replace(config.lab, sharpen=0.0, chroma_denoise=0.0, skin_protection=0.0),
        export=replace(config.export, export_fmt=fmt, export_bit_depth=16, export_color_space=space, jpeg_quality=100),
    )


def _export(processor, path: str, config: WorkspaceConfig) -> bytes:
    data, error = processor.process_export(path, config, config.export, path, prefer_gpu=False)
    assert data is not None, error
    return data


@pytest.mark.parametrize("fmt, ext", [(ExportFormat.TIFF, "tif"), (ExportFormat.JPEG, "jpg")])
@pytest.mark.parametrize("space", [ColorSpace.ADOBE_RGB.value, ColorSpace.SRGB.value])
def test_an_export_reopened_as_positive_exports_the_same_pixels(tmp_path, fmt, ext, space) -> None:
    """With the Positive edits at neutral, the decode inverts the export encoding. The colors stay
    inside sRGB: an 8-bit export near the sRGB gamut edge has its own rounding error."""
    from negpy.services.rendering.image_processor import ImageProcessor

    y, x = np.mgrid[0:48, 0:64] / np.array([48.0, 64.0])[:, None, None]
    encoded = np.stack([0.2 + 0.6 * x, 0.5 + 0.15 * np.sin(3 * y + 1), 0.3 + 0.4 * x * y], axis=-1)
    source = _write(tmp_path, np.round(encoded * 65535).astype(np.uint16), _icc("AdobeCompat-v4.icc"))
    processor = ImageProcessor()
    config = _neutral_positive(fmt, space)
    first = tmp_path / f"first.{ext}"
    first.write_bytes(_export(processor, source, config))
    second = _export(processor, str(first), config)

    def pixels(data: bytes) -> np.ndarray:
        img = tifffile.imread(io.BytesIO(data)) if fmt is ExportFormat.TIFF else np.asarray(Image.open(io.BytesIO(data)))
        return img.astype(np.float64) / np.iinfo(img.dtype).max

    diff = np.abs(pixels(second) - pixels(first.read_bytes())) * 255.0
    assert diff.mean() < 1.0 and np.percentile(diff, 99) < 4.0


def test_a_positive_jpeg_decodes_by_auto_whatever_the_stored_encoding(tmp_path) -> None:
    from negpy.infrastructure.loaders.jpeg_loader import JpegLoader

    data = np.random.default_rng(3).integers(0, 256, (16, 24, 3), dtype=np.uint8)
    path = str(tmp_path / "export.jpg")
    Image.fromarray(data).save(path, quality=100, subsampling=0, icc_profile=_icc("AdobeCompat-v4.icc"))
    decoded = np.asarray(Image.open(path)).astype(np.float32) / 255.0
    for encoding in (SourceEncoding.AUTO, SourceEncoding.LINEAR):
        ctx, metadata = JpegLoader().load(path, positive_decode=encoding)
        with ctx as raw:
            np.testing.assert_allclose(raw.data, working_oetf_decode(decoded), atol=5e-4)
        assert metadata["source_encoding"] == "profile"
