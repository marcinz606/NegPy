import os
import struct
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import tifffile

from negpy.features.process.models import DemosaicMode
from negpy.infrastructure.loaders import pixel_shift, raw_shots
from negpy.infrastructure.loaders.helpers import NonStandardFileWrapper, camera_wb_multipliers, camera_xyz_matrix, resolve_demosaic
from negpy.infrastructure.loaders.pixel_shift import (
    LIGHT_WARNING,
    MISMATCH_WARNING,
    SHOT_SHIFT,
    PixelShiftImage,
    PixelShiftStats,
    PixelShiftWrapper,
    combine_shots,
    libraw_wb_gains,
    shot_levels,
)
from negpy.infrastructure.loaders.raw_shots import (
    CombinedShots,
    ShotKind,
    ShotLayout,
    ShotsInfo,
    SingleShot,
    composite_shots,
    makernote_quality,
    probe_shots,
    shot_source,
)

RGGB = np.array([[0, 1], [3, 2]])
BLACK = (256, 256, 256, 256)


@pytest.fixture(autouse=True)
def _fresh_caches():
    pixel_shift.clear_cache()
    raw_shots._probe.cache_clear()
    yield
    pixel_shift.clear_cache()
    raw_shots._probe.cache_clear()


def _scene(h: int, w: int, seed: int = 1) -> np.ndarray:
    """A known RGB scene in sensor counts above black, with texture on every scale."""
    rng = np.random.default_rng(seed)
    base = rng.integers(500, 12000, size=(h, w, 3))
    return base.astype(np.int64)


def _shots_of(scene: np.ndarray, shift=SHOT_SHIFT, cfa=RGGB, black=BLACK) -> np.ndarray:
    """Sample `scene` the way the camera does: shot k at photosite (y, x) reads the scene at
    (y - dy, x - dx) through the photosite's own color filter."""
    h, w, _ = scene.shape
    color = np.where(np.tile(cfa, (h // 2 + 1, w // 2 + 1))[:h, :w] == 3, 1, np.tile(cfa, (h // 2 + 1, w // 2 + 1))[:h, :w])
    site = np.tile(cfa, (h // 2 + 1, w // 2 + 1))[:h, :w]
    shots = np.zeros((len(shift), h, w), np.uint16)
    yy, xx = np.mgrid[0:h, 0:w]
    for k, (dy, dx) in enumerate(shift):
        sy, sx = np.clip(yy - dy, 0, h - 1), np.clip(xx - dx, 0, w - 1)
        shots[k] = scene[sy, sx, color] + np.array(black)[site]
    return shots


def test_combine_returns_the_scene_exactly():
    scene = _scene(32, 48)
    counts, _check = combine_shots(_shots_of(scene), RGGB, BLACK)
    assert counts.shape == (32, 48, 3) and counts.dtype == np.uint16
    inner = (slice(0, 31), slice(0, 47))
    assert np.array_equal(counts[inner][..., 0], scene[inner][..., 0])
    assert np.array_equal(counts[inner][..., 2], scene[inner][..., 2])
    assert np.array_equal(counts[inner][..., 1], 2 * scene[inner][..., 1])
    # The last row and column copy their neighbors.
    assert np.array_equal(counts[31], counts[30]) and np.array_equal(counts[:, 47], counts[:, 46])


def test_swapped_shots_do_not_return_the_scene():
    scene = _scene(32, 48)
    shots = _shots_of(scene)
    counts, _check = combine_shots(shots[[0, 3, 2, 1]], RGGB, BLACK)
    assert not np.array_equal(counts[:31, :47, 0], scene[:31, :47, 0])


def test_combine_clips_counts_below_black_to_zero():
    shots = np.full((4, 8, 8), 100, np.uint16)
    counts, _check = combine_shots(shots, RGGB, BLACK)
    assert counts.max() == 0


def test_combine_rejects_a_cfa_without_two_greens():
    with pytest.raises(ValueError):
        combine_shots(np.zeros((4, 8, 8), np.uint16), np.array([[0, 0], [2, 2]]), BLACK)


def test_aligned_still_shots_agree():
    rng = np.random.default_rng(3)
    scene = _scene(256, 256)
    shots = _shots_of(scene).astype(np.int64)
    shots += rng.normal(0, 20, size=shots.shape).astype(np.int64)  # sensor noise, independent per shot
    _counts, check = combine_shots(np.clip(shots, 0, 65535).astype(np.uint16), RGGB, BLACK)
    mismatch, _pairs, (gain, floor) = check.measure()
    assert mismatch < MISMATCH_WARNING
    assert floor >= 1.0 and gain >= 0.0


def _blocks(h: int, w: int, size: int = 16, seed: int = 2) -> np.ndarray:
    """Flat patches with sharp edges: where the noise model reads its flat pixels."""
    rng = np.random.default_rng(seed)
    levels = rng.integers(500, 12000, size=(h // size + 1, w // size + 1, 3))
    return np.repeat(np.repeat(levels, size, axis=0), size, axis=1)[:h, :w].astype(np.int64)


def test_a_shot_that_moved_disagrees():
    scene = _blocks(256, 256)
    shift = list(SHOT_SHIFT)
    shift[2] = (2, 2)  # the third shot landed one photosite off
    _counts, check = combine_shots(_shots_of(scene, shift=tuple(shift)), RGGB, BLACK)
    mismatch, (pair_13, pair_24), _noise = check.measure()
    assert mismatch > MISMATCH_WARNING
    assert pair_13 > pair_24  # the third shot pairs with the first


def test_stats_flag_a_light_change():
    stats = PixelShiftStats(0.0, (0.0, 0.0), (1.0, 1.0, 1.0 + 2 * LIGHT_WARNING, 1.0), (1.0, 1.0), 0.1)
    assert stats.shots_disagree
    assert not PixelShiftStats(0.0, (0.0, 0.0), (1.0, 1.0, 1.0, 1.0), (1.0, 1.0), 0.1).shots_disagree


def test_shot_levels_are_relative_to_the_first_shot():
    shots = np.stack([np.full((4, 4), 256 + v, np.uint16) for v in (1000, 1000, 1050, 950)])
    assert shot_levels(shots, BLACK) == pytest.approx((1.0, 1.0, 1.05, 0.95))


def _image(counts: np.ndarray, **kw: Any) -> PixelShiftImage:
    values: dict[str, Any] = dict(
        counts=counts,
        render_white=16119,
        export_white=16127,
        rgb_xyz_matrix=np.eye(4, 3),
        camera_whitebalance=(26048.0, 8192.0, 14208.0, 8160.0),
        daylight_whitebalance=(2.0, 1.0, 1.5, 0.0),
        sizes=SimpleNamespace(height=counts.shape[0], width=counts.shape[1], iheight=counts.shape[0], iwidth=counts.shape[1]),
        other=SimpleNamespace(timestamp=None),
        stats=PixelShiftStats(0.0, (0.0, 0.0), (1.0, 1.0, 1.0, 1.0), (2.0, 800.0), 1.0),
    )
    values.update(kw)
    return PixelShiftImage(**values)


def test_to_u16_scales_like_libraw():
    counts = np.array([[[1, 2, 3], [8000, 16000, 16119]], [[16119, 32238, 20000], [0, 0, 0]]], np.uint16)
    u16 = _image(counts).to_u16(16119)
    expected_r = np.floor(counts[..., 0].astype(np.float32) * np.float32(65535 / 16119))
    expected_g = np.floor(counts[..., 1].astype(np.float32) * np.float32(65535 / (2 * 16119)))
    assert np.array_equal(u16[..., 0], np.minimum(expected_r, 65535).astype(np.uint16))
    assert np.array_equal(u16[..., 1], np.minimum(expected_g, 65535).astype(np.uint16))
    assert u16[1, 0, 0] == 65535 and u16[1, 0, 1] == 65535 and u16[1, 0, 2] == 65535


def test_libraw_wb_gains_divide_by_the_smallest_of_four():
    assert libraw_wb_gains((26048.0, 8192.0, 14208.0, 8160.0)) == pytest.approx((26048 / 8160, 8176 / 8160, 14208 / 8160))
    assert libraw_wb_gains((2.0, 1.0, 1.5)) == pytest.approx((2.0, 1.0, 1.5))
    assert libraw_wb_gains([1, 1, 1, 1]) == (1.0, 1.0, 1.0)
    assert libraw_wb_gains(None) == (1.0, 1.0, 1.0)
    assert libraw_wb_gains((0.0, 1.0, 1.0)) == (1.0, 1.0, 1.0)


def test_wrapper_reads_like_a_camera_raw():
    counts = np.full((8, 12, 3), 4000, np.uint16)
    wrapper = PixelShiftWrapper(_image(counts), 16119)
    assert isinstance(wrapper, NonStandardFileWrapper)
    assert resolve_demosaic(wrapper, DemosaicMode.AHD)[1] is None
    assert camera_xyz_matrix(wrapper) == np.eye(3).tolist()
    assert camera_wb_multipliers(wrapper) == [26048.0, 8192.0, 14208.0]
    linear = wrapper.postprocess(gamma=(1, 1), output_bps=16, user_wb=[1, 1, 1, 1], use_camera_wb=False)
    assert linear.dtype == np.uint16 and linear.shape == (8, 12, 3)
    assert np.array_equal(linear, _image(counts).to_u16(16119))
    balanced = wrapper.postprocess(gamma=(1, 1), output_bps=16, use_camera_wb=True)
    assert balanced[0, 0, 0] > linear[0, 0, 0] and balanced[0, 0, 2] > linear[0, 0, 2]
    pinned = wrapper.postprocess(gamma=(1, 1), output_bps=16, use_camera_wb=False, user_wb=[2.0, 1.0, 1.5, 1.0])
    assert abs(int(pinned[0, 0, 0]) - 2 * int(linear[0, 0, 0])) <= 1
    assert wrapper.postprocess(half_size=True, gamma=(1, 1), output_bps=16).shape == (4, 6, 3)
    assert wrapper.postprocess().dtype == np.uint8
    assert wrapper.data.shape == (8, 12, 3) and wrapper.data.dtype == np.float32


def _makernote(quality: int, order: bytes = b"II") -> bytes:
    end = "<" if order == b"II" else ">"
    entry = struct.pack(end + "HHI", 0x0008, 3, 1) + struct.pack(end + "H", quality) + b"\0\0"
    return b"PENTAX \0" + order + struct.pack(end + "H", 1) + entry + b"\0\0\0\0"


def test_makernote_quality_reads_every_header():
    assert makernote_quality(_makernote(7)) == 7
    assert makernote_quality(_makernote(6, b"MM")) == 6
    aoc = b"AOC\0II" + _makernote(7)[10:]
    assert makernote_quality(aoc) == 7
    adobe = b"Adobe\0MakN" + struct.pack(">I", 100) + b"II" + b"\0\0\0\0" + _makernote(7)
    assert makernote_quality(adobe) == 7
    assert makernote_quality(b"NIKON\0\0") is None
    assert makernote_quality(b"PENTAX \0II\x05") is None


_CFA_TAGS = [
    (33421, "H", 2, (2, 2), True),  # CFARepeatPatternDim
    (33422, "B", 4, (0, 1, 1, 2), True),  # CFAPattern RGGB
    (50714, "H", 1, (256,), True),  # BlackLevel
    (50717, "H", 1, (16383,), True),  # WhiteLevel
]


def _write_multishot_dng(path, shots: np.ndarray, make: str = "PENTAX", quality: int = 7) -> None:
    """A Pentax-style DNG: thumbnail IFD0 with the makernote in DNGPrivateData, one CFA SubIFD per shot."""
    ifd0 = [
        (50706, "B", 4, (1, 4, 0, 0), True),  # DNGVersion
        (50708, "s", 0, "Sim", True),  # UniqueCameraModel
        (50721, "2i", 9, (1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1), True),  # ColorMatrix1
        (50728, "2I", 3, (1, 1, 1, 1, 1, 1), True),  # AsShotNeutral
        (271, "s", 0, make, True),  # Make
        (50740, "B", len(_makernote(quality)), _makernote(quality), True),  # DNGPrivateData
    ]
    with tifffile.TiffWriter(path) as tw:
        tw.write(np.zeros((8, 12, 3), np.uint8), photometric="rgb", subfiletype=1, subifds=len(shots), extratags=ifd0)
        for shot in shots:
            tw.write(shot, photometric=32803, subfiletype=0, extratags=_CFA_TAGS)


def test_probe_finds_a_pixel_shift_dng(tmp_path):
    path = tmp_path / "ps.dng"
    _write_multishot_dng(path, _shots_of(_scene(32, 48)))
    assert probe_shots(str(path)) == ShotLayout(4, ShotKind.PIXEL_SHIFT)


def test_probe_reads_other_multishot_files_as_shots(tmp_path):
    hdr = tmp_path / "hdr.dng"
    _write_multishot_dng(hdr, _shots_of(_scene(32, 48))[:3], quality=6)
    assert probe_shots(str(hdr)) == ShotLayout(3, ShotKind.SHOTS)
    other = tmp_path / "other.dng"
    _write_multishot_dng(other, _shots_of(_scene(32, 48)), make="Sim")
    assert probe_shots(str(other)) == ShotLayout(4, ShotKind.SHOTS)


def test_probe_ignores_single_shot_unreadable_and_other_files(tmp_path):
    single = tmp_path / "one.dng"
    _write_multishot_dng(single, _shots_of(_scene(32, 48))[:1])
    assert probe_shots(str(single)) is None
    assert probe_shots(str(tmp_path / "missing.dng")) is None
    broken = tmp_path / "broken.pef"
    broken.write_bytes(b"not a tiff")
    assert probe_shots(str(broken)) is None
    assert probe_shots(str(tmp_path / "frame.cr3")) is None


def test_shot_source_follows_the_choice(monkeypatch):
    layout = ShotLayout(4, ShotKind.PIXEL_SHIFT)
    monkeypatch.setattr(raw_shots, "probe_shots", lambda _path: layout)
    assert isinstance(shot_source("x.pef", 0), CombinedShots)
    single = shot_source("x.pef", 3)
    assert isinstance(single, SingleShot) and single.index == 2
    assert isinstance(shot_source("x.pef", 9), CombinedShots)  # a stale shot number reads as Auto
    monkeypatch.setattr(raw_shots, "probe_shots", lambda _path: ShotLayout(3, ShotKind.SHOTS))
    auto = shot_source("x.pef", 0)
    assert isinstance(auto, SingleShot) and auto.index == 0
    monkeypatch.setattr(raw_shots, "probe_shots", lambda _path: None)
    assert shot_source("x.pef", 0) is None


def test_one_combine_per_file(monkeypatch, tmp_path):
    path = tmp_path / "ps.pef"
    path.write_bytes(b"x")
    built = []

    def fake_build(file_path, _should_cancel):
        built.append(file_path)
        return _image(np.zeros((4, 4, 3), np.uint16))

    monkeypatch.setattr(pixel_shift, "_build", fake_build)
    first = pixel_shift.combined_image(str(path))
    assert pixel_shift.combined_image(str(path)) is first
    assert len(built) == 1
    os.utime(path, ns=(1, 1))  # a new modification time is a new file
    pixel_shift.combined_image(str(path))
    assert len(built) == 2


def test_combine_from_a_synthetic_dng(tmp_path):
    """End to end through LibRaw: the SubIFDs of a DNG are its shots."""
    path = tmp_path / "ps.dng"
    scene = _scene(32, 48)
    _write_multishot_dng(path, _shots_of(scene))
    source = shot_source(str(path), 0)
    assert isinstance(source, CombinedShots)
    wrapper, info = source.open()
    assert info.shot == 0 and info.count == 4 and info.stats is not None
    counts = wrapper.image.counts
    h, w = counts.shape[:2]
    assert np.array_equal(counts[: h - 1, : w - 1, 0], scene[: h - 1, : w - 1, 0])
    shot2, info2 = shot_source(str(path), 2).open()
    with shot2:
        assert info2.shot == 2
        assert not np.array_equal(shot2.raw_image_visible, _shots_of(scene)[0])


@pytest.mark.slow
def test_a_real_pixel_shift_raw():
    """Set NEGPY_PIXEL_SHIFT_RAW to a Pentax Pixel Shift PEF or DNG shot with a self-timer."""
    path = os.environ.get("NEGPY_PIXEL_SHIFT_RAW")
    if not path:
        pytest.skip("NEGPY_PIXEL_SHIFT_RAW is not set")
    source = shot_source(path, 0)
    assert isinstance(source, CombinedShots)
    wrapper, info = source.open()
    with raw_shots.open_shot(path, 0) as first:
        assert wrapper.image.counts.shape[:2] == first.raw_image_visible.shape
    assert not info.stats.shots_disagree


def test_rawpy_loader_combines_only_when_the_edit_asks(tmp_path):
    from negpy.infrastructure.loaders.rawpy_loader import RawpyLoader

    path = tmp_path / "ps.dng"
    scene = _scene(32, 48)
    _write_multishot_dng(path, _shots_of(scene))
    decoder, metadata = RawpyLoader().load(str(path), raw_shot=0)
    assert isinstance(decoder, PixelShiftWrapper)
    assert metadata["raw_shots"].shot == 0 and metadata["raw_shots"].stats is not None
    assert metadata["raw_flip"] == 0 and metadata["color_space"] is None and metadata["ir"] is None
    assert "lens_correction" in metadata
    plain, plain_meta = RawpyLoader().load(str(path))
    with plain:
        assert not isinstance(plain, NonStandardFileWrapper)
        assert "raw_shots" not in plain_meta
    shot, shot_meta = RawpyLoader().load(str(path), preview_max_edge=64, raw_shot=3)
    with shot:
        assert not isinstance(shot, NonStandardFileWrapper)
        assert shot_meta["raw_shots"].shot == 3


def test_rawpy_loader_falls_back_to_the_first_shot(tmp_path, monkeypatch):
    from negpy.infrastructure.loaders.rawpy_loader import RawpyLoader

    path = tmp_path / "ps.dng"
    _write_multishot_dng(path, _shots_of(_scene(32, 48)))

    def broken(*_args, **_kw):
        raise ValueError("combine failed")

    monkeypatch.setattr(pixel_shift, "_build", broken)
    decoder, metadata = RawpyLoader().load(str(path), raw_shot=0)
    with decoder:
        assert not isinstance(decoder, NonStandardFileWrapper)
        assert metadata["raw_shots"].failed and metadata["raw_shots"].shot == 1


def test_linear_export_uses_the_export_white(tmp_path):
    from negpy.services.export.linear_output import _decode_camera_raw_buffer

    path = tmp_path / "ps.dng"
    scene = _scene(32, 48)
    _write_multishot_dng(path, _shots_of(scene))
    f32, _wb, meta, _lens = _decode_camera_raw_buffer(str(path), raw_shot=0)
    assert meta.demosaic == "Pixel Shift"
    image = pixel_shift.combined_image(str(path))
    expected = image.to_u16(image.export_white).astype(np.float32) / 65535.0
    assert np.allclose(f32, expected, atol=1 / 65535)
    _f32, _wb2, meta2, _lens2 = _decode_camera_raw_buffer(str(path), raw_shot=2)
    assert meta2.demosaic.endswith("shot 2")


def test_a_thumbnail_never_combines(tmp_path):
    path = tmp_path / "ps.dng"
    _write_multishot_dng(path, _shots_of(_scene(32, 48)))
    assert raw_shots.thumbnail_shot(str(path), 0) == 1
    assert raw_shots.thumbnail_shot(str(path), 3) == 3
    plain = tmp_path / "one.dng"
    _write_multishot_dng(plain, _shots_of(_scene(32, 48))[:1])
    assert raw_shots.thumbnail_shot(str(plain), 0) == 0


def test_the_shot_choice_is_a_decode_input():
    from dataclasses import replace

    from negpy.domain.models import WorkspaceConfig
    from negpy.services.rendering.source_identity import source_token

    config = WorkspaceConfig()
    other = replace(config, process=replace(config.process, raw_shot=2))
    assert source_token(config) != source_token(other)
    assert replace(config.process, raw_shot="junk").raw_shot == 0
    assert replace(config.process, raw_shot=-3).raw_shot == 0
    assert WorkspaceConfig.from_flat_dict(other.to_dict()).process.raw_shot == 2


def test_shot_texts():
    from negpy.desktop.session import raw_shots_details, raw_shots_summary, raw_shots_warning
    from negpy.infrastructure.loaders.raw_shots import ShotsInfo

    good = PixelShiftStats(0.0001, (0.0001, 0.0001), (1.0, 1.0, 1.0, 1.0), (2.0, 800.0), 3.2)
    moved = PixelShiftStats(0.05, (0.07, 0.03), (1.0, 1.0, 1.0, 1.0), (2.0, 800.0), 3.2)
    assert raw_shots_summary(ShotsInfo(4, ShotKind.PIXEL_SHIFT, 0, good)) == "Pixel Shift: 4 shots combined."
    assert raw_shots_summary(ShotsInfo(3, ShotKind.SHOTS, 2)) == "Shot 2 of 3."
    assert raw_shots_summary(None) == ""
    assert raw_shots_warning(ShotsInfo(4, ShotKind.PIXEL_SHIFT, 0, good)) == ""
    assert "5.00%" in raw_shots_warning(ShotsInfo(4, ShotKind.PIXEL_SHIFT, 0, moved))
    assert "3.2 s" in raw_shots_details(ShotsInfo(4, ShotKind.PIXEL_SHIFT, 0, good))
    assert raw_shots_details(ShotsInfo(3, ShotKind.SHOTS, 1)) == ""


def test_a_failed_decode_warns():
    from negpy.desktop.session import raw_shots_warning
    from negpy.infrastructure.loaders.raw_shots import ShotsInfo

    assert "first shot" in raw_shots_warning(ShotsInfo(4, ShotKind.PIXEL_SHIFT, 1, failed=True))


def test_a_failed_decode_is_not_cached(tmp_path, monkeypatch):
    from negpy.services.rendering.preview_manager import PreviewManager

    path = tmp_path / "ps.dng"
    _write_multishot_dng(path, _shots_of(_scene(32, 48)))

    def broken(*_args, **_kw):
        raise ValueError("combine failed")

    monkeypatch.setattr(pixel_shift, "_build", broken)
    manager = PreviewManager()
    manager.load_linear_preview(str(path), color_space="Adobe RGB", file_hash="h", raw_shot=0)
    assert len(manager._cache._data) == 0
    monkeypatch.undo()
    manager.load_linear_preview(str(path), color_space="Adobe RGB", file_hash="h", raw_shot=0)
    assert len(manager._cache._data) == 1


def _info(failed: bool = False, mismatch: float = 0.0, light: float = 0.0) -> ShotsInfo:
    stats = PixelShiftStats(mismatch, (mismatch, mismatch), (1.0, 1.0 + light, 1.0, 1.0), (2.0, 1.0), 0.0)
    return ShotsInfo(count=4, kind=ShotKind.PIXEL_SHIFT, shot=1 if failed else 0, stats=None if failed else stats, failed=failed)


def test_a_composite_reports_its_worst_part():
    still, moving, flicker, broken = _info(), _info(mismatch=4 * MISMATCH_WARNING), _info(light=3 * LIGHT_WARNING), _info(failed=True)
    assert composite_shots([still, flicker, moving]) is moving
    assert composite_shots([still, flicker]) is flicker
    assert composite_shots([moving, broken]) is broken
    assert composite_shots([None, still, _info()]) is still
    assert composite_shots([None, None]) is None


def _composite_manager(tmp_path, infos):
    """A PreviewManager whose single-file decode returns each part with its own shots."""
    from unittest.mock import MagicMock

    from negpy.services.rendering.preview_manager import PreviewManager

    paths = [tmp_path / f"part{i}.pef" for i in range(len(infos))]
    for path in paths:
        path.write_bytes(b"x")
    shots = dict(zip(map(str, paths), infos))

    def decode(path, *_args, **_kwargs):
        meta = {} if shots[path] is None else {"raw_shots": shots[path]}
        return np.full((80, 80, 3), 0.4, dtype=np.float32), (80, 80), meta

    manager = PreviewManager()
    manager.load_linear_preview = MagicMock(side_effect=decode)
    return manager, [str(p) for p in paths]


def test_a_triplet_with_a_failed_exposure_warns_and_decodes_again(tmp_path):
    from negpy.features.rgbscan.models import RgbScanConfig

    manager, (red, green, blue) = _composite_manager(tmp_path, (_info(), _info(failed=True), _info()))
    rgbscan = RgbScanConfig(enabled=True, green_path=green, blue_path=blue, align=False)
    _out, _dims, meta = manager.load_linear_preview_rgb(red, rgbscan, "Adobe RGB", file_hash="red")
    assert meta["raw_shots"].failed
    manager.load_linear_preview_rgb(red, rgbscan, "Adobe RGB", file_hash="red")
    assert manager.load_linear_preview.call_count == 6


def test_a_triplet_with_a_moving_exposure_warns_and_stays_cached(tmp_path):
    from negpy.features.rgbscan.models import RgbScanConfig

    moving = _info(mismatch=4 * MISMATCH_WARNING)
    manager, (red, green, blue) = _composite_manager(tmp_path, (_info(), _info(), moving))
    rgbscan = RgbScanConfig(enabled=True, green_path=green, blue_path=blue, align=False)
    _out, _dims, meta = manager.load_linear_preview_rgb(red, rgbscan, "Adobe RGB", file_hash="red")
    _out, _dims, again = manager.load_linear_preview_rgb(red, rgbscan, "Adobe RGB", file_hash="red")
    assert meta["raw_shots"] is moving and again["raw_shots"] is moving
    assert manager.load_linear_preview.call_count == 3


def test_a_bracket_with_a_failed_frame_warns_and_decodes_again(tmp_path):
    from negpy.features.hdr.models import HdrConfig

    manager, (reference, sibling) = _composite_manager(tmp_path, (_info(), _info(failed=True)))
    hdr = HdrConfig(hdr_enabled=True, hdr_paths=(sibling,), hdr_ratios=(1.0, 2.0), hdr_align=False)
    _out, _dims, meta = manager.load_linear_preview_hdr(reference, hdr, "Adobe RGB", file_hash="ref")
    assert meta["raw_shots"].failed
    manager.load_linear_preview_hdr(reference, hdr, "Adobe RGB", file_hash="ref")
    assert manager.load_linear_preview.call_count == 4


def test_a_stitch_with_a_failed_part_warns_and_decodes_again(tmp_path):
    from negpy.features.stitch.models import StitchConfig

    manager, (first, second) = _composite_manager(tmp_path, (None, _info(failed=True)))
    stitch = StitchConfig(
        stitch_enabled=True,
        stitch_paths=(second,),
        stitch_transforms=((1.0, 0.0, 0.0, 0.0, 1.0, 0.0), (1.0, 0.0, 40.0, 0.0, 1.0, 0.0)),
        stitch_canvas=(120, 80),
        stitch_sizes=((80, 80), (80, 80)),
    )
    _out, _dims, meta = manager.load_linear_preview_stitch(first, stitch, "Adobe RGB", file_hash="first#stitch")
    assert meta["raw_shots"].failed
    manager.load_linear_preview_stitch(first, stitch, "Adobe RGB", file_hash="first#stitch")
    assert manager.load_linear_preview.call_count == 4


def test_one_combine_after_a_cancelled_one(monkeypatch, tmp_path):
    import threading

    path = tmp_path / "ps.pef"
    path.write_bytes(b"x")
    started, release = threading.Event(), threading.Event()
    builds = []

    def fake_build(_file_path, should_cancel):
        builds.append(should_cancel)
        if should_cancel is not None:
            raise InterruptedError("cancelled")
        started.set()
        release.wait(5)
        return _image(np.zeros((4, 4, 3), np.uint16))

    monkeypatch.setattr(pixel_shift, "_build", fake_build)
    with pytest.raises(InterruptedError):
        pixel_shift.combined_image(str(path), should_cancel=lambda: True)
    first = threading.Thread(target=pixel_shift.combined_image, args=(str(path),))
    first.start()
    started.wait(5)
    second = threading.Thread(target=pixel_shift.combined_image, args=(str(path),))
    second.start()
    release.set()
    first.join(5)
    second.join(5)
    assert len(builds) == 2  # the cancelled build and one real build, shared by both callers
