"""Imported Camera Raw Look profiles as print stocks (rgb_table.py)."""

import hashlib
import struct
import unittest
import zlib
from dataclasses import replace
from unittest.mock import MagicMock

import numpy as np
import pytest

from negpy.domain.models import WorkspaceConfig
from negpy.features.exposure import rgb_table
from negpy.features.exposure.models import EXPOSURE_CONSTANTS
from negpy.features.exposure.papers import (
    DEFAULT_PROFILE_KEY,
    PAPER_PROFILES,
    exposure_paper,
    profiles_for_mode,
    resolve_flat_matrix,
    resolve_paper,
    resolve_print_color,
)
from negpy.features.exposure.print_lut import DyeLut, tetra_lookup
from negpy.features.exposure.rgb_table import (
    LOOK_KEY_PREFIX,
    apply_rgb_table,
    b85_decode,
    b85_encode,
    delete_look,
    import_look_profile,
    list_looks,
    look_on_digital,
    look_print_lut,
    look_table_lut,
    read_look_profile,
)
from negpy.features.process.models import ProcessMode
from negpy.infrastructure.gpu.device import GPUDevice
from negpy.kernel.image.logic import rgb_to_lab_working
from negpy.kernel.system.config import APP_CONFIG

PROPHOTO_18 = (2, 2, 0)
SRGB_EXTEND = (0, 1, 1)


def _identity(n: int = 17) -> np.ndarray:
    axis = np.linspace(0.0, 1.0, n, dtype=np.float32)
    return np.ascontiguousarray(np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), -1))


def _warm(n: int = 17) -> np.ndarray:
    """A test look: red up and blue down in the midtones, black and white held."""
    t = _identity(n).astype(np.float64)
    bump = 4 * t * (1 - t)
    t[..., 0] += 0.06 * bump[..., 0]
    t[..., 2] -= 0.06 * bump[..., 2]
    return np.clip(t, 0, 1)


def _payload(values: np.ndarray, space=PROPHOTO_18, amounts=(0.0, 2.0)) -> bytes:
    """The dng_rgb_table stream of an absolute (N, N, N, 3) table in 0..1."""
    n = values.shape[0]
    u16 = np.round(np.clip(values, 0, 1) * 65535).astype(np.int64)
    ident = (np.arange(n) * 0xFFFF + (n >> 1)) // (n - 1)
    nominal = np.stack(np.meshgrid(ident, ident, ident, indexing="ij"), -1)
    deltas = ((u16 - nominal) & 0xFFFF).astype("<u2")
    return struct.pack("<4I", 1, 1, 3, n) + deltas.tobytes() + struct.pack("<3I", *space) + struct.pack("<2d", *amounts)


def _xmp(values: np.ndarray, name: str = "Test Warm", attrs: str = "", **kw) -> str:
    payload = _payload(values, **kw)
    table_id = hashlib.md5(payload).hexdigest().upper()
    encoded = b85_encode(struct.pack("<I", len(payload)) + zlib.compress(payload))
    return (
        f'<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF><rdf:Description crs:PresetType="Look" {attrs} '
        f'crs:RGBTable="{table_id}" crs:Table_{table_id}="{encoded}">'
        f'<crs:Name><rdf:Alt><rdf:li xml:lang="x-default">{name}</rdf:li></rdf:Alt></crs:Name>'
        "</rdf:Description></rdf:RDF></x:xmpmeta>"
    )


@pytest.fixture
def looks(tmp_path, monkeypatch):
    """An empty looks folder; returns a function that writes and imports an .xmp."""
    monkeypatch.setattr(APP_CONFIG, "looks_dir", str(tmp_path / "looks"))
    for fn in (rgb_table.load_look, rgb_table.list_looks, rgb_table._look_lut):
        fn.cache_clear()

    def add(values=None, name="Test Warm", **kw):
        path = tmp_path / f"{name}.xmp"
        path.write_text(_xmp(_warm() if values is None else values, name, **kw), encoding="utf-8")
        return import_look_profile(str(path))

    yield add
    for fn in (rgb_table.load_look, rgb_table.list_looks, rgb_table._look_lut):
        fn.cache_clear()


def test_base85_round_trips_every_tail_length():
    for n in range(0, 13):
        data = bytes(range(n))
        assert b85_decode(b85_encode(data)) == data


def test_a_look_profile_reads_back_its_name_id_and_table():
    values = _warm()
    name, table_id, table = read_look_profile(_xmp(values, "Warm Look"))
    assert name == "Warm Look"
    assert table_id == hashlib.md5(_payload(values)).hexdigest().upper()
    assert table.space == PROPHOTO_18 and table.amounts == (0.0, 2.0)
    np.testing.assert_array_equal(table.values, np.round(values * 65535).astype(np.uint16))


def test_a_damaged_or_tableless_profile_is_refused():
    text = _xmp(_warm())
    with pytest.raises(ValueError):
        read_look_profile(text.replace('crs:RGBTable="', 'crs:Other="'))
    start = text.index("crs:Table_") + 50
    with pytest.raises(ValueError):
        read_look_profile(text[:start] + "00000" + text[start + 5 :])


def test_a_profile_that_shares_another_files_table_is_refused_with_why():
    text = _xmp(_warm())
    shared = text[: text.index("crs:Table_")] + text[text.index('">', text.index("crs:Table_")) + 1 :]
    with pytest.raises(ValueError, match="another file"):
        read_look_profile(shared)


def test_a_profile_name_reads_as_text():
    assert read_look_profile(_xmp(_warm(), "Clean &amp; Rich"))[0] == "Clean & Rich"


def test_a_table_amount_is_the_strength_at_100_percent_under_its_own_key(looks):
    plain = looks()
    half = looks(name="Test Half", attrs='crs:RGBTableAmount="0.5"')
    assert half != plain and half.startswith(plain)
    look = rgb_table.load_look(half)
    np.testing.assert_array_equal(look_print_lut(half, 1.0).table, look_table_lut(look.table, look.space, 0.5)[0])


def test_a_malformed_table_amount_is_ignored(looks):
    assert looks(attrs='crs:RGBTableAmount="1.2.3"') == looks(name="Test Plain")


def test_a_key_that_is_not_a_look_id_names_no_file(looks, tmp_path):
    looks()
    outside = tmp_path / "outside.npz"
    np.savez(outside, x=np.zeros(3))
    key = LOOK_KEY_PREFIX + "../outside"
    assert rgb_table.load_look(key) is None and resolve_paper(key) is PAPER_PROFILES[DEFAULT_PROFILE_KEY]
    rgb_table.delete_look(key)
    assert outside.exists()


def test_a_damaged_look_file_is_skipped(looks):
    key = looks()
    folder = rgb_table.looks_dir()
    (folder / ("0" * 32 + ".npz")).write_bytes(b"not a zip")
    np.savez(folder / ("1" * 32 + ".npz"), x=np.zeros(3))
    rgb_table.list_looks.cache_clear()
    assert [k for k, _ in list_looks()] == [key]
    assert resolve_paper(LOOK_KEY_PREFIX + "1" * 32) is PAPER_PROFILES[DEFAULT_PROFILE_KEY]


def test_a_profile_without_an_amount_holds_full_strength(looks):
    key = looks(attrs='crs:SupportsAmount="False"')
    assert rgb_table.load_look(key).amounts == (1.0, 1.0)
    assert look_print_lut(key, 0.3) is look_print_lut(key, 1.0)


def _stream_xmp(payload: bytes, declared: int) -> str:
    table_id = hashlib.md5(payload).hexdigest().upper()
    encoded = b85_encode(struct.pack("<I", declared) + zlib.compress(payload))
    return f'<x crs:RGBTable="{table_id}" crs:Table_{table_id}="{encoded}"/>'


@pytest.mark.parametrize(
    "payload, declared",
    [
        (b"\1" * 8, 8),  # shorter than the header
        (struct.pack("<4I", 1, 1, 3, 2), 16),  # a header with no table
        (bytes(5_000_000), 100),  # a stream longer than its length
        (bytes(5_000_000), 5_000_000),  # longer than the largest table
    ],
)
def test_a_cut_or_oversized_table_is_refused_as_damaged(payload, declared):
    with pytest.raises(ValueError, match="damaged"):
        read_look_profile(_stream_xmp(payload, declared))


def test_an_imported_look_is_a_color_negative_print_stock(looks):
    key = looks(name="Test Warm")
    assert key.startswith(LOOK_KEY_PREFIX) and list_looks() == [(key, "Test Warm")]
    paper = resolve_paper(key)
    assert paper.look and paper.label == "Test Warm" and paper.medium == "look" and paper.print_color == DyeLut(key)
    assert key in [k for k, _ in profiles_for_mode(ProcessMode.C41)]
    assert key not in [k for k, _ in profiles_for_mode(ProcessMode.BW)]
    assert resolve_flat_matrix(paper) is None


def test_a_deleted_look_leaves_the_stocks_and_prints_on_digital(looks):
    key = looks()
    delete_look(key)
    assert list_looks() == [] and resolve_paper(key) is PAPER_PROFILES[DEFAULT_PROFILE_KEY]
    assert key not in dict(profiles_for_mode(ProcessMode.C41))


def test_a_removed_look_falls_back_to_digital(looks):
    key = looks()
    (rgb_table.looks_dir() / f"{key[len(LOOK_KEY_PREFIX) :]}.npz").unlink()
    rgb_table.load_look.cache_clear()
    assert resolve_paper(key) is PAPER_PROFILES[DEFAULT_PROFILE_KEY]


@pytest.mark.parametrize("space", [PROPHOTO_18, SRGB_EXTEND, (1, 3, 0), (4, 4, 0), (3, 0, 0)])
def test_an_identity_table_returns_its_input(space):
    rng = np.random.default_rng(1)
    rgb = rng.uniform(0.0, 1.0, (400, 3))
    if space[2] == 0:
        rgb = np.clip(rgb @ rgb_table._adobe_to(space[0]).T, 0, 1) @ np.linalg.inv(rgb_table._adobe_to(space[0])).T
    for amount in (0.0, 1.0, 2.0):
        np.testing.assert_allclose(apply_rgb_table(rgb, _identity(), space, amount), rgb, atol=2e-5)


def test_the_amount_blends_in_the_table_encoding():
    """A table that halves each encoded value: Amount a gives (1 - a/2) times the encoded input."""
    table = np.ascontiguousarray(_identity() * 0.5)
    gray = np.linspace(0.05, 1.0, 12)
    rgb = np.stack([gray] * 3, axis=1)
    for amount in (0.5, 1.0, 1.5):
        want = rgb_table._srgb_decode(rgb_table._srgb_encode(gray) * (1 - amount / 2))
        np.testing.assert_allclose(apply_rgb_table(rgb, table, (0, 1, 0), amount)[:, 1], want, atol=1e-5)


def test_an_extend_table_carries_a_color_past_its_edge():
    out_of_gamut = np.array([[0.0, 0.6, 0.0]])  # Adobe RGB green, outside sRGB
    clipped = apply_rgb_table(out_of_gamut, _identity(), (0, 1, 0), 1.0)
    extended = apply_rgb_table(out_of_gamut, _identity(), SRGB_EXTEND, 1.0)
    np.testing.assert_allclose(extended, out_of_gamut, atol=1e-5)
    assert np.abs(clipped - out_of_gamut).max() > 0.05


def test_zero_strength_is_digital(looks):
    key = looks()
    look = rgb_table.load_look(key)
    e = np.random.default_rng(0).uniform(0.0, float(EXPOSURE_CONSTANTS["d_max"]), (500, 3))
    y_b = 10.0 ** -float(EXPOSURE_CONSTANTS["d_max"])
    digital = np.clip((10.0**-e - y_b) / (1 - y_b), 0, 1) * (1 - y_b) + y_b
    np.testing.assert_allclose(look_on_digital(e, look.table, look.space, 0.0), digital, atol=2e-6)


def test_strength_clamps_to_the_profile_amount_range_and_keeps_the_key(looks):
    key = looks(amounts=(0.0, 1.5))
    full, over = look_print_lut(key, 1.5), look_print_lut(key, 2.0)
    assert full is over and full.key == key
    assert look_print_lut(key, 1.0) is look_print_lut(key, 1.0)
    assert np.abs(look_print_lut(key, 1.0).table - look_print_lut(key, 0.0).table).max() > 0.01


def test_the_lut_reproduces_the_table_on_digitals_print(looks):
    key = looks()
    look, lut = rgb_table.load_look(key), look_print_lut(key, 1.0)
    e = np.random.default_rng(2).uniform(0.05, 2.0, (300, 3))
    looked = np.array([tetra_lookup(lut.table, lut.e_max, *p) for p in e])
    direct = look_on_digital(e, look.table, look.space, 1.0)
    de = np.sqrt(((_lab(looked) - _lab(direct)) ** 2).sum(-1))
    assert de.mean() < 0.3 and np.percentile(de, 95) < 1.0


def test_exposure_paper_carries_the_strength(looks):
    key = looks()
    exp = replace(WorkspaceConfig().exposure, paper_profile=key, stock_strength=0.5)
    paper = exposure_paper(exp, ProcessMode.C41)
    assert paper.print_color == DyeLut(key, strength=0.5)
    assert resolve_print_color(paper)[1] is look_print_lut(key, 0.5)
    assert exposure_paper(replace(exp, paper_profile="pakon"), ProcessMode.C41) is PAPER_PROFILES["pakon"]
    assert exposure_paper(exp, ProcessMode.E6) is PAPER_PROFILES[DEFAULT_PROFILE_KEY]


def _lab(v: np.ndarray) -> np.ndarray:
    return rgb_to_lab_working(np.clip(v, 0, 1).astype(np.float32).reshape(-1, 1, 3)).reshape(-1, 3)


def _negative(h: int = 48, w: int = 48) -> np.ndarray:
    rng = np.random.default_rng(4)
    yy, xx = np.mgrid[0:h, 0:w] / max(h, w)
    base = np.stack([0.75 - 0.5 * xx, 0.55 - 0.35 * yy, 0.40 - 0.25 * xx * yy], -1)
    patches = rng.uniform(0.6, 1.4, (6, 6, 3)).repeat(-(-h // 6), 0).repeat(-(-w // 6), 1)[:h, :w]
    return np.ascontiguousarray(np.clip(base * patches, 0.02, 0.95).astype(np.float32))


def _render(settings: WorkspaceConfig, img: np.ndarray, gpu: bool) -> np.ndarray:
    from negpy.services.rendering.image_processor import ImageProcessor

    result, _ = ImageProcessor().run_pipeline(
        img, settings, "look-src", render_size_ref=float(max(img.shape[:2])), prefer_gpu=gpu, readback_metrics=False
    )
    arr = np.asarray(result.readback() if hasattr(result, "readback") else result)[:, :, :3]
    return arr.astype(np.float64)


def _settings(paper: str, strength: float = 1.0) -> WorkspaceConfig:
    s = WorkspaceConfig()
    lab = replace(s.lab, sharpen=0.0, chroma_denoise=0.0)
    return replace(s, lab=lab, exposure=replace(s.exposure, paper_profile=paper, stock_strength=strength))


@pytest.mark.parametrize("strength", [1.0, 0.5])
def test_a_look_stock_render_is_the_table_on_the_digital_render(looks, strength):
    key = looks()
    img = _negative()
    gamma = 563 / 256
    digital = _render(_settings(DEFAULT_PROFILE_KEY), img, gpu=False) ** gamma
    looked = _render(_settings(key, strength), img, gpu=False) ** gamma
    look = rgb_table.load_look(key)
    want = apply_rgb_table(digital.reshape(-1, 3), look.table, look.space, strength)
    de = np.sqrt(((_lab(looked.reshape(-1, 3)) - _lab(want)) ** 2).sum(-1))
    assert de.mean() < 1.0, de.mean()
    assert np.sqrt(((_lab(looked.reshape(-1, 3)) - _lab(digital.reshape(-1, 3))) ** 2).sum(-1)).mean() > 1.0


def test_the_stock_rail_shows_only_on_a_look_stock(qapp, looks):
    from negpy.desktop.session import AppState
    from negpy.desktop.view.sidebar.paper import PaperSidebar

    key = looks()
    controller = MagicMock()
    controller.state = AppState()
    sidebar = PaperSidebar(controller)
    sidebar.sync_ui()
    assert not sidebar.stock_rail.isVisibleTo(sidebar) and sidebar.look_import_btn.isVisibleTo(sidebar)
    assert not sidebar.look_delete_btn.isVisibleTo(sidebar)
    assert sidebar.paper_combo.findData(key) >= 0
    cfg = controller.state.config
    controller.state.config = replace(cfg, exposure=replace(cfg.exposure, paper_profile=key, stock_strength=1.25))
    sidebar.sync_ui()
    assert sidebar.stock_rail.isVisibleTo(sidebar) and sidebar.look_delete_btn.isVisibleTo(sidebar)
    assert sidebar.stock_strength_slider.value() == pytest.approx(125.0)
    sidebar.stock_strength_slider.valueCommitted.emit(80.0)
    assert controller.apply_config.call_args.args[0].exposure.stock_strength == pytest.approx(0.8)
    assert controller.apply_config.call_args.kwargs == {"persist": True, "readback_metrics": True}


@unittest.skipUnless(GPUDevice.get().is_available, "GPU not available")
class TestLookStockGpuParity(unittest.TestCase):
    @pytest.fixture(autouse=True)
    def _looks(self, looks):
        self.add = looks

    def test_cpu_gpu_match_on_look_strengths(self):
        img = _negative(64, 64)
        key = self.add()
        extend = self.add(values=_warm(), name="Test Extend", space=SRGB_EXTEND)
        for k, strength in ((key, 1.0), (key, 1.75), (extend, 0.6)):
            settings = _settings(k, strength)
            cpu, gpu = _render(settings, img, gpu=False), _render(settings, img, gpu=True)
            self.assertLess(float(np.mean(np.abs(cpu - gpu))), 0.01, k)
            self.assertLess(float(np.max(np.abs(cpu - gpu))), 0.04, k)

    def test_one_gpu_engine_follows_a_strength_change(self):
        from negpy.services.rendering.image_processor import ImageProcessor

        img, key = _negative(64, 64), self.add()
        processor = ImageProcessor()
        for strength in (1.0, 0.3):
            result, _ = processor.run_pipeline(
                img, _settings(key, strength), "look-src", render_size_ref=64.0, prefer_gpu=True, readback_metrics=False
            )
        gpu = np.asarray(result.readback() if hasattr(result, "readback") else result)[:, :, :3].astype(np.float64)
        cpu = _render(_settings(key, 0.3), img, gpu=False)
        self.assertLess(float(np.mean(np.abs(cpu - gpu))), 0.01)


def test_the_strength_slider_ends_at_the_profile_amount_range(qapp, looks):
    from negpy.desktop.session import AppState
    from negpy.desktop.view.sidebar.paper import PaperSidebar

    key = looks(amounts=(0.0, 1.0))
    controller = MagicMock()
    controller.state = AppState()
    controller.state.config = replace(controller.state.config, exposure=replace(controller.state.config.exposure, paper_profile=key))
    sidebar = PaperSidebar(controller)
    sidebar.sync_ui()
    assert sidebar.stock_strength_slider.spin.maximum() == pytest.approx(100.0)


def test_delete_look_profile_asks_then_prints_the_frame_on_digital(qapp, looks, monkeypatch):
    from negpy.desktop.session import AppState
    from negpy.desktop.view.sidebar import paper as paper_sidebar

    key = looks()
    controller = MagicMock()
    controller.state = AppState()
    controller.state.config = replace(controller.state.config, exposure=replace(controller.state.config.exposure, paper_profile=key))
    sidebar = paper_sidebar.PaperSidebar(controller)
    sidebar.sync_ui()
    monkeypatch.setattr(paper_sidebar, "confirm_delete_named", lambda *a, **k: False)
    sidebar.look_delete_btn.click()
    assert list_looks() != [] and not controller.apply_config.called
    monkeypatch.setattr(paper_sidebar, "confirm_delete_named", lambda *a, **k: True)
    sidebar.look_delete_btn.click()
    assert list_looks() == [] and sidebar.paper_combo.findData(key) < 0
    assert controller.apply_config.call_args.args[0].exposure.paper_profile == DEFAULT_PROFILE_KEY


def test_a_profile_without_an_amount_shows_no_strength_slider(qapp, looks):
    from negpy.desktop.session import AppState
    from negpy.desktop.view.sidebar.paper import PaperSidebar

    key = looks(attrs='crs:SupportsAmount="False"')
    controller = MagicMock()
    controller.state = AppState()
    controller.state.config = replace(controller.state.config, exposure=replace(controller.state.config.exposure, paper_profile=key))
    sidebar = PaperSidebar(controller)
    sidebar.sync_ui()
    assert not sidebar.stock_rail.isVisibleTo(sidebar) and sidebar.look_delete_btn.isVisibleTo(sidebar)


def test_another_stock_starts_at_full_strength_and_the_same_one_keeps_it(qapp, looks):
    from negpy.desktop.session import AppState
    from negpy.desktop.view.sidebar.paper import PaperSidebar

    first, second = looks(name="First"), looks(_warm() * 0.9, name="Second")
    controller = MagicMock()
    controller.state = AppState()
    exposure = replace(controller.state.config.exposure, paper_profile=first, stock_strength=0.4)
    controller.state.config = replace(controller.state.config, exposure=exposure)
    sidebar = PaperSidebar(controller)
    sidebar.sync_ui()
    sidebar._on_paper_changed(0)
    assert not controller.apply_config.called
    sidebar.paper_combo.setCurrentIndex(sidebar.paper_combo.findData(second))
    picked = controller.apply_config.call_args.args[0].exposure
    assert (picked.paper_profile, picked.stock_strength) == (second, 1.0)
