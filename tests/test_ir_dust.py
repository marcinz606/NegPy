import os
import tempfile

import numpy as np
import tifffile

from negpy.domain.models import WorkspaceConfig
from negpy.features.retouch.logic import (
    apply_manual_heals,
    build_heal_regions,
    detect_ir_regions,
    ir_bake_token,
    ir_detect_cutoff,
    ir_ratio_and_gain,
    normalize_ir,
)
from negpy.features.retouch.models import RetouchConfig
from negpy.infrastructure.loaders.factory import LoaderFactory


def test_retouch_config_defaults_include_ir_fields():
    cfg = RetouchConfig()
    assert cfg.ir_dust_remove is False
    assert 0.05 < cfg.ir_threshold < 0.95
    assert cfg.ir_inpaint_radius >= 1


def test_workspace_config_backcompat_for_ir_fields():
    """Old config dicts without IR fields must deserialize with sane defaults."""
    cfg = WorkspaceConfig.from_flat_dict({})
    assert cfg.retouch.ir_dust_remove is False


def test_workspace_config_roundtrip_ir_fields():
    cfg = WorkspaceConfig(
        retouch=RetouchConfig(ir_dust_remove=True, ir_threshold=0.4, ir_inpaint_radius=5),
    )
    flat = cfg.to_dict()
    assert flat["ir_dust_remove"] is True
    assert flat["ir_threshold"] == 0.4

    restored = WorkspaceConfig.from_flat_dict(flat)
    assert restored.retouch.ir_dust_remove is True
    assert restored.retouch.ir_threshold == 0.4


def test_detect_ir_regions_heals_defect_end_to_end():
    """IR speck → synthesized ungated stroke → membrane clone removes it."""
    h, w = 80, 80
    rng = np.random.default_rng(17)
    img = (np.full((h, w, 3), 0.5) + rng.normal(0, 0.01, (h, w, 3))).astype(np.float32)
    img[39:42, 39:42] = 0.95
    ir = np.full((h, w), 0.9, dtype=np.float32)
    ir[39:42, 39:42] = 0.05

    strokes, _ = detect_ir_regions(normalize_ir(ir), 0.5, pad_px=3.0)
    assert len(strokes) == 1
    assert strokes[0][4] == 0.0  # IR regions are ungated

    regions = build_heal_regions(strokes, [], (h, w), 0, 0.0, False, False, 0.0, (w, h))
    out = apply_manual_heals(img, *regions)
    assert out[40, 40, 0] < 0.7


def test_detect_ir_regions_no_defect_is_empty():
    ir = np.full((40, 40), 0.9, dtype=np.float32)
    strokes, hair = detect_ir_regions(normalize_ir(ir), 0.5)
    assert strokes == [] and hair is None


def test_ir_detect_cutoff_mapping_and_direction():
    """The slider→ratio-cutoff map: lower slider catches more (higher cutoff) in
    both modes; the attenuation band sits lower than detection-only."""
    assert ir_detect_cutoff(0.1, True) > ir_detect_cutoff(0.9, True)
    assert ir_detect_cutoff(0.1, False) > ir_detect_cutoff(0.9, False)
    assert ir_detect_cutoff(0.35, True) < ir_detect_cutoff(0.35, False)
    assert abs(ir_detect_cutoff(0.35, True) - 0.71) < 1e-6


def test_normalize_ir_flat_plane_is_unity():
    """Clean film → ratio ~1.0 everywhere; a dust dip on a mild illumination
    gradient is still detected at the default cutoff (raw-IR thresholding missed
    dips that sat above the global cutoff)."""
    ir = np.full((120, 120), 0.8, dtype=np.float32)
    assert abs(float(normalize_ir(ir).mean()) - 1.0) < 0.01

    grad = np.linspace(0.7, 0.85, 120, dtype=np.float32)[:, None].repeat(120, axis=1)
    grad[60:63, 60:63] = grad[60:63, 60:63] * 0.4  # dust dip on the gradient
    strokes, _ = detect_ir_regions(normalize_ir(grad), ir_detect_cutoff(0.35, True))
    assert len(strokes) == 1


def test_detect_ir_regions_coverage_abort():
    """A cutoff that marks the whole frame returns nothing (never smears the preview)."""
    ratio = np.full((80, 80), 0.5, dtype=np.float32)  # 100% below any sane cutoff
    assert detect_ir_regions(ratio, 0.8)[0] == []


def test_tiff_loader_reads_ir_from_extrasamples():
    h, w = 16, 24
    rgb = np.full((h, w, 3), 30000, dtype=np.uint16)
    ir = np.full((h, w), 50000, dtype=np.uint16)
    rgba_with_ir = np.dstack([rgb, ir])
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "scan.tif")
        tifffile.imwrite(path, rgba_with_ir, photometric="rgb", extrasamples=("unspecified",))
        ctx_mgr, metadata = LoaderFactory().get_loader(path)
        with ctx_mgr:
            pass
        assert metadata["ir"] is not None
        assert metadata["ir"].shape == (h, w)
        assert metadata["ir"].dtype == np.float32
        assert abs(float(metadata["ir"].mean()) - (50000.0 / 65535.0)) < 1e-3


def test_tiff_loader_sidecar_ir_file():
    h, w = 12, 18
    rgb = np.full((h, w, 3), 20000, dtype=np.uint16)
    ir = np.full((h, w), 60000, dtype=np.uint16)
    with tempfile.TemporaryDirectory() as td:
        rgb_path = os.path.join(td, "scan.tif")
        ir_path = os.path.join(td, "scan_IR.tif")
        tifffile.imwrite(rgb_path, rgb, photometric="rgb")
        tifffile.imwrite(ir_path, ir, photometric="minisblack")
        ctx_mgr, metadata = LoaderFactory().get_loader(rgb_path)
        with ctx_mgr:
            pass
        assert metadata["ir"] is not None
        assert metadata["ir"].shape == (h, w)
        assert abs(float(metadata["ir"].mean()) - (60000.0 / 65535.0)) < 1e-3


def test_tiff_loader_no_ir_when_rgb_only():
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "rgb_only.tif")
        tifffile.imwrite(path, np.full((10, 10, 3), 30000, dtype=np.uint16), photometric="rgb")
        _, metadata = LoaderFactory().get_loader(path)
        assert metadata["ir"] is None


def test_rawpy_loader_dng_thumbnail_subifd_ir():
    """VueScan/Adobe-style DNG: thumbnail IFD0 + SubIFD carrying the 4-sample LinearRaw
    RGB+IR data (img02.dng's structure) — not NegPy's own single-IFD DNG output."""
    h, w = 8, 10
    thumb = np.zeros((4, 5, 3), dtype=np.uint8)
    full = np.random.randint(0, 65535, (h, w, 4)).astype(np.uint16)
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "scan.dng")
        with tifffile.TiffWriter(path) as tw:
            tw.write(thumb, photometric="rgb", subfiletype=1, subifds=1)
            tw.write(full, photometric=34892, subfiletype=0, planarconfig="CONTIG")
        ctx_mgr, metadata = LoaderFactory().get_loader(path)
        with ctx_mgr:
            pass
        assert metadata["ir"] is not None
        assert metadata["ir"].shape == (h, w)


def test_tiff_loader_silverfast_multipage_ir():
    """SilverFast iSRD: IR stored as page 2 with NewSubfileType=4 (transparency mask)."""
    h, w = 16, 24
    rgb = np.full((h, w, 3), 30000, dtype=np.uint16)
    ir = np.full((h, w), 50000, dtype=np.uint16)
    thumb = np.full((4, 6, 3), 30000, dtype=np.uint16)
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "silverfast.tif")
        with tifffile.TiffWriter(path) as tw:
            tw.write(rgb, photometric="rgb", subfiletype=0)
            tw.write(thumb, photometric="rgb", subfiletype=1)
            tw.write(ir, photometric="minisblack", subfiletype=0)
        ctx_mgr, metadata = LoaderFactory().get_loader(path)
        with ctx_mgr:
            pass
        assert metadata["ir"] is not None
        assert metadata["ir"].shape == (h, w)
        assert metadata["ir"].dtype == np.float32
        assert abs(float(metadata["ir"].mean()) - (50000.0 / 65535.0)) < 1e-3


def test_ir_dust_remove_field_invalidates_retouch_hash():
    from negpy.kernel.caching.logic import calculate_config_hash

    a = RetouchConfig(ir_dust_remove=False)
    b = RetouchConfig(ir_dust_remove=True)
    assert calculate_config_hash(a) != calculate_config_hash(b)


def test_ir_attenuation_field_invalidates_retouch_hash():
    from negpy.kernel.caching.logic import calculate_config_hash

    assert calculate_config_hash(RetouchConfig(ir_attenuation=True)) != calculate_config_hash(RetouchConfig(ir_attenuation=False))


def test_ir_bake_token_active_and_empty():
    on = RetouchConfig(ir_dust_remove=True, ir_attenuation=True)
    assert ir_bake_token(on, has_ir=True) == "|irdiv1"
    assert ir_bake_token(on, has_ir=False) == ""  # no IR plane → nothing to bake
    assert ir_bake_token(RetouchConfig(ir_dust_remove=True, ir_attenuation=False), True) == ""
    assert ir_bake_token(RetouchConfig(ir_dust_remove=False, ir_attenuation=True), True) == ""


def test_ir_ratio_and_gain_properties():
    """Gain never darkens a clean pixel, clamps at 2.0, and is identity on clean
    film (ratio≈1); γ stays inside the clamp."""
    h = w = 200
    ir = np.full((h, w), 0.9, dtype=np.float32)
    ir[95:105, 95:105] = 0.2  # opaque-ish core
    ir[60:64, 60:64] = 0.78 * 0.9  # semi-transparent speck (ratio ≈ 0.78)
    img = np.full((h, w, 3), 0.5, dtype=np.float32)
    img[95:105, 95:105] = 0.15
    img[60:64, 60:64] = 0.42

    ratio, gain, degenerate, gammas = ir_ratio_and_gain(ir, img)
    assert not degenerate
    assert gain.shape == (h, w, 3)
    assert gain.min() >= 1.0 - 1e-4
    assert gain.max() <= 2.0 + 1e-4
    clean = ratio > 0.99
    assert clean.any()
    assert abs(float(gain[np.broadcast_to(clean[..., None], gain.shape)].reshape(-1, 3).mean()) - 1.0) < 1e-3
    assert all(1.0 <= g <= 2.2 for g in gammas)


def test_ir_ratio_and_gain_degenerate_on_image_content():
    """An IR plane carrying image content (a broad gradient, like B&W silver) is
    flagged degenerate so the caller skips both the bake and IR strokes."""
    grad = np.linspace(0.2, 0.9, 200, dtype=np.float32)[None, :].repeat(200, axis=0)
    img = np.stack([grad] * 3, axis=-1)
    _, _, degenerate, _ = ir_ratio_and_gain(grad.copy(), img)
    assert degenerate
