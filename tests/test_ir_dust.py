import os
import tempfile

import numpy as np
import tifffile

from negpy.domain.models import WorkspaceConfig
from negpy.features.retouch.logic import apply_manual_heals, build_heal_regions, detect_ir_regions
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

    strokes = detect_ir_regions(ir, threshold=0.5, pad_px=3.0)
    assert len(strokes) == 1
    assert strokes[0][4] == 0.0  # IR regions are ungated

    regions = build_heal_regions(strokes, [], (h, w), 0, 0.0, False, False, 0.0, (w, h))
    out = apply_manual_heals(img, *regions)
    assert out[40, 40, 0] < 0.7


def test_detect_ir_regions_no_defect_is_empty():
    ir = np.full((40, 40), 0.9, dtype=np.float32)
    assert detect_ir_regions(ir, threshold=0.5) == []


def test_detect_ir_regions_threshold_inversion_convention():
    """Callers pass 1−ir_threshold (the old processor/gpu inversion): a UI
    threshold of t marks IR transmittance below 1−t as defective."""
    ir = np.full((60, 60), 0.9, dtype=np.float32)
    ir[30, 30] = 0.45
    assert detect_ir_regions(ir, threshold=1.0 - 0.6) == []  # ir < 0.4 misses the 0.45 dip
    assert len(detect_ir_regions(ir, threshold=1.0 - 0.5)) == 1  # ir < 0.5 catches it


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
