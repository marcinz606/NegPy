import os
import tempfile

import numpy as np
import tifffile

from negpy.domain.interfaces import PipelineContext
from negpy.domain.models import WorkspaceConfig
from negpy.features.geometry.models import GeometryConfig
from negpy.features.geometry.processor import GeometryProcessor
from negpy.features.retouch.logic import apply_dust_removal, apply_ir_dust_removal
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


def test_apply_ir_dust_removal_heals_defect():
    img = np.full((80, 80, 3), 0.5, dtype=np.float32)
    img[40, 40] = 0.95
    ir = np.full((80, 80), 0.9, dtype=np.float32)
    ir[39:42, 39:42] = 0.05

    out, mask = apply_ir_dust_removal(img.copy(), ir, threshold=0.5, inpaint_radius=3, scale_factor=1.0)
    assert mask.shape == (80, 80)
    assert mask[40, 40] == 255
    # Defect pixel pulled toward 0.5 background.
    assert out[40, 40, 0] < 0.9


def test_apply_ir_dust_removal_no_defect_returns_untouched():
    img = np.full((40, 40, 3), 0.5, dtype=np.float32)
    ir = np.full((40, 40), 0.9, dtype=np.float32)
    out, mask = apply_ir_dust_removal(img.copy(), ir, threshold=0.5, inpaint_radius=3, scale_factor=1.0)
    assert not np.any(mask)
    np.testing.assert_array_equal(out, img)


def test_apply_dust_removal_with_ir_enabled_heals_defect():
    img = np.full((50, 50, 3), 0.5, dtype=np.float32)
    img[25, 25] = 0.95
    ir = np.full((50, 50), 0.9, dtype=np.float32)
    ir[25, 25] = 0.1
    out = apply_dust_removal(
        img.copy(),
        dust_remove=False,
        dust_threshold=0.5,
        dust_size=2,
        manual_spots=[],
        scale_factor=1.0,
        ir_buffer=ir,
        ir_dust_remove=True,
        ir_threshold=0.5,
        ir_inpaint_radius=3,
    )
    assert out[25, 25, 0] < 0.9


def test_apply_dust_removal_noop_when_all_disabled():
    img = np.full((40, 40, 3), 0.5, dtype=np.float32)
    out = apply_dust_removal(
        img,
        dust_remove=False,
        dust_threshold=0.5,
        dust_size=2,
        manual_spots=[],
        scale_factor=1.0,
        ir_buffer=None,
        ir_dust_remove=False,
    )
    np.testing.assert_array_equal(out, img)


def test_geometry_processor_transforms_ir_alongside_rgb():
    img = np.zeros((20, 30, 3), dtype=np.float32)
    img[5, 10] = 1.0
    ir = np.zeros((20, 30), dtype=np.float32)
    ir[5, 10] = 1.0
    ctx = PipelineContext(original_size=(20, 30), scale_factor=1.0, ir_buffer=ir)
    geom = GeometryConfig(rotation=1, flip_horizontal=True)
    out = GeometryProcessor(geom).process(img, ctx)
    ir_out = ctx.metrics["ir_post_geometry"]
    assert ir_out is not None
    assert ir_out.shape == out.shape[:2]
    # Hot pixel must end up where the RGB hot pixel ended up.
    rgb_hot = np.unravel_index(np.argmax(out[..., 0]), out.shape[:2])
    ir_hot = np.unravel_index(np.argmax(ir_out), ir_out.shape)
    assert rgb_hot == ir_hot


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
