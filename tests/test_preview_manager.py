"""
Contract tests for PreviewManager.load_linear_preview.

These guard decode parameters and output shape/dtype so preview-speed work
(half_size, demosaic mode, cache keys) does not regress silently.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import rawpy

from negpy.infrastructure.loaders.tiff_loader import NonStandardFileWrapper
from negpy.services.rendering.preview_manager import PreviewManager


def test_load_linear_preview_nonstandard_wrapper_round_trip() -> None:
    """Uses NonStandardFileWrapper (no rawpy) to exercise float/resize path — no half_size fast path."""
    data = np.full((120, 160, 3), 0.25, dtype=np.float32)
    ctx = NonStandardFileWrapper(data)
    meta = {"color_space": "Adobe RGB"}

    with patch("negpy.services.rendering.preview_manager.loader_factory") as lf:
        lf.get_loader.return_value = (ctx, meta)
        buf, dims, out_meta = PreviewManager().load_linear_preview("/fake/path.tif")

    assert out_meta == meta
    assert dims == (120, 160)
    assert buf.shape == (120, 160, 3)
    assert buf.dtype == np.float32
    assert np.allclose(buf, 0.25, atol=1e-5)


def test_load_linear_preview_downscales_to_preview_render_size() -> None:
    """When long edge exceeds preview_render_size, output is scaled down."""
    h, w = 800, 600
    data = np.full((h, w, 3), 0.5, dtype=np.float32)
    ctx = NonStandardFileWrapper(data)
    fake_cfg = SimpleNamespace(
        preview_render_size=400,
        canvas_zoom_min=0.25,
        canvas_zoom_max=8.0,
        preview_cache_max_entries=8,
        preview_cache_max_bytes=10**9,
    )

    with (
        patch("negpy.services.rendering.preview_manager.loader_factory") as lf,
        patch("negpy.services.rendering.preview_manager.APP_CONFIG", fake_cfg),
    ):
        lf.get_loader.return_value = (ctx, {"color_space": "Adobe RGB"})
        buf, dims, _ = PreviewManager().load_linear_preview("/fake/path.tif")

    assert dims == (800, 600)
    assert max(buf.shape[0], buf.shape[1]) == 400
    assert buf.dtype == np.float32


def test_load_linear_preview_fast_path_line_and_half() -> None:
    """Default (non-HQ) raw: LINEAR + half_size."""
    rgb_u16 = np.zeros((32, 24, 3), dtype=np.uint16)
    rgb_u16[..., 0] = 1000

    raw = MagicMock()
    raw.raw_type = rawpy.RawType.Flat
    raw.raw_pattern = np.zeros((2, 2), dtype=np.uint8)
    raw.sizes = SimpleNamespace(raw_height=32, raw_width=24, iheight=32, iwidth=24)
    raw.postprocess = MagicMock(return_value=rgb_u16)

    class _Ctx:
        def __enter__(self) -> MagicMock:
            return raw

        def __exit__(self, *args: object) -> None:
            return None

    with patch("negpy.services.rendering.preview_manager.loader_factory") as lf:
        lf.get_loader.return_value = (_Ctx(), {"color_space": "Adobe RGB"})
        buf, dims, _ = PreviewManager().load_linear_preview(
            "/fake/path.dng",
            color_space="Adobe RGB",
            use_camera_wb=False,
        )

    raw.postprocess.assert_called_once()
    _, kwargs = raw.postprocess.call_args
    assert kwargs["gamma"] == (1, 1)
    assert kwargs["no_auto_bright"] is True
    assert kwargs["use_camera_wb"] is False
    assert kwargs["user_wb"] == [1, 1, 1, 1]
    assert kwargs["output_bps"] == 16
    assert kwargs["demosaic_algorithm"] == rawpy.DemosaicAlgorithm.LINEAR
    assert kwargs.get("half_size") is True
    assert kwargs["user_flip"] == 0

    assert dims == (32, 24)
    assert buf.shape == (32, 24, 3)
    assert buf.dtype == np.float32


def test_load_linear_preview_hq_uses_best_demosaic_no_half() -> None:
    """full_resolution: AHD (Bayer) and no half_size."""
    rgb_u16 = np.zeros((64, 48, 3), dtype=np.uint16)
    raw = MagicMock()
    raw.raw_type = rawpy.RawType.Flat
    raw.raw_pattern = np.zeros((2, 2), dtype=np.uint8)
    raw.sizes = SimpleNamespace(raw_height=64, raw_width=48, iheight=64, iwidth=48)
    raw.postprocess = MagicMock(return_value=rgb_u16)

    class _Ctx:
        def __enter__(self) -> MagicMock:
            return raw

        def __exit__(self, *args: object) -> None:
            return None

    with patch("negpy.services.rendering.preview_manager.loader_factory") as lf:
        lf.get_loader.return_value = (_Ctx(), {"color_space": "Adobe RGB"})
        PreviewManager().load_linear_preview("/fake/path.dng", color_space="Adobe RGB", use_camera_wb=False, full_resolution=True)

    _, kwargs = raw.postprocess.call_args
    assert kwargs["demosaic_algorithm"] == rawpy.DemosaicAlgorithm.AHD
    assert kwargs.get("half_size") is not True


@pytest.mark.parametrize("cfa_block", [2, 6])
def test_load_linear_preview_hq_demosaic_xtrans_vs_bayer(cfa_block: int) -> None:
    """HQ: X-Trans (6) uses DHT; Bayer (2) uses AHD."""
    rgb_u16 = np.ones((32, 32, 3), dtype=np.uint16) * 128

    raw = MagicMock()
    raw.raw_type = rawpy.RawType.Flat
    raw.raw_pattern = np.zeros((cfa_block, cfa_block), dtype=np.uint8)
    raw.sizes = SimpleNamespace(raw_height=32, raw_width=32, iheight=32, iwidth=32)
    raw.postprocess = MagicMock(return_value=rgb_u16)

    class _Ctx:
        def __enter__(self) -> MagicMock:
            return raw

        def __exit__(self, *args: object) -> None:
            return None

    with patch("negpy.services.rendering.preview_manager.loader_factory") as lf:
        lf.get_loader.return_value = (_Ctx(), {"color_space": "Adobe RGB"})
        PreviewManager().load_linear_preview("/fake/path.dng", full_resolution=True)

    _, kwargs = raw.postprocess.call_args
    expected = rawpy.DemosaicAlgorithm.DHT if cfa_block == 6 else rawpy.DemosaicAlgorithm.AHD
    assert kwargs["demosaic_algorithm"] == expected


@pytest.mark.parametrize(
    "cfa_block,use_camera_wb,half_expected",
    [
        (6, False, False),  # X-Trans + linear: half_size aliases the 6x6 CFA → skip it
        (6, True, True),  # X-Trans + camera WB: tolerated, keep the fast path
        (2, False, True),  # Bayer: 2x2 averages cleanly → keep half_size
    ],
)
def test_load_linear_preview_fast_half_size_gated_on_xtrans_linear(cfa_block: int, use_camera_wb: bool, half_expected: bool) -> None:
    """Fast preview always uses LINEAR; half_size is dropped only for linear X-Trans decodes."""
    rgb_u16 = np.ones((32, 32, 3), dtype=np.uint16) * 128

    raw = MagicMock()
    raw.raw_type = rawpy.RawType.Flat
    raw.raw_pattern = np.zeros((cfa_block, cfa_block), dtype=np.uint8)
    raw.sizes = SimpleNamespace(raw_height=32, raw_width=32, iheight=32, iwidth=32)
    raw.postprocess = MagicMock(return_value=rgb_u16)

    class _Ctx:
        def __enter__(self) -> MagicMock:
            return raw

        def __exit__(self, *args: object) -> None:
            return None

    with patch("negpy.services.rendering.preview_manager.loader_factory") as lf:
        lf.get_loader.return_value = (_Ctx(), {"color_space": "Adobe RGB"})
        PreviewManager().load_linear_preview("/fake/path.dng", color_space="Adobe RGB", use_camera_wb=use_camera_wb, full_resolution=False)

    _, kwargs = raw.postprocess.call_args
    assert kwargs["demosaic_algorithm"] == rawpy.DemosaicAlgorithm.LINEAR
    assert (kwargs.get("half_size") is True) == half_expected


def test_load_linear_preview_decodes_in_raw_colorspace() -> None:
    """Preview must decode in rawpy ColorSpace.raw — the pipeline assumes raw-space linear input.

    Guards against decoding into a display space (e.g. Adobe), which silently shifts
    colour and breaks normalization / process-mode detection.
    """
    rgb_u16 = np.zeros((32, 24, 3), dtype=np.uint16)
    raw = MagicMock()
    raw.raw_type = rawpy.RawType.Flat
    raw.raw_pattern = np.zeros((2, 2), dtype=np.uint8)
    raw.sizes = SimpleNamespace(raw_height=32, raw_width=24, iheight=32, iwidth=24)
    raw.postprocess = MagicMock(return_value=rgb_u16)

    class _Ctx:
        def __enter__(self) -> MagicMock:
            return raw

        def __exit__(self, *args: object) -> None:
            return None

    with patch("negpy.services.rendering.preview_manager.loader_factory") as lf:
        lf.get_loader.return_value = (_Ctx(), {"color_space": "Adobe RGB"})
        PreviewManager().load_linear_preview("/fake/path.dng", color_space="Adobe RGB", use_camera_wb=False)

    _, kwargs = raw.postprocess.call_args
    assert kwargs["output_color"] == rawpy.ColorSpace.raw


def test_load_linear_preview_bakes_exif_orientation() -> None:
    """Orientation from metadata is baked into pixels (postprocess runs user_flip=0)."""
    data = np.zeros((120, 160, 3), dtype=np.float32)
    data[0, 0, :] = 1.0  # top-left marker
    ctx = NonStandardFileWrapper(data)

    with patch("negpy.services.rendering.preview_manager.loader_factory") as lf:
        # orientation 3 == rot180: top-left marker must land bottom-right.
        lf.get_loader.return_value = (ctx, {"color_space": "Adobe RGB", "orientation": 3})
        buf, dims, _ = PreviewManager().load_linear_preview("/fake/path.tif")

    assert dims == (120, 160)  # 180° keeps dims
    assert buf.shape == (120, 160, 3)
    assert np.allclose(buf[119, 159, :], 1.0, atol=1e-4)
    assert np.allclose(buf[0, 0, :], 0.0, atol=1e-4)


def test_load_linear_preview_orientation_swaps_reported_dims() -> None:
    """90° orientation swaps both the buffer and the reported full-res dims."""
    data = np.full((120, 160, 3), 0.25, dtype=np.float32)
    ctx = NonStandardFileWrapper(data)

    with patch("negpy.services.rendering.preview_manager.loader_factory") as lf:
        lf.get_loader.return_value = (ctx, {"color_space": "Adobe RGB", "orientation": 6})
        buf, dims, _ = PreviewManager().load_linear_preview("/fake/path.tif")

    assert dims == (160, 120)
    assert buf.shape == (160, 120, 3)


def test_load_linear_preview_builds_ir_preview() -> None:
    """An IR channel in metadata is surfaced as ir_preview, oriented and sized to the preview."""
    data = np.full((120, 160, 3), 0.5, dtype=np.float32)
    ir = np.full((120, 160), 0.3, dtype=np.float32)
    ctx = NonStandardFileWrapper(data)

    with patch("negpy.services.rendering.preview_manager.loader_factory") as lf:
        lf.get_loader.return_value = (ctx, {"color_space": "Adobe RGB", "orientation": 1, "ir": ir})
        buf, _dims, out_meta = PreviewManager().load_linear_preview("/fake/path.tif")

    assert out_meta["ir_preview"] is not None
    assert out_meta["ir_preview"].shape == buf.shape[:2]
    assert out_meta["ir_preview"].dtype == np.float32
    assert np.allclose(out_meta["ir_preview"], 0.3, atol=1e-4)


def test_load_linear_preview_ir_preview_resized_with_downscale() -> None:
    """IR preview tracks the downscaled preview dims, not the full-res IR shape."""
    data = np.full((120, 160, 3), 0.5, dtype=np.float32)
    ir = np.full((120, 160), 0.3, dtype=np.float32)
    ctx = NonStandardFileWrapper(data)
    fake_cfg = SimpleNamespace(
        preview_render_size=80,
        canvas_zoom_min=0.25,
        canvas_zoom_max=8.0,
        preview_cache_max_entries=8,
        preview_cache_max_bytes=10**9,
    )

    with (
        patch("negpy.services.rendering.preview_manager.loader_factory") as lf,
        patch("negpy.services.rendering.preview_manager.APP_CONFIG", fake_cfg),
    ):
        lf.get_loader.return_value = (ctx, {"color_space": "Adobe RGB", "ir": ir})
        buf, _dims, out_meta = PreviewManager().load_linear_preview("/fake/path.tif")

    assert out_meta["ir_preview"].shape == buf.shape[:2]
    assert max(out_meta["ir_preview"].shape) == 80


def test_ir_preview_survives_warm_cache_hit() -> None:
    """A warm cache hit must still carry ir_preview — the cache stores/returns metadata."""
    data = np.full((120, 160, 3), 0.5, dtype=np.float32)
    ir = np.full((120, 160), 0.3, dtype=np.float32)
    ctx = NonStandardFileWrapper(data)
    pm = PreviewManager()

    with patch("negpy.services.rendering.preview_manager.loader_factory") as lf:
        lf.get_loader.return_value = (ctx, {"color_space": "Adobe RGB", "ir": ir})
        # Cold: populates the cache (file_hash + explicit color_space activate the cache path).
        pm.load_linear_preview("/fake/path.tif", color_space="Adobe RGB", file_hash="abc123")
        # Warm: must hit the cache without re-opening the loader.
        lf.get_loader.reset_mock()
        buf, _dims, out_meta = pm.load_linear_preview("/fake/path.tif", color_space="Adobe RGB", file_hash="abc123")

    lf.get_loader.assert_not_called()
    assert out_meta["ir_preview"] is not None
    assert out_meta["ir_preview"].shape == buf.shape[:2]
    assert np.allclose(out_meta["ir_preview"], 0.3, atol=1e-4)


def test_load_linear_preview_no_ir_preview_when_absent() -> None:
    """No IR channel -> ir_preview is None (not missing)."""
    data = np.full((64, 64, 3), 0.5, dtype=np.float32)
    ctx = NonStandardFileWrapper(data)

    with patch("negpy.services.rendering.preview_manager.loader_factory") as lf:
        lf.get_loader.return_value = (ctx, {"color_space": "Adobe RGB"})
        _buf, _dims, out_meta = PreviewManager().load_linear_preview("/fake/path.tif")

    assert out_meta["ir_preview"] is None


def test_output_dimensions_from_raw_sizes_not_postprocessed_shape() -> None:
    """When postprocessed array is half-res, dims use raw.sizes."""
    # Simulated half decode output 16x12 but full image is 32x24
    rgb_u16 = np.ones((16, 12, 3), dtype=np.uint16) * 1000
    raw = MagicMock()
    raw.raw_type = rawpy.RawType.Flat
    raw.raw_pattern = np.zeros((2, 2), dtype=np.uint8)
    raw.sizes = SimpleNamespace(raw_height=32, raw_width=24, iheight=32, iwidth=24)
    raw.postprocess = MagicMock(return_value=rgb_u16)

    class _Ctx:
        def __enter__(self) -> MagicMock:
            return raw

        def __exit__(self, *args: object) -> None:
            return None

    with patch("negpy.services.rendering.preview_manager.loader_factory") as lf:
        lf.get_loader.return_value = (_Ctx(), {"color_space": "Adobe RGB"})
        _buf, dims, _ = PreviewManager().load_linear_preview("/fake/path.dng")

    assert dims == (32, 24)
