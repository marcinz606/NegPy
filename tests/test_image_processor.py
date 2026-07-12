import numpy as np
from negpy.services.rendering.image_processor import ImageProcessor
from negpy.domain.models import WorkspaceConfig


def test_image_service_buffer_to_pil_8bit() -> None:
    service = ImageProcessor()
    buffer = np.array([[[0.0, 0.5, 1.0]]], dtype=np.float32)
    settings = WorkspaceConfig()

    img = service.buffer_to_pil(buffer, settings, bit_depth=8)
    assert img.mode == "RGB"
    assert img.size == (1, 1)
    assert img.getpixel((0, 0)) == (0, 127, 255)


def test_image_service_buffer_to_pil_16bit_bw() -> None:
    service = ImageProcessor()
    buffer = np.array([[0.0, 1.0]], dtype=np.float32)  # Single channel (grayscale)
    settings = WorkspaceConfig.from_flat_dict({"process_mode": "B&W"})

    img = service.buffer_to_pil(buffer, settings, bit_depth=16)
    # PIL uses 'I;16' for 16-bit single channel
    assert img.mode == "I;16"
    assert img.getpixel((1, 0)) == 65535


def test_image_service_bw_conversion() -> None:
    service = ImageProcessor()
    # 3-channel input but B&W mode
    buffer = np.zeros((10, 10, 3), dtype=np.float32)
    settings = WorkspaceConfig.from_flat_dict({"process_mode": "B&W"})

    img = service.buffer_to_pil(buffer, settings, bit_depth=8)
    assert img.mode == "L"


def test_image_service_jit_conversions() -> None:
    from negpy.kernel.image.logic import uint16_to_float32, uint8_to_float32

    # Test uint16 to float32 JIT
    u16_arr = np.array([[[0, 32767, 65535]]], dtype=np.uint16)
    f32_res = uint16_to_float32(np.ascontiguousarray(u16_arr))
    assert f32_res.dtype == np.float32
    assert np.allclose(f32_res, [[[0.0, 32767 / 65535, 1.0]]])

    # Test uint8 to float32 JIT
    u8_arr = np.array([[[0, 127, 255]]], dtype=np.uint8)
    f32_res_u8 = uint8_to_float32(np.ascontiguousarray(u8_arr))
    assert f32_res_u8.dtype == np.float32
    assert np.allclose(f32_res_u8, [[[0.0, 127 / 255, 1.0]]])


def test_use_half_size_decode_rules(monkeypatch) -> None:
    import negpy.services.rendering.image_processor as ip

    class _Raw:
        pass

    monkeypatch.setattr(ip, "is_xtrans", lambda raw: False)
    assert ip._use_half_size_decode(_Raw(), linear_raw=False)
    assert ip._use_half_size_decode(_Raw(), linear_raw=True)

    # X-Trans + linear decode: half_size aliases the 6x6 CFA -> stay full-res.
    monkeypatch.setattr(ip, "is_xtrans", lambda raw: True)
    assert ip._use_half_size_decode(_Raw(), linear_raw=False)
    assert not ip._use_half_size_decode(_Raw(), linear_raw=True)

    monkeypatch.setattr(ip, "is_xtrans", lambda raw: False)
    wrapper = object.__new__(ip.NonStandardFileWrapper)
    assert not ip._use_half_size_decode(wrapper, linear_raw=False)


def _fake_decode_recorder(calls):
    def fake(file_path, linear_raw, fast=False):
        calls.append(fast)
        return np.zeros((4, 4, 3), dtype=np.uint16), {"orientation": 1, "color_space": "sRGB"}

    return fake


def test_load_source_f32_cache_key_separates_fast_decode(monkeypatch) -> None:
    service = ImageProcessor()
    calls: list = []
    monkeypatch.setattr(service, "_decode_sensor_rgb", _fake_decode_recorder(calls))
    cfg = WorkspaceConfig()

    service._load_source_f32("/nonexistent/a.raw", cfg, fast_decode=True)
    service._load_source_f32("/nonexistent/a.raw", cfg, fast_decode=True)
    assert calls == [True]  # second call is a cache hit

    # A full-res consumer (real export) must not reuse the half-size buffer.
    service._load_source_f32("/nonexistent/a.raw", cfg, fast_decode=False)
    assert calls == [True, False]


def test_load_source_f32_never_fast_decodes_rgbscan_triplets(monkeypatch, tmp_path) -> None:
    from dataclasses import replace

    from negpy.features.rgbscan.models import RgbScanConfig

    r, g, b = (tmp_path / n for n in ("r.raw", "g.raw", "b.raw"))
    for f in (r, g, b):
        f.write_bytes(b"x")

    service = ImageProcessor()
    calls: list = []
    monkeypatch.setattr(service, "_decode_sensor_rgb", _fake_decode_recorder(calls))
    cfg = replace(
        WorkspaceConfig(),
        rgbscan=RgbScanConfig(enabled=True, green_path=str(g), blue_path=str(b), align=False),
    )

    service._load_source_f32(str(r), cfg, fast_decode=True)
    assert calls and calls[0] is False
