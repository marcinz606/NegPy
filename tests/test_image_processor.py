import numpy as np
from src.services.rendering.image_processor import ImageProcessor
from src.domain.models import WorkspaceConfig


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
    from src.kernel.image.logic import uint16_to_float32, uint8_to_float32

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
