import numpy as np
import wgpu  # type: ignore
from src.infrastructure.gpu.device import GPUDevice


class GPUTexture:
    """
    Hardware-backed texture wrapper.
    Defaults to rgba32float for high-dynamic-range processing.
    """

    def __init__(
        self, width: int, height: int, format: str = "rgba32float", usage: int = 0
    ) -> None:
        self.width, self.height, self.format = width, height, format
        gpu = GPUDevice.get()
        if not gpu.device:
            raise RuntimeError("Hardware device required")

        if usage == 0:
            usage = (
                wgpu.TextureUsage.TEXTURE_BINDING
                | wgpu.TextureUsage.STORAGE_BINDING
                | wgpu.TextureUsage.COPY_DST
                | wgpu.TextureUsage.COPY_SRC
            )

        self.texture = gpu.device.create_texture(
            size=(width, height, 1), format=format, usage=usage
        )
        self.view = self.texture.create_view()

    def upload(self, data: np.ndarray) -> None:
        """Transfers ndarray to VRAM."""
        gpu = GPUDevice.get()
        if not gpu.device:
            return

        if data.dtype != np.float32:
            data = data.astype(np.float32)
        if data.shape[2] == 3:
            rgba = np.ones((data.shape[0], data.shape[1], 4), dtype=np.float32)
            rgba[:, :, :3] = data
            data = rgba

        gpu.device.queue.write_texture(
            {"texture": self.texture},
            data,
            {"bytes_per_row": data.shape[1] * 16, "rows_per_image": data.shape[0]},
            (data.shape[1], data.shape[0], 1),
        )

    def destroy(self) -> None:
        """Forces hardware resource release."""
        try:
            self.view = None
            if self.texture:
                self.texture.destroy()
                self.texture = None
        except Exception:
            pass


class GPUBuffer:
    """Uniform or storage buffer wrapper."""

    def __init__(self, size: int, usage: int) -> None:
        gpu = GPUDevice.get()
        if not gpu.device:
            raise RuntimeError("Hardware device required")
        self.buffer = gpu.device.create_buffer(size=size, usage=usage)

    def upload(self, data: np.ndarray) -> None:
        gpu = GPUDevice.get()
        if not gpu.device:
            return
        gpu.device.queue.write_buffer(self.buffer, 0, data)

    def destroy(self) -> None:
        try:
            if self.buffer:
                self.buffer.destroy()
                self.buffer = None
        except Exception:
            pass
