import numpy as np
from src.infrastructure.gpu.device import GPUDevice


class GPUTexture:
    """
    Wrapper for WebGPU textures.
    """

    def __init__(
        self, width: int, height: int, format: str = "rgba32float", usage: int = 0
    ) -> None:
        self.width = width
        self.height = height
        self.format = format

        gpu = GPUDevice.get()
        if not gpu.device:
            raise RuntimeError("GPU device not available")

        # Combine default usages if not provided
        if usage == 0:
            import wgpu  # type: ignore

            usage = (
                wgpu.TextureUsage.TEXTURE_BINDING
                | wgpu.TextureUsage.STORAGE_BINDING
                | wgpu.TextureUsage.COPY_DST
                | wgpu.TextureUsage.COPY_SRC
            )

        self.texture = gpu.device.create_texture(
            size=(width, height, 1),
            format=format,
            usage=usage,
        )
        self.view = self.texture.create_view()

    def upload(self, data: np.ndarray) -> None:
        """
        Uploads a numpy array to the texture.
        """
        gpu = GPUDevice.get()
        if not gpu.device:
            return

        # Ensure data is contiguous and correct type
        if data.dtype != np.float32:
            data = data.astype(np.float32)

        # WebGPU expects 4 components for rgba32float
        if data.shape[2] == 3:
            # Add alpha channel
            h, w = data.shape[:2]
            rgba = np.ones((h, w, 4), dtype=np.float32)
            rgba[:, :, :3] = data
            data = rgba

        gpu.device.queue.write_texture(
            {"texture": self.texture},
            data,
            {"bytes_per_row": data.shape[1] * 16, "rows_per_image": data.shape[0]},
            (data.shape[1], data.shape[0], 1),
        )


class GPUBuffer:
    """
    Wrapper for WebGPU buffers (Uniform/Storage).
    """

    def __init__(self, size: int, usage: int) -> None:
        gpu = GPUDevice.get()
        if not gpu.device:
            raise RuntimeError("GPU device not available")

        self.buffer = gpu.device.create_buffer(size=size, usage=usage)

    def upload(self, data: np.ndarray) -> None:
        """
        Uploads data to the buffer.
        """
        gpu = GPUDevice.get()
        if not gpu.device:
            return
        gpu.device.queue.write_buffer(self.buffer, 0, data)
