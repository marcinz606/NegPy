import threading

import numpy as np
import wgpu  # type: ignore
from negpy.infrastructure.gpu.device import GPUDevice
from negpy.kernel.system.logging import get_logger

logger = get_logger(__name__)


class GPUTexture:
    """
    Hardware-backed texture wrapper.
    Defaults to rgba32float for high-dynamic-range processing.
    """

    def __init__(self, width: int, height: int, format: str = "rgba32float", usage: int = 0) -> None:
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

        self.texture = gpu.device.create_texture(size=(width, height, 1), format=format, usage=usage)
        self.view = self.texture.create_view()

        # Persistent staging buffers — allocated lazily, reused across calls
        self._readback_staging = None  # full-texture readback
        self._region_staging = None  # sub-region readback
        self._region_staging_size: int = 0  # allocated size of _region_staging
        # Serializes readback vs. destroy: the UI thread probes textures (densitometer,
        # pixel readout) while the render worker may destroy them (engine cleanup on file
        # switch). Destroying a mapped staging buffer panics in wgpu-native — uncatchable.
        self._lock = threading.Lock()

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

    def readback_region(self, x: int, y: int, rw: int, rh: int) -> np.ndarray:
        """Downloads a sub-region from VRAM. Significantly faster than a full readback."""
        gpu = GPUDevice.get()
        with self._lock:
            if not gpu.device or not self.texture:
                return np.zeros((rh, rw, 4), dtype=np.float32)

            rw = min(rw, max(1, self.width - x))
            rh = min(rh, max(1, self.height - y))

            bytes_per_row = (rw * 16 + 255) & ~255
            required_size = bytes_per_row * rh

            if self._region_staging is None or self._region_staging_size < required_size:
                if self._region_staging is not None:
                    try:
                        self._region_staging.destroy()
                    except Exception as e:
                        logger.warning("Failed to destroy region staging buffer", exc_info=e)
                    try:
                        gpu.poll()
                    except Exception as e:
                        logger.warning("Failed to poll GPU device", exc_info=e)
                self._region_staging = None
                self._region_staging_size = 0
                try:
                    self._region_staging = gpu.device.create_buffer(
                        size=required_size, usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ
                    )
                    self._region_staging_size = required_size
                except Exception as e:
                    logger.warning("Failed to create region staging buffer", exc_info=e)
                    self._region_staging = None
                    self._region_staging_size = 0
                    raise
            staging = self._region_staging

            enc = gpu.device.create_command_encoder()
            enc.copy_texture_to_buffer(
                {"texture": self.texture, "origin": (x, y, 0)},
                {"buffer": staging, "bytes_per_row": bytes_per_row},
                (rw, rh, 1),
            )
            gpu.device.queue.submit([enc.finish()])

            try:
                staging.map_sync(mode=wgpu.MapMode.READ)
                view = staging.read_mapped()
                arr = np.frombuffer(view, dtype=np.float32).reshape((rh, bytes_per_row // 4))
                result = arr[:, : rw * 4].reshape((rh, rw, 4)).copy()
                staging.unmap()
                return result
            except Exception:
                self._region_staging = None
                self._region_staging_size = 0
                raise

    def readback(self) -> np.ndarray:
        """Downloads pixels from VRAM to CPU ndarray (float32)."""
        gpu = GPUDevice.get()
        with self._lock:
            if not gpu.device or not self.texture:
                return np.zeros((self.height, self.width, 4), dtype=np.float32)

            bytes_per_row = (self.width * 16 + 255) & ~255
            size = bytes_per_row * self.height

            if self._readback_staging is None:
                self._readback_staging = gpu.device.create_buffer(size=size, usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ)
            staging = self._readback_staging

            enc = gpu.device.create_command_encoder()
            enc.copy_texture_to_buffer(
                {"texture": self.texture},
                {"buffer": staging, "bytes_per_row": bytes_per_row},
                (self.width, self.height, 1),
            )
            gpu.device.queue.submit([enc.finish()])

            try:
                staging.map_sync(mode=wgpu.MapMode.READ)
                view = staging.read_mapped()
                arr = np.frombuffer(view, dtype=np.float32).reshape((self.height, bytes_per_row // 4))
                result = arr[:, : self.width * 4].reshape((self.height, self.width, 4)).copy()
                staging.unmap()
                return result
            except Exception:
                self._readback_staging = None
                raise

    def destroy(self) -> None:
        """Forces hardware resource release."""
        with self._lock:
            try:
                self.view = None
                if self.texture:
                    self.texture.destroy()
                    self.texture = None
            except Exception as e:
                logger.warning("Failed to destroy GPU texture", exc_info=e)
            for attr in ("_readback_staging", "_region_staging"):
                buf = getattr(self, attr, None)
                if buf is not None:
                    try:
                        buf.destroy()
                    except Exception as e:
                        logger.warning(f"Failed to destroy GPU buffer ({attr})", exc_info=e)
                    setattr(self, attr, None)
            self._region_staging_size = 0


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
        except Exception as e:
            logger.warning("Failed to destroy GPU buffer", exc_info=e)
