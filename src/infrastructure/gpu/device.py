from typing import Optional
import wgpu  # type: ignore
from src.kernel.system.logging import get_logger

logger = get_logger(__name__)


class GPUDevice:
    """
    Singleton manager for the WebGPU device and adapter.
    """

    _instance: Optional["GPUDevice"] = None

    def __init__(self) -> None:
        if GPUDevice._instance is not None:
            raise RuntimeError("GPUDevice is a singleton")

        self.adapter: Optional[wgpu.GPUAdapter] = None
        self.device: Optional[wgpu.GPUDevice] = None
        self._initialize()

    @classmethod
    def get(cls) -> "GPUDevice":
        """
        Returns the singleton instance of GPUDevice.
        """
        if cls._instance is None:
            cls._instance = GPUDevice()
        return cls._instance

    def _initialize(self) -> None:
        """
        Requests adapter and device from the WebGPU implementation.
        """
        try:
            self.adapter = wgpu.gpu.request_adapter_sync(
                power_preference="high-performance"
            )
            if self.adapter:
                self.device = self.adapter.request_device_sync()
                self.limits = self.device.limits
                logger.info(f"GPU Initialized: {self.adapter.summary}")
                logger.info(
                    f"GPU Limits: Max Texture 2D Size = {self.limits.get('max_texture_dimension_2d', 'Unknown')}"
                )
                logger.info(
                    f"GPU Limits: Uniform Alignment = {self.limits.get('min_uniform_buffer_offset_alignment', 256)}"
                )
            else:
                logger.warning("No compatible GPU adapter found")
                self.limits = {}

        except Exception as e:
            logger.error(f"Failed to initialize WebGPU: {e}")
            self.adapter = None
            self.device = None

    @property
    def is_available(self) -> bool:
        """
        Indicates if a valid GPU device is active.
        """
        return self.device is not None

    def poll(self) -> None:
        """
        Polls the GPU device to progress async tasks.
        """
        if self.device:
            if hasattr(self.device, "poll"):
                self.device.poll()
            elif hasattr(self.device, "_poll"):
                self.device._poll()
