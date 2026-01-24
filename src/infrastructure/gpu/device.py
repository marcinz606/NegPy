from typing import Optional
import wgpu  # type: ignore
from src.kernel.system.logging import get_logger

logger = get_logger(__name__)


class GPUDevice:
    """
    Hardware adapter and device manager.
    Singleton pattern ensures single initialization per process.
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
        if cls._instance is None:
            cls._instance = GPUDevice()
        return cls._instance

    def _initialize(self) -> None:
        try:
            self.adapter = wgpu.gpu.request_adapter_sync(
                power_preference="high-performance"
            )
            if self.adapter:
                self.device = self.adapter.request_device_sync()
                self.limits = self.device.limits
                logger.info(f"WebGPU Backend: {self.adapter.summary}")
            else:
                logger.warning("No compatible GPU adapter found")
                self.limits = {}
        except Exception as e:
            logger.error(f"Hardware initialization failed: {e}")
            self.adapter = None
            self.device = None

    @property
    def is_available(self) -> bool:
        return self.device is not None

    @property
    def backend_name(self) -> Optional[str]:
        if not self.adapter:
            return None
        summary = str(self.adapter.summary)
        if "(" in summary:
            return str(summary.split("(")[-1].replace(")", "").strip())
        return str(summary.split()[-1])

    def poll(self) -> None:
        """Forces hardware queue processing for async operations."""
        if self.device:
            if hasattr(self.device, "poll"):
                self.device.poll()
            elif hasattr(self.device, "_poll"):
                self.device._poll()
