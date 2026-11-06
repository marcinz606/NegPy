import os
import threading
from typing import Any
from negpy.infrastructure.gpu.device import GPUDevice


class ShaderLoader:
    """
    On-demand WGSL shader compiler and cache.
    Reduces pipeline initialization overhead by reusing modules.
    """

    _cache: dict[str, Any] = {}
    # A module belongs to the device that compiled it: a new device after destroy_singleton empties the cache.
    _device: Any = None
    _lock = threading.Lock()

    @classmethod
    def load(cls, path: str) -> Any:
        with cls._lock:
            gpu = GPUDevice.get()
            if gpu.device is not cls._device:
                cls._cache.clear()
                cls._device = gpu.device
            if path in cls._cache:
                return cls._cache[path]

            if not os.path.exists(path):
                raise FileNotFoundError(f"Shader source missing: {path}")

            with open(path, "r", encoding="utf-8") as f:
                code = f.read()

            if not gpu.device:
                raise RuntimeError("Hardware device required for shader compilation")

            module = gpu.device.create_shader_module(code=code)
            cls._cache[path] = module
            return module
