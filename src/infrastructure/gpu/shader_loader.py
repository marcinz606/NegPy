import os
from typing import Dict, Any
from src.infrastructure.gpu.device import GPUDevice
from src.kernel.system.logging import get_logger

logger = get_logger(__name__)


class ShaderLoader:
    """
    Utility for loading and caching WGSL shader modules.
    """

    _cache: Dict[str, Any] = {}

    @classmethod
    def load(cls, path: str) -> Any:
        """
        Loads a WGSL shader from the filesystem and creates a shader module.
        """
        if path in cls._cache:
            return cls._cache[path]

        if not os.path.exists(path):
            raise FileNotFoundError(f"Shader not found at {path}")

        with open(path, "r", encoding="utf-8") as f:
            code = f.read()

        gpu = GPUDevice.get()
        if not gpu.device:
            raise RuntimeError("GPU device not initialized")

        shader_module = gpu.device.create_shader_module(code=code)
        cls._cache[path] = shader_module

        logger.info(f"Loaded shader: {os.path.basename(path)}")
        return shader_module
