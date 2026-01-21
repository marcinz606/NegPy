import os
from typing import Any
from src.infrastructure.gpu.device import GPUDevice


class ShaderLoader:
    """
    On-demand WGSL shader compiler and cache.
    Reduces pipeline initialization overhead by reusing modules.
    """

    _cache: dict[str, Any] = {}

    @classmethod
    def load(cls, path: str) -> Any:
        if path in cls._cache:
            return cls._cache[path]

        if not os.path.exists(path):
            raise FileNotFoundError(f"Shader source missing: {path}")

        with open(path, "r") as f:
            code = f.read()

        gpu = GPUDevice.get()
        if not gpu.device:
            raise RuntimeError("Hardware device required for shader compilation")

        module = gpu.device.create_shader_module(code=code)
        cls._cache[path] = module
        return module
