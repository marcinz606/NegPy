import unittest
import numpy as np
import wgpu  # type: ignore
from src.infrastructure.gpu.device import GPUDevice
from src.infrastructure.gpu.resources import GPUTexture, GPUBuffer
from src.infrastructure.gpu.shader_loader import ShaderLoader


class TestGPUInfrastructure(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gpu = GPUDevice.get()

    def test_device_singleton(self):
        """GPUDevice should be a singleton."""
        device2 = GPUDevice.get()
        self.assertIs(self.gpu, device2)

    def test_shader_loader(self):
        """ShaderLoader should load valid WGSL files."""
        import os

        # Test with a known shader file
        shader_path = os.path.join(
            "src", "features", "geometry", "shaders", "transform.wgsl"
        )
        if os.path.exists(shader_path):
            module = ShaderLoader.load(shader_path)
            self.assertIsNotNone(module)
            # Second load should return from cache
            module2 = ShaderLoader.load(shader_path)
            self.assertIs(module, module2)

    def test_gpu_texture(self):
        """GPUTexture should initialize and upload data."""
        if not self.gpu.is_available:
            self.skipTest("GPU not available")

        tex = GPUTexture(100, 100)
        self.assertEqual(tex.width, 100)
        self.assertEqual(tex.height, 100)
        self.assertIsNotNone(tex.texture)
        self.assertIsNotNone(tex.view)

        # Test upload
        data = np.random.rand(100, 100, 3).astype(np.float32)
        tex.upload(data)

        # Test explicit destroy
        tex.destroy()
        self.assertIsNone(tex.texture)
        self.assertIsNone(tex.view)

    def test_gpu_buffer(self):
        """GPUBuffer should initialize and upload data."""
        if not self.gpu.is_available:
            self.skipTest("GPU not available")

        buf = GPUBuffer(1024, wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)
        self.assertIsNotNone(buf.buffer)
        self.assertEqual(buf.buffer.size, 1024)

        # Test upload
        data = np.zeros(256, dtype=np.float32)
        buf.upload(data)

        # Test explicit destroy
        buf.destroy()
        self.assertIsNone(buf.buffer)


if __name__ == "__main__":
    unittest.main()
