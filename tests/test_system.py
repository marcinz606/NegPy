import os
from negpy.kernel.system.version import get_app_version
from negpy.kernel.system.paths import get_resource_path


def test_get_app_version(tmp_path):
    # Mock VERSION file
    version_dir = tmp_path / "src" / "negpy" / "kernel" / "system"
    version_dir.mkdir(parents=True)

    # We need to mock get_resource_path or just hope it finds VERSION in root
    # Since we are in tests, root is the current directory.
    v = get_app_version()
    assert isinstance(v, str)


def test_get_resource_path():
    p = get_resource_path("negpy/kernel/system/paths.py")
    assert os.path.exists(p)
    assert os.path.isabs(p)


def test_shader_files_exist():
    """All GPU shader files must be present alongside the package (pip install .)."""
    shaders = [
        "negpy/features/geometry/shaders/transform.wgsl",
        "negpy/features/geometry/shaders/autocrop.wgsl",
        "negpy/features/exposure/shaders/exposure.wgsl",
        "negpy/features/exposure/shaders/normalization.wgsl",
        "negpy/features/lab/shaders/lab.wgsl",
        "negpy/features/lab/shaders/clahe_hist.wgsl",
        "negpy/features/lab/shaders/clahe_cdf.wgsl",
        "negpy/features/lab/shaders/clahe_apply.wgsl",
        "negpy/features/lab/shaders/metrics.wgsl",
        "negpy/features/retouch/shaders/retouch.wgsl",
        "negpy/features/toning/shaders/toning.wgsl",
        "negpy/features/toning/shaders/layout.wgsl",
    ]
    for shader in shaders:
        p = get_resource_path(shader)
        assert os.path.exists(p), f"Shader missing: {p}"
