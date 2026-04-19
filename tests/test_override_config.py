import logging
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

from negpy.domain.types import AppConfig
from negpy.kernel.system.override import (
    OverrideConfig,
    _parse,
    _platform_defaults,
    apply,
    load_or_create,
)

_ENV_VARS = ("WGPU_BACKEND_TYPE", "QSG_RHI_BACKEND", "QT_QPA_PLATFORM")


def _make_app_config(**kwargs) -> AppConfig:
    defaults = dict(
        thumbnail_size=120,
        max_workers=1,
        preview_render_size=2000,
        max_history_steps=100,
        edits_db_path="/tmp/e.db",
        settings_db_path="/tmp/s.db",
        presets_dir="/tmp/presets",
        cache_dir="/tmp/cache",
        user_icc_dir="/tmp/icc",
        default_export_dir="/tmp/export",
        adobe_rgb_profile="/tmp/adobe.icc",
        override_toml_path="/tmp/override.toml",
    )
    defaults.update(kwargs)
    return AppConfig(**defaults)


class TestOverrideConfigParsing(unittest.TestCase):
    def test_defaults(self):
        cfg = OverrideConfig()
        self.assertEqual(cfg.backend, "auto")
        self.assertEqual(cfg.qt_rhi_backend, "auto")
        self.assertEqual(cfg.qt_platform, "auto")
        self.assertIsNone(cfg.force_hq_preview)
        self.assertIsNone(cfg.max_texture_size)
        self.assertEqual(cfg.log_level, "info")

    def test_log_level_int_info(self):
        self.assertEqual(OverrideConfig(log_level="info").log_level_int, logging.INFO)

    def test_log_level_int_debug(self):
        self.assertEqual(OverrideConfig(log_level="debug").log_level_int, logging.DEBUG)

    def test_log_level_int_warning(self):
        self.assertEqual(OverrideConfig(log_level="warning").log_level_int, logging.WARNING)

    def test_log_level_int_error(self):
        self.assertEqual(OverrideConfig(log_level="error").log_level_int, logging.ERROR)

    def test_log_level_int_unknown_falls_back_to_info(self):
        self.assertEqual(OverrideConfig(log_level="verbose").log_level_int, logging.INFO)

    def test_parse_all_fields(self):
        data = {
            "rendering": {"backend": "dx12"},
            "display": {"qt_rhi_backend": "d3d12", "qt_platform": "xcb"},
            "performance": {"force_hq_preview": True, "max_texture_size": 4096},
            "logging": {"level": "debug"},
        }
        cfg = _parse(data)
        self.assertEqual(cfg.backend, "dx12")
        self.assertEqual(cfg.qt_rhi_backend, "d3d12")
        self.assertEqual(cfg.qt_platform, "xcb")
        self.assertTrue(cfg.force_hq_preview)
        self.assertEqual(cfg.max_texture_size, 4096)
        self.assertEqual(cfg.log_level, "debug")

    def test_parse_invalid_backend_falls_back_to_auto(self):
        cfg = _parse({"rendering": {"backend": "directx9"}})
        self.assertEqual(cfg.backend, "auto")

    def test_parse_invalid_qt_rhi_falls_back_to_auto(self):
        cfg = _parse({"display": {"qt_rhi_backend": "dx11"}})
        self.assertEqual(cfg.qt_rhi_backend, "auto")

    def test_parse_invalid_qt_platform_falls_back_to_auto(self):
        cfg = _parse({"display": {"qt_platform": "windows"}})
        self.assertEqual(cfg.qt_platform, "auto")

    def test_parse_invalid_log_level_falls_back_to_info(self):
        cfg = _parse({"logging": {"level": "trace"}})
        self.assertEqual(cfg.log_level, "info")

    def test_parse_max_texture_size_string_auto_yields_none(self):
        cfg = _parse({"performance": {"max_texture_size": "auto"}})
        self.assertIsNone(cfg.max_texture_size)

    def test_parse_max_texture_size_zero_yields_none(self):
        cfg = _parse({"performance": {"max_texture_size": 0}})
        self.assertIsNone(cfg.max_texture_size)

    def test_parse_force_hq_preview_absent_yields_none(self):
        cfg = _parse({})
        self.assertIsNone(cfg.force_hq_preview)

    def test_parse_empty_dict_returns_all_defaults(self):
        cfg = _parse({})
        self.assertEqual(cfg.backend, "auto")
        self.assertEqual(cfg.qt_rhi_backend, "auto")
        self.assertIsNone(cfg.max_texture_size)

    def test_platform_defaults_macos(self):
        with patch.object(sys, "platform", "darwin"):
            cfg = _platform_defaults()
        self.assertEqual(cfg.backend, "metal")

    def test_platform_defaults_linux(self):
        with patch.object(sys, "platform", "linux"):
            cfg = _platform_defaults()
        self.assertEqual(cfg.backend, "vulkan")

    def test_platform_defaults_windows(self):
        with patch.object(sys, "platform", "win32"):
            cfg = _platform_defaults()
        self.assertEqual(cfg.backend, "vulkan")


class TestLoadOrCreate(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.override_path = os.path.join(self._tmpdir.name, "override.toml")

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_creates_file_when_missing(self):
        load_or_create(self.override_path)
        self.assertTrue(os.path.exists(self.override_path))

    def test_created_file_is_valid_toml(self):
        import tomllib

        load_or_create(self.override_path)
        with open(self.override_path, "rb") as f:
            data = tomllib.load(f)
        self.assertIn("rendering", data)

    def test_created_file_contains_backend_key(self):
        import tomllib

        load_or_create(self.override_path)
        with open(self.override_path, "rb") as f:
            data = tomllib.load(f)
        self.assertIn("backend", data["rendering"])

    def test_linux_default_file_uses_vulkan(self):
        with patch.object(sys, "platform", "linux"):
            load_or_create(self.override_path)
        with open(self.override_path) as f:
            content = f.read()
        self.assertIn('backend = "vulkan"', content)

    def test_macos_default_file_uses_metal(self):
        with patch.object(sys, "platform", "darwin"):
            load_or_create(self.override_path)
        with open(self.override_path) as f:
            content = f.read()
        self.assertIn('backend = "metal"', content)

    def test_loads_existing_file(self):
        with open(self.override_path, "w") as f:
            f.write('[rendering]\nbackend = "dx12"\n')
        cfg = load_or_create(self.override_path)
        self.assertEqual(cfg.backend, "dx12")

    def test_falls_back_to_defaults_on_corrupt_toml(self):
        with open(self.override_path, "w") as f:
            f.write("this is not valid toml ][[\n")
        cfg = load_or_create(self.override_path)
        self.assertIn(cfg.backend, ("vulkan", "metal"))

    def test_does_not_overwrite_existing_file(self):
        with open(self.override_path, "w") as f:
            f.write('[rendering]\nbackend = "cpu"\n')
        load_or_create(self.override_path)
        with open(self.override_path) as f:
            content = f.read()
        self.assertIn('backend = "cpu"', content)

    def test_creates_parent_directory_if_missing(self):
        nested_path = os.path.join(self._tmpdir.name, "subdir", "override.toml")
        load_or_create(nested_path)
        self.assertTrue(os.path.exists(nested_path))


class TestApplyOverride(unittest.TestCase):
    def setUp(self):
        self._saved_env = {k: os.environ.get(k) for k in _ENV_VARS}

    def tearDown(self):
        for k, v in self._saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def test_cpu_disables_gpu(self):
        cfg = OverrideConfig(backend="cpu")
        app_config = _make_app_config(use_gpu=True)
        apply(cfg, app_config)
        self.assertFalse(app_config.use_gpu)

    def test_cpu_does_not_set_wgpu_env_var(self):
        os.environ.pop("WGPU_BACKEND_TYPE", None)
        cfg = OverrideConfig(backend="cpu")
        apply(cfg, _make_app_config())
        self.assertNotIn("WGPU_BACKEND_TYPE", os.environ)

    def test_vulkan_sets_wgpu_backend(self):
        cfg = OverrideConfig(backend="vulkan")
        apply(cfg, _make_app_config())
        self.assertEqual(os.environ.get("WGPU_BACKEND_TYPE"), "Vulkan")

    def test_vulkan_sets_qt_rhi_backend(self):
        cfg = OverrideConfig(backend="vulkan")
        apply(cfg, _make_app_config())
        self.assertEqual(os.environ.get("QSG_RHI_BACKEND"), "vulkan")

    def test_dx12_sets_wgpu_backend(self):
        cfg = OverrideConfig(backend="dx12")
        apply(cfg, _make_app_config())
        self.assertEqual(os.environ.get("WGPU_BACKEND_TYPE"), "Dx12")

    def test_dx12_sets_qt_rhi_backend(self):
        cfg = OverrideConfig(backend="dx12")
        apply(cfg, _make_app_config())
        self.assertEqual(os.environ.get("QSG_RHI_BACKEND"), "d3d12")

    def test_metal_sets_wgpu_backend(self):
        cfg = OverrideConfig(backend="metal")
        apply(cfg, _make_app_config())
        self.assertEqual(os.environ.get("WGPU_BACKEND_TYPE"), "Metal")

    def test_metal_sets_qt_rhi_backend(self):
        cfg = OverrideConfig(backend="metal")
        apply(cfg, _make_app_config())
        self.assertEqual(os.environ.get("QSG_RHI_BACKEND"), "metal")

    def test_auto_backend_does_not_set_wgpu_env_var(self):
        os.environ.pop("WGPU_BACKEND_TYPE", None)
        cfg = OverrideConfig(backend="auto")
        apply(cfg, _make_app_config())
        self.assertNotIn("WGPU_BACKEND_TYPE", os.environ)

    def test_auto_backend_does_not_set_qt_rhi_env_var(self):
        os.environ.pop("QSG_RHI_BACKEND", None)
        cfg = OverrideConfig(backend="auto")
        apply(cfg, _make_app_config())
        self.assertNotIn("QSG_RHI_BACKEND", os.environ)

    def test_auto_backend_keeps_gpu_enabled(self):
        cfg = OverrideConfig(backend="auto")
        app_config = _make_app_config(use_gpu=True)
        apply(cfg, app_config)
        self.assertTrue(app_config.use_gpu)

    def test_independent_qt_rhi_overrides_backend_derived_value(self):
        # backend=vulkan would set vulkan, but qt_rhi_backend=opengl should win
        cfg = OverrideConfig(backend="vulkan", qt_rhi_backend="opengl")
        apply(cfg, _make_app_config())
        self.assertEqual(os.environ.get("QSG_RHI_BACKEND"), "opengl")

    def test_independent_qt_rhi_software_overrides_auto_backend(self):
        cfg = OverrideConfig(backend="auto", qt_rhi_backend="software")
        apply(cfg, _make_app_config())
        self.assertEqual(os.environ.get("QSG_RHI_BACKEND"), "software")

    def test_qt_platform_set_on_linux(self):
        cfg = OverrideConfig(qt_platform="wayland")
        with patch.object(sys, "platform", "linux"):
            apply(cfg, _make_app_config())
        self.assertEqual(os.environ.get("QT_QPA_PLATFORM"), "wayland")

    def test_qt_platform_not_set_on_non_linux(self):
        os.environ.pop("QT_QPA_PLATFORM", None)
        cfg = OverrideConfig(qt_platform="xcb")
        with patch.object(sys, "platform", "darwin"):
            apply(cfg, _make_app_config())
        self.assertNotIn("QT_QPA_PLATFORM", os.environ)

    def test_qt_platform_auto_does_not_set_env_var(self):
        os.environ.pop("QT_QPA_PLATFORM", None)
        cfg = OverrideConfig(qt_platform="auto")
        with patch.object(sys, "platform", "linux"):
            apply(cfg, _make_app_config())
        self.assertNotIn("QT_QPA_PLATFORM", os.environ)

    def test_max_texture_size_propagated_to_app_config(self):
        cfg = OverrideConfig(max_texture_size=4096)
        app_config = _make_app_config()
        apply(cfg, app_config)
        self.assertEqual(app_config.max_texture_size, 4096)

    def test_max_texture_size_none_leaves_app_config_unchanged(self):
        cfg = OverrideConfig(max_texture_size=None)
        app_config = _make_app_config()
        apply(cfg, app_config)
        self.assertIsNone(app_config.max_texture_size)

    def test_force_hq_preview_true_propagated(self):
        cfg = OverrideConfig(force_hq_preview=True)
        app_config = _make_app_config()
        apply(cfg, app_config)
        self.assertTrue(app_config.force_hq_preview)

    def test_force_hq_preview_false_propagated(self):
        cfg = OverrideConfig(force_hq_preview=False)
        app_config = _make_app_config()
        apply(cfg, app_config)
        self.assertFalse(app_config.force_hq_preview)

    def test_force_hq_preview_none_leaves_app_config_unchanged(self):
        cfg = OverrideConfig(force_hq_preview=None)
        app_config = _make_app_config()
        apply(cfg, app_config)
        self.assertIsNone(app_config.force_hq_preview)


if __name__ == "__main__":
    unittest.main()
