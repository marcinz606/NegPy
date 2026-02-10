"""Tests for negpy.cli.batch — CLI batch converter."""

import os

os.environ.setdefault("NUMBA_THREADING_LAYER", "workqueue")

import pytest

import argparse
import json

from unittest.mock import patch, MagicMock

from negpy.cli.batch import (
    build_parser, discover_files, build_config, main,
    load_user_config, generate_default_config, list_available_presets,
    CONFIG_SCHEMA_URL,
)
from negpy.domain.models import ExportFormat
from negpy.features.process.models import ProcessMode
from negpy.kernel.system.config import DEFAULT_WORKSPACE_CONFIG


@pytest.fixture(scope="session", autouse=True)
def qapp():
    """Override conftest.py qapp — CLI tests don't need Qt."""
    yield None


class TestBuildParser:
    def test_minimal_args(self):
        """Single file with all defaults."""
        parser = build_parser()
        args = parser.parse_args(["input.dng"])
        assert args.inputs == ["input.dng"]
        assert args.mode == "c41"
        assert args.output_format == "tiff"
        assert args.output == "./export"
        assert args.color_space == "adobe-rgb"
        assert args.density is None
        assert args.grade is None
        assert args.sharpen is None
        assert args.dpi is None
        assert args.print_size is None
        assert args.original_res is False
        assert args.filename_pattern is None
        assert args.no_gpu is False
        assert args.settings is None
        assert args.crop_offset is None
        assert args.flat_field is None
        assert args.preset is None
        assert args.list_presets is False
        assert args.init_config is False

    def test_all_flags(self):
        """Every flag specified."""
        parser = build_parser()
        args = parser.parse_args([
            "--mode", "bw",
            "--format", "jpeg",
            "--output", "/tmp/out",
            "--color-space", "prophoto",
            "--density", "1.5",
            "--grade", "3.0",
            "--sharpen", "0.5",
            "--dpi", "600",
            "--print-size", "40.0",
            "--original-res",
            "--filename-pattern", "{{ original_name }}_final",
            "--no-gpu",
            "--settings", "my_settings.json",
            "--crop-offset", "5",
            "--flat-field", "blank_scan.tiff",
            "--preset", "portra-400",
            "file1.dng", "file2.tiff",
        ])
        assert args.mode == "bw"
        assert args.output_format == "jpeg"
        assert args.output == "/tmp/out"
        assert args.color_space == "prophoto"
        assert args.density == 1.5
        assert args.grade == 3.0
        assert args.sharpen == 0.5
        assert args.dpi == 600
        assert args.print_size == 40.0
        assert args.original_res is True
        assert args.filename_pattern == "{{ original_name }}_final"
        assert args.no_gpu is True
        assert args.settings == "my_settings.json"
        assert args.crop_offset == 5
        assert args.flat_field == "blank_scan.tiff"
        assert args.preset == "portra-400"
        assert args.inputs == ["file1.dng", "file2.tiff"]

    def test_init_config_no_inputs_required(self):
        """--init-config should work without input files."""
        parser = build_parser()
        args = parser.parse_args(["--init-config"])
        assert args.init_config is True
        assert args.inputs == []

    def test_list_presets_no_inputs_required(self):
        """--list-presets should work without input files."""
        parser = build_parser()
        args = parser.parse_args(["--list-presets"])
        assert args.list_presets is True
        assert args.inputs == []

    def test_multiple_inputs(self):
        parser = build_parser()
        args = parser.parse_args(["a.dng", "b.tiff", "/scans/"])
        assert len(args.inputs) == 3

    def test_no_inputs_returns_error(self):
        """No inputs and no special command -> exit code 1."""
        exit_code = main([])
        assert exit_code == 1

    def test_invalid_mode_raises(self):
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--mode", "invalid", "file.dng"])

    def test_invalid_format_raises(self):
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--format", "png", "file.dng"])


class TestDiscoverFiles:
    def test_single_supported_file(self, tmp_path):
        f = tmp_path / "scan.dng"
        f.write_bytes(b"fake")
        result = discover_files([str(f)])
        assert result == [str(f)]

    def test_unsupported_file_skipped(self, tmp_path, capsys):
        f = tmp_path / "readme.txt"
        f.write_text("hello")
        result = discover_files([str(f)])
        assert result == []
        captured = capsys.readouterr()
        assert "Skipping unsupported" in captured.err

    def test_directory_walk(self, tmp_path):
        (tmp_path / "a.dng").write_bytes(b"fake")
        (tmp_path / "b.tiff").write_bytes(b"fake")
        (tmp_path / "c.txt").write_text("skip")
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "d.nef").write_bytes(b"fake")
        result = discover_files([str(tmp_path)])
        extensions = {os.path.splitext(f)[1] for f in result}
        assert extensions == {".dng", ".tiff", ".nef"}
        assert len(result) == 3

    def test_missing_path_warns(self, capsys):
        result = discover_files(["/nonexistent/path"])
        assert result == []
        captured = capsys.readouterr()
        assert "Path not found" in captured.err

    def test_mixed_files_and_dirs(self, tmp_path):
        f = tmp_path / "direct.cr2"
        f.write_bytes(b"fake")
        d = tmp_path / "folder"
        d.mkdir()
        (d / "inside.arw").write_bytes(b"fake")
        result = discover_files([str(f), str(d)])
        assert len(result) == 2

    def test_jpeg_and_tiff_accepted(self, tmp_path):
        (tmp_path / "photo.jpg").write_bytes(b"fake")
        (tmp_path / "scan.tif").write_bytes(b"fake")
        result = discover_files([str(tmp_path)])
        assert len(result) == 2


class TestBuildConfig:
    def _make_args(self, **overrides):
        defaults = {
            "mode": "c41",
            "output_format": "tiff",
            "output": "./export",
            "color_space": "adobe-rgb",
            "density": None,
            "grade": None,
            "sharpen": None,
            "dpi": None,
            "print_size": None,
            "original_res": False,
            "filename_pattern": None,
            "no_gpu": False,
            "settings": None,
            "crop_offset": None,
            "preset": None,
            "inputs": ["file.dng"],
        }
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    def _empty_user_config(self):
        return {"cli": {}, "processing": {}}

    def _build(self, user_config=None, **args_overrides):
        """Helper: build_config with args overrides and optional user_config."""
        return build_config(self._make_args(**args_overrides), user_config or self._empty_user_config())

    def test_defaults_preserve_workspace_config(self):
        args = self._make_args()
        config = build_config(args, {"cli": {}, "processing": {}})
        assert config.process.process_mode == ProcessMode.C41
        assert config.exposure.density == DEFAULT_WORKSPACE_CONFIG.exposure.density
        assert config.exposure.grade == DEFAULT_WORKSPACE_CONFIG.exposure.grade
        assert config.lab.sharpen == DEFAULT_WORKSPACE_CONFIG.lab.sharpen

    def test_mode_bw(self):
        config = build_config(self._make_args(mode="bw"), {"cli": {}, "processing": {}})
        assert config.process.process_mode == ProcessMode.BW

    def test_mode_e6(self):
        config = build_config(self._make_args(mode="e6"), {"cli": {}, "processing": {}})
        assert config.process.process_mode == ProcessMode.E6

    def test_exposure_overrides(self):
        config = build_config(self._make_args(density=2.0, grade=4.0), {"cli": {}, "processing": {}})
        assert config.exposure.density == 2.0
        assert config.exposure.grade == 4.0

    def test_sharpen_override(self):
        config = build_config(self._make_args(sharpen=0.8), {"cli": {}, "processing": {}})
        assert config.lab.sharpen == 0.8

    def test_export_format_jpeg(self):
        config = build_config(self._make_args(output_format="jpeg"), {"cli": {}, "processing": {}})
        assert config.export.export_fmt == ExportFormat.JPEG

    def test_export_format_tiff_default(self):
        config = build_config(self._make_args(), {"cli": {}, "processing": {}})
        assert config.export.export_fmt == ExportFormat.TIFF

    def test_color_space_prophoto(self):
        config = build_config(self._make_args(color_space="prophoto"), {"cli": {}, "processing": {}})
        assert config.export.export_color_space == "ProPhoto RGB"

    def test_dpi_and_print_size(self):
        config = build_config(self._make_args(dpi=600, print_size=40.0), {"cli": {}, "processing": {}})
        assert config.export.export_dpi == 600
        assert config.export.export_print_size == 40.0

    def test_original_res(self):
        config = build_config(self._make_args(original_res=True), {"cli": {}, "processing": {}})
        assert config.export.use_original_res is True

    def test_filename_pattern(self):
        config = build_config(self._make_args(filename_pattern="{{ date }}_{{ original_name }}"), {"cli": {}, "processing": {}})
        assert config.export.filename_pattern == "{{ date }}_{{ original_name }}"

    def test_crop_offset_override(self):
        config = build_config(self._make_args(crop_offset=10), {"cli": {}, "processing": {}})
        assert config.geometry.autocrop_offset == 10

    def test_crop_offset_negative(self):
        config = build_config(self._make_args(crop_offset=-3), {"cli": {}, "processing": {}})
        assert config.geometry.autocrop_offset == -3

    def test_crop_offset_none_preserves_default(self):
        config = build_config(self._make_args(), {"cli": {}, "processing": {}})
        assert config.geometry.autocrop_offset == DEFAULT_WORKSPACE_CONFIG.geometry.autocrop_offset

    def test_none_overrides_preserve_base(self):
        config = build_config(self._make_args(), {"cli": {}, "processing": {}})
        assert config.exposure.density == DEFAULT_WORKSPACE_CONFIG.exposure.density
        assert config.lab.sharpen == DEFAULT_WORKSPACE_CONFIG.lab.sharpen
        assert config.export.export_dpi == DEFAULT_WORKSPACE_CONFIG.export.export_dpi

    def test_settings_file_load(self, tmp_path):
        settings = {"process_mode": "B&W", "density": 0.8, "grade": 1.5}
        settings_file = tmp_path / "settings.json"
        settings_file.write_text(json.dumps(settings))
        args = self._make_args(settings=str(settings_file))
        config = build_config(args, {"cli": {}, "processing": {}})
        # --mode c41 still overrides the JSON process_mode
        assert config.process.process_mode == ProcessMode.C41
        # JSON values used as base
        assert config.exposure.density == 0.8

    def test_settings_file_with_cli_override(self, tmp_path):
        settings = {"density": 0.8, "grade": 1.5}
        settings_file = tmp_path / "settings.json"
        settings_file.write_text(json.dumps(settings))
        args = self._make_args(settings=str(settings_file), density=2.5)
        config = build_config(args, {"cli": {}, "processing": {}})
        assert config.exposure.density == 2.5
        assert config.exposure.grade == 1.5

    def test_missing_settings_file_raises(self):
        args = self._make_args(settings="/nonexistent/file.json")
        with pytest.raises(FileNotFoundError):
            build_config(args, {"cli": {}, "processing": {}})

    def test_invalid_json_raises(self, tmp_path):
        bad_file = tmp_path / "bad.json"
        bad_file.write_text("not json {{{")
        args = self._make_args(settings=str(bad_file))
        with pytest.raises(json.JSONDecodeError):
            build_config(args, {"cli": {}, "processing": {}})


class TestMain:
    @pytest.fixture(autouse=True)
    def isolate_test_environment(self, tmp_path, monkeypatch):
        """Ensure tests don't affect global state or load the real user config."""
        # Isolate user config
        fake_config_dir = tmp_path / ".negpy_test_isolation"
        fake_config_dir.mkdir()
        monkeypatch.setattr("negpy.cli.batch.CONFIG_FILE", str(fake_config_dir / "config.json"))
        monkeypatch.setattr("negpy.cli.batch.PRESETS_DIR", str(fake_config_dir / "presets"))
        # Preserve APP_CONFIG.use_gpu state (--no-gpu modifies global state)
        from negpy.kernel.system.config import APP_CONFIG
        monkeypatch.setattr(APP_CONFIG, "use_gpu", APP_CONFIG.use_gpu)

    @patch("negpy.cli.batch.ImageProcessor")
    def test_no_files_returns_1(self, mock_proc_cls, tmp_path):
        """Empty directory -> exit code 1."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        exit_code = main([str(empty_dir)])
        assert exit_code == 1

    @patch("negpy.cli.batch.ImageProcessor")
    def test_successful_batch(self, mock_proc_cls, tmp_path):
        """Two files, both succeed."""
        (tmp_path / "a.dng").write_bytes(b"fake")
        (tmp_path / "b.dng").write_bytes(b"fake")
        out_dir = tmp_path / "output"

        mock_processor = MagicMock()
        mock_processor.process_export.return_value = (b"fake_image_bytes", "tiff")
        mock_proc_cls.return_value = mock_processor

        exit_code = main(["--output", str(out_dir), str(tmp_path / "a.dng"), str(tmp_path / "b.dng")])

        assert exit_code == 0
        assert mock_processor.process_export.call_count == 2
        assert mock_processor.cleanup.call_count == 2
        assert mock_processor.destroy_all.call_count == 1

    @patch("negpy.cli.batch.ImageProcessor")
    def test_partial_failure_returns_1(self, mock_proc_cls, tmp_path):
        """One succeeds, one fails -> exit code 1."""
        (tmp_path / "good.dng").write_bytes(b"fake")
        (tmp_path / "bad.dng").write_bytes(b"fake")
        out_dir = tmp_path / "output"

        mock_processor = MagicMock()
        mock_processor.process_export.side_effect = [
            (b"image_bytes", "tiff"),
            (None, "decode error"),
        ]
        mock_proc_cls.return_value = mock_processor

        exit_code = main(["--output", str(out_dir), str(tmp_path / "good.dng"), str(tmp_path / "bad.dng")])
        assert exit_code == 1

    @patch("negpy.cli.batch.ImageProcessor")
    def test_output_files_written(self, mock_proc_cls, tmp_path):
        """Verify actual files appear in the output directory."""
        (tmp_path / "scan.dng").write_bytes(b"fake")
        out_dir = tmp_path / "output"

        mock_processor = MagicMock()
        mock_processor.process_export.return_value = (b"TIFF_BYTES", "tiff")
        mock_proc_cls.return_value = mock_processor

        main(["--output", str(out_dir), str(tmp_path / "scan.dng")])

        output_files = list(out_dir.iterdir())
        assert len(output_files) == 1
        assert output_files[0].suffix == ".tiff"
        assert output_files[0].read_bytes() == b"TIFF_BYTES"

    @patch("negpy.cli.batch.ImageProcessor")
    def test_jpeg_output_extension(self, mock_proc_cls, tmp_path):
        """JPEG format uses .jpg extension."""
        (tmp_path / "scan.dng").write_bytes(b"fake")
        out_dir = tmp_path / "output"

        mock_processor = MagicMock()
        mock_processor.process_export.return_value = (b"JPEG_BYTES", "jpg")
        mock_proc_cls.return_value = mock_processor

        main(["--format", "jpeg", "--output", str(out_dir), str(tmp_path / "scan.dng")])

        output_files = list(out_dir.iterdir())
        assert len(output_files) == 1
        assert output_files[0].suffix == ".jpg"

    @patch("negpy.cli.batch.ImageProcessor")
    def test_no_gpu_flag(self, mock_proc_cls, tmp_path):
        """--no-gpu should pass prefer_gpu=False to process_export."""
        (tmp_path / "a.dng").write_bytes(b"fake")
        out_dir = tmp_path / "output"

        mock_processor = MagicMock()
        mock_processor.process_export.return_value = (b"bytes", "tiff")
        mock_proc_cls.return_value = mock_processor

        main(["--no-gpu", "--output", str(out_dir), str(tmp_path / "a.dng")])

        _, kwargs = mock_processor.process_export.call_args
        assert kwargs["prefer_gpu"] is False

    @patch("negpy.cli.batch.ImageProcessor")
    def test_exception_during_processing_continues(self, mock_proc_cls, tmp_path):
        """An exception on one file doesn't abort the batch."""
        (tmp_path / "a.dng").write_bytes(b"fake")
        (tmp_path / "b.dng").write_bytes(b"fake")
        out_dir = tmp_path / "output"

        mock_processor = MagicMock()
        mock_processor.process_export.side_effect = [
            RuntimeError("corrupt file"),
            (b"ok_bytes", "tiff"),
        ]
        mock_proc_cls.return_value = mock_processor

        exit_code = main(["--output", str(out_dir), str(tmp_path / "a.dng"), str(tmp_path / "b.dng")])
        assert exit_code == 1  # partial failure
        assert mock_processor.process_export.call_count == 2

    @patch("negpy.cli.batch.ImageProcessor")
    def test_directory_input(self, mock_proc_cls, tmp_path):
        """Passing a directory processes all supported files inside."""
        (tmp_path / "roll01.cr2").write_bytes(b"fake")
        (tmp_path / "roll02.cr2").write_bytes(b"fake")
        (tmp_path / "notes.txt").write_text("ignore")
        out_dir = tmp_path / "output"

        mock_processor = MagicMock()
        mock_processor.process_export.return_value = (b"bytes", "tiff")
        mock_proc_cls.return_value = mock_processor

        exit_code = main(["--output", str(out_dir), str(tmp_path)])
        assert exit_code == 0
        assert mock_processor.process_export.call_count == 2

    @patch("negpy.cli.batch.ImageProcessor")
    @patch("negpy.cli.batch.load_raw_to_float32")
    @patch("negpy.cli.batch.load_flatfield")
    @patch("negpy.cli.batch.apply_flatfield")
    def test_flat_field_loads_and_applies(self, mock_apply, mock_load_ff, mock_load_raw, mock_proc_cls, tmp_path):
        """--flat-field should load the flat, load each raw, apply correction, and use run_pipeline."""
        (tmp_path / "a.dng").write_bytes(b"fake")
        out_dir = tmp_path / "output"
        ff_file = tmp_path / "blank.tiff"
        ff_file.write_bytes(b"flat")

        import numpy as np
        fake_flat = np.ones((100, 100, 3), dtype=np.float32)
        fake_raw = np.ones((100, 100, 3), dtype=np.float32) * 0.5
        fake_corrected = np.ones((100, 100, 3), dtype=np.float32) * 0.5
        fake_result = np.ones((50, 50, 3), dtype=np.float32) * 0.7

        mock_load_ff.return_value = fake_flat
        mock_load_raw.return_value = fake_raw
        mock_apply.return_value = fake_corrected

        mock_processor = MagicMock()
        mock_processor.run_pipeline.return_value = (fake_result, {})
        mock_proc_cls.return_value = mock_processor

        exit_code = main(["--flat-field", str(ff_file), "--output", str(out_dir), str(tmp_path / "a.dng")])

        assert exit_code == 0
        mock_load_ff.assert_called_once_with(str(ff_file))
        mock_load_raw.assert_called_once()
        mock_apply.assert_called_once()
        mock_processor.run_pipeline.assert_called_once()
        # process_export should NOT be called when --flat-field is used
        mock_processor.process_export.assert_not_called()

    @patch("negpy.cli.batch.ImageProcessor")
    @patch("negpy.cli.batch.load_raw_to_float32")
    @patch("negpy.cli.batch.load_flatfield")
    @patch("negpy.cli.batch.apply_flatfield")
    def test_flat_field_with_gpu_texture_readback(self, mock_apply, mock_load_ff, mock_load_raw, mock_proc_cls, tmp_path):
        """When GPU returns a texture, readback() should be called on the texture itself."""
        (tmp_path / "a.dng").write_bytes(b"fake")
        out_dir = tmp_path / "output"
        ff_file = tmp_path / "blank.tiff"
        ff_file.write_bytes(b"flat")

        import numpy as np
        fake_flat = np.ones((100, 100, 3), dtype=np.float32)
        fake_raw = np.ones((100, 100, 3), dtype=np.float32) * 0.5
        fake_corrected = np.ones((100, 100, 3), dtype=np.float32) * 0.5
        # Simulate RGBA output from GPU (4 channels)
        fake_rgba_result = np.ones((50, 50, 4), dtype=np.float32) * 0.7

        mock_load_ff.return_value = fake_flat
        mock_load_raw.return_value = fake_raw
        mock_apply.return_value = fake_corrected

        # Create a mock GPUTexture that has a readback() method
        mock_texture = MagicMock()
        mock_texture.readback.return_value = fake_rgba_result

        mock_processor = MagicMock()
        mock_processor.run_pipeline.return_value = (mock_texture, {})
        mock_proc_cls.return_value = mock_processor

        exit_code = main(["--flat-field", str(ff_file), "--output", str(out_dir), str(tmp_path / "a.dng")])

        assert exit_code == 0
        # Verify readback() was called on the texture itself (not on engine_gpu)
        mock_texture.readback.assert_called_once()
        mock_processor.engine_gpu.readback.assert_not_called()

    @patch("negpy.cli.batch.ImageProcessor")
    def test_flat_field_missing_file_returns_1(self, mock_proc_cls, tmp_path):
        """Non-existent flat-field path should return exit code 1."""
        (tmp_path / "a.dng").write_bytes(b"fake")
        out_dir = tmp_path / "output"

        exit_code = main(["--flat-field", "/nonexistent/blank.tiff", "--output", str(out_dir), str(tmp_path / "a.dng")])
        assert exit_code == 1


class TestUserConfig:
    def test_load_user_config_missing_file(self, tmp_path, monkeypatch):
        """When config file doesn't exist, returns empty dicts."""
        monkeypatch.setattr("negpy.cli.batch.CONFIG_FILE", str(tmp_path / "nope.json"))
        result = load_user_config()
        assert result == {"cli": {}, "processing": {}}

    def test_load_user_config_valid(self, tmp_path, monkeypatch):
        """Loads cli and processing sections from config."""
        cfg = {"cli": {"mode": "bw", "flat_field": "/flat.tiff"}, "processing": {"density": 1.5}}
        cfg_file = tmp_path / "config.json"
        cfg_file.write_text(json.dumps(cfg))
        monkeypatch.setattr("negpy.cli.batch.CONFIG_FILE", str(cfg_file))
        result = load_user_config()
        assert result["cli"]["mode"] == "bw"
        assert result["cli"]["flat_field"] == "/flat.tiff"
        assert result["processing"]["density"] == 1.5

    def test_generate_default_config_creates_file(self, tmp_path, monkeypatch):
        """--init-config creates config.json and presets dir with $schema."""
        monkeypatch.setattr("negpy.cli.batch.CONFIG_DIR", str(tmp_path / ".negpy"))
        monkeypatch.setattr("negpy.cli.batch.CONFIG_FILE", str(tmp_path / ".negpy" / "config.json"))
        monkeypatch.setattr("negpy.cli.batch.PRESETS_DIR", str(tmp_path / ".negpy" / "presets"))
        code = generate_default_config()
        assert code == 0
        assert (tmp_path / ".negpy" / "config.json").exists()
        assert (tmp_path / ".negpy" / "presets").is_dir()
        data = json.loads((tmp_path / ".negpy" / "config.json").read_text())
        assert data["$schema"] == CONFIG_SCHEMA_URL
        assert "cli" in data
        assert "processing" in data

    def test_generate_default_config_refuses_overwrite(self, tmp_path, monkeypatch):
        """--init-config refuses to overwrite existing config."""
        negpy_dir = tmp_path / ".negpy"
        negpy_dir.mkdir()
        (negpy_dir / "config.json").write_text("{}")
        monkeypatch.setattr("negpy.cli.batch.CONFIG_DIR", str(negpy_dir))
        monkeypatch.setattr("negpy.cli.batch.CONFIG_FILE", str(negpy_dir / "config.json"))
        monkeypatch.setattr("negpy.cli.batch.PRESETS_DIR", str(negpy_dir / "presets"))
        code = generate_default_config()
        assert code == 1


class TestPresets:
    def test_list_presets_empty(self, tmp_path, monkeypatch):
        """No presets dir -> informational message, exit 0."""
        monkeypatch.setattr("negpy.cli.batch.PRESETS_DIR", str(tmp_path / "nonexistent"))
        code = list_available_presets()
        assert code == 0

    def test_list_presets_with_files(self, tmp_path, monkeypatch, capsys):
        """Lists preset names from presets dir."""
        presets_dir = tmp_path / "presets"
        presets_dir.mkdir()
        (presets_dir / "portra-400.json").write_text('{"density": 1.2}')
        (presets_dir / "tri-x.json").write_text('{"density": 0.9}')
        (presets_dir / "not-json.txt").write_text("skip me")
        monkeypatch.setattr("negpy.cli.batch.PRESETS_DIR", str(presets_dir))
        code = list_available_presets()
        assert code == 0
        captured = capsys.readouterr()
        assert "portra-400" in captured.err
        assert "tri-x" in captured.err
        assert "not-json" not in captured.err


class TestBuildConfigPriority:
    """Tests for the config loading priority chain."""

    def _make_args(self, **overrides):
        defaults = {
            "mode": "c41",
            "output_format": "tiff",
            "output": "./export",
            "color_space": "adobe-rgb",
            "density": None,
            "grade": None,
            "sharpen": None,
            "dpi": None,
            "print_size": None,
            "original_res": False,
            "filename_pattern": None,
            "no_gpu": False,
            "settings": None,
            "crop_offset": None,
            "preset": None,
            "inputs": ["file.dng"],
        }
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    def test_user_config_processing_applied(self):
        """User config processing values should be used as base."""
        user_config = {"cli": {}, "processing": {"density": 1.5, "grade": 3.0}}
        config = build_config(self._make_args(), user_config)
        assert config.exposure.density == 1.5
        assert config.exposure.grade == 3.0

    def test_cli_flag_overrides_user_config(self):
        """CLI flag should override user config value."""
        user_config = {"cli": {}, "processing": {"density": 1.5}}
        config = build_config(self._make_args(density=3.0), user_config)
        assert config.exposure.density == 3.0

    def test_preset_overrides_user_config(self, tmp_path, monkeypatch):
        """Preset should override user config processing values."""
        presets_dir = tmp_path / "presets"
        presets_dir.mkdir()
        (presets_dir / "test-preset.json").write_text(json.dumps({"density": 2.0, "grade": 4.0}))
        monkeypatch.setattr("negpy.cli.batch.PRESETS_DIR", str(presets_dir))

        user_config = {"cli": {}, "processing": {"density": 1.5, "grade": 1.0}}
        config = build_config(self._make_args(preset="test-preset"), user_config)
        assert config.exposure.density == 2.0
        assert config.exposure.grade == 4.0

    def test_cli_flag_overrides_preset(self, tmp_path, monkeypatch):
        """CLI flag should win over preset value."""
        presets_dir = tmp_path / "presets"
        presets_dir.mkdir()
        (presets_dir / "test-preset.json").write_text(json.dumps({"density": 2.0}))
        monkeypatch.setattr("negpy.cli.batch.PRESETS_DIR", str(presets_dir))

        user_config = {"cli": {}, "processing": {}}
        config = build_config(self._make_args(preset="test-preset", density=5.0), user_config)
        assert config.exposure.density == 5.0

    def test_settings_overrides_preset(self, tmp_path, monkeypatch):
        """--settings file should override preset values."""
        presets_dir = tmp_path / "presets"
        presets_dir.mkdir()
        (presets_dir / "test-preset.json").write_text(json.dumps({"density": 2.0}))
        monkeypatch.setattr("negpy.cli.batch.PRESETS_DIR", str(presets_dir))

        settings_file = tmp_path / "settings.json"
        settings_file.write_text(json.dumps({"density": 9.0, "grade": 8.0}))

        user_config = {"cli": {}, "processing": {}}
        config = build_config(
            self._make_args(preset="test-preset", settings=str(settings_file)),
            user_config,
        )
        assert config.exposure.density == 9.0

    def test_missing_preset_raises(self, tmp_path, monkeypatch):
        """Unknown preset name should raise FileNotFoundError."""
        presets_dir = tmp_path / "presets"
        presets_dir.mkdir()
        monkeypatch.setattr("negpy.cli.batch.PRESETS_DIR", str(presets_dir))

        user_config = {"cli": {}, "processing": {}}
        with pytest.raises(FileNotFoundError):
            build_config(self._make_args(preset="nonexistent"), user_config)
