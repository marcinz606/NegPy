"""Tests for negpy.cli.batch — CLI batch converter."""

import os

os.environ.setdefault("NUMBA_THREADING_LAYER", "workqueue")

import pytest

import argparse
import json

from unittest.mock import patch, MagicMock

from negpy.cli.batch import build_parser, discover_files, build_config, main
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
        assert args.inputs == ["file1.dng", "file2.tiff"]

    def test_multiple_inputs(self):
        parser = build_parser()
        args = parser.parse_args(["a.dng", "b.tiff", "/scans/"])
        assert len(args.inputs) == 3

    def test_no_inputs_raises(self):
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])

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
            "inputs": ["file.dng"],
        }
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    def test_defaults_preserve_workspace_config(self):
        args = self._make_args()
        config = build_config(args)
        assert config.process.process_mode == ProcessMode.C41
        assert config.exposure.density == DEFAULT_WORKSPACE_CONFIG.exposure.density
        assert config.exposure.grade == DEFAULT_WORKSPACE_CONFIG.exposure.grade
        assert config.lab.sharpen == DEFAULT_WORKSPACE_CONFIG.lab.sharpen

    def test_mode_bw(self):
        config = build_config(self._make_args(mode="bw"))
        assert config.process.process_mode == ProcessMode.BW

    def test_mode_e6(self):
        config = build_config(self._make_args(mode="e6"))
        assert config.process.process_mode == ProcessMode.E6

    def test_exposure_overrides(self):
        config = build_config(self._make_args(density=2.0, grade=4.0))
        assert config.exposure.density == 2.0
        assert config.exposure.grade == 4.0

    def test_sharpen_override(self):
        config = build_config(self._make_args(sharpen=0.8))
        assert config.lab.sharpen == 0.8

    def test_export_format_jpeg(self):
        config = build_config(self._make_args(output_format="jpeg"))
        assert config.export.export_fmt == ExportFormat.JPEG

    def test_export_format_tiff_default(self):
        config = build_config(self._make_args())
        assert config.export.export_fmt == ExportFormat.TIFF

    def test_color_space_prophoto(self):
        config = build_config(self._make_args(color_space="prophoto"))
        assert config.export.export_color_space == "ProPhoto RGB"

    def test_dpi_and_print_size(self):
        config = build_config(self._make_args(dpi=600, print_size=40.0))
        assert config.export.export_dpi == 600
        assert config.export.export_print_size == 40.0

    def test_original_res(self):
        config = build_config(self._make_args(original_res=True))
        assert config.export.use_original_res is True

    def test_filename_pattern(self):
        config = build_config(self._make_args(filename_pattern="{{ date }}_{{ original_name }}"))
        assert config.export.filename_pattern == "{{ date }}_{{ original_name }}"

    def test_none_overrides_preserve_base(self):
        config = build_config(self._make_args())
        assert config.exposure.density == DEFAULT_WORKSPACE_CONFIG.exposure.density
        assert config.lab.sharpen == DEFAULT_WORKSPACE_CONFIG.lab.sharpen
        assert config.export.export_dpi == DEFAULT_WORKSPACE_CONFIG.export.export_dpi

    def test_settings_file_load(self, tmp_path):
        settings = {"process_mode": "B&W", "density": 0.8, "grade": 1.5}
        settings_file = tmp_path / "settings.json"
        settings_file.write_text(json.dumps(settings))
        args = self._make_args(settings=str(settings_file))
        config = build_config(args)
        # --mode c41 still overrides the JSON process_mode
        assert config.process.process_mode == ProcessMode.C41
        # JSON values used as base
        assert config.exposure.density == 0.8

    def test_settings_file_with_cli_override(self, tmp_path):
        settings = {"density": 0.8, "grade": 1.5}
        settings_file = tmp_path / "settings.json"
        settings_file.write_text(json.dumps(settings))
        args = self._make_args(settings=str(settings_file), density=2.5)
        config = build_config(args)
        assert config.exposure.density == 2.5
        assert config.exposure.grade == 1.5

    def test_missing_settings_file_raises(self):
        args = self._make_args(settings="/nonexistent/file.json")
        with pytest.raises(FileNotFoundError):
            build_config(args)

    def test_invalid_json_raises(self, tmp_path):
        bad_file = tmp_path / "bad.json"
        bad_file.write_text("not json {{{")
        args = self._make_args(settings=str(bad_file))
        with pytest.raises(json.JSONDecodeError):
            build_config(args)


class TestMain:
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
