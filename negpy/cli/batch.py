"""NegPy CLI batch converter.

Converts film negatives to positives without requiring a GUI.
"""

import os
import sys

os.environ.setdefault("NUMBA_THREADING_LAYER", "workqueue")

import argparse
import dataclasses
import json
import time
from typing import List, Optional

from negpy.domain.models import WorkspaceConfig, ExportFormat, ColorSpace
from negpy.features.process.models import ProcessMode
from negpy.infrastructure.loaders.constants import SUPPORTED_RAW_EXTENSIONS
from negpy.kernel.image.logic import calculate_file_hash
from negpy.kernel.system.config import DEFAULT_WORKSPACE_CONFIG, APP_CONFIG
from negpy.services.export.templating import render_export_filename
from negpy.services.rendering.image_processor import ImageProcessor


MODE_MAP = {
    "c41": ProcessMode.C41,
    "bw": ProcessMode.BW,
    "e6": ProcessMode.E6,
}

FORMAT_MAP = {
    "jpeg": ExportFormat.JPEG,
    "tiff": ExportFormat.TIFF,
}

COLOR_SPACE_MAP = {
    "srgb": ColorSpace.SRGB.value,
    "adobe-rgb": ColorSpace.ADOBE_RGB.value,
    "prophoto": ColorSpace.PROPHOTO.value,
    "wide-gamut": ColorSpace.WIDE.value,
    "aces": ColorSpace.ACES.value,
    "p3": ColorSpace.P3_D65.value,
    "rec2020": ColorSpace.REC2020.value,
    "greyscale": ColorSpace.GREYSCALE.value,
}

MODE_CHOICES = tuple(MODE_MAP.keys())
FORMAT_CHOICES = tuple(FORMAT_MAP.keys())
COLOR_SPACE_CHOICES = tuple(COLOR_SPACE_MAP.keys())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="negpy",
        description="NegPy -- Film negative batch converter",
        epilog="Example: negpy --mode c41 --format tiff --output ./export /path/to/scans/",
    )

    parser.add_argument(
        "inputs",
        nargs="+",
        metavar="FILE_OR_DIR",
        help="Input files or directories containing film negatives",
    )

    parser.add_argument(
        "--mode",
        choices=MODE_CHOICES,
        default="c41",
        help="Film type: c41 (color negative), bw (black & white), e6 (slide) (default: c41)",
    )

    parser.add_argument(
        "--format",
        choices=FORMAT_CHOICES,
        default="tiff",
        dest="output_format",
        help="Output file format (default: tiff, 16-bit for maximum quality)",
    )

    parser.add_argument(
        "--output",
        default="./export",
        metavar="DIR",
        help="Output directory (default: ./export)",
    )

    parser.add_argument(
        "--color-space",
        choices=COLOR_SPACE_CHOICES,
        default="adobe-rgb",
        help="Output color space (default: adobe-rgb)",
    )

    parser.add_argument(
        "--density",
        type=float,
        default=None,
        metavar="FLOAT",
        help="Print density / brightness (default: 1.0)",
    )

    parser.add_argument(
        "--grade",
        type=float,
        default=None,
        metavar="FLOAT",
        help="Contrast grade (default: 2.0)",
    )

    parser.add_argument(
        "--sharpen",
        type=float,
        default=None,
        metavar="FLOAT",
        help="Sharpening amount (default: 0.25)",
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=None,
        metavar="INT",
        help="Export DPI (default: 300)",
    )

    parser.add_argument(
        "--print-size",
        type=float,
        default=None,
        metavar="CM",
        help="Print long-edge size in cm (default: 30.0)",
    )

    parser.add_argument(
        "--original-res",
        action="store_true",
        default=False,
        help="Export at original sensor resolution (ignores --dpi and --print-size)",
    )

    parser.add_argument(
        "--filename-pattern",
        default=None,
        metavar="TEMPLATE",
        help='Jinja2 filename template (default: "positive_{{ original_name }}")',
    )

    parser.add_argument(
        "--no-gpu",
        action="store_true",
        default=False,
        help="Disable GPU acceleration, use CPU only",
    )

    parser.add_argument(
        "--settings",
        default=None,
        metavar="JSON_FILE",
        help="Load full WorkspaceConfig from a JSON settings file",
    )

    return parser


def discover_files(inputs: List[str]) -> List[str]:
    """Resolves input paths to a sorted list of supported image files."""
    files = []
    for input_path in inputs:
        path = os.path.abspath(input_path)
        if os.path.isfile(path):
            ext = os.path.splitext(path)[1].lower()
            if ext in SUPPORTED_RAW_EXTENSIONS:
                files.append(path)
            else:
                print(f"Warning: Skipping unsupported file: {path}", file=sys.stderr)
        elif os.path.isdir(path):
            for root, _dirs, filenames in os.walk(path):
                for fname in sorted(filenames):
                    fpath = os.path.join(root, fname)
                    ext = os.path.splitext(fname)[1].lower()
                    if ext in SUPPORTED_RAW_EXTENSIONS:
                        files.append(fpath)
        else:
            print(f"Warning: Path not found: {path}", file=sys.stderr)
    return files


def build_config(args: argparse.Namespace) -> WorkspaceConfig:
    """Builds WorkspaceConfig from base defaults (or --settings JSON) + CLI overrides."""
    if args.settings:
        with open(os.path.abspath(args.settings), "r") as f:
            data = json.load(f)
        config = WorkspaceConfig.from_flat_dict(data)
    else:
        config = DEFAULT_WORKSPACE_CONFIG

    process = dataclasses.replace(config.process, process_mode=MODE_MAP[args.mode])

    exposure_overrides = {}
    if args.density is not None:
        exposure_overrides["density"] = args.density
    if args.grade is not None:
        exposure_overrides["grade"] = args.grade
    exposure = dataclasses.replace(config.exposure, **exposure_overrides) if exposure_overrides else config.exposure

    lab_overrides = {}
    if args.sharpen is not None:
        lab_overrides["sharpen"] = args.sharpen
    lab = dataclasses.replace(config.lab, **lab_overrides) if lab_overrides else config.lab

    export_overrides = {
        "export_path": os.path.abspath(args.output),
        "export_fmt": FORMAT_MAP[args.output_format],
        "export_color_space": COLOR_SPACE_MAP[args.color_space],
    }
    if args.dpi is not None:
        export_overrides["export_dpi"] = args.dpi
    if args.print_size is not None:
        export_overrides["export_print_size"] = args.print_size
    if args.original_res:
        export_overrides["use_original_res"] = True
    if args.filename_pattern is not None:
        export_overrides["filename_pattern"] = args.filename_pattern
    export = dataclasses.replace(config.export, **export_overrides)

    return dataclasses.replace(config, process=process, exposure=exposure, lab=lab, export=export)


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point. Returns 0 on success, 1 on failure."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.no_gpu:
        APP_CONFIG.use_gpu = False

    files = discover_files(args.inputs)
    if not files:
        print("Error: No supported image files found.", file=sys.stderr)
        return 1

    try:
        config = build_config(args)
    except (json.JSONDecodeError, FileNotFoundError, KeyError, TypeError) as e:
        print(f"Error loading settings: {e}", file=sys.stderr)
        return 1

    export_settings = config.export
    use_gpu = not args.no_gpu

    os.makedirs(export_settings.export_path, exist_ok=True)

    processor = ImageProcessor()

    total = len(files)
    failed = 0
    print(f"Processing {total} file(s) -> {export_settings.export_path}", file=sys.stderr)
    t_start = time.monotonic()

    for i, file_path in enumerate(files, 1):
        name = os.path.splitext(os.path.basename(file_path))[0]
        print(f"  [{i}/{total}] {name} ...", file=sys.stderr, end="", flush=True)
        t_file = time.monotonic()

        try:
            source_hash = calculate_file_hash(file_path)

            bits, fmt_or_error = processor.process_export(
                file_path,
                config,
                export_settings,
                source_hash,
                prefer_gpu=use_gpu,
            )

            if bits is None:
                print(f" FAILED ({fmt_or_error})", file=sys.stderr)
                failed += 1
                continue

            ext = "jpg" if export_settings.export_fmt == ExportFormat.JPEG else "tiff"
            filename = render_export_filename(file_path, export_settings)
            out_path = os.path.join(export_settings.export_path, f"{filename}.{ext}")

            with open(out_path, "wb") as f:
                f.write(bits)

            elapsed = time.monotonic() - t_file
            print(f" OK ({elapsed:.1f}s)", file=sys.stderr)

        except Exception as e:
            print(f" ERROR: {e}", file=sys.stderr)
            failed += 1

        processor.cleanup()

    total_time = time.monotonic() - t_start
    succeeded = total - failed
    print(f"Done: {succeeded}/{total} succeeded in {total_time:.1f}s", file=sys.stderr)

    processor.destroy_all()

    return 1 if failed > 0 else 0


def cli_entry() -> None:
    """Console script entry point."""
    sys.exit(main())


if __name__ == "__main__":
    sys.exit(main())
