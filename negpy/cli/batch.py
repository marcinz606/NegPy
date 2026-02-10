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
from negpy.features.flatfield.logic import load_flatfield, load_raw_to_float32, apply_flatfield
from negpy.features.process.models import ProcessMode
from negpy.infrastructure.loaders.constants import SUPPORTED_RAW_EXTENSIONS
from negpy.kernel.image.logic import calculate_file_hash, float_to_uint16, float_to_uint8, float_to_uint_luma
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

CONFIG_DIR = os.path.expanduser("~/.negpy")
CONFIG_FILE = os.path.join(CONFIG_DIR, "config.json")
PRESETS_DIR = os.path.join(CONFIG_DIR, "presets")

SCHEMA_BASE_URL = "https://raw.githubusercontent.com/andreyleonardo/negpy/main/schemas"
CONFIG_SCHEMA_URL = f"{SCHEMA_BASE_URL}/config.schema.json"
PRESET_SCHEMA_URL = f"{SCHEMA_BASE_URL}/preset.schema.json"


def load_user_config() -> dict:
    """Loads ~/.negpy/config.json if it exists. Returns {"cli": {}, "processing": {}}."""
    if not os.path.isfile(CONFIG_FILE):
        return {"cli": {}, "processing": {}}
    with open(CONFIG_FILE, "r") as f:
        data = json.load(f)
    return {
        "cli": data.get("cli", {}),
        "processing": data.get("processing", {}),
    }


def generate_default_config() -> int:
    """Creates ~/.negpy/config.json with documented defaults. Returns 0 on success, 1 if exists."""
    if os.path.isfile(CONFIG_FILE):
        print(f"Config already exists: {CONFIG_FILE}", file=sys.stderr)
        return 1
    os.makedirs(CONFIG_DIR, exist_ok=True)
    os.makedirs(PRESETS_DIR, exist_ok=True)
    default = {
        "$schema": CONFIG_SCHEMA_URL,
        "cli": {
            "flat_field": None,
            "output": "./export",
            "mode": "c41",
            "format": "tiff",
            "color_space": "adobe-rgb",
            "no_gpu": False,
            "crop_offset": None,
            "filename_pattern": "positive_{{ original_name }}",
        },
        "processing": {
            "density": 1.0,
            "grade": 2.0,
            "wb_cyan": 0.0,
            "wb_magenta": 0.0,
            "wb_yellow": 0.0,
            "sharpen": 0.25,
            "color_separation": 1.0,
            "saturation": 1.0,
        },
    }
    with open(CONFIG_FILE, "w") as f:
        json.dump(default, f, indent=4)
    print(f"Config created: {CONFIG_FILE}", file=sys.stderr)
    print(f"Presets directory: {PRESETS_DIR}", file=sys.stderr)
    return 0


def list_available_presets() -> int:
    """Lists preset files from ~/.negpy/presets/ and exits."""
    if not os.path.isdir(PRESETS_DIR):
        print("No presets directory found. Run 'negpy --init-config' to create it.", file=sys.stderr)
        return 0
    presets = sorted(f[:-5] for f in os.listdir(PRESETS_DIR) if f.endswith(".json"))
    if not presets:
        print("No presets found. Place .json files in ~/.negpy/presets/", file=sys.stderr)
    else:
        print("Available presets:", file=sys.stderr)
        for name in presets:
            print(f"  {name}", file=sys.stderr)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="negpy",
        description="NegPy -- Film negative batch converter",
        epilog="Example: negpy --mode c41 --format tiff --output ./export /path/to/scans/",
    )

    parser.add_argument(
        "inputs",
        nargs="*",
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
        "--crop-offset",
        type=int,
        default=None,
        metavar="INT",
        help="Autocrop border offset in pixels, range -5..20 (default: 1)",
    )

    parser.add_argument(
        "--flat-field",
        default=None,
        metavar="FILE",
        help="Path to a flat-field reference frame (blank scan) for vignetting correction",
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

    parser.add_argument(
        "--preset",
        default=None,
        metavar="NAME",
        help="Load a film preset by name (e.g. portra-400)",
    )

    parser.add_argument(
        "--list-presets",
        action="store_true",
        default=False,
        help="List available presets and exit",
    )

    parser.add_argument(
        "--init-config",
        action="store_true",
        default=False,
        help="Generate default config at ~/.negpy/config.json and exit",
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


def build_config(args: argparse.Namespace, user_config: dict) -> WorkspaceConfig:
    """Builds WorkspaceConfig with loading priority:
    DEFAULT → user config → preset → --settings → CLI flags
    """
    # Layer 1: defaults as flat dict
    base_dict = DEFAULT_WORKSPACE_CONFIG.to_dict()

    # Layer 2: user config processing overrides
    processing = user_config.get("processing", {})
    if processing:
        base_dict.update(processing)

    # Layer 3: preset overrides
    if getattr(args, "preset", None):
        preset_path = os.path.join(PRESETS_DIR, f"{args.preset}.json")
        if not os.path.isfile(preset_path):
            raise FileNotFoundError(f"Preset not found: {preset_path}")
        with open(preset_path, "r") as f:
            preset_data = json.load(f)
        base_dict.update(preset_data)

    # Layer 4: --settings file overrides
    if args.settings:
        with open(os.path.abspath(args.settings), "r") as f:
            settings_data = json.load(f)
        base_dict.update(settings_data)

    # Build workspace config from merged flat dict
    config = WorkspaceConfig.from_flat_dict(base_dict)

    # Layer 5: CLI flags always win
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

    geometry_overrides = {}
    if args.crop_offset is not None:
        geometry_overrides["autocrop_offset"] = args.crop_offset
    geometry = dataclasses.replace(config.geometry, **geometry_overrides) if geometry_overrides else config.geometry

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

    return dataclasses.replace(config, process=process, exposure=exposure, geometry=geometry, lab=lab, export=export)


def encode_export(buffer, export_settings) -> bytes:
    """Encodes a float32 buffer to TIFF or JPEG bytes."""
    import io
    import tifffile

    is_tiff = export_settings.export_fmt != ExportFormat.JPEG
    if is_tiff:
        img_int = float_to_uint16(buffer)
        output_buf = io.BytesIO()
        tifffile.imwrite(
            output_buf,
            img_int,
            photometric="rgb" if img_int.ndim == 3 else "minisblack",
            compression="lzw",
        )
        return output_buf.getvalue()
    else:
        from PIL import Image
        img_int = float_to_uint8(buffer)
        pil_img = Image.fromarray(img_int)
        output_buf = io.BytesIO()
        pil_img.save(output_buf, format="JPEG", quality=95)
        return output_buf.getvalue()


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point. Returns 0 on success, 1 on failure."""
    parser = build_parser()
    args = parser.parse_args(argv)

    # Handle early-exit commands
    if args.init_config:
        return generate_default_config()
    if args.list_presets:
        return list_available_presets()

    if args.no_gpu:
        APP_CONFIG.use_gpu = False

    # Load user config
    user_config = load_user_config()

    # Apply CLI defaults from user config (only where arg was not explicitly set)
    cli_defaults = user_config.get("cli", {})
    if args.flat_field is None and cli_defaults.get("flat_field"):
        args.flat_field = cli_defaults["flat_field"]
    if args.output == "./export" and "output" in cli_defaults:
        args.output = cli_defaults["output"]
    if args.crop_offset is None and cli_defaults.get("crop_offset") is not None:
        args.crop_offset = cli_defaults["crop_offset"]

    files = discover_files(args.inputs)
    if not files:
        print("Error: No supported image files found.", file=sys.stderr)
        return 1

    try:
        config = build_config(args, user_config)
    except (json.JSONDecodeError, FileNotFoundError, KeyError, TypeError) as e:
        print(f"Error loading settings: {e}", file=sys.stderr)
        return 1

    export_settings = config.export
    use_gpu = not args.no_gpu

    os.makedirs(export_settings.export_path, exist_ok=True)

    # Load flat-field reference if provided
    flatfield_map = None
    if args.flat_field:
        ff_path = os.path.abspath(args.flat_field)
        if not os.path.isfile(ff_path):
            print(f"Error: Flat-field file not found: {ff_path}", file=sys.stderr)
            return 1
        try:
            flatfield_map = load_flatfield(ff_path)
            print(f"Flat-field loaded: {ff_path}", file=sys.stderr)
        except Exception as e:
            print(f"Error loading flat-field: {e}", file=sys.stderr)
            return 1

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

            if flatfield_map is not None:
                # Flat-field path: load raw, correct, then run pipeline
                f32_buffer = load_raw_to_float32(file_path)
                f32_corrected = apply_flatfield(f32_buffer, flatfield_map)
                result_buffer, _metrics = processor.run_pipeline(
                    f32_corrected,
                    config,
                    source_hash,
                    render_size_ref=float(APP_CONFIG.preview_render_size),
                    prefer_gpu=use_gpu,
                )
                import numpy as np
                if not isinstance(result_buffer, np.ndarray):
                    result_buffer = processor.engine_gpu.readback(result_buffer)
                bits = encode_export(result_buffer, export_settings)
            else:
                # Standard path
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
