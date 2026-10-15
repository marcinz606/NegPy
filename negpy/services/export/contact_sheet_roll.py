"""What a contact sheet reads from each frame. No Qt: the header reads run on a worker thread."""

import math
import os
import re
import statistics
from collections import Counter
from dataclasses import dataclass, replace
from typing import Any, Mapping, Optional, Sequence

from negpy.domain.models import ExportResolutionMode, WorkspaceConfig
from negpy.features.exposure.models import EXPOSURE_CONSTANTS
from negpy.features.geometry.models import AspectRatio, AutocropMode
from negpy.features.metadata.gear_models import FilmColorType, FilmStock
from negpy.features.process.models import ProcessMode
from negpy.services.export.contact_sheet_edge import EdgeFamily, EdgeStyle, ascii_upper, edge_family
from negpy.services.export.contact_sheet_layout import (
    DEFAULT_FRAME_120,
    FRAME_SIZES_120,
    FilmGeometry,
    SheetFormat,
    frame_turns,
)


@dataclass(frozen=True)
class FrameFacts:
    """`scan_size` is (width, height) after EXIF orientation and the half-frame split."""

    capture_time: str = ""
    birth_time: float = 0.0
    scan_size: Optional[tuple[int, int]] = None
    # log2(shutter × ISO / f-number²) of a camera raw scan.
    scan_ev: Optional[float] = None


@dataclass(frozen=True)
class SheetFrame:
    asset: dict
    config: WorkspaceConfig
    facts: FrameFacts = FrameFacts()

    @property
    def name(self) -> str:
        return str(self.asset.get("name") or os.path.basename(str(self.asset.get("path", ""))))

    @property
    def half(self) -> int:
        return int(self.asset.get("half") or 0)


def asset_paths(asset: dict) -> list[str]:
    parts = asset.get("hdr_paths") or asset.get("stitch_paths")
    if parts:
        return [str(p) for p in parts]
    return [str(asset["path"])]


def read_capture_time(path: str) -> str:
    """EXIF capture time as "YYYY-MM-DD HH:MM:SS", or "". piexif reads a whole TIFF or raw, so only JPEG uses it."""
    from negpy.features.metadata.exif_read import format_exif_datetime

    try:
        with open(path, "rb") as fh:
            magic = fh.read(4)
    except OSError:
        return ""
    try:
        if magic[:2] == b"\xff\xd8":
            import piexif

            exif = piexif.load(path)
            value = exif.get("Exif", {}).get(piexif.ExifIFD.DateTimeOriginal) or exif.get("0th", {}).get(piexif.ImageIFD.DateTime)
            return format_exif_datetime(value) if value else ""
        if magic in (b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+"):
            import tifffile

            with tifffile.TiffFile(path) as tif:
                tags = tif.pages[0].tags
                exif_tag = tags.get("ExifTag")
                value = exif_tag.value.get("DateTimeOriginal") if exif_tag is not None and isinstance(exif_tag.value, dict) else None
                if not value:
                    datetime_tag = tags.get("DateTime")
                    value = datetime_tag.value if datetime_tag is not None else None
                return format_exif_datetime(value) if value else ""
    except Exception:
        return ""
    return ""


def read_birth_time(path: str) -> float:
    try:
        st = os.stat(path)
    except OSError:
        return 0.0
    return float(getattr(st, "st_birthtime", 0.0) or st.st_mtime)


def read_scan_size(asset: dict) -> Optional[tuple[int, int]]:
    from negpy.infrastructure.loaders.constants import (
        SUPPORTED_JPEG_EXTENSIONS,
        SUPPORTED_JXL_EXTENSIONS,
        SUPPORTED_RAW_EXTENSIONS,
        SUPPORTED_TIFF_EXTENSIONS,
    )
    from negpy.infrastructure.loaders.helpers import read_orientation
    from negpy.services.assets.half_frame import slice_half_dimensions

    if asset.get("stitch_paths"):
        return None
    path = str(asset["path"])
    ext = os.path.splitext(path)[1].lower()
    camera_raw = SUPPORTED_RAW_EXTENSIONS - SUPPORTED_TIFF_EXTENSIONS - SUPPORTED_JPEG_EXTENSIONS - SUPPORTED_JXL_EXTENSIONS
    size: Optional[tuple[int, int]] = None
    try:
        if ext in SUPPORTED_JXL_EXTENSIONS:
            return None
        if ext in camera_raw:
            import rawpy

            with rawpy.imread(path) as raw:
                sizes = raw.sizes
                size = (int(sizes.width), int(sizes.height))
                if sizes.flip in (5, 6):
                    size = (size[1], size[0])
        else:
            if ext in SUPPORTED_TIFF_EXTENSIONS:
                import tifffile

                with tifffile.TiffFile(path) as tif:
                    page = tif.pages[0]
                    size = (int(page.imagewidth), int(page.imagelength))
            else:
                from PIL import Image

                with Image.open(path) as img:
                    size = (int(img.size[0]), int(img.size[1]))
            if read_orientation(path) in (5, 6, 7, 8):
                size = (size[1], size[0])
    except Exception:
        return None
    if size is None:
        return None
    half = int(asset.get("half") or 0)
    crop = asset.get("crop_rect")
    if half or crop:
        height, width = slice_half_dimensions(
            (size[1], size[0]),
            half,
            float(asset.get("split_x") or 0.5),
            tuple(crop) if crop else None,
            float(asset.get("gutter_thickness") or 0.0),
        )
        size = (width, height)
    return size


def read_scan_ev(path: str) -> Optional[float]:
    """Camera raw only: its decode is linear, so an exposure difference is a pure scale on the data."""
    from negpy.infrastructure.loaders.constants import (
        SUPPORTED_JPEG_EXTENSIONS,
        SUPPORTED_JXL_EXTENSIONS,
        SUPPORTED_RAW_EXTENSIONS,
        SUPPORTED_TIFF_EXTENSIONS,
    )

    camera_raw = SUPPORTED_RAW_EXTENSIONS - SUPPORTED_TIFF_EXTENSIONS - SUPPORTED_JPEG_EXTENSIONS - SUPPORTED_JXL_EXTENSIONS
    if os.path.splitext(path)[1].lower() not in camera_raw:
        return None
    try:
        import rawpy

        with rawpy.imread(path) as raw:
            other = raw.other
            shutter, iso, aperture = float(other.shutter_speed), float(other.iso_speed), float(other.aperture)
    except Exception:
        return None
    if shutter <= 0 or iso <= 0 or aperture <= 0:
        return None
    return math.log2(shutter * iso / (aperture * aperture))


def read_frame_facts(asset: dict) -> FrameFacts:
    paths = asset_paths(asset)
    times = [t for t in (read_capture_time(p) for p in paths) if t]
    births = [b for b in (read_birth_time(p) for p in paths) if b]
    return FrameFacts(
        capture_time=min(times) if len(times) == len(paths) else "",
        birth_time=min(births) if births else 0.0,
        scan_size=read_scan_size(asset),
        scan_ev=read_scan_ev(paths[0]) if len(paths) == 1 else None,
    )


def creation_order(frames: Sequence[SheetFrame]) -> list[SheetFrame]:
    """EXIF capture time when every frame has one, else file creation time: mixing them interleaves two clocks."""
    by_exif = bool(frames) and all(f.facts.capture_time for f in frames)

    def key(frame: SheetFrame) -> tuple[Any, ...]:
        when: Any = frame.facts.capture_time if by_exif else frame.facts.birth_time
        return (when, frame.name.lower(), frame.half)

    return sorted(frames, key=key)


def _crop_fractions(config: WorkspaceConfig) -> tuple[float, float]:
    rect = config.geometry.crop_rect
    if not rect:
        return 1.0, 1.0
    x1, y1, x2, y2 = rect
    return max(1e-3, x2 - x1), max(1e-3, y2 - y1)


def upright_aspect(frame: SheetFrame, tile_shape: Optional[tuple[int, int]] = None) -> Optional[float]:
    """Width over height of the whole picture after the user's rotation, uncropped."""
    size = frame.facts.scan_size
    if size and size[0] > 0 and size[1] > 0:
        w, h = size
        if frame.config.geometry.rotation % 2:
            w, h = h, w
        return w / h
    if tile_shape and tile_shape[0] > 0 and tile_shape[1] > 0:
        fx, fy = _crop_fractions(frame.config)
        return (tile_shape[1] / tile_shape[0]) * (fy / fx)
    return None


def cropped_aspect(frame: SheetFrame) -> Optional[float]:
    whole = upright_aspect(frame)
    if whole is None:
        return None
    fx, fy = _crop_fractions(frame.config)
    aspect = whole * fx / fy
    return max(aspect, 1 / aspect)


def turns_for(frame: SheetFrame, geometry: FilmGeometry, tile_shape: Optional[tuple[int, int]] = None) -> int:
    geo = frame.config.geometry
    return frame_turns(upright_aspect(frame, tile_shape), geometry, geo.rotation, geo.flip_horizontal, geo.flip_vertical)


def _effective_format(frame: SheetFrame) -> str:
    meta = frame.config.metadata
    return meta.format_other if meta.format == "Other" else meta.format


_FRAME_120_PATTERN = re.compile(r"6\s*[x×]\s*(4[.,]5|6|7|8|9|12|17)|645")


def parse_frame_120(text: str) -> str:
    match = _FRAME_120_PATTERN.search(text.lower())
    if not match:
        return ""
    if match.group(0) == "645":
        return "6×4.5"
    side = match.group(1).replace(",", ".")
    size = f"6×{side}"
    return size if size in FRAME_SIZES_120 else ""


def _nearest_frame_120(aspect: float) -> str:
    candidates = {size: FRAME_SIZES_120[size] / 56.0 for size in ("6×4.5", "6×6", "6×7", "6×9", "6×12", "6×17")}
    candidates = {size: max(a, 1 / a) for size, a in candidates.items()}
    return min(candidates, key=lambda size: abs(math.log(candidates[size]) - math.log(aspect)))


def infer_format(frames: Sequence[SheetFrame], half_frame_roll: bool = False) -> tuple[SheetFormat, str]:
    """Without format metadata a 6×9 roll reads as 35mm: both are 3:2."""
    if half_frame_roll or any(f.half for f in frames):
        return SheetFormat.HALF_FRAME, DEFAULT_FRAME_120

    votes: Counter[str] = Counter()
    named_sizes: Counter[str] = Counter()
    for frame in frames:
        fmt = _effective_format(frame).strip()
        size = parse_frame_120(fmt)
        if fmt == "35mm":
            votes["35mm"] += 1
        elif fmt == "120" or size:
            votes["120"] += 1
            if size:
                named_sizes[size] += 1

    aspects = [a for a in (cropped_aspect(f) for f in frames) if a]
    median = statistics.median(aspects) if aspects else None

    if votes:
        fmt = SheetFormat.FULL_FRAME if votes.most_common(1)[0][0] == "35mm" else SheetFormat.MEDIUM
    else:
        fmt = SheetFormat.MEDIUM if median is not None and median <= 1.3 else SheetFormat.FULL_FRAME

    if fmt != SheetFormat.MEDIUM:
        return fmt, DEFAULT_FRAME_120
    if named_sizes:
        return fmt, named_sizes.most_common(1)[0][0]
    return fmt, _nearest_frame_120(median) if median else DEFAULT_FRAME_120


def _gear_stock(frame: SheetFrame, library: Any) -> Optional[FilmStock]:
    stock_id = frame.config.metadata.film_stock_id
    if not stock_id or library is None:
        return None
    try:
        return library.get_film_stock(stock_id)
    except Exception:
        return None


def film_name(frame: SheetFrame, library: Any = None) -> str:
    stock = _gear_stock(frame, library)
    if stock is not None:
        if stock.display_name.strip():
            return stock.display_name.strip()
        maker, name = stock.manufacturer.strip(), stock.stock_name.strip()
        if maker and not name.lower().startswith(maker.lower()):
            return f"{maker} {name}".strip()
        if name:
            return name
    return frame.config.metadata.film.strip()


def film_color_type(frame: SheetFrame, library: Any = None) -> Optional[FilmColorType]:
    """A Positive frame's process mode says nothing about the film it came from."""
    stated = frame.config.metadata.film_color_type
    if stated:
        return FilmColorType.from_storage(stated)
    stock = _gear_stock(frame, library)
    if stock is not None:
        return stock.color_type
    process = frame.config.process
    if process.process_mode == ProcessMode.C41:
        return FilmColorType.COLOR_NEGATIVE
    if process.process_mode == ProcessMode.BW:
        return FilmColorType.BW_NEGATIVE
    if process.process_mode == ProcessMode.E6 and not process.positive_source:
        return FilmColorType.COLOR_SLIDE
    return None


def _majority(values: Sequence[Any], default: Any) -> Any:
    present = [v for v in values if v not in (None, "")]
    return Counter(present).most_common(1)[0][0] if present else default


@dataclass(frozen=True)
class SheetLook:
    palette: str = "bw"
    black: int = 0
    edge: EdgeStyle = EdgeStyle()
    label: str = ""
    white_paper: bool = False
    film_base: bool = True


def sheet_black(paper_black: bool) -> int:
    if not paper_black:
        return 0
    linear = 10.0 ** -float(EXPOSURE_CONSTANTS["d_max"])
    encoded = 12.92 * linear if linear <= 0.0031308 else 1.055 * linear ** (1 / 2.4) - 0.055
    return int(round(encoded * 255))


def sheet_look(frames: Sequence[SheetFrame], fmt: SheetFormat, label: str = "", library: Any = None) -> SheetLook:
    names = [film_name(f, library) for f in frames]
    stock_name = _majority(names, "")
    manufacturer = _majority([f.config.metadata.film_manufacturer for f in frames], "")
    color_type = _majority([film_color_type(f, library) for f in frames], None)
    if color_type == FilmColorType.COLOR_NEGATIVE:
        palette = "color"
    elif color_type in (FilmColorType.COLOR_SLIDE, FilmColorType.BW_SLIDE):
        palette = "slide"
    else:
        palette = "bw"
    family = edge_family(stock_name, manufacturer)
    dx = (
        fmt != SheetFormat.MEDIUM
        and palette != "slide"
        and family != EdgeFamily.CINE
        and (palette == "color" or family == EdgeFamily.ILFORD)
    )
    paper_black = _majority([f.config.exposure.paper_black for f in frames], False)
    return SheetLook(palette, sheet_black(bool(paper_black)), EdgeStyle(family, ascii_upper(stock_name), dx), label)


def roll_label_text(roll_name: str, frames: Sequence[SheetFrame], library: Any = None) -> str:
    if not frames:
        return roll_name
    metas = [f.config.metadata for f in frames]
    parts: list[str] = []
    if roll_name.strip():
        parts.append(roll_name.strip())

    stock = _majority([film_name(f, library) for f in frames], "")
    if stock:
        push = _majority([m.push_pull for m in metas if m.push_pull], 0)
        iso = _majority([m.film_iso for m in metas], None)
        if push and iso:
            stock = f"{stock} @ EI {int(iso * 2**push)}"
        parts.append(stock)

    developer = _majority([" ".join(p for p in (m.developer.strip(), m.process_dilution.strip()) if p) for m in metas], "")
    if developer:
        parts.append(developer)

    def camera(m: Any) -> str:
        make, model = m.camera_make.strip(), m.camera_model.strip()
        return model if make and model.lower().startswith(make.lower()) else " ".join(p for p in (make, model) if p)

    cam = _majority([camera(m) for m in metas], "")
    if cam:
        parts.append(cam)

    date = _majority([m.capture_date.strip()[:10] for m in metas], "")
    if not date:
        times = sorted(f.facts.capture_time for f in frames if f.facts.capture_time)
        date = times[0][:10] if times else ""
    if date:
        parts.append(date)
    return " · ".join(parts)


def tile_params(config: WorkspaceConfig) -> WorkspaceConfig:
    """The sheet draws its own rebate, so a Film edge auto crop re-detects as Image."""
    geometry = config.geometry
    if geometry.crop_from_auto and geometry.autocrop_mode == AutocropMode.FILM:
        geometry = replace(geometry, autocrop_mode=AutocropMode.IMAGE)
    return replace(
        config,
        geometry=geometry,
        finish=replace(config.finish, border_size=0.0, carrier_width=0.0),
        export=replace(
            config.export,
            export_resolution_mode=ExportResolutionMode.ORIGINAL,
            paper_aspect_ratio=AspectRatio.ORIGINAL,
        ),
    )


# ISO R of grade 2 paper.
PROOF_GRADE = 110.0
# Stops: shutter speeds are nominal to about this much.
_SAME_EXPOSURE_EV = 1 / 6


@dataclass(frozen=True)
class StraightProof:
    """`reason` says why the proof cannot be made; `note` what it assumes or corrected."""

    frames: tuple[SheetFrame, ...] = ()
    reason: str = ""
    note: str = ""

    @property
    def available(self) -> bool:
        return bool(self.frames) and not self.reason


def scan_exposure_offsets(frames: Sequence[SheetFrame]) -> list[float]:
    """Per-frame log10 shift that brings each scan to the median scan exposure; 0.0 when unknown."""
    evs = [f.facts.scan_ev for f in frames]
    known = [ev for ev in evs if ev is not None]
    if not known:
        return [0.0] * len(frames)
    reference = statistics.median(known)
    return [0.0 if ev is None else (ev - reference) * math.log10(2.0) for ev in evs]


def straight_proof_config(config: WorkspaceConfig, floors: Optional[tuple], ceils: Optional[tuple], shift: float = 0.0) -> WorkspaceConfig:
    """One exposure and grade for the roll. A slide keeps its own transfer window."""
    from negpy.kernel.system.config import DEFAULT_WORKSPACE_CONFIG

    frame_exposure = config.exposure
    exposure = replace(
        DEFAULT_WORKSPACE_CONFIG.exposure,
        auto_exposure=False,
        auto_normalize_contrast=False,
        cast_removal_strength=0.0,
        grade=PROOF_GRADE,
        paper_profile=frame_exposure.paper_profile,
        paper_black=frame_exposure.paper_black,
        paper_dmin=frame_exposure.paper_dmin,
        render_intent=frame_exposure.render_intent,
    )
    process = config.process
    if floors is not None and ceils is not None and process.process_mode != ProcessMode.E6:
        process = replace(
            process,
            use_luma_average=True,
            use_color_average=True,
            locked_floors=tuple(float(v) + shift for v in floors),
            locked_ceils=tuple(float(v) + shift for v in ceils),
            use_cast_average=False,
            locked_neutral_axis=None,
        )
    defaults = DEFAULT_WORKSPACE_CONFIG
    return replace(
        config,
        process=process,
        exposure=exposure,
        lab=defaults.lab,
        local=defaults.local,
        toning=defaults.toning,
        altproc=defaults.altproc,
        finish=defaults.finish,
    )


def scene_of(frame: SheetFrame) -> Optional[str]:
    scene = frame.asset.get("scene")
    return str(scene[1]) if scene else None


def scene_order(frames: Sequence[SheetFrame]) -> list[int]:
    def rank(index: int) -> tuple[int, int]:
        scene = frames[index].asset.get("scene")
        return (1, 0) if not scene else (0, int(scene[0]))

    return sorted(range(len(frames)), key=rank)


def scene_breaks(frames: Sequence[SheetFrame]) -> list[int]:
    return [i for i in range(1, len(frames)) if scene_of(frames[i]) != scene_of(frames[i - 1])]


def straight_proof(
    frames: Sequence[SheetFrame],
    baseline: Optional[dict],
    scene_baselines: Optional[Mapping[str, dict]] = None,
) -> StraightProof:
    """With `scene_baselines`, each scene prints at its own baseline; the roll's covers a scene with none."""
    if not frames:
        return StraightProof(reason="No frames to proof.")
    positives = sum(1 for f in frames if f.config.process.positive_source)
    if positives:
        return StraightProof(
            reason=f"{positives} of the frames are positives made by the scanning software. A straight proof needs the negatives."
        )

    by_scene = scene_baselines is not None
    groups: list[Optional[str]] = []
    unmetered: set[str] = set()
    for frame in frames:
        scene = scene_of(frame) if by_scene else None
        if scene is not None and not (scene_baselines or {}).get(scene):
            unmetered.add(scene)
            scene = None
        groups.append(scene)

    def baseline_for(group: Optional[str]) -> Optional[dict]:
        return (scene_baselines or {})[group] if group is not None else baseline

    for frame, group in zip(frames, groups):
        if frame.config.process.process_mode != ProcessMode.E6 and baseline_for(group) is None:
            what = "Roll Analysis, or Scene Analysis on each scene," if by_scene else "Roll Analysis"
            return StraightProof(reason=f"Run {what} first: the proof prints every negative at one metered exposure.")

    shifts = [0.0] * len(frames)
    for group in set(groups):
        members = [i for i, g in enumerate(groups) if g == group]
        for i, shift in zip(members, scan_exposure_offsets([frames[i] for i in members])):
            shifts[i] = shift

    proofed = []
    for frame, group, shift in zip(frames, groups, shifts):
        saved = baseline_for(group)
        floors = tuple(saved["floors"]) if saved else None
        ceils = tuple(saved["ceils"]) if saved else None
        proofed.append(replace(frame, config=straight_proof_config(frame.config, floors, ceils, shift)))

    evs = [f.facts.scan_ev for f in frames]
    known = [ev for ev in evs if ev is not None]
    if not known:
        note = "Assumes every frame was scanned at one exposure (an exposure lock, or auto exposure off)."
    else:
        spread = max(known) - min(known)
        missing = len(evs) - len(known)
        if spread < _SAME_EXPOSURE_EV:
            note = "Every raw was exposed alike."
        else:
            note = f"Scan exposures differ by {spread:.1f} stops; each raw's EXIF evens them out."
        if missing:
            note += f" {missing} frames state no exposure and print as scanned."
    if by_scene:
        note += " Each scene prints at its own metering."
        if unmetered:
            note += f" {len(unmetered)} scenes have none of their own and print at the roll's."
    if any(f.config.process.process_mode == ProcessMode.E6 for f in frames):
        note += " Slides print through their own window, as scanned."
    return StraightProof(tuple(proofed), "", note)
