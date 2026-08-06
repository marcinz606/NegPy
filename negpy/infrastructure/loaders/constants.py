import os
from typing import Set

SUPPORTED_TIFF_EXTENSIONS: Set[str] = {
    ".tif",
    ".tiff",
}

SUPPORTED_JPEG_EXTENSIONS: Set[str] = {
    ".jpg",
    ".jpeg",
}

SUPPORTED_JXL_EXTENSIONS: Set[str] = {
    ".jxl",
}

SUPPORTED_RAW_EXTENSIONS: Set[str] = (
    {
        ".3fr",
        ".ari",
        ".arw",
        ".bay",
        ".braw",
        ".crw",
        ".cr2",
        ".cr3",
        ".cap",
        ".data",
        ".dcs",
        ".dcr",
        ".dng",
        ".drf",
        ".eip",
        ".erf",
        ".fff",
        ".gpr",
        ".iiq",
        ".k25",
        ".kdc",
        ".mdc",
        ".mef",
        ".mos",
        ".mrw",
        ".nef",
        ".nrw",
        ".obm",
        ".orf",
        ".pef",
        ".ptx",
        ".pxn",
        ".r3d",
        ".raf",
        ".raw",
        ".rwl",
        ".rw2",
        ".rwz",
        ".sr2",
        ".srf",
        ".srw",
        ".x3f",
    }
    | SUPPORTED_TIFF_EXTENSIONS
    | SUPPORTED_JPEG_EXTENSIONS
    | SUPPORTED_JXL_EXTENSIONS
)

# Longest first: "_ir_valid" must win before "_ir" is tested against the same stem.
IR_SIDECAR_SUFFIXES: tuple[str, ...] = ("_ir_valid", "_ir")


def is_ir_sidecar_path(path: str) -> bool:
    """True for an `_IR`/`_IR_VALID` TIFF whose main TIFF sits next to it. Asset discovery hides
    these; TiffLoader reads them off the main file instead. Case-insensitive name compare (not
    os.path.exists) so a `Frame1.TIF` / `frame1_ir.tif` mix matches on case-sensitive filesystems.
    An orphan sidecar stays visible — nothing else would ever open it."""
    base, ext = os.path.splitext(path)
    if ext.lower() not in SUPPORTED_TIFF_EXTENSIONS:
        return False
    stem = os.path.basename(base).lower()
    for suffix in IR_SIDECAR_SUFFIXES:
        if not stem.endswith(suffix):
            continue
        main = stem[: -len(suffix)]
        if not main:
            return False
        wanted = tuple(main + e for e in sorted(SUPPORTED_TIFF_EXTENSIONS))
        try:
            entries = os.listdir(os.path.dirname(path) or ".")
        except OSError:
            return False
        return any(name.lower() in wanted for name in entries)
    return False
