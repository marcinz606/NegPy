from typing import Set

SUPPORTED_TIFF_EXTENSIONS: Set[str] = {
    ".tif",
    ".tiff",
}

SUPPORTED_JPEG_EXTENSIONS: Set[str] = {
    ".jpg",
    ".jpeg",
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
)
