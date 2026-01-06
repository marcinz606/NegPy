import os
from typing import List, Set


class FolderWatchService:
    """
    Service for scanning directories to discover new RAW assets.
    """

    SUPPORTED_EXTS = {".dng", ".tiff", ".tif", ".nef", ".arw", ".raw", ".raf"}

    @classmethod
    def scan_for_new_files(
        cls, folder_path: str, existing_paths: Set[str]
    ) -> List[str]:
        """
        Scans a folder and returns absolute paths of supported files not in existing_paths.
        """
        if not os.path.exists(folder_path):
            return []

        new_files = []
        for root, _, files in os.walk(folder_path):
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in cls.SUPPORTED_EXTS:
                    full_path = os.path.abspath(os.path.join(root, file))
                    if full_path not in existing_paths:
                        new_files.append(full_path)

        return new_files
