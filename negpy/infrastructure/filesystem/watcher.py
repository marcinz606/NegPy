import os
from typing import List, Set
from negpy.infrastructure.loaders.constants import SUPPORTED_RAW_EXTENSIONS, is_ir_sidecar_path


class FolderWatchService:
    """
    Scans for new RAW/TIFF assets.
    """

    SUPPORTED_EXTS = SUPPORTED_RAW_EXTENSIONS

    @classmethod
    def scan_for_new_files(cls, folder_path: str, existing_paths: Set[str]) -> List[str]:
        """
        Shallow scan for unindexed files.
        """
        if not os.path.exists(folder_path):
            return []

        new_files = []
        try:
            with os.scandir(folder_path) as it:
                for entry in it:
                    if entry.is_file():
                        ext = os.path.splitext(entry.name)[1].lower()
                        if ext in cls.SUPPORTED_EXTS and not is_ir_sidecar_path(entry.path):
                            full_path = os.path.abspath(entry.path)
                            if full_path not in existing_paths:
                                new_files.append(full_path)
        except Exception:
            pass

        return new_files
