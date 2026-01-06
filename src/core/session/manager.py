from typing import List, Dict, Optional, Any, Set
from src.domain_objects import ImageSettings
from src.core.persistence.interfaces import IRepository, IAssetStore
from src.orchestration.engine import DarkroomEngine
from src.config import DEFAULT_SETTINGS


class WorkspaceSession:
    """
    Pure domain workspace session.
    Manages state and orchestrates interactions without UI coupling.
    """

    def __init__(
        self,
        session_id: str,
        repository: IRepository,
        asset_store: IAssetStore,
        engine: DarkroomEngine,
    ):
        self.session_id = session_id
        self.repository = repository
        self.asset_store = asset_store
        self.engine = engine

        # State
        self.uploaded_files: List[Dict[str, str]] = []
        self.file_settings: Dict[str, ImageSettings] = {}
        self.thumbnails: Dict[str, Any] = {}
        self.selected_file_idx: int = 0
        self.clipboard: Optional[Dict[str, Any]] = None
        self.icc_profile_path: Optional[str] = None
        self.show_curve: bool = False

    def sync_files(
        self, current_uploaded_names: Set[str], raw_files: List[Any]
    ) -> None:
        """
        Synchronizes the session's file list with the provided set of file names.
        """
        last_names = {f["name"] for f in self.uploaded_files}
        new_names = current_uploaded_names - last_names

        if new_names:
            for f in raw_files:
                if f.name in new_names:
                    res = self.asset_store.register_asset(f, self.session_id)
                    if res:
                        cached_path, f_hash = res
                        self.uploaded_files.append(
                            {"name": f.name, "path": cached_path, "hash": f_hash}
                        )

        removed_from_widget = last_names - current_uploaded_names
        if removed_from_widget:
            for f_meta in self.uploaded_files:
                if f_meta["name"] in removed_from_widget:
                    self.asset_store.remove(f_meta["path"])

            self.uploaded_files = [
                f for f in self.uploaded_files if f["name"] not in removed_from_widget
            ]
            if self.selected_file_idx >= len(self.uploaded_files):
                self.selected_file_idx = max(0, len(self.uploaded_files) - 1)

    def get_active_settings(self) -> Optional[ImageSettings]:
        """
        Returns settings for the currently selected file.
        Loads from repository if not in memory.
        """
        if not self.uploaded_files:
            return None

        f_hash = self.uploaded_files[self.selected_file_idx]["hash"]
        if f_hash not in self.file_settings:
            settings = self.repository.load_file_settings(f_hash)
            if settings is None:
                settings = ImageSettings.from_dict(DEFAULT_SETTINGS.to_dict())
            self.file_settings[f_hash] = settings

        return self.file_settings[f_hash]

    def update_active_settings(self, settings: ImageSettings) -> None:
        """
        Updates memory and persistent storage with provided settings for the active file.
        """
        if not self.uploaded_files:
            return

        f_hash = self.uploaded_files[self.selected_file_idx]["hash"]
        self.file_settings[f_hash] = settings
        self.repository.save_file_settings(f_hash, settings)

    @property
    def current_file(self) -> Optional[Dict[str, str]]:
        if not self.uploaded_files:
            return None
        return self.uploaded_files[self.selected_file_idx]
