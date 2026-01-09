import os
from typing import List, Dict, Optional, Any, Set
from src.core.models import WorkspaceConfig
from src.core.interfaces import IRepository, IAssetStore
from src.application.engine import DarkroomEngine


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
        self.file_settings: Dict[str, WorkspaceConfig] = {}
        self.thumbnails: Dict[str, Any] = {}
        self.selected_file_idx: int = 0
        self.clipboard: Optional[Dict[str, Any]] = None
        self.icc_profile_path: Optional[str] = None
        self.show_curve: bool = False
        self.watched_folders: Set[str] = set()

    def sync_files(
        self, current_uploaded_names: Set[str], raw_files: List[Any]
    ) -> None:
        """
        Synchronizes the session's file list with the provided set of file names.
        Only affects 'managed' assets (those in the cache).
        """
        # In this architecture, we know that assets in the cache_dir are managed by UI
        # We need to access cache_dir from the asset_store.
        # Since IAssetStore is a Protocol, we might need a runtime check or cast.
        cache_dir = getattr(self.asset_store, "cache_dir", "")

        managed_files = [
            f
            for f in self.uploaded_files
            if cache_dir
            and os.path.abspath(f["path"]).startswith(os.path.abspath(cache_dir))
        ]
        last_managed_names = {f["name"] for f in managed_files}
        new_names = current_uploaded_names - last_managed_names

        # 1. Add new uploads
        if new_names:
            for f in raw_files:
                if f.name in new_names:
                    res = self.asset_store.register_asset(f, self.session_id)
                    if res:
                        cached_path, f_hash = res
                        self.uploaded_files.append(
                            {"name": f.name, "path": cached_path, "hash": f_hash}
                        )

        # 2. Remove only managed files that are no longer in the uploader
        removed_from_widget = last_managed_names - current_uploaded_names
        if removed_from_widget:
            self.uploaded_files = [
                f
                for f in self.uploaded_files
                if not (f["name"] in removed_from_widget and f in managed_files)
            ]

            if self.selected_file_idx >= len(self.uploaded_files):
                self.selected_file_idx = max(0, len(self.uploaded_files) - 1)

    def add_local_assets(self, paths: List[str]) -> None:
        """
        Directly adds local files via path strings.
        Bypasses Streamlit's buffer copying.
        """
        current_paths = {f["path"] for f in self.uploaded_files}

        for p in paths:
            if p not in current_paths:
                res = self.asset_store.register_asset(p, self.session_id)
                if res:
                    cached_path, f_hash = res
                    self.uploaded_files.append(
                        {
                            "name": os.path.basename(p),
                            "path": cached_path,
                            "hash": f_hash,
                        }
                    )

    def create_default_config(self) -> WorkspaceConfig:
        """
        Creates a fresh WorkspaceConfig with correct defaults, ensuring
        environment-dependent values (like export_path) are applied.
        """
        from src.config import DEFAULT_WORKSPACE_CONFIG, APP_CONFIG

        # Start with the static default
        defaults = DEFAULT_WORKSPACE_CONFIG.to_dict()

        # Enforce dynamic defaults that might depend on env vars
        # This fixes the issue where default_factory in models.py uses hardcoded "export"
        if "export_path" in defaults:
            defaults["export_path"] = APP_CONFIG.default_export_dir

        return WorkspaceConfig.from_flat_dict(defaults)

    def get_settings_for_file(self, f_hash: str) -> WorkspaceConfig:
        """
        Retrieves settings for a specific file hash.
        Loads from repository if not in memory.
        """
        if f_hash in self.file_settings:
            return self.file_settings[f_hash]

        settings = self.repository.load_file_settings(f_hash)
        if settings is None:
            settings = self.create_default_config()

        self.file_settings[f_hash] = settings
        return settings

    def get_active_settings(self) -> Optional[WorkspaceConfig]:
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
                # Use centralized default creation
                settings = self.create_default_config()

            self.file_settings[f_hash] = settings

        return self.file_settings[f_hash]

    def update_active_settings(
        self, settings: WorkspaceConfig, persist: bool = True
    ) -> None:
        """
        Updates memory and persistent storage with provided settings for the active file.
        """
        if not self.uploaded_files:
            return

        f_hash = self.uploaded_files[self.selected_file_idx]["hash"]
        self.file_settings[f_hash] = settings
        if persist:
            self.repository.save_file_settings(f_hash, settings)

    @property
    def current_file(self) -> Optional[Dict[str, str]]:
        if not self.uploaded_files:
            return None
        return self.uploaded_files[self.selected_file_idx]
