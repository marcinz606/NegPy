import os
import shutil
import uuid
from typing import Optional, Tuple, Any
from src.helpers import calculate_file_hash
from src.logging_config import get_logger
from src.core.persistence.interfaces import IAssetStore

logger = get_logger(__name__)


class LocalAssetStore(IAssetStore):
    """
    Manages the persistence and lifecycle of RAW files on disk using local file system.
    """

    def __init__(self, cache_dir: str, icc_dir: str) -> None:
        self.cache_dir = cache_dir
        self.icc_dir = icc_dir

    def initialize(self) -> None:
        """
        Prepares the base cache and ICC directories.
        """
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
            os.makedirs(self.icc_dir, exist_ok=True)
            logger.info(
                f"LocalAssetStore initialized (cache: {self.cache_dir}, icc: {self.icc_dir})"
            )
        except Exception as e:
            logger.error(f"Failed to initialize LocalAssetStore: {e}")

    def _get_session_dir(self, session_id: str) -> str:
        """
        Returns and creates a session-specific cache directory.
        """
        session_dir = os.path.join(self.cache_dir, session_id)
        os.makedirs(session_dir, exist_ok=True)
        return session_dir

    def persist(self, uploaded_file: Any, session_id: str) -> Optional[Tuple[str, str]]:
        """
        Saves an uploaded file to the session-specific cache directory and returns (path, hash).
        """
        try:
            session_dir = self._get_session_dir(session_id)
            # Create a unique filename to avoid collisions
            unique_name = f"{uuid.uuid4()}_{uploaded_file.name}"
            file_path = os.path.join(session_dir, unique_name)

            with open(file_path, "wb") as f_out:
                f_out.write(uploaded_file.getbuffer())

            # Calculate hash after writing to disk
            f_hash = calculate_file_hash(file_path)

            logger.info(
                f"Asset persisted: {uploaded_file.name} -> {file_path} (hash: {f_hash[:8]})"
            )
            return file_path, f_hash
        except Exception as e:
            logger.error(f"Failed to persist asset {uploaded_file.name}: {e}")
            return None

    def remove(self, file_path: str) -> None:
        """
        Removes a specific file from the cache.
        """
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Asset removed: {file_path}")
        except Exception as e:
            logger.error(f"Failed to remove asset {file_path}: {e}")

    def clear_session_assets(self, session_id: str) -> None:
        """
        Clears all assets for a specific session.
        """
        session_dir = os.path.join(self.cache_dir, session_id)
        if os.path.exists(session_dir):
            shutil.rmtree(session_dir)

    def clear_all(self) -> None:
        """
        Wipes the entire cache directory.
        """
        try:
            if os.path.exists(self.cache_dir):
                for item in os.listdir(self.cache_dir):
                    item_path = os.path.join(self.cache_dir, item)
                    if os.path.isdir(item_path):
                        shutil.rmtree(item_path)
                    else:
                        os.remove(item_path)
            os.makedirs(self.cache_dir, exist_ok=True)
            logger.info("Global Asset cache completely cleared.")
        except Exception as e:
            logger.error(f"Failed to clear Asset cache: {e}")
