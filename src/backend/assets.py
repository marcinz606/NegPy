import os
import shutil
import uuid
from typing import Optional, Tuple, Any
from src.config import APP_CONFIG
from src.helpers import calculate_file_hash
from src.logging_config import get_logger

logger = get_logger(__name__)


class AssetManager:
    """
    Manages the persistence and lifecycle of RAW files on disk to minimize RAM usage.
    """

    @staticmethod
    def initialize() -> None:
        """
        Prepares the base cache directory.
        """
        cache_dir = APP_CONFIG["cache_dir"]
        try:
            os.makedirs(cache_dir, exist_ok=True)
            logger.info(f"AssetManager base directory ensured at {cache_dir}")
        except Exception as e:
            logger.error(f"Failed to initialize AssetManager: {e}")

    @staticmethod
    def get_session_dir(session_id: str) -> str:
        """
        Returns and creates a session-specific cache directory.
        """
        session_dir = os.path.join(APP_CONFIG["cache_dir"], session_id)
        os.makedirs(session_dir, exist_ok=True)
        return session_dir

    @staticmethod
    def persist(
        uploaded_file: Any, session_id: str
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Saves an uploaded file to the session-specific cache directory and returns (path, hash).
        """
        try:
            session_dir = AssetManager.get_session_dir(session_id)
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
            return None, None

    @staticmethod
    def remove(file_path: str) -> None:
        """
        Removes a specific file from the cache.
        """
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Asset removed: {file_path}")
        except Exception as e:
            logger.error(f"Failed to remove asset {file_path}: {e}")

    @staticmethod
    def clear_all() -> None:
        """
        Wipes the entire cache directory (Global Cleanup).
        """
        try:
            cache_dir = APP_CONFIG["cache_dir"]
            if os.path.exists(cache_dir):
                # We iterate through subdirs to be safe
                for item in os.listdir(cache_dir):
                    item_path = os.path.join(cache_dir, item)
                    if os.path.isdir(item_path):
                        shutil.rmtree(item_path)
                    else:
                        os.remove(item_path)
            os.makedirs(cache_dir, exist_ok=True)
            logger.info("Global Asset cache completely cleared.")
        except Exception as e:
            logger.error(f"Failed to clear Asset cache: {e}")
