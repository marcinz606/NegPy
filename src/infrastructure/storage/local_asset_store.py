import os
import shutil
import uuid
from typing import Optional, Tuple, Any
from PIL import Image
from src.kernel.image.logic import calculate_file_hash
from src.kernel.system.logging import get_logger
from src.domain.interfaces import IAssetStore

logger = get_logger(__name__)


class LocalAssetStore(IAssetStore):
    """
    Manages file access and thumbnail cache.
    """

    def __init__(self, cache_dir: str, icc_dir: str) -> None:
        self.cache_dir = cache_dir
        self.icc_dir = icc_dir
        self.thumb_dir = os.path.join(cache_dir, "thumbnails")

    def initialize(self) -> None:
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
            os.makedirs(self.icc_dir, exist_ok=True)
            os.makedirs(self.thumb_dir, exist_ok=True)
            logger.info(f"LocalAssetStore initialized at {self.cache_dir}")
        except Exception as e:
            logger.error(f"Failed to initialize LocalAssetStore: {e}")

    def register_asset(self, source: Any, session_id: str) -> Optional[Tuple[str, str]]:
        """
        Registers file (zero-copy for local paths, copies uploads).
        """
        try:
            # Zero-Copy (Desktop/Local)
            if isinstance(source, str) and os.path.exists(source):
                f_hash = calculate_file_hash(source)
                return source, f_hash
            elif isinstance(source, str):
                logger.warning(f"Registration failed: Path does not exist: {source}")

            # Docker/Streamlit upload
            if hasattr(source, "getbuffer") and hasattr(source, "name"):
                session_dir = self._get_session_dir(session_id)
                unique_name = f"{uuid.uuid4()}_{source.name}"
                file_path = os.path.join(session_dir, unique_name)

                with open(file_path, "wb") as f_out:
                    f_out.write(source.getbuffer())

                f_hash = calculate_file_hash(file_path)
                return file_path, f_hash

            return None
        except Exception as e:
            logger.error(f"Asset registration failed: {e}")
            return None

    def get_thumbnail(self, file_hash: str) -> Optional[Image.Image]:
        """Loads cached thumb."""
        thumb_path = os.path.join(self.thumb_dir, f"{file_hash}.jpg")
        if os.path.exists(thumb_path):
            try:
                return Image.open(thumb_path)
            except Exception:
                return None
        return None

    def save_thumbnail(self, file_hash: str, image: Image.Image) -> None:
        """Persists thumb to disk."""
        try:
            thumb_path = os.path.join(self.thumb_dir, f"{file_hash}.jpg")
            # Save as JPEG for speed and smaller file size
            image.save(thumb_path, "JPEG", quality=85)
        except Exception as e:
            logger.error(f"Failed to save thumbnail {file_hash}: {e}")

    def _get_session_dir(self, session_id: str) -> str:
        session_dir = os.path.join(self.cache_dir, session_id)
        os.makedirs(session_dir, exist_ok=True)
        return session_dir

    def remove(self, file_path: str) -> None:
        try:
            if os.path.exists(file_path):
                # Don't delete if it's NOT in our cache (referenced file)
                if os.path.abspath(file_path).startswith(
                    os.path.abspath(self.cache_dir)
                ):
                    os.remove(file_path)
                    logger.info(f"Managed asset removed: {file_path}")
        except Exception as e:
            logger.error(f"Failed to remove asset {file_path}: {e}")

    def clear_session_assets(self, session_id: str) -> None:
        session_dir = os.path.join(self.cache_dir, session_id)
        if os.path.exists(session_dir):
            shutil.rmtree(session_dir)

    def clear_all(self) -> None:
        """Nukes the cache."""
        try:
            if os.path.exists(self.cache_dir):
                for item in os.listdir(self.cache_dir):
                    item_path = os.path.join(self.cache_dir, item)
                    if os.path.isdir(item_path):
                        shutil.rmtree(item_path)
                    else:
                        os.remove(item_path)
            self.initialize()
            logger.info("LocalAssetStore completely cleared.")
        except Exception as e:
            logger.error(f"Failed to clear store: {e}")
