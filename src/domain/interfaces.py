from typing import (
    Protocol,
    Optional,
    Any,
    Tuple,
    ContextManager,
    List,
)
from dataclasses import dataclass, field
from src.domain.types import ImageBuffer, ROI, Dimensions
from src.domain.models import WorkspaceConfig


@dataclass
class PipelineContext:
    """
    Shared state passed through the pipeline.
    """

    original_size: Dimensions
    scale_factor: float
    process_mode: str = "C41"
    active_roi: Optional[ROI] = None
    metrics: dict[str, Any] = field(default_factory=dict)


class IImageSource(Protocol):
    """
    Interface for loading images.
    """

    def read(self) -> ImageBuffer: ...


class IRepository(Protocol):
    """
    Persists application and image settings.
    """

    def save_file_settings(self, file_hash: str, settings: WorkspaceConfig) -> None: ...

    def load_file_settings(self, file_hash: str) -> Optional[WorkspaceConfig]: ...

    def save_global_setting(self, key: str, value: Any) -> None: ...
    def get_global_setting(self, key: str, default: Any = None) -> Any: ...
    def initialize(self) -> None: ...


class IAssetStore(Protocol):
    """
    Manages physical assets (files, thumbnails).
    """

    def register_asset(
        self, source: Any, session_id: str
    ) -> Optional[Tuple[str, str]]: ...

    def get_thumbnail(self, file_hash: str) -> Optional[Any]: ...
    def save_thumbnail(self, file_hash: str, image: Any) -> None: ...

    def remove(self, file_path: str) -> None: ...
    def clear_session_assets(self, session_id: str) -> None: ...
    def initialize(self) -> None: ...
    def clear_all(self) -> None: ...


class IImageLoader(Protocol):
    """
    Loads specific image formats. Returns (context, metadata).
    """

    def load(self, file_path: str) -> Tuple[ContextManager[Any], dict]: ...


class IFilePicker(Protocol):
    """
    System file dialog wrapper.
    """

    def pick_files(self, initial_dir: Optional[str] = None) -> List[str]: ...

    def pick_folder(
        self, initial_dir: Optional[str] = None
    ) -> tuple[str, List[str]]: ...

    def pick_export_folder(self, initial_dir: Optional[str] = None) -> str: ...
