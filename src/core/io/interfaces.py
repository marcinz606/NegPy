from typing import Protocol, Any, ContextManager, List, Optional


class IImageLoader(Protocol):
    """
    Strategy interface for loading different image formats.
    """

    def load(self, file_path: str) -> ContextManager[Any]: ...


class IFilePicker(Protocol):
    """


    Interface for picking assets from the filesystem.


    """

    def pick_files(self, initial_dir: Optional[str] = None) -> List[str]: ...

    def pick_folder(
        self, initial_dir: Optional[str] = None
    ) -> tuple[str, List[str]]: ...

    def pick_export_folder(self, initial_dir: Optional[str] = None) -> str: ...
