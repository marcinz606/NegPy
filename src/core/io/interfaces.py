from typing import Protocol, Any, ContextManager


class IImageLoader(Protocol):
    """
    Strategy interface for loading different image formats.
    """

    def load(self, file_path: str) -> ContextManager[Any]: ...
