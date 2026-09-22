"""CLIP embedding generation for search by meaning, mirroring ThumbnailWorker exactly:
its own thread-bound QObject, the same chunked-progress shape, feeding the same
batch-lane machinery the controller already runs thumbnails through."""

from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot

from negpy.kernel.system.logging import get_logger

logger = get_logger(__name__)

# Chunked, not per-file: every emit costs the search-ranked model a full relayout.
_EMBED_CHUNK = 8


class EmbeddingWorker(QObject):
    """Asynchronous embedding generation."""

    progress = pyqtSignal(int, int, str)
    finished = pyqtSignal(dict)
    # Chunks of the running batch, so ranking improves as it goes rather than
    # waiting for the whole session to finish indexing.
    partial = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, asset_store, repo) -> None:
        super().__init__()
        self._store = asset_store
        self._repo = repo
        self._cancelled = False

    def cancel(self) -> None:
        """Stops the running batch before its next unstarted file -- a file already
        mid-decode finishes normally. Reset at the start of the next generate()."""
        self._cancelled = True

    @pyqtSlot(list)
    def generate(self, files: list) -> None:
        """Embeds a list of files with progress reporting."""
        import asyncio

        from negpy.services.assets import embeddings as embedding_service

        self._cancelled = False
        try:
            total = len(files)

            async def _progress_callback(current: int, name: str):
                self.progress.emit(current, total, name)

            pending: dict = {}

            def _ready_callback(file_hash: str, vector) -> None:
                pending[file_hash] = vector
                if len(pending) >= _EMBED_CHUNK:
                    self.partial.emit(dict(pending))
                    pending.clear()

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                new_embeddings = loop.run_until_complete(
                    embedding_service.generate_batch_embeddings(
                        files,
                        self._store,
                        self._repo,
                        progress_callback=_progress_callback,
                        ready_callback=_ready_callback,
                        is_cancelled=lambda: self._cancelled,
                    )
                )
            finally:
                loop.close()
                asyncio.set_event_loop(None)
            self.finished.emit(new_embeddings)
        except Exception as e:
            logger.error(f"Embedding generation failure: {e}")
            self.error.emit(str(e))
