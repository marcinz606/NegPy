"""The open frame rendered through the export path, for the Export Size view and the loupe."""

from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot

from negpy.desktop.workers.export import ExportTask
from negpy.kernel.system.logging import get_logger
from negpy.services.rendering.image_processor import ImageProcessor

logger = get_logger(__name__)


class ExportViewWorker(QObject):
    """Owns its processor, so a batch export never queues ahead of the view. The processor keeps
    its decoded source, so a re-render after an edit skips the decode."""

    finished = pyqtSignal(object, object)  # key, (buffer, content_rect) or None on failure

    def __init__(self) -> None:
        super().__init__()
        self._processor = ImageProcessor()

    @pyqtSlot(object, object)
    def render(self, key: object, task: ExportTask) -> None:
        info = task.file_info
        try:
            result = self._processor.render_export_view(
                info["path"],
                task.params,
                task.export_settings,
                info["hash"],
                prefer_gpu=task.gpu_enabled,
                bounds_override=task.bounds_override,
                half=int(info.get("half") or 0),
                split_x=float(info.get("split_x") or 0.5),
                crop_rect=tuple(info["crop_rect"]) if info.get("crop_rect") else None,
                gutter_thickness=float(info.get("gutter_thickness") or 0.0),
                split_axis=str(info.get("split_axis") or "x"),
                diptych=task.diptych,
            )
        except Exception:
            logger.exception("Export Size render failed for %s", info.get("path"))
            result = None
        self.finished.emit(key, result)

    def destroy(self) -> None:
        self._processor.destroy_all()
