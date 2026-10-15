from unittest.mock import MagicMock

from PyQt6.QtWidgets import QApplication

from negpy.desktop.controller import AppController
from negpy.desktop.session import AppState, AssetListModel
from negpy.services.assets.thumbnails import asset_thumbnail_key

_app = QApplication.instance() or QApplication([])


def _model() -> AssetListModel:
    state = AppState()
    state.uploaded_files = [{"name": n, "path": f"/r/{n}", "hash": n} for n in ("b.tif", "a.tif", "c.tif")]
    return AssetListModel(state)


def test_a_new_thumbnail_repaints_its_row_without_a_relayout() -> None:
    model = _model()
    relayouts, repainted = [], []
    model.layoutChanged.connect(lambda *_: relayouts.append(True))
    model.dataChanged.connect(lambda top, bottom, *_: repainted.append((top.row(), bottom.row())))

    model.refresh_thumbnails({asset_thumbnail_key(model._state.uploaded_files[0])})

    assert relayouts == []
    assert repainted == [(1, 1)]  # sorted by name: a, b, c


def test_applying_thumbnails_repaints_rather_than_reindexes() -> None:
    controller = MagicMock()
    controller.state = AppState()
    controller.state.uploaded_files = [{"name": "a.tif", "path": "/r/a.tif", "hash": "a"}]
    controller._thumbnail_pending_correction = {}
    controller._set_thumbnail.return_value = True

    AppController._apply_thumbnails(controller, {"a": object()})

    controller.session.asset_model.refresh.assert_not_called()
    controller.session.asset_model.refresh_thumbnails.assert_called_once()
