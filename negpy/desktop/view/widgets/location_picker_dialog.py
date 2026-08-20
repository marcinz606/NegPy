"""Pick the capture place on a map, by place name, or by typed coordinates."""

from __future__ import annotations

from typing import Optional

import qtawesome as qta
from PyQt6.QtCore import QEvent, QModelIndex, QObject, QRunnable, QStringListModel, Qt, QThreadPool, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QCompleter,
    QDialog,
    QDialogButtonBox,
    QGridLayout,
    QLineEdit,
    QVBoxLayout,
)

from negpy.desktop.view.styles.templates import field_label, hint_label
from negpy.desktop.view.styles.theme import THEME
from negpy.desktop.view.widgets.slippy_map import SlippyMapWidget
from negpy.features.metadata.capture import format_coords, parse_coords
from negpy.services.maps import place_fields, result_coords, reverse_place, search_places

_OFFLINE_HINT = "Map unavailable — enter coordinates manually."
_NO_MATCH_HINT = "No place matched, or the lookup is unreachable."
_SHUTDOWN_WAIT_MS = 6000
# Nominatim asks for at most one request a second, so a keystroke must not be a request.
_SEARCH_DEBOUNCE_MS = 500
_MIN_QUERY_CHARS = 3


class _LookupSignals(QObject):
    search_done = pyqtSignal(object)
    reverse_done = pyqtSignal(int, object)

    def __init__(self):
        super().__init__()
        self.stopped = False


class _SearchJob(QRunnable):
    def __init__(self, signals: _LookupSignals, query: str):
        super().__init__()
        self._signals = signals
        self._query = query

    def run(self) -> None:
        if self._signals.stopped:
            return
        self._signals.search_done.emit(search_places(self._query))


class _ReverseJob(QRunnable):
    def __init__(self, signals: _LookupSignals, token: int, lat: float, lon: float):
        super().__init__()
        self._signals = signals
        self._token = token
        self._lat, self._lon = lat, lon

    def run(self) -> None:
        if self._signals.stopped:
            return
        self._signals.reverse_done.emit(self._token, reverse_place(self._lat, self._lon))


class LocationPickerDialog(QDialog):
    """Coordinates are authoritative; the place names are a proposal the user can edit."""

    def __init__(
        self,
        lat: Optional[float] = None,
        lon: Optional[float] = None,
        city: str = "",
        state: str = "",
        country: str = "",
        center: Optional[tuple[float, float]] = None,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Capture location")
        self.setMinimumSize(560, 560)
        self.setStyleSheet(f"QDialog {{ background: {THEME.bg_dark}; }}")

        # The pool is owned by the dialog, so closing it joins any running lookup before the
        # signal object goes away. done() drops the queue first to keep that join short.
        self._pool = QThreadPool(self)
        self._pool.setMaxThreadCount(2)
        self._signals = _LookupSignals()
        self._signals.search_done.connect(self._on_search_done)
        self._signals.reverse_done.connect(self._on_reverse_done)
        self._reverse_token = 0
        self._results: dict[str, dict] = {}

        root = QVBoxLayout(self)
        root.setContentsMargins(THEME.space_xl, THEME.space_xl, THEME.space_xl, THEME.space_xl)
        root.setSpacing(THEME.space_lg)

        root.addWidget(
            hint_label("Search a place, click the map, or paste coordinates or a map link. Opening this dialog contacts OpenStreetMap.")
        )

        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("e.g. Tokyo, Japan")
        self.search_edit.addAction(
            qta.icon("fa5s.search", color=THEME.text_secondary),
            QLineEdit.ActionPosition.LeadingPosition,
        )
        self.search_edit.textEdited.connect(self._on_search_text_edited)
        root.addWidget(self.search_edit)

        # A QCompleter, not a hand-rolled Qt.Popup: a popup grabs the keyboard, so the next
        # keystroke would go to the list instead of the field. Unfiltered, because the hits
        # come from the geocoder — filtering them again locally would hide "Tōkyō" for "tokyo".
        self._suggestions = QStringListModel(self)
        self._completer = QCompleter(self._suggestions, self)
        self._completer.setCompletionMode(QCompleter.CompletionMode.UnfilteredPopupCompletion)
        self._completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self._completer.setMaxVisibleItems(8)
        self._completer.setWidget(self.search_edit)
        self._completer.activated[QModelIndex].connect(self._on_suggestion_chosen)
        # Installed after the completer takes the field, so this filter is asked first and can
        # stand aside while the popup owns Return.
        self.search_edit.installEventFilter(self)

        self._search_timer = QTimer(self)
        self._search_timer.setSingleShot(True)
        self._search_timer.setInterval(_SEARCH_DEBOUNCE_MS)
        self._search_timer.timeout.connect(self._on_search)

        self.map_view = SlippyMapWidget()
        self.map_view.pin_moved.connect(self._on_pin_moved)
        root.addWidget(self.map_view, 1)

        fields = QGridLayout()
        fields.setHorizontalSpacing(THEME.space_lg)
        fields.setVerticalSpacing(THEME.space_sm)

        self.coords_edit = QLineEdit()
        self.coords_edit.setPlaceholderText("35.67620, 139.65030")
        self.coords_edit.editingFinished.connect(self._on_coords_edited)
        self.city_edit = QLineEdit(city)
        self.state_edit = QLineEdit(state)
        self.country_edit = QLineEdit(country)

        for column, (label, widget) in enumerate(
            (
                ("Coordinates", self.coords_edit),
                ("City", self.city_edit),
                ("State", self.state_edit),
                ("Country", self.country_edit),
            )
        ):
            fields.addWidget(field_label(label), (column // 2) * 2, column % 2)
            fields.addWidget(widget, (column // 2) * 2 + 1, column % 2)
        root.addLayout(fields)

        self.status_label = hint_label("")
        root.addWidget(self.status_label)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

        if lat is not None and lon is not None:
            self.coords_edit.setText(format_coords(lat, lon))
            self.map_view.set_pin(lat, lon)
            self.map_view.set_zoom(10)
        elif center is not None:
            # The scan file's own position only frames the view. Adopting it as the capture
            # place would claim the frame was shot where it was digitized.
            self.map_view.set_center(*center)
            self.map_view.set_zoom(8)
            self.status_label.setText("Centred on the scan file's coordinates.")

    def eventFilter(self, obj, event) -> bool:  # noqa: N802 - Qt override
        """Return in the search field searches; without this the dialog's OK button takes it."""
        if obj is self.search_edit and event.type() == QEvent.Type.KeyPress:
            if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and not self._completer.popup().isVisible():
                self._on_search()
                return True
        return super().eventFilter(obj, event)

    def done(self, result: int) -> None:
        # Join the lookup threads here: the pool's destructor would wait with the GIL held,
        # which a worker needs to finish, and the app would hang instead of closing.
        self._signals.stopped = True
        self._pool.clear()
        self._pool.waitForDone(_SHUTDOWN_WAIT_MS)
        self.map_view.shutdown()
        super().done(result)

    def location(self) -> tuple[Optional[float], Optional[float], str, str, str]:
        coords = parse_coords(self.coords_edit.text())
        lat, lon = coords if coords else (None, None)
        return (
            lat,
            lon,
            self.city_edit.text().strip(),
            self.state_edit.text().strip(),
            self.country_edit.text().strip(),
        )

    # ── search ───────────────────────────────────────────────────────────

    def _on_search_text_edited(self, text: str) -> None:
        if len(text.strip()) < _MIN_QUERY_CHARS:
            self._search_timer.stop()
            return
        self._search_timer.start()

    def _on_search(self) -> None:
        self._search_timer.stop()
        query = self.search_edit.text().strip()
        if len(query) < _MIN_QUERY_CHARS:
            return
        self.status_label.setText("Searching…")
        self._pool.start(_SearchJob(self._signals, query))

    def _on_search_done(self, results: object) -> None:
        self._results = {}
        for item in results if isinstance(results, list) else []:
            name = str(item.get("display_name", ""))
            if name and name not in self._results:
                self._results[name] = item

        self._suggestions.setStringList(list(self._results))
        if self._results:
            self._completer.complete()
            # Highlight the top hit, so Return in the field takes it instead of falling
            # through to the dialog's OK button.
            self._completer.popup().setCurrentIndex(self._completer.completionModel().index(0, 0))
            self.status_label.setText("")
        else:
            self._completer.popup().hide()
            self.status_label.setText(_NO_MATCH_HINT)

    def _on_suggestion_chosen(self, index: QModelIndex) -> None:
        result = self._results.get(str(index.data()))
        if result is None:
            return
        coords = result_coords(result)
        if coords is None:
            return
        self.coords_edit.setText(format_coords(*coords))
        self.map_view.set_pin(*coords)
        self.map_view.set_zoom(10)
        self._apply_place(result)

    # ── map and coordinates ──────────────────────────────────────────────

    def _on_pin_moved(self, lat: float, lon: float) -> None:
        self.coords_edit.setText(format_coords(lat, lon))
        self._start_reverse(lat, lon)

    def _on_coords_edited(self) -> None:
        coords = parse_coords(self.coords_edit.text())
        if coords is None:
            self.status_label.setText("Coordinates not recognised.")
            return
        self.coords_edit.setText(format_coords(*coords))
        self.map_view.set_pin(*coords)
        self._start_reverse(*coords)

    def _start_reverse(self, lat: float, lon: float) -> None:
        self._reverse_token += 1
        self.status_label.setText("Looking up place…")
        self._pool.start(_ReverseJob(self._signals, self._reverse_token, lat, lon))

    def _on_reverse_done(self, token: int, result: object) -> None:
        if token != self._reverse_token:
            return
        if not isinstance(result, dict):
            self.status_label.setText(_OFFLINE_HINT)
            return
        self._apply_place(result)
        self.status_label.setText("")

    def _apply_place(self, result: dict) -> None:
        city, state, country = place_fields(result)
        self.city_edit.setText(city)
        self.state_edit.setText(state)
        self.country_edit.setText(country)
