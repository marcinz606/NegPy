"""Offline tests for the capture-location picker. Every network call is patched."""

import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication  # noqa: E402

from negpy.desktop.view.widgets.location_picker_dialog import LocationPickerDialog  # noqa: E402

if not QApplication.instance():
    _app = QApplication(sys.argv)


_TOKYO = {
    "display_name": "Tokyo, Japan",
    "lat": "35.6762",
    "lon": "139.6503",
    "address": {"city": "Tokyo", "state": "Tokyo", "country": "Japan"},
}


def _dialog(monkeypatch, **kwargs) -> LocationPickerDialog:
    """Lookups run inline: no thread may outlive the test and reach the network."""
    monkeypatch.setattr("negpy.services.maps.reverse_place", lambda *a, **k: None)
    monkeypatch.setattr("negpy.desktop.view.widgets.slippy_map.fetch_tile", lambda *a, **k: None)
    monkeypatch.setattr(
        "negpy.desktop.view.widgets.location_picker_dialog.reverse_place",
        lambda *a, **k: None,
    )
    dialog = LocationPickerDialog(**kwargs)
    monkeypatch.setattr(dialog._pool, "start", lambda job, *args: job.run())
    return dialog


def test_opens_with_the_existing_location(monkeypatch) -> None:
    dlg = _dialog(monkeypatch, lat=35.6586, lon=139.7454, city="Tokyo", country="Japan")
    assert dlg.location() == (35.6586, 139.7454, "Tokyo", "", "Japan")
    assert dlg.map_view.pin() == (35.6586, 139.7454)


def test_search_result_sets_pin_and_place(monkeypatch) -> None:
    dlg = _dialog(monkeypatch)
    dlg._on_search_done([_TOKYO])
    assert dlg.results_list.isVisible() is False or dlg.results_list.count() == 1
    dlg._on_result_selected(0)
    lat, lon, city, state, country = dlg.location()
    assert (round(lat, 4), round(lon, 4)) == (35.6762, 139.6503)
    assert (city, state, country) == ("Tokyo", "Tokyo", "Japan")


def test_empty_search_result_reports_unavailable(monkeypatch) -> None:
    dlg = _dialog(monkeypatch)
    dlg._on_search_done([])
    assert "unavailable" in dlg.status_label.text()


def test_pasted_map_link_moves_the_pin(monkeypatch) -> None:
    dlg = _dialog(monkeypatch)
    dlg.coords_edit.setText("https://www.openstreetmap.org/#map=13/49.5/19.5")
    dlg._on_coords_edited()
    assert dlg.map_view.pin() == (49.5, 19.5)
    assert dlg.location()[:2] == (49.5, 19.5)


def test_unparsable_coordinates_are_reported_and_not_applied(monkeypatch) -> None:
    dlg = _dialog(monkeypatch)
    dlg.coords_edit.setText("somewhere nice")
    dlg._on_coords_edited()
    assert dlg.location()[:2] == (None, None)
    assert "not recognised" in dlg.status_label.text()


def test_clicking_the_map_fills_place_from_reverse_lookup(monkeypatch) -> None:
    dlg = _dialog(monkeypatch)
    dlg._on_pin_moved(35.6762, 139.6503)
    dlg._on_reverse_done(dlg._reverse_token, _TOKYO)
    assert dlg.location()[2:] == ("Tokyo", "Tokyo", "Japan")


def test_stale_reverse_lookup_is_ignored(monkeypatch) -> None:
    dlg = _dialog(monkeypatch, city="Kyoto")
    dlg._on_pin_moved(35.6762, 139.6503)
    dlg._on_reverse_done(dlg._reverse_token - 1, _TOKYO)
    assert dlg.location()[2] == "Kyoto"


def test_reverse_failure_keeps_the_coordinates(monkeypatch) -> None:
    dlg = _dialog(monkeypatch)
    dlg._on_pin_moved(35.6762, 139.6503)
    dlg._on_reverse_done(dlg._reverse_token, None)
    assert dlg.location()[:2] == (35.6762, 139.6503)
    assert "unavailable" in dlg.status_label.text()


def test_centre_frames_the_view_without_claiming_the_place(monkeypatch) -> None:
    """A scan file's coordinates say where it was digitized, not where it was shot."""
    dlg = _dialog(monkeypatch, center=(35.6762, 139.6503))
    assert dlg.map_view.pin() is None
    assert dlg.location() == (None, None, "", "", "")
    assert dlg.map_view._center == (35.6762, 139.6503)
    assert "scan file" in dlg.status_label.text()


def test_an_existing_place_wins_over_the_centre(monkeypatch) -> None:
    dlg = _dialog(monkeypatch, lat=35.6586, lon=139.7454, center=(0.0, 0.0))
    assert dlg.map_view.pin() == (35.6586, 139.7454)
