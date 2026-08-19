"""Offline tests for the OpenStreetMap helpers: no request leaves the process."""

from __future__ import annotations

import json

import pytest

from negpy.services import maps


class _Response:
    def __init__(self, payload: bytes, status: int = 200):
        self._payload = payload
        self.status = status

    def read(self) -> bytes:
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, *_exc) -> None:
        return None


def _patch_urlopen(monkeypatch, payload: object, status: int = 200) -> list[str]:
    urls: list[str] = []

    def fake(request, timeout=None):
        urls.append(request.full_url)
        assert request.headers["User-agent"].startswith("NegPy/")
        if payload is None:
            raise OSError("offline")
        body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
        return _Response(body, status)

    monkeypatch.setattr(maps.urllib.request, "urlopen", fake)
    return urls


class TestSearchPlaces:
    def test_returns_results(self, monkeypatch) -> None:
        urls = _patch_urlopen(monkeypatch, [{"display_name": "Tokyo"}])
        assert maps.search_places("Tokyo") == [{"display_name": "Tokyo"}]
        assert "addressdetails=1" in urls[0]

    def test_blank_query_makes_no_request(self, monkeypatch) -> None:
        urls = _patch_urlopen(monkeypatch, [])
        assert maps.search_places("   ") == []
        assert urls == []

    def test_offline_is_empty(self, monkeypatch) -> None:
        _patch_urlopen(monkeypatch, None)
        assert maps.search_places("Tokyo") == []

    def test_non_json_is_empty(self, monkeypatch) -> None:
        _patch_urlopen(monkeypatch, b"<html>rate limited</html>")
        assert maps.search_places("Tokyo") == []


class TestReversePlace:
    def test_returns_result(self, monkeypatch) -> None:
        _patch_urlopen(monkeypatch, {"address": {"city": "Tokyo"}})
        assert maps.reverse_place(35.0, 139.0) == {"address": {"city": "Tokyo"}}

    def test_error_payload_is_none(self, monkeypatch) -> None:
        _patch_urlopen(monkeypatch, {"error": "Unable to geocode"})
        assert maps.reverse_place(0.0, 0.0) is None

    def test_offline_is_none(self, monkeypatch) -> None:
        _patch_urlopen(monkeypatch, None)
        assert maps.reverse_place(35.0, 139.0) is None


class TestPlaceFields:
    @pytest.mark.parametrize(
        "address, city",
        [
            ({"city": "Tokyo"}, "Tokyo"),
            ({"village": "Ōgimi"}, "Ōgimi"),
            ({"town": "Hakone", "city": "Tokyo"}, "Tokyo"),
            ({}, ""),
        ],
    )
    def test_city_falls_back_through_the_settlement_keys(self, address: dict, city: str) -> None:
        assert maps.place_fields({"address": address})[0] == city

    def test_missing_address_is_empty(self) -> None:
        assert maps.place_fields({"lat": "35"}) == ("", "", "")
        assert maps.place_fields(None) == ("", "", "")


class TestFetchTile:
    def test_disk_cache_is_used_before_the_network(self, monkeypatch, tmp_path) -> None:
        path = tmp_path / "9" / "1" / "2.png"
        path.parent.mkdir(parents=True)
        path.write_bytes(b"cached")
        monkeypatch.setattr(maps, "tile_cache_path", lambda z, x, y: str(path))
        urls = _patch_urlopen(monkeypatch, b"downloaded")
        assert maps.fetch_tile(9, 1, 2) == b"cached"
        assert urls == []

    def test_download_writes_the_cache(self, monkeypatch, tmp_path) -> None:
        path = tmp_path / "9" / "1" / "2.png"
        monkeypatch.setattr(maps, "tile_cache_path", lambda z, x, y: str(path))
        _patch_urlopen(monkeypatch, b"downloaded")
        assert maps.fetch_tile(9, 1, 2) == b"downloaded"
        assert path.read_bytes() == b"downloaded"

    def test_offline_returns_none(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setattr(maps, "tile_cache_path", lambda z, x, y: str(tmp_path / "t.png"))
        _patch_urlopen(monkeypatch, None)
        assert maps.fetch_tile(9, 1, 2) is None


class TestAcceptLanguage:
    def test_asks_for_the_locale_language_with_an_english_fallback(self, monkeypatch) -> None:
        monkeypatch.setattr(maps.locale, "getlocale", lambda *a: ("pl_PL", "UTF-8"))
        assert maps.accept_language() == "pl-PL,en"

    def test_falls_back_to_the_environment_then_to_english(self, monkeypatch) -> None:
        monkeypatch.setattr(maps.locale, "getlocale", lambda *a: (None, None))
        monkeypatch.setenv("LANG", "ja_JP.UTF-8")
        assert maps.accept_language() == "ja-JP,en"

        monkeypatch.setenv("LANG", "")
        assert maps.accept_language() == "en"

    def test_both_lookups_send_it(self, monkeypatch) -> None:
        monkeypatch.setattr(maps, "accept_language", lambda: "pl-PL,en")
        urls = _patch_urlopen(monkeypatch, [])
        maps.search_places("Tokyo")
        maps.reverse_place(35.0, 139.0)
        assert all("accept-language=pl-PL%2Cen" in url for url in urls)
