import os

from negpy.infrastructure.loaders import helpers
from negpy.services.assets.toml_cache import load_toml_cached


def _touch_newer(path):
    st = os.stat(path)
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))


def test_toml_cache_rereads_on_change(tmp_path):
    p = tmp_path / "a.toml"
    p.write_text('name = "one"\n')
    assert load_toml_cached(str(p))["name"] == "one"
    p.write_text('name = "two"\n')
    _touch_newer(p)
    assert load_toml_cached(str(p))["name"] == "two"
    p.write_text("name = [\n")
    _touch_newer(p)
    assert load_toml_cached(str(p)) is None
    assert load_toml_cached(str(tmp_path / "missing.toml")) is None


def test_exif_cache_returns_copies_and_invalidates(tmp_path, monkeypatch):
    p = tmp_path / "x.jpg"
    p.write_bytes(b"not an image")
    calls = []

    def fake_read(path):
        calls.append(path)
        return {"0th": {274: len(calls)}}

    monkeypatch.setattr(helpers, "_read_exif_uncached", fake_read)
    helpers._exif_cache.clear()
    first = helpers.read_exif_from_file(str(p))
    first["0th"][274] = 99
    second = helpers.read_exif_from_file(str(p))
    assert second == {"0th": {274: 1}} and len(calls) == 1
    _touch_newer(p)
    assert helpers.read_exif_from_file(str(p)) == {"0th": {274: 2}}
