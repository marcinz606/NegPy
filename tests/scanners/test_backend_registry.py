from negpy.infrastructure.scanners import registry


def test_backend_choices_includes_sane():
    assert ("sane", "SANE") in registry.backend_choices()


def test_create_backend_resolves_and_falls_back(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(registry, "BACKENDS", {"sane": ("SANE", lambda: sentinel)})

    assert registry.create_backend("sane") is sentinel
    # An unknown/removed persisted id must fall back to the default, not crash.
    assert registry.create_backend("bogus") is sentinel
