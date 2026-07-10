"""Film-stock preset store unit tests (fake repo)."""

from negpy.services.capture.presets import PresetStore, ScanlightPreset


class FakeRepo:
    def __init__(self):
        self._d = {}

    def get_global_setting(self, key, default=None):
        return self._d.get(key, default)

    def save_global_setting(self, key, value):
        self._d[key] = value


def test_save_get_list_delete():
    store = PresetStore(FakeRepo())
    assert store.names() == []

    p = ScanlightPreset(r_level=200, g_level=180, b_level=255, shutter_b="1/4")
    store.save("Portra 400", p)
    assert store.names() == ["Portra 400"]
    assert store.get("Portra 400") == p

    store.save("Ektar 100", ScanlightPreset(r_level=210))
    assert store.names() == ["Ektar 100", "Portra 400"]  # sorted

    store.delete("Portra 400")
    assert store.names() == ["Ektar 100"]
    assert store.get("missing") is None


def test_get_tolerates_garbage():
    repo = FakeRepo()
    repo.save_global_setting("scanlight_presets", {"x": "not a dict"})
    assert PresetStore(repo).get("x") is None


def test_get_ignores_unknown_keys():
    repo = FakeRepo()
    repo.save_global_setting("scanlight_presets", {"y": {"r_level": 100, "bogus": 5}})
    p = PresetStore(repo).get("y")
    assert p is not None and p.r_level == 100
