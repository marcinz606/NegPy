"""Film-stock preset store unit tests (fake repo)."""

from negpy.services.capture.presets import PresetStore, ScanlightPreset, framing_levels


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


def test_iso_and_aperture_round_trip():
    store = PresetStore(FakeRepo())
    p = ScanlightPreset(r_level=200, shutter_r="1/5", iso="100", aperture="f/8")
    store.save("Portra 400", p)
    assert store.get("Portra 400") == p  # the baked exposure survives persist + reload


def test_legacy_preset_without_exposure_defaults_blank():
    repo = FakeRepo()  # a preset saved before ISO/aperture existed must still load
    repo.save_global_setting("scanlight_presets", {"Old": {"r_level": 200, "shutter_r": "1/5"}})
    p = PresetStore(repo).get("Old")
    assert p is not None and p.iso == "" and p.aperture == ""


def test_framing_levels_dim_three_stops():
    assert framing_levels(210, 95, 80) == (26, 11, 10)  # the reference start point, 2^3 dimmer


def test_framing_levels_keep_lit_channels_lit_and_dark_channels_dark():
    assert framing_levels(255, 4, 0) == (31, 1, 0)  # a lit channel never dims to 0; off stays off
