from negpy.kernel.system.config import APP_CONFIG
from negpy.services.assets.presets import Presets


def test_save_load_delete_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(APP_CONFIG, "presets_dir", str(tmp_path))

    Presets.save_preset("Portra Look", {"density": 1.5})
    assert "Portra Look" in Presets.list_presets()
    assert Presets.load_preset("Portra Look") == {"density": 1.5}

    assert Presets.delete_preset("Portra Look") is True
    assert Presets.list_presets() == []


def test_delete_missing_is_false(tmp_path, monkeypatch):
    monkeypatch.setattr(APP_CONFIG, "presets_dir", str(tmp_path))
    assert Presets.delete_preset("Nope") is False
