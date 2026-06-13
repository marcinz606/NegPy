import os

from negpy.kernel.system.config import APP_CONFIG
from negpy.services.assets.crosstalk import CrosstalkProfiles


def _write(path, content):
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def test_list_and_get_custom(tmp_path, monkeypatch):
    monkeypatch.setattr(APP_CONFIG, "crosstalk_dir", str(tmp_path))

    _write(
        os.path.join(tmp_path, "portra.toml"),
        'name = "Portra 400"\nmatrix = [[1.0, -0.1, 0.0], [0.0, 1.0, -0.1], [0.0, 0.0, 1.0]]\n',
    )

    assert CrosstalkProfiles.list_profiles() == ["Default", "Portra 400"]
    assert CrosstalkProfiles.get_matrix("Portra 400") == [1.0, -0.1, 0.0, 0.0, 1.0, -0.1, 0.0, 0.0, 1.0]


def test_name_falls_back_to_stem(tmp_path, monkeypatch):
    monkeypatch.setattr(APP_CONFIG, "crosstalk_dir", str(tmp_path))
    _write(
        os.path.join(tmp_path, "my_film.toml"),
        "matrix = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]\n",
    )
    assert "my_film" in CrosstalkProfiles.list_profiles()


def test_default_returns_none(tmp_path, monkeypatch):
    monkeypatch.setattr(APP_CONFIG, "crosstalk_dir", str(tmp_path))
    assert CrosstalkProfiles.get_matrix("Default") is None
    assert CrosstalkProfiles.get_matrix("nonexistent") is None


def test_malformed_skipped(tmp_path, monkeypatch):
    monkeypatch.setattr(APP_CONFIG, "crosstalk_dir", str(tmp_path))
    _write(os.path.join(tmp_path, "bad_shape.toml"), "matrix = [[1.0, 0.0], [0.0, 1.0]]\n")
    _write(os.path.join(tmp_path, "bad_toml.toml"), "matrix = [[[not valid\n")
    _write(os.path.join(tmp_path, "no_matrix.toml"), 'name = "x"\n')
    assert CrosstalkProfiles.list_profiles() == ["Default"]


def test_seed_example(tmp_path, monkeypatch):
    user_dir = tmp_path / "user"
    bundled_dir = tmp_path / "bundled"
    user_dir.mkdir()
    bundled_dir.mkdir()
    _write(
        os.path.join(bundled_dir, "example.toml"),
        'name = "Example"\nmatrix = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]\n',
    )
    _write(os.path.join(bundled_dir, "README.md"), "not a matrix\n")

    monkeypatch.setattr(APP_CONFIG, "crosstalk_dir", str(user_dir))
    monkeypatch.setattr("negpy.services.assets.crosstalk.get_resource_path", lambda _: str(bundled_dir))

    CrosstalkProfiles.seed_example()
    tomls = [f for f in os.listdir(user_dir) if f.endswith(".toml")]
    assert tomls == ["example.toml"]  # README.md not copied
    assert CrosstalkProfiles.list_profiles() == ["Default", "Example"]

    # Second call is a no-op once a file exists.
    CrosstalkProfiles.seed_example()
    assert [f for f in os.listdir(user_dir) if f.endswith(".toml")] == tomls
