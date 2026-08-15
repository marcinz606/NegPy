import os

import pytest

from negpy.features.process.models import ProcessMode
from negpy.kernel.system.config import APP_CONFIG
from negpy.services.assets.crosstalk import CrosstalkProfiles


def _write(path, content):
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


@pytest.fixture(autouse=True)
def _isolate_bundled(tmp_path, monkeypatch):
    monkeypatch.setattr("negpy.services.assets.crosstalk.get_resource_path", lambda _: str(tmp_path / "_no_bundled"))


def test_list_and_get_custom(tmp_path, monkeypatch):
    monkeypatch.setattr(APP_CONFIG, "crosstalk_dir", str(tmp_path))

    _write(
        os.path.join(tmp_path, "portra.toml"),
        'name = "Portra 400"\nmatrix = [[1.0, -0.1, 0.0], [0.0, 1.0, -0.1], [0.0, 0.0, 1.0]]\n',
    )

    assert CrosstalkProfiles.list_profiles() == ["Generic C41", "Portra 400"]
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
    assert CrosstalkProfiles.get_matrix("Generic C41") is None
    assert CrosstalkProfiles.get_matrix("nonexistent") is None


def test_malformed_skipped(tmp_path, monkeypatch):
    monkeypatch.setattr(APP_CONFIG, "crosstalk_dir", str(tmp_path))
    _write(os.path.join(tmp_path, "bad_shape.toml"), "matrix = [[1.0, 0.0], [0.0, 1.0]]\n")
    _write(os.path.join(tmp_path, "bad_toml.toml"), "matrix = [[[not valid\n")
    _write(os.path.join(tmp_path, "no_matrix.toml"), 'name = "x"\n')
    assert CrosstalkProfiles.list_profiles() == ["Generic C41"]


def test_ensure_user_dir_creates_directory(tmp_path, monkeypatch):
    target = tmp_path / "nested" / "crosstalk"
    monkeypatch.setattr(APP_CONFIG, "crosstalk_dir", str(target))

    assert not target.exists()

    CrosstalkProfiles.ensure_user_dir()

    assert target.is_dir()


def test_list_profiles_merges_bundled_and_user(tmp_path, monkeypatch):
    user_dir = tmp_path / "user"
    bundled_dir = tmp_path / "bundled"
    user_dir.mkdir()
    bundled_dir.mkdir()
    _write(
        os.path.join(bundled_dir, "portra_400.toml"),
        'name = "Portra 400"\nmatrix = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]\n',
    )
    _write(
        os.path.join(user_dir, "my_film.toml"),
        'name = "My Film"\nmatrix = [[0.9, 0.1, 0.0], [0.0, 0.9, 0.1], [0.0, 0.0, 0.9]]\n',
    )

    monkeypatch.setattr(APP_CONFIG, "crosstalk_dir", str(user_dir))
    monkeypatch.setattr("negpy.services.assets.crosstalk.get_resource_path", lambda _: str(bundled_dir))

    assert CrosstalkProfiles.list_profiles() == ["Generic C41", "My Film", "Portra 400"]
    assert CrosstalkProfiles.get_matrix("Portra 400") == [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    assert CrosstalkProfiles.get_matrix("My Film") == [0.9, 0.1, 0.0, 0.0, 0.9, 0.1, 0.0, 0.0, 0.9]


def test_save_round_trips_through_get_matrix(tmp_path, monkeypatch):
    monkeypatch.setattr(APP_CONFIG, "crosstalk_dir", str(tmp_path))
    matrix = [1.004, -0.02, 0.001, -0.118, 1.01, -0.04, -0.042, -0.149, 1.006]

    path = CrosstalkProfiles.save("My Film", matrix)

    assert os.path.isfile(path)
    assert path.startswith(str(tmp_path))
    assert CrosstalkProfiles.get_matrix("My Film") == matrix
    assert "My Film" in CrosstalkProfiles.list_profiles()


def test_is_bundled_distinguishes_origin(tmp_path, monkeypatch):
    user_dir = tmp_path / "user"
    bundled_dir = tmp_path / "bundled"
    user_dir.mkdir()
    bundled_dir.mkdir()
    _write(
        os.path.join(bundled_dir, "portra.toml"),
        'name = "Portra 400"\nmatrix = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]\n',
    )
    monkeypatch.setattr(APP_CONFIG, "crosstalk_dir", str(user_dir))
    monkeypatch.setattr("negpy.services.assets.crosstalk.get_resource_path", lambda _: str(bundled_dir))
    CrosstalkProfiles.save("My Film", [1, 0, 0, 0, 1, 0, 0, 0, 1])

    assert CrosstalkProfiles.is_bundled("Generic C41")
    assert CrosstalkProfiles.is_bundled("Portra 400")
    assert not CrosstalkProfiles.is_bundled("My Film")


def test_delete_removes_only_matching_display_name(tmp_path, monkeypatch):
    monkeypatch.setattr(APP_CONFIG, "crosstalk_dir", str(tmp_path))
    CrosstalkProfiles.save("Alpha", [1, 0, 0, 0, 1, 0, 0, 0, 1])
    CrosstalkProfiles.save("Beta", [1, 0, 0, 0, 1, 0, 0, 0, 1])

    CrosstalkProfiles.delete("Alpha")

    assert CrosstalkProfiles.get_matrix("Alpha") is None
    assert CrosstalkProfiles.get_matrix("Beta") is not None
    CrosstalkProfiles.delete("Nonexistent")  # no-op, no raise


def test_dialog_pure_helpers():
    from negpy.desktop.view.widgets.crosstalk_editor_dialog import (
        flat_to_grid,
        grid_to_flat,
        unique_copy_name,
    )

    flat = [1.0, -0.05, -0.02, -0.04, 1.0, -0.08, -0.01, -0.1, 1.0]
    assert grid_to_flat(flat_to_grid(flat)) == flat
    assert flat_to_grid(flat)[1][2] == -0.08

    assert unique_copy_name("Portra", []) == "Portra Copy"
    assert unique_copy_name("Portra", ["Portra Copy"]) == "Portra Copy 2"
    assert unique_copy_name("Portra", ["Portra Copy", "Portra Copy 2"]) == "Portra Copy 3"


def test_bundled_wins_on_name_collision_dedup(tmp_path, monkeypatch):
    user_dir = tmp_path / "user"
    bundled_dir = tmp_path / "bundled"
    user_dir.mkdir()
    bundled_dir.mkdir()
    _write(
        os.path.join(bundled_dir, "portra_400.toml"),
        'name = "Portra 400"\nmatrix = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]\n',
    )
    _write(
        os.path.join(user_dir, "old_seeded_copy.toml"),
        'name = "Portra 400"\nmatrix = [[9.0, 9.0, 9.0], [9.0, 9.0, 9.0], [9.0, 9.0, 9.0]]\n',
    )

    monkeypatch.setattr(APP_CONFIG, "crosstalk_dir", str(user_dir))
    monkeypatch.setattr("negpy.services.assets.crosstalk.get_resource_path", lambda _: str(bundled_dir))

    assert CrosstalkProfiles.list_profiles() == ["Generic C41", "Portra 400"]
    assert CrosstalkProfiles.get_matrix("Portra 400") == [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]


def test_type_groups_profiles_for_the_dropdown(tmp_path, monkeypatch):
    """A datasheet estimate must not share a bucket with one measured on a real rig."""
    monkeypatch.setattr(APP_CONFIG, "crosstalk_dir", str(tmp_path))
    for fname, name, ptype in (
        ("sheet.toml", "Sheet Stock", "specsheet-based"),
        ("rig.toml", "My Rig", "tuned"),
        ("meas.toml", "Measured Stock", "measured"),
    ):
        _write(
            os.path.join(tmp_path, fname),
            f'name = "{name}"\ntype = "{ptype}"\nmatrix = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]\n',
        )

    groups = CrosstalkProfiles.grouped_profiles()
    assert [h for h, _ in groups] == ["Built-in", "Measured", "Tuned on a rig", "From spec sheets (approx)"]
    assert dict(groups)["Built-in"] == [CrosstalkProfiles.DEFAULT_NAME]
    assert dict(groups)["Measured"] == ["Measured Stock"]
    assert dict(groups)["From spec sheets (approx)"] == ["Sheet Stock"]
    assert CrosstalkProfiles.get_type("My Rig") == "tuned"


def test_grouping_never_drops_a_profile(tmp_path, monkeypatch):
    """An unrecognised or missing type falls through to "Other" instead of vanishing."""
    monkeypatch.setattr(APP_CONFIG, "crosstalk_dir", str(tmp_path))
    _write(
        os.path.join(tmp_path, "weird.toml"),
        'name = "Weird"\ntype = "handed-down-by-owls"\nmatrix = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]\n',
    )
    _write(
        os.path.join(tmp_path, "bare.toml"),
        'name = "Bare"\nmatrix = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]\n',
    )

    groups = CrosstalkProfiles.grouped_profiles()
    flat = [n for _h, names in groups for n in names]
    assert sorted(flat) == sorted(CrosstalkProfiles.list_profiles()), "grouping lost or invented a profile"
    assert dict(groups)["Other"] == ["Bare", "Weird"]
    assert CrosstalkProfiles.get_matrix("Weird") is not None


def test_saved_profiles_are_marked_tuned(tmp_path, monkeypatch):
    """Anything saved from the editor was dialled in on a real rig, so it must not be grouped
    with the bundled spec-sheet estimates."""
    monkeypatch.setattr(APP_CONFIG, "crosstalk_dir", str(tmp_path))
    CrosstalkProfiles.save("Mine", [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
    assert CrosstalkProfiles.get_type("Mine") == "tuned"
    assert dict(CrosstalkProfiles.grouped_profiles())["Tuned on a rig"] == ["Mine"]


def test_legacy_process_names_still_match_their_mode(tmp_path, monkeypatch):
    """A matrix written before the mode rename declares `process = "E-6"`. It must still
    reach the Transparency dropdown instead of matching no mode and disappearing."""
    monkeypatch.setattr(APP_CONFIG, "crosstalk_dir", str(tmp_path))
    _write(
        os.path.join(tmp_path, "slide.toml"),
        'name = "Old Slide"\nprocess = "E-6"\nmatrix = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]\n',
    )

    assert CrosstalkProfiles.get_process("Old Slide") == ProcessMode.E6
    flat = [n for _h, names in CrosstalkProfiles.grouped_profiles(ProcessMode.E6) for n in names]
    assert flat == ["Old Slide"]
    assert CrosstalkProfiles.grouped_profiles(ProcessMode.C41) == [("Built-in", [CrosstalkProfiles.DEFAULT_NAME])]
