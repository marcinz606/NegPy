import json
from types import SimpleNamespace
from unittest.mock import patch

from negpy.infrastructure.storage.repository import StorageRepository
from negpy.services.assets import rolls


def _repo(tmp_path) -> StorageRepository:
    repo = StorageRepository(str(tmp_path / "edits.db"), str(tmp_path / "settings.db"))
    repo.initialize()
    return repo


def test_roll_readers_parse_the_store_once_per_write(tmp_path) -> None:
    repo = _repo(tmp_path)
    roll_id = rolls.create_virtual_roll(repo, "Portra", [])
    rolls.set_roll_defaults(repo, roll_id, film_type="c41")

    parses = []

    def loads(raw, *a, **k):
        parses.append(raw)
        return json.loads(raw, *a, **k)

    with patch("negpy.infrastructure.storage.repository.json", SimpleNamespace(loads=loads, dumps=json.dumps)):
        for _ in range(20):
            assert rolls.roll_defaults(repo, roll_id) == {"film_type": "c41"}
            rolls.roll_scenes(repo, roll_id)
            rolls.is_forked(repo, roll_id, "h")
    assert len(parses) == 1

    rolls.set_roll_defaults(repo, roll_id, film_type="bw")
    assert rolls.roll_defaults(repo, roll_id) == {"film_type": "bw"}


def test_a_writer_never_changes_what_a_reader_holds(tmp_path) -> None:
    repo = _repo(tmp_path)
    roll_id = rolls.create_virtual_roll(repo, "Portra", [])
    before = rolls.roll_for_id(repo, roll_id)
    rolls.rename_roll(repo, roll_id, "Ektar")
    assert before["name"] == "Portra"
    assert rolls.roll_for_id(repo, roll_id)["name"] == "Ektar"
