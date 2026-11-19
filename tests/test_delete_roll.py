"""Delete and Refresh close the open roll when the Library stops listing it: the frames stay
loaded on their shared edits, and a deleted roll's forks leave the edits DB."""

from unittest.mock import MagicMock

from negpy.desktop.controller import AppController
from negpy.services.assets import rolls


def _controller():
    ctrl = AppController.__new__(AppController)
    ctrl.session = MagicMock()
    store: dict = {}
    repo = ctrl.session.repo
    repo.get_global_setting.side_effect = lambda key, default=None: store.get(key, default)
    repo.save_global_setting.side_effect = lambda key, value: store.__setitem__(key, value)
    repo.read_global_setting.side_effect = lambda key, default=None: store.get(key, default)
    ctrl.state = MagicMock()
    ctrl._announce_roll_modes = MagicMock()
    ctrl._reselect_active_frame = MagicMock()
    ctrl.config_updated = MagicMock()
    return ctrl, repo


def test_deleting_the_open_roll_closes_it_before_its_forks_go():
    ctrl, repo = _controller()
    roll_id = rolls.create_virtual_roll(repo, "Picks", ["/a/1.tif", "/a/2.tif"])
    rolls.fork_edit(repo, roll_id, "h1", "/a/1.tif", MagicMock())
    fork = rolls.roll_edit_hash("h1", roll_id)
    ctrl.state.active_roll_id = roll_id
    ctrl.state.current_file_hash = fork
    ctrl.state.uploaded_files = [{"path": "/a/1.tif", "hash": fork}, {"path": "/a/2.tif", "hash": "h2"}]
    order: list = []
    ctrl.session.save_active_edit.side_effect = lambda: order.append("save")
    repo.delete_file_settings.side_effect = lambda h: order.append(("delete", h))

    AppController.delete_rolls(ctrl, [roll_id])

    assert order == ["save", ("delete", fork)]
    assert ctrl.state.active_roll_id is None
    assert [f["hash"] for f in ctrl.state.uploaded_files] == ["h1", "h2"]
    assert rolls.roll_for_id(repo, roll_id) is None
    ctrl._reselect_active_frame.assert_called_once()


def test_deleting_another_roll_leaves_the_open_one():
    ctrl, repo = _controller()
    open_id = rolls.create_virtual_roll(repo, "Open", ["/a/1.tif"])
    other = rolls.create_virtual_roll(repo, "Other", ["/a/1.tif"])
    rolls.fork_edit(repo, open_id, "h1", "/a/1.tif", MagicMock())
    ctrl.state.active_roll_id = open_id
    ctrl.state.uploaded_files = [{"path": "/a/1.tif", "hash": rolls.roll_edit_hash("h1", open_id)}]

    AppController.delete_rolls(ctrl, [other])

    assert ctrl.state.active_roll_id == open_id
    assert ctrl.state.uploaded_files[0]["hash"] == rolls.roll_edit_hash("h1", open_id)
    repo.delete_file_settings.assert_not_called()


def test_refresh_closes_an_open_roll_a_filter_hides_and_keeps_its_forks(tmp_path):
    folder = tmp_path / "scratch" / "roll_b"
    folder.mkdir(parents=True)
    (folder / "1.tif").write_bytes(b"x")
    ctrl, repo = _controller()
    (roll_id,) = rolls.import_subfolders_as_rolls(repo, str(tmp_path))
    rolls.fork_edit(repo, roll_id, "h1", str(folder / "1.tif"), MagicMock())
    ctrl.state.active_roll_id = roll_id
    ctrl.state.current_file_hash = None
    ctrl.state.uploaded_files = [{"path": str(folder / "1.tif"), "hash": rolls.roll_edit_hash("h1", roll_id)}]
    rolls.set_discovery_filters(repo, ["scratch"])

    assert AppController.rediscover_rolls(ctrl) == (0, 1)
    assert ctrl.state.active_roll_id is None
    assert ctrl.state.uploaded_files[0]["hash"] == "h1"
    repo.delete_file_settings.assert_not_called()

    rolls.set_discovery_filters(repo, [])
    assert AppController.rediscover_rolls(ctrl) == (1, 0)
    assert rolls.is_forked(repo, roll_id, "h1")
