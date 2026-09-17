"""Per-roll edit fork: an opt-in, per-frame variant of the half-frame hash-suffix trick
(``#roll:<id>`` instead of ``#<half>``), so a photo shared by two rolls can have its own
edit under one of them without disturbing the shared edit every other roll still uses."""

from dataclasses import replace
from unittest.mock import MagicMock

from negpy.domain.models import WorkspaceConfig
from negpy.services.assets import rolls


def _cfg(density: float) -> WorkspaceConfig:
    c = WorkspaceConfig()
    return replace(c, exposure=replace(c.exposure, density=density))


def _controller(roll_entries: dict):
    """A bare AppController (no __init__) with a dict-backed global-settings store, so a
    roll written in one call is readable back in the next -- same pattern as
    tests/test_rolls.py's own repo fixture."""
    from negpy.desktop.controller import AppController

    ctrl = AppController.__new__(AppController)
    ctrl.session = MagicMock()
    settings_store = {rolls.ROLLS_KEY: roll_entries}
    ctrl.session.repo.get_global_setting.side_effect = lambda key, default=None: settings_store.get(key, default)
    ctrl.session.repo.save_global_setting.side_effect = lambda key, value: settings_store.__setitem__(key, value)
    ctrl.set_status = MagicMock()
    ctrl.load_file = MagicMock()
    return ctrl


class TestApplyRollForks:
    def test_rewrites_the_hash_of_a_forked_asset(self):
        from negpy.desktop.controller import AppController

        rolls_store = {"r1": {"kind": "virtual", "name": "A", "member_paths": [], "forked_hashes": ["ha"]}}
        ctrl = _controller(rolls_store)
        ctrl.state = MagicMock()
        ctrl.state.active_roll_id = "r1"
        assets = [{"path": "/p/a.tif", "hash": "ha"}, {"path": "/p/b.tif", "hash": "hb"}]

        AppController._apply_roll_forks(ctrl, assets)

        assert assets[0]["hash"] == "ha#roll:r1"
        assert assets[1]["hash"] == "hb"  # not forked, untouched

    def test_no_active_roll_leaves_hashes_alone(self):
        from negpy.desktop.controller import AppController

        rolls_store = {"r1": {"kind": "virtual", "name": "A", "member_paths": [], "forked_hashes": ["ha"]}}
        ctrl = _controller(rolls_store)
        ctrl.state = MagicMock()
        ctrl.state.active_roll_id = None
        assets = [{"path": "/p/a.tif", "hash": "ha"}]

        AppController._apply_roll_forks(ctrl, assets)

        assert assets[0]["hash"] == "ha"

    def test_preserves_a_half_frame_suffix(self):
        """Each half of a shared diptych scan forks to its own identity."""
        from negpy.desktop.controller import AppController

        rolls_store = {"r1": {"kind": "virtual", "name": "A", "member_paths": [], "forked_hashes": ["ha#1"]}}
        ctrl = _controller(rolls_store)
        ctrl.state = MagicMock()
        ctrl.state.active_roll_id = "r1"
        assets = [{"path": "/p/a.tif", "hash": "ha#1", "half": 1}, {"path": "/p/a.tif", "hash": "ha#2", "half": 2}]

        AppController._apply_roll_forks(ctrl, assets)

        assert assets[0]["hash"] == "ha#1#roll:r1"
        assert assets[1]["hash"] == "ha#2"  # half 2 was never forked, stays shared


class TestRequestForkEditForRoll:
    def test_forks_the_active_frame_seeded_from_the_live_in_memory_config(self):
        from negpy.desktop.controller import AppController

        rolls_store = {"r1": {"kind": "virtual", "name": "A", "member_paths": ["/p/a.tif"]}}
        ctrl = _controller(rolls_store)
        asset = {"path": "/p/a.tif", "hash": "ha"}
        ctrl.state = MagicMock()
        ctrl.state.selected_file_idx = 0
        ctrl.state.uploaded_files = [asset]
        ctrl.state.active_roll_id = "r1"
        ctrl.state.current_file_hash = "ha"
        ctrl.state.config = _cfg(1.5)

        AppController.request_fork_edit_for_roll(ctrl)

        assert asset["hash"] == "ha#roll:r1"
        assert rolls.is_forked(ctrl.session.repo, "r1", "ha")
        ctrl.session.repo.save_file_settings.assert_called_once_with("ha#roll:r1", ctrl.state.config, file_path="/p/a.tif")
        ctrl.load_file.assert_called_once_with("/p/a.tif")
        ctrl.set_status.assert_called_once()

    def test_forking_a_non_active_frame_seeds_from_its_own_stored_config(self):
        from negpy.desktop.controller import AppController

        rolls_store = {"r1": {"kind": "virtual", "name": "A", "member_paths": ["/p/a.tif", "/p/b.tif"]}}
        ctrl = _controller(rolls_store)
        asset = {"path": "/p/b.tif", "hash": "hb"}
        ctrl.state = MagicMock()
        ctrl.state.selected_file_idx = 0
        ctrl.state.uploaded_files = [asset]
        ctrl.state.active_roll_id = "r1"
        ctrl.state.current_file_hash = "ha"  # a different frame is the one on screen
        stored = _cfg(0.7)
        ctrl.session.config_for_asset.return_value = stored

        AppController.request_fork_edit_for_roll(ctrl)

        ctrl.session.config_for_asset.assert_called_once_with(asset)
        ctrl.session.repo.save_file_settings.assert_called_once_with("hb#roll:r1", stored, file_path="/p/b.tif")
        ctrl.load_file.assert_not_called()  # a frame that is not on screen does not reload

    def test_no_active_roll_does_nothing(self):
        from negpy.desktop.controller import AppController

        ctrl = _controller({})
        asset = {"path": "/p/a.tif", "hash": "ha"}
        ctrl.state = MagicMock()
        ctrl.state.selected_file_idx = 0
        ctrl.state.uploaded_files = [asset]
        ctrl.state.active_roll_id = None

        AppController.request_fork_edit_for_roll(ctrl)

        assert asset["hash"] == "ha"
        ctrl.session.repo.save_file_settings.assert_not_called()


class TestRequestUnforkEditForRoll:
    def test_reverts_to_the_shared_edit_and_deletes_the_fork(self):
        from negpy.desktop.controller import AppController

        rolls_store = {"r1": {"kind": "virtual", "name": "A", "member_paths": ["/p/a.tif"], "forked_hashes": ["ha"]}}
        ctrl = _controller(rolls_store)
        asset = {"path": "/p/a.tif", "hash": "ha#roll:r1"}
        ctrl.state = MagicMock()
        ctrl.state.selected_file_idx = 0
        ctrl.state.uploaded_files = [asset]
        ctrl.state.active_roll_id = "r1"
        ctrl.state.current_file_hash = "ha#roll:r1"

        AppController.request_unfork_edit_for_roll(ctrl)

        assert asset["hash"] == "ha"
        assert not rolls.is_forked(ctrl.session.repo, "r1", "ha")
        ctrl.session.repo.delete_file_settings.assert_called_once_with("ha#roll:r1")
        ctrl.load_file.assert_called_once_with("/p/a.tif")
        ctrl.set_status.assert_called_once()

    def test_a_frame_with_no_fork_is_a_noop(self):
        from negpy.desktop.controller import AppController

        ctrl = _controller({"r1": {"kind": "virtual", "name": "A", "member_paths": ["/p/a.tif"]}})
        asset = {"path": "/p/a.tif", "hash": "ha"}
        ctrl.state = MagicMock()
        ctrl.state.selected_file_idx = 0
        ctrl.state.uploaded_files = [asset]
        ctrl.state.active_roll_id = "r1"

        AppController.request_unfork_edit_for_roll(ctrl)

        assert asset["hash"] == "ha"
        ctrl.session.repo.delete_file_settings.assert_not_called()
