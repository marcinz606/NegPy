"""A composite must inherit the film process of the frames it was built from.

A merge or a stitch gets a fresh content hash, so it has no saved edit and hydrates from
defaults plus the *sticky* global process mode. That sticky value is stale whenever the
source frames got their mode from autodetect rather than a manual switch — which is how
merging five E-6 exposures lands you in C41.
"""

import unittest
from dataclasses import replace
from unittest.mock import MagicMock

from negpy.desktop.session import resolve_asset_process_mode
from negpy.domain.models import WorkspaceConfig
from negpy.features.hdr.models import HdrConfig
from negpy.features.process.models import ProcessMode


class TestProcessModeOverlay(unittest.TestCase):
    def test_overlays_the_composites_inherited_mode(self):
        cfg = WorkspaceConfig()  # process_mode defaults to C41
        out = resolve_asset_process_mode(cfg, {"process_mode": "E-6"})
        self.assertEqual(out.process.process_mode, ProcessMode.E6)

    def test_absent_or_empty_leaves_the_config_alone(self):
        cfg = replace(WorkspaceConfig(), process=replace(WorkspaceConfig().process, process_mode=ProcessMode.BW))
        self.assertIs(resolve_asset_process_mode(cfg, {}), cfg)
        self.assertIs(resolve_asset_process_mode(cfg, {"process_mode": ""}), cfg)


class TestHydration(unittest.TestCase):
    """The bug is in the path select_file uses, so assert against that, not the helper."""

    def _session(self, saved_config=None, sticky_mode="C41"):
        from negpy.desktop.session import DesktopSessionManager

        session = DesktopSessionManager.__new__(DesktopSessionManager)
        session.repo = MagicMock()
        session.repo.get_global_setting.side_effect = lambda k, d=None: {"last_process_mode": sticky_mode}.get(k, d)
        session._apply_sticky_settings = lambda c, only_global=False: (
            c if only_global else replace(c, process=replace(c.process, process_mode=sticky_mode))
        )
        import negpy.desktop.session as mod

        self._orig_load = mod.load_or_promote
        mod.load_or_promote = lambda *a, **k: saved_config
        self.addCleanup(lambda: setattr(mod, "load_or_promote", self._orig_load))
        return session

    def test_merged_asset_inherits_e6_over_a_stale_sticky_c41(self):
        """The reported failure: five E-6 exposures merged, composite opened in C41."""
        session = self._session(saved_config=None, sticky_mode="C41")
        asset = {"hash": "merged#hdr", "path": "/x/a.nef", "process_mode": "E-6", "hdr_paths": ("/x/b.nef",), "hdr_ratios": (1.0, 0.5)}
        config, is_new = session._hydrate_asset_config(asset)
        self.assertTrue(is_new)
        self.assertEqual(config.process.process_mode, ProcessMode.E6)

    def test_a_saved_edit_still_wins(self):
        """Inheritance seeds a composite that has no edit of its own; once the user sets a
        mode on it, that must not be overwritten on every reopen."""
        saved = replace(WorkspaceConfig(), process=replace(WorkspaceConfig().process, process_mode=ProcessMode.BW))
        session = self._session(saved_config=saved, sticky_mode="C41")
        asset = {"hash": "merged#hdr", "path": "/x/a.nef", "process_mode": "E-6"}
        config, is_new = session._hydrate_asset_config(asset)
        self.assertFalse(is_new)
        self.assertEqual(config.process.process_mode, ProcessMode.BW)

    def test_hydration_applies_the_bracket(self):
        """Separate bug found alongside: _hydrate_asset_config is what select_file uses to
        build state.config, and the preview task reads state.config.hdr — so omitting
        resolve_asset_hdr there left the on-screen render unmerged while export, which
        composes its params elsewhere, merged correctly."""
        session = self._session(saved_config=None)
        asset = {"hash": "merged#hdr", "path": "/x/a.nef", "hdr_paths": ("/x/b.nef",), "hdr_ratios": (1.0, 0.5), "hdr_align": True}
        config, _ = session._hydrate_asset_config(asset)
        self.assertTrue(config.hdr.hdr_enabled)
        self.assertEqual(config.hdr.hdr_paths, ("/x/b.nef",))
        self.assertEqual(config.hdr.hdr_ratios, (1.0, 0.5))

    def test_a_plain_frame_never_inherits_a_stale_bracket(self):
        session = self._session(saved_config=None)
        config, _ = session._hydrate_asset_config({"hash": "plain", "path": "/x/a.nef"})
        self.assertEqual(config.hdr, HdrConfig())
        self.assertFalse(config.hdr.hdr_enabled)


class TestModeVote(unittest.TestCase):
    """_composite_process_mode takes the majority, not just the primary's."""

    def _controller(self, modes):
        from negpy.desktop.controller import AppController

        ctrl = AppController.__new__(AppController)
        ctrl.session = MagicMock()
        by_hash = {
            f"h{i}": replace(WorkspaceConfig(), process=replace(WorkspaceConfig().process, process_mode=m)) for i, m in enumerate(modes)
        }
        ctrl.session.config_for_asset.side_effect = lambda a: by_hash[a["hash"]]
        return ctrl, [{"hash": f"h{i}"} for i in range(len(modes))]

    def test_unanimous(self):
        ctrl, files = self._controller(["E-6"] * 5)
        self.assertEqual(ctrl._composite_process_mode(files), "E-6")

    def test_one_odd_frame_out_does_not_decide_it(self):
        """A bracket's extreme exposures can autodetect differently — the frame blowing
        46% of its area is not a reliable vote — so the majority decides."""
        ctrl, files = self._controller(["C41", "E-6", "E-6", "E-6", "E-6"])
        self.assertEqual(ctrl._composite_process_mode(files), "E-6")

    def test_ties_go_to_the_reference_frame(self):
        ctrl, files = self._controller(["E-6", "C41"])
        self.assertEqual(ctrl._composite_process_mode(files), "E-6")


if __name__ == "__main__":
    unittest.main()


class TestCompositeDoesNotAdoptItsSourcesEdit(unittest.TestCase):
    """The root cause of both reported symptoms.

    `load_or_promote` falls back to matching by *path*, and a composite's path is its
    reference frame's. So a merge inherited that frame's whole edit — its rotation and its
    film process — and `rehome_file_settings` then *moved* the row, leaving the source
    frame without its own edit. Half-frames were already guarded against exactly this;
    composites were not.
    """

    def _repo(self, frame_edit):
        repo = MagicMock()
        repo.load_file_settings.return_value = None  # the composite's own hash: miss
        repo.load_file_settings_by_path.return_value = ("hash_of_reference_frame", frame_edit)
        return repo

    def _rotated_c41(self):
        base = WorkspaceConfig()
        return replace(
            base,
            process=replace(base.process, process_mode=ProcessMode.C41),
            geometry=replace(base.geometry, rotation=180),
        )

    def test_composite_does_not_inherit_the_reference_frames_edit(self):
        from negpy.services.assets.sidecar import load_or_promote

        repo = self._repo(self._rotated_c41())
        self.assertIsNone(load_or_promote(repo, "merged#hdr", "/scans/_DSC1716.NEF", composite=True))

    def test_composite_does_not_steal_the_reference_frames_edit(self):
        """`rehome_file_settings` moves the row rather than copying it, so the bug did not
        just mislabel the composite — it took the source frame's edit away."""
        from negpy.services.assets.sidecar import load_or_promote

        repo = self._repo(self._rotated_c41())
        load_or_promote(repo, "merged#hdr", "/scans/_DSC1716.NEF", composite=True)
        repo.rehome_file_settings.assert_not_called()

    def test_a_plain_frame_still_gets_the_path_fallback(self):
        """The fallback exists so an EXIF-modified file keeps its edit; only assets whose
        hash is not the identity of the file at that path opt out."""
        from negpy.services.assets.sidecar import load_or_promote

        edit = self._rotated_c41()
        repo = self._repo(edit)
        self.assertIs(load_or_promote(repo, "new_hash", "/scans/_DSC1716.NEF"), edit)
        repo.rehome_file_settings.assert_called_once()

    def test_a_composites_own_saved_edit_is_still_loaded(self):
        """Only the path-keyed fallbacks are skipped — an edit saved under the composite's
        own hash must still come back."""
        from negpy.services.assets.sidecar import load_or_promote

        own = self._rotated_c41()
        repo = MagicMock()
        repo.load_file_settings.return_value = own
        self.assertIs(load_or_promote(repo, "merged#hdr", "/scans/_DSC1716.NEF", composite=True), own)


class TestMergeNaming(unittest.TestCase):
    """A merge's output name must not depend on picture content, and must not collide
    with the export of the source frame it is named after."""

    _FRAMES = ["/x/_DSC1716.NEF", "/x/_DSC1715.NEF", "/x/_DSC1717.NEF", "/x/_DSC1718.NEF", "/x/_DSC1719.NEF"]

    def test_named_from_the_first_frame_not_the_reference(self):
        """The asset's path is the *exposure reference*, chosen from the images (longest
        without clipping), so naming after it made a bracket of 1715..19 export as 1716."""
        from negpy.features.hdr.models import hdr_name, hdr_stem

        self.assertEqual(hdr_stem(self._FRAMES), "_DSC1715")
        self.assertEqual(hdr_name(self._FRAMES), "_DSC1715 +4 (HDR)")

    def test_naming_is_independent_of_the_order_passed(self):
        from negpy.features.hdr.models import hdr_stem

        self.assertEqual(hdr_stem(self._FRAMES), hdr_stem(list(reversed(self._FRAMES))))
        self.assertEqual(hdr_stem(sorted(self._FRAMES)), hdr_stem(self._FRAMES))

    def test_export_filename_suffixes_hdr_so_it_cannot_overwrite_the_source(self):
        from negpy.domain.models import ExportConfig
        from negpy.services.export.templating import render_export_filename

        settings = ExportConfig()
        plain = render_export_filename("/x/_DSC1715.NEF", settings)
        merged = render_export_filename("/x/_DSC1715.NEF", settings, composite="HDR")
        self.assertEqual(merged, f"{plain}-HDR")
        self.assertNotEqual(plain, merged)

    def test_frame_paths_helper_ignores_a_plain_asset(self):
        from negpy.features.hdr.models import hdr_frame_paths

        self.assertEqual(hdr_frame_paths({"path": "/x/a.NEF"}), [])
        self.assertEqual(hdr_frame_paths({"path": "/x/a.NEF", "hdr_paths": ()}), [])
        self.assertEqual(hdr_frame_paths({"path": "/x/a.NEF", "hdr_paths": ("/x/b.NEF",)}), ["/x/a.NEF", "/x/b.NEF"])

    def test_export_naming_end_to_end_uses_the_first_frame_plus_suffix(self):
        """Through resolve_export_naming, which is what both the write and the
        overwrite-conflict check call, so they cannot disagree."""
        from negpy.desktop.workers.export import ExportTask, resolve_export_naming
        from negpy.domain.models import ExportConfig, ExportPresetOutputMode

        cfg = WorkspaceConfig()
        task = ExportTask(
            file_info={"path": "/x/_DSC1716.NEF", "hash": "m#hdr", "hdr_paths": tuple(self._FRAMES[1:])},
            params=cfg,
            export_settings=replace(ExportConfig(), output_mode=ExportPresetOutputMode.SAME_AS_SOURCE),
            metadata_config=cfg.metadata,
        )
        _out_dir, filename, _ext = resolve_export_naming(task)
        self.assertEqual(filename, "_DSC1715-HDR")


class TestHdrSeedHydration(unittest.TestCase):
    """The seed reaches a fresh merge, and only a fresh one."""

    def _session(self, saved_config=None):
        from negpy.desktop.session import DesktopSessionManager

        session = DesktopSessionManager.__new__(DesktopSessionManager)
        session.repo = MagicMock()
        session.repo.get_global_setting.side_effect = lambda k, d=None: d
        session._apply_sticky_settings = lambda c, only_global=False: c
        import negpy.desktop.session as mod

        orig = mod.load_or_promote
        mod.load_or_promote = lambda *a, **k: saved_config
        self.addCleanup(lambda: setattr(mod, "load_or_promote", orig))
        return session

    _ASSET = {
        "hash": "m#hdr",
        "path": "/x/a.nef",
        "hdr_paths": ("/x/b.nef", "/x/c.nef"),
        "hdr_ratios": (1.0, 2.0, 4.0),  # 2 stops of reach
    }

    def test_a_fresh_merge_opens_with_its_shadows_lifted(self):
        config, is_new = self._session()._hydrate_asset_config(self._ASSET)
        self.assertTrue(is_new)
        self.assertAlmostEqual(config.exposure.shadow_density, -0.6, places=5)

    def test_zeroing_the_slider_sticks(self):
        """The whole reason to seed rather than bake in: a user who wants the render that
        is faithful to the metered frame zeroes the slider, and it must stay zeroed."""
        neutral = replace(WorkspaceConfig(), exposure=replace(WorkspaceConfig().exposure, shadow_density=0.0))
        config, is_new = self._session(saved_config=neutral)._hydrate_asset_config(self._ASSET)
        self.assertFalse(is_new)
        self.assertEqual(config.exposure.shadow_density, 0.0)

    def test_a_plain_frame_is_never_seeded(self):
        config, _ = self._session()._hydrate_asset_config({"hash": "plain", "path": "/x/a.nef"})
        self.assertEqual(config.exposure.shadow_density, 0.0)

    def test_a_bracket_that_recovered_nothing_is_not_seeded(self):
        asset = {**self._ASSET, "hdr_ratios": (1.0,), "hdr_paths": ("/x/b.nef",)}
        config, _ = self._session()._hydrate_asset_config(asset)
        self.assertEqual(config.exposure.shadow_density, 0.0)


class TestResetSettingsOnAComposite(unittest.TestCase):
    """Reset Settings has to land where opening the asset fresh does.

    It wrote a bare WorkspaceConfig, which on a merged frame dropped the bracket wiring
    itself — silently un-merging the render — along with the inherited film process and the
    seeded shadow lift, with no supported way back to any of it.
    """

    def _session(self):
        from negpy.desktop.session import DesktopSessionManager

        s = DesktopSessionManager.__new__(DesktopSessionManager)
        s.repo = MagicMock()
        s.repo.get_global_setting.side_effect = lambda k, d=None: d
        s.state = MagicMock()
        s.state.selected_file_idx = 0
        s.state.uploaded_files = [
            {
                "hash": "m#hdr",
                "path": "/x/a.nef",
                "hdr_paths": ("/x/b.nef", "/x/c.nef"),
                "hdr_ratios": (1.0, 2.0, 4.0),
                "hdr_align": True,
                "process_mode": "E-6",
            }
        ]
        s.update_config = MagicMock()
        return s

    def _reset_result(self, session):
        session.reset_settings()
        session.update_config.assert_called_once()
        return session.update_config.call_args.args[0]

    def test_reset_keeps_the_merge_active(self):
        cfg = self._reset_result(self._session())
        self.assertTrue(cfg.hdr.hdr_enabled, "reset un-merged the frame")
        self.assertEqual(cfg.hdr.hdr_paths, ("/x/b.nef", "/x/c.nef"))

    def test_reset_keeps_the_inherited_film_process(self):
        self.assertEqual(self._reset_result(self._session()).process.process_mode, ProcessMode.E6)

    def test_reset_restores_the_seeded_shadow_lift(self):
        """The supported route back to a merge's starting point — which is what made the
        stale-edit trap escapable."""
        self.assertAlmostEqual(self._reset_result(self._session()).exposure.shadow_density, -0.6, places=5)

    def test_reset_on_a_plain_frame_is_still_plain_defaults(self):
        session = self._session()
        session.state.uploaded_files = [{"hash": "plain", "path": "/x/a.nef"}]
        cfg = self._reset_result(session)
        self.assertEqual(cfg.hdr, HdrConfig())
        self.assertEqual(cfg.exposure.shadow_density, 0.0)
        self.assertEqual(cfg.process.process_mode, WorkspaceConfig().process.process_mode)

    def test_reset_with_no_selection_does_not_raise(self):
        session = self._session()
        session.state.selected_file_idx = -1
        self.assertEqual(self._reset_result(session).hdr, HdrConfig())

    def test_reset_matches_what_opening_the_asset_fresh_gives(self):
        """The contract that keeps the two paths from drifting apart."""
        import negpy.desktop.session as mod

        session = self._session()
        asset = session.state.uploaded_files[0]
        orig = mod.load_or_promote
        mod.load_or_promote = lambda *a, **k: None
        try:
            session._apply_sticky_settings = lambda c, only_global=False: c
            fresh, is_new = session._hydrate_asset_config(asset)
        finally:
            mod.load_or_promote = orig
        self.assertTrue(is_new)
        reset = self._reset_result(session)
        self.assertEqual(reset.hdr, fresh.hdr)
        self.assertEqual(reset.process.process_mode, fresh.process.process_mode)
        self.assertEqual(reset.exposure.shadow_density, fresh.exposure.shadow_density)


class TestMergeRefusals(unittest.TestCase):
    """Combinations that build one frame from several files in another way are refused.

    Each of them owns the asset's single primary path, so merging on top of one has no
    defined composition order. A half-frame pair shares its path with its sibling, which
    would also silently collapse to one whole-frame asset.
    """

    def _controller(self, files):
        ctrl = MagicMock()
        ctrl.state.uploaded_files = files
        ctrl.state.selected_indices = list(range(len(files)))
        ctrl._batch_busy.return_value = False
        ctrl._begin_batch.return_value = None  # stops before the worker
        return ctrl

    def _merge(self, ctrl):
        from negpy.desktop.controller import AppController

        AppController.request_hdr_merge_selected(ctrl)

    def _refusal(self, files):
        ctrl = self._controller(files)
        self._merge(ctrl)
        ctrl._begin_batch.assert_not_called()
        return ctrl.set_status.call_args.args[0] if ctrl.set_status.called else ""

    def test_a_lone_frame_is_not_a_bracket(self):
        self.assertIn("two or more", self._refusal([{"path": "/x/a.nef"}]))

    def test_already_merged(self):
        files = [{"path": "/x/a.nef", "hdr_paths": ("/x/b.nef",)}, {"path": "/x/c.nef"}]
        self.assertIn("already-merged", self._refusal(files))

    def test_stitched(self):
        files = [{"path": "/x/a.nef", "stitch_paths": ("/x/b.nef",)}, {"path": "/x/c.nef"}]
        self.assertIn("stitched", self._refusal(files))

    def test_rgb_triplet(self):
        files = [{"path": "/x/a.nef", "green_path": "/x/g.nef"}, {"path": "/x/c.nef"}]
        self.assertIn("RGB-scan", self._refusal(files))

    def test_half_frame(self):
        """Halves share a path, so the by-path dedupe drops one of each pair — merging
        them would quietly produce a whole-frame composite."""
        files = [{"path": "/x/a.nef", "half": 1}, {"path": "/x/a.nef", "half": 2}, {"path": "/x/b.nef", "half": 1}]
        self.assertIn("half-frame", self._refusal(files))

    def test_a_plain_bracket_is_accepted(self):
        ctrl = self._controller([{"path": "/x/a.nef"}, {"path": "/x/b.nef"}])
        self._merge(ctrl)
        ctrl._begin_batch.assert_called_once()


class TestStitchGetsTheSameTreatment(unittest.TestCase):
    """Every fix in this file is keyed on `hdr_paths or stitch_paths`, so a stitch takes
    the same path as a merge. Asserted here rather than inferred: the merge tests above
    would keep passing if the stitch arm of any of those conditions were dropped.
    """

    _ASSET = {"hash": "c#stitch", "path": "/x/left.nef", "stitch_paths": ("/x/right.nef",), "process_mode": "E-6"}

    def _session(self, saved_config=None, sticky_mode="C41"):
        from negpy.desktop.session import DesktopSessionManager

        session = DesktopSessionManager.__new__(DesktopSessionManager)
        session.repo = MagicMock()
        session.repo.get_global_setting.side_effect = lambda k, d=None: d
        session._apply_sticky_settings = lambda c, only_global=False: (
            c if only_global else replace(c, process=replace(c.process, process_mode=sticky_mode))
        )
        import negpy.desktop.session as mod

        orig = mod.load_or_promote
        self.seen = {}
        mod.load_or_promote = lambda *a, **k: (self.seen.update(k), saved_config)[1]
        self.addCleanup(lambda: setattr(mod, "load_or_promote", orig))
        return session

    def test_a_stitch_opts_out_of_the_path_fallback(self):
        """Its path is the primary part's, so a path match hands it that part's edit and
        then moves the row off the part."""
        self._session()._hydrate_asset_config(self._ASSET)
        self.assertTrue(self.seen.get("composite"))

    def test_a_stitch_inherits_its_parts_film_process(self):
        config, is_new = self._session(sticky_mode="C41")._hydrate_asset_config(self._ASSET)
        self.assertTrue(is_new)
        self.assertEqual(config.process.process_mode, ProcessMode.E6)

    def test_a_stitch_is_never_seeded_with_a_shadow_lift(self):
        """The seed is derived from bracket ratios; a stitch has none and must not be
        given a lift because it is a composite."""
        config, _ = self._session()._hydrate_asset_config(self._ASSET)
        self.assertEqual(config.exposure.shadow_density, 0.0)

    def test_reset_keeps_a_stitch_assembled(self):
        session = self._session()
        session.state = MagicMock()
        session.state.selected_file_idx = 0
        session.state.uploaded_files = [
            {
                **self._ASSET,
                "stitch_transforms": ((1.0, 0.0, 0.0, 0.0, 1.0, 0.0),),
                "stitch_canvas": (100, 50),
                "stitch_sizes": ((60, 50),),
            }
        ]
        session.update_config = MagicMock()
        session.reset_settings()
        cfg = session.update_config.call_args.args[0]
        self.assertTrue(cfg.stitch.stitch_enabled, "reset un-stitched the composite")
        self.assertEqual(cfg.stitch.stitch_paths, ("/x/right.nef",))
        self.assertEqual(cfg.process.process_mode, ProcessMode.E6)


class TestMergeIsOfferedOnlyForTransparencies:
    """A bracket buys nothing on a negative.

    Colour negative holds ~5-6 stops between base and Dmax and an ordinary black-and-white
    negative nearer 4, both inside one capture; a transparency runs to 10-12, which is what
    the merge exists for. Hidden on C-41, disabled with a reason on B&W, because reversal
    monochrome (Scala, dr5, Fomapan R) really is a transparency and is simply not wired up.
    """

    def _menu_labels(self, mode):
        from unittest.mock import MagicMock

        from negpy.desktop.view.sidebar.files import FileBrowser

        browser = MagicMock()
        state = MagicMock()
        state.selected_file_idx = 0
        state.uploaded_files = [{"path": "/x/a.nef", "process_mode": mode}]
        menu = MagicMock()
        actions = []

        def add_action(label):
            act = MagicMock()
            act.label = label
            actions.append(act)
            return act

        menu.addAction.side_effect = add_action
        FileBrowser._add_hdr_merge_action(browser, menu, state)
        return actions

    def test_hidden_for_colour_negative(self):
        assert self._menu_labels("C41") == []

    def test_present_for_a_slide(self):
        acts = self._menu_labels("E-6")
        assert [a.label for a in acts] == ["Merge exposures (HDR)"]
        acts[0].setEnabled.assert_not_called()

    def test_disabled_with_a_reason_for_black_and_white(self):
        acts = self._menu_labels("B&W")
        assert [a.label for a in acts] == ["Merge exposures (HDR)"]
        acts[0].setEnabled.assert_called_once_with(False)
        assert "reversal" in acts[0].setToolTip.call_args.args[0]
