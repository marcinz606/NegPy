"""Stack Scans: rescans of one frame averaged through the HDR merge at ratio 1.0, placed on
the reference's canvas, keeping the autos a bracket switches off."""

from dataclasses import replace
from unittest.mock import MagicMock

import numpy as np

from negpy.desktop.session import composite_summary, resolve_asset_hdr
from negpy.domain.models import WorkspaceConfig
from negpy.features.hdr.logic import merge_providers
from negpy.features.hdr.models import HdrConfig, hdr_bracket, hdr_hash, hdr_name
from negpy.services.assets.composites import composite_entry


def _scene(h=240, w=360, seed=0):
    rng = np.random.default_rng(seed)
    base = np.zeros((h, w), np.float32)
    for _ in range(40):
        y, x = rng.integers(0, h), rng.integers(0, w)
        base[max(0, y - 8) : y + 8, max(0, x - 12) : x + 12] += rng.uniform(0.05, 0.2)
    return np.clip(0.2 + base, 0, 0.8)


def _scan(scene, y0, x0, h, w, noise, seed):
    rng = np.random.default_rng(seed)
    crop = scene[y0 : y0 + h, x0 : x0 + w] + rng.normal(0, noise, (h, w)).astype(np.float32)
    return np.repeat(crop[..., None], 3, 2).astype(np.float32)


class TestStackMerge:
    def test_rescans_of_other_sizes_are_placed_and_averaged(self):
        scene = _scene()
        ref = _scan(scene, 20, 20, 200, 300, 0.02, 1)
        others = [_scan(scene, 26, 14, 196, 310, 0.02, 2), _scan(scene, 15, 25, 205, 290, 0.02, 3)]
        merged = merge_providers([lambda f=f: f for f in [ref, *others]], [1.0, 1.0, 1.0], stack=True)
        assert merged.shape == ref.shape
        truth = scene[20:220, 20:320]
        inner = (slice(20, -20), slice(20, -20))
        err_single = np.abs(ref[..., 0] - truth)[inner].mean()
        err_stack = np.abs(merged[..., 0] - truth)[inner].mean()
        assert err_stack < 0.7 * err_single

    def test_canvas_a_scan_does_not_reach_takes_only_the_scans_that_do(self):
        scene = _scene()
        ref = _scan(scene, 20, 20, 200, 300, 0.0, 1)
        narrow = _scan(scene, 20, 20, 200, 150, 0.0, 2) * 0.0 + 0.5  # covers only the left half
        merged = merge_providers([lambda: ref, lambda: narrow], [1.0, 1.0], align=False, stack=True)
        np.testing.assert_allclose(merged[:, 200:], ref[:, 200:], atol=1e-6)


class TestStackConfig:
    def test_a_stack_keeps_the_autos_and_a_bracket_does_not(self):
        base = WorkspaceConfig()
        autos = replace(base.exposure, auto_exposure=True, auto_normalize_contrast=True)
        stack = HdrConfig(hdr_enabled=True, hdr_paths=("/b.tif",), hdr_ratios=(1.0, 1.0), hdr_stack=True)
        bracket = replace(stack, hdr_stack=False)
        assert WorkspaceConfig(exposure=autos, hdr=stack).exposure.auto_exposure
        assert not WorkspaceConfig(exposure=autos, hdr=bracket).exposure.auto_exposure
        assert not hdr_bracket(stack) and hdr_bracket(bracket)

    def test_a_stack_keeps_highlight_reconstruction_and_a_bracket_does_not(self):
        base = WorkspaceConfig()
        process = replace(base.process, highlight_reconstruction=2)
        stack = HdrConfig(hdr_enabled=True, hdr_paths=("/b.tif",), hdr_ratios=(1.0, 1.0), hdr_stack=True)
        assert WorkspaceConfig(process=process, hdr=stack).process.highlight_reconstruction == 2
        assert WorkspaceConfig(process=process, hdr=replace(stack, hdr_stack=False)).process.highlight_reconstruction == 0

    def test_flat_round_trip(self):
        cfg = WorkspaceConfig(hdr=HdrConfig(hdr_enabled=True, hdr_paths=("/b.tif",), hdr_ratios=(1.0, 1.0), hdr_stack=True))
        assert WorkspaceConfig.from_flat_dict(cfg.to_dict()).hdr.hdr_stack


class TestStackAsset:
    def _asset(self):
        return {
            "path": "/r/a.tif",
            "hash": hdr_hash(["h1", "h2", "h3"], stack=True),
            "hdr_paths": ("/r/b.tif", "/r/c.tif"),
            "hdr_ratios": (1.0, 1.0, 1.0),
            "hdr_stack": True,
        }

    def test_name_hash_and_summary_say_stack(self):
        asset = self._asset()
        assert hdr_name(["/r/a.tif", "/r/b.tif", "/r/c.tif"], stack=True) == "a +2 (Stack)"
        assert asset["hash"].endswith("#stack") and asset["hash"].split("#")[0] == hdr_hash(["h1", "h2", "h3"]).split("#")[0]
        assert composite_summary(asset) == "Stack of 3 scans"

    def test_the_flag_reaches_the_config_and_the_store(self):
        asset = self._asset()
        assert resolve_asset_hdr(WorkspaceConfig(), asset).hdr.hdr_stack
        assert composite_entry(asset)["stack"] is True


class TestStackFrames:
    def _controller(self, files):
        from negpy.desktop.controller import AppController

        ctrl = MagicMock()
        ctrl.state.uploaded_files = files
        ctrl._composite_process_mode.return_value = "C41"
        return ctrl, AppController.stack_frames.__get__(ctrl)

    def test_builds_a_stack_on_the_chosen_reference(self):
        files = [{"path": f"/r/{n}.tif", "hash": f"h{n}"} for n in "abc"]
        ctrl, stack = self._controller(files)
        assert stack(["/r/a.tif", "/r/b.tif", "/r/c.tif"], reference="/r/b.tif")
        indices, composite = ctrl.session.apply_composite.call_args.args
        assert sorted(indices) == [0, 1, 2]
        assert composite["path"] == "/r/b.tif"
        assert composite["hdr_paths"] == ("/r/a.tif", "/r/c.tif")
        assert composite["hdr_ratios"] == (1.0, 1.0, 1.0)
        assert composite["hdr_stack"] is True
        assert composite["name"] == "a +2 (Stack)"

    def test_refuses_a_composite_or_a_lone_frame(self):
        ctrl, stack = self._controller([{"path": "/r/a.tif", "hash": "h", "hdr_paths": ("/r/x.tif",)}, {"path": "/r/b.tif", "hash": "g"}])
        assert not stack(["/r/a.tif", "/r/b.tif"])
        assert not stack(["/r/b.tif"])
        ctrl.session.apply_composite.assert_not_called()


class TestStackElsewhere:
    def _controller(self, files):
        from negpy.desktop.controller import AppController

        ctrl = MagicMock()
        ctrl.state.uploaded_files = files
        ctrl.state.selected_indices = list(range(len(files)))
        ctrl._batch_busy.return_value = False
        return ctrl, AppController

    def test_stitch_refuses_a_stack(self):
        files = [{"path": "/r/a.tif", "hash": "s", "hdr_paths": ("/r/b.tif",), "hdr_stack": True}, {"path": "/r/c.tif", "hash": "c"}]
        ctrl, cls = self._controller(files)
        cls.request_stitch_selected.__get__(ctrl)()
        ctrl.stitch_requested.emit.assert_not_called()
        assert "stacked" in ctrl.set_status.call_args.args[0]

    def test_merge_to_tiff_names_a_stack_as_a_stack(self):
        files = [{"name": "a +1 (Stack)", "path": "/r/a.tif", "hash": "s", "hdr_paths": ("/r/b.tif",), "hdr_stack": True}]
        ctrl, cls = self._controller(files)
        mergeable, skipped = cls.frame_merge_plan.__get__(ctrl)()
        assert mergeable == [] and skipped == ["a +1 (Stack): a stack cannot merge"]
