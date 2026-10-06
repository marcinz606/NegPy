"""Find Duplicates: grouping frames scanned more than once, the review's defaults, and the
decisions it hands the controller."""

from unittest.mock import MagicMock, patch

import cv2
import numpy as np

from negpy.services.assets.duplicates import DuplicateGroup, find_groups, fingerprint


def _scene(seed, h=600, w=900):
    rng = np.random.default_rng(seed)
    img = np.full((h, w), 0.3, np.float32)
    for _ in range(60):
        y, x = rng.integers(0, h), rng.integers(0, w)
        img[max(0, y - 20) : y + 20, max(0, x - 30) : x + 30] += rng.uniform(0.05, 0.3)
    return cv2.GaussianBlur(img, (0, 0), 2)


def _rescan(scene, dy, dx, noise, seed):
    rng = np.random.default_rng(seed)
    crop = scene[20 + dy : 560 + dy, 30 + dx : 850 + dx]
    return np.clip(crop + rng.normal(0, noise, crop.shape), 0, 1).astype(np.float32)


def _toned(img):
    """A frame with a full tonal range, dark at one edge and bright at the other."""
    return (img * np.linspace(0.05, 0.9, img.shape[1], dtype=np.float32)[None, :]).astype(np.float32)


def _display_u8(linear):
    return (np.clip(linear, 0, 1) ** (1 / 2.2) * 255).round().astype(np.uint8)


class TestGrouping:
    def test_rescans_group_and_other_frames_stay_apart(self):
        a, b = _scene(1), _scene(2)
        prints = {
            "/r/a1.tif": fingerprint(_rescan(a, 0, 0, 0.05, 1)),
            "/r/a2.tif": fingerprint(_rescan(a, 6, -9, 0.05, 2)),
            "/r/a3.tif": fingerprint(_rescan(a, -5, 7, 0.05, 3)),
            "/r/b1.tif": fingerprint(_rescan(b, 0, 0, 0.01, 4)),
        }
        groups = find_groups(prints)
        assert [g.paths for g in groups] == [("/r/a1.tif", "/r/a2.tif", "/r/a3.tif")]
        assert not groups[0].same_scan

    def test_a_resized_copy_is_the_same_scan(self):
        scan = _rescan(_scene(3), 0, 0, 0.01, 1)
        copy = cv2.resize(scan, (scan.shape[1] // 2, scan.shape[0] // 2), interpolation=cv2.INTER_AREA)
        groups = find_groups({"/r/x.tif": fingerprint(scan), "/r/x.jpg": fingerprint(copy)})
        assert len(groups) == 1 and groups[0].same_scan

    def test_exposures_of_one_bracket_stay_apart(self):
        scan = _toned(_rescan(_scene(5), 0, 0, 0.005, 1))
        for encode in (lambda x: x, _display_u8):
            for ev in (-2, -1, 1, 2):
                other = np.clip(scan * 2.0**ev, 0, 1)
                assert find_groups({"/r/a.tif": fingerprint(encode(scan)), "/r/b.tif": fingerprint(encode(other))}) == [], ev

    def test_rescans_a_little_apart_in_level_still_group(self):
        scene = _scene(6)
        a = _toned(_rescan(scene, 0, 0, 0.01, 1))
        b = _toned(_rescan(scene, 4, -3, 0.01, 2)) * 1.1
        assert len(find_groups({"/r/a.tif": fingerprint(_display_u8(a)), "/r/b.tif": fingerprint(_display_u8(b))})) == 1

    def test_a_gamma_encoded_copy_is_the_same_scan(self):
        scan = _toned(_rescan(_scene(7), 0, 0, 0.005, 1))
        linear = (scan * 65535).astype(np.uint16)
        groups = find_groups({"/r/x.tif": fingerprint(linear), "/r/x.jpg": fingerprint(_display_u8(scan))})
        assert len(groups) == 1 and groups[0].same_scan

    def test_a_different_crop_shape_never_matches(self):
        scan = _rescan(_scene(4), 0, 0, 0.01, 1)
        assert find_groups({"/r/a.tif": fingerprint(scan), "/r/b.tif": fingerprint(scan[:, :300])}) == []


class TestReviewDefaults:
    def test_rescans_stack_and_copies_or_mixed_types_reject(self):
        from negpy.desktop.view.widgets.duplicates_dialog import default_action

        assert default_action(DuplicateGroup(("/r/a.tif", "/r/b.tif"), 0.97)) == "stack"
        assert default_action(DuplicateGroup(("/r/a.tif", "/r/b.tif"), 0.999)) == "reject"
        assert default_action(DuplicateGroup(("/r/a.tif", "/r/a.jpg"), 0.97)) == "reject"

    def test_keeps_the_largest_file(self, tmp_path):
        from negpy.desktop.view.widgets.duplicates_dialog import default_keep

        small, big = tmp_path / "a.jpg", tmp_path / "a.tif"
        small.write_bytes(b"x" * 10)
        big.write_bytes(b"x" * 100)
        assert default_keep([str(small), str(big)]) == str(big)


class TestDecisions:
    def _controller(self):
        from negpy.desktop.controller import AppController

        ctrl = MagicMock()
        ctrl.stack_frames.return_value = True
        ctrl.session.mark_paths.return_value = 1
        ctrl.session.remove_paths.side_effect = lambda paths: len(paths)
        return ctrl, AppController.apply_duplicate_decisions.__get__(ctrl)

    def test_each_action_reaches_the_frames_not_kept(self, tmp_path):
        trash_me = tmp_path / "c2.tif"
        trash_me.write_bytes(b"x")
        ctrl, apply = self._controller()
        with patch("negpy.desktop.controller._move_to_trash", return_value=True) as moved:
            apply(
                [
                    (("/r/a1.tif", "/r/a2.tif"), "/r/a2.tif", "stack"),
                    (("/r/b1.tif", "/r/b2.tif"), "/r/b1.tif", "reject"),
                    (("/r/c1.tif", str(trash_me)), "/r/c1.tif", "trash"),
                    (("/r/d1.tif", "/r/d2.tif"), "/r/d1.tif", "skip"),
                ]
            )
        ctrl.stack_frames.assert_called_once_with(["/r/a1.tif", "/r/a2.tif"], reference="/r/a2.tif")
        ctrl.session.mark_paths.assert_called_once_with(["/r/b2.tif"], "excluded")
        moved.assert_called_once_with(str(trash_me))
        ctrl.session.remove_paths.assert_called_once_with([str(trash_me)])
