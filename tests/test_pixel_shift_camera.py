"""PixelShiftCamera tests: a fake set camera, with the merge, check and DNG steps replaced."""

import threading

import numpy as np
import pytest
import tifffile

from negpy.services.capture import service

from negpy.services.capture import pixel_shift as ps
from negpy.services.capture.pixel_shift_merge import WobbleReport, shot_positions
from negpy.services.capture.service import CaptureError

STEADY = WobbleReport(rms=0.0, worst=0.0, bands=8)
MOVED = WobbleReport(rms=2.0, worst=3.0, bands=8)
BLANK = WobbleReport(rms=0.0, worst=0.0, bands=0)


class _Camera:
    def __init__(self):
        self.sets = 0
        self.args = None
        self.closed = False

    def capture_set(self, out_path, count, shutter=None, iso=None, aperture=None, on_shot=None):
        self.sets += 1
        self.args = (shutter, iso, aperture)
        stem = out_path.rsplit(".", 1)[0]
        paths = []
        for n in range(1, count + 1):
            path = f"{stem}_S{n:02d}.ARW"
            with open(path, "w") as handle:
                handle.write(f"set{self.sets} shot{n}")
            paths.append(path)
            on_shot(n, count)
        return paths

    def close(self):
        self.closed = True


@pytest.fixture
def rig(tmp_path, monkeypatch):
    class Rig:
        camera = _Camera()
        reports = [STEADY]
        lines: list[str] = []
        frames_folder = str(tmp_path / ps.FRAMES_FOLDER)

        def shoot(self, **kwargs):
            shifted = ps.PixelShiftCamera(
                self.camera, shot_positions(4), frames_folder=self.frames_folder, report=self.lines.append, **kwargs
            )
            return shifted.capture(str(tmp_path / "Roll_Frame001.raw"), shutter="1/5", iso="100", aperture="8")

        def files(self):
            return sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*") if p.is_file())

    r = Rig()
    r.lines = []

    def _measure(mosaics, positions, black):
        assert len(list(mosaics)) == len(positions)
        return r.reports.pop(0)

    def _merge(mosaics, colors, positions, allocate=None):
        assert len(list(mosaics)) == len(positions)
        return np.zeros((2, 2, 3), np.uint16)

    def _write(rgb, color, out_path, shots):
        assert len([f for f in r.files() if "_S" in f and ps.FRAMES_FOLDER not in f]) == 4  # no frame is gone before the DNG exists
        tifffile.imwrite(out_path, rgb, software=f"NegPy Pixel Shift {shots}")

    monkeypatch.setattr(ps, "_load_mosaic", lambda path: np.zeros((2, 2), np.uint16))
    monkeypatch.setattr(ps, "_load_colors", lambda path: np.zeros((2, 2), np.uint8))
    monkeypatch.setattr(ps, "_load_black", lambda path: 512.0)
    monkeypatch.setattr(ps, "measure_wobble", _measure)
    monkeypatch.setattr(ps, "merge_set", _merge)
    monkeypatch.setattr(ps, "raw_color", lambda path: None)
    monkeypatch.setattr(ps, "write_linear_dng", _write)
    return r


SHOTS = [f"Shot {n} of 4" for n in range(1, 5)]
FRAMES = [f"Roll_Frame001_S{n:02d}.ARW" for n in range(1, 5)]
CHECK = "Checking the frames for movement…"
SAVE = ["Merging 4 frames…", "Saving the merged frame…"]


def test_a_steady_set_is_merged_and_its_frames_deleted(rig, tmp_path):
    assert rig.shoot() == str(tmp_path / "Roll_Frame001.dng")
    assert rig.files() == ["Roll_Frame001.dng"]
    assert rig.camera.sets == 1 and rig.camera.args == ("1/5", "100", "8")
    assert rig.lines == [*SHOTS, CHECK, *SAVE]


def test_keep_frames_moves_them_into_their_folder(rig):
    rig.shoot(keep_frames=True)
    assert rig.files() == [f"{ps.FRAMES_FOLDER}/{name}" for name in FRAMES] + ["Roll_Frame001.dng"]


def test_a_moved_set_is_shot_again_and_removed(rig):
    rig.reports = [MOVED, STEADY]
    rig.shoot(keep_frames=True)
    assert rig.camera.sets == 2
    assert rig.files() == [f"{ps.FRAMES_FOLDER}/{name}" for name in FRAMES] + ["Roll_Frame001.dng"]
    assert (rig.lines[len(SHOTS) : len(SHOTS) + 2]) == [CHECK, "The set moved or a frame is blurred. Shooting again (2 of 3)…"]
    assert rig.lines[-len(SAVE) :] == SAVE


def test_three_moved_sets_keep_one_plain_frame(rig, tmp_path):
    rig.reports = [MOVED, MOVED, MOVED]
    assert rig.shoot() == str(tmp_path / "Roll_Frame001.ARW")
    assert rig.camera.sets == 3
    assert rig.files() == ["Roll_Frame001.ARW"]
    assert (tmp_path / "Roll_Frame001.ARW").read_text() == "set3 shot1"
    assert rig.lines == [
        *SHOTS,
        CHECK,
        "The set moved or a frame is blurred. Shooting again (2 of 3)…",
        *SHOTS,
        CHECK,
        "The set moved or a frame is blurred. Shooting again (3 of 3)…",
        *SHOTS,
        CHECK,
        "Every set moved or was blurred. Kept one frame without Pixel Shift.",
    ]


def test_a_set_that_cannot_be_judged_is_merged_and_said_so(rig):
    rig.reports = [BLANK]
    rig.shoot()
    assert rig.lines == [*SHOTS, CHECK, "The frame has too little detail to check for movement.", *SAVE]


def test_a_failed_dng_write_moves_the_frames_out_of_staging(rig, monkeypatch):
    def _fail(*_args):
        raise OSError("disk full")

    monkeypatch.setattr(ps, "write_linear_dng", _fail)
    with pytest.raises(OSError):
        rig.shoot()
    assert rig.files() == [f"{ps.FRAMES_FOLDER}/{name}" for name in FRAMES]


def test_a_failed_merge_moves_the_frames_out_of_staging(rig, monkeypatch):
    def _fail(*_args):
        raise MemoryError

    monkeypatch.setattr(ps, "merge_set", _fail)
    with pytest.raises(MemoryError):
        rig.shoot()
    assert rig.files() == [f"{ps.FRAMES_FOLDER}/{name}" for name in FRAMES]


def test_three_moved_sets_fail_where_a_plain_frame_cannot_stand_in(rig):
    rig.reports = [MOVED, MOVED, MOVED]
    with pytest.raises(ps.CaptureError, match="Every set moved or was blurred"):
        rig.shoot(fallback=False)
    assert rig.files() == []


def test_cancel_between_steps_removes_the_set(rig):
    cancel = threading.Event()
    rig.lines = type("Lines", (list,), {"append": lambda self, line: cancel.set()})()
    with pytest.raises(CaptureError, match="capture cancelled"):
        rig.shoot(cancel=cancel)
    assert rig.camera.sets == 1 and rig.files() == []


def test_close_and_other_attributes_reach_the_camera():
    camera = _Camera()
    shifted = ps.PixelShiftCamera(camera, shot_positions(4), frames_folder="unused")
    assert shifted.sets == 0
    shifted.close()
    assert camera.closed


def test_a_failed_write_inside_the_service_keeps_the_frames(rig, tmp_path, monkeypatch):
    """The service deletes its staging folder, so the frames must leave it first."""

    def _fail(*_args):
        raise OSError("disk full")

    monkeypatch.setattr(ps, "write_linear_dng", _fail)
    shifted = ps.PixelShiftCamera(rig.camera, shot_positions(4), frames_folder=rig.frames_folder)
    with pytest.raises(OSError):
        service._capture_validated_single(
            shifted, final_stem=str(tmp_path / "Roll_Frame001"), shutter=None, min_raw_bytes=0, max_raw_bytes=10**9
        )
    assert rig.files() == [f"{ps.FRAMES_FOLDER}/{name}" for name in FRAMES]


def test_a_retake_under_the_other_setting_replaces_the_old_file(rig, tmp_path):
    for name in ("Roll_Frame001.ARW", "Roll_Frame001.tif", "Roll_Frame002.ARW"):
        (tmp_path / name).write_text("old")
    shifted = ps.PixelShiftCamera(rig.camera, shot_positions(4), frames_folder=rig.frames_folder)
    service._capture_validated_single(
        shifted, final_stem=str(tmp_path / "Roll_Frame001"), shutter=None, min_raw_bytes=0, max_raw_bytes=10**9
    )
    assert rig.files() == ["Roll_Frame001.dng", "Roll_Frame001.tif", "Roll_Frame002.ARW"]


def test_a_set_reports_its_share_of_the_exposure(rig):
    shares: list[float] = []
    rig.shoot(part=shares.append)
    assert shares == pytest.approx([0.2, 0.4, 0.6, 0.8, 0.8, 0.9, 0.95])


def test_the_plain_frame_that_stands_in_is_the_sharpest(rig, tmp_path):
    rig.reports = [MOVED, MOVED, WobbleReport(rms=2.0, worst=3.0, bands=8, soft_shot=1, sharpest_shot=3)]
    rig.shoot()
    assert rig.files() == ["Roll_Frame001.ARW"]
    assert (tmp_path / "Roll_Frame001.ARW").read_text() == "set3 shot3"


def test_stop_after_the_merge_still_saves_the_set(rig, monkeypatch):
    cancel = threading.Event()
    merge = ps.merge_set
    monkeypatch.setattr(ps, "merge_set", lambda *args: (cancel.set(), merge(*args))[1])
    rig.shoot(cancel=cancel)
    assert rig.files() == ["Roll_Frame001.dng"]


def test_a_retake_replaces_the_kept_frames_of_the_earlier_take(rig, tmp_path):
    folder = tmp_path / ps.FRAMES_FOLDER
    folder.mkdir()
    for name in ("Roll_Frame001_S01.ARW", "Roll_Frame001_S16.ARW", "Roll_Frame0012_S01.ARW", "Roll_Frame002_S01.ARW"):
        (folder / name).write_text("old")
    rig.shoot(keep_frames=True)
    kept = sorted(p.name for p in folder.iterdir())
    assert kept == sorted([*FRAMES, "Roll_Frame0012_S01.ARW", "Roll_Frame002_S01.ARW"])
    assert (folder / "Roll_Frame001_S01.ARW").read_text() == "set1 shot1"


def test_frames_that_cannot_be_kept_are_said_so(rig, tmp_path):
    (tmp_path / ps.FRAMES_FOLDER).write_text("a file where the folder goes")
    rig.shoot(keep_frames=True)
    assert rig.lines[-1] == "Could not keep the source frames."
    assert "Roll_Frame001.dng" in rig.files()


def test_only_a_merged_file_and_its_plain_twin_replace_each_other(tmp_path):
    """A capture never removes a same-named file that no Pixel Shift take wrote."""
    for name in ("Roll_Frame001.ARW", "Roll_Frame001.NEF"):
        (tmp_path / name).write_text("x")
    tifffile.imwrite(tmp_path / "Roll_Frame001.dng", np.zeros((2, 2, 3), np.uint16), software="a converter")
    service._drop_other_formats(str(tmp_path / "Roll_Frame001.ARW"))
    assert sorted(p.name for p in tmp_path.iterdir()) == ["Roll_Frame001.ARW", "Roll_Frame001.NEF", "Roll_Frame001.dng"]
    tifffile.imwrite(tmp_path / "Roll_Frame001.dng", np.zeros((2, 2, 3), np.uint16), software="NegPy Pixel Shift 4")
    service._drop_other_formats(str(tmp_path / "Roll_Frame001.ARW"))  # a plain retake of a merged frame
    assert sorted(p.name for p in tmp_path.iterdir()) == ["Roll_Frame001.ARW", "Roll_Frame001.NEF"]


def test_an_earlier_take_that_cannot_be_removed_does_not_fail_the_capture(tmp_path, monkeypatch):
    tifffile.imwrite(tmp_path / "Roll_Frame001.dng", np.zeros((2, 2, 3), np.uint16), software="NegPy Pixel Shift 4")
    (tmp_path / "Roll_Frame001.ARW").write_text("x")
    monkeypatch.setattr(service.os, "remove", lambda _path: (_ for _ in ()).throw(PermissionError("locked")))
    service._drop_other_formats(str(tmp_path / "Roll_Frame001.ARW"))


GB = 2**30


@pytest.mark.parametrize(
    "megapixels, available, in_memory",
    [
        (61, 2 * GB, True),  # 4 shots, a 4 GB machine with half its memory free
        (244, 2 * GB, False),  # 16 shots on the same machine
        (244, 512 * GB, True),
        (1600, 8 * GB, False),  # 16 shots of a 400 MP body
        (1600, 512 * GB, True),
        (244, None, False),  # the OS does not say
    ],
)
def test_the_merge_works_in_memory_only_when_it_fits(tmp_path, megapixels, available, in_memory):
    """`megapixels` is the merged frame; the merge holds four samples per output site."""
    assert ps.fits_in_memory(2 * 4 * megapixels * 10**6, available) is in_memory
    (image, plane), mapped = ps.merge_buffers((6, 2), str(tmp_path / "m"), GB if in_memory else 0, free_disk=GB)
    assert (image.dtype, image.shape, plane.shape) == (np.uint16, (6,), (2,)) and mapped is not in_memory
    assert isinstance(image, np.memmap) is not in_memory
    image[:], plane[:] = 7, 9  # the two buffers share one file and no sample
    assert image.tolist() == [7] * 6 and plane.tolist() == [9] * 2


@pytest.mark.parametrize("available, held", [(512 * GB, 6), (0, 6 + 8)])
def test_a_disk_that_cannot_hold_the_merge_takes_no_file(tmp_path, available, held):
    """The disk holds the merged file, and the merge's own file when memory is short."""
    sizes = (3 * 10**6, 10**6)  # 6 MB of merged file, 8 MB of buffers
    enough = held * 10**6 + ps.DISK_MARGIN_BYTES
    ps.merge_buffers(sizes, str(tmp_path / "passed"), available, enough)
    for short in (10**8, 1):  # the message never asks for 0.0 GB
        with pytest.raises(CaptureError, match="Free 0.1 GB"):
            ps.merge_buffers(sizes, str(tmp_path / "refused"), available, enough - short)
    assert [path.stat().st_size for path in tmp_path.iterdir()] == ([] if available else [8 * 10**6])  # none from the refused call


@pytest.mark.parametrize("count", [4, 16])
def test_a_merge_in_a_file_gives_the_same_frame(tmp_path, count):
    from negpy.services.capture.pixel_shift_merge import merge_set, shot_positions

    rng = np.random.default_rng(count)
    mosaics = [rng.integers(0, 16000, (32, 48), np.uint16) for _ in range(count)]
    colors = np.array([[0, 1], [3, 2]], np.uint8)
    in_memory = merge_set(iter(mosaics), colors, shot_positions(count))
    mapped = merge_set(iter(mosaics), colors, shot_positions(count), lambda sizes: ps.merge_buffers(sizes, str(tmp_path / "m"), 0, GB)[0])
    assert isinstance(mapped.base, np.memmap)
    assert np.array_equal(mapped, in_memory)


def test_a_set_merged_in_a_file_leaves_only_the_merged_file(rig, tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(ps, "available_system_memory_bytes", lambda: 0)
    monkeypatch.setattr(ps.shutil, "disk_usage", lambda _folder: type("Usage", (), {"free": GB})())
    monkeypatch.setattr(
        ps, "merge_set", lambda mosaics, colors, positions, allocate: seen.extend(allocate((6, 2))) or np.zeros((2, 2, 3), np.uint16)
    )
    rig.shoot()
    assert all(isinstance(buffer, np.memmap) for buffer in seen)
    assert rig.files() == ["Roll_Frame001.dng"]


def test_a_full_disk_is_said_and_leaves_no_file(rig, tmp_path, monkeypatch):
    monkeypatch.setattr(ps.shutil, "disk_usage", lambda _folder: type("Usage", (), {"free": 0})())
    monkeypatch.setattr(ps, "merge_set", lambda mosaics, colors, positions, allocate: allocate((6, 2)))
    with pytest.raises(CaptureError, match="Not enough free disk space"):
        rig.shoot()
    assert rig.files() == []


@pytest.mark.parametrize("step", ["merge_set", "write_linear_dng"])
def test_a_failed_merge_in_a_file_releases_the_file_first(rig, monkeypatch, step):
    """Windows cannot remove a mapped file, and an error's traceback holds the mapping."""
    import weakref

    alive = []
    monkeypatch.setattr(ps, "available_system_memory_bytes", lambda: 0)
    monkeypatch.setattr(ps.shutil, "disk_usage", lambda _folder: type("Usage", (), {"free": GB})())

    def merge(mosaics, colors, positions, allocate):
        image, _plane = allocate((12, 4))
        alive.append(weakref.ref(image))
        if step == "merge_set":
            raise ValueError("a bad shot")
        return image.reshape(2, 2, 3)

    def write(rgb, *args):
        raise OSError("disk")

    unlink = ps.safe_unlink
    monkeypatch.setattr(ps, "merge_set", merge)
    monkeypatch.setattr(ps, "write_linear_dng", write)
    monkeypatch.setattr(ps, "safe_unlink", lambda path: (alive.append(alive[0]() is None), unlink(path)))
    with pytest.raises((ValueError, OSError)):
        rig.shoot()
    assert alive[1:] == [True]
    assert not [name for name in rig.files() if name.endswith(".merge")]


class _MergedCamera:
    """Returns a merged set, with Stop pressed while it merged."""

    def __init__(self, cancel):
        self.cancel = cancel

    def capture(self, out_path, shutter=None, iso=None, aperture=None):
        path = out_path.rsplit(".", 1)[0] + ".dng"
        tifffile.imwrite(path, np.zeros((2, 2, 3), np.uint16), software="NegPy Pixel Shift 4")
        self.cancel.set()
        return path


def test_stop_during_the_merge_keeps_the_merged_frame(tmp_path):
    cancel = threading.Event()
    path = service.capture_single(
        _MergedCamera(cancel), roll_name="Roll", frame_number=1, output_folder=str(tmp_path), min_raw_bytes=0, cancel=cancel
    )
    assert [p.name for p in tmp_path.iterdir()] == ["Roll_Frame001.dng"] == [path.rsplit("/", 1)[1]]
