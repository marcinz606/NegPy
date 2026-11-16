import numpy as np

from negpy.features.process.fade import measure_crossover
from negpy.features.process.fade_roll import FrameReading, pick_crossover, pool_red, pool_survivals


def _readings(pairs):
    return [FrameReading(f"f{i}", g, b) for i, (g, b) in enumerate(pairs)]


def test_pool_survivals_leaves_out_a_frame_that_reads_far_from_the_roll():
    pooled = pool_survivals(_readings([(0.93, 0.57), (0.94, 0.56), (0.92, 0.58), (0.95, 0.55), (0.86, 0.32)]))
    assert pooled is not None
    assert pooled.outliers == ("f4",)
    assert abs(pooled.green - 0.935) < 1e-9 and abs(pooled.blue - 0.565) < 1e-9


def test_pool_survivals_needs_three_frames():
    assert pool_survivals(_readings([(0.9, 0.5), (0.9, 0.5)])) is None


def test_pool_survivals_keeps_frames_a_hair_apart():
    pooled = pool_survivals(_readings([(0.900, 0.5), (0.900, 0.5), (0.900, 0.5), (0.905, 0.5)]))
    assert pooled is not None and pooled.outliers == ()


def _shift_at(median, red=0.8):
    """A frame whose median sits at density *median* at *red*: its shift at a new red takes back what the median gains."""
    return lambda new: -median * (red / new - 1.0)


def test_pool_red_takes_the_median_and_drops_frames_far_from_it():
    shift_at = [_shift_at(m) for m in (1.0, 1.1, 0.9, 1.2, 0.5)]
    red, shift = pool_red([0.53, 0.52, 0.51, 0.62, 0.40], shift_at)
    assert red == 0.52
    assert shift == round(float(np.median([f(0.52) for f in shift_at[:3]])), 4)


def test_pool_red_takes_each_frame_s_shift_at_the_pooled_red():
    """A dense frame's shift found at its own short red is far larger than at the roll's red, so a median of
    the shifts as found would pair the roll's red with a shift no frame needs there."""
    reds = [0.90, 0.92, 0.68, 0.40, 0.40]
    shift_at = [_shift_at(m) for m in (1.0, 1.1, 0.1, 0.8, 0.9)]
    as_found = [f(r) for f, r in zip(shift_at, reds)]
    red, shift = pool_red(reds, shift_at)
    assert red == 0.68
    assert shift == round(float(np.median([f(0.68) for f in shift_at])), 4)
    assert abs(shift - float(np.median(as_found))) > 0.1


def _table(zero, rows):
    return tuple([zero, *[v for row in rows for v in row]])


def _gray_frame(lean_per_d, seed):
    """A log grid of gray cells from D 0.15 to 2.6 whose channels lean by *lean_per_d* times (D - 1)."""
    rng = np.random.default_rng(seed)
    d = np.repeat(np.linspace(0.15, 2.6, 60), 40)
    lean = np.outer(d - 1.0, lean_per_d)
    cells = -(d[:, None] + lean) + rng.normal(0, 0.003, (len(d), 3))
    grid = cells.reshape(48, 50, 3).astype(np.float32)
    return grid, -1.0


def test_pick_crossover_takes_the_table_the_film_shares_over_one_scene_light():
    film = np.array([0.06, 0.0, -0.06])
    frames = []
    for i, lean in enumerate((film, film * 1.1, film * 0.9, -film * 1.5)):
        grid, zero = _gray_frame(lean, i)
        frames.append((f"f{i}", grid, zero, measure_crossover(grid, zero)))
    pick = pick_crossover(frames)
    assert pick is not None and pick.key in ("f0", "f1", "f2")
    assert pick.tried == 3 and pick.helped >= 2


def test_pick_crossover_skips_a_table_whose_grays_sit_in_few_tones(monkeypatch):
    import negpy.features.process.fade_roll as fade_roll

    film = np.array([0.06, 0.0, -0.06])
    frames = []
    for i, lean in enumerate((film, film * 1.1, film * 0.9, film * 1.05)):
        grid, zero = _gray_frame(lean, i)
        frames.append((f"f{i}", grid, zero, measure_crossover(grid, zero)))
    best = pick_crossover(frames).key
    narrow = next(t for k, _, _, t in frames if k == best)
    span = fade_roll._gray_span
    monkeypatch.setattr(fade_roll, "_gray_span", lambda t: 0.1 if t is narrow else span(t))
    assert pick_crossover(frames).key != best


def test_pick_crossover_needs_two_frames_with_a_table():
    grid, zero = _gray_frame(np.array([0.06, 0.0, -0.06]), 0)
    assert pick_crossover([("a", grid, zero, measure_crossover(grid, zero))]) is None
    assert pick_crossover([("a", grid, zero, None), ("b", grid, zero, None)]) is None


def test_the_roll_remembers_which_frame_its_table_came_from():
    from negpy.services.assets import rolls

    class Repo:
        def __init__(self):
            self.store = {}

        def get_global_setting(self, key, default=None):
            return self.store.get(key, default)

        read_global_setting = get_global_setting

        def save_global_setting(self, key, value):
            self.store[key] = value

    repo = Repo()
    rolls._write(repo, {"r1": {"defaults": {}}})
    table = _table(-0.6, [(-0.5, 0.05, 0.0, 0.0, 0.0), (-0.7, 0.05, 0.0, 0.0, 0.0)])
    rolls.set_roll_crossover_source(repo, "r1", "Image (7).tif", table)
    assert rolls.roll_crossover_source(repo, "r1", table) == "Image (7).tif"
    assert rolls.roll_crossover_source(repo, "r1", tuple(v + 0.1 for v in table)) == ""
    assert rolls.roll_crossover_source(repo, "unknown", table) == ""


def _roll_worker(grids):
    """A worker whose frames read as *grids* (name -> Estimate grid, None for an unreadable frame)."""
    from negpy.desktop.workers.fade_roll import FadeRollTask, FadeRollWorker
    from negpy.domain.models import WorkspaceConfig

    worker = FadeRollWorker(preview_service=None)
    worker._read = lambda info, config, task: (grids[info["name"]], "")
    task = FadeRollTask([({"name": name, "hash": name}, WorkspaceConfig()) for name in grids], "Adobe RGB")
    return worker, task


def _faded_grid(survivals, seed=0):
    from fade_crossover_scene import REFERENCE, build_scene, estimate_grid, linear_fade, scan_image

    return estimate_grid(scan_image(linear_fade(REFERENCE, survivals), build_scene(96, 72), seed=seed))


def test_the_roll_worker_pools_the_frames_and_names_the_ones_it_leaves_out():
    grids = {f"f{i}.tif": _faded_grid((1.0, 0.8, 0.65), seed=i) for i in range(4)}
    grids["odd.tif"] = _faded_grid((1.0, 1.25, 0.65), seed=9)
    grids["flat.tif"] = np.full_like(grids["f0.tif"], 0.5)
    worker, task = _roll_worker(grids)

    result = worker._run(task)

    assert (result.read, result.counted) == (6, 4)
    assert result.outliers == ["odd.tif"] and result.no_grays == ["flat.tif"]
    assert set(result.changes) == {
        "fade_ratio_r",
        "fade_ratio_g",
        "fade_ratio_b",
        "fade_density_shift",
        "fade_auto_red",
        "fade_white_anchor",
    }
    assert abs(result.changes["fade_ratio_g"] - 0.8) < 0.03 and abs(result.changes["fade_ratio_b"] - 0.65) < 0.03


def test_the_roll_worker_gives_a_reason_with_too_few_readable_frames():
    grids = {f"f{i}.tif": _faded_grid((1.0, 0.8, 0.65), seed=i) for i in range(2)}
    worker, task = _roll_worker(grids)

    result = worker._run(task)

    assert not result.changes and result.reason


def test_a_cancelled_roll_worker_reports_cancelled_and_no_result():
    worker, task = _roll_worker({f"f{i}.tif": _faded_grid((1.0, 0.8, 0.65), seed=i) for i in range(3)})
    ended = []
    worker.cancelled.connect(lambda: ended.append("cancelled"))
    worker.finished.connect(lambda _result: ended.append("finished"))
    worker.cancel()

    worker.process(task)

    assert ended == ["cancelled"]


def test_the_roll_worker_names_a_frame_it_could_not_read():
    grids = {f"f{i}.tif": _faded_grid((1.0, 0.8, 0.65), seed=i) for i in range(3)}
    grids["broken.tif"] = None
    worker, task = _roll_worker(grids)

    result = worker._run(task)

    assert result.unread == ["broken.tif"] and (result.read, result.counted) == (3, 3)


def test_the_roll_worker_tells_two_frames_of_one_name_apart():
    from negpy.desktop.workers.fade_roll import FadeRollTask, FadeRollWorker
    from negpy.domain.models import WorkspaceConfig

    grids = {f"h{i}": _faded_grid((1.0, 0.8, 0.65), seed=i) for i in range(4)}
    grids["odd"] = _faded_grid((1.0, 1.25, 0.65), seed=9)
    worker = FadeRollWorker(preview_service=None)
    worker._read = lambda info, config, task: (grids[info["hash"]], "")
    names = {"h3": "same.tif", "odd": "same.tif"}
    task = FadeRollTask([({"name": names.get(h, f"{h}.tif"), "hash": h}, WorkspaceConfig()) for h in grids], "Adobe RGB")

    result = worker._run(task)

    assert result.counted == 4 and result.outliers == ["same.tif"]
