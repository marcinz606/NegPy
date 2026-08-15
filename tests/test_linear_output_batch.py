from unittest.mock import patch

from negpy.desktop.workers.export import ExportWorker, LinearOutputTask


def _task(name: str) -> LinearOutputTask:
    return LinearOutputTask(
        file_info={"name": f"{name}.nef", "path": f"/tmp/{name}.nef", "hash": name},
        out_path=f"/tmp/out/{name}_linear.tiff",
        options={"gamma_key": "linear"},
    )


def test_linear_output_runs_in_the_worker_with_progress() -> None:
    """Linear Output used to loop on the GUI thread, so it reported no progress and
    could not be aborted."""
    worker = ExportWorker()
    progress: list[tuple[int, int, str]] = []
    finished: list[bool] = []
    worker.progress.connect(lambda *a: progress.append(a))
    worker.finished.connect(lambda: finished.append(True))

    with patch("negpy.services.export.linear_output.export_linear_output") as export:
        worker.run_linear_output([_task("a"), _task("b")])

    assert progress == [(1, 2, "a"), (2, 2, "b")]
    assert finished == [True]
    assert export.call_count == 2
    assert export.call_args_list[0].args == ("/tmp/a.nef", "/tmp/out/a_linear.tiff")
    assert export.call_args_list[0].kwargs == {"gamma_key": "linear"}


def test_linear_output_cancel_stops_the_batch() -> None:
    worker = ExportWorker()
    cancelled: list[bool] = []
    worker.cancelled.connect(lambda: cancelled.append(True))

    with patch("negpy.services.export.linear_output.export_linear_output", side_effect=lambda *a, **k: worker.cancel()) as export:
        worker.run_linear_output([_task("a"), _task("b"), _task("c")])

    assert cancelled == [True]
    assert export.call_count == 1


def test_linear_output_failure_is_reported_and_the_batch_continues() -> None:
    """Each error raises the controller's failure count, so the completion toast can
    say how many frames did not make it."""
    worker = ExportWorker()
    errors: list[str] = []
    finished: list[bool] = []
    worker.error.connect(errors.append)
    worker.finished.connect(lambda: finished.append(True))

    def _fail_first(path, *_a, **_k):
        if path.endswith("a.nef"):
            raise OSError("disk full")

    with patch("negpy.services.export.linear_output.export_linear_output", side_effect=_fail_first) as export:
        worker.run_linear_output([_task("a"), _task("b")])

    assert len(errors) == 1
    assert "disk full" in errors[0]
    assert export.call_count == 2
    assert finished == [True]
