from __future__ import annotations

from negpy.desktop.workers.scan_worker import ScanWorker


class _EjectService:
    def __init__(self, result: bool = True, error: Exception | None = None) -> None:
        self.result = result
        self.error = error
        self.calls: list[str] = []

    def eject(self, device_id: str) -> bool:
        self.calls.append(device_id)
        if self.error is not None:
            raise self.error
        return self.result


def test_scan_worker_emits_eject_result() -> None:
    worker = ScanWorker()
    service = _EjectService()
    worker._service = service  # type: ignore[assignment]
    results: list[bool] = []
    worker.ejected.connect(results.append)

    worker.eject("coolscan3:test")

    assert service.calls == ["coolscan3:test"]
    assert results == [True]


def test_scan_worker_reports_eject_failure() -> None:
    worker = ScanWorker()
    service = _EjectService(error=RuntimeError("transport refused"))
    worker._service = service  # type: ignore[assignment]
    errors: list[str] = []
    worker.eject_error.connect(errors.append)

    worker.eject("coolscan3:test")

    assert service.calls == ["coolscan3:test"]
    assert errors == ["transport refused"]
