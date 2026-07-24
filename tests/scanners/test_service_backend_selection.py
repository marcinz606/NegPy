from negpy.services.scanning.service import ScannerService


def test_injected_backend_wins_over_id():
    fake = object()
    svc = ScannerService(backend=fake, backend_id="anything")
    assert svc._get_backend() is fake


def test_backend_id_routes_through_registry(monkeypatch):
    fake = object()
    seen: list[str] = []

    def _create(backend_id):
        seen.append(backend_id)
        return fake

    # service imports create_backend lazily inside _get_backend, so patch the
    # registry module attribute it resolves.
    monkeypatch.setattr("negpy.infrastructure.scanners.registry.create_backend", _create)

    svc = ScannerService(backend_id="fake")
    assert svc._get_backend() is fake
    assert seen == ["fake"]


def test_default_backend_id_used_when_none(monkeypatch):
    seen: list[str] = []

    def _create(backend_id):
        seen.append(backend_id)
        return object()

    monkeypatch.setattr("negpy.infrastructure.scanners.registry.create_backend", _create)

    ScannerService()._get_backend()
    assert seen == ["sane"]
