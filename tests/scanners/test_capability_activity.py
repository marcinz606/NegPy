"""Presence of a SANE option is not capability. A backend advertises options it
compiled in, then marks them SANE_CAP_INACTIVE per device — coolscan3 carries
`ae` on an LS-50 only where the hardware meters. Gating the UI on presence would
offer a control whose every non-default value SaneBackend.scan() then refuses."""

from negpy.infrastructure.scanners.sane_backend import (
    _detect_auto_exposure,
    _detect_eject,
    _has_usable_option,
)


class _Option:
    def __init__(self, active: bool = True, settable: bool = True) -> None:
        self._active = active
        self._settable = settable

    def is_active(self) -> bool:
        return self._active

    def is_settable(self) -> bool:
        return self._settable


def _ls50_opt() -> dict:
    """A real LS-50: the options exist, but the driver says no to some."""
    return {
        "ae": _Option(active=True),
        "eject": _Option(active=True),
    }


def test_active_auto_exposure_survives() -> None:
    """The LS-50 does meter in hardware — this must stay available."""
    assert _detect_auto_exposure(_ls50_opt()) is True


def test_inactive_auto_exposure_is_not_a_capability() -> None:
    opt = _ls50_opt()
    opt["ae"] = _Option(active=False)
    assert _detect_auto_exposure(opt) is False


def test_absent_auto_exposure_is_not_a_capability() -> None:
    assert _detect_auto_exposure({}) is False


def test_active_eject_is_a_capability() -> None:
    assert _detect_eject(_ls50_opt()) is True


def test_inactive_eject_is_not_a_capability() -> None:
    opt = _ls50_opt()
    opt["eject"] = _Option(active=False)
    assert _detect_eject(opt) is False


def test_absent_eject_is_not_a_capability() -> None:
    assert _detect_eject({}) is False


def test_options_without_activity_methods_are_trusted() -> None:
    """Not every backend's option object implements is_active; those must not
    be treated as unusable."""

    class _Bare:
        pass

    assert _has_usable_option({"ae": _Bare()}, "ae") is True
