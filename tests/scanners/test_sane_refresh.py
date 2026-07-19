"""Refreshing devices must re-open, never tear the backend down.

coolscan3(5) BUGS claims the --frame option is fixed at backend init, which
invites a sane.exit()/init() cycle in refresh_devices() so film loaded after
startup is sensed. That reading is wrong and the cure is dangerous: coolscan3
senses the adapter in cs3_full_inquiry(), which sane_open() calls, so a re-open
already rebuilds the frame option from the live strip — while sane_exit() frees
the device list that open handles still point into, and sane_get_devices()
refuses to run at all while any handle is open.
"""

import threading
from unittest.mock import MagicMock

from negpy.infrastructure.scanners.sane_backend import SaneBackend


def _backend() -> tuple[SaneBackend, MagicMock]:
    fake = MagicMock()
    fake.get_devices.return_value = []
    backend = SaneBackend.__new__(SaneBackend)
    backend._sane = fake
    backend._sane_initialized = False
    backend._devices_cache = None
    backend._id_remap = {}
    backend._active_sessions = {}
    backend._session_lock = threading.Lock()
    return backend, fake


def test_refresh_never_tears_the_backend_down() -> None:
    backend, fake = _backend()
    backend.list_devices()

    backend.refresh_devices()

    fake.exit.assert_not_called()
    assert fake.init.call_count == 1


def test_refresh_re_enumerates_so_a_reopen_re_senses_the_adapter() -> None:
    backend, fake = _backend()
    backend.list_devices()
    assert fake.get_devices.call_count == 1

    backend.refresh_devices()

    # The re-open inside list_devices is what picks up newly loaded film.
    assert fake.get_devices.call_count == 2


def test_repeat_listings_stay_cached() -> None:
    backend, fake = _backend()
    backend.list_devices()
    backend.list_devices()

    assert fake.get_devices.call_count == 1
