"""Tests for the Tier-2 repair engine seam (negpy.infrastructure.roll.repair).

No repair engine ships in this package, so `available()` must be False and
`repair()` must raise until a caller registers one -- these tests pin that
contract down, since `RollScanningService.write_frame`'s Tier 2/3 degrade
path depends on it. `fake_repair_engine` (tests/roll/conftest.py) scripts a
registered engine for the tests that need one; it always unregisters on
teardown, so a test that does not use it sees the unavailable state.
"""

from __future__ import annotations

import numpy as np
import pytest

from negpy.infrastructure.roll import repair as roll_repair


class TestUnavailableByDefault:
    def test_available_is_false_with_nothing_registered(self) -> None:
        assert roll_repair.available() is False

    def test_repair_raises_when_unavailable(self) -> None:
        rgb = np.zeros((2, 2, 3), dtype=np.uint16)
        ir = np.zeros((2, 2), dtype=np.uint16)
        with pytest.raises(RuntimeError, match="no dust-repair engine is registered"):
            roll_repair.repair(rgb, ir, roll_repair.RepairMode.EXACT)


class TestRegisteredEngine:
    def test_available_becomes_true(self, fake_repair_engine) -> None:
        assert roll_repair.available() is True

    def test_repair_delegates_to_the_registered_engine(self, fake_repair_engine) -> None:
        rgb = np.full((2, 2, 3), 100, dtype=np.uint16)
        ir = np.full((2, 2), 50, dtype=np.uint16)

        result = roll_repair.repair(rgb, ir, roll_repair.RepairMode.HYBRID)

        assert fake_repair_engine.calls == [(rgb, ir, roll_repair.RepairMode.HYBRID)]
        assert result.engine == "test-repair-engine"
        assert result.engine_version == "0.0.1-test"
        assert result.mode == roll_repair.RepairMode.HYBRID
        np.testing.assert_array_equal(result.rgb, rgb)  # default transform is identity

    def test_unregister_reverts_to_unavailable(self, fake_repair_engine) -> None:
        roll_repair.unregister_engine()
        assert roll_repair.available() is False
        with pytest.raises(RuntimeError):
            roll_repair.repair(np.zeros((1, 1, 3), dtype=np.uint16), np.zeros((1, 1), dtype=np.uint16), roll_repair.RepairMode.EXACT)


class TestRepairMode:
    def test_values_are_exact_and_hybrid(self) -> None:
        assert roll_repair.RepairMode.EXACT.value == "exact"
        assert roll_repair.RepairMode.HYBRID.value == "hybrid"

    def test_constructible_from_its_own_string_value(self) -> None:
        """RollScanningService.write_frame coerces a persisted settings string
        back to a RepairMode this way; pin the round-trip."""
        assert roll_repair.RepairMode("exact") is roll_repair.RepairMode.EXACT
        assert roll_repair.RepairMode("hybrid") is roll_repair.RepairMode.HYBRID
