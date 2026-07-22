"""Tests for the Tier-2 repair engine seam (negpy.infrastructure.roll.repair).

The optional digital-fauxice bridge auto-registers only when its engine is
installed. These tests force the seam's independently-unavailable state,
then pin that degrade contract down because `RollScanningService.write_frame`'s
Tier 2/3 path depends on it. `fake_repair_engine` (tests/roll/conftest.py)
scripts a registered engine for tests that need one.
"""

from __future__ import annotations

import numpy as np
import pytest

from negpy.infrastructure.roll import repair as roll_repair


def _acquisition() -> roll_repair.RepairAcquisition:
    main = np.full((3, 2, 4), 100, dtype=np.uint16)
    return roll_repair.RepairAcquisition.from_arrays(
        acquisition_id="dice-" + "a" * 64,
        slot=7,
        reservation_id="reservation-007",
        capture_attempt_id="fine-slot-7-attempt-001",
        storage_transform=roll_repair.DIGITAL_ICE_STORAGE_TRANSFORM,
        evidence_sha256="b" * 64,
        main_rgbi=main,
        prepass_rgbi=np.full((2, 2, 4), 50, dtype=np.uint16),
        ir_validity=np.ones((3, 2), dtype=np.bool_),
    )


class TestUnavailableByDefault:
    def test_available_is_false_with_nothing_registered(self, no_repair_engine) -> None:
        assert roll_repair.available() is False

    def test_repair_raises_when_unavailable(self, no_repair_engine) -> None:
        with pytest.raises(RuntimeError, match="no dust-repair engine is registered"):
            roll_repair.repair(_acquisition(), roll_repair.RepairMode.EXACT)


class TestRegisteredEngine:
    def test_available_becomes_true(self, fake_repair_engine) -> None:
        assert roll_repair.available() is True

    def test_repair_delegates_to_the_registered_engine(self, fake_repair_engine) -> None:
        main = np.full((3, 2, 4), 100, dtype=np.uint16)
        prepass = np.full((2, 2, 4), 50, dtype=np.uint16)
        validity = np.ones((3, 2), dtype=np.bool_)
        acquisition = roll_repair.RepairAcquisition.from_arrays(
            acquisition_id="dice-" + "a" * 64,
            slot=7,
            reservation_id="reservation-007",
            capture_attempt_id="fine-slot-7-attempt-001",
            storage_transform="rot90-k1-scanner-native-to-upright-v1",
            evidence_sha256="b" * 64,
            main_rgbi=main,
            prepass_rgbi=prepass,
            ir_validity=validity,
        )

        result = roll_repair.repair(acquisition, roll_repair.RepairMode.HYBRID)

        assert fake_repair_engine.calls == [(acquisition, roll_repair.RepairMode.HYBRID, None)]
        assert result.engine == "test-repair-engine"
        assert result.engine_version == "0.0.1-test"
        assert result.mode_requested == roll_repair.RepairMode.HYBRID
        assert result.mode_resolved == roll_repair.RepairMode.EXACT
        assert result.degraded is True
        np.testing.assert_array_equal(
            result.rgb,
            np.rot90(main[..., :3], k=1, axes=(0, 1)),
        )

    def test_unregister_reverts_to_unavailable(self, fake_repair_engine) -> None:
        roll_repair.unregister_engine()
        assert roll_repair.available() is False
        with pytest.raises(RuntimeError):
            roll_repair.repair(_acquisition(), roll_repair.RepairMode.EXACT)


class TestRepairMode:
    def test_values_are_exact_and_hybrid(self) -> None:
        assert roll_repair.RepairMode.EXACT.value == "exact"
        assert roll_repair.RepairMode.HYBRID.value == "hybrid"

    def test_constructible_from_its_own_string_value(self) -> None:
        """RollScanningService.write_frame coerces a persisted settings string
        back to a RepairMode this way; pin the round-trip."""
        assert roll_repair.RepairMode("exact") is roll_repair.RepairMode.EXACT
        assert roll_repair.RepairMode("hybrid") is roll_repair.RepairMode.HYBRID
