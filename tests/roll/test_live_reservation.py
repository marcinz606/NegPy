from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest

from negpy.services.roll import live_reservation
from negpy.services.roll.live_reservation import (
    OUTPUT_LOCK_NAME,
    ExclusiveReceiptReservation,
    FixedOutputLease,
    InventoryConflict,
    LiveReservationError,
    OwnershipLost,
    ReservationConflict,
)


def _in_progress() -> dict[str, object]:
    return {
        "schema": "negpy.test-live-reservation.v1",
        "status": "in_progress",
    }


def _lock_document() -> dict[str, object]:
    return {
        "reservation_id": "0123456789abcdef",
        "run_receipt": "/outside/output/run-receipt.json",
    }


def test_receipt_is_exclusive_durable_canonical_and_finalized_in_place(
    tmp_path: Path,
) -> None:
    path = tmp_path / "run-receipt.json"
    reservation = ExclusiveReceiptReservation.reserve(path, _in_progress())
    try:
        initial_inode = reservation.inode
        assert path.read_bytes() == (b'{"schema":"negpy.test-live-reservation.v1","status":"in_progress"}\n')

        final = {
            "schema": "negpy.test-live-reservation.v1",
            "status": "succeeded",
            "frames": [1, 2, 3, 4, 5, 6],
        }
        reservation.publish(final)

        assert reservation.inode == initial_inode
        assert (path.stat().st_dev, path.stat().st_ino) == initial_inode
        assert json.loads(path.read_bytes()) == final
    finally:
        reservation.close()


def test_preexisting_receipt_is_preserved_and_never_opened_for_writing(
    tmp_path: Path,
) -> None:
    path = tmp_path / "run-receipt.json"
    path.write_bytes(b"existing receipt")
    before = path.stat()

    with pytest.raises(ReservationConflict, match="already exists"):
        ExclusiveReceiptReservation.reserve(path, _in_progress())

    after = path.stat()
    assert path.read_bytes() == b"existing receipt"
    assert (after.st_dev, after.st_ino) == (before.st_dev, before.st_ino)


def test_receipt_reservation_requires_in_progress_status(tmp_path: Path) -> None:
    path = tmp_path / "run-receipt.json"

    with pytest.raises(LiveReservationError, match="exactly 'in_progress'"):
        ExclusiveReceiptReservation.reserve(path, {"status": "succeeded"})

    assert not path.exists()


def test_receipt_publication_refuses_replaced_path_and_preserves_collision(
    tmp_path: Path,
) -> None:
    path = tmp_path / "run-receipt.json"
    reservation = ExclusiveReceiptReservation.reserve(path, _in_progress())
    path.unlink()
    path.write_bytes(b"collision")

    try:
        with pytest.raises(OwnershipLost, match="no longer identifies"):
            reservation.publish({"status": "succeeded"})
        assert path.read_bytes() == b"collision"
    finally:
        reservation.close()


def test_receipt_publication_refuses_an_added_hard_link(tmp_path: Path) -> None:
    path = tmp_path / "run-receipt.json"
    linked = tmp_path / "unexpected-hard-link.json"
    reservation = ExclusiveReceiptReservation.reserve(path, _in_progress())
    os.link(path, linked)

    try:
        with pytest.raises(OwnershipLost, match="no longer identifies"):
            reservation.publish({"status": "succeeded"})
        assert json.loads(path.read_bytes()) == _in_progress()
        assert linked.read_bytes() == path.read_bytes()
    finally:
        reservation.close()


def test_oversized_final_receipt_is_rejected_before_mutating_reservation(
    tmp_path: Path,
) -> None:
    path = tmp_path / "run-receipt.json"
    reservation = ExclusiveReceiptReservation.reserve(
        path,
        {"status": "in_progress"},
        maximum_bytes=64,
    )
    before = path.read_bytes()

    try:
        with pytest.raises(LiveReservationError, match="size limit"):
            reservation.publish({"status": "failed", "error": "x" * 100})
        assert path.read_bytes() == before
    finally:
        reservation.close()


def test_fixed_output_lease_is_exclusive_visible_and_removed_on_release(
    tmp_path: Path,
) -> None:
    output = tmp_path / "outputs"
    output.mkdir()

    lease = FixedOutputLease.acquire(output, _lock_document())
    assert lease.lock_path == output / OUTPUT_LOCK_NAME
    assert json.loads(lease.lock_path.read_bytes()) == _lock_document()
    assert sorted(path.name for path in output.iterdir()) == [OUTPUT_LOCK_NAME]
    assert lease.assert_inventory(()).directories == ()

    lease.release()

    assert lease.released is True
    assert list(output.iterdir()) == []


def test_preexisting_fixed_lock_is_preserved(tmp_path: Path) -> None:
    output = tmp_path / "outputs"
    output.mkdir()
    lock = output / OUTPUT_LOCK_NAME
    lock.write_bytes(b"existing lock")
    before = lock.stat()

    with pytest.raises(ReservationConflict, match="already reserved"):
        FixedOutputLease.acquire(output, _lock_document())

    after = lock.stat()
    assert lock.read_bytes() == b"existing lock"
    assert (after.st_dev, after.st_ino) == (before.st_dev, before.st_ino)


def test_nonempty_output_fails_and_removes_only_its_new_lock(tmp_path: Path) -> None:
    output = tmp_path / "outputs"
    output.mkdir()
    existing = output / "existing.tif"
    existing.write_bytes(b"do not overwrite")

    with pytest.raises(InventoryConflict, match="unexpected files: existing.tif"):
        FixedOutputLease.acquire(output, _lock_document())

    assert existing.read_bytes() == b"do not overwrite"
    assert not (output / OUTPUT_LOCK_NAME).exists()


def test_inventory_allows_only_declared_files_and_implied_directories(
    tmp_path: Path,
) -> None:
    output = tmp_path / "outputs"
    output.mkdir()
    lease = FixedOutputLease.acquire(output, _lock_document())
    evidence_dir = output / ".negpy-native-builder" / "receipt-a"
    evidence_dir.mkdir(parents=True)
    first = evidence_dir / "evidence.json"
    first.write_bytes(b"one")

    try:
        first_snapshot = lease.assert_inventory([first])
        assert first_snapshot.directories == (
            ".negpy-native-builder",
            os.path.join(".negpy-native-builder", "receipt-a"),
        )

        second = output / "acceptance_slot01.tif"
        second.write_bytes(b"two")
        second_snapshot = lease.assert_inventory(
            [first, second],
            previous=first_snapshot,
        )
        assert dict(second_snapshot.files)[os.fspath(second.relative_to(output))].size == 3

        rogue = output / "acceptance_slot02.tif"
        rogue.write_bytes(b"collision")
        with pytest.raises(InventoryConflict, match="unexpected files: acceptance_slot02.tif"):
            lease.assert_inventory([first, second], previous=second_snapshot)
        rogue.unlink()
    finally:
        lease.release()


def test_inventory_ignores_only_regular_finder_metadata(tmp_path: Path) -> None:
    output = tmp_path / "outputs"
    output.mkdir()
    lease = FixedOutputLease.acquire(output, _lock_document())
    artifact = output / "acceptance_slot01.tif"
    artifact.write_bytes(b"captured")
    (output / ".DS_Store").write_bytes(b"finder metadata")

    try:
        snapshot = lease.assert_inventory([artifact])
    finally:
        lease.release()

    assert [name for name, _identity in snapshot.files] == [
        OUTPUT_LOCK_NAME,
        artifact.name,
    ]


def test_inventory_detects_mutation_of_previously_owned_file(tmp_path: Path) -> None:
    output = tmp_path / "outputs"
    output.mkdir()
    lease = FixedOutputLease.acquire(output, _lock_document())
    artifact = output / "acceptance_slot01.tif"
    artifact.write_bytes(b"before")
    snapshot = lease.assert_inventory([artifact])

    try:
        artifact.write_bytes(b"after!")
        with pytest.raises(InventoryConflict, match="previously owned output changed"):
            lease.assert_inventory([artifact], previous=snapshot)
    finally:
        lease.release()


def test_inventory_rejects_symlink_and_owned_path_escape(tmp_path: Path) -> None:
    output = tmp_path / "outputs"
    output.mkdir()
    outside = tmp_path / "outside.tif"
    outside.write_bytes(b"outside")
    lease = FixedOutputLease.acquire(output, _lock_document())
    symlink = output / "linked.tif"
    symlink.symlink_to(outside)

    try:
        with pytest.raises(InventoryConflict, match="symbolic link"):
            lease.assert_inventory([symlink])
        symlink.unlink()
        with pytest.raises(InventoryConflict, match="escapes"):
            lease.assert_inventory([outside])
    finally:
        lease.release()


def test_output_path_replacement_is_detected_but_original_lock_is_released(
    tmp_path: Path,
) -> None:
    output = tmp_path / "outputs"
    moved = tmp_path / "moved-outputs"
    output.mkdir()
    lease = FixedOutputLease.acquire(output, _lock_document())
    output.rename(moved)
    output.mkdir()

    with pytest.raises(OwnershipLost, match="pathname no longer identifies"):
        lease.assert_owned()

    lease.release()

    assert not (moved / OUTPUT_LOCK_NAME).exists()
    assert list(output.iterdir()) == []


def test_release_refuses_to_unlink_replacement_lock(tmp_path: Path) -> None:
    output = tmp_path / "outputs"
    output.mkdir()
    lease = FixedOutputLease.acquire(output, _lock_document())
    lease.lock_path.unlink()
    lease.lock_path.write_bytes(b"replacement lock")

    with pytest.raises(OwnershipLost, match="no longer exclusively owned"):
        lease.release()

    assert lease.released is True
    assert lease.lock_path.read_bytes() == b"replacement lock"


def test_verified_release_checks_inventory_around_final_publication(
    tmp_path: Path,
) -> None:
    output = tmp_path / "outputs"
    output.mkdir()
    artifact = output / "acceptance_slot01.tif"
    lease = FixedOutputLease.acquire(output, _lock_document())
    artifact.write_bytes(b"stable")
    snapshot = lease.assert_inventory([artifact])
    finalized: list[bool] = []

    result = lease.release_verified(
        [artifact],
        previous=snapshot,
        finalize=lambda: finalized.append(True),
    )

    assert finalized == [True]
    assert lease.released is True
    assert [name for name, _identity in result.files] == [artifact.name]
    assert sorted(path.name for path in output.iterdir()) == [artifact.name]


def test_verified_release_rejects_output_path_swap_during_final_publication(
    tmp_path: Path,
) -> None:
    output = tmp_path / "outputs"
    moved = tmp_path / "moved-outputs"
    output.mkdir()
    artifact = output / "acceptance_slot01.tif"
    lease = FixedOutputLease.acquire(output, _lock_document())
    artifact.write_bytes(b"stable")
    snapshot = lease.assert_inventory([artifact])

    def replace_output_path() -> None:
        output.rename(moved)
        output.mkdir()

    with pytest.raises(OwnershipLost, match="pathname no longer identifies"):
        lease.release_verified(
            [artifact],
            previous=snapshot,
            finalize=replace_output_path,
        )

    assert lease.released is True
    assert not (moved / OUTPUT_LOCK_NAME).exists()
    assert (moved / artifact.name).read_bytes() == b"stable"
    assert list(output.iterdir()) == []


def test_receipt_lock_and_directories_are_fsynced(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_fsync = os.fsync
    synced_modes: list[int] = []

    def recording_fsync(descriptor: int) -> None:
        synced_modes.append(stat.S_IFMT(os.fstat(descriptor).st_mode))
        real_fsync(descriptor)

    monkeypatch.setattr(live_reservation.os, "fsync", recording_fsync)
    receipt = ExclusiveReceiptReservation.reserve(
        tmp_path / "run-receipt.json",
        _in_progress(),
    )
    receipt.publish({"status": "succeeded"})
    receipt.close()
    output = tmp_path / "outputs"
    output.mkdir()
    lease = FixedOutputLease.acquire(output, _lock_document())
    lease.release()

    assert synced_modes.count(stat.S_IFREG) >= 3
    assert synced_modes.count(stat.S_IFDIR) >= 4
