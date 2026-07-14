"""Transport-safe, chunked full-roll orchestration contracts."""

from __future__ import annotations

import hashlib
import threading
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from negpy.services.scanning.forward_roll import ForwardRollWorkflow
from negpy.services.scanning.provenance import PlanIdentity


IDENTITY = PlanIdentity(
    "11111111-1111-4111-8111-111111111111",
    "22222222-2222-4222-8222-222222222222",
    "33333333-3333-4333-8333-333333333333",
)


@dataclass
class FakeForwardOperations:
    physical_frames: list[int] = field(default_factory=list)
    calls: list[tuple[str, tuple[int, ...]]] = field(default_factory=list)
    invalid_frame: int | None = None
    stop_after_prepare_once: bool = False
    prior_contexts: list[tuple[str, ...]] = field(default_factory=list)

    def prepare_chunk(self, *, frames, plan_dir, identity, prior_plan_paths, stop, progress):
        assert identity == IDENTITY
        self.prior_contexts.append(tuple(path.parent.name for path in prior_plan_paths))
        assert tuple(path.parent.name for path in prior_plan_paths) == tuple(
            f"chunk-{start:03d}-{end:03d}" for start, end in ((1, 3), (4, 6), (7, 8))[: len(prior_plan_paths)]
        )
        selected = tuple(frames)
        self.calls.append(("prepare", selected))
        # Wide pass, then aligned pass.  A chunk-local rewind is bounded.
        self.physical_frames.extend(selected)
        self.physical_frames.extend(selected)
        plan_dir.mkdir(parents=True, exist_ok=True)
        if self.stop_after_prepare_once:
            self.stop_after_prepare_once = False
            stop.set()
        return plan_dir / "roll-scan.json"

    def approve_chunk(self, *, plan_path, allow_unverified_frames):
        selected = _chunk_from_path(plan_path.parent)
        self.calls.append(("approve", selected))

    def capture_chunk(self, *, plan_path, archive_dir, identity, frames, stop, progress):
        assert identity == IDENTITY
        selected = tuple(frames)
        self.calls.append(("capture", selected))
        manifests = {}
        for frame in selected:
            self.physical_frames.extend((frame, frame + 1))
            path = archive_dir / "manifests" / f"frame{frame:03d}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"frame={frame}\n", encoding="utf-8")
            manifests[frame] = path
        return manifests

    def validate_manifest(self, *, manifest_path, identity, frame):
        assert identity == IDENTITY
        if frame == self.invalid_frame:
            raise RuntimeError("synthetic missing artifact")
        assert manifest_path.read_text(encoding="utf-8") == f"frame={frame}\n"
        return hashlib.sha256(manifest_path.read_bytes()).hexdigest()


def _chunk_from_path(path: Path) -> tuple[int, ...]:
    _, start, end = path.name.split("-")
    return tuple(range(int(start), int(end) + 1))


def test_forward_roll_previews_approves_and_archives_each_chunk_before_advancing(tmp_path: Path) -> None:
    operations = FakeForwardOperations()
    workflow = ForwardRollWorkflow(
        root=tmp_path / "roll",
        device_id="coolscan3:usb:test",
        identity=IDENTITY,
        frames=tuple(range(1, 9)),
        chunk_size=3,
        operations=operations,
        resume=False,
    )

    checkpoint = workflow.run(
        approval_confirmed=True,
        allow_unverified_frames=(1,),
        stop=threading.Event(),
    )

    assert operations.calls == [
        ("prepare", (1, 2, 3)),
        ("approve", (1, 2, 3)),
        ("capture", (1, 2, 3)),
        ("prepare", (4, 5, 6)),
        ("approve", (4, 5, 6)),
        ("capture", (4, 5, 6)),
        ("prepare", (7, 8)),
        ("approve", (7, 8)),
        ("capture", (7, 8)),
    ]
    reverse_moves = [before - after for before, after in zip(operations.physical_frames, operations.physical_frames[1:])]
    assert max(reverse_moves) <= 2
    assert checkpoint.complete is True
    assert checkpoint.completed_frames == tuple(range(1, 9))


def test_forward_roll_stops_before_advancing_when_a_chunk_manifest_is_not_live(tmp_path: Path) -> None:
    operations = FakeForwardOperations(invalid_frame=2)
    workflow = ForwardRollWorkflow(
        root=tmp_path / "roll",
        device_id="coolscan3:usb:test",
        identity=IDENTITY,
        frames=tuple(range(1, 7)),
        chunk_size=3,
        operations=operations,
        resume=False,
    )

    try:
        workflow.run(
            approval_confirmed=True,
            stop=threading.Event(),
        )
    except RuntimeError as exc:
        assert "frame(s) 2" in str(exc)
    else:
        raise AssertionError("an invalid frame manifest must prevent roll completion")

    assert operations.calls == [
        ("prepare", (1, 2, 3)),
        ("approve", (1, 2, 3)),
        ("capture", (1, 2, 3)),
    ]
    checkpoint = (tmp_path / "roll" / "forward-roll.json").read_text(encoding="utf-8")
    assert '"complete": false' in checkpoint


@pytest.mark.parametrize(
    "changed",
    [
        {
            "identity": PlanIdentity(
                "44444444-4444-4444-8444-444444444444",
                "55555555-5555-4555-8555-555555555555",
                "66666666-6666-4666-8666-666666666666",
            )
        },
        {"frames": (1, 2, 3, 4)},
        {"chunk_size": 2},
        {"calibration_excluded_frames": ()},
    ],
)
def test_forward_roll_resume_refuses_identity_frame_chunk_or_calibration_changes(tmp_path: Path, changed: dict) -> None:
    root = tmp_path / "roll"
    common = {
        "root": root,
        "device_id": "coolscan3:usb:test",
        "identity": IDENTITY,
        "frames": (1, 2, 3),
        "chunk_size": 3,
        "calibration_excluded_frames": (1,),
        "operations": FakeForwardOperations(),
    }
    ForwardRollWorkflow(**common, resume=False)

    with pytest.raises(RuntimeError, match="identity, device, frames, chunk size, or calibration exclusions"):
        ForwardRollWorkflow(**(common | changed), resume=True)


def test_resume_does_not_use_an_incomplete_chunk_plan_as_its_own_calibration_context(tmp_path: Path) -> None:
    root = tmp_path / "roll"
    operations = FakeForwardOperations(stop_after_prepare_once=True)
    stop = threading.Event()
    common = {
        "root": root,
        "device_id": "coolscan3:usb:test",
        "identity": IDENTITY,
        "frames": (1, 2, 3, 4, 5, 6),
        "chunk_size": 3,
        "operations": operations,
    }

    interrupted = ForwardRollWorkflow(**common, resume=False).run(
        approval_confirmed=True,
        stop=stop,
    )
    assert interrupted.complete is False
    assert interrupted.completed_frames == ()

    stop.clear()
    completed = ForwardRollWorkflow(**common, resume=True).run(
        approval_confirmed=True,
        stop=stop,
    )

    assert completed.complete is True
    assert operations.prior_contexts[:2] == [(), ()]
    assert operations.prior_contexts[2] == ("chunk-001-003",)
