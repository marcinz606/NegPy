"""Forward-only, chunked orchestration for a registered full-film roll.

The workflow deliberately interleaves preview, approval, and archival capture
for each small sequential chunk.  That prevents a prepared whole roll from
rewinding from its final frame to the first archival frame.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Callable, Mapping, Protocol, Sequence, cast

from negpy.services.scanning.provenance import FullNegativeProvenanceManifest, PlanIdentity
from negpy.services.scanning.quality import inspect_tiff_payload


CHECKPOINT_FILENAME = "forward-roll.json"


def validate_full_negative_manifest(
    *,
    manifest_path: Path,
    identity: PlanIdentity,
    frame: int,
) -> str:
    """Reparse a frame manifest and revalidate every TIFF against live bytes."""

    payload = manifest_path.read_bytes()
    manifest = FullNegativeProvenanceManifest.from_json(payload.decode("utf-8"))
    if manifest.identity != identity or manifest.source_pair.logical_frame != frame:
        raise RuntimeError(f"frame {frame}: manifest belongs to another roll identity or logical frame")
    root = manifest_path.parent.parent
    artifacts = []
    for capture in manifest.captures:
        artifacts.extend(capture.artifacts.values())
    artifacts.extend(manifest.reconstruction.artifacts.values())
    for artifact in artifacts:
        live = inspect_tiff_payload(root / artifact.path)
        if not (
            live.sha256 == artifact.sha256
            and live.byte_length == artifact.byte_length
            and tuple(live.shape) == artifact.shape
            and live.dtype == artifact.dtype
            and live.page_count == artifact.page_count
            and live.payload_within_file == artifact.payload_within_file
            and live.x_resolution == artifact.x_resolution
            and live.y_resolution == artifact.y_resolution
            and live.resolution_unit == artifact.resolution_unit
        ):
            raise RuntimeError(f"frame {frame}: live artifact differs from its manifest: {artifact.path}")
    return sha256(payload).hexdigest()


class ForwardRollOperations(Protocol):
    """Scanner and persistence boundary used by :class:`ForwardRollWorkflow`."""

    def prepare_chunk(
        self,
        *,
        frames: Sequence[int],
        plan_dir: Path,
        identity: PlanIdentity,
        prior_plan_paths: Sequence[Path],
        stop: threading.Event,
        progress: Callable[..., None] | None,
    ) -> Path: ...

    def approve_chunk(
        self,
        *,
        plan_path: Path,
        allow_unverified_frames: Sequence[int],
    ) -> None: ...

    def capture_chunk(
        self,
        *,
        plan_path: Path,
        archive_dir: Path,
        identity: PlanIdentity,
        frames: Sequence[int],
        stop: threading.Event,
        progress: Callable[..., None] | None,
    ) -> Mapping[int, Path]: ...

    def validate_manifest(
        self,
        *,
        manifest_path: Path,
        identity: PlanIdentity,
        frame: int,
    ) -> str: ...


@dataclass(frozen=True)
class ForwardRollCheckpoint:
    """Live aggregate state derived from content-validated frame manifests."""

    path: Path
    identity: PlanIdentity
    device_id: str
    frames: tuple[int, ...]
    chunk_size: int
    calibration_excluded_frames: tuple[int, ...]
    completed_frames: tuple[int, ...]
    invalid_frames: tuple[int, ...]
    complete: bool


def _atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _strict_frames(frames: Sequence[int]) -> tuple[int, ...]:
    selected = tuple(frames)
    if not selected or any(type(frame) is not int or frame < 1 for frame in selected):
        raise ValueError("forward-roll frames must be positive integers")
    if tuple(sorted(set(selected))) != selected:
        raise ValueError("forward-roll frames must be unique and increasing")
    return selected


class ForwardRollWorkflow:
    """Run small sequential chunks without a whole-roll transport rewind."""

    _VERSION = 1

    def __init__(
        self,
        *,
        root: str | Path,
        device_id: str,
        identity: PlanIdentity,
        frames: Sequence[int],
        chunk_size: int,
        operations: ForwardRollOperations,
        resume: bool,
        calibration_excluded_frames: Sequence[int] = (),
    ) -> None:
        self._root = Path(root).expanduser().resolve()
        self._path = self._root / CHECKPOINT_FILENAME
        self._frames = _strict_frames(frames)
        if type(chunk_size) is not int or chunk_size < 1:
            raise ValueError("forward-roll chunk size must be a positive integer")
        if not isinstance(identity, PlanIdentity):
            raise ValueError("forward-roll identity must be a PlanIdentity")
        if type(device_id) is not str or not device_id:
            raise ValueError("forward-roll device ID must be nonempty")
        self._device_id = device_id
        self._identity = identity
        self._chunk_size = chunk_size
        self._operations = operations
        self._calibration_excluded_frames = _strict_frames(calibration_excluded_frames) if calibration_excluded_frames else ()
        unknown_exclusions = sorted(set(self._calibration_excluded_frames) - set(self._frames))
        if unknown_exclusions:
            raise ValueError("calibration exclusions are absent from this roll: " + ", ".join(map(str, unknown_exclusions)))

        expected_chunks = [
            {
                "frames": list(self._frames[index : index + chunk_size]),
                "plan_path": None,
                "archive_dir": str(Path("archive") / self._chunk_name(self._frames[index : index + chunk_size])),
                "manifest_paths": {},
            }
            for index in range(0, len(self._frames), chunk_size)
        ]
        if self._path.exists():
            if not resume:
                raise RuntimeError(
                    f"forward-roll root {self._root} already exists; pass resume only if the film was not ejected or reinserted"
                )
            self._payload = self._read_payload()
            persisted_chunks_value = self._payload.get("chunks")
            if type(persisted_chunks_value) is not list:
                raise ValueError(f"malformed forward-roll checkpoint {self._path}")
            persisted_chunk_frames: list[object] = []
            for raw_chunk_value in persisted_chunks_value:
                if type(raw_chunk_value) is not dict:
                    raise ValueError(f"malformed forward-roll checkpoint {self._path}")
                persisted_chunk_frames.append(cast(dict[str, object], raw_chunk_value).get("frames"))
            if (
                self._payload.get("version") != self._VERSION
                or self._payload.get("identity") != identity.to_dict()
                or self._payload.get("device_id") != device_id
                or self._payload.get("frames") != list(self._frames)
                or self._payload.get("chunk_size") != chunk_size
                or self._payload.get("calibration_excluded_frames") != list(self._calibration_excluded_frames)
                or persisted_chunk_frames != [chunk["frames"] for chunk in expected_chunks]
            ):
                raise RuntimeError(
                    "forward-roll resume identity, device, frames, chunk size, or calibration exclusions differ from its checkpoint"
                )
        else:
            if resume:
                raise RuntimeError(f"cannot resume: {self._path} does not exist")
            if self._root.exists() and any(self._root.iterdir()):
                raise RuntimeError(f"new forward-roll root {self._root} is not empty")
            self._payload: dict[str, object] = {
                "version": self._VERSION,
                "identity": identity.to_dict(),
                "device_id": device_id,
                "frames": list(self._frames),
                "chunk_size": chunk_size,
                "calibration_excluded_frames": list(self._calibration_excluded_frames),
                "chunks": expected_chunks,
                "completed_frames": [],
                "invalid_frames": [],
                "complete": False,
            }
            self._write()

    @staticmethod
    def _chunk_name(frames: Sequence[int]) -> str:
        return f"chunk-{frames[0]:03d}-{frames[-1]:03d}"

    def _read_payload(self) -> dict[str, object]:
        try:
            value = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"could not read forward-roll checkpoint {self._path}: {exc}") from exc
        if type(value) is not dict or type(value.get("chunks")) is not list:
            raise ValueError(f"malformed forward-roll checkpoint {self._path}")
        return value

    def _write(self) -> None:
        _atomic_write_json(self._path, self._payload)

    def _refresh(self) -> ForwardRollCheckpoint:
        completed: set[int] = set()
        invalid: set[int] = set()
        chunks = cast(list[object], self._payload["chunks"])
        for raw_chunk_value in chunks:
            if type(raw_chunk_value) is not dict:
                raise ValueError(f"malformed forward-roll checkpoint {self._path}")
            raw_chunk = cast(dict[str, object], raw_chunk_value)
            raw_manifests_value = raw_chunk.get("manifest_paths", {})
            raw_manifests = cast(dict[object, object], raw_manifests_value)
            if type(raw_manifests) is not dict:
                raise ValueError(f"malformed forward-roll checkpoint {self._path}")
            live_hashes: dict[str, str] = {}
            for raw_frame, raw_path in raw_manifests.items():
                if type(raw_frame) is not str or not raw_frame.isdigit():
                    raise ValueError(f"malformed forward-roll checkpoint {self._path}")
                frame = int(raw_frame)
                try:
                    if frame not in self._frames or type(raw_path) is not str:
                        raise ValueError("unexpected frame manifest")
                    manifest_path = self._root / raw_path
                    live_hashes[str(frame)] = self._operations.validate_manifest(
                        manifest_path=manifest_path,
                        identity=self._identity,
                        frame=frame,
                    )
                    completed.add(frame)
                except (OSError, RuntimeError, ValueError):
                    invalid.add(frame)
            raw_chunk["manifest_sha256"] = live_hashes
        ordered_completed = tuple(frame for frame in self._frames if frame in completed)
        exact = ordered_completed == self._frames and not invalid
        self._payload["completed_frames"] = list(ordered_completed)
        self._payload["invalid_frames"] = sorted(invalid)
        self._payload["complete"] = exact
        self._write()
        return ForwardRollCheckpoint(
            path=self._path,
            identity=self._identity,
            device_id=self._device_id,
            frames=self._frames,
            chunk_size=self._chunk_size,
            calibration_excluded_frames=self._calibration_excluded_frames,
            completed_frames=ordered_completed,
            invalid_frames=tuple(sorted(invalid)),
            complete=exact,
        )

    def run(
        self,
        *,
        approval_confirmed: bool,
        allow_unverified_frames: Sequence[int] = (),
        stop: threading.Event,
        progress: Callable[..., None] | None = None,
        max_chunks: int | None = None,
    ) -> ForwardRollCheckpoint:
        """Process one chunk completely before advancing to the next chunk."""

        if approval_confirmed is not True:
            raise RuntimeError(
                "forward-roll policy confirmation refused: pass --yes to accept automatic verification and the named frame exceptions"
            )
        if max_chunks is not None and (type(max_chunks) is not int or max_chunks < 1):
            raise ValueError("max_chunks must be a positive integer when supplied")
        overrides = _strict_frames(allow_unverified_frames) if allow_unverified_frames else ()
        unknown = sorted(set(overrides) - set(self._frames))
        if unknown:
            raise ValueError("unverified override frames are absent from this roll: " + ", ".join(map(str, unknown)))

        checkpoint = self._refresh()
        chunks = cast(list[object], self._payload["chunks"])
        prior_plan_paths: list[Path] = []
        processed_chunks = 0
        for raw_chunk_value in chunks:
            if type(raw_chunk_value) is not dict:
                raise ValueError(f"malformed forward-roll checkpoint {self._path}")
            raw_chunk = cast(dict[str, object], raw_chunk_value)
            raw_frames = raw_chunk.get("frames")
            if type(raw_frames) is not list or any(type(frame) is not int for frame in raw_frames):
                raise ValueError(f"malformed forward-roll checkpoint {self._path}")
            frames = tuple(cast(list[int], raw_frames))
            raw_plan_path = raw_chunk.get("plan_path")
            if set(frames).issubset(checkpoint.completed_frames):
                if type(raw_plan_path) is not str:
                    raise ValueError(f"completed forward-roll chunk has no plan path in {self._path}")
                prior_plan_paths.append(self._root / raw_plan_path)
                continue
            if stop.is_set():
                return checkpoint

            chunk_name = self._chunk_name(frames)
            plan_dir = self._root / "plans" / chunk_name
            plan_path = self._operations.prepare_chunk(
                frames=frames,
                plan_dir=plan_dir,
                identity=self._identity,
                prior_plan_paths=tuple(prior_plan_paths),
                stop=stop,
                progress=progress,
            ).resolve()
            try:
                relative_plan = plan_path.relative_to(self._root)
            except ValueError as exc:
                raise RuntimeError("forward-roll chunk plan escaped its roll root") from exc
            raw_chunk["plan_path"] = str(relative_plan)
            self._write()
            if stop.is_set():
                return self._refresh()

            self._operations.approve_chunk(
                plan_path=plan_path,
                allow_unverified_frames=tuple(frame for frame in overrides if frame in frames),
            )
            if stop.is_set():
                return self._refresh()

            archive_dir = self._root / str(raw_chunk["archive_dir"])
            manifests = self._operations.capture_chunk(
                plan_path=plan_path,
                archive_dir=archive_dir,
                identity=self._identity,
                frames=frames,
                stop=stop,
                progress=progress,
            )
            if set(manifests) != set(frames):
                missing = sorted(set(frames) - set(manifests))
                raise RuntimeError("forward-roll chunk is incomplete; manifests are missing for frame(s) " + ", ".join(map(str, missing)))
            relative_manifests: dict[str, str] = {}
            for frame, path in manifests.items():
                try:
                    relative_manifests[str(frame)] = str(Path(path).resolve().relative_to(self._root))
                except ValueError as exc:
                    raise RuntimeError("forward-roll frame manifest escaped its roll root") from exc
            raw_chunk["manifest_paths"] = relative_manifests
            checkpoint = self._refresh()
            invalid_chunk_frames = tuple(frame for frame in frames if frame not in checkpoint.completed_frames)
            if invalid_chunk_frames:
                raise RuntimeError(
                    "forward-roll chunk failed live manifest validation for frame(s) " + ", ".join(map(str, invalid_chunk_frames))
                )
            prior_plan_paths.append(plan_path)
            processed_chunks += 1
            if max_chunks is not None and processed_chunks >= max_chunks:
                return checkpoint

        checkpoint = self._refresh()
        if not checkpoint.complete:
            missing = sorted(set(self._frames) - set(checkpoint.completed_frames))
            raise RuntimeError(
                "forward roll is incomplete; live validated manifests are missing for frame(s) " + ", ".join(map(str, missing))
            )
        return checkpoint


__all__ = [
    "CHECKPOINT_FILENAME",
    "ForwardRollCheckpoint",
    "ForwardRollOperations",
    "ForwardRollWorkflow",
    "validate_full_negative_manifest",
]
