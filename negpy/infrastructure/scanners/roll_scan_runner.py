#!/usr/bin/env python3
"""Fail-closed command-line bridge for registered Coolscan roll scans.

Preview preparation and fine scanning are separate commands so a person can
inspect the aligned previews before any archival transfer starts.  Signals set
an outer stop event; they never cancel an in-flight scanner read.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import threading
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from types import FrameType
from typing import Callable, Iterator, Sequence
from uuid import uuid4

import numpy as np
import tifffile

from negpy.infrastructure.scanners.params import ScanParams
from negpy.services.scanning import (
    PreviewRollRegistration,
    RollRegistrationConfig,
    RollScanPlan,
    RollScanService,
    RollStage,
    ScannerService,
)
from negpy.services.scanning.roll_service import PreviewScanRecipe, _read_plan
from negpy.services.scanning.roll_service import approved_plan_binding_sha256
from negpy.services.scanning.provenance import PlanIdentity
from negpy.services.scanning.roll_service import _atomic_write_json
from negpy.services.scanning.quality import inspect_tiff_payload
from negpy.services.scanning.forward_roll import (
    CHECKPOINT_FILENAME as FORWARD_ROLL_CHECKPOINT_FILENAME,
    ForwardRollOperations,
    ForwardRollWorkflow,
    validate_full_negative_manifest,
)


LIVE_ENV = "NEGPY_ROLL_LIVE"
FULL_NEGATIVE_IDENTITY_FILENAME = "full-negative-identity.json"


def parse_frames(value: str) -> tuple[int, ...]:
    """Parse a strictly increasing list such as ``2-4,7,10-12``."""

    frames: list[int] = []
    try:
        for raw_part in value.split(","):
            part = raw_part.strip()
            if not part:
                raise ValueError
            if "-" not in part:
                frames.append(int(part))
                continue
            start_text, separator, end_text = part.partition("-")
            if not separator or "-" in end_text:
                raise ValueError
            start = int(start_text)
            end = int(end_text)
            if end < start:
                raise ValueError
            frames.extend(range(start, end + 1))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("frames must look like 2-4,7,10-12") from exc

    normalized = tuple(frames)
    if not normalized or any(frame < 1 for frame in normalized):
        raise argparse.ArgumentTypeError("frames must be positive integers")
    if tuple(sorted(set(normalized))) != normalized:
        raise argparse.ArgumentTypeError("frames must be unique and increasing")
    return normalized


def _registration_signature(excluded_frames: Sequence[int] = ()) -> dict[str, object]:
    config = RollRegistrationConfig(excluded_frames=frozenset(excluded_frames))
    return dict(PreviewRollRegistration(config).registration_signature)


def _context_signature_matches(signature: object, *, expected: dict[str, object] | None = None) -> bool:
    expected = _registration_signature() if expected is None else expected
    if signature == expected:
        return True
    # Version 3 changes only how a *new* clipped-boundary registration is
    # positioned.  Version-2 wide TIFFs remain valid calibration samples and
    # can safely seed the new roll-wide model.
    legacy = json.loads(json.dumps(expected))
    legacy["version"] = 2
    config = legacy["config"]
    for field in ("frame_pitch_mm", "maximum_subframe_mm", "maximum_br_y_device_px"):
        config.pop(field)
    return signature == legacy


def _build_service(excluded_frames: Sequence[int] = ()) -> RollScanService:
    return RollScanService(
        scanner=ScannerService(),
        registration=PreviewRollRegistration(RollRegistrationConfig(excluded_frames=frozenset(excluded_frames))),
    )


def _load_context_previews(
    plan_paths: Sequence[Path],
    *,
    device_id: str,
    selected_frames: Sequence[int],
    preview_params: ScanParams,
    registration_signature: dict[str, object] | None = None,
) -> dict[int, np.ndarray]:
    """Load validated committed wides from earlier chunks of this roll."""

    expected_recipe = PreviewScanRecipe.from_params(preview_params)
    selected = set(selected_frames)
    previews: dict[int, np.ndarray] = {}
    context_identity: PlanIdentity | None = None
    for raw_path in plan_paths:
        path = raw_path.expanduser().resolve()
        plan = _read_plan(path)
        if context_identity is None:
            context_identity = plan.identity
        elif plan.identity != context_identity:
            raise ValueError("context plans do not belong to the same roll insertion/session")
        if plan.device_id != device_id:
            raise ValueError(f"context plan {path} belongs to {plan.device_id!r}, not {device_id!r}")
        if plan.preview_recipe != expected_recipe:
            raise ValueError(f"context plan {path} uses a different preview recipe")
        if not _context_signature_matches(plan.registration_signature, expected=registration_signature):
            raise ValueError(f"context plan {path} uses a different registration policy")
        for record in plan.frames:
            if record.frame in selected:
                raise ValueError(f"context plan {path} overlaps selected frame {record.frame}")
            if record.frame in previews:
                raise ValueError(f"duplicate context frame {record.frame}")
            if record.wide_path is None or record.wide_rgb_shape is None or record.wide_dtype is None:
                raise ValueError(f"context plan {path} has no committed wide preview for frame {record.frame}")
            preview_path = path.parent / record.wide_path
            try:
                preview = tifffile.imread(preview_path)
            except (OSError, ValueError, tifffile.TiffFileError) as exc:
                raise ValueError(f"could not read context preview {preview_path}: {exc}") from exc
            if tuple(int(value) for value in preview.shape) != record.wide_rgb_shape:
                raise ValueError(f"context preview {preview_path} shape differs from its plan")
            if np.dtype(preview.dtype).name != record.wide_dtype or preview.dtype != np.uint16:
                raise ValueError(f"context preview {preview_path} is not the recorded uint16 payload")
            inspection = inspect_tiff_payload(preview_path)
            if (
                inspection.sha256 != record.wide_sha256
                or inspection.x_resolution != (record.wide_dpi, 1)
                or inspection.y_resolution != (record.wide_dpi, 1)
                or inspection.resolution_unit != "INCH"
            ):
                raise ValueError(f"context preview {preview_path} hash/DPI evidence differs from its plan")
            previews[record.frame] = preview
    return dict(sorted(previews.items()))


def _context_plan_identity(plan_paths: Sequence[Path]) -> PlanIdentity | None:
    identities = {_read_plan(path.expanduser().resolve()).identity for path in plan_paths}
    if len(identities) > 1:
        raise ValueError("context plans do not belong to the same roll insertion/session")
    return next(iter(identities), None)


def _discover_coolscan_device() -> str:
    scanner = ScannerService()
    devices = scanner.list_devices()
    matches = [device.id for device in devices if device.id.startswith("coolscan3:")]
    if not matches:
        raise RuntimeError(f"no direct coolscan3 device found (saw: {[device.id for device in devices] or 'none'})")
    if len(matches) > 1:
        raise RuntimeError(f"multiple direct coolscan3 devices found; pass --device explicitly: {matches}")
    return matches[0]


def _require_live(args: argparse.Namespace) -> None:
    if not args.live or os.environ.get(LIVE_ENV) != "YES":
        raise RuntimeError(f"hardware command refused: pass --live and set {LIVE_ENV}=YES")


@contextmanager
def _stop_between_transfers(stop: threading.Event) -> Iterator[None]:
    previous: dict[
        signal.Signals,
        signal.Handlers | int | Callable[[int, FrameType | None], object] | None,
    ] = {}

    def request_stop(signum: int, _frame: FrameType | None) -> None:
        stop.set()
        name = signal.Signals(signum).name
        print(f"\n{name} received; stopping after the current scanner transfer completes", file=sys.stderr)

    for signum in (signal.SIGINT, signal.SIGTERM):
        previous[signum] = signal.getsignal(signum)
        signal.signal(signum, request_stop)
    try:
        yield
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


class _ProgressPrinter:
    def __init__(self) -> None:
        self._last: tuple[RollStage, int, int] | None = None

    def __call__(self, stage: RollStage, frame: int, fraction: float) -> None:
        bucket = max(0, min(10, int(fraction * 10)))
        state = (stage, frame, bucket)
        if state == self._last:
            return
        self._last = state
        print(f"{stage.value}: frame {frame:03d}: {bucket * 10:3d}%", file=sys.stderr)


class _RunnerForwardOperations:
    """Adapt the existing roll service to the forward-only workflow boundary."""

    def __init__(
        self,
        service: RollScanService,
        *,
        device_id: str,
        registration_signature: dict[str, object],
    ) -> None:
        self._service = service
        self._device_id = device_id
        self._registration_signature = registration_signature

    def prepare_chunk(
        self,
        *,
        frames: Sequence[int],
        plan_dir: Path,
        identity: PlanIdentity,
        prior_plan_paths: Sequence[Path],
        stop: threading.Event,
        progress: Callable[..., None] | None,
    ) -> Path:
        preview_params = _preview_params()
        calibration_previews = _load_context_previews(
            prior_plan_paths,
            device_id=self._device_id,
            selected_frames=frames,
            preview_params=preview_params,
            registration_signature=self._registration_signature,
        )
        plan = self._service.prepare_roll(
            device_id=self._device_id,
            frames=frames,
            output_dir=plan_dir,
            preview_params=preview_params,
            stop=stop,
            progress=progress,
            calibration_previews=calibration_previews,
            identity=identity,
        )
        if plan.identity != identity:
            raise RuntimeError("prepared chunk did not retain the shared forward-roll identity")
        return plan_dir / "roll-scan.json"

    def approve_chunk(
        self,
        *,
        plan_path: Path,
        allow_unverified_frames: Sequence[int],
    ) -> None:
        plan = _read_plan(plan_path)
        requested_overrides = tuple(allow_unverified_frames)
        if plan.approved:
            if plan.visual_override_frames != requested_overrides:
                raise RuntimeError("approved chunk has different frame-specific visual overrides")
            return
        self._service.approve_roll(
            plan_path,
            allow_unverified_frames=requested_overrides,
        )

    def capture_chunk(
        self,
        *,
        plan_path: Path,
        archive_dir: Path,
        identity: PlanIdentity,
        frames: Sequence[int],
        stop: threading.Event,
        progress: Callable[..., None] | None,
    ) -> dict[int, Path]:
        results = self._service.scan_full_negative(
            plan_path=plan_path,
            output_dir=archive_dir,
            identity=identity,
            fine_params=_fine_params(4000),
            stop=stop,
            frames=frames,
            progress=progress,
        )
        return {frame: result.manifest_path for frame, result in results.items()}

    def validate_manifest(
        self,
        *,
        manifest_path: Path,
        identity: PlanIdentity,
        frame: int,
    ) -> str:
        return validate_full_negative_manifest(
            manifest_path=manifest_path,
            identity=identity,
            frame=frame,
        )

    def eject(self) -> bool:
        """Trigger a capability-gated eject of the just-completed roll."""
        return self._service.eject(self._device_id)


def _plan_summary(plan: RollScanPlan, *, plan_path: Path | None = None) -> dict[str, object]:
    frames: list[dict[str, object]] = []
    for record in plan.frames:
        registration = record.registration
        verification = record.verification
        frames.append(
            {
                "frame": record.frame,
                "wide_path": record.wide_path,
                "aligned_path": record.aligned_path,
                "geometry": asdict(registration.geometry) if registration is not None else None,
                "registration_confidence": registration.confidence if registration is not None else None,
                "verification": asdict(verification) if verification is not None else None,
                "verification_passed": verification.passed if verification is not None else None,
                "fine_path": record.fine_path,
            }
        )
    return {
        "plan_path": str(plan_path) if plan_path is not None else None,
        "device_id": plan.device_id,
        "stage": plan.stage.value,
        "approved": plan.approved,
        "frames": frames,
    }


def _print_plan(plan: RollScanPlan, *, plan_path: Path | None = None) -> None:
    print(json.dumps(_plan_summary(plan, plan_path=plan_path), indent=2))


def _preview_params() -> ScanParams:
    return ScanParams(
        dpi=400,
        depth=16,
        capture_ir=False,
        autofocus=False,
        samples_per_scan=1,
        auto_exposure=False,
        area=None,
    )


def _fine_params(dpi: int = 4000) -> ScanParams:
    return ScanParams(
        dpi=dpi,
        depth=16,
        capture_ir=True,
        autofocus=True,
        samples_per_scan=4,
        auto_exposure=True,
        area=None,
    )


def _full_negative_identity(
    output: Path,
    *,
    resume: bool,
    identity: PlanIdentity,
    approved_plan_binding: str,
) -> PlanIdentity:
    root = output.expanduser().resolve()
    path = root / FULL_NEGATIVE_IDENTITY_FILENAME
    if path.exists():
        if not resume:
            raise RuntimeError(
                f"full-negative output {root} already has an insertion identity; pass --resume only if the film was not ejected or reinserted"
            )
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"could not read full-negative identity {path}: {exc}") from exc
        if type(payload) is not dict or set(payload) != {"identity", "approved_plan_binding_sha256"}:
            raise ValueError(f"malformed full-negative identity {path}")
        persisted = PlanIdentity.from_dict(payload["identity"])
        if persisted != identity or payload["approved_plan_binding_sha256"] != approved_plan_binding:
            raise RuntimeError("full-negative output belongs to a different approved roll plan")
        return persisted
    if resume:
        raise RuntimeError(f"cannot resume: {path} does not exist")
    _atomic_write_json(
        path,
        {
            "identity": identity.to_dict(),
            "approved_plan_binding_sha256": approved_plan_binding,
        },
    )
    return identity


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="capture wide and aligned 400 dpi previews")
    prepare.add_argument("--frames", type=parse_frames, required=True, help="explicit exposures, for example 2-4")
    prepare.add_argument("--output", type=Path, required=True, help="new or resumable roll-plan directory")
    prepare.add_argument(
        "--context-plan",
        type=Path,
        action="append",
        default=[],
        help="earlier same-roll chunk plan whose wide previews seed roll-wide registration; repeat as needed",
    )
    prepare.add_argument("--device", help="direct coolscan3 device ID; default is fresh discovery")
    prepare.add_argument("--live", action="store_true", help=f"allow hardware I/O when {LIVE_ENV}=YES")

    status = subparsers.add_parser("status", help="read and summarize a saved roll plan without scanner I/O")
    status.add_argument("--plan", type=Path, required=True)

    approve = subparsers.add_parser("approve", help="approve previews under automatic verification plus named exceptions")
    approve.add_argument("--plan", type=Path, required=True)
    approve.add_argument("--allow-unverified", type=parse_frames, default=(), help="visually accepted frame numbers")
    approve.add_argument("--yes", action="store_true", help="confirm the verification policy and named exceptions")

    fine = subparsers.add_parser("fine", help="scan explicitly selected approved frames at 4000 dpi with RGB4x + RGBI1x")
    fine.add_argument("--plan", type=Path, required=True)
    fine.add_argument("--frames", type=parse_frames, required=True, help="explicit approved exposures")
    fine.add_argument("--dpi", type=int, choices=(1000, 4000), default=4000, help="1000 dpi discriminator or 4000 dpi archival scan")
    fine.add_argument("--allow-indeterminate-smear", type=parse_frames, default=(), help="visually accepted smear-QC frames")
    fine.add_argument("--live", action="store_true", help=f"allow hardware I/O when {LIVE_ENV}=YES")

    full = subparsers.add_parser(
        "full-negative",
        help="capture approved full negatives as RGB4x plus aligned IR using a bounded successor tail",
    )
    full.add_argument("--plan", type=Path, required=True)
    full.add_argument("--output", type=Path, required=True)
    full.add_argument("--frames", type=parse_frames, required=True, help="explicit approved exposures")
    full.add_argument(
        "--resume",
        action="store_true",
        help="reuse this output identity only when the film has not been ejected or reinserted",
    )
    full.add_argument("--live", action="store_true", help=f"allow hardware I/O when {LIVE_ENV}=YES")

    forward = subparsers.add_parser(
        "forward-roll",
        help="preview, explicitly approve, and archive small sequential chunks without a whole-roll rewind",
    )
    forward.add_argument(
        "--frames",
        type=parse_frames,
        default=tuple(range(1, 38)),
        help="explicit exposures (default: 1-37)",
    )
    forward.add_argument("--output", type=Path, required=True, help="one root for chunk plans, archives, and aggregate state")
    forward.add_argument("--chunk-size", type=int, default=3, help="sequential preview/archive chunk size (default: 3)")
    forward.add_argument("--device", help="direct coolscan3 device ID; default is fresh discovery")
    forward.add_argument(
        "--allow-unverified",
        type=parse_frames,
        default=(),
        help="explicitly authorized frame exceptions to automatic alignment verification",
    )
    forward.add_argument(
        "--exclude-calibration",
        type=parse_frames,
        default=(),
        help="damaged frames to capture normally but exclude from the roll-wide calibration fit",
    )
    forward.add_argument(
        "--yes",
        action="store_true",
        help="confirm automatic verification and the explicit --allow-unverified exceptions for this run",
    )
    forward.add_argument(
        "--max-chunks",
        type=int,
        help="stop cleanly after this many newly completed chunks (use 1 for a deterministic review pause)",
    )
    forward.add_argument(
        "--resume",
        action="store_true",
        help="resume only if the film has not been ejected or reinserted",
    )
    forward.add_argument("--live", action="store_true", help=f"allow hardware I/O when {LIVE_ENV}=YES")
    forward.add_argument(
        "--no-eject",
        action="store_true",
        help="do not eject the film once the entire declared roll completes (default: eject, capability-gated)",
    )
    return parser


def _forward_roll_identity(output: Path, *, resume: bool) -> PlanIdentity:
    checkpoint_path = output.expanduser().resolve() / FORWARD_ROLL_CHECKPOINT_FILENAME
    if resume:
        try:
            payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            return PlanIdentity.from_dict(payload["identity"])
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"cannot recover forward-roll identity from {checkpoint_path}: {exc}") from exc
    return PlanIdentity(str(uuid4()), str(uuid4()), str(uuid4()))


def _eject_after_forward_roll(operations: object, *, complete: bool, requested: bool) -> bool:
    """Fail-soft, capability-gated eject after a fully completed forward-roll.

    Mirrors Nikon Scan ejecting film at batch completion instead of leaving
    it parked (the LS-5000 feeder auto-parks a few minutes after any session
    closes, and a parked feeder needs a power-cycle to recover mid-roll).
    Fires only when this call finished the *entire* declared roll — a
    ``--max-chunks``-truncated pause leaves ``complete`` False and must not
    eject, since the next invocation resumes the same insertion.

    Never raises: an eject failure must not take down an otherwise-complete,
    already-persisted archive. ``operations`` is duck-typed so plain
    ``ForwardRollOperations`` fakes without an ``eject`` method are a clean
    no-op, matching the device-level capability gate in SaneBackend.eject.
    """
    if not complete or not requested:
        return False
    eject = getattr(operations, "eject", None)
    if not callable(eject):
        return False
    try:
        ejected = bool(eject())
    except Exception as exc:
        print(f"eject after forward-roll completion failed (archive unaffected): {exc}", file=sys.stderr)
        return False
    if not ejected:
        print("forward-roll complete; device reports no eject capability, film left in the feeder", file=sys.stderr)
    return ejected


def main(
    argv: Sequence[str] | None = None,
    *,
    service_factory: Callable[[], RollScanService] | None = None,
    forward_operations_factory: Callable[[RollScanService, str], ForwardRollOperations] | None = None,
) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    make_service = service_factory or _build_service

    try:
        if args.command == "status":
            path = args.plan.expanduser().resolve()
            _print_plan(_read_plan(path), plan_path=path)
            return 0

        if args.command == "prepare":
            _require_live(args)
            service = make_service()
            device_id = args.device or _discover_coolscan_device()
            stop = threading.Event()
            preview_params = _preview_params()
            calibration_previews = _load_context_previews(
                args.context_plan,
                device_id=device_id,
                selected_frames=args.frames,
                preview_params=preview_params,
            )
            identity = _context_plan_identity(args.context_plan)
            with _stop_between_transfers(stop):
                plan = service.prepare_roll(
                    device_id=device_id,
                    frames=args.frames,
                    output_dir=args.output,
                    preview_params=preview_params,
                    stop=stop,
                    progress=_ProgressPrinter(),
                    calibration_previews=calibration_previews,
                    identity=identity,
                )
            plan_path = args.output.expanduser().resolve() / "roll-scan.json"
            _print_plan(plan, plan_path=plan_path)
            return 0

        if args.command == "approve":
            if not args.yes:
                raise RuntimeError("approval refused: pass --yes to confirm automatic verification and named exceptions")
            service = make_service()
            path = args.plan.expanduser().resolve()
            plan = service.approve_roll(path, allow_unverified_frames=args.allow_unverified)
            _print_plan(plan, plan_path=path)
            return 0

        if args.command == "fine":
            _require_live(args)
            service = make_service()
            stop = threading.Event()
            path = args.plan.expanduser().resolve()
            with _stop_between_transfers(stop):
                plan = service.scan_fine(
                    plan_path=path,
                    fine_params=_fine_params(args.dpi),
                    stop=stop,
                    frames=args.frames,
                    progress=_ProgressPrinter(),
                    allow_indeterminate_smear_frames=args.allow_indeterminate_smear,
                )
            _print_plan(plan, plan_path=path)
            return 0

        if args.command == "full-negative":
            _require_live(args)
            service = make_service()
            stop = threading.Event()
            path = args.plan.expanduser().resolve()
            approved_plan = _read_plan(path)
            approved_binding = approved_plan_binding_sha256(approved_plan)
            available_frames = {record.frame for record in approved_plan.frames}
            unknown_frames = sorted(set(args.frames) - available_frames)
            if unknown_frames:
                raise ValueError(
                    "full-negative frames are absent from the approved plan: " + ", ".join(str(frame) for frame in unknown_frames)
                )
            if not approved_plan.device_id.startswith("coolscan3:"):
                raise RuntimeError("full-negative capture currently requires a direct coolscan3 device")
            identity = _full_negative_identity(
                args.output,
                resume=args.resume,
                identity=approved_plan.identity,
                approved_plan_binding=approved_binding,
            )
            with _stop_between_transfers(stop):
                completed = service.scan_full_negative(
                    plan_path=path,
                    output_dir=args.output,
                    identity=identity,
                    fine_params=_fine_params(4000),
                    stop=stop,
                    frames=args.frames,
                    progress=_ProgressPrinter(),
                )
            print(
                json.dumps(
                    {
                        "plan_path": str(path),
                        "output": str(args.output.expanduser().resolve()),
                        "identity": identity.to_dict(),
                        "frames": {str(frame): str(result.manifest_path) for frame, result in completed.items()},
                    },
                    indent=2,
                )
            )
            return 0

        if args.command == "forward-roll":
            if not args.yes:
                raise RuntimeError(
                    "forward-roll policy confirmation refused: pass --yes to accept automatic verification and the named exceptions"
                )
            _require_live(args)
            unknown_exclusions = sorted(set(args.exclude_calibration) - set(args.frames))
            if unknown_exclusions:
                raise ValueError(
                    "calibration exclusions are absent from this roll: " + ", ".join(str(frame) for frame in unknown_exclusions)
                )
            service = make_service() if service_factory is not None else _build_service(args.exclude_calibration)
            device_id = args.device or _discover_coolscan_device()
            if not device_id.startswith("coolscan3:"):
                raise RuntimeError("forward-roll capture currently requires a direct coolscan3 device")
            identity = _forward_roll_identity(args.output, resume=args.resume)
            operations = (
                forward_operations_factory(service, device_id)
                if forward_operations_factory is not None
                else _RunnerForwardOperations(
                    service,
                    device_id=device_id,
                    registration_signature=_registration_signature(args.exclude_calibration),
                )
            )
            workflow = ForwardRollWorkflow(
                root=args.output,
                device_id=device_id,
                identity=identity,
                frames=args.frames,
                chunk_size=args.chunk_size,
                operations=operations,
                resume=args.resume,
                calibration_excluded_frames=args.exclude_calibration,
            )
            stop = threading.Event()
            with _stop_between_transfers(stop):
                checkpoint = workflow.run(
                    approval_confirmed=True,
                    allow_unverified_frames=args.allow_unverified,
                    stop=stop,
                    progress=_ProgressPrinter(),
                    max_chunks=args.max_chunks,
                )
            ejected = _eject_after_forward_roll(
                operations,
                complete=checkpoint.complete,
                requested=not args.no_eject,
            )
            print(
                json.dumps(
                    {
                        "checkpoint": str(checkpoint.path),
                        "identity": checkpoint.identity.to_dict(),
                        "frames": list(checkpoint.frames),
                        "completed_frames": list(checkpoint.completed_frames),
                        "invalid_frames": list(checkpoint.invalid_frames),
                        "complete": checkpoint.complete,
                        "ejected": ejected,
                    },
                    indent=2,
                )
            )
            return 0
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2

    parser.error(f"unknown command {args.command!r}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
