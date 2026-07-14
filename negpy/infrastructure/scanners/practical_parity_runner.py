#!/usr/bin/env python3
"""Observable, fail-closed runner for the LS-5000
RGB+IR practical-parity capture through NegPy's production scan path.

Replaces the quarantined parity-test.sh/preflight.sh. This runner:

  * DEFAULTS TO DRY-RUN. Without --live it only opens the device, snapshots
    option state, evaluates every film/frame precondition, writes a JSON
    report, and closes. It never calls sane_start and never moves film.
  * Discovers the current coolscan3 device ID through SANE instead of
    hard-coding the ephemeral USB address.
  * Refuses to proceed (dry or live) while any precondition fails — an
    inactive or zero-max `frame` constraint (no film / exhausted feeder)
    is a refusal, never a "try anyway".
  * In live mode, drives negpy.SaneBackend.scan() — the exact production
    split-capture path — through a recording wrapper that logs every option
    write, start, read, cancel and close with timestamps, then derives the
    acceptance assertions from that log: one device open, exactly two starts,
    no geometry write after the first start, samples N->1, infrared
    false->true, AF/AE true->false, alignment transform + confidence.
  * Writes the RGB + IR TIFF pair transactionally via
    negpy.services.scanning.writer.write_tiff_16bit, reloads both with
    tifffile to verify dtype/shape/DPI, and records SHA-256 hashes.
  * Re-opens the device afterwards to prove the scanner is still responsive.

Hardware safety: never interrupt a running transfer; never issue
infrared=yes with samples-per-scan > 1 (the NegPy path guarantees this);
do not run --live until film is confirmed loaded and a fresh dry-run passes.

Run from the NegPy project so `negpy` and python-sane resolve. The canonical
entry point is the package module:

    cd negpy-src
    uv run python -m negpy.infrastructure.scanners.practical_parity_runner
    uv run python -m negpy.infrastructure.scanners.practical_parity_runner --profile small --frame 7
    uv run python -m negpy.infrastructure.scanners.practical_parity_runner --profile full
    uv run python -m negpy.infrastructure.scanners.practical_parity_runner --live

Exit codes: 0 = ok (dry-run passed, or live capture passed every assertion),
2 = preconditions failed (nothing started), 1 = live run failed an assertion.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
import threading
import time
from dataclasses import asdict
from datetime import datetime

from negpy.services.scanning.quality import (
    StoppedTransportSmearAssessment as StoppedTransportSmearAssessment,
    assess_stopped_transport_smear,
    file_sha256,
    inspect_tiff_payload,
    split_alignment_metrics_confident,
)

sha256 = file_sha256

DEFAULT_OUT_DIR = os.path.abspath("practical-parity-results")

PROFILES = {
    # Small discriminator first; full Frame 03 only after small passes.
    "small": {"dpi": 1000, "frame": 3, "subframe_mm": 6.35, "br_y": 800, "samples": 4},
    "full": {"dpi": 4000, "frame": 3, "subframe_mm": 6.35, "br_y": 5003, "samples": 4},
}

WATCHED_OPTIONS = (
    "frame",
    "frame_count",
    "subframe",
    "br_y",
    "autofocus",
    "ae",
    "samples_per_scan",
    "infrared",
    "depth",
    "resolution",
)

GEOMETRY_OPTIONS = {"frame", "subframe", "br_y"}


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def _profile_output_stem(profile_name: str, profile: dict) -> str:
    return f"{profile_name}-{profile['dpi']}dpi-frame{profile['frame']:02d}"


def strip_net_prefix(device_id: str) -> str:
    if not device_id.startswith("net:"):
        return device_id
    rest = device_id[len("net:") :]
    if rest.startswith("["):
        close = rest.find("]:")
        return rest[close + 2 :] if close > 0 else device_id
    host, sep, backend_part = rest.partition(":")
    return backend_part if sep and host else device_id


def discover_coolscan_device(sane_module) -> str:
    """Find exactly one coolscan3 device; the USB address is ephemeral so it
    must be discovered fresh every run, never hard-coded."""
    devices = [d[0] for d in sane_module.get_devices()]
    matches = [d for d in devices if strip_net_prefix(d).startswith("coolscan3:")]
    if not matches:
        raise RuntimeError(f"no coolscan3 device found (saw: {devices or 'none'})")
    if len(matches) > 1:
        raise RuntimeError(f"multiple coolscan3 devices found, pass --device explicitly: {matches}")
    return matches[0]


def snapshot_options(dev) -> dict:
    """Read-only view of the option state this runner gates on.

    python-sane encodes a (min, max, step) RANGE as a tuple and an enumerated
    value list as a list; JSON flattens both to arrays, so the kind is
    recorded explicitly — misreading an enumeration as a range inverts
    membership checks (e.g. resolution [4000, 2000, 1000])."""
    opt = dev.opt if hasattr(dev, "opt") else {}
    snapshot: dict[str, dict] = {}
    for name in WATCHED_OPTIONS:
        if name not in opt:
            snapshot[name] = {"present": False}
            continue
        option = opt[name]
        is_active = getattr(option, "is_active", None)
        try:
            active = bool(is_active()) if callable(is_active) else True
        except Exception:
            active = False
        is_settable = getattr(option, "is_settable", None)
        try:
            settable = bool(is_settable()) if callable(is_settable) else None
        except Exception:
            settable = False
        constraint = getattr(option, "constraint", None)
        if isinstance(constraint, tuple):
            kind = "range"
        elif isinstance(constraint, list):
            kind = "values"
        else:
            kind = "none"
        snapshot[name] = {
            "present": True,
            "active": active,
            "settable": settable,
            "constraint": list(constraint) if isinstance(constraint, (list, tuple)) else constraint,
            "constraint_kind": kind,
        }
    return snapshot


def _constraint_max(state: dict) -> int | None:
    constraint = state.get("constraint")
    if state.get("constraint_kind") == "range" and isinstance(constraint, (list, tuple)) and len(constraint) >= 2:
        return int(constraint[1])
    if state.get("constraint_kind") == "values" and isinstance(constraint, (list, tuple)) and constraint:
        numeric = [int(v) for v in constraint if isinstance(v, (int, float))]
        return max(numeric) if numeric else None
    return None


def _constraint_allows(state: dict, value: int) -> bool:
    constraint = state.get("constraint")
    kind = state.get("constraint_kind")
    if kind == "none" or constraint is None:
        return True  # unconstrained option
    if kind == "range" and len(constraint) >= 2:
        lower, upper = constraint[:2]
        if not lower <= value <= upper:
            return False
        quantum = constraint[2] if len(constraint) >= 3 else 0
        if not quantum:
            return True
        steps = (value - lower) / quantum
        return math.isclose(steps, round(steps), abs_tol=1e-9)
    if kind == "values":
        return any(
            isinstance(candidate, (int, float)) and math.isclose(float(candidate), float(value), rel_tol=0.0, abs_tol=1e-9)
            for candidate in constraint
        )
    return False


def evaluate_preconditions(snapshot: dict, profile: dict) -> list[str]:
    """Every reason NOT to scan. Empty list means all preconditions hold."""
    failures: list[str] = []

    def _require_active(name: str) -> dict | None:
        state = snapshot.get(name, {"present": False})
        if not state.get("present"):
            failures.append(f"option '{name}' is missing")
            return None
        if not state.get("active"):
            failures.append(f"option '{name}' is inactive")
            return None
        if state.get("settable") is False:
            failures.append(f"option '{name}' is not settable")
            return None
        return state

    frame_state = _require_active("frame")
    if frame_state is not None:
        frame_max = _constraint_max(frame_state)
        if frame_max is None or frame_max < max(3, profile["frame"]):
            failures.append(
                f"frame constraint {frame_state.get('constraint')} has no selectable frame >= "
                f"{max(3, profile['frame'])} (film not loaded, or feeder at end state)"
            )
        elif not _constraint_allows(frame_state, profile["frame"]):
            failures.append(f"frame constraint {frame_state.get('constraint')} cannot select frame {profile['frame']}")

    # This capture targets a roll frame, so the exposure counter must be live.
    frame_count_state = _require_active("frame_count")
    if frame_count_state is not None:
        count_max = _constraint_max(frame_count_state)
        if count_max is not None and count_max < 1:
            failures.append(f"frame_count constraint {frame_count_state.get('constraint')} allows no exposures")
        elif not _constraint_allows(frame_count_state, 1):
            failures.append(f"frame_count constraint {frame_count_state.get('constraint')} cannot reset to 1")

    subframe_state = _require_active("subframe")
    if subframe_state is not None and not _constraint_allows(subframe_state, profile["subframe_mm"]):
        failures.append(f"subframe constraint {subframe_state.get('constraint')} cannot honor {profile['subframe_mm']}")

    br_y_state = _require_active("br_y")
    if br_y_state is not None and not _constraint_allows(br_y_state, profile["br_y"]):
        failures.append(f"br_y constraint {br_y_state.get('constraint')} cannot honor {profile['br_y']}")

    for name in ("autofocus", "ae", "infrared"):
        _require_active(name)

    samples_state = _require_active("samples_per_scan")
    if samples_state is not None and not _constraint_allows(samples_state, profile["samples"]):
        failures.append(f"samples_per_scan constraint {samples_state.get('constraint')} cannot honor {profile['samples']}x")

    resolution_state = _require_active("resolution")
    if resolution_state is not None and not _constraint_allows(resolution_state, profile["dpi"]):
        failures.append(f"resolution constraint {resolution_state.get('constraint')} lacks {profile['dpi']} dpi")

    depth_state = _require_active("depth")
    if depth_state is not None and not _constraint_allows(depth_state, 16):
        failures.append(f"depth constraint {depth_state.get('constraint')} lacks 16-bit")

    return failures


def derive_assertions(events: list[dict], open_count: int, result, profile: dict) -> dict:
    """Turn the recorded event log + ScanResult into the acceptance booleans."""
    starts = [i for i, e in enumerate(events) if e["event"] == "start"]
    completed_reads = [i for i, e in enumerate(events) if e["event"] == "read_end"]
    sets = [(i, e) for i, e in enumerate(events) if e["event"] == "set"]

    def _geometry_after_first_start() -> bool:
        if not starts:
            return False
        return any(i > starts[0] for i, e in sets if e["option"] in GEOMETRY_OPTIONS)

    def _written_once_before_first_start(option: str, expected_value) -> bool:
        writes = [(i, event) for i, event in sets if event["option"] == option]
        return bool(starts) and len(writes) == 1 and writes[0][0] < starts[0] and writes[0][1]["value"] == expected_value

    def _written_once_between_captures(option: str, expected_value) -> bool:
        writes = [(i, event) for i, event in sets if event["option"] == option]
        return (
            len(starts) == 2
            and bool(completed_reads)
            and len(writes) == 1
            and completed_reads[0] < writes[0][0] < starts[1]
            and writes[0][1]["value"] == expected_value
        )

    def _two_capture_writes(option: str, first_value, second_value) -> bool:
        writes = [(i, event) for i, event in sets if event["option"] == option]
        return (
            len(starts) == 2
            and bool(completed_reads)
            and len(writes) == 2
            and writes[0][0] < starts[0]
            and writes[0][1]["value"] == first_value
            and completed_reads[0] < writes[1][0] < starts[1]
            and writes[1][1]["value"] == second_value
        )

    alignment = getattr(result, "ir_alignment", None) if result is not None else None
    alignment_finite = alignment is not None and all(math.isfinite(value) for value in (alignment.dx_px, alignment.dy_px))
    image_width = result.rgb.shape[1] if result is not None else None
    alignment_confident = split_alignment_metrics_confident(alignment, image_width=image_width)
    checks = {
        "one_device_open": open_count == 1,
        "exactly_two_starts": len(starts) == 2,
        "exactly_two_completed_reads_in_order": (
            len(starts) == 2 and len(completed_reads) == 2 and starts[0] < completed_reads[0] < starts[1] < completed_reads[1]
        ),
        "frame_written_once_before_first_start": _written_once_before_first_start("frame", profile["frame"]),
        "subframe_written_once_before_first_start": _written_once_before_first_start("subframe", profile["subframe_mm"]),
        "br_y_written_once_before_first_start": _written_once_before_first_start("br_y", profile["br_y"]),
        "depth_16_written_once_before_first_start": _written_once_before_first_start("depth", 16),
        "resolution_written_once_before_first_start": _written_once_before_first_start("resolution", profile["dpi"]),
        "frame_count_reset_once_between_captures": _written_once_between_captures("frame_count", 1),
        "no_geometry_write_after_first_start": not _geometry_after_first_start(),
        "samples_n_then_1": _two_capture_writes("samples_per_scan", profile["samples"], 1),
        "infrared_false_then_true": _two_capture_writes("infrared", False, True),
        "autofocus_true_then_false": _two_capture_writes("autofocus", True, False),
        "ae_true_then_false": _two_capture_writes("ae", True, False),
        "rgb_is_uint16": result is not None and str(result.rgb.dtype) == "uint16",
        "ir_is_uint16": result is not None and result.ir is not None and str(result.ir.dtype) == "uint16",
        "rgb_ir_dimensions_match": (result is not None and result.ir is not None and result.rgb.shape[:2] == result.ir.shape),
        "alignment_reported": alignment is not None,
        "alignment_finite": alignment_finite,
        "alignment_confident": alignment_confident,
    }
    checks["all_passed"] = all(checks.values())
    return checks


# --------------------------------------------------------------------------
# Recording instrumentation around the real python-sane module
# --------------------------------------------------------------------------


class RecordingDev:
    """Forwarding wrapper that logs every interaction with the SANE device."""

    def __init__(self, real, events: list[dict]) -> None:
        object.__setattr__(self, "_real", real)
        object.__setattr__(self, "_events", events)

    def _log(self, event: str, **fields) -> None:
        self._events.append({"ts": _now(), "event": event, **fields})

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_real"), name)

    def __setattr__(self, name, value) -> None:
        try:
            setattr(object.__getattribute__(self, "_real"), name, value)
        except Exception as exc:
            self._log("set_failed", option=name, value=value, error=f"{type(exc).__name__}: {exc}")
            raise
        self._log("set", option=name, value=value)

    def start(self) -> None:
        self._log("start")
        self._real.start()

    def get_parameters(self):
        params = self._real.get_parameters()
        self._log("get_parameters", parameters=str(params))
        return params

    def arr_snap(self, progress=None):
        self._log("read_begin")
        began = time.monotonic()
        array = self._real.arr_snap(progress=progress)
        self._log("read_end", seconds=round(time.monotonic() - began, 3), shape=list(array.shape), dtype=str(array.dtype))
        return array

    def cancel(self) -> None:
        self._log("cancel")
        self._real.cancel()

    def close(self) -> None:
        self._log("close")
        self._real.close()


class RecordingSaneModule:
    """Stands in for the `sane` module inside SaneBackend; counts opens."""

    def __init__(self, real_module, events: list[dict]) -> None:
        self._real = real_module
        self._events = events
        self.opened: list[str] = []

    def open(self, device_id: str) -> RecordingDev:
        self.opened.append(device_id)
        self._events.append({"ts": _now(), "event": "open", "device": device_id})
        return RecordingDev(self._real.open(device_id), self._events)


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------


def probe_preconditions(sane_module, device_id: str, profile: dict) -> tuple[dict, list[str]]:
    dev = sane_module.open(device_id)
    try:
        snapshot = snapshot_options(dev)
    finally:
        dev.close()
    return snapshot, evaluate_preconditions(snapshot, profile)


def probe_responsiveness(sane_module, device_id: str, profile: dict) -> tuple[bool, dict | None, str | None]:
    """Re-open, query, and close the device; every stage must succeed."""
    try:
        snapshot, _ = probe_preconditions(sane_module, device_id, profile)
    except Exception as exc:
        return False, None, f"{type(exc).__name__}: {exc}"
    if not all(snapshot.get(name, {}).get("present") for name in ("depth", "resolution", "infrared")):
        return False, snapshot, "post-scan probe did not expose depth, resolution, and infrared options"
    return True, snapshot, None


def _resolution_matches_dpi(resolution: list[int] | None, expected_dpi: int) -> bool:
    if resolution is None or len(resolution) != 2 or expected_dpi <= 0:
        return False
    numerator, denominator = resolution
    return numerator > 0 and denominator > 0 and numerator == expected_dpi * denominator


def inspect_artifacts(result, rgb_path: str, ir_path: str, *, expected_dpi: int) -> tuple[dict, dict[str, bool]]:
    """Reload the committed TIFF pair and compare it with the capture in memory."""
    import tifffile

    artifacts = {}
    rgb_smear_assessment = None
    for label, path in (("rgb", rgb_path), ("ir", ir_path)):
        inspection = inspect_tiff_payload(path)
        array = None
        if inspection.payload_within_file:
            with tifffile.TiffFile(path) as tif:
                array = tif.pages.first.asarray()
        artifacts[label] = {
            "path": path,
            "sha256": inspection.sha256,
            "bytes": inspection.byte_length,
            "dtype": str(array.dtype) if array is not None else inspection.dtype,
            "shape": list(array.shape) if array is not None else list(inspection.shape),
            "page_count": inspection.page_count,
            "x_resolution": (list(inspection.x_resolution) if inspection.x_resolution is not None else None),
            "y_resolution": (list(inspection.y_resolution) if inspection.y_resolution is not None else None),
            "resolution_unit": inspection.resolution_unit,
            "payload_within_file": inspection.payload_within_file,
        }
        if label == "rgb" and array is not None:
            rgb_smear_assessment = assess_stopped_transport_smear(array, dpi=expected_dpi)
            artifacts[label]["stopped_transport_smear"] = asdict(rgb_smear_assessment)

    dpi_ok = all(
        _resolution_matches_dpi(artifact["x_resolution"], expected_dpi) and _resolution_matches_dpi(artifact["y_resolution"], expected_dpi)
        for artifact in artifacts.values()
    )
    expected_shapes = {
        "rgb": list(result.rgb.shape),
        "ir": list(result.ir.shape) if result.ir is not None else None,
    }
    checks = {
        "tiff_pair_reloads_16bit_at_capture_dpi": dpi_ok
        and all(
            artifact["dtype"] == "uint16"
            and artifact["page_count"] == 1
            and artifact["payload_within_file"]
            and artifact["resolution_unit"] == "INCH"
            for artifact in artifacts.values()
        ),
        "tiff_shapes_match_in_memory": all(
            expected_shapes[label] is not None and artifact["shape"] == expected_shapes[label] for label, artifact in artifacts.items()
        ),
        "no_stopped_transport_smear_suffix": (rgb_smear_assessment is not None and rgb_smear_assessment.verdict == "clean"),
    }
    return artifacts, checks


def finalize_live_capture(
    *,
    report: dict,
    checks: dict[str, bool],
    result,
    scan_error: str | None,
    sane_module,
    device_id: str,
    args,
    profile: dict,
    write_tiff_fn=None,
) -> tuple[dict, int]:
    """Persist and validate artifacts, then prove the scanner still responds."""
    checks["artifact_pipeline_succeeded"] = False
    checks["no_stopped_transport_smear_suffix"] = False
    if result is not None and scan_error is None:
        try:
            if write_tiff_fn is None:
                from negpy.services.scanning.writer import write_tiff_16bit

                write_tiff_fn = write_tiff_16bit
            os.makedirs(args.out_dir, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            stem = f"parity-{_profile_output_stem(args.profile, profile)}-{stamp}"
            rgb_path = write_tiff_fn(result, os.path.join(args.out_dir, f"{stem}.tif"))
            ir_path = os.path.splitext(rgb_path)[0] + "_IR.tif"
            artifacts, artifact_checks = inspect_artifacts(result, rgb_path, ir_path, expected_dpi=profile["dpi"])
            report["artifacts"] = artifacts
            checks.update(artifact_checks)
            checks["artifact_pipeline_succeeded"] = all(artifact_checks.values())
        except Exception as exc:
            report["artifact_error"] = f"{type(exc).__name__}: {exc}"

    # This probe runs even when artifact persistence or validation failed.
    responsive, followup_snapshot, probe_error = probe_responsiveness(sane_module, device_id, profile)
    if responsive:
        report["post_scan_option_snapshot"] = followup_snapshot
        checks["scanner_responsive_after_scan"] = True
    else:
        report["post_scan_probe_error"] = probe_error
        checks["scanner_responsive_after_scan"] = False

    checks["all_passed"] = all(value for name, value in checks.items() if name != "all_passed")
    report["assertions"] = checks
    report["verdict"] = "live-ok" if checks["all_passed"] and scan_error is None else "live-failed"
    return report, 0 if report["verdict"] == "live-ok" else 1


def write_report(report: dict, args, *, fallback_dir: str | None = None) -> str:
    """Write the report, falling back to a timestamped temp path if needed."""
    report["finished"] = _now()
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    profile = report.get("profile")
    profile_suffix = f"-{_profile_output_stem(profile['name'], profile)}" if profile is not None else ""
    primary_path = args.json or os.path.join(args.out_dir, f"report-{report['mode']}{profile_suffix}-{stamp}.json")

    def _write(path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, default=str)

    report["report_path"] = primary_path
    try:
        _write(primary_path)
        return primary_path
    except Exception as exc:
        report["report_write_error"] = f"{type(exc).__name__}: {exc}"

    fallback_path = os.path.join(
        fallback_dir or tempfile.gettempdir(),
        f"negpy-practical-parity-report-{report['mode']}-{stamp}.json",
    )
    report["report_path"] = fallback_path
    _write(fallback_path)
    return fallback_path


def run(sane_module, args, profile: dict) -> tuple[dict, int]:
    """Full runner flow against `sane_module` (real or fake). Returns
    (report, exit_code). Dry-run and precondition refusal never start a scan."""
    report: dict = {
        "generator": "negpy.infrastructure.scanners.practical_parity_runner",
        "started": _now(),
        "mode": "live" if args.live else "dry-run",
        "profile": {"name": args.profile, **profile},
    }

    if args.device:
        device_id = args.device
    else:
        try:
            device_id = discover_coolscan_device(sane_module)
        except Exception as exc:
            report["option_snapshot"] = {}
            report["precondition_failures"] = [f"device discovery failed: {type(exc).__name__}: {exc}"]
            report["verdict"] = "preconditions-failed"
            return report, 2
    report["device_id"] = device_id
    if args.device and not strip_net_prefix(device_id).startswith("coolscan3:"):
        report["option_snapshot"] = {}
        report["precondition_failures"] = [f"explicit device is not a coolscan3 backend: {device_id}"]
        report["verdict"] = "preconditions-failed"
        return report, 2

    try:
        snapshot, failures = probe_preconditions(sane_module, device_id, profile)
    except Exception as exc:
        report["option_snapshot"] = {}
        report["precondition_failures"] = [f"preflight probe failed: {type(exc).__name__}: {exc}"]
        report["verdict"] = "preconditions-failed"
        return report, 2
    report["option_snapshot"] = snapshot
    report["precondition_failures"] = failures
    if failures:
        report["verdict"] = "preconditions-failed"
        return report, 2
    if not args.live:
        report["verdict"] = "dry-run-ok"
        return report, 0

    # ---- live capture through the production NegPy path -------------------
    from negpy.infrastructure.scanners.params import RegisteredScanGeometry, ScanParams
    from negpy.infrastructure.scanners.sane_backend import SaneBackend

    events: list[dict] = []
    recording_module = RecordingSaneModule(sane_module, events)
    backend = SaneBackend.__new__(SaneBackend)
    backend._sane = recording_module
    backend._sane_initialized = True
    backend._devices_cache = None

    params = ScanParams(
        dpi=profile["dpi"],
        depth=16,
        capture_ir=True,
        autofocus=True,
        auto_exposure=True,
        samples_per_scan=profile["samples"],
        registered_geometry=RegisteredScanGeometry(
            frame=profile["frame"],
            subframe_mm=profile["subframe_mm"],
            br_y_device_px=profile["br_y"],
        ),
    )
    report["scan_params"] = {
        "dpi": params.dpi,
        "depth": params.depth,
        "samples_per_scan": params.samples_per_scan,
        "frame": profile["frame"],
        "subframe_mm": profile["subframe_mm"],
        "br_y_device_px": profile["br_y"],
    }

    def progress(fraction: float) -> None:
        sys.stderr.write(f"\rscan progress: {fraction * 100.0:6.2f}%")
        sys.stderr.flush()

    result = None
    scan_error = None
    try:
        result = backend.scan(device_id, params, progress, threading.Event())
    except Exception as e:  # the report must capture the failure, not hide it
        scan_error = f"{type(e).__name__}: {e}"
    finally:
        sys.stderr.write("\n")
    report["events"] = events
    report["scan_error"] = scan_error

    checks = derive_assertions(events, len(recording_module.opened), result, profile)
    report["assertions"] = checks
    if result is not None and result.ir_alignment is not None:
        report["ir_alignment"] = asdict(result.ir_alignment)
    return finalize_live_capture(
        report=report,
        checks=checks,
        result=result,
        scan_error=scan_error,
        sane_module=sane_module,
        device_id=device_id,
        args=args,
        profile=profile,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--profile", choices=sorted(PROFILES), default="small")
    parser.add_argument("--frame", type=int, help="override the profile's film frame number")
    parser.add_argument("--registration-json", help="closed-loop registration manifest path")
    parser.add_argument("--live", action="store_true", help="actually scan; default is a read-only dry-run")
    parser.add_argument("--device", help="explicit SANE device id (default: discover the coolscan3 device)")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--json", help="report path (default: <out-dir>/report-<mode>-<timestamp>.json)")
    args = parser.parse_args(argv)
    profile = dict(PROFILES[args.profile])
    if args.live and args.frame is not None and args.frame != profile["frame"] and args.registration_json is None:
        parser.error("live scans for non-default --frame require --registration-json with coupled subframe_mm and br_y geometry")
    if args.frame is not None:
        profile["frame"] = args.frame
    if args.registration_json is not None:
        if args.frame is None:
            parser.error("--registration-json requires --frame")
        try:
            with open(args.registration_json) as stream:
                manifest = json.load(stream)
        except json.JSONDecodeError as error:
            parser.error(f"registration manifest {args.registration_json} is not valid JSON: {error}")
        except OSError as error:
            parser.error(f"cannot read registration manifest {args.registration_json}: {error}")
        if not isinstance(manifest, dict) or not isinstance(manifest.get("frames"), list):
            parser.error("registration manifest must contain a 'frames' array")
        frames = manifest["frames"]
        seen_frames: set[int] = set()
        for index, item in enumerate(frames, start=1):
            if not isinstance(item, dict) or type(item.get("frame")) is not int:
                parser.error(f"registration manifest frame entry {index} must contain an integer 'frame'")
            manifest_frame = item["frame"]
            if manifest_frame in seen_frames:
                parser.error(f"registration manifest has duplicate entries for frame {manifest_frame}")
            seen_frames.add(manifest_frame)
        matches = [item for item in frames if item["frame"] == args.frame]
        if not matches:
            parser.error(f"registration manifest has no entry for frame {args.frame}")
        if len(matches) > 1:
            parser.error(f"registration manifest has duplicate entries for frame {args.frame}")
        entry = matches[0]
        if "subframe_mm" not in entry or "br_y" not in entry:
            parser.error(f"registration entry for frame {args.frame} must contain subframe_mm and br_y")
        subframe_mm = entry["subframe_mm"]
        if type(subframe_mm) not in (int, float) or not math.isfinite(subframe_mm) or subframe_mm <= 0:
            parser.error(f"registration entry for frame {args.frame} has invalid subframe_mm; expected a finite number > 0")
        br_y = entry["br_y"]
        if type(br_y) is not int or not 0 <= br_y <= 5958:
            parser.error(f"registration entry for frame {args.frame} has invalid br_y; expected an integer in [0, 5958]")
        profile["subframe_mm"] = subframe_mm
        profile["br_y"] = br_y

    import sane

    sane.init()

    report, code = run(sane, args, profile)

    report_path = write_report(report, args)

    print(f"device:  {report.get('device_id', '(not discovered)')}")
    for failure in report.get("precondition_failures", []):
        print(f"REFUSED: {failure}")
    if "assertions" in report:
        for name, ok in report["assertions"].items():
            print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print(f"verdict: {report['verdict']}")
    print(f"report:  {report_path}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
