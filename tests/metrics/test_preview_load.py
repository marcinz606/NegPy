"""
Wall-clock metrics for PreviewManager (real RAW files on disk).

Fixtures are sourced from rawsamples.ch and cached in ~/.cache/negpy-metrics/.

Environment:
  NEGPY_PERF_RAW_CR2             — override CR2 fixture path
  NEGPY_PERF_RAW_NEF             — override NEF fixture path
  NEGPY_PERF_RAW_ARW             — override ARW fixture path
  NEGPY_PERF_RAW_MAX_SEC         — cold-load budget in seconds (default 30)
  NEGPY_METRICS_OUT              — write session metrics to this JSON path
  NEGPY_METRICS_BASELINE         — path to prior-run JSON; enables regression test
  NEGPY_METRICS_MAX_REGRESSION_PCT — max allowed regression % vs baseline (default 40.0)
  NEGPY_METRICS_MACHINE          — override auto machine label
  NEGPY_METRICS_BASELINE_UNLABELED=1 — accept legacy baselines with no machine_label

On CI fixtures are pre-downloaded by the workflow cache step so tests never skip.
Locally, get_fixture_path() downloads on first run (~20-30 MB each) and caches in
~/.cache/negpy-metrics/.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from negpy.services.rendering.preview_manager import PreviewManager

from . import fixture, recorder
from .labeling import metrics_machine_label, resolve_baseline_metrics

pytestmark = [pytest.mark.slow, pytest.mark.metrics]


def _max_seconds() -> float:
    return float(os.environ.get("NEGPY_PERF_RAW_MAX_SEC", "30").strip())


def _regression_pct() -> float:
    return float(os.environ.get("NEGPY_METRICS_MAX_REGRESSION_PCT", "40.0").strip())


@pytest.mark.parametrize("fix", fixture.FIXTURES, ids=lambda f: f.key)
def test_preview_load_cold_within_budget(fix: fixture.RawFixture) -> None:
    path = fixture.get_fixture_path(fix)
    if path is None:
        pytest.skip(f"No perf fixture for {fix.key!r} (network down and NEGPY_PERF_RAW_{fix.key.upper()} not set)")

    max_sec = _max_seconds()
    t0 = time.perf_counter()
    buf, dims, meta = PreviewManager().load_linear_preview(path)
    elapsed = time.perf_counter() - t0

    h, w = dims
    mp = (h * w) / 1_000_000
    recorder.record(f"preview.load.cold_s.{fix.key}", elapsed)
    recorder.record(f"preview.load.cold_mpx_per_s.{fix.key}", mp / elapsed)

    assert buf is not None and buf.ndim == 3
    assert len(dims) == 2
    assert isinstance(meta, dict)

    suggest = max(elapsed * 1.05, max_sec + 0.5)
    assert elapsed < max_sec, (
        f"Preview load took {elapsed:.2f}s (budget {max_sec}s) for {path!r} ({mp:.1f} MPx).\n"
        f"Raise NEGPY_PERF_RAW_MAX_SEC (e.g. {suggest:.0f}) to match this machine, or optimize decode."
    )


def test_regression_against_baseline_if_configured() -> None:
    base = os.environ.get("NEGPY_METRICS_BASELINE", "").strip()
    if not base:
        pytest.skip("Set NEGPY_METRICS_BASELINE to a JSON file from a prior run to enforce regression bounds")

    p = Path(base)
    if not p.is_file():
        pytest.skip(f"NEGPY_METRICS_BASELINE is not a file: {base!r}")

    cur = recorder.snapshot()
    if not cur:
        pytest.skip("No metrics recorded this session (did the timing tests run?)")

    label = metrics_machine_label()
    prev_data = json.loads(p.read_text(encoding="utf-8"))
    prev, err = resolve_baseline_metrics(prev_data, label)
    if prev is None:
        pytest.skip(f"Regression: {err}")

    pct = _regression_pct()
    failures: list[str] = []
    for k, new_v in cur.items():
        if k not in prev:
            continue
        old_v = float(prev[k])
        if old_v <= 0:
            continue
        if new_v > old_v * (1.0 + pct / 100.0):
            failures.append(f"{k}: {new_v:.3f}s vs baseline {old_v:.3f}s (>{pct}% worse) [machine {label!r}]")

    assert not failures, "Metrics regression vs baseline:\n" + "\n".join(failures)
