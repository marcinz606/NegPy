"""
Session hooks: write `NEGPY_METRICS_OUT` JSON, order regression test last, optional options.
"""

from __future__ import annotations

import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from . import fixture as _fixture_mod
from . import recorder
from .labeling import metrics_machine_label

_SCHEMA_VERSION = 2


def pytest_sessionstart(session: pytest.Session) -> None:
    recorder.clear()


def pytest_collection_modifyitems(session: pytest.Session, config: pytest.Config, items: list) -> None:
    """Run regression test after all other metrics tests in this package."""
    metrics = [i for i in items if i.nodeid.startswith("tests/metrics/")]
    outside = [i for i in items if not i.nodeid.startswith("tests/metrics/")]
    reg = [i for i in metrics if "regression" in i.nodeid]
    non_reg = [i for i in metrics if "regression" not in i.nodeid]
    items[:] = outside + non_reg + reg


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    out = _metrics_output_path(session.config)
    if not out:
        return

    mlabel = metrics_machine_label()
    payload: dict[str, Any] = {
        "schema_version": _SCHEMA_VERSION,
        "machine_label": mlabel,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "exitstatus": exitstatus,
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "implementation": platform.python_implementation(),
            "machine_label": mlabel,
        },
        "ci": {
            "github_ref": os.environ.get("GITHUB_REF", ""),
            "github_sha": os.environ.get("GITHUB_SHA", ""),
            "github_workflow": os.environ.get("GITHUB_WORKFLOW", ""),
        },
        "input": {f"negpy_perf_raw_{f.key}": os.environ.get(f"NEGPY_PERF_RAW_{f.key.upper()}", "") for f in _fixture_mod.FIXTURES},
        "metrics": recorder.snapshot(),
    }
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\n[metrics] machine_label={mlabel!r} wrote {path} — metric keys: {list(payload['metrics'].keys())}")


def _metrics_output_path(config: pytest.Config) -> str | None:
    opt = config.getoption("--metrics-out", default=None)
    if opt:
        return str(opt).strip() or None
    ev = os.environ.get("NEGPY_METRICS_OUT", "").strip()
    return ev or None
