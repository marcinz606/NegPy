"""Package-resource and provenance checks for the LS-5000 single-pass core."""

import hashlib
import inspect
import os
import shutil
import subprocess
import sys
import textwrap
import tomllib
import zipfile
from pathlib import Path

import pytest

from negpy.infrastructure.scanners.ls5000_single_pass import (
    bundle as capture_bundle,
    meter,
    packed,
    roll_index,
)
from negpy.infrastructure.scanners.ls5000_single_pass.bundle import (
    CAPTURE_BUNDLE_SHA256,
    CAPTURE_WORKER_SHA256,
    CaptureBundleIntegrityError,
    verify_capture_bundle,
)
from negpy.infrastructure.scanners.ls5000_single_pass.capture_process import (
    CAPTURE_HELPER_FLAG,
)
from negpy.infrastructure.scanners.ls5000_single_pass.plan import (
    CANONICAL_FINE_READ_BYTES,
    CANONICAL_FINE_READ_CDB,
    CANONICAL_FINE_READ_COUNT,
    CANONICAL_PLAN_LINE_COUNT,
    CANONICAL_PLAN_SHA256,
    CanonicalPlanError,
    canonical_plan_bytes,
    load_canonical_plan,
    verify_canonical_plan,
)


REPO = Path(__file__).resolve().parents[2]


def test_bundled_plan_has_exact_proven_hash_and_fine_read_contract() -> None:
    payload = canonical_plan_bytes()
    plan = load_canonical_plan()

    assert hashlib.sha256(payload).hexdigest() == CANONICAL_PLAN_SHA256
    assert len(plan) == CANONICAL_PLAN_LINE_COUNT
    assert plan[0]["name"] == "INQUIRY"
    assert plan[0]["expected_data_in"].endswith("312e3033")  # LS-5000 ED firmware 1.03
    assert plan[-1]["role"] == "fine-rgbi4-template"
    assert plan[-1]["cdb"] == CANONICAL_FINE_READ_CDB
    assert plan[-1]["request_len"] == CANONICAL_FINE_READ_BYTES
    assert plan[-1]["repeat"] == CANONICAL_FINE_READ_COUNT
    assert plan[-1]["capture"] is True


def test_plan_integrity_check_refuses_even_one_changed_byte() -> None:
    payload = bytearray(canonical_plan_bytes())
    payload[100] ^= 1
    with pytest.raises(CanonicalPlanError, match="sha256 mismatch"):
        verify_canonical_plan(payload)


def test_core_modules_have_no_usb_or_campaign_tree_imports() -> None:
    for module in (meter, packed, roll_index):
        source = inspect.getsource(module)
        assert "single-pass-wire" not in source
        assert "negfit/data/wire" not in source
        assert "import usb" not in source
        assert "import pyusb" not in source


def test_packaged_capture_bundle_sources_match_their_stable_identity() -> None:
    assert verify_capture_bundle(require_python_sources=True) == CAPTURE_BUNDLE_SHA256
    assert len(CAPTURE_WORKER_SHA256) == 64


def test_capture_bundle_pins_the_continuation_parser_and_wire_resource(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(
        capture_bundle.CAPTURE_BUNDLE_COMPONENT_SHA256,
        "continuation_plan.py",
        "0" * 64,
    )
    with pytest.raises(CaptureBundleIntegrityError, match="continuation_plan.py"):
        verify_capture_bundle(require_python_sources=True)

    monkeypatch.undo()
    canonical = capture_bundle.canonical_continuation_plan_bytes()
    monkeypatch.setattr(
        capture_bundle,
        "canonical_continuation_plan_bytes",
        lambda: canonical + b"changed",
    )
    with pytest.raises(CaptureBundleIntegrityError, match="continuation plan"):
        verify_capture_bundle(require_python_sources=False)


def test_real_packaged_worker_dry_run_does_not_touch_the_scanner() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "-m",
            "negpy.infrastructure.scanners.ls5000_single_pass.worker",
            "--frame",
            "18",
            "--boundary-offset-rows",
            "0",
            "--confirm-full-capture",
        ],
        cwd=REPO,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "619458560 bytes" in completed.stdout
    assert "dry run only; scanner was not accessed" in completed.stdout


def test_desktop_helper_dispatch_runs_worker_without_starting_the_ui(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from negpy.desktop.main import _dispatch_capture_helper

    handled = _dispatch_capture_helper(
        [
            "NegPy",
            CAPTURE_HELPER_FLAG,
            "--frame",
            "18",
            "--confirm-full-capture",
        ]
    )

    assert handled is True
    assert "dry run only; scanner was not accessed" in capsys.readouterr().out
    assert _dispatch_capture_helper(["NegPy"]) is False


def test_all_negpy_packages_and_plan_resource_are_declared_for_distribution() -> None:
    pyproject_text = (REPO / "pyproject.toml").read_text(encoding="utf-8")
    pyproject = tomllib.loads(pyproject_text)
    build_script = (REPO / "build.py").read_text(encoding="utf-8")
    discovery = pyproject["tool"]["setuptools"]["packages"]["find"]
    data_package = "negpy.infrastructure.scanners.ls5000_single_pass.data"
    pyinstaller_resource = "negpy/infrastructure/scanners/ls5000_single_pass/data"
    pyinstaller_worker = (
        "--hidden-import=negpy.infrastructure.scanners.ls5000_single_pass.worker"
    )
    assert discovery == {"include": ["negpy*"], "namespaces": False}
    assert pyproject["tool"]["setuptools"]["package-data"][data_package] == [
        "*.json",
        "*.jsonl",
    ]
    assert pyinstaller_resource in build_script
    assert pyinstaller_worker in build_script
    assert "_dispatch_capture_helper(sys.argv)" in (
        REPO / "negpy/desktop/main.py"
    ).read_text(encoding="utf-8")


def test_built_wheel_contains_and_imports_representative_application_packages(tmp_path: Path) -> None:
    source = tmp_path / "source"
    wheel_dir = tmp_path / "wheel"
    source.mkdir()
    wheel_dir.mkdir()
    shutil.copytree(
        REPO / "negpy",
        source / "negpy",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    for filename in ("pyproject.toml", "README.md", "VERSION", "LICENSE"):
        shutil.copy2(REPO / filename, source / filename)

    build = subprocess.run(
        [
            sys.executable,
            "-c",
            "from setuptools.build_meta import build_wheel; import sys; build_wheel(sys.argv[1])",
            str(wheel_dir),
        ],
        cwd=source,
        check=True,
        capture_output=True,
        text=True,
    )
    wheels = list(wheel_dir.glob("*.whl"))
    assert len(wheels) == 1, build.stdout + build.stderr
    wheel = wheels[0]
    expected_members = {
        "negpy/domain/interfaces.py",
        "negpy/features/geometry/processor.py",
        "negpy/services/scanning/writer.py",
        "negpy/services/scanning/ls5000_roll_outputs.py",
        "negpy/services/scanning/ls5000_roll_session.py",
        "negpy/services/scanning/ls5000_sane_rgb.py",
        "negpy/services/scanning/ls5000_single_pass_workflow.py",
        "negpy/services/scanning/roll_preview_controls.py",
        "negpy/desktop/controller.py",
        "negpy/desktop/workers/ls5000_roll_worker.py",
        "negpy/desktop/view/widgets/roll_slot_model.py",
        "negpy/desktop/view/widgets/roll_slot_selector.py",
        "negpy/infrastructure/scanners/ls5000_single_pass/packed.py",
        "negpy/infrastructure/scanners/ls5000_single_pass/meter.py",
        "negpy/infrastructure/scanners/ls5000_single_pass/roll_index.py",
        "negpy/infrastructure/scanners/ls5000_single_pass/plan.py",
        "negpy/infrastructure/scanners/ls5000_single_pass/bundle.py",
        "negpy/infrastructure/scanners/ls5000_single_pass/capture_process.py",
        "negpy/infrastructure/scanners/ls5000_single_pass/continuation_plan.py",
        "negpy/infrastructure/scanners/ls5000_single_pass/window.py",
        "negpy/infrastructure/scanners/ls5000_single_pass/worker.py",
        "negpy/infrastructure/scanners/ls5000_single_pass/data/replay-first-rgbi4-plan.jsonl",
        "negpy/infrastructure/scanners/ls5000_single_pass/data/replay-first-rgbi4-manifest.json",
        "negpy/infrastructure/scanners/ls5000_single_pass/data/replay-next-rgbi4-plan.json",
    }
    installed = tmp_path / "installed-wheel"
    with zipfile.ZipFile(wheel) as archive:
        assert expected_members <= set(archive.namelist())
        archive.extractall(installed)

    script = textwrap.dedent(
        """
        import hashlib
        import importlib
        import sys

        modules = [
            "negpy.domain.interfaces",
            "negpy.features.geometry.processor",
            "negpy.services.scanning.writer",
            "negpy.desktop.controller",
            "negpy.infrastructure.scanners.ls5000_single_pass",
        ]
        for name in modules:
            module = importlib.import_module(name)
            assert module.__file__.startswith(sys.argv[1]), (name, module.__file__)
        from negpy.infrastructure.scanners.ls5000_single_pass.plan import (
            CANONICAL_PLAN_SHA256,
            canonical_plan_bytes,
        )
        assert hashlib.sha256(canonical_plan_bytes()).hexdigest() == CANONICAL_PLAN_SHA256
        from negpy.infrastructure.scanners.ls5000_single_pass.continuation_plan import (
            CANONICAL_CONTINUATION_PLAN_SHA256,
            canonical_continuation_plan_bytes,
        )
        assert hashlib.sha256(canonical_continuation_plan_bytes()).hexdigest() == CANONICAL_CONTINUATION_PLAN_SHA256
        from negpy.infrastructure.scanners.ls5000_single_pass.bundle import (
            CAPTURE_BUNDLE_SHA256,
            verify_capture_bundle,
        )
        assert verify_capture_bundle(require_python_sources=True) == CAPTURE_BUNDLE_SHA256
        """
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(installed)
    imported = subprocess.run(
        [sys.executable, "-c", script, str(installed)],
        cwd=tmp_path,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert imported.returncode == 0, imported.stdout + imported.stderr

    dry_run = subprocess.run(
        [
            sys.executable,
            "-m",
            "negpy.infrastructure.scanners.ls5000_single_pass.worker",
            "--frame",
            "18",
            "--confirm-full-capture",
        ],
        cwd=tmp_path,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert dry_run.returncode == 0, dry_run.stdout + dry_run.stderr
    assert "dry run only; scanner was not accessed" in dry_run.stdout
