"""Regression coverage for the installable NegPy wheel boundary."""

from __future__ import annotations

import hashlib
import os
import site as site_module
import subprocess
import sys
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ROLL_MODULES = {
    f"negpy/services/roll/{name}"
    for name in (
        "__init__.py",
        "exact_color.py",
        "nikon_icc.py",
        "portable_builder.py",
        "portable_cms.py",
        "portable_oracle_evaluator.py",
        "positive.py",
        "service.py",
    )
}


def _run(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, (
        f"command failed ({completed.returncode}): {' '.join(command)}\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )
    return completed


def test_wheel_contains_and_loads_all_negpy_packages_and_portable_assets(tmp_path: Path) -> None:
    dist = tmp_path / "dist"
    _run(["uv", "build", "--wheel", "--out-dir", str(dist)], cwd=ROOT)
    wheels = list(dist.glob("*.whl"))
    assert len(wheels) == 1
    wheel = wheels[0]

    expected_python = {path.relative_to(ROOT).as_posix() for path in (ROOT / "negpy").rglob("*.py")}
    expected_assets = {
        path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for directory in (
            ROOT / "negpy/assets/portable_builder",
            ROOT / "negpy/assets/portable_cms",
        )
        for path in directory.iterdir()
        if path.suffix in {".bin", ".json"}
    }
    assert len(expected_assets) == 11
    assert sum(name.endswith(".bin") for name in expected_assets) == 10
    assert sum(name.endswith(".json") for name in expected_assets) == 1

    with zipfile.ZipFile(wheel) as archive:
        packaged = set(archive.namelist())
        packaged_python = {name for name in packaged if name.startswith("negpy/") and name.endswith(".py")}
        packaged_assets = {
            name
            for name in packaged
            if name.startswith("negpy/assets/portable_")
            and Path(name).suffix in {".bin", ".json"}
        }
        assert packaged_python == expected_python
        assert ROLL_MODULES <= packaged
        assert packaged_assets == set(expected_assets)
        assert {name: hashlib.sha256(archive.read(name)).hexdigest() for name in expected_assets} == expected_assets

    venv = tmp_path / "venv"
    _run(["uv", "venv", "--python", sys.executable, str(venv)], cwd=tmp_path)
    venv_python = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    _run(
        ["uv", "pip", "install", "--python", str(venv_python), "--no-deps", str(wheel)],
        cwd=tmp_path,
    )
    venv_site = Path(
        _run(
            [str(venv_python), "-c", "import site; print(site.getsitepackages()[0])"],
            cwd=tmp_path,
        ).stdout.strip()
    )
    dependency_site = next(Path(path) for path in site_module.getsitepackages() if (Path(path) / "numpy").is_dir())
    (venv_site / "negpy-test-dependencies.pth").write_text(
        str(dependency_site) + "\n",
        encoding="utf-8",
    )
    smoke = """
import importlib
from pathlib import Path
import negpy.services.roll.exact_color as exact_color
import negpy.services.roll.nikon_icc as nikon_icc
import negpy.services.roll.portable_builder as portable_builder
import negpy.services.roll.portable_cms as portable_cms
import negpy.services.roll.service as service

site = Path(__import__('sys').argv[1]).resolve()
roll_modules = [
    importlib.import_module(name)
    for name in (
        'negpy.services.roll',
        'negpy.services.roll.exact_color',
        'negpy.services.roll.nikon_icc',
        'negpy.services.roll.portable_builder',
        'negpy.services.roll.portable_cms',
        'negpy.services.roll.positive',
        'negpy.services.roll.service',
    )
]
assert len(roll_modules) == 7
for module in roll_modules:
    assert Path(module.__file__).resolve().is_relative_to(site), module.__file__
builder = portable_builder.PortableStage1Builder()
cms = portable_cms.PortableCMSOnEvaluator()
assert 'negpy.services.roll.portable_oracle_evaluator' not in __import__('sys').modules
assert len(list(builder.assets_dir.glob('*.bin'))) == 1
assert len(list(cms.assets_dir.glob('*.bin'))) == 9
validation = cms.assets_dir / portable_cms.VALIDATION_RECEIPT_FILENAME
assert validation.is_file()
assert validation.stat().st_size == portable_cms.VALIDATION_RECEIPT_BYTES
assert __import__('hashlib').sha256(validation.read_bytes()).hexdigest() == portable_cms.VALIDATION_RECEIPT_SHA256
profile = nikon_icc.nikon_adobe_rgb_profile()
assert len(profile) == 492
assert __import__('hashlib').sha256(profile).hexdigest() == nikon_icc.NIKON_ADOBE_RGB_PROFILE_SHA256
"""
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    _run([str(venv_python), "-c", smoke, str(venv_site)], cwd=tmp_path, env=env)

    validation_path = (
        venv_site
        / "negpy/assets/portable_cms/portable-oracle-receipt.json"
    )
    validation_payload = validation_path.read_bytes()
    for failure, expected in (
        ("missing", "portable CMS validation receipt is unavailable"),
        ("one-byte", "portable CMS validation receipt hash mismatch"),
    ):
        if failure == "missing":
            validation_path.unlink()
        else:
            tampered = bytearray(validation_payload)
            tampered[len(tampered) // 2] ^= 1
            validation_path.write_bytes(tampered)
        try:
            completed = subprocess.run(
                [
                    str(venv_python),
                    "-c",
                    "from negpy.services.roll.portable_cms import PortableCMSOnEvaluator; PortableCMSOnEvaluator()",
                ],
                cwd=tmp_path,
                env=env,
                check=False,
                capture_output=True,
                text=True,
            )
            assert completed.returncode != 0
            assert expected in completed.stderr
        finally:
            validation_path.write_bytes(validation_payload)
