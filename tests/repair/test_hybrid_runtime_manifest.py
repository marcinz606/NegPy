from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from negpy.services.repair import hybrid_runtime_manifest as runtime_manifest
from negpy.services.repair.hybrid_runtime_manifest import (
    HybridRuntimeManifestError,
    RUNTIME_MANIFEST_SCHEMA,
    canonical_runtime_manifest_bytes,
    load_default_hybrid_runtime_manifest,
    load_hybrid_runtime_manifest,
    runtime_manifest_pin_path,
)


def _document(root: Path) -> dict[str, object]:
    return {
        "core_source_manifest_sha256": "1" * 64,
        "executable": str(root / "hybrid" / "bin" / "fauxce-hybrid"),
        "hybrid_python": str(root / "hybrid" / "bin" / "python"),
        "hybrid_source_manifest_sha256": "2" * 64,
        "inpaint_device": "cpu",
        "inpaint_seed": 0,
        "inpaint_threads": 1,
        "iopaint_executable": str(root / "iopaint" / "bin" / "iopaint"),
        "iopaint_python": str(root / "iopaint" / "bin" / "python"),
        "iopaint_source_manifest_sha256": "3" * 64,
        "max_synthesis_fraction": 0.1,
        "model_dir": str(root / "models"),
        "model_weights": str(root / "models" / "torch" / "hub" / "checkpoints" / "big-lama.pt"),
        "model_weights_sha256": "4" * 64,
        "schema": RUNTIME_MANIFEST_SCHEMA,
    }


def _write_pair(path: Path, document: dict[str, object]) -> str:
    payload = canonical_runtime_manifest_bytes(document)
    path.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    runtime_manifest_pin_path(path).write_text(digest + "\n", encoding="ascii")
    return digest


def test_loads_canonical_hash_pinned_runtime(tmp_path: Path) -> None:
    path = tmp_path / "fauxce-hybrid-runtime.json"
    digest = _write_pair(path, _document(tmp_path))

    runtime = load_hybrid_runtime_manifest(path, expected_sha256=digest)

    assert runtime.hybrid_python == tmp_path / "hybrid" / "bin" / "python"
    assert runtime.core_source_manifest_sha256 == "1" * 64
    assert runtime.hybrid_source_manifest_sha256 == "2" * 64
    assert runtime.max_synthesis_fraction == 0.1


def test_default_loader_requires_manifest_and_independent_pin(tmp_path: Path) -> None:
    path = tmp_path / "fauxce-hybrid-runtime.json"
    assert load_default_hybrid_runtime_manifest(path) is None

    path.write_bytes(canonical_runtime_manifest_bytes(_document(tmp_path)))
    with pytest.raises(HybridRuntimeManifestError, match="must both exist"):
        load_default_hybrid_runtime_manifest(path)


def test_default_loader_rejects_tampering_after_pin(tmp_path: Path) -> None:
    path = tmp_path / "fauxce-hybrid-runtime.json"
    document = _document(tmp_path)
    _write_pair(path, document)
    document["inpaint_threads"] = 2
    path.write_bytes(canonical_runtime_manifest_bytes(document))

    with pytest.raises(HybridRuntimeManifestError, match="SHA-256 mismatch"):
        load_default_hybrid_runtime_manifest(path)


def test_loader_rejects_noncanonical_or_extended_manifest(tmp_path: Path) -> None:
    path = tmp_path / "fauxce-hybrid-runtime.json"
    document = _document(tmp_path)
    document["unexpected"] = True
    payload = (json.dumps(document, indent=2) + "\n").encode()
    path.write_bytes(payload)

    with pytest.raises(HybridRuntimeManifestError, match="not canonical"):
        load_hybrid_runtime_manifest(
            path,
            expected_sha256=hashlib.sha256(payload).hexdigest(),
        )


def test_loader_rejects_symlinked_manifest(tmp_path: Path) -> None:
    target = tmp_path / "target.json"
    payload = canonical_runtime_manifest_bytes(_document(tmp_path))
    target.write_bytes(payload)
    path = tmp_path / "fauxce-hybrid-runtime.json"
    path.symlink_to(target)

    with pytest.raises(HybridRuntimeManifestError, match="non-symlink"):
        load_hybrid_runtime_manifest(
            path,
            expected_sha256=hashlib.sha256(payload).hexdigest(),
        )


def test_loader_does_not_block_if_manifest_is_swapped_to_fifo(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "fauxce-hybrid-runtime.json"
    payload = canonical_runtime_manifest_bytes(_document(tmp_path))
    path.write_bytes(payload)
    real_open = runtime_manifest.os.open
    swapped = False

    def swap_before_open(candidate, flags, *args, **kwargs):
        nonlocal swapped
        if os.fspath(candidate) == os.fspath(path) and not swapped:
            swapped = True
            assert flags & os.O_NONBLOCK
            path.unlink()
            os.mkfifo(path)
        return real_open(candidate, flags, *args, **kwargs)

    monkeypatch.setattr(runtime_manifest.os, "open", swap_before_open)

    with pytest.raises(HybridRuntimeManifestError, match="changed while opening"):
        load_hybrid_runtime_manifest(
            path,
            expected_sha256=hashlib.sha256(payload).hexdigest(),
        )

    assert swapped is True
