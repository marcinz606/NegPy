"""Adversarial contracts for the file-backed Stage-3 replay bridge."""

from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path

import numpy as np
import pytest

from negpy.services.roll import exact_color
from tests.roll._exact_fixtures import write_stage3_replay_fixture


def _arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    identity = np.arange(65_536, dtype=np.uint16)
    return identity, 65_535 - identity, identity // 3


def _rewrite_report(path: Path, mutate) -> None:
    report = json.loads(path.read_bytes())
    mutate(report)
    path.write_bytes(json.dumps(report, sort_keys=True, separators=(",", ":")).encode())


def test_trusted_loader_builds_a_bound_replay_receipt(tmp_path: Path) -> None:
    report_path = write_stage3_replay_fixture(tmp_path / "capture", _arrays())

    receipt = exact_color.load_stage3_replay_builder_receipt(report_path)

    payload = exact_color.builder_receipt_payload(receipt)
    assert receipt.attested is True
    assert payload["scope"] == exact_color.STAGE3_REPLAY_SCOPE
    assert payload["native_per_acquisition_builder"] is False
    assert receipt.stage3_receipt == report_path.read_bytes()


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda report: report.update(errors=[{"code": "hidden"}]), "errors"),
        (lambda report: report["provenance"].pop("observer_source"), "provenance"),
        (lambda report: report["provenance"]["observer_executable"].pop("sha256"), "observer executable"),
        (lambda report: report["provenance"]["module"].update(sha256="0" * 64), "module provenance"),
        (lambda report: report["provenance"]["resource"].update(bytes=1_023), "resource provenance"),
        (lambda report: report["artifacts"].pop(), "exact required file set"),
        (lambda report: report["artifacts"].append(dict(report["artifacts"][0])), "repeats"),
    ],
)
def test_report_provenance_and_inventory_mutations_fail_closed(
    tmp_path: Path,
    mutate,
    message: str,
) -> None:
    report_path = write_stage3_replay_fixture(tmp_path / "capture", _arrays())
    _rewrite_report(report_path, mutate)

    with pytest.raises(exact_color.ExactColorIntegrityError, match=message):
        exact_color.load_stage3_replay_builder_receipt(report_path)


def test_pre_f_file_must_match_the_report_inventory(tmp_path: Path) -> None:
    report_path = write_stage3_replay_fixture(tmp_path / "capture", _arrays())
    path = report_path.parent / "builder-preF-g.bin"
    payload = bytearray(path.read_bytes())
    payload[4_096] ^= 1
    path.write_bytes(payload)

    with pytest.raises(exact_color.ExactColorIntegrityError, match="does not bind"):
        exact_color.load_stage3_replay_builder_receipt(report_path)


def test_symlinked_evidence_is_rejected(tmp_path: Path) -> None:
    report_path = write_stage3_replay_fixture(tmp_path / "capture", _arrays())
    target = report_path.parent / "builder-preF-b.bin"
    real = report_path.parent / "real-b.bin"
    target.replace(real)
    target.symlink_to(real)

    with pytest.raises(exact_color.ExactColorUnavailable, match="non-symlink"):
        exact_color.load_stage3_replay_builder_receipt(report_path)


def test_report_swap_between_lstat_and_open_is_detected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report_path = write_stage3_replay_fixture(tmp_path / "capture", _arrays())
    replacement = tmp_path / "replacement.json"
    replacement.write_bytes(report_path.read_bytes() + b" ")
    real_open = os.open
    swapped = False

    def swap_then_open(path, flags, *args, **kwargs):
        nonlocal swapped
        if not swapped and Path(path) == report_path:
            swapped = True
            os.replace(replacement, report_path)
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(exact_color.os, "open", swap_then_open)

    with pytest.raises(exact_color.ExactColorIntegrityError, match="changed while being read"):
        exact_color.load_stage3_replay_builder_receipt(report_path)


def test_callers_cannot_self_attest_a_builder_receipt(tmp_path: Path) -> None:
    report_path = write_stage3_replay_fixture(tmp_path / "capture", _arrays())
    receipt = exact_color.load_stage3_replay_builder_receipt(report_path)
    forged = dataclasses.replace(receipt, _factory_token=object())

    with pytest.raises(exact_color.ExactColorIntegrityError, match="trusted Stage-3 replay file loader"):
        exact_color.builder_receipt_payload(forged)
