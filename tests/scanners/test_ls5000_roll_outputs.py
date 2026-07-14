from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import pytest

from negpy.services.scanning.ls5000_roll_outputs import (
    RollOutputError,
    promote_single_pass_frame,
)


@dataclass(frozen=True)
class _Finalization:
    manifest_path: Path
    output_paths: dict[str, Path]
    manifest: dict[str, object]


def _finalization(tmp_path: Path) -> _Finalization:
    source = tmp_path / "attempt" / "final"
    source.mkdir(parents=True)
    outputs = {
        "rgb": source / "frame007.tif",
        "ir": source / "frame007_IR.tif",
        "ir_valid_mask": source / "frame007_IR_VALID.tif",
    }
    for role, path in outputs.items():
        path.write_bytes((role + "\n").encode() * 11)
    manifest_path = source / "manifest.json"
    manifest = {
        "identity": {"selected_slot": 7, "boundary_offset_rows": -3},
        "frame_evidence": {
            "roll_identity": {
                "reviewed_fingerprint_sha256": "c" * 64,
                "fresh_fingerprint_sha256": "d" * 64,
                "comparison": {"matches": True, "reason": "matched"},
                "selected_slot_comparison": {
                    "matches": True,
                    "reason": "matched",
                    "slot": 7,
                },
            },
            "manual_review_approval": {
                "binding_sha256": "e" * 64,
                "slot": 7,
            },
        },
        "quality_control": {
            "capture_clipping": {"warning": False},
            "focus_detail": {
                "verdict": "measured",
                "score": 0.03125,
            },
        },
    }
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    return _Finalization(manifest_path, outputs, manifest)


def test_promotes_triplet_without_overwrite_and_binds_receipt(tmp_path: Path) -> None:
    finalization = _finalization(tmp_path)
    output = tmp_path / "scans"

    first = promote_single_pass_frame(
        finalization,
        output_folder=output,
        filename_pattern='roll_slot{{ "%02d" % slot }}_{{ "%03d" % seq }}',
        selected_slot=7,
    )
    second = promote_single_pass_frame(
        finalization,
        output_folder=output,
        filename_pattern='roll_slot{{ "%02d" % slot }}_{{ "%03d" % seq }}',
        selected_slot=7,
    )

    assert first.sequence == 1
    assert first.rgb_path.name == "roll_slot07_001.tif"
    assert first.ir_path.name == "roll_slot07_001_IR.tif"
    assert first.ir_valid_mask_path.name == "roll_slot07_001_IR_VALID.tif"
    assert second.sequence == 2
    assert first.rgb_path.read_bytes() == finalization.output_paths["rgb"].read_bytes()
    receipt = json.loads(first.receipt_path.read_text(encoding="utf-8"))
    assert receipt["kind"] == "negpy.ls5000-promoted-frame"
    assert receipt["roll_slot"] == 7
    assert receipt["capture_summary"] == {
        "boundary_offset_rows": -3,
        "manual_review_approval": finalization.manifest["frame_evidence"][
            "manual_review_approval"
        ],
        "quality_control": finalization.manifest["quality_control"],
        "roll_identity": finalization.manifest["frame_evidence"]["roll_identity"],
    }
    assert receipt["outputs"]["rgb"]["path"] == first.rgb_path.name


def test_existing_sidecar_reserves_the_whole_basename(tmp_path: Path) -> None:
    finalization = _finalization(tmp_path)
    output = tmp_path / "scans"
    output.mkdir()
    (output / "roll_001_IR.tif").write_bytes(b"unrelated")

    result = promote_single_pass_frame(
        finalization,
        output_folder=output,
        filename_pattern='roll_{{ "%03d" % seq }}',
    )

    assert result.sequence == 2
    assert (output / "roll_001_IR.tif").read_bytes() == b"unrelated"


def test_promoter_refuses_a_slot_label_that_disagrees_with_capture_evidence(
    tmp_path: Path,
) -> None:
    with pytest.raises(RollOutputError, match="slot differs"):
        promote_single_pass_frame(
            _finalization(tmp_path),
            output_folder=tmp_path / "scans",
            filename_pattern='roll_slot{{ "%02d" % slot }}_{{ seq }}',
            selected_slot=8,
        )


def test_promoter_refuses_a_slot_label_when_capture_evidence_omits_the_slot(
    tmp_path: Path,
) -> None:
    finalization = _finalization(tmp_path)
    finalization.manifest["identity"].pop("selected_slot")
    finalization.manifest_path.write_text(
        json.dumps(finalization.manifest) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(RollOutputError, match="has no selected slot"):
        promote_single_pass_frame(
            finalization,
            output_folder=tmp_path / "scans",
            filename_pattern='roll_slot{{ "%02d" % slot }}_{{ seq }}',
            selected_slot=7,
        )


def test_constant_filename_pattern_is_rejected_before_collision_search(
    tmp_path: Path,
) -> None:
    finalization = _finalization(tmp_path)

    with pytest.raises(ValueError, match="must produce a different basename"):
        promote_single_pass_frame(
            finalization,
            output_folder=tmp_path / "scans",
            filename_pattern="one-name-for-every-frame",
        )


def test_repeating_filename_pattern_cannot_cycle_forever(tmp_path: Path) -> None:
    finalization = _finalization(tmp_path)
    output = tmp_path / "scans"
    output.mkdir()
    (output / "roll_1.tif").write_bytes(b"occupied")
    (output / "roll_0.tif").write_bytes(b"occupied")

    with pytest.raises(RollOutputError, match="repeated a basename"):
        promote_single_pass_frame(
            finalization,
            output_folder=output,
            filename_pattern="roll_{{ seq % 2 }}",
        )


def test_partial_publication_is_removed_on_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    finalization = _finalization(tmp_path)
    output = tmp_path / "scans"
    original_link = os.link
    calls = 0

    def fail_second(source, destination):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("simulated link and copy failure")
        return original_link(source, destination)

    monkeypatch.setattr(os, "link", fail_second)
    monkeypatch.setattr("shutil.copyfileobj", lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("copy failed")))

    with pytest.raises(RollOutputError, match="could not publish"):
        promote_single_pass_frame(
            finalization,
            output_folder=output,
            filename_pattern="roll_{{ seq }}",
        )

    assert list(output.iterdir()) == []


def test_missing_verified_artifact_refuses_before_writing(tmp_path: Path) -> None:
    finalization = _finalization(tmp_path)
    finalization.output_paths["ir"].unlink()

    with pytest.raises(RollOutputError, match="missing"):
        promote_single_pass_frame(
            finalization,
            output_folder=tmp_path / "scans",
            filename_pattern="roll_{{ seq }}",
        )
