"""Tests for the fauxce-hybrid subprocess bridge, with a fake CLI runner.

No real ``fauxce-hybrid`` install is required: ``run_hybrid_repair`` never
imports it, it only shells out to the console script named on
``HybridRuntimeConfig.executable``. Every test here supplies its own fake
``runner`` callable in place of ``subprocess.run`` and writes the output
files the real CLI would have written, so these tests exercise the argv
construction and output-reading contract without a pinned IOPaint runtime.
"""

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from negpy.services.repair.fauxice_hybrid_runner import (
    HybridRunError,
    HybridRuntimeConfig,
    run_hybrid_repair,
)


def _runtime_config(tmp_path: Path) -> HybridRuntimeConfig:
    return HybridRuntimeConfig(
        iopaint_python=tmp_path / "iopaint-venv" / "bin" / "python",
        iopaint_executable=tmp_path / "iopaint-venv" / "bin" / "iopaint",
        iopaint_source_manifest_sha256=hashlib.sha256(b"iopaint-source").hexdigest(),
        model_dir=tmp_path / "lama-models",
        model_weights=tmp_path / "lama-models" / "torch" / "hub" / "checkpoints" / "big-lama.pt",
        model_weights_sha256=hashlib.sha256(b"big-lama-weights").hexdigest(),
    )


def _rgbi(height: int, width: int, base: int) -> np.ndarray:
    return np.full((height, width, 4), base, dtype=np.uint16)


def _receipt(*, requested: str = "cpu", used: str = "cpu", fraction: float = 0.01) -> dict:
    return {
        "core": {
            "version": "0.3.0",
            "backend": {"requested": requested, "used": used, "reason": "stub receipt"},
        },
        "synthesis": {"fraction": fraction},
        "routing": {
            "counts": {
                "final_regions": 3,
                "synthesis_pixels": 120,
                "frame_pixels": 10000,
                "at_floor_pixels": 500,
            }
        },
    }


def _write_success_outputs(out_dir: Path, *, hybrid_rgb16: np.ndarray, receipt: dict) -> None:
    out_dir.mkdir(parents=True)
    np.save(out_dir / "output-hybrid.rgb16.npy", hybrid_rgb16, allow_pickle=False)
    (out_dir / "synth-mask.png").write_bytes(b"fake-png-bytes")
    (out_dir / "hybrid-receipt.json").write_text(json.dumps(receipt), encoding="utf-8")


class _RecordingRunner:
    """Fake ``subprocess.run`` that records argv and stages the CLI's outputs."""

    def __init__(self, *, hybrid_rgb16: np.ndarray, receipt: dict, returncode: int = 0) -> None:
        self._hybrid_rgb16 = hybrid_rgb16
        self._receipt = receipt
        self._returncode = returncode
        self.calls: list[list[str]] = []

    def __call__(self, argv, **kwargs) -> subprocess.CompletedProcess:
        self.calls.append(list(argv))
        out_index = argv.index("--out")
        out_dir = Path(argv[out_index + 1])
        if self._returncode == 0:
            _write_success_outputs(out_dir, hybrid_rgb16=self._hybrid_rgb16, receipt=self._receipt)
            return subprocess.CompletedProcess(argv, 0, stdout="ok", stderr="")
        return subprocess.CompletedProcess(argv, self._returncode, stdout="", stderr="synthetic failure")


class TestRunHybridRepairSuccess:
    def test_reads_hybrid_output_and_mask(self, tmp_path: Path) -> None:
        hybrid_rgb16 = np.full((4, 4, 3), 4242, dtype=np.uint16)
        runner = _RecordingRunner(hybrid_rgb16=hybrid_rgb16, receipt=_receipt())

        result = run_hybrid_repair(
            _rgbi(4, 4, 1000),
            _rgbi(4, 4, 500),
            same_frame_id="frame-001",
            backend="cpu",
            runtime=_runtime_config(tmp_path),
            scratch_dir=tmp_path,
            runner=runner,
        )

        np.testing.assert_array_equal(result.hybrid_rgb16, hybrid_rgb16)
        assert result.synth_mask_png == b"fake-png-bytes"
        assert result.synth_mask_sha256 == hashlib.sha256(b"fake-png-bytes").hexdigest()

    def test_receipt_fields_are_extracted(self, tmp_path: Path) -> None:
        hybrid_rgb16 = np.zeros((2, 2, 3), dtype=np.uint16)
        receipt = _receipt(requested="auto", used="cpu-fast", fraction=0.0136)
        runner = _RecordingRunner(hybrid_rgb16=hybrid_rgb16, receipt=receipt)

        result = run_hybrid_repair(
            _rgbi(2, 2, 1000),
            _rgbi(2, 2, 500),
            same_frame_id="frame-002",
            backend="auto",
            runtime=_runtime_config(tmp_path),
            scratch_dir=tmp_path,
            runner=runner,
        )

        assert result.engine_version == "0.3.0"
        assert result.backend_requested == "auto"
        assert result.backend_used == "cpu-fast"
        assert result.synthesis_fraction == 0.0136
        assert result.routing_counts == {
            "final_regions": 3,
            "synthesis_pixels": 120,
            "frame_pixels": 10000,
            "at_floor_pixels": 500,
        }

    def test_routing_counts_none_when_receipt_omits_expected_keys(self, tmp_path: Path) -> None:
        hybrid_rgb16 = np.zeros((2, 2, 3), dtype=np.uint16)
        receipt = _receipt()
        del receipt["routing"]["counts"]["frame_pixels"]
        runner = _RecordingRunner(hybrid_rgb16=hybrid_rgb16, receipt=receipt)

        result = run_hybrid_repair(
            _rgbi(2, 2, 1000),
            _rgbi(2, 2, 500),
            same_frame_id="frame-003",
            backend="cpu",
            runtime=_runtime_config(tmp_path),
            scratch_dir=tmp_path,
            runner=runner,
        )

        assert result.routing_counts is None
        # A partial/malformed routing block must not silently degrade the
        # fields this module is confident about.
        assert result.synthesis_fraction == 0.01

    def test_argv_carries_the_documented_cli_flags(self, tmp_path: Path) -> None:
        hybrid_rgb16 = np.zeros((2, 2, 3), dtype=np.uint16)
        runner = _RecordingRunner(hybrid_rgb16=hybrid_rgb16, receipt=_receipt())
        runtime = _runtime_config(tmp_path)

        run_hybrid_repair(
            _rgbi(2, 2, 1000),
            _rgbi(2, 2, 500),
            same_frame_id="frame-004",
            backend="cpu",
            runtime=runtime,
            scratch_dir=tmp_path,
            runner=runner,
        )

        assert len(runner.calls) == 1
        argv = runner.calls[0]
        assert argv[0] == "fauxce-hybrid"
        for flag, value in (
            ("--same-frame-id", "frame-004"),
            ("--backend", "cpu"),
            ("--iopaint-python", str(runtime.iopaint_python)),
            ("--iopaint-executable", str(runtime.iopaint_executable)),
            ("--iopaint-source-manifest-sha256", runtime.iopaint_source_manifest_sha256),
            ("--model-dir", str(runtime.model_dir)),
            ("--model-weights", str(runtime.model_weights)),
            ("--model-weights-sha256", runtime.model_weights_sha256),
            ("--inpaint-device", "cpu"),
            ("--inpaint-threads", "1"),
            ("--inpaint-seed", "0"),
        ):
            assert flag in argv, f"missing {flag}"
            assert argv[argv.index(flag) + 1] == value
        assert "--assert-focus-exposure-locked" in argv
        # A real hybrid run must not pass --no-inpaint, or the CLI would
        # never touch the model and every hybrid call would silently
        # degrade to routing-only.
        assert "--no-inpaint" not in argv

    def test_scratch_dir_out_subdirectory_does_not_preexist(self, tmp_path: Path) -> None:
        """The CLI refuses a pre-existing --out directory; this module must create it fresh."""
        hybrid_rgb16 = np.zeros((2, 2, 3), dtype=np.uint16)

        def runner(argv, **kwargs) -> subprocess.CompletedProcess:
            out_dir = Path(argv[argv.index("--out") + 1])
            assert not out_dir.exists()
            _write_success_outputs(out_dir, hybrid_rgb16=hybrid_rgb16, receipt=_receipt())
            return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

        scratch = tmp_path / "scratch"
        scratch.mkdir()
        run_hybrid_repair(
            _rgbi(2, 2, 1000),
            _rgbi(2, 2, 500),
            same_frame_id="frame-005",
            backend="cpu",
            runtime=_runtime_config(tmp_path),
            scratch_dir=scratch,
            runner=runner,
        )


class TestRunHybridRepairFailure:
    def test_nonzero_exit_raises_hybrid_run_error(self, tmp_path: Path) -> None:
        runner = _RecordingRunner(
            hybrid_rgb16=np.zeros((2, 2, 3), dtype=np.uint16),
            receipt=_receipt(),
            returncode=2,
        )

        with pytest.raises(HybridRunError, match="synthetic failure"):
            run_hybrid_repair(
                _rgbi(2, 2, 1000),
                _rgbi(2, 2, 500),
                same_frame_id="frame-006",
                backend="cpu",
                runtime=_runtime_config(tmp_path),
                scratch_dir=tmp_path,
                runner=runner,
            )

    def test_missing_output_file_raises_hybrid_run_error(self, tmp_path: Path) -> None:
        def runner(argv, **kwargs) -> subprocess.CompletedProcess:
            out_dir = Path(argv[argv.index("--out") + 1])
            out_dir.mkdir(parents=True)
            # Deliberately omit synth-mask.png and hybrid-receipt.json.
            np.save(out_dir / "output-hybrid.rgb16.npy", np.zeros((2, 2, 3), dtype=np.uint16))
            return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

        with pytest.raises(HybridRunError, match="missing"):
            run_hybrid_repair(
                _rgbi(2, 2, 1000),
                _rgbi(2, 2, 500),
                same_frame_id="frame-007",
                backend="cpu",
                runtime=_runtime_config(tmp_path),
                scratch_dir=tmp_path,
                runner=runner,
            )

    def test_launch_oserror_raises_hybrid_run_error(self, tmp_path: Path) -> None:
        def runner(argv, **kwargs):
            raise OSError("fauxce-hybrid executable not found")

        with pytest.raises(HybridRunError, match="could not launch"):
            run_hybrid_repair(
                _rgbi(2, 2, 1000),
                _rgbi(2, 2, 500),
                same_frame_id="frame-008",
                backend="cpu",
                runtime=_runtime_config(tmp_path),
                scratch_dir=tmp_path,
                runner=runner,
            )
