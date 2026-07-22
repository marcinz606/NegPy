"""Tests for the fauxce-hybrid subprocess bridge, with a fake CLI runner.

No real ``fauxce-hybrid`` install is required: ``run_hybrid_repair`` never
imports it, it only shells out to the console script named on
``HybridRuntimeConfig.executable``. Every test here supplies its own fake
``runner`` callable in place of ``subprocess.run`` and writes the output
files the real CLI would have written, so these tests exercise the argv
construction and output-reading contract without a pinned IOPaint runtime.
"""

import hashlib
import io
import json
import os
import subprocess
import sys
import threading
import time
import types
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from negpy.services.repair import fauxice_hybrid_runner as hybrid_runner
from negpy.services.repair.fauxice_hybrid_runner import (
    HybridRunCancelled,
    HybridRunError,
    HybridRuntimeConfig,
    run_hybrid_repair,
)

EXPECTED_CORE_SOURCE_SHA256 = "c" * 64
EXPECTED_HYBRID_SOURCE_SHA256 = "d" * 64


def _runtime_config(tmp_path: Path) -> HybridRuntimeConfig:
    hybrid_python = tmp_path / "hybrid-venv" / "bin" / "python"
    executable = tmp_path / "hybrid-venv" / "bin" / "fauxce-hybrid"
    iopaint_python = tmp_path / "iopaint-venv" / "bin" / "python"
    iopaint_executable = tmp_path / "iopaint-venv" / "bin" / "iopaint"
    model_dir = tmp_path / "lama-models"
    model_weights = model_dir / "torch" / "hub" / "checkpoints" / "big-lama.pt"
    for path in (hybrid_python, executable, iopaint_python, iopaint_executable):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"#!/bin/sh\nexit 0\n")
        path.chmod(0o700)
    model_weights.parent.mkdir(parents=True, exist_ok=True)
    model_weights.write_bytes(b"big-lama-weights")
    return HybridRuntimeConfig(
        hybrid_python=hybrid_python,
        executable=executable,
        core_source_manifest_sha256=EXPECTED_CORE_SOURCE_SHA256,
        hybrid_source_manifest_sha256=EXPECTED_HYBRID_SOURCE_SHA256,
        iopaint_python=iopaint_python,
        iopaint_executable=iopaint_executable,
        iopaint_source_manifest_sha256=hashlib.sha256(b"iopaint-source").hexdigest(),
        model_dir=model_dir,
        model_weights=model_weights,
        model_weights_sha256=hashlib.sha256(b"big-lama-weights").hexdigest(),
    )


def _rgbi(height: int, width: int, base: int) -> np.ndarray:
    return np.full((height, width, 4), base, dtype=np.uint16)


def _receipt(*, requested: str = "cpu", used: str = "cpu", fraction: float = 0.01) -> dict:
    return {"fraction": fraction, "requested": requested, "used": used}


def _arg(argv: list[str], flag: str) -> str:
    return argv[argv.index(flag) + 1]


def _raw_sha256(array: np.ndarray) -> str:
    canonical = np.array(array, dtype="<u2", order="C", copy=True)
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def _write_success_outputs(
    out_dir: Path,
    *,
    hybrid_rgb16: np.ndarray,
    receipt_options: dict,
    argv: list[str],
) -> None:
    out_dir.mkdir(parents=True)
    np.save(out_dir / "output-hybrid.rgb16.npy", hybrid_rgb16, allow_pickle=False)
    main = np.load(_arg(argv, "--main"), allow_pickle=False)
    prepass = np.load(_arg(argv, "--prepass"), allow_pickle=False)
    manifest_bytes = Path(_arg(argv, "--acquisition-manifest")).read_bytes()
    requested_fraction = float(receipt_options["fraction"])
    synthesis_pixels = min(
        int(round(requested_fraction * main.shape[0] * main.shape[1])),
        main.shape[0] * main.shape[1],
    )
    mask = np.zeros(main.shape[:2], dtype=np.uint8)
    mask.reshape(-1)[:synthesis_pixels] = 255
    mask_path = out_dir / "synth-mask.png"
    Image.fromarray(mask, mode="L").save(mask_path, format="PNG")
    mask_bytes = mask_path.read_bytes()
    fraction = synthesis_pixels / mask.size
    output_hash = _raw_sha256(hybrid_rgb16)
    receipt = {
        "artifacts": [
            {
                "dtype": "<u2",
                "file_sha256": hashlib.sha256((out_dir / "output-hybrid.rgb16.npy").read_bytes()).hexdigest(),
                "id": "hybrid-output",
                "raw_encoding": "npy_array_c_order",
                "raw_sha256": output_hash,
                "relative_path": "output-hybrid.rgb16.npy",
                "role": "hybrid_output_rgb16",
                "shape": list(hybrid_rgb16.shape),
            },
            {
                "dtype": "|u1",
                "file_sha256": hashlib.sha256(mask_bytes).hexdigest(),
                "id": "synthesis-mask",
                "raw_encoding": "png_decoded_uint8_c_order",
                "raw_sha256": hashlib.sha256(mask.tobytes()).hexdigest(),
                "relative_path": "synth-mask.png",
                "role": "synthesis_mask_png",
                "shape": list(mask.shape),
            },
        ],
        "composite": {"hybrid_rgb16_raw_sha256": output_hash},
        "core": {
            "version": "0.3.0",
            "source_manifest_sha256": EXPECTED_CORE_SOURCE_SHA256,
            "backend": {
                "requested": receipt_options["requested"],
                "used": receipt_options["used"],
                "reason": "stub receipt",
            },
        },
        "generation": {
            "hybrid_source_manifest_sha256": EXPECTED_HYBRID_SOURCE_SHA256,
        },
        "inpainting": {
            "invoked": True,
            "model": {"weights_sha256": _arg(argv, "--model-weights-sha256")},
            "runtime": {
                "device": _arg(argv, "--inpaint-device"),
                "seed": int(_arg(argv, "--inpaint-seed")),
                "threads": int(_arg(argv, "--inpaint-threads")),
            },
            "tool": {"iopaint_source_manifest_sha256": _arg(argv, "--iopaint-source-manifest-sha256")},
        },
        "inputs": {
            "geometry": {
                "mask_shape": list(mask.shape),
                "output_shape": list(hybrid_rgb16.shape),
            },
            "main": {
                "canonical_encoding": "uint16_little_endian_c_order",
                "raw_sha256": _raw_sha256(main),
                "shape": list(main.shape),
            },
            "prepass": {
                "canonical_encoding": "uint16_little_endian_c_order",
                "raw_sha256": _raw_sha256(prepass),
                "shape": list(prepass.shape),
            },
            "provenance": {
                "basis": "caller_asserted",
                "source_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            },
        },
        "routing": {
            "counts": {
                "at_floor_pixels": synthesis_pixels,
                "final_regions": 1 if synthesis_pixels else 0,
                "frame_pixels": int(mask.size),
                "synthesis_pixels": synthesis_pixels,
            }
        },
        "schema": "fauxce-hybrid-receipt-v2",
        "synthesis": {
            "fraction": fraction,
            "frame_pixel_count": int(mask.size),
            "pixel_count": synthesis_pixels,
            "within_budget": True,
        },
    }
    (out_dir / "hybrid-receipt.json").write_bytes(json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n")


def _test_receipt_verifier(path: Path, _runtime: HybridRuntimeConfig):
    document = json.loads(path.read_text(encoding="utf-8"))
    out_dir = path.parent
    hybrid = np.load(out_dir / "output-hybrid.rgb16.npy", allow_pickle=False)
    with Image.open(out_dir / "synth-mask.png") as image:
        mask = np.ascontiguousarray(np.asarray(image.convert("L")) != 0)
    return types.SimpleNamespace(
        document=document,
        receipt_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        hybrid_output_rgb16=np.ascontiguousarray(hybrid),
        synthesis_mask=mask,
        model_weights_rehashed=True,
    )


def _attest_only(path: Path, _runtime: HybridRuntimeConfig):
    return types.SimpleNamespace(
        receipt_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        model_weights_rehashed=True,
    )


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
            _write_success_outputs(
                out_dir,
                hybrid_rgb16=self._hybrid_rgb16,
                receipt_options=self._receipt,
                argv=list(argv),
            )
            return subprocess.CompletedProcess(argv, 0, stdout="ok", stderr="")
        return subprocess.CompletedProcess(argv, self._returncode, stdout="", stderr="synthetic failure")


class TestRunHybridRepairSuccess:
    def test_reports_hybrid_and_verification_progress(self, tmp_path: Path) -> None:
        runner = _RecordingRunner(
            hybrid_rgb16=np.zeros((2, 2, 3), dtype=np.uint16),
            receipt=_receipt(),
        )
        seen: list[float] = []

        run_hybrid_repair(
            _rgbi(2, 2, 1000),
            _rgbi(2, 2, 500),
            same_frame_id="progress-bound",
            backend="cpu",
            runtime=_runtime_config(tmp_path),
            scratch_dir=tmp_path,
            runner=runner,
            receipt_verifier=_test_receipt_verifier,
            progress=seen.append,
        )

        assert seen == [0.0, 0.8, 0.9, 1.0]

    @pytest.mark.parametrize("boundary", [0.9, 1.0])
    def test_cancellation_at_late_progress_boundary_never_returns_a_result(
        self,
        tmp_path: Path,
        boundary: float,
    ) -> None:
        runner = _RecordingRunner(
            hybrid_rgb16=np.zeros((2, 2, 3), dtype=np.uint16),
            receipt=_receipt(),
        )
        cancel = threading.Event()

        def progress(value: float) -> None:
            if value == boundary:
                cancel.set()

        with pytest.raises(HybridRunCancelled, match="cancelled"):
            run_hybrid_repair(
                _rgbi(2, 2, 1000),
                _rgbi(2, 2, 500),
                same_frame_id=f"late-cancel-{boundary}",
                backend="cpu",
                runtime=_runtime_config(tmp_path),
                scratch_dir=tmp_path,
                runner=runner,
                receipt_verifier=_test_receipt_verifier,
                progress=progress,
                cancel=cancel,
            )

    def test_subprocess_environment_ignores_python_injection_variables(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("PYTHONHOME", "/attacker/python-home")
        monkeypatch.setenv("PYTHONPATH", "/attacker/imports")
        monkeypatch.setenv("DYLD_INSERT_LIBRARIES", "/attacker/inject.dylib")
        monkeypatch.setenv("LD_PRELOAD", "/attacker/inject.so")
        seen_environment: dict[str, str] = {}
        hybrid_rgb16 = np.zeros((2, 2, 3), dtype=np.uint16)

        def runner(argv, **kwargs):
            seen_environment.update(kwargs["env"])
            out_dir = Path(_arg(list(argv), "--out"))
            _write_success_outputs(
                out_dir,
                hybrid_rgb16=hybrid_rgb16,
                receipt_options=_receipt(),
                argv=list(argv),
            )
            return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

        run_hybrid_repair(
            _rgbi(2, 2, 1000),
            _rgbi(2, 2, 500),
            same_frame_id="sanitized-environment",
            backend="cpu",
            runtime=_runtime_config(tmp_path),
            scratch_dir=tmp_path,
            runner=runner,
            receipt_verifier=_test_receipt_verifier,
        )

        assert "PYTHONHOME" not in seen_environment
        assert "PYTHONPATH" not in seen_environment
        assert "DYLD_INSERT_LIBRARIES" not in seen_environment
        assert "LD_PRELOAD" not in seen_environment
        assert seen_environment["PYTHONHASHSEED"] == "0"
        assert seen_environment["PYTHONNOUSERSITE"] == "1"
        assert seen_environment["LC_ALL"] == "C"

    def test_resolves_scratch_to_link_free_real_path_before_cli(self, tmp_path: Path) -> None:
        real_scratch = tmp_path / "real-scratch"
        real_scratch.mkdir()
        linked_scratch = tmp_path / "linked-scratch"
        linked_scratch.symlink_to(real_scratch, target_is_directory=True)
        hybrid_rgb16 = np.zeros((2, 2, 3), dtype=np.uint16)

        def runner(argv, **kwargs):
            out_dir = Path(_arg(list(argv), "--out"))
            assert out_dir.parent.parent == real_scratch.resolve(strict=True)
            assert out_dir.parent.name.startswith("negpy-hybrid-run-")
            _write_success_outputs(
                out_dir,
                hybrid_rgb16=hybrid_rgb16,
                receipt_options=_receipt(),
                argv=list(argv),
            )
            return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

        run_hybrid_repair(
            _rgbi(2, 2, 1000),
            _rgbi(2, 2, 500),
            same_frame_id="link-free-scratch",
            backend="cpu",
            runtime=_runtime_config(tmp_path / "runtime"),
            scratch_dir=linked_scratch,
            runner=runner,
            receipt_verifier=_test_receipt_verifier,
        )

    def test_default_receipt_verification_runs_in_external_hybrid_python(self, tmp_path: Path) -> None:
        hybrid_rgb16 = np.full((3, 4, 3), 4242, dtype=np.uint16)
        runner = _RecordingRunner(hybrid_rgb16=hybrid_rgb16, receipt=_receipt())
        verification_calls: list[list[str]] = []

        def verification_runner(argv, **kwargs):
            verification_calls.append(list(argv))
            receipt_path = Path(argv[-3])
            attestation = {
                "model_weights_rehashed": True,
                "receipt_sha256": hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
                "schema": "negpy.external-fauxce-receipt-verification-v1",
            }
            return subprocess.CompletedProcess(
                argv,
                0,
                stdout=json.dumps(attestation, sort_keys=True, separators=(",", ":")),
                stderr="",
            )

        runtime = _runtime_config(tmp_path)
        result = run_hybrid_repair(
            _rgbi(3, 4, 1000),
            _rgbi(2, 2, 500),
            same_frame_id="frame-external-verifier",
            backend="cpu",
            runtime=runtime,
            scratch_dir=tmp_path,
            runner=runner,
            verification_runner=verification_runner,
        )

        assert result.receipt_sha256
        assert len(verification_calls) == 1
        assert verification_calls[0][0] == str(runtime.hybrid_python)
        assert "fauxce_hybrid.receipts" in verification_calls[0][3]

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
            receipt_verifier=_test_receipt_verifier,
        )

        np.testing.assert_array_equal(result.hybrid_rgb16, hybrid_rgb16)
        assert result.synth_mask_png.startswith(b"\x89PNG\r\n\x1a\n")
        assert result.synth_mask_sha256 == hashlib.sha256(result.synth_mask_png).hexdigest()
        assert result.receipt_sha256 == hashlib.sha256(result.receipt).hexdigest()
        assert result.provenance_class == "caller_asserted_bare_npy"

    def test_receipt_fields_are_extracted(self, tmp_path: Path) -> None:
        hybrid_rgb16 = np.zeros((10, 10, 3), dtype=np.uint16)
        receipt = _receipt(requested="auto", used="cpu-fast", fraction=0.01)
        runner = _RecordingRunner(hybrid_rgb16=hybrid_rgb16, receipt=receipt)

        result = run_hybrid_repair(
            _rgbi(10, 10, 1000),
            _rgbi(5, 5, 500),
            same_frame_id="frame-002",
            backend="auto",
            runtime=_runtime_config(tmp_path),
            scratch_dir=tmp_path,
            runner=runner,
            receipt_verifier=_test_receipt_verifier,
        )

        assert result.engine_version == "0.3.0"
        assert result.backend_requested == "auto"
        assert result.backend_used == "cpu-fast"
        assert result.synthesis_fraction == 0.01
        assert result.routing_counts == {
            "final_regions": 1,
            "synthesis_pixels": 1,
            "frame_pixels": 100,
            "at_floor_pixels": 1,
        }

    def test_rejects_receipt_that_omits_routing_counts(self, tmp_path: Path) -> None:
        hybrid_rgb16 = np.zeros((2, 2, 3), dtype=np.uint16)
        base_runner = _RecordingRunner(hybrid_rgb16=hybrid_rgb16, receipt=_receipt())

        def runner(argv, **kwargs):
            completed = base_runner(argv, **kwargs)
            receipt_path = Path(argv[argv.index("--out") + 1]) / "hybrid-receipt.json"
            document = json.loads(receipt_path.read_text(encoding="utf-8"))
            del document["routing"]["counts"]["frame_pixels"]
            receipt_path.write_bytes(json.dumps(document, sort_keys=True, separators=(",", ":")).encode() + b"\n")
            return completed

        with pytest.raises(HybridRunError, match="routing counts"):
            run_hybrid_repair(
                _rgbi(2, 2, 1000),
                _rgbi(2, 2, 500),
                same_frame_id="frame-003",
                backend="cpu",
                runtime=_runtime_config(tmp_path),
                scratch_dir=tmp_path,
                runner=runner,
                receipt_verifier=_test_receipt_verifier,
            )

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
            receipt_verifier=_test_receipt_verifier,
        )

        assert len(runner.calls) == 1
        argv = runner.calls[0]
        assert argv[0] == str(runtime.executable)
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
            _write_success_outputs(
                out_dir,
                hybrid_rgb16=hybrid_rgb16,
                receipt_options=_receipt(),
                argv=list(argv),
            )
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
            receipt_verifier=_test_receipt_verifier,
        )


class TestRunHybridRepairFailure:
    @pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="POSIX FIFO test")
    def test_model_hash_regular_to_fifo_swap_never_blocks(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        weights = tmp_path / "weights.bin"
        weights.write_bytes(b"weights")
        real_open = hybrid_runner.os.open
        swapped = False

        def swap_before_open(path, flags, *args):
            nonlocal swapped
            if not swapped and Path(path) == weights:
                swapped = True
                weights.unlink()
                os.mkfifo(weights)
            return real_open(path, flags, *args)

        monkeypatch.setattr(hybrid_runner.os, "open", swap_before_open)
        started = time.monotonic()
        with pytest.raises(ValueError, match="changed while it was opened"):
            hybrid_runner._stable_regular_sha256(weights, label="model weights")
        assert time.monotonic() - started < 1.0

    @pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="POSIX FIFO test")
    def test_result_regular_to_fifo_swap_never_blocks(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        artifact = tmp_path / "artifact.bin"
        artifact.write_bytes(b"artifact")
        real_open = hybrid_runner.os.open
        swapped = False

        def swap_before_open(path, flags, *args):
            nonlocal swapped
            if not swapped and Path(path) == artifact:
                swapped = True
                artifact.unlink()
                os.mkfifo(artifact)
            return real_open(path, flags, *args)

        monkeypatch.setattr(hybrid_runner.os, "open", swap_before_open)
        started = time.monotonic()
        with pytest.raises(HybridRunError, match="changed while being opened"):
            hybrid_runner._stable_regular_bytes(
                artifact,
                maximum_bytes=1024,
                label="hybrid artifact",
            )
        assert time.monotonic() - started < 1.0

    def test_npy_header_geometry_is_rejected_before_array_allocation(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        stream = io.BytesIO()
        np.lib.format.write_array_header_1_0(
            stream,
            {
                "descr": "<u2",
                "fortran_order": False,
                "shape": (1_000_000, 1_000_000, 3),
            },
        )
        monkeypatch.setattr(
            hybrid_runner.np,
            "load",
            lambda *_args, **_kwargs: pytest.fail("np.load must not run before NPY header validation"),
        )

        with pytest.raises(HybridRunError, match="shape"):
            hybrid_runner._decode_npy_rgb16(
                stream.getvalue(),
                expected_shape=(2, 2, 3),
            )

    def test_png_geometry_is_rejected_before_pixel_decompression(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        class HugeHeader:
            format = "PNG"
            size = (1_000_000, 1_000_000)

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def convert(self, _mode):
                pytest.fail("oversized PNG pixels must not be decompressed")

        monkeypatch.setattr(
            hybrid_runner.Image,
            "open",
            lambda *_args, **_kwargs: HugeHeader(),
        )

        with pytest.raises(HybridRunError, match="geometry"):
            hybrid_runner._decode_mask_png(
                b"synthetic-png-header",
                expected_shape=(2, 2),
            )

    def test_external_process_logs_are_bounded(self, tmp_path: Path) -> None:
        runtime = _runtime_config(tmp_path)
        runtime.executable.write_text(
            "#!/bin/sh\nhead -c 17825792 /dev/zero\n",
            encoding="utf-8",
        )
        runtime.executable.chmod(0o700)

        with pytest.raises(HybridRunError, match="output exceeded"):
            run_hybrid_repair(
                _rgbi(2, 2, 1000),
                _rgbi(2, 2, 500),
                same_frame_id="bounded-log",
                backend="cpu",
                runtime=runtime,
                scratch_dir=tmp_path,
                timeout_seconds=10,
            )

    def test_cancellation_terminates_blocking_external_process_group(self, tmp_path: Path) -> None:
        runtime = _runtime_config(tmp_path)
        runtime.executable.write_text("#!/bin/sh\nsleep 30\n", encoding="utf-8")
        runtime.executable.chmod(0o700)
        cancel = threading.Event()
        timer = threading.Timer(0.2, cancel.set)
        started = time.monotonic()
        timer.start()
        try:
            with pytest.raises(HybridRunCancelled, match="cancelled"):
                run_hybrid_repair(
                    _rgbi(2, 2, 1000),
                    _rgbi(2, 2, 500),
                    same_frame_id="cancel-blocking-process",
                    backend="cpu",
                    runtime=runtime,
                    scratch_dir=tmp_path,
                    cancel=cancel,
                )
        finally:
            timer.cancel()

        assert time.monotonic() - started < 5.0
        assert not (tmp_path / "out" / "hybrid-receipt.json").exists()

    def test_cancellation_kills_child_that_ignores_term_after_leader_exits(
        self,
        tmp_path: Path,
    ) -> None:
        runtime = _runtime_config(tmp_path)
        child_pid_path = tmp_path / "child.pid"
        child_program = tmp_path / "ignore-term-child.py"
        child_program.write_text(
            "\n".join(
                (
                    "import os",
                    "from pathlib import Path",
                    "import signal",
                    "import sys",
                    "import time",
                    "signal.signal(signal.SIGTERM, signal.SIG_IGN)",
                    "Path(sys.argv[1]).write_text(str(os.getpid()), encoding='ascii')",
                    "while True:",
                    "    time.sleep(0.1)",
                )
            )
            + "\n",
            encoding="utf-8",
        )
        runtime.executable.write_text(
            "\n".join(
                (
                    "#!/bin/sh",
                    f"{sys.executable!s} {child_program!s} {child_pid_path!s} &",
                    "trap 'exit 0' TERM",
                    "while :; do sleep 1; done",
                )
            )
            + "\n",
            encoding="utf-8",
        )
        runtime.executable.chmod(0o700)
        cancel = threading.Event()

        def cancel_after_child_starts() -> None:
            deadline = time.monotonic() + 5.0
            while not child_pid_path.exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            cancel.set()

        canceller = threading.Thread(target=cancel_after_child_starts, daemon=True)
        canceller.start()
        with pytest.raises(HybridRunCancelled, match="cancelled"):
            run_hybrid_repair(
                _rgbi(2, 2, 1000),
                _rgbi(2, 2, 500),
                same_frame_id="cancel-orphan-process",
                backend="cpu",
                runtime=runtime,
                scratch_dir=tmp_path,
                cancel=cancel,
            )
        canceller.join(timeout=1.0)
        assert child_pid_path.exists()
        child_pid = int(child_pid_path.read_text(encoding="ascii"))
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            try:
                os.kill(child_pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.05)
        else:
            pytest.fail(f"hybrid child process {child_pid} survived cancellation")

    @pytest.mark.parametrize(
        ("section", "field", "match"),
        [
            ("core", "source_manifest_sha256", "core source manifest"),
            (
                "generation",
                "hybrid_source_manifest_sha256",
                "hybrid source manifest",
            ),
        ],
    )
    def test_rejects_independent_source_manifest_mismatch(
        self,
        tmp_path: Path,
        section: str,
        field: str,
        match: str,
    ) -> None:
        base = _RecordingRunner(
            hybrid_rgb16=np.zeros((2, 2, 3), dtype=np.uint16),
            receipt=_receipt(),
        )

        def runner(argv, **kwargs):
            completed = base(argv, **kwargs)
            receipt_path = Path(_arg(list(argv), "--out")) / "hybrid-receipt.json"
            document = json.loads(receipt_path.read_bytes())
            document[section][field] = "0" * 64
            receipt_path.write_bytes(json.dumps(document, sort_keys=True, separators=(",", ":")).encode() + b"\n")
            return completed

        with pytest.raises(HybridRunError, match=match):
            run_hybrid_repair(
                _rgbi(2, 2, 1000),
                _rgbi(2, 2, 500),
                same_frame_id="source-manifest-mismatch",
                backend="cpu",
                runtime=_runtime_config(tmp_path),
                scratch_dir=tmp_path,
                runner=runner,
                receipt_verifier=_attest_only,
            )

    def test_rejects_malformed_receipt_json(self, tmp_path: Path) -> None:
        base = _RecordingRunner(
            hybrid_rgb16=np.zeros((2, 2, 3), dtype=np.uint16),
            receipt=_receipt(),
        )

        def runner(argv, **kwargs):
            completed = base(argv, **kwargs)
            receipt_path = Path(_arg(list(argv), "--out")) / "hybrid-receipt.json"
            receipt_path.write_bytes(b"{")
            return completed

        with pytest.raises(HybridRunError, match="receipt JSON is invalid"):
            run_hybrid_repair(
                _rgbi(2, 2, 1000),
                _rgbi(2, 2, 500),
                same_frame_id="malformed-receipt",
                backend="cpu",
                runtime=_runtime_config(tmp_path),
                scratch_dir=tmp_path,
                runner=runner,
                receipt_verifier=_attest_only,
            )

    def test_rejects_receipt_input_hash_swap(self, tmp_path: Path) -> None:
        base = _RecordingRunner(
            hybrid_rgb16=np.zeros((2, 2, 3), dtype=np.uint16),
            receipt=_receipt(),
        )

        def runner(argv, **kwargs):
            completed = base(argv, **kwargs)
            receipt_path = Path(_arg(list(argv), "--out")) / "hybrid-receipt.json"
            document = json.loads(receipt_path.read_bytes())
            document["inputs"]["main"]["raw_sha256"] = "0" * 64
            receipt_path.write_bytes(json.dumps(document, sort_keys=True, separators=(",", ":")).encode() + b"\n")
            return completed

        with pytest.raises(HybridRunError, match="main SHA-256 changed"):
            run_hybrid_repair(
                _rgbi(2, 2, 1000),
                _rgbi(2, 2, 500),
                same_frame_id="input-hash-swap",
                backend="cpu",
                runtime=_runtime_config(tmp_path),
                scratch_dir=tmp_path,
                runner=runner,
                receipt_verifier=_attest_only,
            )

    def test_rejects_invalid_disclosure_png_even_when_file_hash_matches_receipt(self, tmp_path: Path) -> None:
        base = _RecordingRunner(
            hybrid_rgb16=np.zeros((2, 2, 3), dtype=np.uint16),
            receipt=_receipt(),
        )

        def runner(argv, **kwargs):
            completed = base(argv, **kwargs)
            out_dir = Path(_arg(list(argv), "--out"))
            bad_png = b"not-a-png"
            (out_dir / "synth-mask.png").write_bytes(bad_png)
            receipt_path = out_dir / "hybrid-receipt.json"
            document = json.loads(receipt_path.read_bytes())
            mask_artifact = next(row for row in document["artifacts"] if row["role"] == "synthesis_mask_png")
            mask_artifact["file_sha256"] = hashlib.sha256(bad_png).hexdigest()
            receipt_path.write_bytes(json.dumps(document, sort_keys=True, separators=(",", ":")).encode() + b"\n")
            return completed

        with pytest.raises(HybridRunError, match="mask PNG is invalid"):
            run_hybrid_repair(
                _rgbi(2, 2, 1000),
                _rgbi(2, 2, 500),
                same_frame_id="bad-mask-png",
                backend="cpu",
                runtime=_runtime_config(tmp_path),
                scratch_dir=tmp_path,
                runner=runner,
                receipt_verifier=_attest_only,
            )

    def test_rejects_symlinked_result_artifact(self, tmp_path: Path) -> None:
        base = _RecordingRunner(
            hybrid_rgb16=np.zeros((2, 2, 3), dtype=np.uint16),
            receipt=_receipt(),
        )

        def runner(argv, **kwargs):
            completed = base(argv, **kwargs)
            out_dir = Path(_arg(list(argv), "--out"))
            mask_path = out_dir / "synth-mask.png"
            target = out_dir / "attacker-mask.png"
            target.write_bytes(mask_path.read_bytes())
            mask_path.unlink()
            mask_path.symlink_to(target)
            return completed

        with pytest.raises(HybridRunError, match="non-symlink"):
            run_hybrid_repair(
                _rgbi(2, 2, 1000),
                _rgbi(2, 2, 500),
                same_frame_id="symlink-mask",
                backend="cpu",
                runtime=_runtime_config(tmp_path),
                scratch_dir=tmp_path,
                runner=runner,
                receipt_verifier=_attest_only,
            )

    def test_rejects_external_verifier_receipt_hash_mismatch(self, tmp_path: Path) -> None:
        runner = _RecordingRunner(
            hybrid_rgb16=np.zeros((2, 2, 3), dtype=np.uint16),
            receipt=_receipt(),
        )

        def verifier(_path: Path, _runtime: HybridRuntimeConfig):
            return types.SimpleNamespace(
                receipt_sha256="0" * 64,
                model_weights_rehashed=True,
            )

        with pytest.raises(HybridRunError, match="changed after verification"):
            run_hybrid_repair(
                _rgbi(2, 2, 1000),
                _rgbi(2, 2, 500),
                same_frame_id="receipt-hash-mismatch",
                backend="cpu",
                runtime=_runtime_config(tmp_path),
                scratch_dir=tmp_path,
                runner=runner,
                receipt_verifier=verifier,
            )

    def test_rejects_a_verifier_result_with_wrong_output_dtype_and_geometry(self, tmp_path: Path) -> None:
        main = _rgbi(4, 5, 1000)
        prepass = _rgbi(2, 3, 500)

        def runner(argv, **kwargs) -> subprocess.CompletedProcess:
            out_dir = Path(argv[argv.index("--out") + 1])
            out_dir.mkdir(parents=True)
            np.save(
                out_dir / "output-hybrid.rgb16.npy",
                np.zeros((1, 2), dtype=np.float32),
                allow_pickle=False,
            )
            (out_dir / "synth-mask.png").write_bytes(b"not-a-png")
            (out_dir / "hybrid-receipt.json").write_text("{}\n", encoding="utf-8")
            return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

        def verifier(_path: Path, _runtime: HybridRuntimeConfig):
            return types.SimpleNamespace(
                document={},
                receipt_sha256=hashlib.sha256(_path.read_bytes()).hexdigest(),
                hybrid_output_rgb16=np.zeros((1, 2), dtype=np.float32),
                synthesis_mask=np.zeros((1, 2), dtype=np.bool_),
                model_weights_rehashed=True,
            )

        with pytest.raises(HybridRunError, match="HxWx3 uint16"):
            run_hybrid_repair(
                main,
                prepass,
                same_frame_id="frame-malformed-output",
                backend="cpu",
                runtime=_runtime_config(tmp_path),
                scratch_dir=tmp_path,
                runner=runner,
                receipt_verifier=verifier,
            )

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


def test_runtime_config_rejects_relative_process_or_artifact_paths() -> None:
    with pytest.raises(ValueError, match="absolute"):
        HybridRuntimeConfig(
            hybrid_python=Path("hybrid/bin/python"),
            executable=Path("hybrid/bin/fauxce-hybrid"),
            core_source_manifest_sha256=EXPECTED_CORE_SOURCE_SHA256,
            hybrid_source_manifest_sha256=EXPECTED_HYBRID_SOURCE_SHA256,
            iopaint_python=Path("iopaint/bin/python"),
            iopaint_executable=Path("iopaint/bin/iopaint"),
            iopaint_source_manifest_sha256="a" * 64,
            model_dir=Path("models"),
            model_weights=Path("models/big-lama.pt"),
            model_weights_sha256="b" * 64,
        )


def test_launch_oserror_raises_hybrid_run_error(tmp_path: Path) -> None:
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
