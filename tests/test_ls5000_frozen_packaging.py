"""Offline contracts for the frozen LS-5000 capture entry point."""

import ast
import hashlib
import os
import platform
import plistlib
import runpy
import subprocess
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
from PyInstaller.utils.hooks import collect_all

from coolscanpy.protocol.ls5000_single_pass.capture_process import (
    CAPTURE_HELPER_FLAG,
)


REPO = Path(__file__).resolve().parents[1]


def test_desktop_script_dispatches_frozen_auxiliary_before_gui_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The real entry point must not import Qt/desktop code for helper mode."""

    dispatched: list[list[str]] = []
    frozen_entry = types.ModuleType("negpy.desktop.frozen_entry")
    frozen_entry.dispatch_frozen_auxiliary = lambda argv: dispatched.append(argv) or True
    monkeypatch.setitem(sys.modules, "negpy.desktop.frozen_entry", frozen_entry)
    monkeypatch.delitem(sys.modules, "negpy.desktop.main", raising=False)
    real_import = __import__

    def reject_gui_import(name, *args, **kwargs):
        if name == "negpy.desktop.main":
            raise AssertionError("GUI desktop imported during frozen auxiliary dispatch")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", reject_gui_import)
    monkeypatch.setattr(sys, "argv", ["NegPy", "--negpy-packaging-smoke"])

    runpy.run_path(str(REPO / "desktop.py"), run_name="__main__")

    assert dispatched == [["NegPy", "--negpy-packaging-smoke"]]


def test_live_acceptance_dispatch_forwards_exact_arguments_without_qt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The private live runner must enter before the frozen Qt desktop."""

    from negpy.desktop import frozen_entry

    forwarded: list[list[str]] = []
    live_acceptance = types.ModuleType("negpy.services.roll.live_acceptance")
    live_acceptance.main = lambda argv: forwarded.append(argv) or 0
    monkeypatch.setitem(
        sys.modules,
        "negpy.services.roll.live_acceptance",
        live_acceptance,
    )
    real_import = __import__

    def reject_gui_import(name, *args, **kwargs):
        if name == "negpy.desktop.main" or name.startswith("PyQt6"):
            raise AssertionError("Qt desktop imported during live acceptance dispatch")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", reject_gui_import)
    arguments = [
        "--device-id",
        "usb:2:7",
        "--confirm-live",
    ]

    handled = frozen_entry.dispatch_frozen_auxiliary(
        ["NegPy", frozen_entry.LIVE_ACCEPTANCE_FLAG, *arguments]
    )

    assert handled is True
    assert forwarded == [arguments]
    assert "LIVE_ACCEPTANCE_FLAG" in frozen_entry.__all__
    assert "dispatch_live_acceptance" in frozen_entry.__all__


def test_live_acceptance_dispatch_propagates_failure_exit_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from negpy.desktop import frozen_entry

    live_acceptance = types.ModuleType("negpy.services.roll.live_acceptance")
    live_acceptance.main = lambda _argv: 23
    monkeypatch.setitem(
        sys.modules,
        "negpy.services.roll.live_acceptance",
        live_acceptance,
    )

    with pytest.raises(SystemExit) as stopped:
        frozen_entry.dispatch_frozen_auxiliary(
            ["NegPy", frozen_entry.LIVE_ACCEPTANCE_FLAG]
        )

    assert stopped.value.code == 23


def test_capture_helper_dispatch_runs_the_real_worker_without_usb(capsys) -> None:
    from negpy.desktop.main import _dispatch_capture_helper

    handled = _dispatch_capture_helper(
        [
            "NegPy",
            CAPTURE_HELPER_FLAG,
            "--frame",
            "18",
            "--boundary-offset-rows",
            "0",
            "--confirm-full-capture",
        ]
    )

    assert handled is True
    output = capsys.readouterr().out
    assert "619458560 bytes" in output
    assert "dry run only; scanner was not accessed" in output


def test_desktop_main_stops_before_bootstrap_for_capture_helper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import negpy.desktop.main as desktop_main

    monkeypatch.setattr(desktop_main, "_dispatch_capture_helper", lambda _argv: True)

    def fail_if_bootstrapped(*_args, **_kwargs):
        raise AssertionError("desktop bootstrap ran in capture-helper mode")

    monkeypatch.setattr(desktop_main, "load_override", fail_if_bootstrapped)
    monkeypatch.setattr(
        desktop_main,
        "_macos_frozen_documents_handoff",
        lambda: (_ for _ in ()).throw(AssertionError("handoff ran in capture-helper mode")),
    )

    desktop_main.main()


class _FakeMacApplication:
    events: list[str] = []

    @staticmethod
    def instance():
        return None

    def __init__(self, _argv) -> None:
        self.events.append("application")

    def processEvents(self) -> None:
        self.events.append("events")


class _FakeMacProgress:
    window_flags: list[tuple[object, bool]] = []

    def __init__(self, *_args) -> None:
        _FakeMacApplication.events.append("progress")

    def setWindowTitle(self, _title: str) -> None:
        pass

    def setCancelButton(self, _button) -> None:
        pass

    def setAutoClose(self, _enabled: bool) -> None:
        pass

    def setAutoReset(self, _enabled: bool) -> None:
        pass

    def setWindowFlag(self, _flag, _enabled: bool) -> None:
        self.window_flags.append((_flag, _enabled))

    def setWindowModality(self, _modality) -> None:
        pass

    def show(self) -> None:
        _FakeMacApplication.events.append("show")

    def close(self) -> None:
        _FakeMacApplication.events.append("close")


def _configure_frozen_macos_handoff(monkeypatch: pytest.MonkeyPatch, desktop_main) -> None:
    _FakeMacApplication.events = []
    _FakeMacProgress.window_flags = []
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.delenv(desktop_main._MACOS_DOCUMENTS_READY_ENV, raising=False)  # noqa: SLF001 - startup seam
    monkeypatch.setattr(desktop_main, "QApplication", _FakeMacApplication)
    monkeypatch.setattr(desktop_main, "_MacOSDocumentsHandoff", _FakeMacProgress)


def test_macos_frozen_handoff_shows_ui_before_probe_and_reexecs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import negpy.desktop.main as desktop_main

    _configure_frozen_macos_handoff(monkeypatch, desktop_main)
    monkeypatch.setattr(desktop_main, "BASE_USER_DIR", str(tmp_path))

    def probe(_path: str) -> None:
        _FakeMacApplication.events.append("probe")

    class ReexecCalled(Exception):
        pass

    def reexec(executable: str, argv: list[str], env: dict[str, str]) -> None:
        _FakeMacApplication.events.append("reexec")
        assert executable == str(Path(sys.executable).absolute())
        assert argv[0] == executable
        assert env[desktop_main._MACOS_DOCUMENTS_READY_ENV] == "1"  # noqa: SLF001 - startup seam
        raise ReexecCalled

    monkeypatch.setattr(desktop_main, "_probe_user_data_access", probe)
    monkeypatch.setattr(desktop_main.os, "execve", reexec)

    with pytest.raises(ReexecCalled):
        desktop_main._macos_frozen_documents_handoff()  # noqa: SLF001 - startup seam

    assert _FakeMacApplication.events.index("application") < _FakeMacApplication.events.index("probe")
    assert _FakeMacApplication.events.index("show") < _FakeMacApplication.events.index("probe")
    assert _FakeMacApplication.events[-1] == "reexec"
    assert (desktop_main.Qt.WindowType.WindowCloseButtonHint, False) in _FakeMacProgress.window_flags


def test_macos_frozen_handoff_consumes_its_one_process_sentinel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import negpy.desktop.main as desktop_main

    _configure_frozen_macos_handoff(monkeypatch, desktop_main)
    monkeypatch.setenv(desktop_main._MACOS_DOCUMENTS_READY_ENV, "1")  # noqa: SLF001 - startup seam
    monkeypatch.setattr(
        desktop_main,
        "_probe_user_data_access",
        lambda _path: (_ for _ in ()).throw(AssertionError("child probed Documents again")),
    )

    assert desktop_main._macos_frozen_documents_handoff() is False  # noqa: SLF001 - startup seam
    assert desktop_main._MACOS_DOCUMENTS_READY_ENV not in os.environ  # noqa: SLF001 - startup seam
    assert _FakeMacApplication.events == []


def test_macos_frozen_handoff_denial_stops_startup_without_bootstrap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import negpy.desktop.main as desktop_main

    _configure_frozen_macos_handoff(monkeypatch, desktop_main)
    errors: list[tuple] = []
    monkeypatch.setattr(desktop_main, "_dispatch_packaging_smoke", lambda _argv: False)
    monkeypatch.setattr(desktop_main, "_dispatch_capture_helper", lambda _argv: False)
    monkeypatch.setattr(desktop_main, "_probe_user_data_access", lambda _path: (_ for _ in ()).throw(PermissionError("denied")))
    monkeypatch.setattr(
        desktop_main,
        "QMessageBox",
        SimpleNamespace(critical=lambda *args: errors.append(args)),
    )
    monkeypatch.setattr(
        desktop_main,
        "load_override",
        lambda _path: (_ for _ in ()).throw(AssertionError("override loaded after denial")),
    )
    monkeypatch.setattr(
        desktop_main,
        "_bootstrap_environment",
        lambda: (_ for _ in ()).throw(AssertionError("bootstrap ran after denial")),
    )

    desktop_main.main()

    assert errors
    assert "Documents access" in errors[0][1]
    assert "progress" in _FakeMacApplication.events


def test_macos_handoff_ignores_close_events(qapp) -> None:
    from PyQt6.QtGui import QCloseEvent

    from negpy.desktop.main import _MacOSDocumentsHandoff

    handoff = _MacOSDocumentsHandoff("Opening workspace", None, 0, 0)
    event = QCloseEvent()
    handoff.closeEvent(event)

    assert not event.isAccepted()


def test_macos_access_probe_removes_its_empty_file(tmp_path: Path) -> None:
    from negpy.desktop.main import _MACOS_ACCESS_PROBE_PREFIX, _probe_user_data_access

    _probe_user_data_access(str(tmp_path))

    assert not list(tmp_path.glob(f"{_MACOS_ACCESS_PROBE_PREFIX}*"))


def test_desktop_packaging_smoke_dispatches_before_bootstrap(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from negpy.desktop import frozen_entry

    checked: list[bool] = []
    monkeypatch.setattr(
        frozen_entry,
        "run_packaging_smoke_checks",
        lambda: checked.append(True),
    )

    handled = frozen_entry.dispatch_packaging_smoke(["NegPy", frozen_entry.PACKAGING_SMOKE_FLAG])

    assert handled is True
    assert checked == [True]
    assert frozen_entry.PACKAGING_SMOKE_OK in capsys.readouterr().out


def test_pyinstaller_collects_the_frozen_scanner_and_dice_runtimes() -> None:
    build_source = (REPO / "build.py").read_text(encoding="utf-8")
    frozen_entry_source = (REPO / "negpy/desktop/frozen_entry.py").read_text(encoding="utf-8")

    required_options = {
        "--collect-all=coolscanpy",
        "--hidden-import=coolscanpy.protocol.ls5000_single_pass.worker",
        "--hidden-import=negpy.services.roll.live_acceptance",
        "--collect-all=portable_digital_ice",
    }
    assert all(option in build_source for option in required_options)
    assert 'LIVE_ACCEPTANCE_FLAG = "--ls5000-live-acceptance"' in build_source
    assert "--add-data=negpy/assets/portable_cms:negpy/assets/portable_cms" in build_source
    # This constructor performs frozen hash/schema validation of the packaged
    # 12-event receipt as well as the nine transform binaries.
    smoke_function = next(
        node
        for node in ast.parse(frozen_entry_source).body
        if isinstance(node, ast.FunctionDef) and node.name == "run_packaging_smoke_checks"
    )
    smoke_calls = {node.func.id for node in ast.walk(smoke_function) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
    assert "check_jit_execution" in smoke_calls
    assert "PortableCMSOnEvaluator" in smoke_calls
    assert "_require_registered_repair_engine" in smoke_calls
    assert "_require_pinned_hybrid_runtime" in smoke_calls
    assert "--codesign-identity=" in build_source


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS bundle identity contract")
def test_macos_build_uses_a_stable_reverse_dns_bundle_identifier() -> None:
    import build

    assert build.MACOS_BUNDLE_IDENTIFIER == "org.negpy.NegPy"
    assert f"--osx-bundle-identifier={build.MACOS_BUNDLE_IDENTIFIER}" in build.params


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS hardened-runtime contract")
def test_macos_build_uses_only_the_llvmlite_executable_memory_entitlement() -> None:
    import build

    entitlements_path = Path(build.MACOS_ENTITLEMENTS_FILE)
    entitlements = plistlib.loads(entitlements_path.read_bytes())

    assert entitlements == {
        "com.apple.security.cs.allow-unsigned-executable-memory": True,
    }
    assert f"--osx-entitlements-file={entitlements_path}" in build.params


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS hardened-runtime contract")
def test_macos_embedded_entitlements_are_read_back_as_xml(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import build

    app_path = tmp_path / "NegPy.app"
    embedded = {"com.apple.security.cs.allow-unsigned-executable-memory": True}
    calls: list[list[str]] = []

    def emit_entitlements(argv, **_kwargs):
        calls.append(list(argv))
        return subprocess.CompletedProcess(argv, 0, stdout=plistlib.dumps(embedded), stderr=b"")

    monkeypatch.setattr(build.subprocess, "run", emit_entitlements)

    assert build.verify_macos_runtime_entitlements(str(app_path)) == embedded
    assert calls == [
        [
            "/usr/bin/codesign",
            "--display",
            "--entitlements",
            "-",
            "--xml",
            str(app_path),
        ]
    ]


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS hardened-runtime contract")
def test_macos_developer_id_signature_exposes_the_hardened_runtime_flag(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import build

    app_path = tmp_path / "NegPy.app"
    calls: list[list[str]] = []

    def emit_signature_details(argv, **_kwargs):
        calls.append(list(argv))
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="CodeDirectory v=20500 flags=0x10000(runtime)\n")

    monkeypatch.setattr(build.subprocess, "run", emit_signature_details)

    assert build.verify_macos_hardened_runtime(str(app_path)) == 0x10000
    assert calls == [["/usr/bin/codesign", "--display", "--verbose=4", str(app_path)]]


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS hardened-runtime contract")
def test_macos_developer_id_signature_rejects_a_non_runtime_seal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import build

    monkeypatch.setattr(
        build.subprocess,
        "run",
        lambda argv, **_kwargs: subprocess.CompletedProcess(
            argv,
            0,
            stdout="",
            stderr="CodeDirectory v=20400 flags=0x2(adhoc)\n",
        ),
    )

    with pytest.raises(RuntimeError, match="missing the hardened-runtime"):
        build.verify_macos_hardened_runtime(str(tmp_path / "NegPy.app"))


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS hardened-runtime contract")
@pytest.mark.parametrize(
    "embedded",
    [
        {},
        {
            "com.apple.security.cs.allow-unsigned-executable-memory": True,
            "com.apple.security.cs.allow-jit": True,
        },
        {
            "com.apple.security.cs.allow-unsigned-executable-memory": True,
            "com.apple.security.cs.disable-library-validation": True,
        },
    ],
)
def test_macos_embedded_entitlement_verification_fails_closed(
    embedded: dict[str, bool],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import build

    def emit_entitlements(argv, **_kwargs):
        return subprocess.CompletedProcess(argv, 0, stdout=plistlib.dumps(embedded), stderr=b"")

    monkeypatch.setattr(build.subprocess, "run", emit_entitlements)

    with pytest.raises(RuntimeError, match="unexpected hardened-runtime entitlements"):
        build.verify_macos_runtime_entitlements(str(tmp_path / "NegPy.app"))


def test_frozen_smoke_rejects_an_unregistered_dice_bridge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from negpy.desktop import frozen_entry
    from negpy.infrastructure.roll import repair as roll_repair

    monkeypatch.setattr(roll_repair, "available", lambda: False)

    with pytest.raises(RuntimeError, match="Digital ICE repair bridge"):
        frozen_entry._require_registered_repair_engine()  # noqa: SLF001 - frozen acceptance seam


def test_frozen_smoke_rehashes_the_pinned_hybrid_model(tmp_path: Path) -> None:
    from negpy.desktop import frozen_entry

    model = tmp_path / "big-lama.pt"
    model.write_bytes(b"offline pinned model fixture")
    expected = hashlib.sha256(model.read_bytes()).hexdigest()
    validated: list[bool] = []
    runtime = SimpleNamespace(
        model_weights=model,
        model_weights_sha256=expected,
        validate_files=lambda: validated.append(True),
    )

    actual = frozen_entry._require_pinned_hybrid_runtime(  # noqa: SLF001 - frozen acceptance seam
        loader=lambda: runtime,
    )

    assert actual == expected
    assert validated == [True]


@pytest.mark.parametrize("failure", ["missing-runtime", "wrong-model", "symlink-model"])
def test_frozen_smoke_rejects_an_unpinned_hybrid_runtime(
    failure: str,
    tmp_path: Path,
) -> None:
    from negpy.desktop import frozen_entry

    model = tmp_path / "big-lama.pt"
    model.write_bytes(b"offline pinned model fixture")
    expected = hashlib.sha256(model.read_bytes()).hexdigest()
    runtime = SimpleNamespace(
        model_weights=model,
        model_weights_sha256=expected,
        validate_files=lambda: None,
    )
    if failure == "missing-runtime":
        runtime_to_load = None
    else:
        runtime_to_load = runtime
    if failure == "wrong-model":
        runtime.model_weights_sha256 = "0" * 64
    elif failure == "symlink-model":
        alias = tmp_path / "model-link.pt"
        alias.symlink_to(model)
        runtime.model_weights = alias

    def loader():
        return runtime_to_load

    with pytest.raises(RuntimeError, match="hybrid runtime|model weights"):
        frozen_entry._require_pinned_hybrid_runtime(  # noqa: SLF001 - frozen acceptance seam
            loader=loader,
        )


def test_normal_desktop_startup_rejects_model_corruption_after_manifest_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from negpy.desktop import main as desktop_main
    from negpy.services.repair.fauxice_hybrid_runner import HybridRuntimeConfig

    executables = [tmp_path / name for name in ("python", "fauxce-hybrid", "iopaint-python", "iopaint")]
    for executable in executables:
        executable.write_bytes(b"offline executable fixture")
        executable.chmod(0o755)
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    model = model_dir / "big-lama.pt"
    model.write_bytes(b"corrupted after the manifest was pinned")
    runtime = HybridRuntimeConfig(
        hybrid_python=executables[0],
        executable=executables[1],
        core_source_manifest_sha256="1" * 64,
        hybrid_source_manifest_sha256="2" * 64,
        iopaint_python=executables[2],
        iopaint_executable=executables[3],
        iopaint_source_manifest_sha256="3" * 64,
        model_dir=model_dir,
        model_weights=model,
        model_weights_sha256="0" * 64,
    )
    monkeypatch.setattr(desktop_main, "load_default_hybrid_runtime_manifest", lambda: runtime)

    assert desktop_main._load_desktop_hybrid_runtime() is None  # noqa: SLF001 - startup gate


def test_pyinstaller_hooks_find_capture_assets_and_dice_backends() -> None:
    coolscan_data, _coolscan_binaries, coolscan_modules = collect_all("coolscanpy")
    _dice_data, _dice_binaries, dice_modules = collect_all("portable_digital_ice")

    captured_names = {Path(source).name for source, _destination in coolscan_data}
    assert {
        "replay-first-rgbi4-manifest.json",
        "replay-first-rgbi4-plan.jsonl",
        "replay-next-rgbi4-plan.json",
    } <= captured_names
    assert "coolscanpy.protocol.ls5000_single_pass.worker" in coolscan_modules
    assert {
        "portable_digital_ice.backend",
        "portable_digital_ice.fast_cpu.engine",
        "portable_digital_ice.metal_backend.engine",
    } <= set(dice_modules)


def test_build_preflight_rejects_a_stale_coolscan_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import build

    real_import = build.importlib.import_module

    def import_without_exact_builder(module_name: str):
        if module_name == "coolscanpy.protocol.ls5000_single_pass.density":
            return SimpleNamespace(
                build_nikon_density_evidence=lambda: None,
                build_nikon_exact_builder_evidence=lambda: None,
            )
        return real_import(module_name)

    monkeypatch.setattr(build.importlib, "import_module", import_without_exact_builder)

    with pytest.raises(RuntimeError, match="NikonExactBuilderEvidence"):
        build.preflight_coolscan_runtime()


def test_build_preflight_accepts_the_current_sealed_capture_bundle() -> None:
    import build
    from coolscanpy.protocol.ls5000_single_pass.bundle import CAPTURE_BUNDLE_SHA256

    assert build.preflight_coolscan_runtime() == CAPTURE_BUNDLE_SHA256


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS dylib packaging contract")
def test_macos_build_declares_a_present_arch_compatible_libusb() -> None:
    import build

    source = Path(build.MACOS_LIBUSB_SOURCE)

    assert source.is_absolute()
    assert source.is_file()
    assert source.name == build.COOLSCAN_LIBUSB_FILENAME
    assert f"--add-binary={source}:coolscanpy/_native" in build.params
    architecture = subprocess.run(
        ["/usr/bin/lipo", str(source), "-verify_arch", platform.machine()],
        check=False,
        capture_output=True,
        text=True,
    )
    assert architecture.returncode == 0, architecture.stderr


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS signing contract")
def test_macos_build_signs_libusb_before_resealing_the_app(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import build

    app_path = tmp_path / "NegPy.app"
    bundled_libusb = app_path / "Contents" / "Frameworks" / build.COOLSCAN_LIBUSB_DESTINATION / build.COOLSCAN_LIBUSB_FILENAME
    bundled_libusb.parent.mkdir(parents=True)
    bundled_libusb.write_bytes(b"offline signing fixture")
    calls: list[list[str]] = []
    entitlement_checks: list[str] = []

    def record_run(argv, **_kwargs):
        calls.append(list(argv))
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(build.subprocess, "run", record_run)
    monkeypatch.setattr(
        build,
        "verify_macos_runtime_entitlements",
        lambda path: entitlement_checks.append(path),
    )
    monkeypatch.setenv("NEGPY_CODESIGN_IDENTITY", "-")

    signed_path = build.sign_macos_scanner_runtime(str(app_path))

    assert signed_path == str(bundled_libusb)
    signing_targets = [call[-1] for call in calls if "--sign" in call]
    assert signing_targets == [str(bundled_libusb), str(app_path)]
    verification_targets = [call[-1] for call in calls if "--verify" in call]
    assert verification_targets == [str(bundled_libusb), str(app_path)]
    assert entitlement_checks == [str(app_path)]


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS signing contract")
def test_macos_reseal_reapplies_the_exact_pyinstaller_entitlements(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import build

    app_path = tmp_path / "NegPy.app"
    bundled_libusb = app_path / "Contents" / "Frameworks" / build.COOLSCAN_LIBUSB_DESTINATION / build.COOLSCAN_LIBUSB_FILENAME
    bundled_libusb.parent.mkdir(parents=True)
    bundled_libusb.write_bytes(b"offline signing fixture")
    calls: list[list[str]] = []

    def record_run(argv, **_kwargs):
        calls.append(list(argv))
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(build.subprocess, "run", record_run)
    monkeypatch.setattr(build, "verify_macos_runtime_entitlements", lambda _path: None)
    monkeypatch.setenv("NEGPY_CODESIGN_IDENTITY", "-")

    build.sign_macos_scanner_runtime(str(app_path))

    app_sign = next(call for call in calls if "--sign" in call and call[-1] == str(app_path))
    entitlement_index = app_sign.index("--entitlements")
    assert app_sign[entitlement_index + 1] == build.MACOS_ENTITLEMENTS_FILE
    preserve_option = next(option for option in app_sign if option.startswith("--preserve-metadata="))
    assert "entitlements" not in preserve_option.split("=", 1)[1].split(",")


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS signing contract")
def test_macos_developer_id_reseal_verifies_hardened_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import build

    app_path = tmp_path / "NegPy.app"
    bundled_libusb = app_path / "Contents" / "Frameworks" / build.COOLSCAN_LIBUSB_DESTINATION / build.COOLSCAN_LIBUSB_FILENAME
    bundled_libusb.parent.mkdir(parents=True)
    bundled_libusb.write_bytes(b"offline signing fixture")
    runtime_checks: list[str] = []

    monkeypatch.setattr(
        build.subprocess,
        "run",
        lambda argv, **_kwargs: subprocess.CompletedProcess(argv, 0),
    )
    monkeypatch.setattr(build, "verify_macos_runtime_entitlements", lambda _path: None)
    monkeypatch.setattr(
        build,
        "verify_macos_hardened_runtime",
        lambda path: runtime_checks.append(path),
    )
    monkeypatch.setenv("NEGPY_CODESIGN_IDENTITY", "Developer ID Application: NegPy Test")

    build.sign_macos_scanner_runtime(str(app_path))

    assert runtime_checks == [str(app_path)]


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS app smoke contract")
def test_macos_postbuild_smoke_uses_only_offline_frozen_entry_points(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import build

    app_path = tmp_path / "NegPy.app"
    executable = app_path / "Contents" / "MacOS" / "NegPy"
    executable.parent.mkdir(parents=True)
    executable.write_bytes(b"offline executable fixture")
    executable.chmod(0o755)
    bundled_libusb = app_path / "Contents" / "Frameworks" / build.COOLSCAN_LIBUSB_DESTINATION / build.COOLSCAN_LIBUSB_FILENAME
    bundled_libusb.parent.mkdir(parents=True)
    bundled_libusb.write_bytes(b"offline dylib fixture")

    class LoadedLibusb:
        libusb_init = object()
        libusb_exit = object()

    monkeypatch.setattr(build.ctypes, "CDLL", lambda _path: LoadedLibusb())
    commands: list[list[str]] = []
    environments: list[dict[str, str]] = []
    ambient_loader_variables = {
        "DYLD_FALLBACK_FRAMEWORK_PATH",
        "DYLD_FALLBACK_LIBRARY_PATH",
        "DYLD_FRAMEWORK_PATH",
        "DYLD_INSERT_LIBRARIES",
        "DYLD_LIBRARY_PATH",
    }
    for variable in ambient_loader_variables:
        monkeypatch.setenv(variable, "/untrusted/host/path")

    def complete(argv, **kwargs):
        command = list(argv)
        commands.append(command)
        environments.append(kwargs["env"])
        if build.PACKAGING_SMOKE_FLAG in command:
            stdout = build.PACKAGING_SMOKE_OK + "\n"
        else:
            stdout = "validated RGBI4x plan: 619458560 bytes\ndry run only; scanner was not accessed\n"
        return subprocess.CompletedProcess(command, 0, stdout, "")

    monkeypatch.setattr(build.subprocess, "run", complete)

    result = build.smoke_macos_scanner_runtime(str(app_path))

    assert result["libusb"] == str(bundled_libusb)
    assert [command[1] for command in commands] == [
        build.PACKAGING_SMOKE_FLAG,
        build.CAPTURE_HELPER_FLAG,
    ]
    assert all("--live" not in command for command in commands)
    assert all(build.LIVE_ACCEPTANCE_FLAG not in command for command in commands)
    assert all(ambient_loader_variables.isdisjoint(environment) for environment in environments)


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS app smoke contract")
def test_macos_postbuild_smoke_refuses_the_live_acceptance_flag(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import build

    app_path = tmp_path / "NegPy.app"
    executable = app_path / "Contents" / "MacOS" / "NegPy"
    executable.parent.mkdir(parents=True)
    executable.write_bytes(b"offline executable fixture")
    executable.chmod(0o755)
    bundled_libusb = app_path / "Contents" / "Frameworks" / build.COOLSCAN_LIBUSB_DESTINATION / build.COOLSCAN_LIBUSB_FILENAME
    bundled_libusb.parent.mkdir(parents=True)
    bundled_libusb.write_bytes(b"offline dylib fixture")

    class LoadedLibusb:
        libusb_init = object()
        libusb_exit = object()

    monkeypatch.setattr(build.ctypes, "CDLL", lambda _path: LoadedLibusb())
    monkeypatch.setattr(build, "PACKAGING_SMOKE_FLAG", build.LIVE_ACCEPTANCE_FLAG)
    monkeypatch.setattr(
        build.subprocess,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("live acceptance reached subprocess")
        ),
    )

    with pytest.raises(RuntimeError, match="attempted to enable live acceptance"):
        build.smoke_macos_scanner_runtime(str(app_path))


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS app smoke contract")
def test_macos_postbuild_smoke_resolves_a_relative_bundle_before_chdir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A build invoked from its project root must not resolve dist twice."""
    import build

    app_path = tmp_path / "dist" / "NegPy.app"
    executable = app_path / "Contents" / "MacOS" / "NegPy"
    executable.parent.mkdir(parents=True)
    executable.write_bytes(b"offline executable fixture")
    executable.chmod(0o755)
    bundled_libusb = app_path / "Contents" / "Frameworks" / build.COOLSCAN_LIBUSB_DESTINATION / build.COOLSCAN_LIBUSB_FILENAME
    bundled_libusb.parent.mkdir(parents=True)
    bundled_libusb.write_bytes(b"offline dylib fixture")

    class LoadedLibusb:
        libusb_init = object()
        libusb_exit = object()

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(build.ctypes, "CDLL", lambda _path: LoadedLibusb())
    commands: list[list[str]] = []
    working_directories: list[str] = []

    def complete(argv, **kwargs):
        command = list(argv)
        commands.append(command)
        working_directories.append(kwargs["cwd"])
        stdout = (
            build.PACKAGING_SMOKE_OK + "\n"
            if build.PACKAGING_SMOKE_FLAG in command
            else "validated RGBI4x plan: 619458560 bytes\ndry run only; scanner was not accessed\n"
        )
        return subprocess.CompletedProcess(command, 0, stdout, "")

    monkeypatch.setattr(build.subprocess, "run", complete)

    result = build.smoke_macos_scanner_runtime("dist/NegPy.app")

    assert result["executable"] == str(executable.resolve())
    assert all(command[0] == str(executable.resolve()) for command in commands)
    assert working_directories == [str(app_path.parent.resolve())] * 2
