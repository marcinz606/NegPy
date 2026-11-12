import os
import threading
import time
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from negpy.desktop.controller import AppController
from negpy.desktop.launch import (
    LaunchArgsError,
    LaunchRequest,
    LaunchServer,
    decode_request,
    encode_request,
    hand_off,
    parse_launch_args,
    server_name,
)
from negpy.desktop.view.sidebar.export import ExportSidebar
from negpy.domain.models import ExportConfig, ExportPreset, ExportPresetOutputMode, ExportResolutionMode, RenderIntent
from negpy.services.export.launch_recipe import LaunchRecipe, apply_recipe, recipe_groups, recipe_intent
from tests.conftest import FakeController, FakeRepo


def test_parse_launch_args_reads_the_recipe_and_ignores_qt_flags(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    request = parse_launch_args(
        [
            "-style",
            "a.dng",
            "--format",
            "jpg",
            "--quality",
            "85",
            "--long-edge",
            "3000",
            "--color-space",
            "adobe-rgb",
            "--output-dir",
            "out",
            "-psn_0_1",
        ]
    )
    assert request.paths == (str(tmp_path / "a.dng"),)
    recipe = request.recipe
    assert (recipe.export_fmt, recipe.quality, recipe.long_edge, recipe.color_space) == ("JPEG", 85, 3000, "Adobe RGB")
    assert recipe.output_dir == str(tmp_path / "out")
    assert recipe.overwrite is True


def test_parse_launch_args_without_arguments():
    assert parse_launch_args([]) == LaunchRequest()


@pytest.mark.parametrize(
    "argv",
    [
        ["--format", "gif"],
        ["--quality", "0"],
        ["--bit-depth", "12"],
        ["--intent", "negative"],
        ["--long-edge", "3000", "--print-size", "30"],
        ["--long-edge", "3000", "--dpi", "300"],
        ["--color-space", "srgb", "--profile", "x.icc"],
        ["--profile", "missing.icc"],
        ["--formt", "jpeg"],
    ],
)
def test_bad_flags_stop_the_launch(argv):
    with pytest.raises(LaunchArgsError):
        parse_launch_args(argv)


def test_request_round_trips_and_rejects_garbage():
    request = LaunchRequest(paths=("/a/b.dng",), recipe=LaunchRecipe(export_fmt="TIFF", output_dir="/out", quit_after_export=True))
    assert decode_request(encode_request(request)) == request
    assert decode_request(b"\xff not json") is None


def test_server_name_depends_on_the_data_folder():
    assert server_name("/data/one") == server_name("/data/one")
    assert server_name("/data/one") != server_name("/data/two")


def test_hand_off_without_a_running_instance_returns_false():
    assert hand_off(f"negpy-test-absent-{os.getpid()}", LaunchRequest(paths=("x",))) is False


def test_hand_off_delivers_the_request_to_the_server(qapp):
    name = f"negpy-test-{os.getpid()}-{time.monotonic_ns()}"
    server = LaunchServer(name)
    assert server.is_listening()
    received = []
    server.received.connect(received.append)
    request = LaunchRequest(paths=("/scans/a.dng", "/scans/b.dng"), recipe=LaunchRecipe(output_dir="/edits"))
    result = {}
    client = threading.Thread(target=lambda: result.setdefault("ok", hand_off(name, request)))
    client.start()
    deadline = time.monotonic() + 10
    while client.is_alive() and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.01)
    client.join(1)
    server.close()
    server.deleteLater()
    qapp.processEvents()
    assert result.get("ok") is True
    assert received == [request]


def test_apply_recipe_sets_only_what_the_recipe_names():
    conf = ExportConfig(export_fmt="TIFF", jpeg_quality=70, filename_pattern="{{ original_name }}_x", overwrite=False)
    out = apply_recipe(conf, LaunchRecipe(export_fmt="JPEG", quality=95, long_edge=2400, output_dir="/edits"), None)
    assert (out.export_fmt, out.jpeg_quality, out.export_target_long_edge_px) == ("JPEG", 95, 2400)
    assert out.export_resolution_mode == ExportResolutionMode.TARGET_PX.value
    assert (out.output_mode, out.export_path) == (ExportPresetOutputMode.ABSOLUTE, "/edits")
    assert out.filename_pattern == "{{ original_name }}_x"
    assert out.overwrite is True


def test_flags_override_the_preset():
    preset = ExportPreset(name="Web", export_fmt="WEBP", webp_quality=80, output_mode=ExportPresetOutputMode.ABSOLUTE, output_path="/web")
    out = apply_recipe(ExportConfig(), LaunchRecipe(preset="Web", quality=60), preset)
    assert (out.export_fmt, out.webp_quality, out.export_path) == ("WEBP", 60, "/web")


def test_groups_and_intent():
    assert recipe_groups(LaunchRecipe(output_dir="/x"), None) == {"destination"}
    assert recipe_groups(LaunchRecipe(intent="flat", export_fmt="TIFF", color_space="sRGB"), None) == {"intent", "format", "color"}
    assert recipe_groups(LaunchRecipe(), ExportPreset(name="p")) == {"intent", "format", "size", "color", "destination"}
    assert recipe_intent(LaunchRecipe(), ExportPreset(name="p", render_intent=RenderIntent.FLAT)) == "flat"
    assert recipe_intent(LaunchRecipe(), None) is None


def _fake_controller():
    return SimpleNamespace(
        _launch_recipes={},
        state=SimpleNamespace(export_presets=[ExportPreset(name="Catalog TIFF")]),
        request_asset_discovery=MagicMock(),
        set_status=MagicMock(),
        _sync_launch_recipe=MagicMock(),
    )


def test_open_launch_request_registers_the_recipe_and_selects_the_first_file(tmp_path):
    folder = tmp_path / "roll"
    folder.mkdir()
    scan = tmp_path / "a.dng"
    scan.write_bytes(b"")
    fake = _fake_controller()
    recipe = LaunchRecipe(preset="catalog tiff", output_dir="/edits")
    AppController.open_launch_request(fake, [str(folder), str(scan), str(tmp_path / "gone.dng")], recipe)
    fake.request_asset_discovery.assert_called_once_with([str(folder), str(scan)], auto_open=True, select_path=str(scan))
    assert AppController.launch_recipe_for(fake, str(scan)) == (recipe, fake.state.export_presets[0])
    assert AppController.launch_recipe_for(fake, str(folder / "sub" / "b.dng"))[0] == recipe
    assert AppController.launch_recipe_for(fake, str(tmp_path / "other.dng")) is None


def test_open_launch_request_with_only_missing_files_warns(tmp_path):
    fake = _fake_controller()
    AppController.open_launch_request(fake, [str(tmp_path / "gone.dng")], None)
    fake.request_asset_discovery.assert_not_called()
    fake.set_status.assert_called_once()


def _sidebar_with(recipe, preset=None):
    controller = FakeController(repo=FakeRepo())
    sidebar = ExportSidebar(controller)
    found = (recipe, preset)
    controller.current_launch_recipe = lambda: found
    controller.recipe_export_config = lambda: apply_recipe(sidebar.state.config.export, recipe, preset)
    sidebar._sync_launch_recipe()
    return sidebar


def test_the_export_tab_hides_what_the_recipe_sets(qapp):
    sidebar = _sidebar_with(LaunchRecipe(intent="print", export_fmt="JPEG", output_dir="/edits"))
    form = sidebar.form
    assert not form._format_section.isVisibleTo(form)
    assert not form._destination_section.isVisibleTo(form)
    assert form._mode_row_widget.isVisibleTo(form)
    assert form._export_profile_row.isVisibleTo(form)
    assert not sidebar.intent_btn.isVisibleTo(sidebar)
    assert sidebar.recipe_hint.isVisibleTo(sidebar)
    assert "/edits" in sidebar.recipe_hint.text()
    assert not sidebar.export_presets_group.isEnabled()
    assert sidebar.presets_recipe_hint.isVisibleTo(sidebar._presets_section)


def test_a_frame_without_a_recipe_shows_the_whole_tab(qapp):
    sidebar = _sidebar_with(LaunchRecipe(long_edge=3000))
    assert not sidebar.form._mode_row_widget.isVisibleTo(sidebar.form)
    sidebar.controller.current_launch_recipe = lambda: None
    sidebar._sync_launch_recipe()
    form = sidebar.form
    assert form._mode_row_widget.isVisibleTo(form)
    assert form._format_section.isVisibleTo(form)
    assert form._destination_section.isVisibleTo(form)
    assert sidebar.intent_btn.isVisibleTo(sidebar)
    assert not sidebar.recipe_hint.isVisibleTo(sidebar)
    assert sidebar.export_presets_group.isEnabled()


def test_arm_launch_quit_only_for_quit_recipes():
    quit_recipe = replace(LaunchRecipe(quit_after_export=True), started_app=True)
    fake = SimpleNamespace(_launch_quit_pending=None)
    fake.launch_recipe_for = lambda p: {"a": (quit_recipe, None), "b": (LaunchRecipe(), None)}.get(p)
    AppController._arm_launch_quit(fake, ["b", "c"])
    assert fake._launch_quit_pending is None
    AppController._arm_launch_quit(fake, ["a", "b"])
    assert fake._launch_quit_pending is True


def test_recipe_frames_follow_their_own_overwrite_flag(tmp_path):
    from negpy.desktop.workers.export import ExportTask, resolve_export_target_path
    from negpy.domain.models import WorkspaceConfig

    def task(name, overwrite):
        settings = ExportPreset(name="t", output_mode=ExportPresetOutputMode.ABSOLUTE, output_path=str(tmp_path), overwrite=overwrite)
        t = ExportTask(file_info={"path": str(tmp_path / f"{name}.tif"), "name": name}, params=WorkspaceConfig(), export_settings=settings)
        open(resolve_export_target_path(t), "w").close()
        return t

    recipe_task, plain_task = task("edit_in", True), task("plain", False)
    fake = SimpleNamespace(
        state=SimpleNamespace(config=SimpleNamespace(export=SimpleNamespace(overwrite=False))),
        launch_recipe_for=lambda p: (LaunchRecipe(), None) if p.endswith("edit_in.tif") else None,
        _prompt_overwrite_conflicts=MagicMock(return_value=(None, False)),
    )
    assert AppController._resolve_export_conflicts(fake, [recipe_task]) == [recipe_task]
    fake._prompt_overwrite_conflicts.assert_not_called()
    assert AppController._resolve_export_conflicts(fake, [recipe_task, plain_task]) is None
    asked = fake._prompt_overwrite_conflicts.call_args[0][0]
    assert [os.path.basename(p) for p in asked] == [os.path.basename(resolve_export_target_path(plain_task))]
